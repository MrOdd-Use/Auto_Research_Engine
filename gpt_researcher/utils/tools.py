"""
Tool-enabled LLM utilities for Auto_Research_Engine

This module provides provider-agnostic tool calling functionality using LangChain's
unified interface. It allows any LLM provider that supports function calling to use
tools seamlessly.
"""

import asyncio
import logging
from typing import Any, Dict, List, Tuple, Callable, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

from .llm import create_chat_completion
from multi_agents.route_agent import current_route_scope, get_global_invoker
from multi_agents.route_agent.utils.model_utils import normalize_app_provider

logger = logging.getLogger(__name__)


async def create_chat_completion_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Callable],
    model: str | None = None,
    temperature: float | None = 0.4,
    max_tokens: int | None = 4000,
    llm_provider: str | None = None,
    llm_kwargs: Dict[str, Any] | None = None,
    cost_callback: Callable = None,
    websocket: Any | None = None,
    **kwargs
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Create a chat completion with tool calling support across all LLM providers.
    
    This function uses LangChain's bind_tools() to enable function calling in a 
    provider-agnostic way. The AI decides autonomously when and how to use tools.
    
    Args:
        messages: List of chat messages with role and content
        tools: List of LangChain tool functions (decorated with @tool)
        model: The model to use (from config)
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        llm_provider: LLM provider name (from config)
        llm_kwargs: Additional LLM keyword arguments
        cost_callback: Callback function for cost tracking
        websocket: Optional websocket for streaming
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (response_content, tool_calls_metadata)
        
    Raises:
        Exception: If tool-enabled completion fails, falls back to simple completion
    """
    try:
        async def _provider_call(selected_model: str, selected_provider: str = "") -> Tuple[str, List[Dict[str, Any]]]:
            from ..llm_provider.generic.base import GenericLLMProvider

            effective_provider = normalize_app_provider(selected_provider or llm_provider or "")
            provider_kwargs = {
                'model': selected_model,
                **(llm_kwargs or {})
            }

            llm_provider_instance = GenericLLMProvider.from_provider(
                effective_provider,
                **provider_kwargs
            )

            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))

            llm_with_tools = llm_provider_instance.llm.bind_tools(tools)
            logger.info(f"Invoking LLM with {len(tools)} available tools")

            from langchain_core.messages import ToolMessage

            response = await llm_with_tools.ainvoke(lc_messages)

            tool_calls_metadata = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"LLM made {len(response.tool_calls)} tool calls")
                lc_messages.append(response)

                for tool_call in response.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_args = tool_call.get('args', {})
                    tool_id = tool_call.get('id', '')

                    logger.info(f"Tool called: {tool_name}")
                    if tool_args:
                        args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                        logger.debug(f"Tool arguments: {args_str}")

                    tool_result = "Tool execution failed"
                    for tool in tools:
                        if tool.name == tool_name:
                            try:
                                if hasattr(tool, 'ainvoke'):
                                    tool_result = await tool.ainvoke(tool_args)
                                elif hasattr(tool, 'invoke'):
                                    tool_result = tool.invoke(tool_args)
                                else:
                                    tool_result = await tool(**tool_args) if asyncio.iscoroutinefunction(tool) else tool(**tool_args)
                                break
                            except Exception as e:
                                error_type = type(e).__name__
                                error_msg = str(e)
                                logger.error(
                                    f"Error executing tool '{tool_name}': {error_type}: {error_msg}",
                                    exc_info=True
                                )
                                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                                    tool_result = f"Tool '{tool_name}' timed out. The operation took too long to complete. Please try again or check your network connection."
                                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                                    tool_result = f"Tool '{tool_name}' failed due to a network issue. Please check your internet connection and try again."
                                elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                                    tool_result = f"Tool '{tool_name}' failed due to insufficient permissions. Please check your API keys or access credentials."
                                else:
                                    tool_result = f"Tool '{tool_name}' encountered an error: {error_msg}. Please check the logs for more details."

                    tool_message = ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                    lc_messages.append(tool_message)
                    tool_calls_metadata.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "call_id": tool_id,
                        "result": str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                    })

                logger.info("Getting final response from LLM after tool execution")
                final_response = await llm_with_tools.ainvoke(lc_messages)

                if cost_callback:
                    from .costs import estimate_llm_cost
                    llm_costs = estimate_llm_cost(str(lc_messages), final_response.content or "")
                    cost_callback(llm_costs)

                return final_response.content, tool_calls_metadata

            if cost_callback:
                from .costs import estimate_llm_cost
                llm_costs = estimate_llm_cost(str(messages), response.content or "")
                cost_callback(llm_costs)

            return response.content, []

        scope = current_route_scope()
        route_request = None
        if scope is not None:
            route_request = scope.build_request(
                task=_extract_user_task(messages),
                system_prompt=_extract_system_prompt(messages),
                requested_model=None,
                llm_provider=llm_provider or "",
                metadata={"tool_count": len(tools)},
            )
        return await get_global_invoker().invoke(
            provider_call=_provider_call,
            requested_model=model,
            llm_provider=llm_provider or "",
            route_request=route_request,
            metadata={
                "messages": messages,
                "system_prompt": _extract_system_prompt(messages),
                "task": _extract_user_task(messages),
                "tool_count": len(tools),
            },
        )
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(
            f"Error in tool-enabled chat completion: {error_type}: {error_msg}",
            exc_info=True
        )
        logger.info("Falling back to simple chat completion without tools")
        
        # Fallback to simple chat completion without tools
        response = await create_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_provider=llm_provider,
            llm_kwargs=llm_kwargs,
            cost_callback=cost_callback,
            websocket=websocket,
            **kwargs
        )
        return response, []


def _extract_system_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        if str(msg.get("role") or "").lower() != "system":
            continue
        content = str(msg.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts).strip()


def _extract_user_task(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        if str(msg.get("role") or "").lower() not in {"user", "assistant"}:
            continue
        content = str(msg.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts).strip()


def create_search_tool(search_function: Callable[[str], Dict]) -> Callable:
    """
    Create a standardized search tool for use with tool-enabled chat completions.
    
    Args:
        search_function: Function that takes a query string and returns search results
        
    Returns:
        LangChain tool function decorated with @tool
    """
    @tool
    def search_tool(query: str) -> str:
        """Search for current events or online information when you need new knowledge that doesn't exist in the current context"""
        try:
            results = search_function(query)
            if results and 'results' in results:
                search_content = f"Search results for '{query}':\n\n"
                for result in results['results'][:5]:
                    search_content += f"Title: {result.get('title', '')}\n"
                    search_content += f"Content: {result.get('content', '')[:300]}...\n"
                    search_content += f"URL: {result.get('url', '')}\n\n"
                return search_content
            else:
                return f"No search results found for: {query}"
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(
                f"Search tool error: {error_type}: {error_msg}",
                exc_info=True
            )
            # Provide context-aware error messages
            if "api" in error_msg.lower() or "key" in error_msg.lower():
                return f"Search failed: API key issue. Please verify your search API credentials are configured correctly."
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                return f"Search timed out. The search request took too long. Please try again with a different query."
            elif "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                return f"Search rate limit exceeded. Please wait a moment before trying again."
            else:
                return f"Search encountered an error: {error_msg}. Please check your search provider configuration."
    
    return search_tool


def create_custom_tool(
    name: str,
    description: str, 
    function: Callable,
    parameter_schema: Optional[Dict] = None
) -> Callable:
    """
    Create a custom tool for use with tool-enabled chat completions.
    
    Args:
        name: Name of the tool
        description: Description of what the tool does
        function: The actual function to execute
        parameter_schema: Optional schema for function parameters
        
    Returns:
        LangChain tool function decorated with @tool
    """
    @tool
    def custom_tool(*args, **kwargs) -> str:
        try:
            result = function(*args, **kwargs)
            return str(result) if result is not None else "Tool executed successfully"
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(
                f"Custom tool '{name}' error: {error_type}: {error_msg}",
                exc_info=True
            )
            # Provide informative error message without exposing internal details
            if "validation" in error_msg.lower() or "invalid" in error_msg.lower():
                return f"Tool '{name}' received invalid input. Please check the parameters and try again."
            elif "not found" in error_msg.lower() or "missing" in error_msg.lower():
                return f"Tool '{name}' could not find required resources. Please verify the input data is correct."
            else:
                return f"Tool '{name}' encountered an error: {error_msg}. Please check the tool configuration."
    
    # Set tool metadata
    custom_tool.name = name
    custom_tool.description = description
    
    return custom_tool


# Utility function for common tool patterns
def get_available_providers_with_tools() -> List[str]:
    """
    Get list of LLM providers that support tool calling.
    
    Returns:
        List of provider names that support function calling
    """
    # These are the providers known to support function calling in LangChain
    return [
        "openai",
        "anthropic", 
        "google_genai",
        "azure_openai",
        "fireworks",
        "groq",
        # Note: This list may expand as more providers add function calling support
    ]


def supports_tools(provider: str) -> bool:
    """
    Check if a given provider supports tool calling.
    
    Args:
        provider: LLM provider name
        
    Returns:
        True if provider supports tools, False otherwise
    """
    return provider in get_available_providers_with_tools()
