"""
MCP Research Execution Skill

Handles research execution using selected MCP tools as a skill component.
"""
import asyncio
import logging
from typing import List, Dict, Any

from multi_agents.route_agent import current_route_scope, get_global_invoker
from multi_agents.route_agent.utils.model_utils import normalize_app_provider

logger = logging.getLogger(__name__)


class MCPResearchSkill:
    """
    Handles research execution using selected MCP tools.
    
    Responsible for:
    - Executing research with LLM and bound tools
    - Processing tool results into standard format
    - Managing tool execution and error handling
    """

    def __init__(self, cfg, researcher=None):
        """
        Initialize the MCP research skill.
        
        Args:
            cfg: Configuration object with LLM settings
            researcher: Researcher instance for cost tracking
        """
        self.cfg = cfg
        self.researcher = researcher

    async def conduct_research_with_tools(self, query: str, selected_tools: List) -> List[Dict[str, str]]:
        """
        Use LLM with bound tools to conduct intelligent research.
        
        Args:
            query: Research query
            selected_tools: List of selected MCP tools
            
        Returns:
            List[Dict[str, str]]: Research results in standard format
        """
        if not selected_tools:
            logger.warning("No tools available for research")
            return []
            
        logger.info(f"Conducting research using {len(selected_tools)} selected tools")
        
        try:
            from ..llm_provider.generic.base import GenericLLMProvider
            from ..prompts import PromptFamily

            research_prompt = PromptFamily.generate_mcp_research_prompt(query, selected_tools)
            messages = [{"role": "user", "content": research_prompt}]

            async def _provider_call(selected_model: str, selected_provider: str = "") -> List[Dict[str, str]]:
                effective_provider = normalize_app_provider(selected_provider or self.cfg.strategic_llm_provider or "")
                provider_kwargs = {
                    'model': selected_model,
                    **self.cfg.llm_kwargs
                }

                llm_provider = GenericLLMProvider.from_provider(
                    effective_provider,
                    **provider_kwargs
                )
                llm_with_tools = llm_provider.llm.bind_tools(selected_tools)

                logger.info("LLM researching with bound tools...")
                response = await llm_with_tools.ainvoke(messages)
                research_results = []

                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.info(f"LLM made {len(response.tool_calls)} tool calls")
                    for i, tool_call in enumerate(response.tool_calls, 1):
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})

                        logger.info(f"Executing tool {i}/{len(response.tool_calls)}: {tool_name}")
                        if tool_args:
                            args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                            logger.debug(f"Tool arguments: {args_str}")

                        try:
                            tool = next((t for t in selected_tools if t.name == tool_name), None)
                            if not tool:
                                logger.warning(f"Tool {tool_name} not found in selected tools")
                                continue

                            if hasattr(tool, 'ainvoke'):
                                result = await tool.ainvoke(tool_args)
                            elif hasattr(tool, 'invoke'):
                                result = tool.invoke(tool_args)
                            else:
                                result = await tool(tool_args) if asyncio.iscoroutinefunction(tool) else tool(tool_args)

                            if result:
                                result_preview = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
                                logger.debug(f"Tool {tool_name} response preview: {result_preview}")

                                formatted_results = self._process_tool_result(tool_name, result)
                                research_results.extend(formatted_results)
                                logger.info(f"Tool {tool_name} returned {len(formatted_results)} formatted results")

                                for j, formatted_result in enumerate(formatted_results):
                                    title = formatted_result.get("title", "No title")
                                    content_preview = formatted_result.get("body", "")[:200] + "..." if len(formatted_result.get("body", "")) > 200 else formatted_result.get("body", "")
                                    logger.debug(f"Result {j+1}: '{title}' - Content: {content_preview}")
                            else:
                                logger.warning(f"Tool {tool_name} returned empty result")

                        except Exception as e:
                            logger.error(f"Error executing tool {tool_name}: {e}")
                            continue

                if hasattr(response, 'content') and response.content:
                    llm_analysis = {
                        "title": f"LLM Analysis: {query}",
                        "href": "mcp://llm_analysis",
                        "body": response.content
                    }
                    research_results.append(llm_analysis)

                    analysis_preview = response.content[:300] + "..." if len(response.content) > 300 else response.content
                    logger.debug(f"LLM Analysis: {analysis_preview}")
                    logger.info("Added LLM analysis to results")

                logger.info(f"Research completed with {len(research_results)} total results")
                return research_results

            scope = current_route_scope()
            route_request = None
            if scope is not None:
                route_request = scope.build_request(
                    task=query,
                    system_prompt="MCP research with bound tools.",
                    requested_model=self.cfg.strategic_llm_model,
                    llm_provider=self.cfg.strategic_llm_provider,
                    metadata={"tool_count": len(selected_tools)},
                )
            return await get_global_invoker().invoke(
                provider_call=_provider_call,
                requested_model=self.cfg.strategic_llm_model,
                llm_provider=self.cfg.strategic_llm_provider,
                route_request=route_request,
                metadata={"task": query, "tool_count": len(selected_tools)},
            )
            
        except Exception as e:
            logger.error(f"Error in LLM research with tools: {e}")
            return []

    def _process_tool_result(self, tool_name: str, result: Any) -> List[Dict[str, str]]:
        """
        Process tool result into search result format.
        
        Args:
            tool_name: Name of the tool that produced the result
            result: The tool result
            
        Returns:
            List[Dict[str, str]]: Formatted search results
        """
        search_results = []
        
        try:
            # 1) First: handle MCP result wrapper with structured_content/content
            if isinstance(result, dict) and ("structured_content" in result or "content" in result):
                search_results = []
                # Prefer structured_content when present
                structured = result.get("structured_content")
                if isinstance(structured, dict):
                    items = structured.get("results")
                    if isinstance(items, list):
                        for i, item in enumerate(items):
                            if isinstance(item, dict):
                                search_results.append({
                                    "title": item.get("title", f"Result from {tool_name} #{i+1}"),
                                    "href": item.get("href", item.get("url", f"mcp://{tool_name}/{i}")),
                                    "body": item.get("body", item.get("content", str(item)))
                                })
                    # If no items array but structured is dict, treat as single
                    elif isinstance(structured, dict):
                        search_results.append({
                            "title": structured.get("title", f"Result from {tool_name}"),
                            "href": structured.get("href", structured.get("url", f"mcp://{tool_name}")),
                            "body": structured.get("body", structured.get("content", str(structured)))
                        })
                # Fallback to content if provided (MCP spec: list of {type: text, text: ...})
                if not search_results:
                    content_field = result.get("content")
                    if isinstance(content_field, list):
                        texts = []
                        for part in content_field:
                            if isinstance(part, dict):
                                if part.get("type") == "text" and isinstance(part.get("text"), str):
                                    texts.append(part["text"])
                                elif "text" in part:
                                    texts.append(str(part.get("text")))
                                else:
                                    # unknown piece; stringify
                                    texts.append(str(part))
                            else:
                                texts.append(str(part))
                        body_text = "\n\n".join([t for t in texts if t])
                    elif isinstance(content_field, str):
                        body_text = content_field
                    else:
                        body_text = str(result)
                    search_results.append({
                        "title": f"Result from {tool_name}",
                        "href": f"mcp://{tool_name}",
                        "body": body_text,
                    })
                return search_results

            # 2) If the result is already a list, process each item normally
            if isinstance(result, list):
                # If the result is already a list, process each item
                for i, item in enumerate(result):
                    if isinstance(item, dict):
                        # Use the item as is if it has required fields
                        if "title" in item and ("content" in item or "body" in item):
                            search_result = {
                                "title": item.get("title", ""),
                                "href": item.get("href", item.get("url", f"mcp://{tool_name}/{i}")),
                                "body": item.get("body", item.get("content", str(item))),
                            }
                            search_results.append(search_result)
                        else:
                            # Create a search result with a generic title
                            search_result = {
                                "title": f"Result from {tool_name}",
                                "href": f"mcp://{tool_name}/{i}",
                                "body": str(item),
                            }
                            search_results.append(search_result)
            # 3) If the result is a dict (non-MCP wrapper), use it as a single search result
            elif isinstance(result, dict):
                # If the result is a dictionary, use it as a single search result
                search_result = {
                    "title": result.get("title", f"Result from {tool_name}"),
                    "href": result.get("href", result.get("url", f"mcp://{tool_name}")),
                    "body": result.get("body", result.get("content", str(result))),
                }
                search_results.append(search_result)
            else:
                # For any other type, convert to string and use as a single search result
                search_result = {
                    "title": f"Result from {tool_name}",
                    "href": f"mcp://{tool_name}",
                    "body": str(result),
                }
                search_results.append(search_result)
                
        except Exception as e:
            logger.error(f"Error processing tool result from {tool_name}: {e}")
            # Fallback: create a basic result
            search_result = {
                "title": f"Result from {tool_name}",
                "href": f"mcp://{tool_name}",
                "body": str(result),
            }
            search_results.append(search_result)
        
        return search_results 
