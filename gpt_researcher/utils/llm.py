"""LLM utilities for Auto_Research_Engine.

This module provides utility functions for interacting with various
LLM providers through a unified interface.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from gpt_researcher.llm_provider.generic.base import (
    NO_SUPPORT_TEMPERATURE_MODELS,
    SUPPORT_REASONING_EFFORT_MODELS,
    ReasoningEfforts,
)

from ..prompts import PromptFamily
from .costs import estimate_llm_cost
from .validators import Subtopics
from multi_agents.route_agent import get_global_invoker
from multi_agents.route_agent.utils.model_utils import normalize_app_provider


def get_llm(llm_provider: str, **kwargs):
    """Get an LLM provider instance.

    Args:
        llm_provider: The name of the LLM provider (e.g., 'openai', 'anthropic').
        **kwargs: Additional keyword arguments passed to the provider.

    Returns:
        A GenericLLMProvider instance configured for the specified provider.
    """
    from gpt_researcher.llm_provider import GenericLLMProvider
    return GenericLLMProvider.from_provider(llm_provider, **kwargs)


async def create_chat_completion(
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = 0.4,
        max_tokens: int | None = 4000,
        llm_provider: str | None = None,
        stream: bool = False,
        websocket: Any | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        cost_callback: callable = None,
        reasoning_effort: str | None = ReasoningEfforts.Medium.value,
        route_context: dict[str, Any] | None = None,
        **kwargs
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion.
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.4.
        max_tokens (int, optional): The max tokens to use. Defaults to 4000.
        llm_provider (str, optional): The LLM Provider to use.
        stream (bool): Whether to stream the response. Defaults to False.
        webocket (WebSocket): The websocket used in the currect request,
        llm_kwargs (dict[str, Any], optional): Additional LLM keyword arguments. Defaults to None.
        cost_callback: Callback function for updating cost.
        reasoning_effort (str, optional): Reasoning effort for OpenAI's reasoning models. Defaults to 'low'.
        **kwargs: Additional keyword arguments.
    Returns:
        str: The response from the chat completion.
    """
    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 32001:
        raise ValueError(
            f"Max tokens cannot be more than 32,000, but got {max_tokens}")

    base_provider_kwargs = dict(llm_kwargs or {})

    async def _provider_call(selected_model: str, selected_provider: str = "") -> str:
        effective_provider = normalize_app_provider(selected_provider or llm_provider or "")
        selected_provider_kwargs = dict(base_provider_kwargs)
        selected_provider_kwargs["model"] = selected_model
        if selected_model in SUPPORT_REASONING_EFFORT_MODELS:
            selected_provider_kwargs["reasoning_effort"] = reasoning_effort
        else:
            selected_provider_kwargs.pop("reasoning_effort", None)

        if selected_model not in NO_SUPPORT_TEMPERATURE_MODELS:
            selected_provider_kwargs["temperature"] = temperature
            selected_provider_kwargs["max_tokens"] = max_tokens
        else:
            selected_provider_kwargs["temperature"] = None
            selected_provider_kwargs["max_tokens"] = None

        if effective_provider == "openai":
            base_url = os.environ.get("OPENAI_BASE_URL", None)
            if base_url:
                selected_provider_kwargs["openai_api_base"] = base_url
        provider = get_llm(effective_provider, **selected_provider_kwargs)
        response = ""
        for _ in range(10):  # maximum of 10 attempts
            response = await provider.get_chat_response(
                messages, stream, websocket, **kwargs
            )
            if cost_callback:
                llm_costs = estimate_llm_cost(str(messages), response)
                cost_callback(llm_costs)
            return response
        logging.error(f"Failed to get response from {effective_provider} API")
        raise RuntimeError(f"Failed to get response from {effective_provider} API")

    invoker = get_global_invoker()
    if route_context:
        return await invoker.invoke(
            provider_call=_provider_call,
            requested_model=model,
            llm_provider=llm_provider or "",
            route_request=route_context.get("route_request"),
            metadata={
                "messages": messages,
                "system_prompt": route_context.get("system_prompt") or _extract_system_prompt(messages),
                "task": route_context.get("task") or _extract_user_task(messages),
                "route_context": route_context,
            },
        )

    return await invoker.invoke(
        provider_call=_provider_call,
        requested_model=model,
        llm_provider=llm_provider or "",
        metadata={
            "messages": messages,
            "system_prompt": _extract_system_prompt(messages),
            "task": _extract_user_task(messages),
        },
    )


async def construct_subtopics(
    task: str,
    data: str,
    config,
    subtopics: list = [],
    prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
    **kwargs
) -> list:
    """
    Construct subtopics based on the given task and data.

    Args:
        task (str): The main task or topic.
        data (str): Additional data for context.
        config: Configuration settings.
        subtopics (list, optional): Existing subtopics. Defaults to [].
        prompt_family (PromptFamily): Family of prompts
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of constructed subtopics.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=Subtopics)

        prompt = PromptTemplate(
            template=prompt_family.generate_subtopics_prompt(),
            input_variables=["task", "data", "subtopics", "max_subtopics"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )

        prompt_value = prompt.format(
            task=task,
            data=data,
            subtopics=subtopics,
            max_subtopics=config.max_subtopics,
        )
        output = await create_chat_completion(
            messages=[{"role": "user", "content": prompt_value}],
            model=config.smart_llm_model,
            temperature=config.temperature,
            max_tokens=config.smart_token_limit,
            llm_provider=config.smart_llm_provider,
            llm_kwargs=config.llm_kwargs,
            reasoning_effort=ReasoningEfforts.High.value,
            **kwargs,
        )
        return parser.parse(output)

    except Exception as e:
        print("Exception in parsing subtopics : ", e)
        logging.getLogger(__name__).error("Exception in parsing subtopics : \n {e}")
        return subtopics


def _extract_system_prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for item in messages or []:
        if str(item.get("role") or "").lower() != "system":
            continue
        content = str(item.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts).strip()


def _extract_user_task(messages: list[dict[str, str]]) -> str:
    parts = []
    for item in messages or []:
        if str(item.get("role") or "").lower() not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts).strip()
