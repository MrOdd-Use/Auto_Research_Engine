from typing import Any

import json_repair
from langchain_community.adapters.openai import convert_openai_messages
from langchain_core.utils.json import parse_json_markdown
from loguru import logger

from gpt_researcher.config.config import Config
from gpt_researcher.llm_provider.generic.base import normalize_response_text
from gpt_researcher.utils.llm import create_chat_completion
from multi_agents.route_agent import RouteExecutionContext, RouteRequest


async def call_model(
    prompt: list,
    model: str,
    response_format: str | None = None,
    route_context: dict | None = None,
):

    cfg = Config()
    lc_messages = convert_openai_messages(prompt)

    try:
        route_request = _build_route_request(route_context, prompt, model, cfg.smart_llm_provider)
        response = await create_chat_completion(
            model=model,
            messages=lc_messages,
            temperature=0,
            llm_provider=cfg.smart_llm_provider,
            llm_kwargs=cfg.llm_kwargs,
            route_context={
                "route_request": route_request,
                "system_prompt": _extract_prompt_content(prompt, "system"),
                "task": _extract_prompt_content(prompt, "user"),
                "route_context": route_context or {},
            },
            # cost_callback=cost_callback,
        )

        response_text = _normalize_model_response(response)

        if response_format == "json":
            return parse_json_markdown(response_text, parser=json_repair.loads)

        return response_text

    except Exception as e:
        logger.error(f"Error in calling model: {e}")
        raise


def _normalize_model_response(response: Any) -> str:
    """Convert one model response payload into a plain-text string."""
    return normalize_response_text(response)


def _extract_prompt_content(prompt: list, role: str) -> str:
    parts = []
    for item in prompt or []:
        if str(item.get("role") or "").lower() != role:
            continue
        content = str(item.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts).strip()


def _build_route_request(
    route_context: dict | None,
    prompt: list,
    model: str,
    llm_provider: str,
) -> RouteRequest | None:
    if not route_context:
        return None

    execution_context = RouteExecutionContext(
        workflow_id=str(route_context.get("workflow_id") or ""),
        run_id=str(route_context.get("run_id") or ""),
        step_id=str(route_context.get("step_id") or ""),
        parent_step_id=str(route_context.get("parent_step_id") or ""),
        rollback_of_step_id=str(route_context.get("rollback_of_step_id") or ""),
        rollback_reason=str(route_context.get("rollback_reason") or ""),
        seed=route_context.get("seed"),
        tags=list(route_context.get("tags") or []),
    )
    return RouteRequest(
        application_name=str(route_context.get("application_name") or "auto_research_engine"),
        shared_agent_class=str(route_context.get("shared_agent_class") or ""),
        agent_role=str(route_context.get("agent_role") or ""),
        stage_name=str(route_context.get("stage_name") or ""),
        system_prompt=str(route_context.get("system_prompt") or _extract_prompt_content(prompt, "system")),
        task=str(route_context.get("task") or _extract_prompt_content(prompt, "user")),
        requested_model=None,
        llm_provider=llm_provider,
        execution_context=execution_context,
        metadata={
            "messages": prompt,
            "route_context": dict(route_context),
        },
    )
