from __future__ import annotations

from typing import Any, Dict

from ..models import RouteExecutionContext, RouteScope


def build_route_context(
    *,
    application_name: str,
    shared_agent_class: str,
    agent_role: str,
    stage_name: str,
    system_prompt: str = "",
    task: str = "",
    state: Dict[str, Any] | None = None,
    task_payload: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    state = state or {}
    task_payload = task_payload or {}
    execution = dict(task_payload.get("route_execution_context") or {})
    execution.update(dict(state.get("route_execution_context") or {}))
    if extra:
        execution.update(extra)
    return {
        "application_name": application_name,
        "shared_agent_class": shared_agent_class,
        "agent_role": agent_role,
        "stage_name": stage_name,
        "system_prompt": system_prompt,
        "task": task,
        "workflow_id": str(execution.get("workflow_id") or ""),
        "run_id": str(execution.get("run_id") or ""),
        "step_id": str(execution.get("step_id") or ""),
        "parent_step_id": str(execution.get("parent_step_id") or ""),
        "rollback_of_step_id": str(execution.get("rollback_of_step_id") or ""),
        "rollback_reason": str(execution.get("rollback_reason") or ""),
        "seed": execution.get("seed"),
        "tags": list(execution.get("tags") or []),
    }


def build_route_scope(
    *,
    application_name: str,
    shared_agent_class: str,
    agent_role: str,
    stage_name: str,
    system_prompt: str = "",
    task: str = "",
    state: Dict[str, Any] | None = None,
    task_payload: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> RouteScope:
    context = build_route_context(
        application_name=application_name,
        shared_agent_class=shared_agent_class,
        agent_role=agent_role,
        stage_name=stage_name,
        system_prompt=system_prompt,
        task=task,
        state=state,
        task_payload=task_payload,
        extra=extra,
    )
    return RouteScope(
        application_name=context["application_name"],
        shared_agent_class=context["shared_agent_class"],
        agent_role=context["agent_role"],
        stage_name=context["stage_name"],
        system_prompt=context["system_prompt"],
        task=context["task"],
        execution_context=RouteExecutionContext(
            workflow_id=context["workflow_id"],
            run_id=context["run_id"],
            step_id=context["step_id"],
            parent_step_id=context["parent_step_id"],
            rollback_of_step_id=context["rollback_of_step_id"],
            rollback_reason=context["rollback_reason"],
            seed=context["seed"],
            tags=context["tags"],
        ),
    )
