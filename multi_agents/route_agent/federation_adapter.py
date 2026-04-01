"""Federation adapter bridging RouteClient into Auto_Research_Engine routing."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from .models import RouteDecision, RouteRequest
from .utils.model_utils import split_model_identifier

logger = logging.getLogger(__name__)

DECLARED_AGENTS: Dict[str, str] = {
    "planner_agent": "summarization",
    "research_agent": "general",
    "scrape_agent": "scrape",
    "check_data_agent": "review",
    "writer_agent": "rewrite",
    "review_agent": "review",
    "reviser_agent": "rewrite",
    "claim_verifier_agent": "review",
    "publisher_agent": "general",
}


def _ensure_route_agent_on_path() -> None:
    """Add the Route_Agent project to sys.path if not already importable."""
    try:
        import route_agent  # noqa: F401
        return
    except ImportError:
        pass
    repo_root = Path(__file__).resolve().parents[3]
    env_path = os.getenv("ROUTE_AGENT_PROJECT_PATH")
    project_path = Path(env_path) if env_path else repo_root / "Route_Agent"
    resolved = str(project_path.resolve())
    if resolved not in sys.path and project_path.exists():
        sys.path.insert(0, resolved)


class FederationAdapter:
    """Bridges the federation RouteClient into the local routing layer."""

    def __init__(
        self,
        *,
        app_id: str = "auto_research_engine",
        server_url: str | None = None,
        local_db_path: str | None = None,
        router_db_path: str | None = None,
    ) -> None:
        _ensure_route_agent_on_path()
        from route_agent.federation.client.route_client import RouteClient

        self._app_id = app_id
        _server_url = server_url or os.getenv("ROUTE_AGENT_FEDERATION_URL", "")
        _local_db = local_db_path or os.getenv(
            "ROUTE_AGENT_FEDERATION_LOCAL_DB", "data/federation_local.db"
        )
        _router_db = router_db_path or os.getenv(
            "ROUTE_AGENT_FEDERATION_ROUTER_DB", "data/router_engine.db"
        )
        self._client = RouteClient(
            app_id=app_id,
            server_url=_server_url,
            local_db_path=_local_db,
            router_db_path=_router_db,
        )
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if not self._started:
            await self._client.start()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._client.stop()
            self._started = False

    async def route(self, request: RouteRequest) -> tuple[RouteDecision, str | None]:
        """Route via federation client, returning (decision, lease_id)."""
        agent_name = (
            request.agent_role
            or request.stage_name
            or request.shared_agent_class
            or "agent"
        )
        task = request.task or request.system_prompt or "route_request"
        result = await self._client.route(agent_name, task)

        candidates = _convert_candidates(result.candidates, request.llm_provider)
        selected_provider = request.llm_provider or ""
        if result.model:
            _, provider, _ = split_model_identifier(result.model)
            if provider:
                selected_provider = provider

        decision = RouteDecision(
            selected_model=result.model or request.requested_model or "",
            selected_provider=selected_provider,
            candidates=candidates,
            resolved_shared_agent_class=result.analysis.task_class,
            class_resolution_source=f"federation_{result.mode}",
            routing_reason=result.local_reason,
            trace_context={
                "application_name": request.application_name,
                "agent_role": request.agent_role,
                "stage_name": request.stage_name,
                "workflow_id": request.execution_context.workflow_id,
                "run_id": request.execution_context.run_id,
                "step_id": request.execution_context.step_id,
                "federation_mode": result.mode,
                "lease_id": result.lease_id,
            },
        )
        return decision, result.lease_id

    async def release(self, lease_id: str) -> None:
        """Release a concurrency lease."""
        if lease_id:
            await self._client.release(lease_id)

    async def report_outcome(
        self,
        *,
        lease_id: str,
        model_id: str,
        agent_class: str,
        outcome_type: str,
        duration_ms: int | None = None,
        quality_score: float | None = None,
    ) -> None:
        """Report execution outcome via dual-write (local + central)."""
        if lease_id:
            await self._client.report_outcome(
                lease_id=lease_id,
                model_id=model_id,
                agent_class=agent_class,
                outcome_type=outcome_type,
                duration_ms=duration_ms,
                quality_score=quality_score,
            )


def _convert_candidates(
    raw_candidates: Any, fallback_provider: str
) -> List[Dict[str, Any]]:
    """Normalize federation RouteResult.candidates to list-of-dict."""
    result: List[Dict[str, Any]] = []
    for candidate in raw_candidates or []:
        if isinstance(candidate, dict):
            result.append(dict(candidate))
        elif hasattr(candidate, "model_id"):
            _, provider, model = split_model_identifier(candidate.model_id)
            result.append({
                "model": model,
                "provider": provider or fallback_provider,
                "model_id": candidate.model_id,
                "score": getattr(candidate, "composite_score", 0.0),
            })
        else:
            result.append({"model": str(candidate)})
    return result
