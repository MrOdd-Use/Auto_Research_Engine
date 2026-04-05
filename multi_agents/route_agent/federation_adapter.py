"""Federation adapter bridging RouteClient into Auto_Research_Engine routing."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx

from route_agent.federation.client.route_client import RouteClient

from .models import RouteDecision, RouteRequest
from .utils.model_utils import split_model_identifier

logger = logging.getLogger(__name__)

DECLARED_AGENTS: Dict[str, str] = {
    "planner_agent":        "planning",
    "research_agent":       "research",
    "scrape_agent":         "scrape",
    "check_data_agent":     "data_adequacy",
    "writer_agent":         "deep_writing",
    "review_agent":         "review",
    "reviser_agent":        "deep_writing",
    "claim_verifier_agent": "claim_verification",
    "publisher_agent":      "general",
}


def _ensure_route_agent_on_path() -> None:
    """No-op: route-agent is now an editable install dependency."""


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
        self._app_id = app_id
        self._server_url = server_url or os.getenv("ROUTE_AGENT_FEDERATION_URL", "")
        _local_db = local_db_path or os.getenv(
            "ROUTE_AGENT_FEDERATION_LOCAL_DB", "data/federation_local.db"
        )
        _router_db = router_db_path or os.getenv(
            "ROUTE_AGENT_FEDERATION_ROUTER_DB", "data/router_engine.db"
        )
        self._client = RouteClient(
            app_id=app_id,
            server_url=self._server_url,
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
            await self._register_agents()
            self._started = True

    async def _register_agents(self) -> None:
        """Register declared agents with the central federation server."""
        agents = [
            {"agent_name": name, "agent_class": cls, "agent_version": "v1"}
            for name, cls in DECLARED_AGENTS.items()
        ]
        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                response = await http.post(
                    f"{self._server_url}/api/v1/apps/register",
                    json={
                        "app_id": self._app_id,
                        "app_name": self._app_id,
                        "agents": agents,
                    },
                )
                response.raise_for_status()
                data = response.json()
                logger.info(
                    "Registered %d declared agents with federation | app_id=%s pool_versions=%s",
                    len(agents),
                    self._app_id,
                    data.get("pool_versions", {}),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent registration failed (non-fatal): %s", exc)

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
