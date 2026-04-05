"""Local-only adapter: Route_Agent full engine without a central federation server."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from route_agent.federation.client.lightweight_analysis import (
    build_lightweight_analysis_for_class,
)
from route_agent.federation.client.local_router import LocalRouter, LocalRouterConfig
from route_agent.federation.client.local_store import LocalStore

from .federation_adapter import DECLARED_AGENTS
from .models import RouteDecision, RouteRequest
from .utils.model_utils import split_model_identifier

logger = logging.getLogger(__name__)


class LocalOnlyAdapter:
    """Routes via Route_Agent's full engine (LocalStore + LocalRouter) with no server.

    Replaces the in-memory LayeredRoutingStore with persistent SQLite-backed scoring:
    - Wilson CI confidence intervals on model success rates
    - Class pool with seed, automatic promotion, downgrade, and eviction
    - Outcome feedback written synchronously to the local router-engine DB
    - Supports all 13 task classes (including the 4 Auto_Research-specific ones)

    Use ROUTE_AGENT_BACKEND=local_full to activate.
    """

    def __init__(
        self,
        *,
        app_id: str = "auto_research_engine",
        local_db_path: str | None = None,
        router_db_path: str | None = None,
    ) -> None:
        self._app_id = app_id
        _local_db = local_db_path or os.getenv(
            "ROUTE_AGENT_FEDERATION_LOCAL_DB", "data/federation_local.db"
        )
        _router_db = router_db_path or os.getenv(
            "ROUTE_AGENT_FEDERATION_ROUTER_DB", "data/router_engine.db"
        )
        self._local_store = LocalStore(_local_db)
        self._local_router = LocalRouter(LocalRouterConfig(router_db_path=_router_db))
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if not self._started:
            await self._save_declared_mappings()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            self._local_router.close()
            self._started = False

    async def _save_declared_mappings(self) -> None:
        """Persist DECLARED_AGENTS into local store at startup (idempotent)."""
        for agent_name, agent_class in DECLARED_AGENTS.items():
            await self._local_store.save_agent_mapping(
                app_id=self._app_id,
                agent_name=agent_name,
                agent_class=agent_class,
                source="declared",
            )
        logger.info(
            "LocalOnlyAdapter: persisted %d agent mappings | app_id=%s",
            len(DECLARED_AGENTS),
            self._app_id,
        )

    async def route(self, request: RouteRequest) -> tuple[RouteDecision, str | None]:
        """Route via local full engine. Returns (decision, lease_id=None)."""
        agent_name = (
            request.agent_role
            or request.stage_name
            or request.shared_agent_class
            or "agent"
        )
        task = request.task or request.system_prompt or "route_request"

        # Resolve agent class from persistent local store
        mapping = await self._local_store.get_agent_mapping(self._app_id, agent_name)
        agent_class = mapping.agent_class if mapping else "general"

        analysis = build_lightweight_analysis_for_class(agent_class)
        engine_decision = await self._local_router.route_known_agent(
            agent_name=agent_name,
            task=task,
            agent_class=agent_class,
            analysis=analysis,
        )

        candidates = _convert_candidates(engine_decision.candidates, request.llm_provider)
        selected_model = engine_decision.primary_model or request.requested_model or ""
        selected_provider = request.llm_provider or ""
        if selected_model:
            _, provider, _ = split_model_identifier(selected_model)
            if provider:
                selected_provider = provider

        decision = RouteDecision(
            selected_model=selected_model,
            selected_provider=selected_provider,
            candidates=candidates,
            resolved_shared_agent_class=agent_class,
            class_resolution_source="local_full",
            routing_reason=engine_decision.reason,
            trace_context={
                "application_name": request.application_name,
                "agent_role": request.agent_role,
                "stage_name": request.stage_name,
                "workflow_id": request.execution_context.workflow_id,
                "run_id": request.execution_context.run_id,
                "step_id": request.execution_context.step_id,
                "federation_mode": "local_full",
                "lease_id": None,
            },
        )
        return decision, None

    async def release(self, lease_id: str) -> None:
        """No-op: no central lease server in local-only mode."""

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
        """Write outcome synchronously to local router-engine DB."""
        try:
            self._local_router.process_outcome(
                lease_id=lease_id or "local",
                model_id=model_id,
                agent_class=agent_class,
                outcome_type=outcome_type,
                duration_ms=duration_ms,
                quality_score=quality_score,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LocalOnlyAdapter outcome processing failed: %s", exc)


def _convert_candidates(
    raw_candidates: Any, fallback_provider: str
) -> List[Dict[str, Any]]:
    """Normalize router-engine candidates to list-of-dict."""
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
