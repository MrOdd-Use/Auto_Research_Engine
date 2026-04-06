"""local_study adapter: Route_Agent full engine + local SQLite learning.

Combines ExternalRouteAgentBridge (routing, escalation, monitoring) with
local persistent learning (LocalRouter Wilson CI scoring + LocalStore).

Use ROUTE_AGENT_BACKEND=local_study to activate.
"""

from __future__ import annotations

import logging
import os
import sys as _sys
from dataclasses import replace
from typing import Any

# Ensure Route_Agent project is importable before loading its modules.
_ra_path = str(os.getenv("ROUTE_AGENT_PROJECT_PATH") or "").strip().strip('"')
if _ra_path and _ra_path not in _sys.path:
    _sys.path.insert(0, _ra_path)

from route_agent.federation.client.local_router import LocalRouter, LocalRouterConfig
from route_agent.federation.client.local_store import LocalStore

from .federation_adapter import DECLARED_AGENTS
from .models import RouteDecision, RouteRequest
from .tools.external_bridge import ExternalRouteAgentBridge, ExternalRouteRuntimeContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local learning layer
# ---------------------------------------------------------------------------


class LocalStudyLearningLayer:
    """Local SQLite persistence: LocalRouter + LocalStore."""

    def __init__(
        self,
        *,
        app_id: str = "auto_research_engine",
        local_db_path: str | None = None,
        router_db_path: str | None = None,
    ) -> None:
        self._app_id = app_id
        _local_db = local_db_path or os.getenv(
            "ROUTE_AGENT_LOCAL_LEARN_DB", "data/local_study.db"
        )
        _router_db = router_db_path or os.getenv(
            "ROUTE_AGENT_FEDERATION_ROUTER_DB", "data/router_engine.db"
        )
        self._local_store = LocalStore(_local_db)
        self._local_router = LocalRouter(LocalRouterConfig(router_db_path=_router_db))
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        for agent_name, agent_class in DECLARED_AGENTS.items():
            await self._local_store.save_agent_mapping(
                app_id=self._app_id,
                agent_name=agent_name,
                agent_class=agent_class,
                source="declared",
            )
        self._started = True
        logger.info(
            "LocalStudyLearningLayer: persisted %d agent mappings | app_id=%s",
            len(DECLARED_AGENTS),
            self._app_id,
        )

    def close(self) -> None:
        self._local_router.close()

    def report_outcome(
        self,
        *,
        model_id: str,
        agent_class: str,
        outcome_type: str,
        duration_ms: int | None = None,
        quality_score: float | None = None,
    ) -> None:
        """Write outcome to local SQLite. Logs warning on failure, never raises."""
        try:
            self._local_router.process_outcome(
                lease_id="local_study",
                model_id=model_id,
                agent_class=agent_class,
                outcome_type=outcome_type,
                duration_ms=duration_ms,
                quality_score=quality_score,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LocalStudyLearningLayer write failed: %s", exc)


# ---------------------------------------------------------------------------
# Runtime context wrapper (escalation + local learning dual-write)
# ---------------------------------------------------------------------------


class LocalStudyRuntimeContext:
    """Wraps ExternalRouteRuntimeContext + local learning dual-write.

    Same interface as ExternalRouteRuntimeContext so the invoker's
    escalation loop works unchanged via duck typing.
    """

    def __init__(
        self,
        external_ctx: ExternalRouteRuntimeContext,
        learning_layer: LocalStudyLearningLayer,
        agent_class: str,
    ) -> None:
        self._external = external_ctx
        self._learning = learning_layer
        self._agent_class = agent_class

    async def handle_execution_success(self, model_id: str) -> None:
        await self._external.handle_execution_success(model_id)
        self._learning.report_outcome(
            model_id=model_id,
            agent_class=self._agent_class,
            outcome_type="exec_success",
        )

    async def handle_execution_failure(
        self,
        model_id: str,
        error_detail: str,
        *,
        hard_unavailable: bool = False,
        priority: str = "normal",
    ) -> str | None:
        next_model = await self._external.handle_execution_failure(
            model_id,
            error_detail,
            hard_unavailable=hard_unavailable,
            priority=priority,
        )
        self._learning.report_outcome(
            model_id=model_id,
            agent_class=self._agent_class,
            outcome_type="exec_fail",
        )
        return next_model

    async def mark_model_unavailable(self, model_id: str) -> None:
        await self._external.mark_model_unavailable(model_id)


# ---------------------------------------------------------------------------
# Unified adapter
# ---------------------------------------------------------------------------


class LocalStudyAdapter:
    """Unified adapter: Route_Agent full engine + local SQLite learning.

    - Primary routing via ExternalRouteAgentBridge (escalation, monitoring, probe)
    - Local learning via LocalStudyLearningLayer (Wilson CI, persistent SQLite)
    - RuntimeContext dual-writes on every success/failure callback
    """

    def __init__(
        self,
        *,
        app_id: str = "auto_research_engine",
        project_path: str | None = None,
        app_env_path: str | None = None,
        local_db_path: str | None = None,
        router_db_path: str | None = None,
    ) -> None:
        self._bridge = ExternalRouteAgentBridge(
            project_path=project_path,
            app_env_path=app_env_path,
        )
        self._learning = LocalStudyLearningLayer(
            app_id=app_id,
            local_db_path=local_db_path,
            router_db_path=router_db_path,
        )
        self._started = False

    @property
    def bridge(self) -> ExternalRouteAgentBridge:
        return self._bridge

    @property
    def learning(self) -> LocalStudyLearningLayer:
        return self._learning

    async def start(self) -> None:
        if not self._started:
            await self._learning.start()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            self._learning.close()
            self._started = False

    def route(self, request: RouteRequest) -> RouteDecision:
        """Route via Route_Agent, wrapping runtime_context with local learning."""
        decision = self._bridge.route(request)
        external_ctx = decision.runtime_context
        if external_ctx is not None:
            agent_class = decision.resolved_shared_agent_class or "general"
            decision = replace(
                decision,
                runtime_context=LocalStudyRuntimeContext(
                    external_ctx=external_ctx,
                    learning_layer=self._learning,
                    agent_class=agent_class,
                ),
            )
        return decision

    def start_execution_tracking(
        self, request: RouteRequest, decision: RouteDecision
    ) -> str:
        return self._bridge.start_execution_tracking(request, decision)

    def end_execution_tracking(
        self,
        *,
        execution_id: str,
        status: str,
        duration_ms: float,
        error_message: str | None = None,
    ) -> bool:
        return self._bridge.end_execution_tracking(
            execution_id=execution_id,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
        )
