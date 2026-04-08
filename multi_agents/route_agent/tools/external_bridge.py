from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import os
import sys
import time
import uuid
from typing import Any, Dict, Iterator, List

from dotenv import dotenv_values

from ..models import RouteDecision, RouteRequest
from ..utils.model_utils import (
    build_model_identifier,
    normalize_route_agent_provider,
    split_model_identifier,
)
from .live_probe import LiveProbeMixin


@dataclass(slots=True)
class ExternalRouteRuntimeContext:
    """Runtime hooks for Route_Agent external routing."""

    request_id: str
    agent_name: str
    agent_class: str
    domain: str
    class_source: str
    record_id: int | None
    engine: Any
    router_storage: Any
    decision: Any
    escalation_manager: Any
    execution_attempt_cls: type[Any]
    _attempt_count: int = 0

    async def handle_execution_success(self, model_id: str) -> None:
        """Persist one successful execution into Route_Agent feedback storage."""
        self._attempt_count += 1
        self.escalation_manager.record_attempt(
            self.execution_attempt_cls(
                model_id=model_id,
                attempt_number=self._attempt_count,
                success=True,
            )
        )
        await self.engine.report_execution_async(
            request_id=self.request_id,
            record_id=self.record_id,
            model_id=model_id,
            agent_name=self.agent_name,
            agent_class=self.agent_class,
            domain=self.domain,
            completed=True,
            class_source=self.class_source,
        )
        await self.engine.report_quality_async(
            request_id=self.request_id,
            record_id=self.record_id,
            model_id=model_id,
            agent_name=self.agent_name,
            agent_class=self.agent_class,
            domain=self.domain,
            rating="good",
            action="keep",
            note="auto_success_completion",
            class_source=self.class_source,
        )

    async def handle_execution_failure(
        self,
        model_id: str,
        error_detail: str,
        *,
        hard_unavailable: bool = False,
        priority: str = "normal",
    ) -> str | None:
        """Persist one failed execution and ask Route_Agent for the next target."""
        self._attempt_count += 1
        self.escalation_manager.record_attempt(
            self.execution_attempt_cls(
                model_id=model_id,
                attempt_number=self._attempt_count,
                success=False,
                failure_type="exec",
                error_detail=error_detail,
            )
        )
        await self.engine.report_execution_async(
            request_id=self.request_id,
            record_id=self.record_id,
            model_id=model_id,
            agent_name=self.agent_name,
            agent_class=self.agent_class,
            domain=self.domain,
            completed=False,
            error_type="provider_error",
            error_detail=error_detail,
            class_source=self.class_source,
        )
        if hard_unavailable:
            await self.mark_model_unavailable(model_id)

        result = await self.escalation_manager.escalate_with_overload_check_async(
            current_model_id=model_id,
            priority=priority,
            decision=self.decision,
        )
        if result.action == "escalate" and result.next_model:
            return str(result.next_model)
        return None

    async def mark_model_unavailable(self, model_id: str) -> None:
        """Mark one Route_Agent model unavailable immediately."""
        if self.router_storage is None:
            return
        await self.router_storage.mark_unable_async(model_id)


class ExternalRouteAgentBridge(LiveProbeMixin):
    """Bridge into the sibling Route_Agent project for real routing decisions."""

    def __init__(
        self,
        *,
        project_path: str | Path | None = None,
        app_env_path: str | Path | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        default_project_path = repo_root.parent / "Route_Agent"
        self.project_path = Path(
            project_path
            or os.getenv("ROUTE_AGENT_PROJECT_PATH")
            or default_project_path
        ).expanduser()
        self.app_env_path = Path(
            app_env_path
            or os.getenv("ROUTE_AGENT_APP_ENV_PATH")
            or repo_root / ".env"
        ).expanduser()
        self._build_registry_context = None
        self._execute_route = None
        self._get_analysis_storage = None
        self._get_engine = None
        self._monitoring = None
        self._resolve_router_runtime = None
        self._route_agent_request_cls = None
        self._route_agent_run_options = None
        self._route_execution_attempt_cls = None
        self._init_probe_cache()

    def is_available(self) -> bool:
        """Return whether the sibling Route_Agent project exists on disk."""
        return self.project_path.exists()

    def describe_global_pool(self) -> List[Dict[str, str]]:
        """Return the Route_Agent global available-model pool."""
        if not self.is_available():
            return []

        self._ensure_imports()
        with self._app_env_overlay():
            options = self._route_agent_run_options(load_env_file=False)
            context = self._build_registry_context(options)
            entries: List[Dict[str, str]] = []
            for model in context.pool.list_available():
                model_id, provider, model_name = split_model_identifier(getattr(model, "model_id", ""))
                entries.append(
                    {
                        "model_id": model_id,
                        "provider": provider,
                        "model": model_name,
                    }
                )
            return entries

    def route(self, request: RouteRequest) -> RouteDecision:
        """Route one business request through Route_Agent and keep runtime hooks."""
        if not self.is_available():
            raise FileNotFoundError(f"Route_Agent project not found: {self.project_path}")

        self._ensure_imports()
        started_at = time.perf_counter()
        with self._app_env_overlay():
            options = self._route_agent_run_options(load_env_file=False)
            route_request = self._route_agent_request_cls.from_mapping(
                self._build_route_payload(request),
                default_agent_name=options.default_agent_name,
            )
            execution = self._execute_route(route_request, options)

        engine = self._build_engine_for_pool(execution.pool, options)
        probe_results = self._run_async(
            self._probe_entries_async(
                self._decision_candidate_entries(execution.decision),
                force=False,
                router_storage=getattr(engine, "_router_storage", None),
            )
        )
        probe_results_by_model = {
            str(item["model_id"]): item
            for item in probe_results
            if str(item.get("model_id") or "").strip()
        }
        filtered_decision = self._apply_live_probe_results(execution.decision, probe_results_by_model)

        route_latency_ms = round((time.perf_counter() - started_at) * 1000.0, 4)
        selected_model_id, selected_provider, selected_model = split_model_identifier(
            str(getattr(filtered_decision, "primary_model", "") or ""),
            fallback_provider=request.llm_provider,
        )
        if not selected_model:
            selected_model_id, selected_provider, selected_model = split_model_identifier(
                request.requested_model,
                fallback_provider=request.llm_provider,
            )

        runtime_context = ExternalRouteRuntimeContext(
            request_id=route_request.request_id,
            agent_name=route_request.agent_name,
            agent_class=str(
                getattr(filtered_decision, "pool_class", None)
                or execution.analysis_result.task_class
                or request.shared_agent_class
                or "general"
            ),
            domain=str(execution.analysis_result.domain or "general"),
            class_source=str(getattr(filtered_decision, "class_source", "route_agent")),
            record_id=execution.record_id,
            engine=engine,
            router_storage=getattr(engine, "_router_storage", None),
            decision=filtered_decision,
            escalation_manager=engine.create_escalation_manager(filtered_decision),
            execution_attempt_cls=self._route_execution_attempt_cls,
        )

        candidates = [
            self._convert_candidate(candidate, probe_results_by_model=probe_results_by_model)
            for candidate in list(getattr(filtered_decision, "candidates", ()) or ())
        ]
        removed_count = sum(1 for item in probe_results if not item.get("ok") and not item.get("skipped"))
        trace_context = {
            "application_name": request.application_name,
            "agent_role": request.agent_role,
            "stage_name": request.stage_name,
            "workflow_id": request.execution_context.workflow_id,
            "run_id": request.execution_context.run_id,
            "step_id": request.execution_context.step_id,
            "selected_model_id": selected_model_id,
            "route_agent_payload": {
                "routing_reason": str(getattr(filtered_decision, "reason", "") or ""),
                "alerts": list(getattr(filtered_decision, "alerts", ()) or ()),
                "default_used": bool(getattr(filtered_decision, "default_used", False)),
                "task_class": str(execution.analysis_result.task_class or ""),
                "domain": str(execution.analysis_result.domain or ""),
                "class_source": str(getattr(filtered_decision, "class_source", "") or ""),
                "pool_class": str(getattr(filtered_decision, "pool_class", "") or ""),
                "candidates": candidates,
                "live_probe_filtered": removed_count,
            },
        }

        return RouteDecision(
            selected_model=selected_model,
            selected_provider=selected_provider,
            selected_model_id=selected_model_id,
            candidates=candidates,
            route_latency_ms=route_latency_ms,
            trace_context=trace_context,
            routing_reason=str(getattr(filtered_decision, "reason", "") or ""),
            runtime_context=runtime_context,
        )

    def start_execution_tracking(self, request: RouteRequest, decision: RouteDecision) -> str:
        """Create one Route_Agent monitoring execution row for the business call."""
        if not self.is_available():
            return ""

        self._ensure_imports()
        execution_id = request.execution_context.step_id or uuid.uuid4().hex
        model_id = build_model_identifier(
            decision.selected_provider,
            decision.selected_model,
            target="route_agent",
        )
        with self._app_env_overlay():
            return self._monitoring.start_execution(
                {
                    "source": request.application_name,
                    "agent_name": self._build_agent_name(request),
                    "execution_id": execution_id,
                    "request_id": self._build_request_id(request),
                    "model_used": model_id or None,
                    "provider": normalize_route_agent_provider(decision.selected_provider) or None,
                    "status": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "metadata": {
                        "kind": "business_llm_execution",
                        "application_name": request.application_name,
                        "shared_agent_class": request.shared_agent_class,
                        "agent_role": request.agent_role,
                        "stage_name": request.stage_name,
                        "workflow_id": request.execution_context.workflow_id,
                        "run_id": request.execution_context.run_id,
                        "step_id": request.execution_context.step_id,
                        "route_latency_ms": decision.route_latency_ms,
                    },
                }
            )

    def end_execution_tracking(
        self,
        *,
        execution_id: str,
        status: str,
        duration_ms: float,
        error_message: str | None = None,
    ) -> bool:
        """Finalize one Route_Agent monitoring execution row."""
        if not execution_id or not self.is_available():
            return False

        self._ensure_imports()
        normalized_status = "success" if status == "completed" else "failed"
        with self._app_env_overlay():
            return bool(
                self._monitoring.end_execution(
                    {
                        "execution_id": execution_id,
                        "status": normalized_status,
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "duration_ms": duration_ms,
                        "error_message": error_message,
                        "metadata": {
                            "kind": "business_llm_execution",
                        },
                    }
                )
            )

    def _ensure_imports(self) -> None:
        """Import Route_Agent modules lazily."""
        if self._execute_route is not None:
            return

        if not self.is_available():
            raise FileNotFoundError(f"Route_Agent project not found: {self.project_path}")

        project_path_str = str(self.project_path)
        if project_path_str not in sys.path:
            sys.path.insert(0, project_path_str)

        self._build_registry_context = import_module("route_agent.app.registry").build_registry_context
        self._execute_route = import_module("route_agent.app.orchestrator").execute_route
        self._resolve_router_runtime = import_module("route_agent.app.orchestrator")._resolve_router_runtime
        self._get_analysis_storage = import_module("route_agent.app.wiring").get_analysis_storage
        self._get_engine = import_module("route_agent.app.wiring").get_engine
        self._monitoring = import_module("route_agent.monitoring")
        self._route_agent_request_cls = import_module("route_agent.app.contracts").RouteAgentRequest
        self._route_agent_run_options = import_module("route_agent.app.contracts").RouteAgentRunOptions
        self._route_execution_attempt_cls = import_module("route_agent.router_engine").ExecutionAttempt

    @contextmanager
    def _app_env_overlay(self) -> Iterator[None]:
        """Temporarily overlay the sibling app's .env into os.environ."""
        if not self.app_env_path.exists():
            yield
            return

        original: Dict[str, str | None] = {}
        updates = {
            key: str(value).strip()
            for key, value in dotenv_values(self.app_env_path).items()
            if value is not None and str(value).strip()
        }
        try:
            for key, value in updates.items():
                original[key] = os.environ.get(key)
                os.environ[key] = value
            yield
        finally:
            for key, previous in original.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

    def _build_route_payload(self, request: RouteRequest) -> Dict[str, Any]:
        """Translate the business route request into Route_Agent's public payload."""
        _, requested_provider, requested_model = split_model_identifier(
            request.requested_model,
            fallback_provider=request.llm_provider,
        )
        requested_model_id = build_model_identifier(
            requested_provider or request.llm_provider,
            requested_model,
            target="route_agent",
        )
        task = str(request.task or "").strip()
        if not task:
            task = str(request.system_prompt or "").strip()
        if not task:
            task = str(request.agent_role or request.stage_name or "route_request").strip()
        constraints: Dict[str, Any] = {}
        if requested_model_id:
            constraints["preferred_model"] = requested_model_id

        return {
            "task": task,
            "agent_name": self._build_agent_name(request),
            "system_prompt": request.system_prompt,
            "request_id": self._build_request_id(request),
            "agent_class": request.shared_agent_class or request.agent_role or request.stage_name or None,
            "constraints": constraints or None,
        }

    def _build_agent_name(self, request: RouteRequest) -> str:
        """Build the stable Route_Agent agent_name for one business request."""
        role = str(request.agent_role or request.stage_name or "agent").strip()
        return f"{request.application_name}.{role}" if role else request.application_name

    def _build_request_id(self, request: RouteRequest) -> str:
        """Build a stable request id for one business call."""
        parts = [
            str(request.application_name or "").strip(),
            str(request.execution_context.run_id or "").strip(),
            str(request.execution_context.step_id or "").strip(),
        ]
        resolved = ":".join(part for part in parts if part)
        return resolved or uuid.uuid4().hex

    def _build_route_agent_runtime_support(self) -> tuple[Any, Any]:
        """Build the shared Route_Agent runtime objects used by preflight hooks."""
        self._ensure_imports()
        with self._app_env_overlay():
            options = self._route_agent_run_options(load_env_file=False)
            context = self._build_registry_context(options)
            engine = self._build_engine_for_pool(context.pool, options)
        return options, engine

    def _build_engine_for_pool(self, pool: Any, options: Any) -> Any:
        """Build the Route_Agent engine for a known pool snapshot."""
        with self._app_env_overlay():
            analysis_storage = self._get_analysis_storage(options.analysis_db_path)
            redis_url, router_db_path, rate_limit_mode, rate_limit_fail_strategy = (
                self._resolve_router_runtime(options)
            )
            return self._get_engine(
                pool,
                analysis_storage,
                redis_url=redis_url,
                router_db_path=router_db_path,
                rate_limit_mode=rate_limit_mode,
                rate_limit_fail_strategy=rate_limit_fail_strategy,
            )

    def _decision_candidate_entries(self, decision: Any) -> List[Dict[str, str]]:
        """Return execution-order model entries for one Route_Agent decision."""
        entries: List[Dict[str, str]] = []
        seen: set[str] = set()

        def _append(model_id: str) -> None:
            resolved_id, provider, model_name = split_model_identifier(model_id)
            if not resolved_id or resolved_id in seen:
                return
            seen.add(resolved_id)
            entries.append(
                {
                    "model_id": resolved_id,
                    "provider": provider,
                    "model": model_name,
                }
            )

        primary_model = str(getattr(decision, "primary_model", "") or "")
        if primary_model:
            _append(primary_model)
        for candidate in list(getattr(decision, "candidates", ()) or ()):
            _append(str(getattr(candidate, "model_id", "") or ""))
        return entries

    def _apply_live_probe_results(self, decision: Any, probe_results_by_model: Dict[str, Dict[str, Any]]) -> Any:
        """Drop candidates that fail the live probe and reseat the start model."""
        candidates = list(getattr(decision, "candidates", ()) or ())
        if not candidates or not probe_results_by_model:
            return decision

        kept_candidates = []
        removed_models: list[str] = []
        kept_model_ids: set[str] = set()
        for candidate in candidates:
            model_id = str(getattr(candidate, "model_id", "") or "")
            probe_result = probe_results_by_model.get(model_id)
            if probe_result and not probe_result.get("ok") and not probe_result.get("skipped"):
                removed_models.append(model_id)
                continue
            kept_candidates.append(candidate)
            kept_model_ids.add(model_id)

        if not kept_candidates:
            return decision

        primary_model = str(getattr(decision, "primary_model", "") or "")
        if primary_model and primary_model not in kept_model_ids:
            ordered_model_ids = [
                primary_model,
                *[
                    str(getattr(candidate, "model_id", "") or "")
                    for candidate in candidates
                    if str(getattr(candidate, "model_id", "") or "") != primary_model
                ],
            ]
            primary_model = next(
                (model_id for model_id in ordered_model_ids if model_id in kept_model_ids),
                str(getattr(kept_candidates[0], "model_id", "") or primary_model),
            )

        if not removed_models:
            return decision

        start_index = next(
            (
                index
                for index, candidate in enumerate(kept_candidates)
                if str(getattr(candidate, "model_id", "") or "") == primary_model
            ),
            0,
        )
        alerts = tuple(getattr(decision, "alerts", ()) or ())
        reason = str(getattr(decision, "reason", "") or "")
        return replace(
            decision,
            primary_model=primary_model,
            candidates=tuple(kept_candidates),
            start_index=start_index,
            alerts=alerts + (f"live_probe_filtered={len(removed_models)}",),
            reason=f"{reason}; live_probe_filtered={len(removed_models)}",
        )

    def _convert_candidate(
        self,
        payload: Any,
        *,
        probe_results_by_model: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Normalize one Route_Agent candidate into the app-facing dict shape."""
        if isinstance(payload, dict):
            candidate = dict(payload)
        else:
            candidate = {
                "model_id": getattr(payload, "model_id", ""),
                "provider": getattr(payload, "provider", ""),
                "display_name": getattr(payload, "display_name", ""),
                "dimension_score": getattr(payload, "dimension_score", 0.0),
                "raw_dimension_score": getattr(payload, "raw_dimension_score", 0.0),
                "cost_score": getattr(payload, "cost_score", 0.0),
                "health_status": getattr(payload, "health_status", ""),
                "rate_limited": getattr(payload, "rate_limited", False),
                "is_default": getattr(payload, "is_default", False),
                "is_pool": getattr(payload, "is_pool", False),
                "is_explore": getattr(payload, "is_explore", False),
                "rank": getattr(payload, "rank", 0),
            }

        model_id, provider, model_name = split_model_identifier(
            str(candidate.get("model_id") or candidate.get("model") or "")
        )
        candidate["model"] = model_name
        candidate["provider"] = provider
        candidate["model_id"] = model_id
        if probe_results_by_model and model_id in probe_results_by_model:
            result = probe_results_by_model[model_id]
            candidate["live_probe"] = {
                "ok": bool(result.get("ok")),
                "skipped": bool(result.get("skipped")),
                "cached": bool(result.get("cached")),
                "message": str(result.get("message") or ""),
            }
        return candidate
