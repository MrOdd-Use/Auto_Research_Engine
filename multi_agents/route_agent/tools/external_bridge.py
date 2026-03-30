from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import os
import sys
import time
import uuid
from typing import Any, Dict, Iterator, List

from dotenv import dotenv_values

from ..utils.model_utils import build_model_identifier, normalize_route_agent_provider, split_model_identifier
from ..models import RouteDecision, RouteRequest


class ExternalRouteAgentBridge:
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
        self._run_route_agent = None
        self._build_registry_context = None
        self._route_agent_run_options = None
        self._monitoring = None

    def is_available(self) -> bool:
        return self.project_path.exists()

    def describe_global_pool(self) -> List[Dict[str, str]]:
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
        if not self.is_available():
            raise FileNotFoundError(f"Route_Agent project not found: {self.project_path}")

        self._ensure_imports()
        started_at = time.perf_counter()
        with self._app_env_overlay():
            payload = self._run_route_agent(
                self._build_route_payload(request),
                load_env_file=False,
            )

        route_latency_ms = round((time.perf_counter() - started_at) * 1000.0, 4)
        selected_model_id, selected_provider, selected_model = split_model_identifier(
            str(payload.get("model_used") or ""),
            fallback_provider=request.llm_provider,
        )
        if not selected_model:
            selected_model_id, selected_provider, selected_model = split_model_identifier(
                request.requested_model,
                fallback_provider=request.llm_provider,
            )
        candidates = [self._convert_candidate(item) for item in list(payload.get("candidates") or [])]
        return RouteDecision(
            selected_model=selected_model,
            selected_provider=selected_provider,
            candidates=candidates,
            resolved_shared_agent_class=str(payload.get("pool_class") or request.shared_agent_class or ""),
            class_resolution_source=str(payload.get("class_source") or "route_agent"),
            route_latency_ms=route_latency_ms,
            analysis_latency_ms=float(payload.get("analysis_latency_ms") or 0.0),
            registry_latency_ms=0.0,
            selection_latency_ms=0.0,
            score_breakdown={},
            trace_context={
                "application_name": request.application_name,
                "agent_role": request.agent_role,
                "stage_name": request.stage_name,
                "workflow_id": request.execution_context.workflow_id,
                "run_id": request.execution_context.run_id,
                "step_id": request.execution_context.step_id,
                "selected_model_id": selected_model_id,
                "route_agent_payload": {
                    "routing_reason": str(payload.get("routing_reason") or ""),
                    "alerts": list(payload.get("alerts") or []),
                    "default_used": bool(payload.get("default_used")),
                    "pool_hit": bool(payload.get("pool_hit")),
                },
            },
            routing_reason=str(payload.get("routing_reason") or ""),
        )

    def start_execution_tracking(self, request: RouteRequest, decision: RouteDecision) -> str:
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
        if self._run_route_agent is not None:
            return

        if not self.is_available():
            raise FileNotFoundError(f"Route_Agent project not found: {self.project_path}")

        project_path_str = str(self.project_path)
        if project_path_str not in sys.path:
            sys.path.insert(0, project_path_str)

        self._run_route_agent = import_module("route_agent.app.service").run_route_agent
        self._build_registry_context = import_module("route_agent.app.registry").build_registry_context
        self._route_agent_run_options = import_module("route_agent.app.contracts").RouteAgentRunOptions
        self._monitoring = import_module("route_agent.monitoring")

    @contextmanager
    def _app_env_overlay(self) -> Iterator[None]:
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
        requested_model_id = build_model_identifier(
            request.llm_provider,
            request.requested_model,
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
        elif request.llm_provider:
            constraints["require_provider"] = normalize_route_agent_provider(request.llm_provider)

        return {
            "task": task,
            "agent_name": self._build_agent_name(request),
            "system_prompt": request.system_prompt,
            "request_id": self._build_request_id(request),
            "agent_class": request.shared_agent_class or request.agent_role or request.stage_name or None,
            "constraints": constraints or None,
        }

    def _build_agent_name(self, request: RouteRequest) -> str:
        role = str(request.agent_role or request.stage_name or "agent").strip()
        return f"{request.application_name}.{role}" if role else request.application_name

    def _build_request_id(self, request: RouteRequest) -> str:
        parts = [
            str(request.application_name or "").strip(),
            str(request.execution_context.run_id or "").strip(),
            str(request.execution_context.step_id or "").strip(),
        ]
        resolved = ":".join(part for part in parts if part)
        return resolved or uuid.uuid4().hex

    def _convert_candidate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        model_id, provider, model_name = split_model_identifier(
            str(payload.get("model_id") or payload.get("model") or "")
        )
        candidate = dict(payload)
        candidate["model"] = model_name
        candidate["provider"] = provider
        candidate["model_id"] = model_id
        return candidate
