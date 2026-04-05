from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from dotenv import dotenv_values

from .tools.external_bridge import ExternalRouteAgentBridge
from .utils.model_utils import build_model_identifier, split_model_identifier
from .models import RouteDecision, RouteRequest
from .storage.store import LayeredRoutingStore

if TYPE_CHECKING:
    from .federation_adapter import FederationAdapter
    from .local_adapter import LocalOnlyAdapter


class RouteAgentClient:
    """Route_Agent client with explicit local-test and external-runtime backends."""

    CLASS_ALIASES = {
        "planner": "planner_agent",
        "planner_agent": "planner_agent",
        "editor": "planner_agent",
        "research": "research_agent",
        "researcher": "research_agent",
        "browser": "research_agent",
        "scrap": "scrape_agent",
        "scrape": "scrape_agent",
        "scrape_agent": "scrape_agent",
        "check_data": "check_data_agent",
        "check_data_agent": "check_data_agent",
        "writer": "writer_agent",
        "writer_agent": "writer_agent",
        "reviewer": "review_agent",
        "review": "review_agent",
        "review_agent": "review_agent",
        "reviser": "reviser_agent",
        "revise": "reviser_agent",
        "reviser_agent": "reviser_agent",
        "claim_verifier": "claim_verifier_agent",
        "claim_verifier_agent": "claim_verifier_agent",
        "challenger": "review_agent",
        "publisher": "publisher_agent",
    }

    MODEL_PROFILES = {
        "gpt-4o-mini": {
            "general": 1.2,
            "reasoning": 0.6,
            "research": 0.8,
            "writing": 0.9,
            "review": 0.7,
            "cost_efficiency": 1.0,
        },
        "gpt-4o": {
            "general": 1.4,
            "reasoning": 1.1,
            "research": 1.2,
            "writing": 1.3,
            "review": 1.2,
            "cost_efficiency": 0.6,
        },
        "o1-preview": {
            "general": 1.2,
            "reasoning": 1.6,
            "research": 1.1,
            "writing": 1.0,
            "review": 1.5,
            "cost_efficiency": 0.2,
        },
        "o4-mini": {
            "general": 1.3,
            "reasoning": 1.3,
            "research": 1.1,
            "writing": 1.1,
            "review": 1.4,
            "cost_efficiency": 0.5,
        },
    }

    def __init__(
        self,
        *,
        store: LayeredRoutingStore | None = None,
        model_pool: List[str] | None = None,
        application_name: str = "auto_research_engine",
        backend: str | None = None,
        external_project_path: str | None = None,
        app_env_path: str | None = None,
        federation_url: str | None = None,
        federation_local_db: str | None = None,
        federation_router_db: str | None = None,
    ) -> None:
        self.store = store or LayeredRoutingStore()
        self.application_name = application_name
        self._model_provider_map: Dict[str, str] = {}
        self._external_error = ""
        self._external_pool_cache: List[Dict[str, str]] | None = None
        self.backend = self._resolve_backend(backend, store=store, model_pool=model_pool)
        self._external_bridge = (
            ExternalRouteAgentBridge(project_path=external_project_path, app_env_path=app_env_path)
            if self.backend == "external"
            else None
        )
        self._federation: "FederationAdapter | None" = None
        self._local_full: "LocalOnlyAdapter | None" = None
        if self.backend == "federation":
            from .federation_adapter import FederationAdapter
            self._federation = FederationAdapter(
                app_id=application_name,
                server_url=federation_url,
                local_db_path=federation_local_db,
                router_db_path=federation_router_db,
            )
        elif self.backend == "local_full":
            from .local_adapter import LocalOnlyAdapter
            self._local_full = LocalOnlyAdapter(
                app_id=application_name,
                local_db_path=federation_local_db,
                router_db_path=federation_router_db,
            )
        self.model_pool = self._resolve_model_pool(model_pool)

    @property
    def is_external_backend(self) -> bool:
        return self.backend == "external"

    @property
    def is_federation(self) -> bool:
        return self._federation is not None

    @property
    def is_local_full(self) -> bool:
        return self._local_full is not None

    @property
    def federation(self) -> "FederationAdapter | None":
        return self._federation

    @property
    def local_full(self) -> "LocalOnlyAdapter | None":
        return self._local_full

    @property
    def external_error(self) -> str:
        return self._external_error

    @property
    def external_bridge(self) -> ExternalRouteAgentBridge | None:
        """Expose the external bridge for runtime preflight and feedback hooks."""
        return self._external_bridge

    async def astart(self) -> None:
        """Start async resources (federation / local_full background tasks)."""
        if self._federation is not None:
            await self._federation.start()
        if self._local_full is not None:
            await self._local_full.start()

    async def astop(self) -> None:
        """Stop async resources."""
        if self._federation is not None:
            await self._federation.stop()
        if self._local_full is not None:
            await self._local_full.stop()

    def route(self, request: RouteRequest) -> RouteDecision:
        if self._external_bridge is not None:
            return self._external_bridge.route(request)
        return self._route_local(request)

    def record_quality_success(self, application_name: str, shared_agent_class: str, model_id: str) -> None:
        if self._external_bridge is not None:
            return
        self.store.mark_success(application_name, shared_agent_class, model_id)

    def record_quality_failure(self, application_name: str, shared_agent_class: str, model_id: str) -> None:
        if self._external_bridge is not None:
            return
        self.store.mark_quality_failure(application_name, shared_agent_class, model_id)

    def record_execution_failure(
        self,
        application_name: str,
        shared_agent_class: str,
        model_id: str,
        *,
        provider_failure: bool = False,
        unavailable: bool = False,
    ) -> None:
        if self._external_bridge is not None:
            return
        self.store.mark_exec_failure(
            application_name,
            shared_agent_class,
            model_id,
            provider_failure=provider_failure,
            unavailable=unavailable,
        )

    def start_execution_tracking(self, request: RouteRequest, decision: RouteDecision) -> str:
        if self._external_bridge is None:
            return ""
        return self._external_bridge.start_execution_tracking(request, decision)

    def end_execution_tracking(
        self,
        *,
        execution_id: str,
        status: str,
        duration_ms: float,
        error_message: str | None = None,
    ) -> bool:
        if self._external_bridge is None:
            return False
        return self._external_bridge.end_execution_tracking(
            execution_id=execution_id,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
        )

    def describe_model_pool(self) -> List[Dict[str, str]]:
        if self._external_bridge is not None:
            try:
                self._external_error = ""
                if self._external_pool_cache is None:
                    self._external_pool_cache = self._external_bridge.describe_global_pool()
                return list(self._external_pool_cache)
            except Exception as exc:  # noqa: BLE001
                self._external_error = str(exc)
                self._external_pool_cache = None
                return []

        entries: List[Dict[str, str]] = []
        for model_name in self.model_pool:
            provider = self._model_provider_map.get(model_name, "")
            entries.append(
                {
                    "model_id": build_model_identifier(provider, model_name),
                    "provider": provider,
                    "model": model_name,
                }
            )
        return entries

    def _resolve_backend(
        self,
        backend: str | None,
        *,
        store: LayeredRoutingStore | None,
        model_pool: List[str] | None,
    ) -> str:
        explicit = str(backend or "").strip().lower()
        if explicit:
            return explicit
        if store is not None or model_pool is not None:
            return "local"
        env_backend = str(os.getenv("ROUTE_AGENT_BACKEND") or "").strip().lower()
        if env_backend:
            return env_backend
        repo_env_path = Path(__file__).resolve().parents[2] / ".env"
        if repo_env_path.exists():
            file_backend = str(dotenv_values(repo_env_path).get("ROUTE_AGENT_BACKEND") or "").strip().lower()
            if file_backend:
                return file_backend
        return "local"

    def _resolve_model_pool(self, model_pool: List[str] | None) -> List[str]:
        if model_pool is not None:
            return self._normalize_model_pool(model_pool)
        if self._external_bridge is not None:
            try:
                self._external_error = ""
                entries = self.describe_model_pool()
                return self._normalize_model_pool([entry["model_id"] for entry in entries])
            except Exception as exc:  # noqa: BLE001
                self._external_error = str(exc)
                self._external_pool_cache = None
                return []
        return self._normalize_model_pool(self._default_model_pool())

    def _route_local(self, request: RouteRequest) -> RouteDecision:
        start = time.perf_counter()
        analysis_start = time.perf_counter()
        resolved_class, source = self._resolve_shared_agent_class(request)
        requirements = self._analyze_request(request, resolved_class)
        analysis_ms = (time.perf_counter() - analysis_start) * 1000

        registry_start = time.perf_counter()
        candidate_models = self._resolve_candidate_models(request)
        registry_ms = (time.perf_counter() - registry_start) * 1000

        selection_start = time.perf_counter()
        default_model = self.store.get_default_model(request.application_name, resolved_class)
        candidates: List[Dict[str, Any]] = []
        score_breakdown: Dict[str, Dict[str, float]] = {}

        for model_id in candidate_models:
            capability_score = self._capability_score(model_id, requirements)
            shared_bonus = self.store.get_shared_bonus(resolved_class, model_id)
            app_bonus = self.store.get_app_bonus(request.application_name, resolved_class, model_id)
            global_penalty = self.store.get_global_penalty(model_id)
            app_penalty = self.store.get_app_penalty(request.application_name, resolved_class, model_id)
            default_bonus = 0.35 if default_model and model_id == default_model else 0.0
            requested_bonus = 0.2 if request.requested_model and model_id == request.requested_model else 0.0
            final_score = (
                capability_score
                + shared_bonus
                + app_bonus
                + default_bonus
                + requested_bonus
                - global_penalty
                - app_penalty
            )
            candidates.append(
                {
                    "model": model_id,
                    "provider": request.llm_provider or self._model_provider_map.get(model_id, ""),
                    "model_id": build_model_identifier(
                        request.llm_provider or self._model_provider_map.get(model_id, ""),
                        model_id,
                    ),
                    "score": round(final_score, 4),
                    "capability_score": round(capability_score, 4),
                    "shared_class_bonus": round(shared_bonus, 4),
                    "app_local_bonus": round(app_bonus + default_bonus + requested_bonus, 4),
                    "global_health_penalty": round(global_penalty, 4),
                    "app_local_penalty": round(app_penalty, 4),
                }
            )
            score_breakdown[model_id] = {
                "capability_score": round(capability_score, 4),
                "shared_class_bonus": round(shared_bonus, 4),
                "app_local_bonus": round(app_bonus, 4),
                "default_bonus": round(default_bonus, 4),
                "requested_bonus": round(requested_bonus, 4),
                "global_health_penalty": round(global_penalty, 4),
                "app_local_penalty": round(app_penalty, 4),
                "final_score": round(final_score, 4),
            }

        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected = candidates[0]["model"] if candidates else request.requested_model or "gpt-4o-mini"
        selection_ms = (time.perf_counter() - selection_start) * 1000
        route_ms = (time.perf_counter() - start) * 1000
        selected_provider = request.llm_provider or self._model_provider_map.get(selected, "")
        return RouteDecision(
            selected_model=selected,
            selected_provider=selected_provider,
            candidates=candidates,
            resolved_shared_agent_class=resolved_class,
            class_resolution_source=source,
            routing_reason="local_layered_learning",
            route_latency_ms=round(route_ms, 4),
            analysis_latency_ms=round(analysis_ms, 4),
            registry_latency_ms=round(registry_ms, 4),
            selection_latency_ms=round(selection_ms, 4),
            score_breakdown=score_breakdown,
            trace_context={
                "application_name": request.application_name,
                "agent_role": request.agent_role,
                "stage_name": request.stage_name,
                "workflow_id": request.execution_context.workflow_id,
                "run_id": request.execution_context.run_id,
                "step_id": request.execution_context.step_id,
            },
        )

    def _resolve_shared_agent_class(self, request: RouteRequest) -> Tuple[str, str]:
        explicit = str(request.shared_agent_class or "").strip().lower()
        if explicit:
            return self.CLASS_ALIASES.get(explicit, explicit), "explicit"

        candidates = [
            str(request.agent_role or "").strip().lower(),
            str(request.stage_name or "").strip().lower(),
        ]
        for candidate in candidates:
            if candidate in self.CLASS_ALIASES:
                return self.CLASS_ALIASES[candidate], "alias"

        text = f"{request.system_prompt} {request.task}".lower()
        if "review" in text or "critique" in text:
            return "review_agent", "inferred"
        if "outline" in text or "plan" in text:
            return "planner_agent", "inferred"
        if "scrap" in text or "scrape" in text or "evidence" in text:
            return "scrape_agent", "inferred"
        if "write" in text or "draft" in text:
            return "writer_agent", "inferred"
        return "general_agent", "fallback"

    def _analyze_request(self, request: RouteRequest, resolved_class: str) -> Dict[str, float]:
        text = " ".join(
            [
                str(request.agent_role or ""),
                str(request.stage_name or ""),
                str(request.system_prompt or ""),
                str(request.task or ""),
                str(request.execution_context.rollback_reason or ""),
            ]
        ).lower()
        requirements = {
            "general": 1.0,
            "reasoning": 0.7,
            "research": 0.6,
            "writing": 0.6,
            "review": 0.6,
            "cost_efficiency": 0.3,
        }
        if resolved_class in {"planner_agent", "claim_verifier_agent"}:
            requirements["reasoning"] += 0.7
        if resolved_class in {"scrape_agent", "research_agent"}:
            requirements["research"] += 0.8
        if resolved_class in {"writer_agent", "reviser_agent"}:
            requirements["writing"] += 0.9
        if resolved_class in {"review_agent", "claim_verifier_agent", "check_data_agent"}:
            requirements["review"] += 0.9
        if "rollback" in text or "challenge" in text or "conflict" in text:
            requirements["reasoning"] += 0.5
            requirements["review"] += 0.4
        if "fast" in text or "brief" in text:
            requirements["cost_efficiency"] += 0.4
        if "evidence" in text or "source" in text or "citation" in text:
            requirements["research"] += 0.4
            requirements["review"] += 0.3
        return requirements

    def _capability_score(self, model_id: str, requirements: Dict[str, float]) -> float:
        profile = self.MODEL_PROFILES.get(model_id, self.MODEL_PROFILES["gpt-4o"])
        total = 0.0
        for key, weight in requirements.items():
            total += profile.get(key, profile.get("general", 1.0)) * weight
        return round(total, 4)

    def _resolve_candidate_models(self, request: RouteRequest) -> List[str]:
        requested_model = request.requested_model
        _, _, normalized_requested_model = split_model_identifier(
            requested_model,
            fallback_provider=request.llm_provider,
        )
        return self.store.iter_unique_models(
            [
                [normalized_requested_model] if normalized_requested_model else [],
                self.model_pool,
            ]
        )

    def _default_model_pool(self) -> List[str]:
        configured_pool = str(os.getenv("ROUTE_AGENT_MODEL_POOL") or "").strip()
        if configured_pool:
            return [item.strip() for item in configured_pool.split(",") if item.strip()]
        default_model = str(os.getenv("ROUTE_AGENT_DEFAULT_MODEL") or "").strip()
        if default_model:
            return [default_model]
        return []

    def _normalize_model_pool(self, pool: List[str]) -> List[str]:
        normalized: List[str] = []
        for item in pool:
            model_id, provider, model = split_model_identifier(item)
            if not model:
                continue
            if provider and model not in self._model_provider_map:
                self._model_provider_map[model] = provider
            if model_id and provider and build_model_identifier(provider, model) != model_id:
                self._model_provider_map[model] = provider
            normalized.append(model)
        return self.store.iter_unique_models([normalized])
