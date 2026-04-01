from __future__ import annotations

import inspect
import time
from typing import Any, Awaitable, Callable, Dict, Optional


def _is_quota_error(exc: BaseException) -> bool:
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg or "RATE_LIMIT" in msg

from .client import RouteAgentClient
from .utils.context import current_route_scope
from .utils.model_utils import build_model_identifier
from .models import RouteDecision, RouteRequest


ProviderCall = Callable[..., Awaitable[Any]]
EventLogger = Callable[[Dict[str, Any]], None]


class RoutedLLMInvoker:
    """Executes Route_Agent control-plane routing before the real provider call."""

    def __init__(self, client: RouteAgentClient | None = None, *, event_logger: EventLogger | None = None) -> None:
        self.client = client or RouteAgentClient()
        self.event_logger = event_logger

    async def invoke(
        self,
        *,
        provider_call: ProviderCall,
        requested_model: str,
        llm_provider: str = "",
        route_request: RouteRequest | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        request = route_request or self._build_request_from_scope(
            requested_model=requested_model,
            llm_provider=llm_provider,
            metadata=metadata,
        )
        if request is None:
            return await provider_call(requested_model)

        if self.client.is_federation and self.client.federation is not None:
            return await self._invoke_federation(
                request=request,
                provider_call=provider_call,
            )

        decision = self.client.route(request)
        selected_model_id = build_model_identifier(decision.selected_provider, decision.selected_model)
        self._emit(
            {
                "type": "route_decision",
                "application_name": request.application_name,
                "shared_agent_class": decision.resolved_shared_agent_class,
                "agent_role": request.agent_role,
                "stage_name": request.stage_name,
                "requested_model": requested_model,
                "selected_model": decision.selected_model,
                "selected_provider": decision.selected_provider,
                "selected_model_id": selected_model_id,
                "routing_reason": decision.routing_reason,
                "route_latency_ms": decision.route_latency_ms,
                "analysis_latency_ms": decision.analysis_latency_ms,
                "registry_latency_ms": decision.registry_latency_ms,
                "selection_latency_ms": decision.selection_latency_ms,
                "trace_context": decision.trace_context,
                "candidates": decision.candidates,
            }
        )
        started_at = time.perf_counter()
        self._emit(
            {
                "type": "execution_start",
                "application_name": request.application_name,
                "shared_agent_class": decision.resolved_shared_agent_class,
                "agent_role": request.agent_role,
                "stage_name": request.stage_name,
                "selected_model": decision.selected_model,
                "selected_provider": decision.selected_provider,
                "selected_model_id": selected_model_id,
                "route_latency_ms": decision.route_latency_ms,
                "trace_context": decision.trace_context,
            }
        )
        execution_id = self.client.start_execution_tracking(request, decision)
        try:
            result = await self._invoke_provider(
                provider_call=provider_call,
                decision=decision,
            )
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 4)
            self.client.record_execution_failure(
                request.application_name,
                decision.resolved_shared_agent_class,
                decision.selected_model,
                provider_failure=True,
            )
            self.client.end_execution_tracking(
                execution_id=execution_id,
                status="failed",
                duration_ms=duration_ms,
                error_message=str(exc),
            )
            self._emit(
                {
                    "type": "execution_end",
                    "status": "failed",
                    "application_name": request.application_name,
                    "shared_agent_class": decision.resolved_shared_agent_class,
                    "agent_role": request.agent_role,
                    "stage_name": request.stage_name,
                    "selected_model": decision.selected_model,
                    "selected_provider": decision.selected_provider,
                    "selected_model_id": selected_model_id,
                    "route_latency_ms": decision.route_latency_ms,
                    "execution_latency_ms": duration_ms,
                    "error": str(exc),
                    "trace_context": decision.trace_context,
                }
            )
            raise

        duration_ms = round((time.perf_counter() - started_at) * 1000, 4)
        self.client.record_quality_success(
            request.application_name,
            decision.resolved_shared_agent_class,
            decision.selected_model,
        )
        self.client.end_execution_tracking(
            execution_id=execution_id,
            status="completed",
            duration_ms=duration_ms,
        )
        self._emit(
            {
                "type": "execution_end",
                "status": "completed",
                "application_name": request.application_name,
                "shared_agent_class": decision.resolved_shared_agent_class,
                "agent_role": request.agent_role,
                "stage_name": request.stage_name,
                "selected_model": decision.selected_model,
                "selected_provider": decision.selected_provider,
                "selected_model_id": selected_model_id,
                "route_latency_ms": decision.route_latency_ms,
                "execution_latency_ms": duration_ms,
                "trace_context": decision.trace_context,
            }
        )
        return result

    async def _invoke_federation(
        self,
        *,
        request: RouteRequest,
        provider_call: ProviderCall,
    ) -> Any:
        """Federation path: route → invoke → release + report_outcome."""
        federation = self.client.federation
        assert federation is not None

        decision, lease_id = await federation.route(request)
        selected_model_id = build_model_identifier(decision.selected_provider, decision.selected_model)
        self._emit({
            "type": "route_decision",
            "backend": "federation",
            "application_name": request.application_name,
            "shared_agent_class": decision.resolved_shared_agent_class,
            "selected_model": decision.selected_model,
            "selected_provider": decision.selected_provider,
            "selected_model_id": selected_model_id,
            "routing_reason": decision.routing_reason,
            "lease_id": lease_id,
            "trace_context": decision.trace_context,
        })

        started_at = time.perf_counter()
        try:
            result = await self._invoke_provider(
                provider_call=provider_call,
                decision=decision,
            )
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 4)
            if lease_id:
                await federation.release(lease_id)
                await federation.report_outcome(
                    lease_id=lease_id,
                    model_id=decision.selected_model,
                    agent_class=decision.resolved_shared_agent_class,
                    outcome_type="failure",
                    duration_ms=int(duration_ms),
                )
            self._emit({
                "type": "execution_end",
                "backend": "federation",
                "status": "failed",
                "selected_model": decision.selected_model,
                "lease_id": lease_id,
                "execution_latency_ms": duration_ms,
                "error": str(exc),
            })
            raise

        duration_ms = round((time.perf_counter() - started_at) * 1000, 4)
        if lease_id:
            await federation.release(lease_id)
            await federation.report_outcome(
                lease_id=lease_id,
                model_id=decision.selected_model,
                agent_class=decision.resolved_shared_agent_class,
                outcome_type="success",
                duration_ms=int(duration_ms),
            )
        self._emit({
            "type": "execution_end",
            "backend": "federation",
            "status": "completed",
            "selected_model": decision.selected_model,
            "lease_id": lease_id,
            "execution_latency_ms": duration_ms,
        })
        return result

    async def _invoke_provider(self, *, provider_call: ProviderCall, decision: RouteDecision) -> Any:
        candidates = list(decision.candidates or [])
        # 把当前选中的模型放在首位，其余候选跟在后面
        current = {"model": decision.selected_model, "provider": decision.selected_provider}
        ordered = [current] + [c for c in candidates if c.get("model") != decision.selected_model]

        last_exc: Exception | None = None
        for candidate in ordered:
            model = candidate.get("model") or decision.selected_model
            provider = candidate.get("provider") or decision.selected_provider
            try:
                if self._provider_call_accepts_provider(provider_call):
                    return await provider_call(model, provider)
                return await provider_call(model)
            except Exception as exc:
                if not _is_quota_error(exc) or candidate is ordered[-1]:
                    raise
                # quota 耗尽：标记失败，尝试下一个候选
                self.client.record_execution_failure(
                    "",
                    decision.resolved_shared_agent_class,
                    model,
                    provider_failure=True,
                )
                self._emit({
                    "type": "quota_fallback",
                    "failed_model": model,
                    "failed_provider": provider,
                    "next_model": ordered[ordered.index(candidate) + 1].get("model"),
                    "error": str(exc)[:200],
                })
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    def _provider_call_accepts_provider(self, provider_call: ProviderCall) -> bool:
        try:
            signature = inspect.signature(provider_call)
        except (TypeError, ValueError):
            return False

        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
            return True
        return len(positional) >= 2

    def _build_request_from_scope(
        self,
        *,
        requested_model: str,
        llm_provider: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RouteRequest | None:
        scope = current_route_scope()
        if scope is None:
            return None
        metadata = metadata or {}
        return scope.build_request(
            task=str(metadata.get("task") or ""),
            system_prompt=str(metadata.get("system_prompt") or ""),
            requested_model=requested_model,
            llm_provider=llm_provider,
            metadata=metadata,
        )

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self.event_logger is not None:
            self.event_logger(payload)


_GLOBAL_INVOKER: RoutedLLMInvoker | None = None


def get_global_invoker() -> RoutedLLMInvoker:
    global _GLOBAL_INVOKER
    if _GLOBAL_INVOKER is None:
        _GLOBAL_INVOKER = RoutedLLMInvoker()
    return _GLOBAL_INVOKER


def set_global_invoker(invoker: RoutedLLMInvoker) -> None:
    global _GLOBAL_INVOKER
    _GLOBAL_INVOKER = invoker


def reset_global_invoker() -> None:
    global _GLOBAL_INVOKER
    _GLOBAL_INVOKER = None
