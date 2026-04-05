from __future__ import annotations

import asyncio
import inspect
import os
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
        self._startup_preflight_done = False

    async def invoke(
        self,
        *,
        provider_call: ProviderCall,
        requested_model: str | None,
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
            return await provider_call(str(requested_model or ""))

        if self.client.is_federation and self.client.federation is not None:
            return await self._invoke_federation(
                request=request,
                provider_call=provider_call,
            )

        if self.client.is_local_full and self.client.local_full is not None:
            return await self._invoke_local_full(
                request=request,
                provider_call=provider_call,
            )

        if getattr(self.client, "is_external_backend", False) is True:
            await self._ensure_external_startup_preflight()

        decision = self.client.route(request)
        runtime_context = getattr(decision, "runtime_context", None)
        selected_model_id = build_model_identifier(decision.selected_provider, decision.selected_model)
        self._emit(
            {
                "type": "route_decision",
                "application_name": request.application_name,
                "shared_agent_class": decision.resolved_shared_agent_class,
                "agent_role": request.agent_role,
                "stage_name": request.stage_name,
                "requested_model": request.requested_model,
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
        executed_model = decision.selected_model
        executed_provider = decision.selected_provider
        executed_model_id = selected_model_id
        try:
            if runtime_context is not None:
                result, executed_model, executed_provider = await self._invoke_external_runtime(
                    provider_call=provider_call,
                    decision=decision,
                )
                executed_model_id = build_model_identifier(executed_provider, executed_model)
            else:
                result = await self._invoke_provider(
                    provider_call=provider_call,
                    decision=decision,
                )
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 4)
            if runtime_context is None:
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
                    "selected_model": executed_model,
                    "selected_provider": executed_provider,
                    "selected_model_id": executed_model_id,
                    "route_latency_ms": decision.route_latency_ms,
                    "execution_latency_ms": duration_ms,
                    "error": str(exc),
                    "trace_context": decision.trace_context,
                }
            )
            raise

        duration_ms = round((time.perf_counter() - started_at) * 1000, 4)
        if runtime_context is None:
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
                "selected_model": executed_model,
                "selected_provider": executed_provider,
                "selected_model_id": executed_model_id,
                "route_latency_ms": decision.route_latency_ms,
                "execution_latency_ms": duration_ms,
                "trace_context": decision.trace_context,
            }
        )
        return result

    async def _ensure_external_startup_preflight(self) -> None:
        """Probe the external Route_Agent pool once before the first routed call."""
        if self._startup_preflight_done:
            return

        bridge = getattr(self.client, "external_bridge", None)
        if bridge is None:
            self._startup_preflight_done = True
            return

        max_models_raw = str(os.getenv("ROUTE_AGENT_PREFLIGHT_MAX_MODELS") or "").strip()
        max_models = int(max_models_raw) if max_models_raw.isdigit() else None
        results = await asyncio.to_thread(
            lambda: bridge.probe_global_pool(
                force=False,
                limit=max_models,
                mark_unavailable=True,
            )
        )
        ok_count = sum(1 for item in results if item.get("ok"))
        skipped_count = sum(1 for item in results if item.get("skipped"))
        filtered = [
            str(item.get("model_id") or "")
            for item in results
            if not item.get("ok") and not item.get("skipped")
        ]
        self._emit(
            {
                "type": "startup_preflight",
                "backend": "external",
                "checked_models": len(results),
                "ok_count": ok_count,
                "skipped_count": skipped_count,
                "filtered_models": filtered,
            }
        )
        self._startup_preflight_done = True
        if results and ok_count == 0 and skipped_count == 0:
            raise RuntimeError("external Route_Agent preflight found no reachable models")

    async def _invoke_external_runtime(
        self,
        *,
        provider_call: ProviderCall,
        decision: RouteDecision,
    ) -> tuple[Any, str, str]:
        """Execute one external-routed request with failure writeback and failover."""
        runtime = getattr(decision, "runtime_context", None)
        if runtime is None:
            result = await self._invoke_provider(provider_call=provider_call, decision=decision)
            return result, decision.selected_model, decision.selected_provider

        queue = self._build_candidate_queue(decision)
        attempted_model_ids: set[str] = set()
        last_exc: Exception | None = None

        while queue:
            candidate = queue.pop(0)
            model = str(candidate.get("model") or decision.selected_model)
            provider = str(candidate.get("provider") or decision.selected_provider)
            model_id = str(candidate.get("model_id") or build_model_identifier(provider, model))
            if model_id in attempted_model_ids:
                continue
            attempted_model_ids.add(model_id)

            try:
                if self._provider_call_accepts_provider(provider_call):
                    result = await provider_call(model, provider)
                else:
                    result = await provider_call(model)
            except Exception as exc:
                last_exc = exc
                next_model_id = await runtime.handle_execution_failure(
                    model_id,
                    str(exc),
                    hard_unavailable=self._is_hard_unavailable_error(exc),
                )
                if _is_quota_error(exc) and queue:
                    self._emit(
                        {
                            "type": "quota_fallback",
                            "failed_model": model,
                            "failed_provider": provider,
                            "next_model": queue[0].get("model"),
                            "error": str(exc)[:200],
                        }
                    )
                    continue

                if next_model_id:
                    promoted = self._promote_candidate(queue, next_model_id, attempted_model_ids)
                    if promoted is not None:
                        self._emit(
                            {
                                "type": "execution_escalation",
                                "kind": "escalate",
                                "failed_model": model,
                                "failed_provider": provider,
                                "next_model": promoted.get("model"),
                                "next_provider": promoted.get("provider"),
                                "error": str(exc)[:200],
                            }
                        )
                        continue

                if queue:
                    self._emit(
                        {
                            "type": "execution_escalation",
                            "kind": "failover",
                            "failed_model": model,
                            "failed_provider": provider,
                            "next_model": queue[0].get("model"),
                            "next_provider": queue[0].get("provider"),
                            "error": str(exc)[:200],
                        }
                    )
                    continue
                raise

            await runtime.handle_execution_success(model_id)
            return result, model, provider

        assert last_exc is not None
        raise last_exc

    def _build_candidate_queue(self, decision: RouteDecision) -> list[dict[str, Any]]:
        """Build the execution queue with the selected model first."""
        current_model_id = build_model_identifier(
            decision.selected_provider,
            decision.selected_model,
            target="route_agent",
        )
        current = {
            "model": decision.selected_model,
            "provider": decision.selected_provider,
            "model_id": current_model_id,
        }
        queue = [current]
        for candidate in list(decision.candidates or []):
            candidate_model_id = str(
                candidate.get("model_id")
                or build_model_identifier(
                    str(candidate.get("provider") or ""),
                    str(candidate.get("model") or ""),
                    target="route_agent",
                )
            )
            if candidate_model_id == current_model_id:
                continue
            queue.append(
                {
                    **dict(candidate),
                    "model_id": candidate_model_id,
                }
            )
        return queue

    def _promote_candidate(
        self,
        queue: list[dict[str, Any]],
        model_id: str,
        attempted_model_ids: set[str],
    ) -> dict[str, Any] | None:
        """Move the requested model to the front of the remaining queue."""
        for index, candidate in enumerate(queue):
            candidate_model_id = str(
                candidate.get("model_id")
                or build_model_identifier(
                    str(candidate.get("provider") or ""),
                    str(candidate.get("model") or ""),
                    target="route_agent",
                )
            )
            if candidate_model_id != model_id:
                continue
            if candidate_model_id in attempted_model_ids:
                return None
            promoted = queue.pop(index)
            queue.insert(0, promoted)
            return promoted
        return None

    def _is_hard_unavailable_error(self, exc: BaseException) -> bool:
        """Return whether the error should immediately mark a model unavailable."""
        message = str(exc).lower()
        patterns = (
            "no available channel",
            "unsupported model",
            "model not found",
            "does not exist",
            "invalid_request_error",
            "authentication",
            "unauthorized",
            "permission denied",
        )
        return any(pattern in message for pattern in patterns)

    async def _invoke_local_full(
        self,
        *,
        request: RouteRequest,
        provider_call: ProviderCall,
    ) -> Any:
        """local_full path: route via LocalOnlyAdapter → invoke → report_outcome (local DB only)."""
        local_full = self.client.local_full
        assert local_full is not None

        decision, _lease_id = await local_full.route(request)
        selected_model_id = build_model_identifier(decision.selected_provider, decision.selected_model)
        self._emit({
            "type": "route_decision",
            "backend": "local_full",
            "application_name": request.application_name,
            "shared_agent_class": decision.resolved_shared_agent_class,
            "selected_model": decision.selected_model,
            "selected_provider": decision.selected_provider,
            "selected_model_id": selected_model_id,
            "routing_reason": decision.routing_reason,
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
            await local_full.report_outcome(
                lease_id="",
                model_id=decision.selected_model,
                agent_class=decision.resolved_shared_agent_class,
                outcome_type="exec_fail",
                duration_ms=int(duration_ms),
            )
            self._emit({
                "type": "execution_end",
                "backend": "local_full",
                "status": "failed",
                "selected_model": decision.selected_model,
                "execution_latency_ms": duration_ms,
                "error": str(exc),
            })
            raise

        duration_ms = round((time.perf_counter() - started_at) * 1000, 4)
        await local_full.report_outcome(
            lease_id="",
            model_id=decision.selected_model,
            agent_class=decision.resolved_shared_agent_class,
            outcome_type="exec_success",
            duration_ms=int(duration_ms),
        )
        self._emit({
            "type": "execution_end",
            "backend": "local_full",
            "status": "completed",
            "selected_model": decision.selected_model,
            "execution_latency_ms": duration_ms,
        })
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
                    outcome_type="exec_fail",
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
                outcome_type="exec_success",
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
        requested_model: str | None,
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
            requested_model=None,
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
