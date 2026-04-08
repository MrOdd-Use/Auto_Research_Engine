import json

import pytest

from multi_agents.route_agent import (
    LayeredRoutingStore,
    RouteAgentClient,
    RouteDecision,
    RouteExecutionContext,
    RouteRequest,
    RoutedLLMInvoker,
)


def _request(application_name: str, shared_agent_class: str, task: str = "collect evidence") -> RouteRequest:
    return RouteRequest(
        application_name=application_name,
        shared_agent_class=shared_agent_class,
        agent_role=shared_agent_class,
        stage_name="test_stage",
        system_prompt="Test system prompt",
        task=task,
        requested_model="gpt-4o-mini",
        llm_provider="openai",
        execution_context=RouteExecutionContext(
            workflow_id="wf",
            run_id="run",
            step_id="step",
        ),
    )


def test_shared_class_bonus_is_shared_but_app_local_penalty_is_isolated():
    store = LayeredRoutingStore()
    client = RouteAgentClient(store=store, model_pool=["gpt-4o-mini", "gpt-4o"])

    before = client.route(_request("app_b", "scrape_agent"))
    before_bonus = before.score_breakdown["gpt-4o"]["shared_class_bonus"]

    client.record_quality_success("app_a", "scrape_agent", "gpt-4o")
    after = client.route(_request("app_b", "scrape_agent"))

    assert before_bonus == 0.0
    assert after.score_breakdown["gpt-4o"]["shared_class_bonus"] > 0.0

    client.record_quality_failure("app_a", "scrape_agent", "gpt-4o")
    isolated = client.route(_request("app_b", "scrape_agent"))

    assert isolated.score_breakdown["gpt-4o"]["app_local_penalty"] == 0.0


def test_global_execution_failure_penalty_is_shared_across_apps():
    store = LayeredRoutingStore()
    client = RouteAgentClient(store=store, model_pool=["gpt-4o-mini", "gpt-4o"])
    before = client.route(_request("app_b", "writer_agent", task="write a polished report"))

    client.record_execution_failure("app_a", "writer_agent", "gpt-4o", provider_failure=True)
    after = client.route(_request("app_b", "writer_agent", task="write a polished report"))

    assert before.score_breakdown["gpt-4o"]["global_health_penalty"] == 0.0
    assert after.score_breakdown["gpt-4o"]["global_health_penalty"] > 0.0


def test_route_decision_exposes_latency_fields():
    client = RouteAgentClient(model_pool=["gpt-4o-mini", "gpt-4o"])
    decision = client.route(_request("app_a", "planner_agent", task="plan a report outline"))

    assert decision.route_latency_ms >= 0.0
    assert decision.analysis_latency_ms >= 0.0
    assert decision.registry_latency_ms >= 0.0
    assert decision.selection_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_invoker_records_route_then_execution_events():
    events = []
    client = RouteAgentClient(model_pool=["gpt-4o-mini", "gpt-4o"])
    invoker = RoutedLLMInvoker(client, event_logger=events.append)
    seen = {}

    async def provider_call(selected_model: str):
        seen["model"] = selected_model
        return {"ok": True}

    await invoker.invoke(
        provider_call=provider_call,
        requested_model="gpt-4o-mini",
        llm_provider="openai",
        route_request=_request("app_a", "review_agent", task="review evidence conflicts"),
    )

    assert seen["model"] in {"gpt-4o-mini", "gpt-4o"}
    assert [event["type"] for event in events] == [
        "route_decision",
        "execution_end",
    ]
    assert events[0]["route_latency_ms"] >= 0.0
    assert events[-1]["status"] == "completed"


class _FakeExternalBridge:
    """Minimal external bridge double for startup preflight tests."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def probe_global_pool(self, *, force=False, limit=None, mark_unavailable=True):
        self.calls.append(
            {
                "force": force,
                "limit": limit,
                "mark_unavailable": mark_unavailable,
            }
        )
        return list(self.results)

    async def probe_global_pool_async(self, *, force=False, limit=None, mark_unavailable=True):
        return self.probe_global_pool(force=force, limit=limit, mark_unavailable=mark_unavailable)


class _FakeRuntimeContext:
    """Minimal runtime context double for escalation-path tests."""

    def __init__(self, *, next_model_id: str | None):
        self.next_model_id = next_model_id
        self.failures = []
        self.successes = []

    async def handle_execution_failure(
        self,
        model_id: str,
        error_detail: str,
        *,
        hard_unavailable: bool = False,
        priority: str = "normal",
    ) -> str | None:
        self.failures.append(
            {
                "model_id": model_id,
                "error_detail": error_detail,
                "hard_unavailable": hard_unavailable,
                "priority": priority,
            }
        )
        return self.next_model_id

    async def handle_execution_success(self, model_id: str) -> None:
        self.successes.append(model_id)


class _FakeExternalClient:
    """Minimal external client double for invoker tests."""

    def __init__(self, *, decision: RouteDecision, bridge: _FakeExternalBridge):
        self.decision = decision
        self.external_bridge = bridge
        self.backend = "local_study"
        self.is_external_backend = True
        self.is_federation = False
        self.is_local_full = False
        self.is_local_study = False
        self.federation = None
        self.local_full = None
        self.local_study = None
        self.started = []
        self.ended = []

    def route(self, request: RouteRequest) -> RouteDecision:
        self.last_request = request
        return self.decision

    def start_execution_tracking(self, request: RouteRequest, decision: RouteDecision) -> str:
        self.started.append((request, decision))
        return "exec-1"

    def end_execution_tracking(
        self,
        *,
        execution_id: str,
        status: str,
        duration_ms: float,
        error_message: str | None = None,
    ) -> bool:
        self.ended.append(
            {
                "execution_id": execution_id,
                "status": status,
                "duration_ms": duration_ms,
                "error_message": error_message,
            }
        )
        return True


@pytest.mark.asyncio
async def test_external_invoker_escalates_after_runtime_failure():
    events = []
    seen_models = []
    runtime = _FakeRuntimeContext(next_model_id="openai:good-model")
    decision = RouteDecision(
        selected_model="bad-model",
        selected_provider="openai",
        candidates=[
            {"model": "bad-model", "provider": "openai", "model_id": "openai:bad-model"},
            {"model": "good-model", "provider": "openai", "model_id": "openai:good-model"},
        ],
        resolved_shared_agent_class="review_agent",
        routing_reason="prefer stronger review model",
        trace_context={},
        runtime_context=runtime,
    )
    client = _FakeExternalClient(
        decision=decision,
        bridge=_FakeExternalBridge(
            [
                {
                    "model_id": "openai:bad-model",
                    "ok": True,
                    "skipped": False,
                },
                {
                    "model_id": "openai:good-model",
                    "ok": True,
                    "skipped": False,
                },
            ]
        ),
    )
    invoker = RoutedLLMInvoker(client, event_logger=events.append)

    async def provider_call(selected_model: str, provider: str):
        seen_models.append((selected_model, provider))
        if selected_model == "bad-model":
            raise RuntimeError("invalid_request_error: no available channel")
        return {"ok": True, "model": selected_model}

    result = await invoker.invoke(
        provider_call=provider_call,
        requested_model="gpt-4o-mini",
        llm_provider="openai",
        route_request=_request("app_a", "review_agent", task="review evidence conflicts"),
    )

    assert result == {"ok": True, "model": "good-model"}
    assert seen_models == [("bad-model", "openai"), ("good-model", "openai")]
    assert runtime.failures[0]["model_id"] == "openai:bad-model"
    assert runtime.failures[0]["hard_unavailable"] is True
    assert runtime.successes == ["openai:good-model"]
    assert [event["type"] for event in events] == [
        "startup_preflight",
        "route_decision",
        "execution_escalation",
        "execution_end",
    ]
    assert events[0]["backend"] == "local_study"
    assert events[2]["kind"] == "escalate"
    assert events[-1]["status"] == "completed"
    assert events[-1]["selected_model"] == "good-model"
    assert client.ended[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_external_invoker_aborts_when_startup_preflight_finds_no_reachable_models():
    events = []
    runtime = _FakeRuntimeContext(next_model_id=None)
    decision = RouteDecision(
        selected_model="bad-model",
        selected_provider="openai",
        candidates=[{"model": "bad-model", "provider": "openai", "model_id": "openai:bad-model"}],
        resolved_shared_agent_class="review_agent",
        routing_reason="prefer stronger review model",
        trace_context={},
        runtime_context=runtime,
    )
    client = _FakeExternalClient(
        decision=decision,
        bridge=_FakeExternalBridge(
            [
                {
                    "model_id": "openai:bad-model",
                    "ok": False,
                    "skipped": False,
                }
            ]
        ),
    )
    invoker = RoutedLLMInvoker(client, event_logger=events.append)

    async def provider_call(selected_model: str, provider: str):
        raise AssertionError("provider should not be called when preflight blocks the run")

    with pytest.raises(RuntimeError, match="no reachable models"):
        await invoker.invoke(
            provider_call=provider_call,
            requested_model="gpt-4o-mini",
            llm_provider="openai",
            route_request=_request("app_a", "review_agent", task="review evidence conflicts"),
        )

    assert [event["type"] for event in events] == ["startup_preflight"]
    assert events[0]["backend"] == "local_study"
    assert events[0]["ok_count"] == 0
