import json

import pytest

from multi_agents.route_agent import (
    LayeredRoutingStore,
    RouteAgentClient,
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
        "execution_start",
        "execution_end",
    ]
    assert events[0]["route_latency_ms"] >= 0.0
    assert events[-1]["status"] == "completed"
