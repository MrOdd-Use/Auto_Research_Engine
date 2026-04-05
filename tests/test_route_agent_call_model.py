import pytest

from multi_agents.agents.utils import llms as llm_module
from multi_agents.route_agent import RouteExecutionContext, RouteRequest
from multi_agents.route_agent.tools.external_bridge import ExternalRouteAgentBridge


@pytest.mark.asyncio
async def test_call_model_builds_route_request(monkeypatch):
    captured = {}

    class _FakeConfig:
        smart_llm_provider = "openai"
        llm_kwargs = {}

    async def fake_create_chat_completion(**kwargs):
        captured["route_context"] = kwargs.get("route_context")
        return '{"ok": true}'

    monkeypatch.setattr(llm_module, "Config", _FakeConfig)
    monkeypatch.setattr(llm_module, "create_chat_completion", fake_create_chat_completion)

    result = await llm_module.call_model(
        prompt=[
            {"role": "system", "content": "You are a planner."},
            {"role": "user", "content": "Plan a report."},
        ],
        model="gpt-4o-mini",
        response_format="json",
        route_context={
            "application_name": "auto_research_engine",
            "shared_agent_class": "planner_agent",
            "agent_role": "planner",
            "stage_name": "outline_planning",
            "workflow_id": "wf",
            "run_id": "run",
            "step_id": "step",
        },
    )

    assert result == {"ok": True}
    route_request = captured["route_context"]["route_request"]
    assert route_request.application_name == "auto_research_engine"
    assert route_request.shared_agent_class == "planner_agent"
    assert route_request.execution_context.workflow_id == "wf"
    assert route_request.requested_model is None


@pytest.mark.asyncio
async def test_call_model_normalizes_structured_json_response(monkeypatch):
    class _FakeConfig:
        smart_llm_provider = "openai"
        llm_kwargs = {}

    async def fake_create_chat_completion(**kwargs):
        return [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "internal"}],
            },
            {"type": "output_text", "text": '{"ok": true, "source": "structured"}'},
        ]

    monkeypatch.setattr(llm_module, "Config", _FakeConfig)
    monkeypatch.setattr(llm_module, "create_chat_completion", fake_create_chat_completion)

    result = await llm_module.call_model(
        prompt=[{"role": "user", "content": "Return JSON."}],
        model="gpt-4.1",
        response_format="json",
    )

    assert result == {"ok": True, "source": "structured"}


def test_external_bridge_does_not_hard_filter_by_provider_without_requested_model():
    bridge = ExternalRouteAgentBridge(project_path="D:/agent/Route_Agent")

    payload = bridge._build_route_payload(
        RouteRequest(
            application_name="auto_research_engine",
            shared_agent_class="planner_agent",
            agent_role="planner",
            stage_name="outline_planning",
            system_prompt="You are a planner.",
            task="Plan a report.",
            requested_model="",
            llm_provider="openai",
            execution_context=RouteExecutionContext(),
        )
    )

    assert payload["constraints"] is None


def test_external_bridge_uses_requested_model_as_preferred_model():
    bridge = ExternalRouteAgentBridge(project_path="D:/agent/Route_Agent")

    payload = bridge._build_route_payload(
        RouteRequest(
            application_name="auto_research_engine",
            shared_agent_class="planner_agent",
            agent_role="planner",
            stage_name="outline_planning",
            system_prompt="You are a planner.",
            task="Plan a report.",
            requested_model="gpt-4.1",
            llm_provider="openai",
            execution_context=RouteExecutionContext(),
        )
    )

    assert payload["constraints"] == {"preferred_model": "openai:gpt-4.1"}
