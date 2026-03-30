import pytest

from multi_agents.agents.utils import llms as llm_module


@pytest.mark.asyncio
async def test_call_model_builds_route_request(monkeypatch):
    captured = {}

    async def fake_create_chat_completion(**kwargs):
        captured["route_context"] = kwargs.get("route_context")
        return '{"ok": true}'

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
