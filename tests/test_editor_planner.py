from datetime import datetime

import pytest

from multi_agents.agents import editor as editor_module
from multi_agents.agents.editor import EditorAgent


@pytest.mark.asyncio
async def test_plan_research_normalizes_sections_and_limits(monkeypatch):
    async def fake_call_model(*args, **kwargs):
        return {
            "title": "  AI Infrastructure Trends  ",
            "date": "",
            "sections": [
                " Introduction ",
                "AI Chips",
                "ai chips",
                "Cloud Cost Optimization",
                "References",
                "Enterprise Adoption",
                "Robotics",
            ],
        }

    monkeypatch.setattr(editor_module, "call_model", fake_call_model)

    agent = EditorAgent()
    research_state = {
        "initial_research": "summary",
        "task": {"query": "AI infrastructure", "max_sections": 3, "model": "gpt-4o"},
    }

    result = await agent.plan_research(research_state)

    assert result["title"] == "AI Infrastructure Trends"
    assert result["sections"] == [
        "AI Chips",
        "Cloud Cost Optimization",
        "Enterprise Adoption",
    ]
    assert result["date"] == datetime.now().strftime("%d/%m/%Y")


@pytest.mark.asyncio
async def test_plan_research_falls_back_when_model_output_is_empty(monkeypatch):
    async def fake_call_model(*args, **kwargs):
        return None

    monkeypatch.setattr(editor_module, "call_model", fake_call_model)

    agent = EditorAgent()
    research_state = {
        "initial_research": "summary",
        "task": {"query": "Edge AI", "max_sections": None, "model": "gpt-4o"},
    }

    result = await agent.plan_research(research_state)

    assert result["title"] == "Edge AI"
    assert result["sections"] == ["Edge AI"]
    assert result["date"] == datetime.now().strftime("%d/%m/%Y")


@pytest.mark.asyncio
async def test_run_parallel_research_uses_fallback_section_when_empty(monkeypatch):
    class FakeChain:
        async def ainvoke(self, payload, config=None):
            return {"draft": {"topic": payload["topic"]}}

    class FakeWorkflow:
        def compile(self):
            return FakeChain()

    agent = EditorAgent()

    monkeypatch.setattr(agent, "_initialize_agents", lambda: {})
    monkeypatch.setattr(agent, "_create_workflow", lambda agents: FakeWorkflow())

    result = await agent.run_parallel_research(
        {
            "task": {"query": "Supply Chain AI", "max_sections": 3},
            "title": "",
            "sections": ["", "  ", "Introduction"],
        }
    )

    assert result["research_data"] == [{"topic": "Supply Chain AI"}]


@pytest.mark.asyncio
async def test_run_parallel_research_collects_scrap_packets(monkeypatch):
    class FakeChain:
        async def ainvoke(self, payload, config=None):
            return {
                "draft": {"topic": payload["topic"]},
                "scrap_packet": {
                    "iteration_index": 1,
                    "model_level": "Level_1_Base",
                    "active_engines": ["Tavily"],
                    "search_log": [],
                },
                "check_data_verdict": {
                    "status": "ACCEPT",
                    "deep_eval_report": {"final_score": 0.9},
                    "feedback_packet": {"instruction": "", "new_query_suggestion": ""},
                },
            }

    class FakeWorkflow:
        def compile(self):
            return FakeChain()

    agent = EditorAgent()

    monkeypatch.setattr(agent, "_initialize_agents", lambda: {})
    monkeypatch.setattr(agent, "_create_workflow", lambda agents: FakeWorkflow())

    result = await agent.run_parallel_research(
        {
            "task": {"query": "AI Safety", "max_sections": 1},
            "sections": ["AI Safety"],
        }
    )

    assert len(result["scrap_packets"]) == 1
    assert result["scrap_packets"][0]["model_level"] == "Level_1_Base"
    assert len(result["check_data_reports"]) == 1
    assert result["check_data_reports"][0]["status"] == "ACCEPT"
