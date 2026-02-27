import pytest

from multi_agents.agents.check_data import CheckDataAgent
from multi_agents.agents.state_controller import StateController


def _make_agent(tmp_path):
    agent = CheckDataAgent()
    agent.state_controller = StateController(str(tmp_path / "agent_state.json"))
    return agent


def _packet_with_text(text: str) -> dict:
    return {
        "iteration_index": 1,
        "model_level": "Level_1_Base",
        "active_engines": ["tavily"],
        "search_log": [
            {
                "target": "nvidia 2026 ai gpu revenue",
                "extra_hints_applied": "",
                "top_10_passages": [
                    {
                        "content": text,
                        "source_url": "https://example.com/source",
                        "metadata": {},
                    }
                ],
            }
        ],
    }


def _packet_with_coverage(text: str, section_coverage: float, uncovered_queries=None) -> dict:
    return {
        "iteration_index": 1,
        "model_level": "Level_1_Base",
        "active_engines": ["tavily"],
        "search_log": [
            {
                "source_query": "nvidia 2026 ai gpu revenue audited",
                "target": "nvidia 2026 audited ai gpu revenue",
                "extra_hints_applied": "",
                "top_10_passages": [
                    {
                        "content": text,
                        "source_url": "https://example.com/source",
                        "metadata": {},
                    }
                ],
            }
        ],
        "coverage_snapshot": {
            "query_coverage": section_coverage,
            "keypoint_coverage": section_coverage,
            "section_coverage": section_coverage,
            "query_total": 1,
            "query_covered": 1 if section_coverage >= 0.7 else 0,
            "uncovered_queries": uncovered_queries or ([] if section_coverage >= 0.7 else ["nvidia 2026 ai gpu revenue audited"]),
            "keypoint_total": 1,
            "keypoint_covered": 1 if section_coverage >= 0.7 else 0,
            "uncovered_key_points": [] if section_coverage >= 0.7 else ["audited revenue"],
            "coverage_threshold": 0.7,
        },
    }


@pytest.mark.asyncio
async def test_check_data_retry_when_constraints_missing(tmp_path):
    agent = _make_agent(tmp_path)
    result = await agent.run(
        {
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 1,
            "scrap_packet": _packet_with_text("Forecast for 2025 suggests rapid growth projection."),
        }
    )

    assert result["check_data_action"] == "retry"
    assert result["check_data_verdict"]["status"] == "RETRY"
    assert result["audit_feedback"]["is_satisfied"] is False
    assert result["audit_feedback"]["confidence_score"] <= 0.69
    assert result["iteration_index"] == 2
    assert result["audit_feedback"]["new_query_suggestion"]
    assert result["extra_hints"]


@pytest.mark.asyncio
async def test_check_data_accept_when_claims_are_verified(tmp_path):
    agent = _make_agent(tmp_path)
    result = await agent.run(
        {
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 1,
            "scrap_packet": _packet_with_text(
                "Nvidia reported actual audited AI GPU revenue in 2026 in its official filing."
            ),
        }
    )

    assert result["check_data_action"] == "accept"
    assert result["check_data_verdict"]["status"] == "ACCEPT"
    assert result["check_data_verdict"]["deep_eval_report"]["final_score"] >= 0.7
    assert result["audit_feedback"]["is_satisfied"] is True


@pytest.mark.asyncio
async def test_check_data_blocks_after_third_failed_attempt(tmp_path):
    agent = _make_agent(tmp_path)
    result = await agent.run(
        {
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 3,
            "scrap_packet": _packet_with_text("Projection only. No audited actual value was found."),
        }
    )

    assert result["check_data_action"] == "blocked"
    assert result["check_data_verdict"]["status"] == "BLOCKED"
    assert "2026 Nvidia AI GPU revenue" in result["draft"]
    assert "需人工复核" in result["draft"]["2026 Nvidia AI GPU revenue"]


@pytest.mark.asyncio
async def test_retry_feedback_is_propagated_as_extra_hints(tmp_path):
    agent = _make_agent(tmp_path)
    result = await agent.run(
        {
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 1,
            "scrap_packet": _packet_with_text("Prediction data from 2025 only."),
        }
    )

    packet = result["check_data_verdict"]["feedback_packet"]
    assert result["check_data_action"] == "retry"
    assert packet["instruction"] in result["extra_hints"]
    assert packet["new_query_suggestion"] in result["extra_hints"]
    assert result["audit_feedback"]["instruction"] == packet["instruction"]
    assert result["audit_feedback"]["new_query_suggestion"] == packet["new_query_suggestion"]


@pytest.mark.asyncio
async def test_check_data_retries_when_coverage_below_threshold(tmp_path):
    agent = _make_agent(tmp_path)
    result = await agent.run(
        {
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {"research_queries": ["nvidia 2026 ai gpu revenue audited"]},
            "iteration_index": 1,
            "scrap_packet": _packet_with_coverage(
                text="Nvidia reported actual audited AI GPU revenue in 2026.",
                section_coverage=0.4,
            ),
        }
    )

    assert result["check_data_action"] == "retry"
    assert result["check_data_verdict"]["coverage_report"]["section_coverage"] == 0.4
    assert result["audit_feedback"]["uncovered_queries"]


@pytest.mark.asyncio
async def test_check_data_blocks_when_coverage_still_low_after_retry(tmp_path):
    agent = _make_agent(tmp_path)
    result = await agent.run(
        {
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {"research_queries": ["nvidia 2026 ai gpu revenue audited"]},
            "iteration_index": 2,
            "scrap_packet": _packet_with_coverage(
                text="Nvidia reported actual audited AI GPU revenue in 2026.",
                section_coverage=0.5,
            ),
        }
    )

    assert result["check_data_action"] == "blocked"
    assert result["check_data_verdict"]["coverage_report"]["section_coverage"] == 0.5
