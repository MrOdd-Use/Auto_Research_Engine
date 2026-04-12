import pytest
from unittest.mock import AsyncMock, patch

from multi_agents.agents.check_data import CheckDataAgent
from multi_agents.agents.state_controller import StateController


def _make_agent(tmp_path):
    agent = CheckDataAgent()
    agent.state_controller = StateController(str(tmp_path / "agent_state.json"))
    return agent


def _packet(targets: list) -> dict:
    """targets: list of {target, passages}"""
    return {
        "search_log": [
            {
                "target": t["target"],
                "top_10_passages": [{"content": t["passages"], "source_url": "https://example.com"}],
            }
            for t in targets
        ]
    }


def _packet_with_coverage(targets: list, section_coverage: float, uncovered_queries=None) -> dict:
    base = _packet(targets)
    base["coverage_snapshot"] = {
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
    }
    return base


LLM_ALL_CHECKED = [{"target": "nvidia 2026 ai gpu revenue", "checked": True, "reason": "Evidence directly states audited revenue figures for 2026."}]
LLM_ALL_FAILED = [{"target": "nvidia 2026 ai gpu revenue", "checked": False, "reason": "Evidence only contains forecast data, no actual figures."}]


@pytest.mark.asyncio
async def test_check_data_accept_when_llm_checks_all(tmp_path):
    agent = _make_agent(tmp_path)
    with patch("multi_agents.agents.check_data.call_model", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLM_ALL_CHECKED
        result = await agent.run({
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 1,
            "scraping_packet": _packet([{
                "target": "nvidia 2026 ai gpu revenue",
                "passages": "Nvidia reported actual audited AI GPU revenue of $40B in fiscal 2026.",
            }]),
        })

    assert result["check_data_action"] == "accept"
    assert result["check_data_verdict"]["status"] == "ACCEPT"
    assert result["audit_feedback"]["is_satisfied"] is True
    assert result["audit_feedback"]["confidence_score"] == 1.0


@pytest.mark.asyncio
async def test_check_data_retry_when_llm_fails_all(tmp_path):
    agent = _make_agent(tmp_path)
    with patch("multi_agents.agents.check_data.call_model", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLM_ALL_FAILED
        result = await agent.run({
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 1,
            "scraping_packet": _packet([{
                "target": "nvidia 2026 ai gpu revenue",
                "passages": "Forecast for 2025 suggests rapid growth projection.",
            }]),
        })

    assert result["check_data_action"] == "retry"
    assert result["check_data_verdict"]["status"] == "RETRY"
    assert result["audit_feedback"]["is_satisfied"] is False
    assert result["audit_feedback"]["confidence_score"] == 0.0
    assert result["iteration_index"] == 2
    assert result["extra_hints"]


@pytest.mark.asyncio
async def test_check_data_blocks_after_max_retries(tmp_path):
    agent = _make_agent(tmp_path)
    with patch("multi_agents.agents.check_data.call_model", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLM_ALL_FAILED
        result = await agent.run({
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 3,
            "scraping_packet": _packet([{
                "target": "nvidia 2026 ai gpu revenue",
                "passages": "Projection only. No audited actual value was found.",
            }]),
        })

    assert result["check_data_action"] == "blocked"
    assert result["check_data_verdict"]["status"] == "BLOCKED"
    assert "2026 Nvidia AI GPU revenue" in result["draft"]
    assert "需人工复核" in result["draft"]["2026 Nvidia AI GPU revenue"]
    assert "2 轮验证" in result["draft"]["2026 Nvidia AI GPU revenue"]


@pytest.mark.asyncio
async def test_blocked_placeholder_uses_configured_retry_count(tmp_path):
    agent = _make_agent(tmp_path)
    with patch("multi_agents.agents.check_data.call_model", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLM_ALL_FAILED
        result = await agent.run({
            "task": {"check_data_max_retries": 3},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 3,
            "scraping_packet": _packet([{
                "target": "nvidia 2026 ai gpu revenue",
                "passages": "Projection only. No audited actual value was found.",
            }]),
        })

    assert result["check_data_action"] == "blocked"
    assert "3 轮验证" in result["draft"]["2026 Nvidia AI GPU revenue"]


@pytest.mark.asyncio
async def test_retry_feedback_propagated_as_extra_hints(tmp_path):
    agent = _make_agent(tmp_path)
    with patch("multi_agents.agents.check_data.call_model", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLM_ALL_FAILED
        result = await agent.run({
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 1,
            "scraping_packet": _packet([{
                "target": "nvidia 2026 ai gpu revenue",
                "passages": "Prediction data from 2025 only.",
            }]),
        })

    packet = result["check_data_verdict"]["feedback_packet"]
    assert result["check_data_action"] == "retry"
    assert packet["instruction"] in result["extra_hints"]
    assert packet["new_query_suggestion"] in result["extra_hints"]
    assert result["audit_feedback"]["instruction"] == packet["instruction"]
    assert result["audit_feedback"]["new_query_suggestion"] == packet["new_query_suggestion"]


@pytest.mark.asyncio
async def test_check_data_retries_when_coverage_below_threshold(tmp_path):
    agent = _make_agent(tmp_path)
    with patch("multi_agents.agents.check_data.call_model", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLM_ALL_CHECKED
        result = await agent.run({
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {"research_queries": ["nvidia 2026 ai gpu revenue audited"]},
            "iteration_index": 1,
            "scraping_packet": _packet_with_coverage(
                targets=[{"target": "nvidia 2026 ai gpu revenue", "passages": "Nvidia reported actual audited AI GPU revenue in 2026."}],
                section_coverage=0.4,
            ),
        })

    assert result["check_data_action"] == "retry"
    assert result["check_data_verdict"]["coverage_report"]["section_coverage"] == 0.4
    assert result["audit_feedback"]["uncovered_queries"]


@pytest.mark.asyncio
async def test_check_data_blocks_when_coverage_still_low_after_retry(tmp_path):
    agent = _make_agent(tmp_path)
    with patch("multi_agents.agents.check_data.call_model", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLM_ALL_CHECKED
        result = await agent.run({
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {"research_queries": ["nvidia 2026 ai gpu revenue audited"]},
            "iteration_index": 2,
            "scraping_packet": _packet_with_coverage(
                targets=[{"target": "nvidia 2026 ai gpu revenue", "passages": "Nvidia reported actual audited AI GPU revenue in 2026."}],
                section_coverage=0.5,
            ),
        })

    assert result["check_data_action"] == "blocked"
    assert result["check_data_verdict"]["coverage_report"]["section_coverage"] == 0.5


@pytest.mark.asyncio
async def test_hard_fail_when_no_passages(tmp_path):
    agent = _make_agent(tmp_path)
    result = await agent.run({
        "task": {},
        "topic": "2026 Nvidia AI GPU revenue",
        "research_context": {},
        "iteration_index": 1,
        "scraping_packet": _packet([{"target": "nvidia 2026 ai gpu revenue", "passages": ""}]),
    })

    assert result["check_data_action"] == "retry"
    assert result["check_data_verdict"]["guard_report"]["hard_fail"] is True
    assert "empty_passages" in result["check_data_verdict"]["guard_report"]["hard_fail_reasons"]


@pytest.mark.asyncio
async def test_llm_error_falls_back_to_unchecked(tmp_path):
    agent = _make_agent(tmp_path)
    with patch("multi_agents.agents.check_data.call_model", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = RuntimeError("connection timeout")
        result = await agent.run({
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 1,
            "scraping_packet": _packet([{
                "target": "nvidia 2026 ai gpu revenue",
                "passages": "Nvidia reported actual audited AI GPU revenue in 2026.",
            }]),
        })

    assert result["check_data_action"] == "retry"
    assert result["audit_feedback"]["confidence_score"] == 0.0
