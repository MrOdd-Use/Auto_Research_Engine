import pytest
from unittest.mock import AsyncMock, patch

from multi_agents.agents.reviser import ReviserAgent


@pytest.mark.asyncio
@patch("multi_agents.agents.reviser.call_model", new_callable=AsyncMock)
async def test_reviser_prompt_is_clean_and_preserves_structure(mock_call):
    mock_call.return_value = {"draft": "updated", "revision_notes": "done"}
    agent = ReviserAgent()

    result = await agent.run(
        {
            "task": {"model": "gpt-4o-mini", "verbose": False},
            "draft": "# Title\n## Introduction\nDraft body",
            "review": "Please tighten the introduction.",
            "revision_notes": None,
        }
    )

    assert result["draft"] == "updated"
    prompt = mock_call.call_args.args[0]
    user_content = prompt[1]["content"]
    assert 'Reviewer\'s notes:' in user_content
    assert '" + "Reviewer\'s notes:' not in user_content
    assert "preserve the existing report structure and section headings" in user_content


@pytest.mark.asyncio
@patch("multi_agents.agents.reviser.call_model", new_callable=AsyncMock)
async def test_reviser_prompt_includes_resolved_opinion_guardrail(mock_call):
    mock_call.return_value = {"draft": "updated", "revision_notes": "done"}
    agent = ReviserAgent()

    await agent.run(
        {
            "task": {"model": "gpt-4o-mini", "verbose": False},
            "draft": "# Title\n## Introduction\nDraft body",
            "review": "Please tighten the conclusion.",
            "revision_notes": None,
            "pending_opinions": "1. [Round 2 / Agent / unresolved] Tighten the conclusion wording",
            "resolved_opinions": "1. [Round 1 / Agent / resolved] Keep the labor statistics citation in the introduction",
        }
    )

    prompt = mock_call.call_args.args[0]
    user_content = prompt[1]["content"]
    assert "Previously Resolved Opinion Items" in user_content
    assert "Do not regress these already-satisfied requirements" in user_content
    assert "Keep the labor statistics citation in the introduction" in user_content
