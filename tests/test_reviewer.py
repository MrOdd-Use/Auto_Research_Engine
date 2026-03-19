import pytest
from unittest.mock import AsyncMock, patch

from multi_agents.agents.reviewer import (
    ReviewerAgent,
    _extract_changed_paragraphs,
    _build_condensed_sources,
)


@pytest.fixture
def agent():
    return ReviewerAgent()


@pytest.fixture
def source_index():
    return {
        "S1": {"content": "Apple revenue $128B in Q1 2024", "source_url": "https://reuters.com/a", "domain": "reuters.com"},
        "S2": {"content": "Apple Q1 earnings beat estimates", "source_url": "https://bloomberg.com/b", "domain": "bloomberg.com"},
    }


# ── _extract_changed_paragraphs ──────────────────────────────────────


class TestExtractChangedParagraphs:
    def test_detects_additions(self):
        original = "Line 1\nLine 2"
        revised = "Line 1\nLine 2\nLine 3 new"

        changed = _extract_changed_paragraphs(original, revised)
        assert any("Line 3" in c for c in changed)

    def test_no_changes(self):
        text = "Line 1\nLine 2"
        changed = _extract_changed_paragraphs(text, text)
        assert changed == []

    def test_detects_modifications(self):
        original = "Revenue was $100B.\nGrowth was 5%."
        revised = "Revenue was $128B.\nGrowth was 5%."

        changed = _extract_changed_paragraphs(original, revised)
        assert any("128B" in c for c in changed)


# ── _build_condensed_sources ─────────────────────────────────────────


class TestBuildCondensedSources:
    def test_basic_output(self, source_index):
        result = _build_condensed_sources(source_index)
        assert "[S1]" in result
        assert "[S2]" in result
        assert "reuters.com" in result

    def test_respects_limit(self, source_index):
        result = _build_condensed_sources(source_index, limit=1)
        assert "[S1]" in result
        assert "[S2]" not in result

    def test_empty_index(self):
        assert _build_condensed_sources({}) == ""


# ── ReviewerAgent.run ────────────────────────────────────────────────


class TestReviewerRun:
    @pytest.mark.asyncio
    @patch("multi_agents.agents.reviewer.call_model", new_callable=AsyncMock)
    async def test_guidelines_review(self, mock_call, agent):
        """With follow_guidelines=True and no source_index, behaves as before."""
        mock_call.return_value = "Section 2 needs more detail."

        result = await agent.run({
            "task": {"follow_guidelines": True, "guidelines": ["Be detailed", "Cite sources"], "model": "gpt-4o-mini"},
            "draft": "Some draft text.",
            "revision_notes": None,
            "source_index": {},
            "previous_draft": "",
        })

        assert result["review"] is not None
        assert "more detail" in result["review"]

    @pytest.mark.asyncio
    @patch("multi_agents.agents.reviewer.call_model", new_callable=AsyncMock)
    async def test_accepts_good_draft(self, mock_call, agent):
        mock_call.return_value = "None."

        result = await agent.run({
            "task": {"follow_guidelines": True, "guidelines": ["Be detailed"], "model": "gpt-4o-mini"},
            "draft": "Good draft.",
            "revision_notes": None,
            "source_index": {},
            "previous_draft": "",
        })

        assert result["review"] is None

    @pytest.mark.asyncio
    @patch("multi_agents.agents.reviewer.call_model", new_callable=AsyncMock)
    async def test_source_verify_without_guidelines(self, mock_call, agent, source_index):
        """With follow_guidelines=False but source_index present, still runs review for hallucination detection."""
        mock_call.return_value = '[HALLUCINATION] "Market cap tripled" — no source supports this.'

        result = await agent.run({
            "task": {"follow_guidelines": False, "guidelines": [], "model": "gpt-4o-mini"},
            "draft": "Revenue was $128B. Market cap tripled.",
            "revision_notes": None,
            "source_index": source_index,
            "previous_draft": "Revenue was $128B.",
        })

        assert result["review"] is not None
        assert "HALLUCINATION" in result["review"]

    @pytest.mark.asyncio
    @patch("multi_agents.agents.reviewer.call_model", new_callable=AsyncMock)
    async def test_no_guidelines_no_sources_skips(self, mock_call, agent):
        """With follow_guidelines=False and no source_index, skips review entirely."""
        result = await agent.run({
            "task": {"follow_guidelines": False, "guidelines": [], "model": "gpt-4o-mini"},
            "draft": "Some text.",
            "revision_notes": None,
            "source_index": {},
            "previous_draft": "",
        })

        assert result["review"] is None
        mock_call.assert_not_called()

    @pytest.mark.asyncio
    @patch("multi_agents.agents.reviewer.call_model", new_callable=AsyncMock)
    async def test_no_guidelines_with_sources_and_no_diff_still_audits_full_draft(self, mock_call, agent, source_index):
        """Without guidelines, source-aware review still audits the full draft when evidence is available."""
        mock_call.return_value = "none"

        result = await agent.run({
            "task": {"follow_guidelines": False, "guidelines": [], "model": "gpt-4o-mini"},
            "draft": "Revenue was $128B.",
            "revision_notes": None,
            "source_index": source_index,
            "previous_draft": "Revenue was $128B.",
        })

        assert result["review"] is None
        mock_call.assert_called_once()

    @pytest.mark.asyncio
    @patch("multi_agents.agents.reviewer.call_model", new_callable=AsyncMock)
    async def test_source_verify_block_included_in_prompt(self, mock_call, agent, source_index):
        """When previous_draft differs from current draft and source_index exists, the prompt includes source verification."""
        mock_call.return_value = "none"

        await agent.run({
            "task": {"follow_guidelines": True, "guidelines": ["Be accurate"], "model": "gpt-4o-mini"},
            "draft": "Revenue was $128B. Market cap reached $3T.",
            "revision_notes": "Fixed revenue figure.",
            "source_index": source_index,
            "previous_draft": "Revenue was $100B.",
        })

        # Verify the prompt included source verification content
        call_args = mock_call.call_args
        prompt_content = call_args[0][0][1]["content"]
        assert "Source Verification" in prompt_content
        assert "Evidence Sources" in prompt_content
        assert "Modified paragraphs" in prompt_content

    @pytest.mark.asyncio
    @patch("multi_agents.agents.reviewer.call_model", new_callable=AsyncMock)
    async def test_full_draft_source_verify_block_included_without_previous_draft(self, mock_call, agent, source_index):
        mock_call.return_value = "none"

        await agent.run({
            "task": {"follow_guidelines": False, "guidelines": [], "model": "gpt-4o-mini"},
            "draft": "Revenue was $128B. Market cap reached $3T.",
            "revision_notes": None,
            "source_index": source_index,
            "previous_draft": "",
        })

        call_args = mock_call.call_args
        prompt_content = call_args[0][0][1]["content"]
        assert "Source Verification" in prompt_content
        assert "Review the full draft above" in prompt_content

    @pytest.mark.asyncio
    @patch("multi_agents.agents.reviewer.call_model", new_callable=AsyncMock)
    async def test_no_diff_skips_source_verify(self, mock_call, agent, source_index):
        """When previous_draft equals current draft, reviewer falls back to full-draft source verification."""
        mock_call.return_value = "none"

        same_draft = "Revenue was $128B."
        await agent.run({
            "task": {"follow_guidelines": True, "guidelines": ["Be accurate"], "model": "gpt-4o-mini"},
            "draft": same_draft,
            "revision_notes": None,
            "source_index": source_index,
            "previous_draft": same_draft,
        })

        call_args = mock_call.call_args
        prompt_content = call_args[0][0][1]["content"]
        assert "Source Verification" in prompt_content
        assert "Review the full draft above" in prompt_content
