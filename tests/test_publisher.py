"""Tests for the publisher report artifact workflow."""

from __future__ import annotations

import pytest

from multi_agents.agents.publisher import PublisherAgent


@pytest.mark.asyncio
async def test_publish_research_report_writes_only_fixed_markdown_artifacts(tmp_path):
    """Ensure publishing does not create random-stem markdown duplicates."""
    publisher = PublisherAgent(str(tmp_path))
    research_state = {
        "final_draft": "# Title\n\nBody text.",
        "source_index": {},
        "claim_confidence_report": [],
    }

    result = await publisher.publish_research_report(
        research_state,
        {"markdown": True, "pdf": False, "docx": False},
    )

    markdown_files = sorted(path.name for path in tmp_path.glob("*.md"))

    report_text = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert result == "# Title\n\nBody text."
    assert markdown_files == ["evidence_base.md", "report.md"]
    assert report_text.startswith("# Title\n\nBody text.")
