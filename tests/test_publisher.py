"""Tests for the publisher report artifact workflow."""

from __future__ import annotations

import pytest

from multi_agents.agents.publisher import PublisherAgent


def test_generate_layout_places_introduction_before_sections_and_conclusion():
    """Ensure the readable report flow is TOC, introduction, sections, conclusion."""
    publisher = PublisherAgent("unused")
    research_state = {
        "headers": {
            "title": "AI and Jobs",
            "date": "Date",
            "table_of_contents": "Table of Contents",
            "introduction": "Introduction",
            "conclusion": "Conclusion",
            "references": "References",
        },
        "date": "2026-04-09",
        "table_of_contents": "- Section One",
        "introduction": "Intro text.",
        "research_data": ["### Section One\n\nBody text."],
        "conclusion": "Closing text.",
        "sources": ["- Ref 1"],
    }

    result = publisher.generate_layout(research_state)

    assert result.index("## Table of Contents") < result.index("## Introduction")
    assert result.index("## Introduction") < result.index("### Section One")
    assert result.index("### Section One") < result.index("## Conclusion")
    assert result.index("## Conclusion") < result.index("## References")


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
