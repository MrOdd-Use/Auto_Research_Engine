"""Tests for multi_agents.agents.utils.output_writers."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from multi_agents.agents.utils.output_writers import (
    collect_cited_ids_from_text,
    format_annotated_report_md,
    format_evidence_base_md,
    format_model_decisions_json,
    format_model_decisions_md,
    write_annotated_report,
    write_evidence_base,
    write_model_decisions,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

SAMPLE_DECISIONS = [
    {
        "type": "route_decision",
        "agent_role": "editor",
        "stage_name": "plan_research",
        "selected_model": "gpt-4o",
        "selected_provider": "openai",
        "routing_reason": "layered_learning",
        "candidates": [
            {"model": "gpt-4o-mini", "provider": "openai"},
            {"model": "gpt-4o", "provider": "openai"},
        ],
        "route_latency_ms": 2.34,
    },
    {
        "type": "route_decision",
        "agent_role": "researcher",
        "stage_name": "web_search",
        "selected_model": "gpt-4o-mini",
        "selected_provider": "openai",
        "routing_reason": "cost_efficient",
        "candidates": [{"model": "gpt-4o-mini", "provider": "openai"}],
        "route_latency_ms": 1.12,
    },
]

SAMPLE_SOURCE_INDEX = {
    "S1": {
        "content": "The global AI market size was valued at $136.6 billion in 2022.",
        "source_url": "https://www.statista.com/statistics/ai-market",
        "domain": "statista.com",
        "section_key": "market_overview",
        "section_title": "Market Overview",
    },
    "S2": {
        "content": "OpenAI revenue reportedly reached $3.4 billion annually.",
        "source_url": "https://reuters.com/tech/openai-revenue",
        "domain": "reuters.com",
        "section_key": "company_analysis",
        "section_title": "Company Analysis",
    },
    "S3": {
        "content": "China accounts for roughly 15% of global AI investment.",
        "source_url": "https://example.com/china-ai",
        "domain": "example.com",
        "section_key": "market_overview",
        "section_title": "Market Overview",
    },
}

SAMPLE_CLAIM_REPORT = [
    {
        "claim_text": "AI market size was valued at $136.6 billion",
        "confidence": "HIGH",
        "source_ids": ["S1"],
        "original_sentence": "The AI market size was valued at $136.6 billion [S1].",
    },
    {
        "claim_text": "OpenAI revenue reached $3.4 billion",
        "confidence": "MEDIUM",
        "source_ids": ["S2"],
        "original_sentence": "OpenAI revenue reached $3.4 billion [S2].",
    },
    {
        "claim_text": "China leads global AI development",
        "confidence": "HALLUCINATION",
        "source_ids": [],
        "original_sentence": "China leads global AI development.",
    },
]


# ── Model Decisions Tests ────────────────────────────────────────────────


class TestModelDecisionsJson:
    def test_valid_json(self):
        result = format_model_decisions_json(SAMPLE_DECISIONS)
        parsed = json.loads(result)
        assert "decisions" in parsed
        assert "summary" in parsed

    def test_summary_fields(self):
        result = json.loads(format_model_decisions_json(SAMPLE_DECISIONS))
        summary = result["summary"]
        assert summary["total_calls"] == 2
        assert "gpt-4o" in summary["models_used"]
        assert "gpt-4o-mini" in summary["models_used"]
        assert summary["total_route_latency_ms"] == pytest.approx(3.46, abs=0.01)

    def test_empty_decisions(self):
        result = json.loads(format_model_decisions_json([]))
        assert result["summary"]["total_calls"] == 0
        assert result["decisions"] == []


class TestModelDecisionsMd:
    def test_contains_table_header(self):
        result = format_model_decisions_md(SAMPLE_DECISIONS)
        assert "| # | Agent Role |" in result
        assert "## Summary" in result

    def test_contains_decision_rows(self):
        result = format_model_decisions_md(SAMPLE_DECISIONS)
        assert "editor" in result
        assert "gpt-4o" in result
        assert "researcher" in result

    def test_empty_decisions(self):
        result = format_model_decisions_md([])
        assert "Total calls**: 0" in result


# ── Evidence Base Tests ──────────────────────────────────────────────────


class TestEvidenceBaseMd:
    def test_contains_all_entries(self):
        result = format_evidence_base_md(SAMPLE_SOURCE_INDEX)
        assert "### S1" in result
        assert "### S2" in result
        assert "### S3" in result

    def test_sorted_by_id(self):
        result = format_evidence_base_md(SAMPLE_SOURCE_INDEX)
        pos_s1 = result.index("### S1")
        pos_s2 = result.index("### S2")
        pos_s3 = result.index("### S3")
        assert pos_s1 < pos_s2 < pos_s3

    def test_fields_present(self):
        result = format_evidence_base_md(SAMPLE_SOURCE_INDEX)
        assert "**URL**:" in result
        assert "**Domain**:" in result
        assert "**Section**:" in result
        assert "**Content**:" in result

    def test_empty_index(self):
        result = format_evidence_base_md({})
        assert "No evidence collected" in result

    def test_cited_annotation(self):
        cited = {"S1", "S3"}
        result = format_evidence_base_md(SAMPLE_SOURCE_INDEX, cited_ids=cited)
        assert "(cited" in result
        assert "(unused)" in result

    def test_no_cited_ids_means_no_annotations(self):
        result = format_evidence_base_md(SAMPLE_SOURCE_INDEX, cited_ids=None)
        assert "(cited" not in result
        assert "(unused)" not in result


# ── Annotated Report Tests ───────────────────────────────────────────────


class TestAnnotatedReportMd:
    DRAFT = (
        "# AI Market Report\n\n"
        "The AI market size was valued at $136.6 billion [S1].\n"
        "OpenAI revenue reached $3.4 billion [S2].\n"
        "China leads global AI development.\n"
    )

    def test_confidence_tags_inserted(self):
        result = format_annotated_report_md(
            self.DRAFT, SAMPLE_CLAIM_REPORT, SAMPLE_SOURCE_INDEX
        )
        assert "[HIGH]" in result
        assert "[MEDIUM]" in result
        assert "[HALLUCINATION]" in result

    def test_citations_preserved(self):
        result = format_annotated_report_md(
            self.DRAFT, SAMPLE_CLAIM_REPORT, SAMPLE_SOURCE_INDEX
        )
        assert "[S1]" in result
        assert "[S2]" in result

    def test_citation_statistics_appended(self):
        result = format_annotated_report_md(
            self.DRAFT, SAMPLE_CLAIM_REPORT, SAMPLE_SOURCE_INDEX
        )
        assert "## Citation Statistics" in result
        assert "Total claims**: 3" in result

    def test_empty_report(self):
        assert format_annotated_report_md("", [], {}) == ""

    def test_no_claims(self):
        result = format_annotated_report_md("Some text", [], SAMPLE_SOURCE_INDEX)
        assert "No claims verified" in result


# ── Collect Cited IDs ────────────────────────────────────────────────────


class TestCollectCitedIds:
    def test_basic(self):
        text = "Fact one [S1][S3]. Fact two [S1]."
        result = collect_cited_ids_from_text(text)
        assert result["S1"] == 2
        assert result["S3"] == 1
        assert "S2" not in result

    def test_empty(self):
        assert len(collect_cited_ids_from_text("no citations here")) == 0


# ── Async Writer Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
class TestAsyncWriters:
    async def test_write_model_decisions(self, tmp_path):
        await write_model_decisions(str(tmp_path), SAMPLE_DECISIONS)
        assert (tmp_path / "model_decisions.json").exists()
        assert (tmp_path / "model_decisions.md").exists()
        content = json.loads((tmp_path / "model_decisions.json").read_text("utf-8"))
        assert content["summary"]["total_calls"] == 2

    async def test_write_evidence_base(self, tmp_path):
        await write_evidence_base(str(tmp_path), SAMPLE_SOURCE_INDEX)
        path = tmp_path / "evidence_base.md"
        assert path.exists()
        assert "### S1" in path.read_text("utf-8")

    async def test_write_annotated_report(self, tmp_path):
        draft = "Fact [S1]. Another fact."
        await write_annotated_report(str(tmp_path), draft, [], SAMPLE_SOURCE_INDEX)
        path = tmp_path / "report.md"
        assert path.exists()
        assert "[S1]" in path.read_text("utf-8")
