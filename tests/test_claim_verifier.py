import pytest

from multi_agents.agents.claim_verifier import ClaimVerifierAgent


@pytest.fixture
def agent():
    return ClaimVerifierAgent()


# ── build_source_index ────────────────────────────────────────────────


def _make_scraping_packets(passages):
    """Build scraping_packets from a list of (content, source_url) tuples."""
    return [
        {
            "search_log": [
                {
                    "target": "test query",
                    "top_10_passages": [
                        {"content": content, "source_url": url}
                        for content, url in passages
                    ],
                }
            ]
        }
    ]


class TestBuildSourceIndex:
    def test_basic_indexing(self, agent):
        packets = _make_scraping_packets([
            ("Apple revenue $128B", "https://reuters.com/article1"),
            ("Apple Q1 earnings", "https://bloomberg.com/news/123"),
        ])
        index, formatted = agent.build_source_index(packets)

        assert "S1" in index
        assert "S2" in index
        assert index["S1"]["domain"] == "reuters.com"
        assert index["S2"]["domain"] == "bloomberg.com"
        assert "[S1]" in formatted
        assert "[S2]" in formatted

    def test_empty_packets(self, agent):
        index, formatted = agent.build_source_index([])
        assert index == {}
        assert formatted == ""

    def test_append_only_with_start_id(self, agent):
        packets = _make_scraping_packets([
            ("New passage", "https://newsite.com/page"),
        ])
        index, formatted = agent.build_source_index(packets, start_id=5)

        assert "S5" in index
        assert "S1" not in index
        assert "[S5]" in formatted

    def test_domain_extraction_www_prefix(self, agent):
        packets = _make_scraping_packets([
            ("Content", "https://www.example.com/page"),
        ])
        index, _ = agent.build_source_index(packets)
        assert index["S1"]["domain"] == "example.com"

    def test_skips_empty_content(self, agent):
        packets = _make_scraping_packets([
            ("", "https://example.com/1"),
            ("Valid content", "https://example.com/2"),
        ])
        index, _ = agent.build_source_index(packets)
        assert len(index) == 1
        assert "S1" in index

    def test_includes_section_metadata(self, agent):
        packets = _make_scraping_packets([
            ("Valid content", "https://example.com/2"),
        ])
        index, _ = agent.build_source_index(
            packets,
            section_contexts=[
                {
                    "section_index": 0,
                    "section_key": "section_0_section_a",
                    "section_title": "Section A",
                }
            ],
        )
        assert index["S1"]["section_index"] == 0
        assert index["S1"]["section_key"] == "section_0_section_a"
        assert index["S1"]["section_title"] == "Section A"


# ── parse_citations ──────────────────────────────────────────────────


class TestParseCitations:
    def test_inline_citations(self, agent):
        source_index = {
            "S1": {"content": "...", "source_url": "https://reuters.com/a", "domain": "reuters.com"},
            "S2": {"content": "...", "source_url": "https://bloomberg.com/b", "domain": "bloomberg.com"},
        }
        writer_output = {
            "introduction": "Revenue was $128B [S1][S2]. Growth was strong.",
            "conclusion": "Overall positive outlook [S1].",
        }

        claims = agent.parse_citations(writer_output, source_index)

        assert len(claims) >= 2
        revenue_claim = next(c for c in claims if "128B" in c["claim_text"])
        assert "S1" in revenue_claim["source_ids"]
        assert "S2" in revenue_claim["source_ids"]
        assert "reuters.com" in revenue_claim["domains"]
        assert "bloomberg.com" in revenue_claim["domains"]

    def test_structured_annotations(self, agent):
        source_index = {
            "S1": {"content": "...", "source_url": "https://reuters.com/a", "domain": "reuters.com"},
        }
        writer_output = {
            "introduction": "Revenue was $128B [S1].",
            "conclusion": "",
            "claim_annotations": [
                {"sentence": "Revenue was $128B", "source_ids": ["S1"], "section": "introduction"},
            ],
        }

        claims = agent.parse_citations(writer_output, source_index)

        assert len(claims) == 1
        assert claims[0]["source_ids"] == ["S1"]
        assert claims[0]["domains"] == ["reuters.com"]

    def test_merges_structured_annotations_with_inline_claims(self, agent):
        source_index = {
            "S1": {"content": "...", "source_url": "https://reuters.com/a", "domain": "reuters.com"},
            "S2": {"content": "...", "source_url": "https://bloomberg.com/b", "domain": "bloomberg.com"},
        }
        writer_output = {
            "introduction": "Revenue was $128B [S1].",
            "conclusion": "Operating margin improved [S2].",
            "claim_annotations": [
                {"sentence": "Revenue was $128B", "source_ids": ["S1"], "section": "introduction"},
            ],
        }

        claims = agent.parse_citations(writer_output, source_index)

        assert len(claims) == 2
        assert any(c["claim_text"] == "Revenue was $128B" for c in claims)
        assert any(c["claim_text"] == "Operating margin improved" for c in claims)

    def test_no_citations(self, agent):
        source_index = {"S1": {"content": "...", "source_url": "https://a.com/x", "domain": "a.com"}}
        writer_output = {
            "introduction": "Revenue was high. Growth was strong.",
            "conclusion": "",
        }

        claims = agent.parse_citations(writer_output, source_index)

        assert len(claims) >= 1
        # All claims should have empty source_ids
        for claim in claims:
            assert claim["source_ids"] == []


# ── classify_claims ──────────────────────────────────────────────────


class TestClassifyClaims:
    @pytest.mark.asyncio
    async def test_hallucination(self, agent):
        claims = [
            {"claim_text": "Revenue was $999T", "source_ids": [], "domains": [], "source_section": "intro", "original_sentence": "Revenue was $999T"},
        ]
        classified = await agent.classify_claims(claims, {}, "gpt-4o-mini")

        assert len(classified) == 1
        assert classified[0]["confidence"] == "HALLUCINATION"

    @pytest.mark.asyncio
    async def test_medium_single_domain(self, agent):
        source_index = {
            "S1": {"content": "Revenue $128B", "source_url": "https://reuters.com/a", "domain": "reuters.com"},
        }
        claims = [
            {"claim_text": "Revenue was $128B", "source_ids": ["S1"], "domains": ["reuters.com"], "source_section": "intro", "original_sentence": "Revenue was $128B [S1]"},
        ]

        classified = await agent.classify_claims(claims, source_index, "gpt-4o-mini")

        assert classified[0]["confidence"] == "MEDIUM"

    @pytest.mark.asyncio
    async def test_high_multiple_domains_no_conflict(self, agent):
        source_index = {
            "S1": {"content": "Revenue $128B", "source_url": "https://reuters.com/a", "domain": "reuters.com"},
            "S2": {"content": "Revenue $128B", "source_url": "https://bloomberg.com/b", "domain": "bloomberg.com"},
        }
        claims = [
            {
                "claim_text": "Revenue was $128B",
                "source_ids": ["S1", "S2"],
                "domains": ["reuters.com", "bloomberg.com"],
                "source_section": "intro",
                "original_sentence": "Revenue was $128B [S1][S2]",
            },
        ]

        # Mock detect_conflicts to return no conflict
        original_detect = agent.detect_conflicts

        async def mock_detect(claim, si, model):
            return {"has_conflict": False, "conflict_detail": ""}

        agent.detect_conflicts = mock_detect
        try:
            classified = await agent.classify_claims(claims, source_index, "gpt-4o-mini")
            assert classified[0]["confidence"] == "HIGH"
        finally:
            agent.detect_conflicts = original_detect

    @pytest.mark.asyncio
    async def test_suspicious_with_conflict(self, agent):
        source_index = {
            "S1": {"content": "Market share 23%", "source_url": "https://statista.com/a", "domain": "statista.com"},
            "S2": {"content": "Market share 21.5%", "source_url": "https://idc.com/b", "domain": "idc.com"},
        }
        claims = [
            {
                "claim_text": "Market share was 23%",
                "source_ids": ["S1", "S2"],
                "domains": ["statista.com", "idc.com"],
                "source_section": "intro",
                "original_sentence": "Market share was 23% [S1][S2]",
            },
        ]

        async def mock_detect(claim, si, model):
            return {"has_conflict": True, "conflict_detail": "Statista says 23%, IDC says 21.5%"}

        agent.detect_conflicts = mock_detect
        classified = await agent.classify_claims(claims, source_index, "gpt-4o-mini")
        assert classified[0]["confidence"] == "SUSPICIOUS"
        assert "21.5%" in classified[0]["note"]


class TestClaimVerifierRun:
    @pytest.mark.asyncio
    async def test_keeps_parsed_claims_when_fallback_returns_nothing(self, agent):
        async def mock_fallback(writer_output, source_index, model):
            return []

        agent.fallback_match_claims = mock_fallback

        result = await agent.run(
            {
                "task": {"model": "gpt-4o-mini"},
                "source_index": {
                    "S1": {
                        "content": "Revenue was reported in filings",
                        "source_url": "https://reuters.com/a",
                        "domain": "reuters.com",
                    }
                },
                "introduction": "Revenue increased sharply.",
                "conclusion": "",
                "claim_annotations": None,
            }
        )

        assert len(result["claim_confidence_report"]) == 1
        assert result["claim_confidence_report"][0]["confidence"] == "HALLUCINATION"


# ── citation coverage ────────────────────────────────────────────────


class TestCitationCoverage:
    def test_full_coverage(self, agent):
        claims = [
            {"source_ids": ["S1"]},
            {"source_ids": ["S2"]},
        ]
        assert agent._check_citation_coverage(claims) == 1.0

    def test_no_coverage(self, agent):
        claims = [
            {"source_ids": []},
            {"source_ids": []},
        ]
        assert agent._check_citation_coverage(claims) == 0.0

    def test_partial_coverage(self, agent):
        claims = [
            {"source_ids": ["S1"]},
            {"source_ids": []},
        ]
        assert agent._check_citation_coverage(claims) == 0.5

    def test_empty_claims(self, agent):
        assert agent._check_citation_coverage([]) == 0.0


# ── annotate_draft ───────────────────────────────────────────────────


class TestAnnotateDraft:
    def test_hallucination_replacement(self, agent):
        draft = "Revenue was $128B. New product launches in March."
        report = [
            {
                "claim_text": "New product launches in March",
                "confidence": "HALLUCINATION",
                "original_sentence": "New product launches in March.",
                "domains": [],
                "note": "无来源支持",
            },
        ]

        result = agent.annotate_draft(draft, report)
        assert "[该方面的知识无从得知" in result
        assert "New product launches in March." not in result

    def test_medium_tag(self, agent):
        draft = "Revenue was $128B [S1]."
        report = [
            {
                "claim_text": "Revenue was $128B",
                "confidence": "MEDIUM",
                "original_sentence": "Revenue was $128B [S1].",
                "domains": ["reuters.com"],
                "note": "仅 reuters.com",
            },
        ]

        result = agent.annotate_draft(draft, report)
        assert "[单一来源]" in result
        assert "[S1]" not in result  # citations stripped

    def test_suspicious_tag(self, agent):
        draft = "Market share 23% [S1][S2]."
        report = [
            {
                "claim_text": "Market share 23%",
                "confidence": "SUSPICIOUS",
                "original_sentence": "Market share 23% [S1][S2].",
                "domains": ["statista.com", "idc.com"],
                "note": "Conflicting data",
            },
        ]

        result = agent.annotate_draft(draft, report)
        assert "[来源冲突" in result

    def test_high_no_annotation(self, agent):
        draft = "Revenue was $128B [S1][S2]."
        report = [
            {
                "claim_text": "Revenue was $128B",
                "confidence": "HIGH",
                "original_sentence": "Revenue was $128B [S1][S2].",
                "domains": ["reuters.com", "bloomberg.com"],
                "note": "",
            },
        ]

        result = agent.annotate_draft(draft, report)
        assert "[单一来源]" not in result
        assert "[来源冲突" not in result
        assert "[该方面的知识无从得知" not in result

    def test_summary_table_appended(self, agent):
        draft = "Some text."
        report = [
            {
                "claim_text": "Revenue was $128B",
                "confidence": "HIGH",
                "original_sentence": "Revenue was $128B.",
                "domains": ["reuters.com", "bloomberg.com"],
                "note": "",
            },
        ]

        result = agent.annotate_draft(draft, report)
        assert "## 声明置信度报告" in result
        assert "高" in result


# ── group_by_section ─────────────────────────────────────────────────


class TestGroupBySection:
    def test_grouping(self, agent):
        claims = [
            {"claim_text": "A", "source_section": "introduction"},
            {"claim_text": "B", "source_section": "conclusion"},
            {"claim_text": "C", "source_section": "introduction"},
        ]

        groups = agent.group_by_section(claims)

        assert len(groups) == 2
        assert len(groups["introduction"]) == 2
        assert len(groups["conclusion"]) == 1

    def test_prefers_source_index_section_keys(self, agent):
        claims = [
            {
                "claim_text": "A",
                "source_section": "introduction",
                "source_ids": ["S1", "S2"],
            },
        ]
        source_index = {
            "S1": {"section_key": "section_0_section_a"},
            "S2": {"section_key": "section_0_section_a"},
        }

        groups = agent.group_by_section(claims, source_index)

        assert list(groups.keys()) == ["section_0_section_a"]


# ── build_reflexion_note ─────────────────────────────────────────────


class TestBuildReflexionNote:
    def test_note_content(self, agent):
        claims = [
            {
                "claim_text": "Market share was 23%",
                "source_ids": ["S1", "S2"],
                "note": "Conflicting data",
            },
        ]
        source_index = {
            "S1": {"content": "Share 23%", "source_url": "https://a.com", "domain": "a.com"},
            "S2": {"content": "Share 21.5%", "source_url": "https://b.com", "domain": "b.com"},
        }

        note = agent.build_reflexion_note(claims, source_index)

        assert "Market share was 23%" in note
        assert "来源冲突" in note or "矛盾" in note
        assert "建议搜索" in note


# ── _extract_domain ──────────────────────────────────────────────────


class TestExtractDomain:
    def test_standard_url(self, agent):
        assert agent._extract_domain("https://reuters.com/article/123") == "reuters.com"

    def test_www_prefix(self, agent):
        assert agent._extract_domain("https://www.bloomberg.com/news") == "bloomberg.com"

    def test_empty_url(self, agent):
        assert agent._extract_domain("") == ""

    def test_url_without_scheme(self, agent):
        assert agent._extract_domain("example.com/page") == "example.com"
