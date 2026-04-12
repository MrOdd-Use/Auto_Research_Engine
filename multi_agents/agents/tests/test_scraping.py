"""
Unit tests for ScrapingAgent (multi_agents/agents/scraping.py).

Covers:
- Pure utility methods (no network / LLM calls)
- Validation and filtering logic
- Coverage snapshot calculation
- run_depth_scraping main flow with mocked dependencies
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Stub heavy imports before ScrapingAgent is loaded ───────────────────────────

_state_stub = MagicMock()
_state_stub.set_tier = MagicMock()
_state_stub.get_current_model = MagicMock(return_value="stub-model")

with (
    patch.dict(
        sys.modules,
        {
            "multi_agents.route_agent": MagicMock(
                build_route_context=MagicMock(return_value={})
            ),
            "multi_agents.agents.utils.llms": MagicMock(
                call_model=AsyncMock(return_value=[])
            ),
            "multi_agents.agents.utils.views": MagicMock(
                print_agent_output=MagicMock()
            ),
        },
    ),
    patch(
        "multi_agents.agents.scraping.StateController",
        return_value=_state_stub,
    ),
    patch("multi_agents.agents.scraping.Config", MagicMock()),
    patch("multi_agents.agents.scraping.WorkerPool", MagicMock()),
    patch(
        "multi_agents.agents.scraping.scrape_urls",
        AsyncMock(return_value=([], [])),
    ),
    patch("multi_agents.agents.scraping.get_retriever", MagicMock(return_value=None)),
):
    from multi_agents.agents.scraping import ScrapingAgent


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_agent(**env_overrides) -> ScrapingAgent:
    """Return a ScrapingAgent with deterministic env config."""
    env = {
        "SCRAPING_MMR_USE_EMBEDDINGS": "false",
        "SCRAPING_MIN_SEARCH_TARGETS": "2",
        "SCRAPING_MAX_SEARCH_TARGETS": "4",
        **env_overrides,
    }
    with patch.dict(os.environ, env, clear=False):
        return ScrapingAgent()


def make_passage(content: str, url: str = "https://example.com") -> dict:
    return {"content": content, "source_url": url, "metadata": {}}


# ═══════════════════════════════════════════════════════════════════════════
# 1. _normalize_iteration
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeIteration:
    def setup_method(self):
        self.agent = make_agent()

    def test_valid_values(self):
        assert self.agent._normalize_iteration(1) == 1
        assert self.agent._normalize_iteration(2) == 2
        assert self.agent._normalize_iteration(3) == 3

    def test_clamps_below_one(self):
        assert self.agent._normalize_iteration(0) == 1
        assert self.agent._normalize_iteration(-5) == 1

    def test_clamps_above_three(self):
        assert self.agent._normalize_iteration(4) == 3
        assert self.agent._normalize_iteration(100) == 3

    def test_non_numeric_defaults_to_one(self):
        assert self.agent._normalize_iteration("abc") == 1
        assert self.agent._normalize_iteration(None) == 1

    def test_string_numeric_parsed(self):
        assert self.agent._normalize_iteration("2") == 2

    def test_float_truncated(self):
        assert self.agent._normalize_iteration(2.9) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 2. _normalize_research_queries
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeResearchQueries:
    def setup_method(self):
        self.agent = make_agent()

    def test_basic_list(self):
        result = self.agent._normalize_research_queries(["A", "B", "C"])
        assert result == ["A", "B", "C"]

    def test_deduplicates_case_insensitive(self):
        result = self.agent._normalize_research_queries(["hello", "HELLO", "world"])
        assert result == ["hello", "world"]

    def test_strips_and_collapses_whitespace(self):
        result = self.agent._normalize_research_queries(["  foo   bar  "])
        assert result == ["foo bar"]

    def test_skips_empty_strings(self):
        result = self.agent._normalize_research_queries(["", "  ", "valid"])
        assert result == ["valid"]

    def test_non_list_returns_empty(self):
        assert self.agent._normalize_research_queries(None) == []
        assert self.agent._normalize_research_queries("string") == []
        assert self.agent._normalize_research_queries(42) == []

    def test_preserves_order(self):
        result = self.agent._normalize_research_queries(["c", "a", "b"])
        assert result == ["c", "a", "b"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. _tokenize_validation
# ═══════════════════════════════════════════════════════════════════════════


class TestTokenizeValidation:
    def setup_method(self):
        self.agent = make_agent()

    def test_basic_english_tokens(self):
        tokens = self.agent._tokenize_validation("AI impact on jobs")
        assert "impact" in tokens
        assert "jobs" in tokens

    def test_stopwords_removed(self):
        tokens = self.agent._tokenize_validation("the impact of AI on the job market")
        assert "the" not in tokens
        assert "of" not in tokens
        assert "on" not in tokens
        assert "impact" in tokens

    def test_single_ascii_letters_removed(self):
        tokens = self.agent._tokenize_validation("a b c impact")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "impact" in tokens

    def test_cjk_characters_included(self):
        tokens = self.agent._tokenize_validation("AI对就业的影响")
        # At least one CJK char should be present
        cjk_in = any("\u4e00" <= ch <= "\u9fff" for ch in tokens)
        assert cjk_in

    def test_empty_string(self):
        assert self.agent._tokenize_validation("") == []

    def test_lowercases_output(self):
        tokens = self.agent._tokenize_validation("Python JAVA")
        assert "python" in tokens
        assert "java" in tokens


# ═══════════════════════════════════════════════════════════════════════════
# 4. _text_to_vector and _cosine_similarity
# ═══════════════════════════════════════════════════════════════════════════


class TestVectorAndCosineSimilarity:
    def setup_method(self):
        self.agent = make_agent()

    def test_text_to_vector_is_counter(self):
        vec = self.agent._text_to_vector("ai ai jobs")
        assert isinstance(vec, Counter)
        assert vec["ai"] == 2
        assert vec["jobs"] == 1

    def test_identical_texts_similarity_is_one(self):
        v = self.agent._text_to_vector("machine learning model")
        assert self.agent._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_disjoint_texts_similarity_is_zero(self):
        v1 = self.agent._text_to_vector("apple orange")
        v2 = self.agent._text_to_vector("quantum physics")
        assert self.agent._cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_empty_vector_returns_zero(self):
        v = self.agent._text_to_vector("hello")
        empty = Counter()
        assert self.agent._cosine_similarity(v, empty) == 0.0
        assert self.agent._cosine_similarity(empty, v) == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        v1 = self.agent._text_to_vector("ai machine learning")
        v2 = self.agent._text_to_vector("ai neural network")
        sim = self.agent._cosine_similarity(v1, v2)
        assert 0.0 < sim < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. _parse_targets
# ═══════════════════════════════════════════════════════════════════════════


class TestParseTargets:
    def setup_method(self):
        self.agent = make_agent()

    def test_list_input(self):
        assert self.agent._parse_targets(["a", "b", "c"]) == ["a", "b", "c"]

    def test_dict_targets_key(self):
        assert self.agent._parse_targets({"targets": ["x", "y"]}) == ["x", "y"]

    def test_dict_queries_key(self):
        assert self.agent._parse_targets({"queries": ["q1", "q2"]}) == ["q1", "q2"]

    def test_dict_search_targets_key(self):
        assert self.agent._parse_targets({"search_targets": ["s1"]}) == ["s1"]

    def test_json_string(self):
        result = self.agent._parse_targets('["foo", "bar"]')
        assert result == ["foo", "bar"]

    def test_quoted_strings_fallback(self):
        # json_repair is bypassed by patching it to raise; regex branch then runs
        with patch("multi_agents.agents.scraping.json_repair.loads", side_effect=ValueError("bad json")):
            result = self.agent._parse_targets('Result: "alpha result query", then "beta data query".')
        assert "alpha result query" in result
        assert "beta data query" in result

    def test_unknown_type_returns_empty(self):
        assert self.agent._parse_targets(42) == []
        assert self.agent._parse_targets({}) == []

    def test_max_targets_enforced(self):
        result = self.agent._parse_targets([f"target_{i}" for i in range(20)])
        assert len(result) <= self.agent._max_search_targets


# ═══════════════════════════════════════════════════════════════════════════
# 6. _clean_targets
# ═══════════════════════════════════════════════════════════════════════════


class TestCleanTargets:
    def setup_method(self):
        self.agent = make_agent()

    def test_deduplicates_case_insensitive(self):
        result = self.agent._clean_targets(["foo", "FOO", "bar"])
        lowered = [x.lower() for x in result]
        assert lowered.count("foo") == 1

    def test_strips_whitespace(self):
        result = self.agent._clean_targets(["  hello  world  "])
        assert result == ["hello world"]

    def test_skips_empty(self):
        result = self.agent._clean_targets(["", "  ", "valid"])
        assert result == ["valid"]

    def test_respects_max_targets(self):
        result = self.agent._clean_targets([f"item_{i}" for i in range(100)])
        assert len(result) <= self.agent._max_search_targets


# ═══════════════════════════════════════════════════════════════════════════
# 7. _fallback_targets and _fallback_targets_from_context
# ═══════════════════════════════════════════════════════════════════════════


class TestFallbackTargets:
    def setup_method(self):
        self.agent = make_agent()

    def test_fallback_targets_contains_topic(self):
        targets = self.agent._fallback_targets("AI employment", "")
        assert all("AI employment" in t for t in targets)
        assert len(targets) == 3

    def test_fallback_targets_appends_hints(self):
        targets = self.agent._fallback_targets("AI", "focus on 2024")
        assert any("focus on 2024" in t for t in targets)

    def test_fallback_from_context_uses_description(self):
        ctx = {"description": "impact on labor", "key_points": []}
        result = self.agent._fallback_targets_from_context("AI employment", ctx)
        assert any("AI employment" in t for t in result)

    def test_fallback_from_context_uses_key_points(self):
        ctx = {"description": "", "key_points": ["wage gap", "skill mismatch"]}
        result = self.agent._fallback_targets_from_context("AI impact", ctx)
        assert any("wage gap" in t for t in result)

    def test_fallback_from_context_no_context_returns_query(self):
        ctx = {"description": "", "key_points": []}
        result = self.agent._fallback_targets_from_context("fallback query", ctx)
        assert result == ["fallback query"]


# ═══════════════════════════════════════════════════════════════════════════
# 8. _merge_extra_hints
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeExtraHints:
    def setup_method(self):
        self.agent = make_agent()

    def test_hint_only(self):
        assert self.agent._merge_extra_hints("hint", None) == "hint"

    def test_audit_instruction_used(self):
        result = self.agent._merge_extra_hints(None, {"instruction": "focus on 2024"})
        assert "focus on 2024" in result

    def test_audit_new_query_suggestion_used(self):
        result = self.agent._merge_extra_hints(None, {"new_query_suggestion": "add studies"})
        assert "add studies" in result

    def test_both_merged_with_separator(self):
        result = self.agent._merge_extra_hints("hint", {"new_query_suggestion": "extra"})
        assert " ; " in result
        assert "hint" in result
        assert "extra" in result

    def test_blank_values_omitted(self):
        assert self.agent._merge_extra_hints("  ", {"instruction": "  "}) == ""

    def test_non_dict_audit_ignored(self):
        assert self.agent._merge_extra_hints("hint", "not a dict") == "hint"

    def test_no_inputs_returns_empty(self):
        assert self.agent._merge_extra_hints(None, None) == ""


# ═══════════════════════════════════════════════════════════════════════════
# 9. _select_engines and _classify_domain
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineSelection:
    def setup_method(self):
        self.agent = make_agent()

    def test_iteration_1_returns_tavily_only(self):
        assert self.agent._select_engines(1, "anything") == ["tavily"]

    def test_iteration_2_with_tech_topic(self):
        engines = self.agent._select_engines(2, "LLM cloud GPU model")
        assert engines == ["arxiv", "semantic_scholar", "tavily"]
        assert "google" not in engines
        assert "bing" not in engines

    def test_classify_medical(self):
        assert self.agent._classify_domain("medical clinical drug patient") == "medical"

    def test_classify_tech(self):
        assert self.agent._classify_domain("LLM model GPU cloud ai") == "tech"

    def test_classify_finance(self):
        assert self.agent._classify_domain("stock earnings revenue valuation bank") == "finance"

    def test_classify_general_default(self):
        assert self.agent._classify_domain("weather cooking travel") == "general"

    def test_medical_takes_priority_over_tech(self):
        # "medical" keyword beats tech keywords
        assert self.agent._classify_domain("medical AI model") == "medical"


# ═══════════════════════════════════════════════════════════════════════════
# 10. _normalize_search_results
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeSearchResults:
    def setup_method(self):
        self.agent = make_agent()

    def test_normalizes_href(self):
        raw = [{"href": "https://example.com", "title": "T", "body": "B"}]
        result = self.agent._normalize_search_results(raw, "google")
        assert result[0]["href"] == "https://example.com"
        assert result[0]["source_engine"] == "google"

    def test_falls_back_to_url_field(self):
        raw = [{"url": "https://fallback.com", "title": "T", "body": "B"}]
        result = self.agent._normalize_search_results(raw, "bing")
        assert result[0]["href"] == "https://fallback.com"

    def test_skips_items_without_url(self):
        raw = [{"title": "no url", "body": "B"}]
        assert self.agent._normalize_search_results(raw, "google") == []

    def test_body_fallback_chain(self):
        raw = [{"href": "https://x.com", "title": "T", "raw_content": "RC"}]
        result = self.agent._normalize_search_results(raw, "g")
        assert result[0]["body"] == "RC"

    def test_non_list_returns_empty(self):
        assert self.agent._normalize_search_results(None, "google") == []
        assert self.agent._normalize_search_results("bad", "google") == []

    def test_non_dict_items_skipped(self):
        raw = ["string_item", {"href": "https://ok.com"}]
        result = self.agent._normalize_search_results(raw, "g")
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 11. _dedupe_search_results and _normalize_url
# ═══════════════════════════════════════════════════════════════════════════


class TestDeduplication:
    def setup_method(self):
        self.agent = make_agent()

    def test_deduplicates_exact_url(self):
        raw = [
            {"href": "https://example.com/page", "title": "A"},
            {"href": "https://example.com/page", "title": "B"},
        ]
        assert len(self.agent._dedupe_search_results(raw)) == 1

    def test_deduplicates_trailing_slash_variant(self):
        raw = [
            {"href": "https://example.com/page/"},
            {"href": "https://example.com/page"},
        ]
        assert len(self.agent._dedupe_search_results(raw)) == 1

    def test_keeps_distinct_urls(self):
        raw = [{"href": "https://a.com"}, {"href": "https://b.com"}]
        assert len(self.agent._dedupe_search_results(raw)) == 2

    def test_skips_items_without_href(self):
        raw = [{"title": "no href"}, {"href": "https://ok.com"}]
        assert len(self.agent._dedupe_search_results(raw)) == 1

    def test_normalize_url_removes_fragment(self):
        url = self.agent._normalize_url("https://example.com/page#section")
        assert "#" not in url

    def test_normalize_url_sorts_query_params(self):
        url1 = self.agent._normalize_url("https://x.com/?b=2&a=1")
        url2 = self.agent._normalize_url("https://x.com/?a=1&b=2")
        assert url1 == url2

    def test_normalize_url_invalid_returns_as_is(self):
        bad = "not a real url %%"
        assert self.agent._normalize_url(bad) == bad


# ═══════════════════════════════════════════════════════════════════════════
# 12. _split_to_passages and _build_passages_from_results
# ═══════════════════════════════════════════════════════════════════════════


class TestPassageBuilding:
    def setup_method(self):
        self.agent = make_agent()

    def test_short_text_single_passage(self):
        text = "Short content."
        passages = self.agent._split_to_passages(text, min_chars=5, max_chars=500)
        assert passages == ["Short content."]

    def test_empty_text_returns_empty(self):
        assert self.agent._split_to_passages("") == []
        assert self.agent._split_to_passages("   ") == []

    def test_long_text_splits_into_chunks(self):
        text = "This is a test sentence. " * 50  # ~1250 chars
        passages = self.agent._split_to_passages(text, min_chars=10, max_chars=200)
        assert len(passages) > 1
        for p in passages:
            assert len(p) <= 200

    def test_build_passages_required_fields(self):
        results = [
            {
                "url": "https://a.com",
                "raw_content": "Alpha content. " * 40,
                "metadata": {"publish_date": "2024-01-01", "author": "X"},
                "source_engine": "google",
            }
        ]
        passages = self.agent._build_passages_from_results(results)
        assert len(passages) >= 1
        assert all("content" in p and "source_url" in p for p in passages)

    def test_build_passages_skips_empty_content(self):
        results = [
            {"url": "https://b.com", "raw_content": "", "metadata": {}, "source_engine": "g"}
        ]
        assert self.agent._build_passages_from_results(results) == []


# ═══════════════════════════════════════════════════════════════════════════
# 13. _resolve_iterations
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveIterations:
    def setup_method(self):
        self.agent = make_agent()

    def test_none_returns_one(self):
        assert self.agent._resolve_iterations(None) == 1

    def test_satisfied_returns_one(self):
        assert self.agent._resolve_iterations({"is_satisfied": True, "confidence_score": 0.5}) == 1

    def test_no_confidence_returns_one(self):
        assert self.agent._resolve_iterations({"is_satisfied": False}) == 1

    def test_low_confidence_returns_three(self):
        assert self.agent._resolve_iterations({"is_satisfied": False, "confidence_score": 0.5}) == 3

    def test_medium_confidence_returns_two(self):
        assert self.agent._resolve_iterations({"is_satisfied": False, "confidence_score": 0.75}) == 2

    def test_invalid_confidence_type_returns_one(self):
        assert self.agent._resolve_iterations({"is_satisfied": False, "confidence_score": "bad"}) == 1

    def test_non_dict_feedback_returns_one(self):
        assert self.agent._resolve_iterations("not a dict") == 1


# ═══════════════════════════════════════════════════════════════════════════
# 14. _validate_and_filter_targets
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateAndFilterTargets:
    def setup_method(self):
        self.agent = make_agent()

    def _ctx(self, description="AI employment impact on job market", key_points=None):
        return {
            "description": description,
            "key_points": key_points or ["job displacement", "wage gap"],
        }

    def test_good_target_is_kept(self):
        result = self.agent._validate_and_filter_targets(
            source_query="AI employment",
            candidate_targets=["AI employment job displacement in 2024"],
            research_context=self._ctx(),
        )
        assert len(result["kept"]) >= 1

    def test_too_short_discarded(self):
        result = self.agent._validate_and_filter_targets(
            source_query="AI employment",
            candidate_targets=["AI"],  # < TARGET_MIN_CHARS = 8
            research_context=self._ctx(),
        )
        reasons = [r for d in result["discarded"] for r in d["reasons"]]
        assert "length_out_of_range" in reasons

    def test_too_long_discarded(self):
        long = "x " * 100  # well over TARGET_MAX_CHARS = 180
        result = self.agent._validate_and_filter_targets(
            source_query="AI employment",
            candidate_targets=[long],
            research_context=self._ctx(),
        )
        reasons = [r for d in result["discarded"] for r in d["reasons"]]
        assert "length_out_of_range" in reasons

    def test_near_duplicate_deduplicated(self):
        target = "AI employment job displacement impact on workers today"
        result = self.agent._validate_and_filter_targets(
            source_query="AI employment",
            candidate_targets=[target, target],
            research_context=self._ctx(),
        )
        # At least one copy must be in discarded or kept count <= 1
        kept_count = len(result["kept"])
        discarded_targets = [d["target"] for d in result["discarded"]]
        assert kept_count == 1 or target in discarded_targets

    def test_coverage_totals_consistent(self):
        targets = [
            "AI employment job displacement impact",
            "wage gap skill mismatch employment market",
        ]
        result = self.agent._validate_and_filter_targets(
            source_query="AI employment",
            candidate_targets=targets,
            research_context=self._ctx(),
        )
        cov = result["coverage"]
        assert cov["kept_total"] + cov["discarded_total"] == cov["candidate_total"]

    def test_empty_candidates_returns_empty(self):
        result = self.agent._validate_and_filter_targets(
            source_query="AI employment",
            candidate_targets=[],
            research_context=self._ctx(),
        )
        assert result["kept"] == []
        assert result["coverage"]["candidate_total"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# 15. _build_coverage_snapshot
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildCoverageSnapshot:
    def setup_method(self):
        self.agent = make_agent()

    def _qtm(self, queries):
        return [{"source_query": q} for q in queries]

    def test_full_query_coverage(self):
        queries = ["AI jobs", "wage gap"]
        search_log = [
            {
                "source_query": "AI jobs",
                "target": "t1",
                "top_10_passages": [make_passage("AI jobs content")],
            },
            {
                "source_query": "wage gap",
                "target": "t2",
                "top_10_passages": [make_passage("wage gap content")],
            },
        ]
        snap = self.agent._build_coverage_snapshot(
            self._qtm(queries), search_log, {"key_points": []}
        )
        assert snap["query_coverage"] == pytest.approx(1.0)
        assert snap["query_covered"] == 2

    def test_partial_query_coverage(self):
        search_log = [
            {
                "source_query": "Q1",
                "target": "t1",
                "top_10_passages": [make_passage("Q1 content")],
            }
        ]
        snap = self.agent._build_coverage_snapshot(
            self._qtm(["Q1", "Q2"]), search_log, {"key_points": []}
        )
        assert snap["query_coverage"] == pytest.approx(0.5)
        assert "Q2" in snap["uncovered_queries"]

    def test_no_key_points_keypoint_coverage_is_one(self):
        snap = self.agent._build_coverage_snapshot(
            self._qtm(["Q1"]),
            [{"source_query": "Q1", "target": "t", "top_10_passages": [make_passage("c")]}],
            {"key_points": []},
        )
        assert snap["keypoint_coverage"] == pytest.approx(1.0)

    def test_uncovered_key_point_tracked(self):
        snap = self.agent._build_coverage_snapshot(
            self._qtm(["q"]),
            [
                {
                    "source_query": "q",
                    "target": "wage polarization topic",
                    "top_10_passages": [make_passage("wage polarization analysis")],
                }
            ],
            {"key_points": ["wage polarization", "automation robots"]},
        )
        assert "automation robots" in snap["uncovered_key_points"]

    def test_empty_snapshot(self):
        snap = self.agent._build_coverage_snapshot([], [], {"key_points": []})
        assert snap["query_coverage"] == pytest.approx(0.0)
        assert snap["section_coverage"] == pytest.approx(0.0)
        assert snap["query_total"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# 16. _fallback_rank
# ═══════════════════════════════════════════════════════════════════════════


class TestFallbackRank:
    def setup_method(self):
        self.agent = make_agent()

    def test_returns_top_k(self):
        passages = [make_passage(f"Content number {i} with extra text") for i in range(10)]
        relevance = [float(i) for i in range(10)]
        result = self.agent._fallback_rank(passages, relevance, [], top_k=3)
        assert len(result) == 3

    def test_highest_relevance_first(self):
        passages = [
            make_passage("Low relevance passage text here", url="u0"),
            make_passage("High relevance passage text here", url="u1"),
            make_passage("Medium relevance passage text", url="u2"),
        ]
        relevance = [0.1, 0.9, 0.5]
        result = self.agent._fallback_rank(passages, relevance, [], top_k=1)
        assert result[0]["source_url"] == "u1"

    def test_chosen_indices_come_first(self):
        passages = [
            make_passage("Already chosen passage unique", url="u0"),
            make_passage("High score but not chosen", url="u1"),
        ]
        relevance = [0.1, 0.9]
        result = self.agent._fallback_rank(passages, relevance, chosen_indices=[0], top_k=2)
        assert result[0]["source_url"] == "u0"

    def test_deduplicates_identical_content(self):
        passages = [
            make_passage("Exact same content repeated", url="u0"),
            make_passage("Exact same content repeated", url="u1"),
            make_passage("Different unique content here", url="u2"),
        ]
        relevance = [0.9, 0.8, 0.7]
        result = self.agent._fallback_rank(passages, relevance, [], top_k=3)
        contents = [r["content"] for r in result]
        assert contents.count("Exact same content repeated") == 1


# ═══════════════════════════════════════════════════════════════════════════
# 17. _env_int and _env_truthy
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvHelpers:
    _KEY_INT = "TEST_SCRAPING_ENV_INT"
    _KEY_BOOL = "TEST_SCRAPING_ENV_BOOL"

    def setup_method(self):
        self.agent = make_agent()
        os.environ.pop(self._KEY_INT, None)
        os.environ.pop(self._KEY_BOOL, None)

    def teardown_method(self):
        os.environ.pop(self._KEY_INT, None)
        os.environ.pop(self._KEY_BOOL, None)

    def test_env_int_default_when_unset(self):
        result = self.agent._env_int(self._KEY_INT, default=5, min_value=1, max_value=10)
        assert result == 5

    def test_env_int_clamps_to_min(self):
        os.environ[self._KEY_INT] = "0"
        assert self.agent._env_int(self._KEY_INT, default=5, min_value=2, max_value=10) == 2

    def test_env_int_clamps_to_max(self):
        os.environ[self._KEY_INT] = "100"
        assert self.agent._env_int(self._KEY_INT, default=5, min_value=1, max_value=10) == 10

    def test_env_int_invalid_uses_default(self):
        os.environ[self._KEY_INT] = "abc"
        assert self.agent._env_int(self._KEY_INT, default=7, min_value=1, max_value=10) == 7

    def test_env_truthy_true_values(self):
        for val in ("1", "true", "True", "yes", "YES", "y", "on"):
            os.environ[self._KEY_BOOL] = val
            assert self.agent._env_truthy(self._KEY_BOOL) is True

    def test_env_truthy_false_values(self):
        for val in ("0", "false", "no", "off"):
            os.environ[self._KEY_BOOL] = val
            assert self.agent._env_truthy(self._KEY_BOOL) is False

    def test_env_truthy_unset_returns_default(self):
        assert self.agent._env_truthy(self._KEY_BOOL, default=True) is True
        assert self.agent._env_truthy(self._KEY_BOOL, default=False) is False


# ═══════════════════════════════════════════════════════════════════════════
# 18. run_depth_scraping — integration with mocked IO
# ═══════════════════════════════════════════════════════════════════════════


def _draft_state(queries=None, topic="AI employment test", iteration_index=None):
    state = {
        "task": {
            "scraping_max_iterations": 1,
            "scraping_max_search_results": 3,
            "application_name": "test_app",
        },
        "topic": topic,
        "research_context": {
            "description": "AI impact on employment",
            "key_points": ["job displacement"],
            "research_queries": queries if queries is not None else ["AI employment 2024"],
        },
    }
    if iteration_index is not None:
        state["iteration_index"] = iteration_index
    return state


@pytest.mark.asyncio
class TestRunDepthScrap:
    def _make_agent_with_mocks(self, decompose_return=None, collect_return=None):
        agent = make_agent()
        agent._decompose_query_to_targets = AsyncMock(
            return_value=decompose_return or ["AI employment job displacement 2024"]
        )
        agent._collect_results_for_target = AsyncMock(
            return_value=collect_return or ([], [])
        )
        agent._select_top_passages_with_mmr = AsyncMock(return_value=[])
        return agent

    async def test_returns_required_keys(self):
        agent = self._make_agent_with_mocks()
        result = await agent.run_depth_scraping(_draft_state())
        assert "draft" in result
        assert "scraping_packet" in result
        assert "iteration_index" in result

    async def test_empty_queries_uses_topic_fallback(self):
        agent = self._make_agent_with_mocks()
        result = await agent.run_depth_scraping(_draft_state(queries=[]))
        assert "draft" in result

    async def test_decompose_returns_empty_triggers_fallback(self):
        agent = self._make_agent_with_mocks(decompose_return=[])
        result = await agent.run_depth_scraping(_draft_state())
        assert "scraping_packet" in result

    async def test_single_iteration_override(self):
        """An explicit iteration override should propagate to the result packet."""
        agent = self._make_agent_with_mocks()
        result = await agent.run_depth_scraping(_draft_state(iteration_index=2))
        assert result["iteration_index"] == 2

    async def test_retry_iteration_preserves_previous_passages(self):
        """Retry iterations should keep prior evidence when the new round returns nothing useful."""
        agent = self._make_agent_with_mocks(collect_return=([], ["duckduckgo"]))
        previous_packet = {
            "iteration_index": 1,
            "model_level": "Level_1_Base",
            "active_engines": ["tavily"],
            "search_log": [
                {
                    "source_query": "AI employment 2024",
                    "target": "AI employment 2024",
                    "extra_hints_applied": "",
                    "top_10_passages": [make_passage("AI employment 2024 objective data point")],
                }
            ],
            "query_target_map": [
                {
                    "source_query": "AI employment 2024",
                    "planning_incomplete": False,
                    "targets_generated": 1,
                    "targets_kept": 1,
                    "targets_discarded": 0,
                    "candidate_targets": ["AI employment 2024"],
                    "kept_targets": ["AI employment 2024"],
                    "discarded_targets": [],
                    "fallback_used": False,
                    "unresolved": False,
                    "validation": {},
                }
            ],
            "coverage_snapshot": {},
            "fallback_used": False,
        }
        state = _draft_state(iteration_index=2)
        state["scraping_packet"] = previous_packet

        result = await agent.run_depth_scraping(state)
        packet = result["scraping_packet"]

        assert packet["iteration_index"] == 2
        assert packet["coverage_snapshot"]["query_covered"] == 1
        assert packet["coverage_snapshot"]["query_total"] == 1
        assert packet["search_log"][0]["top_10_passages"][0]["content"] == "AI employment 2024 objective data point"
        assert packet["active_engines"] == ["duckduckgo", "tavily"]

    async def test_merge_incremental_packet_recomputes_coverage_across_rounds(self):
        """Merged retry packets should compute coverage from the union of old and new queries."""
        agent = self._make_agent_with_mocks()
        previous_packet = {
            "iteration_index": 1,
            "model_level": "Level_1_Base",
            "active_engines": ["tavily"],
            "search_log": [
                {
                    "source_query": "AI employment 2024",
                    "target": "AI employment 2024",
                    "extra_hints_applied": "",
                    "top_10_passages": [make_passage("AI employment 2024 objective data point")],
                }
            ],
            "query_target_map": [
                {
                    "source_query": "AI employment 2024",
                    "planning_incomplete": False,
                    "targets_generated": 1,
                    "targets_kept": 1,
                    "targets_discarded": 0,
                    "candidate_targets": ["AI employment 2024"],
                    "kept_targets": ["AI employment 2024"],
                    "discarded_targets": [],
                    "fallback_used": False,
                    "unresolved": False,
                    "validation": {},
                }
            ],
            "coverage_snapshot": {},
            "fallback_used": False,
        }
        new_packet = {
            "iteration_index": 2,
            "model_level": "Level_2_Pro",
            "active_engines": ["duckduckgo"],
            "search_log": [
                {
                    "source_query": "AI wages 2024",
                    "target": "AI wages 2024",
                    "extra_hints_applied": "focus on wage impacts",
                    "top_10_passages": [make_passage("AI wages 2024 objective wage data point")],
                }
            ],
            "query_target_map": [
                {
                    "source_query": "AI wages 2024",
                    "planning_incomplete": False,
                    "targets_generated": 1,
                    "targets_kept": 1,
                    "targets_discarded": 0,
                    "candidate_targets": ["AI wages 2024"],
                    "kept_targets": ["AI wages 2024"],
                    "discarded_targets": [],
                    "fallback_used": False,
                    "unresolved": False,
                    "validation": {},
                }
            ],
            "coverage_snapshot": {},
            "fallback_used": False,
        }

        merged = await agent._merge_incremental_packet(
            previous_packet=previous_packet,
            new_packet=new_packet,
            research_context={"key_points": []},
        )

        assert merged["iteration_index"] == 2
        assert merged["coverage_snapshot"]["query_total"] == 2
        assert merged["coverage_snapshot"]["query_covered"] == 2
        assert merged["coverage_snapshot"]["section_coverage"] == 1.0
        assert {row["source_query"] for row in merged["query_target_map"]} == {
            "AI employment 2024",
            "AI wages 2024",
        }

    async def test_decompose_exception_does_not_crash(self):
        """503-style LLM failure is caught inside _decompose_query_to_targets (not at the call site)."""
        agent = self._make_agent_with_mocks()
        agent._collect_results_for_target = AsyncMock(return_value=([], []))
        agent._select_top_passages_with_mmr = AsyncMock(return_value=[])
        # Patch call_model (used inside _decompose_query_to_targets) to raise
        with patch("multi_agents.agents.scraping.call_model", side_effect=Exception("HTTP 503")):
            result = await agent.run_depth_scraping(_draft_state())
        assert "scraping_packet" in result

    async def test_state_controller_tier_reset_after_loop(self):
        agent = self._make_agent_with_mocks()
        sc_mock = MagicMock()
        sc_mock.get_current_model.return_value = "stub-model"
        agent.state_controller = sc_mock
        await agent.run_depth_scraping(_draft_state())
        sc_mock.set_tier.assert_called_with("scraping", 3)

    async def test_scraping_packet_has_search_log_and_query_target_map(self):
        agent = self._make_agent_with_mocks()
        result = await agent.run_depth_scraping(_draft_state())
        packet = result["scraping_packet"]
        assert "search_log" in packet
        assert "query_target_map" in packet
        assert "coverage_snapshot" in packet

    async def test_draft_text_contains_topic(self):
        agent = self._make_agent_with_mocks()
        result = await agent.run_depth_scraping(_draft_state(topic="test topic name"))
        draft = result["draft"]
        assert any("test topic name" in v for v in draft.values())
