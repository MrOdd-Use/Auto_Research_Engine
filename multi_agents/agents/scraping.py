import asyncio
import json
import logging
import os
import re
import time
from math import sqrt
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import json_repair

from gpt_researcher.actions.retriever import get_retriever
from gpt_researcher.actions.web_scraping import scrape_urls
from gpt_researcher.config.config import Config
from gpt_researcher.memory.embeddings import Memory
from gpt_researcher.utils.workers import WorkerPool

from multi_agents.route_agent import build_route_context
from .state_controller import StateController
from .utils.llms import call_model
from .utils.views import print_agent_output


class ScrapingAgent:
    """Adaptive Scraping Agent: multi-iteration search + dedup + MMR rerank."""

    ROUND_TO_TIER = {1: 3, 2: 2, 3: 1}
    ROUND_TO_LEVEL = {
        1: "Level_1_Base",
        2: "Level_2_Pro",
        3: "Level_3_Max",
    }
    # Domains known to block headless scrapers; skip BS scraping and rely on
    # the body text already returned by the search engine (Tavily raw_content).
    ANTI_SCRAPE_DOMAINS = {
        "reddit.com",
        "toutiao.com",
        "h5.ifeng.com",
        "ifeng.com",
        "wallstreetcn.com",
        "post.smzdm.com",
        "vzkoo.com",
        "m.caijing.com.cn",
        "caijing.com.cn",
        "k.sina.com.cn",
        "allpku.com",
        "ecommercefastlane.com",
    }
    DOMAIN_TO_ENGINES = {
        "tech": ["arxiv", "semantic_scholar", "tavily", "duckduckgo"],
        "medical": ["pubmed_central", "tavily", "duckduckgo"],
        "finance": ["tavily", "duckduckgo"],
        "general": ["tavily", "duckduckgo"],
    }
    VERTICAL_ENGINES = {"arxiv", "semantic_scholar", "pubmed_central"}
    SEARCH_FALLBACK_ENGINES = ("tavily", "duckduckgo")
    MODEL_LEVEL_FALLBACK = {
        1: "gpt-4o-mini",
        2: "gpt-4o",
        3: "o1-preview",
    }
    TARGET_MIN_CHARS = 8
    TARGET_MAX_CHARS = 180
    TARGET_SIMILARITY_MIN = 0.25
    TARGET_OUT_OF_CONTEXT_MAX_RATIO = 0.60
    TARGET_NEAR_DUPLICATE_SIM = 0.92
    COVERAGE_THRESHOLD = 0.70
    VALIDATION_STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "with",
        "is", "are", "was", "were", "be", "been", "being", "as", "that", "this", "these",
        "those", "from", "into", "about", "vs", "versus", "than", "how", "what", "why",
        "who", "when", "where", "which", "it", "its", "their", "our", "your",
        "的", "了", "和", "与", "及", "在", "对", "就", "也", "或", "是", "有", "及其",
    }

    def __init__(self, websocket=None, stream_output=None, tone=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.tone = tone
        self.headers = headers or {}
        self.logger = logging.getLogger(__name__)
        state_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_state.json")
        self.state_controller = StateController(state_file=state_file)
        self._mmr_embeddings = None
        self._mmr_embeddings_error: Optional[str] = None
        self._use_embedding_mmr = self._env_truthy("SCRAPING_MMR_USE_EMBEDDINGS", default=True)
        self._min_search_targets = self._env_int("SCRAPING_MIN_SEARCH_TARGETS", default=2, min_value=1, max_value=8)
        self._max_search_targets = self._env_int("SCRAPING_MAX_SEARCH_TARGETS", default=4, min_value=1, max_value=12)
        if self._max_search_targets < self._min_search_targets:
            self._max_search_targets = self._min_search_targets

    async def run_depth_scraping(self, draft_state: dict) -> dict:
        """Run one or more scraping iterations and preserve prior evidence on retries."""
        task = draft_state.get("task") or {}
        topic = str(draft_state.get("topic") or "").strip()
        if not topic:
            topic = "Main Research Focus"

        research_context = draft_state.get("research_context") or {}
        audit_feedback = draft_state.get("audit_feedback")
        extra_hints = draft_state.get("extra_hints")
        iteration_override = draft_state.get("iteration_index")
        query_domains = task.get("query_domains") or []
        max_results = self._task_int(task, "scraping_max_search_results", "max_search_results_per_query", default=7)
        max_iterations = self._task_int(task, "scraping_max_iterations", default=3)

        use_single_iteration = iteration_override is not None
        if use_single_iteration:
            single_iteration = self._normalize_iteration(iteration_override)
            iteration_plan = [single_iteration]
        else:
            planned_iterations = min(self._resolve_iterations(audit_feedback), max_iterations)
            iteration_plan = list(range(1, planned_iterations + 1))
        packets: List[dict] = []
        cumulative_packet = None
        if use_single_iteration:
            single_iteration = iteration_plan[0]
            prior_packet = draft_state.get("scraping_packet") or {}
            if single_iteration > 1 and self._packet_has_mergeable_results(prior_packet):
                cumulative_packet = prior_packet

        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "scraping_start",
                f"Running ASA Scraping loop for topic: {topic}",
                self.websocket,
            )
        else:
            print_agent_output(f"Running ASA Scraping loop for topic: {topic}", agent="SCRAPING")

        try:
            for iteration in iteration_plan:
                tier_idx = self.ROUND_TO_TIER.get(iteration, 3)
                self.state_controller.set_tier("scraping", tier_idx)
                extra_hints_applied = self._merge_extra_hints(extra_hints, audit_feedback)
                source_queries = self._normalize_research_queries(
                    research_context.get("research_queries")
                )
                uncovered_retry_queries = self._normalize_research_queries(
                    (audit_feedback or {}).get("uncovered_queries")
                )
                if uncovered_retry_queries:
                    source_queries = uncovered_retry_queries

                planning_incomplete = bool(research_context.get("planning_incomplete")) or not source_queries
                if not source_queries:
                    fallback_source_query = topic or str(research_context.get("description") or "").strip()
                    source_queries = [fallback_source_query] if fallback_source_query else []

                target_jobs: List[dict] = []
                query_target_map: List[dict] = []
                fallback_used_any = False

                for source_query in source_queries:
                    # source_queries are already concrete search targets produced by
                    # the per-keypoint search_queries in the planner. Skip LLM
                    # decomposition and use the query directly as the search target.
                    kept_targets = [source_query]
                    fallback_used = False

                    unresolved = False
                    query_target_map.append(
                        {
                            "source_query": source_query,
                            "planning_incomplete": planning_incomplete,
                            "targets_generated": 1,
                            "targets_kept": 1,
                            "targets_discarded": 0,
                            "candidate_targets": [source_query],
                            "kept_targets": kept_targets,
                            "discarded_targets": [],
                            "fallback_used": fallback_used,
                            "unresolved": unresolved,
                            "validation": {},
                        }
                    )
                    for target in kept_targets:
                        target_jobs.append({"source_query": source_query, "target": target})

                engines = self._select_engines(iteration, topic)
                search_log = []
                active_engines = set()

                for job in target_jobs:
                    source_query = job["source_query"]
                    target = job["target"]
                    target_results, used_engines = await self._collect_results_for_target(
                        target=target,
                        engines=engines,
                        query_domains=query_domains,
                        max_results=max_results,
                    )
                    active_engines.update(used_engines)
                    passages = self._build_passages_from_results(target_results)
                    top_passages = await self._select_top_passages_with_mmr(
                        query=target,
                        passages=passages,
                        top_k=10,
                        time_budget_s=5.0,
                    )
                    search_log.append(
                        {
                            "source_query": source_query,
                            "target": target,
                            "extra_hints_applied": extra_hints_applied,
                            "top_10_passages": top_passages,
                        }
                    )

                coverage_snapshot = self._build_coverage_snapshot(
                    query_target_map=query_target_map,
                    search_log=search_log,
                    research_context=research_context,
                )

                packet = {
                    "iteration_index": iteration,
                    "model_level": self.ROUND_TO_LEVEL.get(iteration, "Level_1_Base"),
                    "active_engines": sorted(active_engines),
                    "search_log": search_log,
                    "query_target_map": query_target_map,
                    "coverage_snapshot": coverage_snapshot,
                    "fallback_used": fallback_used_any,
                }
                if cumulative_packet is not None:
                    packet = await self._merge_incremental_packet(
                        previous_packet=cumulative_packet,
                        new_packet=packet,
                        research_context=research_context,
                    )
                packets.append(packet)
                cumulative_packet = packet
                self.logger.info(
                    "scraping_iteration_complete",
                    extra={
                        "iteration": iteration,
                        "tier": tier_idx,
                        "engines": packet["active_engines"],
                        "passage_count": sum(len(x["top_10_passages"]) for x in search_log),
                    },
                )
        finally:
            # Reset Scraping to default base tier after loop
            self.state_controller.set_tier("scraping", 3)

        final_packet = packets[-1] if packets else self._empty_packet()
        draft_text = self._build_compatible_draft(topic, final_packet)
        return {
            "draft": {topic: draft_text},
            "scraping_packet": final_packet,
            "iteration_index": final_packet.get("iteration_index", 1),
        }

    def _resolve_iterations(self, audit_feedback: dict | None) -> int:
        if not isinstance(audit_feedback, dict):
            return 1
        if audit_feedback.get("is_satisfied") is not False:
            return 1
        if audit_feedback.get("confidence_score") is None:
            return 1
        try:
            score = float(audit_feedback.get("confidence_score"))
        except (TypeError, ValueError):
            return 1
        return 3 if score < 0.7 else 2

    def _resolve_model_for_iteration(self, task: dict, iteration: int) -> str:
        explicit = self._task_value(task, f"scraping_level_{iteration}_model")
        if explicit:
            return str(explicit)
        fallback = self.MODEL_LEVEL_FALLBACK.get(iteration, "gpt-4o-mini")
        return self.state_controller.get_current_model("scraping", fallback_model=fallback)

    def _normalize_research_queries(self, raw_queries: Any) -> List[str]:
        if not isinstance(raw_queries, list):
            return []
        normalized = []
        seen = set()
        for item in raw_queries:
            query = re.sub(r"\s+", " ", str(item or "").strip())
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(query)
        return normalized

    def _normalize_iteration(self, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 1
        return min(max(parsed, 1), 3)

    async def _decompose_query_to_targets(
        self,
        source_query: str,
        research_context: dict,
        extra_hints: str,
        model_name: str,
        task_payload: dict | None = None,
        draft_state: dict | None = None,
    ) -> List[str]:
        description = str(research_context.get("description") or "")
        key_points = research_context.get("key_points") or []
        key_points_text = ", ".join(str(x) for x in key_points if str(x).strip())
        min_targets = self._min_search_targets
        max_targets = self._max_search_targets
        prompt = [
            {
                "role": "system",
                "content": (
                    "You decompose one source query into smaller search targets. "
                    "Do NOT introduce new topics outside the source query and context. "
                    f"Return JSON list only with {min_targets}-{max_targets} concise strings."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Source query: {source_query}\n"
                    f"Description: {description}\n"
                    f"Key points: {key_points_text}\n"
                    f"Extra hints: {extra_hints}\n"
                    f"Constraints: Generate {min_targets}-{max_targets} independent, searchable targets. "
                    "Keep targets tightly scoped to the source query and key points."
                ),
            },
        ]
        try:
            route_context = build_route_context(
                application_name=str((task_payload or {}).get("application_name") or "auto_research_engine"),
                shared_agent_class="scrape_agent",
                agent_role="scraping",
                stage_name="target_decomposition",
                system_prompt="You decompose one source query into smaller search targets.",
                task=source_query,
                state=draft_state,
                task_payload=task_payload or {},
            )
            response = await call_model(prompt=prompt, model=model_name, route_context=route_context)
        except Exception as exc:
            self.logger.warning(f"Failed to decompose source query '{source_query}': {exc}")
            return []

        parsed = self._parse_targets(response)
        return parsed[:max_targets]

    def _validate_and_filter_targets(
        self,
        source_query: str,
        candidate_targets: List[str],
        research_context: dict,
    ) -> dict:
        cleaned = self._clean_targets(candidate_targets or [])
        description = str(research_context.get("description") or "")
        key_points = [str(x or "") for x in (research_context.get("key_points") or []) if str(x or "").strip()]
        context_text = " ".join([source_query, description, *key_points]).strip()
        context_terms = set(self._tokenize_validation(context_text))
        anchor_terms = set(self._tokenize_validation(source_query))
        context_vec = self._text_to_vector(context_text)

        kept_payload: List[dict] = []
        discarded: List[dict] = []
        for target in cleaned:
            reasons = []
            length = len(target)
            if length < self.TARGET_MIN_CHARS or length > self.TARGET_MAX_CHARS:
                reasons.append("length_out_of_range")

            target_terms = set(self._tokenize_validation(target))
            anchor_hits = len(anchor_terms & target_terms)
            if anchor_terms and anchor_hits < 1:
                reasons.append("missing_anchor_terms")

            target_vec = self._text_to_vector(target)
            similarity = self._cosine_similarity(context_vec, target_vec)
            if similarity < self.TARGET_SIMILARITY_MIN:
                reasons.append("low_semantic_similarity")

            if target_terms:
                new_terms = [term for term in target_terms if term not in context_terms]
                out_of_context_ratio = len(new_terms) / max(len(target_terms), 1)
            else:
                out_of_context_ratio = 1.0
            if out_of_context_ratio > self.TARGET_OUT_OF_CONTEXT_MAX_RATIO:
                reasons.append("too_many_out_of_context_terms")

            payload = {
                "target": target,
                "reasons": reasons,
                "scores": {
                    "anchor_hits": anchor_hits,
                    "semantic_similarity": round(similarity, 4),
                    "out_of_context_ratio": round(out_of_context_ratio, 4),
                },
            }
            if reasons:
                discarded.append(payload)
            else:
                kept_payload.append(payload)

        deduped_kept: List[str] = []
        deduped_vecs: List[Counter] = []
        for item in kept_payload:
            target = item["target"]
            target_vec = self._text_to_vector(target)
            duplicate = False
            for existing_vec in deduped_vecs:
                if self._cosine_similarity(existing_vec, target_vec) > self.TARGET_NEAR_DUPLICATE_SIM:
                    duplicate = True
                    break
            if duplicate:
                discarded.append(
                    {
                        "target": target,
                        "reasons": ["near_duplicate_target"],
                        "scores": item["scores"],
                    }
                )
                continue
            deduped_kept.append(target)
            deduped_vecs.append(target_vec)
            if len(deduped_kept) >= self._max_search_targets:
                break

        coverage = {
            "candidate_total": len(cleaned),
            "kept_total": len(deduped_kept),
            "discarded_total": len(discarded),
        }
        return {
            "kept": deduped_kept,
            "discarded": discarded,
            "coverage": coverage,
        }

    def _tokenize_validation(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", (text or "").lower())
        normalized = []
        for token in tokens:
            if token in self.VALIDATION_STOPWORDS:
                continue
            if len(token) == 1 and re.match(r"[a-z]", token):
                continue
            normalized.append(token)
        return normalized

    def _fallback_targets_from_context(self, source_query: str, research_context: dict) -> List[str]:
        description = str(research_context.get("description") or "").strip()
        key_points = [str(x or "").strip() for x in (research_context.get("key_points") or []) if str(x or "").strip()]
        targets: List[str] = []
        if description:
            targets.append(f"{source_query} {description}")
        for point in key_points:
            targets.append(f"{source_query} {point}")
            if len(targets) >= self._max_search_targets:
                break
        if not targets:
            targets.append(source_query)
        return self._clean_targets(targets)

    def _build_coverage_snapshot(
        self,
        query_target_map: List[dict],
        search_log: List[dict],
        research_context: dict,
    ) -> dict:
        total_queries = len(query_target_map)
        covered_queries_set = set()
        for row in search_log:
            source_query = str(row.get("source_query") or "").strip()
            passages = row.get("top_10_passages") or []
            if source_query and passages:
                covered_queries_set.add(source_query)

        covered_queries = sum(
            1 for item in query_target_map if str(item.get("source_query") or "").strip() in covered_queries_set
        )
        query_coverage = (covered_queries / total_queries) if total_queries else 0.0

        key_points = [str(x or "").strip() for x in (research_context.get("key_points") or []) if str(x or "").strip()]
        content_blob = []
        for row in search_log:
            content_blob.append(str(row.get("target") or ""))
            for passage in row.get("top_10_passages") or []:
                content_blob.append(str(passage.get("content") or ""))
        corpus_terms = set(self._tokenize_validation(" ".join(content_blob)))

        if not key_points:
            keypoint_coverage = 1.0
            covered_key_points = []
            uncovered_key_points = []
        else:
            covered_key_points = []
            uncovered_key_points = []
            for point in key_points:
                point_terms = set(self._tokenize_validation(point))
                if not point_terms:
                    covered_key_points.append(point)
                    continue
                if point_terms & corpus_terms:
                    covered_key_points.append(point)
                else:
                    uncovered_key_points.append(point)
            keypoint_coverage = len(covered_key_points) / len(key_points)

        section_coverage = min(query_coverage, keypoint_coverage)
        uncovered_queries = [
            str(item.get("source_query") or "").strip()
            for item in query_target_map
            if str(item.get("source_query") or "").strip() not in covered_queries_set
        ]
        return {
            "query_coverage": round(query_coverage, 4),
            "keypoint_coverage": round(keypoint_coverage, 4),
            "section_coverage": round(section_coverage, 4),
            "query_total": total_queries,
            "query_covered": covered_queries,
            "uncovered_queries": [q for q in uncovered_queries if q],
            "keypoint_total": len(key_points),
            "keypoint_covered": len(covered_key_points),
            "uncovered_key_points": uncovered_key_points,
            "coverage_threshold": self.COVERAGE_THRESHOLD,
        }

    def _packet_has_mergeable_results(self, packet: dict | None) -> bool:
        """Return whether a scraping packet contains reusable research output."""
        if not isinstance(packet, dict):
            return False
        return bool(packet.get("search_log") or packet.get("query_target_map") or packet.get("active_engines"))

    async def _merge_incremental_packet(
        self,
        previous_packet: dict | None,
        new_packet: dict | None,
        research_context: dict,
    ) -> dict:
        """Merge retry output into the previous scraping packet and recompute coverage."""
        previous_packet = previous_packet or {}
        new_packet = new_packet or {}
        merged_query_target_map = self._merge_query_target_map(
            previous_packet.get("query_target_map") or [],
            new_packet.get("query_target_map") or [],
        )
        merged_search_log = await self._merge_search_log(
            previous_packet.get("search_log") or [],
            new_packet.get("search_log") or [],
        )
        iteration_index = self._normalize_iteration(
            new_packet.get("iteration_index") or previous_packet.get("iteration_index")
        )
        model_level = str(
            new_packet.get("model_level")
            or previous_packet.get("model_level")
            or self.ROUND_TO_LEVEL.get(iteration_index, "Level_1_Base")
        )
        active_engines = sorted(
            {
                str(engine).strip()
                for engine in [*(previous_packet.get("active_engines") or []), *(new_packet.get("active_engines") or [])]
                if str(engine).strip()
            }
        )
        merged_packet = {
            "iteration_index": iteration_index,
            "model_level": model_level,
            "active_engines": active_engines,
            "search_log": merged_search_log,
            "query_target_map": merged_query_target_map,
            "coverage_snapshot": self._build_coverage_snapshot(
                query_target_map=merged_query_target_map,
                search_log=merged_search_log,
                research_context=research_context,
            ),
            "fallback_used": bool(previous_packet.get("fallback_used") or new_packet.get("fallback_used")),
        }
        return merged_packet

    async def _merge_search_log(self, previous_log: List[dict], new_log: List[dict]) -> List[dict]:
        """Merge search-log rows by target, preserving prior passages when retries add less."""
        merged_rows: Dict[Tuple[str, str], dict] = {}
        merge_counts: Counter = Counter()
        ordered_keys: List[Tuple[str, str]] = []

        for row in [*previous_log, *new_log]:
            normalized = self._normalize_search_log_row(row)
            if not normalized:
                continue
            row_key = self._search_log_row_key(normalized)
            merge_counts[row_key] += 1
            if row_key not in merged_rows:
                merged_rows[row_key] = normalized
                ordered_keys.append(row_key)
                continue

            merged_rows[row_key]["extra_hints_applied"] = self._merge_hint_text(
                merged_rows[row_key].get("extra_hints_applied"),
                normalized.get("extra_hints_applied"),
            )
            merged_rows[row_key]["top_10_passages"] = self._dedupe_passages(
                [
                    *(merged_rows[row_key].get("top_10_passages") or []),
                    *(normalized.get("top_10_passages") or []),
                ],
                limit=None,
            )

        final_log: List[dict] = []
        for row_key in ordered_keys:
            row = dict(merged_rows[row_key])
            row["top_10_passages"] = await self._finalize_merged_passages(
                target=row.get("target", ""),
                passages=row.get("top_10_passages") or [],
                rerank=merge_counts[row_key] > 1,
            )
            final_log.append(row)
        return final_log

    def _normalize_search_log_row(self, row: Any) -> dict:
        """Normalize a search-log row into a merge-safe shape."""
        if not isinstance(row, dict):
            return {}
        target = str(row.get("target") or "").strip()
        source_query = str(row.get("source_query") or "").strip()
        if not target and not source_query:
            return {}
        return {
            "source_query": source_query,
            "target": target or source_query,
            "extra_hints_applied": str(row.get("extra_hints_applied") or "").strip(),
            "top_10_passages": self._dedupe_passages(row.get("top_10_passages") or [], limit=None),
        }

    def _search_log_row_key(self, row: dict) -> Tuple[str, str]:
        """Build a stable dedupe key for a search-log row."""
        source_query = str(row.get("source_query") or "").strip().lower()
        target = str(row.get("target") or "").strip().lower()
        return (source_query or target, target)

    async def _finalize_merged_passages(
        self,
        target: str,
        passages: List[dict],
        rerank: bool,
    ) -> List[dict]:
        """Deduplicate merged passages and rerank only when a target was seen multiple times."""
        deduped = self._dedupe_passages(passages, limit=None)
        if not deduped:
            return []
        if not rerank:
            return deduped[:10]
        try:
            reranked = await self._select_top_passages_with_mmr(
                query=target,
                passages=deduped,
                top_k=10,
                time_budget_s=5.0,
            )
        except Exception as exc:
            self.logger.warning("Failed to rerank merged passages for target '%s': %s", target, exc)
            return deduped[:10]
        return self._dedupe_passages(reranked or deduped, limit=10)

    def _dedupe_passages(self, passages: List[dict], limit: int | None = 10) -> List[dict]:
        """Deduplicate passages by URL and content while preserving order."""
        unique_passages: List[dict] = []
        seen_keys = set()
        for passage in passages:
            if not isinstance(passage, dict):
                continue
            content = str(passage.get("content") or "").strip()
            source_url = str(passage.get("source_url") or "").strip()
            if not content:
                continue
            dedupe_key = (source_url.lower(), content.lower())
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            unique_passages.append(
                {
                    "content": content,
                    "source_url": source_url,
                    "metadata": dict(passage.get("metadata") or {}),
                }
            )
            if limit is not None and len(unique_passages) >= limit:
                break
        return unique_passages

    def _merge_query_target_map(self, previous_rows: List[dict], new_rows: List[dict]) -> List[dict]:
        """Merge query-target rows by source query so retries extend prior coverage."""
        merged_rows: Dict[str, dict] = {}
        ordered_keys: List[str] = []
        index_seed = 0

        for row in [*previous_rows, *new_rows]:
            normalized = self._normalize_query_target_row(row)
            if not normalized:
                continue
            row_key = self._query_target_row_key(normalized, index_seed)
            index_seed += 1
            if row_key not in merged_rows:
                merged_rows[row_key] = normalized
                ordered_keys.append(row_key)
                continue

            existing = merged_rows[row_key]
            candidate_targets = self._merge_string_list(
                existing.get("candidate_targets") or [],
                normalized.get("candidate_targets") or [],
            )
            kept_targets = self._merge_string_list(
                existing.get("kept_targets") or [],
                normalized.get("kept_targets") or [],
            )
            discarded_targets = self._merge_target_payloads(
                existing.get("discarded_targets") or [],
                normalized.get("discarded_targets") or [],
            )
            validation = dict(existing.get("validation") or {})
            validation.update(normalized.get("validation") or {})
            merged_rows[row_key] = {
                "source_query": existing.get("source_query") or normalized.get("source_query"),
                "planning_incomplete": bool(
                    existing.get("planning_incomplete") or normalized.get("planning_incomplete")
                ),
                "targets_generated": len(candidate_targets),
                "targets_kept": len(kept_targets),
                "targets_discarded": len(discarded_targets),
                "candidate_targets": candidate_targets,
                "kept_targets": kept_targets,
                "discarded_targets": discarded_targets,
                "fallback_used": bool(existing.get("fallback_used") or normalized.get("fallback_used")),
                "unresolved": not bool(kept_targets),
                "validation": validation,
            }

        return [merged_rows[row_key] for row_key in ordered_keys]

    def _normalize_query_target_row(self, row: Any) -> dict:
        """Normalize a query-target row into a consistent merge shape."""
        if not isinstance(row, dict):
            return {}
        source_query = str(row.get("source_query") or "").strip()
        candidate_targets = self._merge_string_list(row.get("candidate_targets") or [])
        kept_targets = self._merge_string_list(row.get("kept_targets") or [])
        discarded_targets = self._merge_target_payloads(row.get("discarded_targets") or [])
        if not source_query and not candidate_targets and not kept_targets:
            return {}
        return {
            "source_query": source_query,
            "planning_incomplete": bool(row.get("planning_incomplete")),
            "targets_generated": len(candidate_targets),
            "targets_kept": len(kept_targets),
            "targets_discarded": len(discarded_targets),
            "candidate_targets": candidate_targets,
            "kept_targets": kept_targets,
            "discarded_targets": discarded_targets,
            "fallback_used": bool(row.get("fallback_used")),
            "unresolved": not bool(kept_targets),
            "validation": dict(row.get("validation") or {}),
        }

    def _query_target_row_key(self, row: dict, fallback_index: int) -> str:
        """Build a stable key for merging query-target rows."""
        source_query = str(row.get("source_query") or "").strip().lower()
        if source_query:
            return f"source::{source_query}"
        kept_targets = row.get("kept_targets") or row.get("candidate_targets") or []
        if kept_targets:
            return f"target::{str(kept_targets[0]).strip().lower()}"
        return f"row::{fallback_index}"

    def _merge_string_list(self, items_a: List[Any], items_b: Optional[List[Any]] = None) -> List[str]:
        """Merge string lists while preserving order and removing duplicates."""
        merged: List[str] = []
        seen = set()
        for item in [*(items_a or []), *((items_b or []) if items_b is not None else [])]:
            text = re.sub(r"\s+", " ", str(item or "").strip())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)
        return merged

    def _merge_target_payloads(self, items_a: List[Any], items_b: Optional[List[Any]] = None) -> List[Any]:
        """Merge payload lists used by discarded-target metadata without dropping structure."""
        merged: List[Any] = []
        seen = set()
        for item in [*(items_a or []), *((items_b or []) if items_b is not None else [])]:
            if isinstance(item, dict):
                normalized = dict(item)
                payload_key = str(normalized.get("target") or json.dumps(normalized, sort_keys=True, ensure_ascii=False))
            else:
                normalized = re.sub(r"\s+", " ", str(item or "").strip())
                payload_key = normalized
            if not payload_key:
                continue
            dedupe_key = payload_key.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged.append(normalized)
        return merged

    def _merge_hint_text(self, existing_hint: str, new_hint: str) -> str:
        """Join hint strings without repeating identical fragments."""
        parts = self._merge_string_list([existing_hint], [new_hint])
        return " ; ".join(parts)

    async def _decompose_targets(
        self,
        topic: str,
        research_context: dict,
        extra_hints: str,
        model_name: str,
    ) -> List[str]:
        description = str(research_context.get("description") or "")
        key_points = research_context.get("key_points") or []
        key_points_text = ", ".join(str(x) for x in key_points if str(x).strip())
        min_targets = self._min_search_targets
        max_targets = self._max_search_targets
        prompt = [
            {
                "role": "system",
                "content": (
                    f"You are a search planner. Produce {min_targets} to {max_targets} independent search targets. "
                    f"Return JSON only as a list of strings ({min_targets}-{max_targets} items)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n"
                    f"Description: {description}\n"
                    f"Key points: {key_points_text}\n"
                    f"Extra hints: {extra_hints}\n"
                    "Constraints: convert hints into explicit search constraints. "
                    f"Output JSON list with {min_targets}-{max_targets} concise targets."
                ),
            },
        ]

        try:
            response = await call_model(prompt=prompt, model=model_name)
            parsed = self._parse_targets(response)
            if min_targets <= len(parsed) <= max_targets:
                return parsed
        except Exception as exc:
            self.logger.warning(f"Failed to decompose targets with model {model_name}: {exc}")

        return self._fallback_targets(topic, extra_hints)

    def _parse_targets(self, response: Any) -> List[str]:
        if isinstance(response, list):
            return self._clean_targets(response)

        if isinstance(response, dict):
            for key in ("targets", "queries", "search_targets"):
                if key in response and isinstance(response[key], list):
                    return self._clean_targets(response[key])

        if isinstance(response, str):
            try:
                repaired = json_repair.loads(response)
                return self._parse_targets(repaired)
            except Exception:
                pass
            match = re.findall(r'"([^"]+)"', response)
            if match:
                return self._clean_targets(match)
        return []

    def _clean_targets(self, raw_targets: Iterable[Any]) -> List[str]:
        cleaned = []
        seen = set()
        for item in raw_targets:
            text = re.sub(r"\s+", " ", str(item or "").strip())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
            if len(cleaned) >= self._max_search_targets:
                break
        return cleaned

    def _fallback_targets(self, topic: str, extra_hints: str) -> List[str]:
        hint_tail = f" | constraints: {extra_hints}" if extra_hints else ""
        return [
            f"{topic} key facts and latest updates{hint_tail}",
            f"{topic} opposing viewpoints and contradictory evidence{hint_tail}",
            f"{topic} quantitative data and primary sources{hint_tail}",
        ]

    def _merge_extra_hints(self, extra_hints: str | None, audit_feedback: dict | None) -> str:
        parts = []
        if isinstance(extra_hints, str) and extra_hints.strip():
            parts.append(extra_hints.strip())
        if isinstance(audit_feedback, dict):
            for key in ("new_query_suggestion", "instruction"):
                value = audit_feedback.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        return " ; ".join(parts)

    def _select_engines(self, iteration: int, topic: str) -> List[str]:
        if iteration <= 1:
            return ["tavily", "duckduckgo"]
        domain = self._classify_domain(topic)
        return self.DOMAIN_TO_ENGINES.get(domain, self.DOMAIN_TO_ENGINES["general"])

    def _classify_domain(self, text: str) -> str:
        lowered = text.lower()
        tech_keywords = {"llm", "model", "software", "chip", "gpu", "ai", "cloud", "open-source", "programming"}
        medical_keywords = {"medical", "clinical", "drug", "patient", "disease", "hospital", "pubmed", "health"}
        finance_keywords = {"finance", "stock", "revenue", "earnings", "fiscal", "market", "bank", "valuation"}
        if any(k in lowered for k in medical_keywords):
            return "medical"
        if any(k in lowered for k in tech_keywords):
            return "tech"
        if any(k in lowered for k in finance_keywords):
            return "finance"
        return "general"

    async def _collect_results_for_target(
        self,
        target: str,
        engines: List[str],
        query_domains: List[str],
        max_results: int,
    ) -> Tuple[List[dict], List[str]]:
        all_results: List[dict] = []
        used_engines = set()

        for engine in engines:
            results, failed = await self._search_with_engine(
                engine=engine,
                query=target,
                query_domains=query_domains,
                max_results=max_results,
            )
            if results:
                all_results.extend(results)
                used_engines.add(engine)
            if failed:
                for fallback_engine in self.SEARCH_FALLBACK_ENGINES:
                    if fallback_engine in engines or fallback_engine in used_engines:
                        continue
                    fallback_results, _ = await self._search_with_engine(
                        engine=fallback_engine,
                        query=target,
                        query_domains=query_domains,
                        max_results=max_results,
                    )
                    if fallback_results:
                        all_results.extend(fallback_results)
                        used_engines.add(fallback_engine)

        deduped = self._dedupe_search_results(all_results)
        enriched = await self._materialize_search_results(deduped)
        return enriched, sorted(used_engines)

    async def _search_with_engine(
        self,
        engine: str,
        query: str,
        query_domains: List[str],
        max_results: int,
    ) -> Tuple[List[dict], bool]:
        retriever_cls = get_retriever(engine)
        if retriever_cls is None:
            return [], True
        try:
            retriever = self._instantiate_retriever(retriever_cls, query, query_domains)
            results = await asyncio.to_thread(retriever.search, max_results=max_results)
            normalized = self._normalize_search_results(results, source_engine=engine)
            return normalized, False
        except Exception as exc:
            self.logger.warning(f"Search engine {engine} failed for query '{query}': {exc}")
            return [], True

    def _instantiate_retriever(self, retriever_cls, query: str, query_domains: List[str]):
        attempts = [
            {"query": query, "headers": self.headers, "query_domains": query_domains},
            {"query": query, "query_domains": query_domains},
            {"query": query},
        ]
        positional_attempts = [
            (query, self.headers, query_domains),
            (query, query_domains),
            (query,),
        ]
        for kwargs in attempts:
            try:
                return retriever_cls(**kwargs)
            except TypeError:
                continue
        for args in positional_attempts:
            try:
                return retriever_cls(*args)
            except TypeError:
                continue
        return retriever_cls(query)

    def _normalize_search_results(self, results: Any, source_engine: str) -> List[dict]:
        normalized = []
        if not isinstance(results, list):
            return normalized

        for item in results:
            if not isinstance(item, dict):
                continue
            href = item.get("href") or item.get("url")
            if not href:
                continue
            normalized.append(
                {
                    "href": str(href),
                    "title": str(item.get("title") or ""),
                    "body": str(item.get("body") or item.get("content") or item.get("raw_content") or ""),
                    "source_engine": source_engine,
                    "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                }
            )
        return normalized

    def _dedupe_search_results(self, results: List[dict]) -> List[dict]:
        deduped = {}
        for item in results:
            url = item.get("href")
            if not url:
                continue
            key = self._normalize_url(url)
            if key not in deduped:
                deduped[key] = item
        return list(deduped.values())

    def _normalize_url(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
            normalized = parsed._replace(
                fragment="",
                query=query,
                path=parsed.path.rstrip("/") or "/",
            )
            return urlunparse(normalized)
        except Exception:
            return url

    async def _materialize_search_results(self, search_results: List[dict]) -> List[dict]:
        def _is_anti_scrape(url: str) -> bool:
            try:
                from urllib.parse import urlparse
                host = urlparse(url).netloc.lower().lstrip("www.")
                return any(host == d or host.endswith("." + d) for d in self.ANTI_SCRAPE_DOMAINS)
            except Exception:
                return False

        # Only scrape URLs that (a) are not on the anti-scrape list and
        # (b) don't already have enough content from the search engine.
        needs_scraping = [
            item for item in search_results
            if item.get("href")
            and not _is_anti_scrape(item["href"])
            and len(item.get("body") or "") < 500
        ]
        urls = [item["href"] for item in needs_scraping]
        scraped_map = {}
        if urls:
            scraped_data = await self._scrape_urls(urls)
            for item in scraped_data:
                if isinstance(item, dict) and item.get("url"):
                    scraped_map[self._normalize_url(item["url"])] = item

        enriched = []
        for item in search_results:
            url = item.get("href")
            norm_url = self._normalize_url(url)
            scraped = scraped_map.get(norm_url, {})
            # Priority: full scraped content > Tavily raw_content (body) > nothing
            raw_content = str(scraped.get("raw_content") or item.get("body") or "")
            if not raw_content.strip():
                continue
            metadata = {}
            metadata.update(item.get("metadata") or {})
            metadata.setdefault("publish_date", scraped.get("publish_date"))
            metadata.setdefault("author", scraped.get("author"))
            enriched.append(
                {
                    "url": url,
                    "title": scraped.get("title") or item.get("title") or "",
                    "raw_content": raw_content,
                    "source_engine": item.get("source_engine"),
                    "metadata": metadata,
                }
            )
        return enriched

    async def _scrape_urls(self, urls: List[str]) -> List[dict]:
        cfg = Config()
        worker_pool = WorkerPool(cfg.max_scraper_workers, cfg.scraper_rate_limit_delay)
        try:
            scraped_data, _ = await scrape_urls(urls, cfg, worker_pool)
            return scraped_data
        except Exception as exc:
            self.logger.warning(f"URL scraping failed: {exc}")
            return []
        finally:
            worker_pool.executor.shutdown(wait=False, cancel_futures=True)

    def _build_passages_from_results(self, results: List[dict]) -> List[dict]:
        passages = []
        for item in results:
            url = item.get("url")
            content = str(item.get("raw_content") or "")
            metadata = item.get("metadata") or {}
            source_engine = item.get("source_engine")
            for chunk in self._split_to_passages(content, min_chars=200, max_chars=500):
                passages.append(
                    {
                        "content": chunk,
                        "source_url": url,
                        "metadata": {
                            "publish_date": metadata.get("publish_date"),
                            "author": metadata.get("author"),
                        },
                        "source_engine": source_engine,
                    }
                )
        return passages

    def _split_to_passages(self, text: str, min_chars: int = 200, max_chars: int = 500) -> List[str]:
        compact = re.sub(r"\s+", " ", text or "").strip()
        if not compact:
            return []
        if len(compact) <= max_chars:
            return [compact]

        sentences = re.split(r"(?<=[.!?。！？])\s+", compact)
        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chars:
                if current:
                    chunks.append(" ".join(current).strip())
                    current, current_len = [], 0
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i : i + max_chars].strip())
                continue

            projected = current_len + len(sentence) + (1 if current else 0)
            if projected > max_chars and current:
                chunks.append(" ".join(current).strip())
                current, current_len = [sentence], len(sentence)
            else:
                current.append(sentence)
                current_len = projected

        if current:
            chunks.append(" ".join(current).strip())

        merged = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue
            if len(chunk) < min_chars and len(merged[-1]) + len(chunk) + 1 <= max_chars:
                merged[-1] = f"{merged[-1]} {chunk}".strip()
            else:
                merged.append(chunk)
        return merged

    async def _select_top_passages_with_mmr(
        self,
        query: str,
        passages: List[dict],
        top_k: int = 10,
        lambda_param: float = 0.7,
        time_budget_s: float = 5.0,
    ) -> List[dict]:
        if not passages:
            return []

        start = time.perf_counter()
        if self._use_embedding_mmr:
            selected = await self._select_top_passages_with_embedding_mmr(
                query=query,
                passages=passages,
                top_k=top_k,
                lambda_param=lambda_param,
                start_time=start,
                time_budget_s=time_budget_s,
            )
            if selected:
                return selected

        query_vec = self._text_to_vector(query)
        passage_vecs = [self._text_to_vector(p["content"]) for p in passages]
        relevance = [self._cosine_similarity(query_vec, vec) for vec in passage_vecs]

        chosen_indices: List[int] = []
        candidates = set(range(len(passages)))

        while candidates and len(chosen_indices) < min(top_k, len(passages)):
            if time.perf_counter() - start > time_budget_s:
                return self._fallback_rank(passages, relevance, chosen_indices, top_k)

            if not chosen_indices:
                best = max(candidates, key=lambda idx: relevance[idx])
            else:
                best = None
                best_score = float("-inf")
                for idx in candidates:
                    diversity_penalty = max(
                        self._cosine_similarity(passage_vecs[idx], passage_vecs[selected])
                        for selected in chosen_indices
                    )
                    score = (lambda_param * relevance[idx]) - ((1.0 - lambda_param) * diversity_penalty)
                    if score > best_score:
                        best_score = score
                        best = idx
                if best is None:
                    break
            chosen_indices.append(best)
            candidates.remove(best)

        top_passages = []
        for idx in chosen_indices:
            passage = passages[idx]
            top_passages.append(
                {
                    "content": passage["content"],
                    "source_url": passage["source_url"],
                    "metadata": passage.get("metadata") or {},
                    "relevance_score": round(float(relevance[idx]), 4),
                }
            )
        return top_passages[:top_k]

    async def _select_top_passages_with_embedding_mmr(
        self,
        query: str,
        passages: List[dict],
        top_k: int,
        lambda_param: float,
        start_time: float,
        time_budget_s: float,
    ) -> List[dict]:
        embeddings = self._get_mmr_embeddings()
        if embeddings is None:
            return []

        remaining = max(0.0, time_budget_s - (time.perf_counter() - start_time))
        if remaining <= 0.05:
            return []

        max_embed_passages = self._env_int("SCRAPING_MMR_MAX_EMBED_PASSAGES", default=80, min_value=10, max_value=500)
        candidates = passages
        if len(passages) > max_embed_passages:
            query_vec = self._text_to_vector(query)
            passage_vecs = [self._text_to_vector(p["content"]) for p in passages]
            relevance = [self._cosine_similarity(query_vec, vec) for vec in passage_vecs]
            ranked = sorted(range(len(passages)), key=lambda idx: relevance[idx], reverse=True)[:max_embed_passages]
            candidates = [passages[idx] for idx in ranked]

        texts = [query] + [p.get("content") or "" for p in candidates]
        texts = [str(t) for t in texts]
        if not texts[0].strip():
            return []

        remaining = max(0.0, time_budget_s - (time.perf_counter() - start_time))
        if remaining <= 0.05:
            return []

        try:
            vectors = await asyncio.wait_for(
                asyncio.to_thread(embeddings.embed_documents, texts),
                timeout=remaining,
            )
        except Exception as exc:
            self._use_embedding_mmr = False
            self._mmr_embeddings_error = f"embedding_call_failed:{type(exc).__name__}"
            self.logger.debug("Embedding MMR failed, falling back to BOW MMR: %s", exc)
            return []

        if not vectors or len(vectors) != len(texts):
            return []

        query_vec = vectors[0]
        passage_vecs = vectors[1:]
        if not passage_vecs:
            return []

        norms = [self._l2_norm(query_vec)] + [self._l2_norm(v) for v in passage_vecs]
        if norms[0] == 0.0:
            return []

        relevance = [
            self._cosine_similarity_dense(query_vec, norms[0], vec, norms[i + 1])
            for i, vec in enumerate(passage_vecs)
        ]

        chosen_indices: List[int] = []
        remaining_indices = set(range(len(candidates)))

        def sim(i: int, j: int) -> float:
            return self._cosine_similarity_dense(
                passage_vecs[i],
                norms[i + 1],
                passage_vecs[j],
                norms[j + 1],
            )

        while remaining_indices and len(chosen_indices) < min(top_k, len(candidates)):
            if time.perf_counter() - start_time > time_budget_s:
                return self._fallback_rank(candidates, relevance, chosen_indices, top_k)

            if not chosen_indices:
                best = max(remaining_indices, key=lambda idx: relevance[idx])
            else:
                best = None
                best_score = float("-inf")
                for idx in remaining_indices:
                    diversity_penalty = max(sim(idx, selected) for selected in chosen_indices)
                    score = (lambda_param * relevance[idx]) - ((1.0 - lambda_param) * diversity_penalty)
                    if score > best_score:
                        best_score = score
                        best = idx
                if best is None:
                    break

            chosen_indices.append(best)
            remaining_indices.remove(best)

        top_passages = []
        for idx in chosen_indices[:top_k]:
            passage = candidates[idx]
            top_passages.append(
                {
                    "content": passage["content"],
                    "source_url": passage["source_url"],
                    "metadata": passage.get("metadata") or {},
                    "relevance_score": round(float(relevance[idx]), 4),
                }
            )
        return top_passages

    def _fallback_rank(
        self,
        passages: List[dict],
        relevance: List[float],
        chosen_indices: List[int],
        top_k: int,
    ) -> List[dict]:
        remaining = [idx for idx in range(len(passages)) if idx not in chosen_indices]
        remaining.sort(key=lambda idx: relevance[idx], reverse=True)
        picked = chosen_indices + remaining
        result = []
        seen_snippets = set()
        for idx in picked:
            content = passages[idx]["content"]
            key = re.sub(r"\s+", " ", content.lower())[:180]
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            result.append(
                {
                    "content": content,
                    "source_url": passages[idx]["source_url"],
                    "metadata": passages[idx].get("metadata") or {},
                    "relevance_score": round(float(relevance[idx]), 4),
                }
            )
            if len(result) >= top_k:
                break
        return result

    def _text_to_vector(self, text: str) -> Counter:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", (text or "").lower())
        return Counter(tokens)

    def _cosine_similarity(self, v1: Counter, v2: Counter) -> float:
        if not v1 or not v2:
            return 0.0
        common = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[token] * v2[token] for token in common)
        norm1 = sum(value * value for value in v1.values()) ** 0.5
        norm2 = sum(value * value for value in v2.values()) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _l2_norm(self, vec: Any) -> float:
        if not vec:
            return 0.0
        try:
            return sqrt(sum(float(x) * float(x) for x in vec))
        except Exception:
            return 0.0

    def _cosine_similarity_dense(self, v1: Any, n1: float, v2: Any, n2: float) -> float:
        if not v1 or not v2 or n1 == 0.0 or n2 == 0.0:
            return 0.0
        try:
            dot = 0.0
            for a, b in zip(v1, v2):
                dot += float(a) * float(b)
            return dot / (n1 * n2)
        except Exception:
            return 0.0

    @staticmethod
    def _task_value(task: dict, *keys: str) -> Any:
        """Return the first explicit task value across canonical and legacy keys."""
        for key in keys:
            if key in task and task.get(key) is not None:
                return task.get(key)
        return None

    def _task_int(self, task: dict, *keys: str, default: int) -> int:
        """Read an integer task setting while honoring legacy key names."""
        raw = self._task_value(task, *keys)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def _env_truthy(self, key: str, default: bool = False) -> bool:
        raw = os.getenv(key)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _env_int(self, key: str, default: int, min_value: int, max_value: int) -> int:
        raw = os.getenv(key)
        if raw is None:
            return default
        try:
            parsed = int(str(raw).strip())
        except ValueError:
            return default
        return max(min_value, min(max_value, parsed))

    def _get_mmr_embeddings(self):
        if self._mmr_embeddings_error is not None:
            return None
        if self._mmr_embeddings is not None:
            return self._mmr_embeddings
        try:
            cfg = Config()
            provider = getattr(cfg, "embedding_provider", None)
            model = getattr(cfg, "embedding_model", None)
            kwargs = getattr(cfg, "embedding_kwargs", {}) or {}
            if not provider or not model:
                self._mmr_embeddings_error = "missing_embedding_config"
                return None
            self._mmr_embeddings = Memory(provider, model, **kwargs).get_embeddings()
            return self._mmr_embeddings
        except Exception as exc:
            self._mmr_embeddings_error = str(exc)
            self.logger.debug("MMR embedding init failed, falling back to BOW MMR: %s", exc)
            return None

    def _empty_packet(self) -> dict:
        return {
            "iteration_index": 1,
            "model_level": "Level_1_Base",
            "active_engines": [],
            "search_log": [],
            "query_target_map": [],
            "coverage_snapshot": {
                "query_coverage": 0.0,
                "keypoint_coverage": 0.0,
                "section_coverage": 0.0,
                "query_total": 0,
                "query_covered": 0,
                "uncovered_queries": [],
                "keypoint_total": 0,
                "keypoint_covered": 0,
                "uncovered_key_points": [],
                "coverage_threshold": self.COVERAGE_THRESHOLD,
            },
            "fallback_used": False,
        }

    def _build_compatible_draft(self, topic: str, scraping_packet: dict) -> str:
        lines = [f"### {topic}", "", "#### Curated Objective Passages"]
        counter = 1
        for target_log in scraping_packet.get("search_log", []):
            target = target_log.get("target", "")
            lines.append("")
            lines.append(f"Target: {target}")
            for passage in target_log.get("top_10_passages", []):
                content = str(passage.get("content") or "").strip()
                source = str(passage.get("source_url") or "").strip()
                if not content:
                    continue
                lines.append(f"{counter}. {content}")
                if source:
                    lines.append(f"   Source: {source}")
                counter += 1
                if counter > 15:
                    break
            if counter > 15:
                break

        if counter == 1:
            lines.append("")
            lines.append("No reliable passages were retrieved in this run.")
        return "\n".join(lines).strip()
