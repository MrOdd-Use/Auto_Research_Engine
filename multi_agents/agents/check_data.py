import logging
import os
import re
from typing import Any, List

from .state_controller import StateController
from .utils.llms import call_model
from .utils.views import print_agent_output


class CheckDataAgent:
    """Check Data Agent: validates scraping passages and decides accept/retry/blocked."""

    ROUND_TO_TIER = {
        1: 2,
        2: 1,
        3: 0,
    }
    MAX_RETRIES = 3
    PASS_THRESHOLD = 0.7
    COVERAGE_THRESHOLD = 0.7
    DEFAULT_RETRY_BUDGET = 1
    MAX_PASSAGES_CHARS = 1500
    SUSPICIOUS_TOKENS = {
        "projection", "projected", "forecast", "estimated",
        "预计", "预测", "预估",
    }
    ACTUAL_TOKENS = {
        "actual", "audited", "reported", "official filing", "actuals",
        "审计", "实绩", "财报",
    }

    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.logger = logging.getLogger(__name__)
        state_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_state.json")
        self.state_controller = StateController(state_file=state_file)

    async def run(self, draft_state: dict) -> dict:
        task = draft_state.get("task") or {}
        topic = str(draft_state.get("topic") or "").strip() or "Main Research Focus"
        research_context = draft_state.get("research_context") or {}
        scraping_packet = draft_state.get("scraping_packet") or {}
        iteration_index = self._normalize_iteration(draft_state.get("iteration_index"))
        max_retries = self._normalize_max_retries(task.get("check_data_max_retries"))
        coverage_report = self._build_coverage_report(scraping_packet, research_context)
        coverage_failed = float(coverage_report.get("section_coverage") or 0.0) < self.COVERAGE_THRESHOLD

        await self._emit_log(
            "check_data_start",
            f"Check Data evaluating topic '{topic}' at iteration {iteration_index}.",
        )

        tier_idx = self.ROUND_TO_TIER.get(iteration_index, 0)
        self.state_controller.set_tier("check_data", tier_idx)

        try:
            target_entries = self._extract_target_entries(scraping_packet)
            precheck = self._constraint_guard(target_entries)
            llm_eval_report = await self._llm_eval_targets(topic, target_entries, precheck)

            final_score = float(llm_eval_report.get("final_score") or 0.0)
            hard_fail = bool(precheck.get("hard_fail"))
            status = self._resolve_status(
                final_score=final_score,
                hard_fail=hard_fail,
                iteration_index=iteration_index,
                max_retries=max_retries,
                coverage_failed=coverage_failed,
            )
            feedback_packet = self._build_feedback_packet(
                topic,
                precheck,
                llm_eval_report,
                coverage_report,
            )
            verdict = {
                "status": status,
                "llm_eval_report": llm_eval_report,
                "feedback_packet": feedback_packet,
                "guard_report": precheck,
                "coverage_report": coverage_report,
            }

            if status == "ACCEPT":
                await self._emit_log(
                    "check_data_accept",
                    f"Check Data accepted topic '{topic}' with score {final_score:.2f}.",
                )
                return {
                    "check_data_action": "accept",
                    "check_data_verdict": verdict,
                    "audit_feedback": {
                        "is_satisfied": True,
                        "confidence_score": final_score,
                        "section_coverage": coverage_report.get("section_coverage"),
                    },
                    "iteration_index": iteration_index,
                    "extra_hints": None,
                }

            if status == "RETRY":
                next_iteration = min(iteration_index + 1, max_retries)
                retry_hints = self._merge_retry_hints(feedback_packet)
                await self._emit_log(
                    "check_data_retry",
                    f"Check Data requested retry for '{topic}' (score {final_score:.2f}, next iteration {next_iteration}).",
                )
                return {
                    "check_data_action": "retry",
                    "check_data_verdict": verdict,
                    "audit_feedback": {
                        "is_satisfied": False,
                        "confidence_score": final_score,
                        "instruction": feedback_packet.get("instruction"),
                        "new_query_suggestion": feedback_packet.get("new_query_suggestion"),
                        "uncovered_queries": coverage_report.get("uncovered_queries") or [],
                        "section_coverage": coverage_report.get("section_coverage"),
                    },
                    "iteration_index": next_iteration,
                    "extra_hints": retry_hints,
                }

            placeholder = self._blocked_placeholder(
                topic,
                llm_eval_report,
                feedback_packet,
                max_retries=max_retries,
            )
            await self._emit_log(
                "check_data_blocked",
                f"Check Data blocked topic '{topic}' after {iteration_index} attempts (score {final_score:.2f}).",
            )
            return {
                "check_data_action": "blocked",
                "check_data_verdict": verdict,
                "audit_feedback": {
                    "is_satisfied": False,
                    "confidence_score": final_score,
                    "instruction": feedback_packet.get("instruction"),
                    "new_query_suggestion": feedback_packet.get("new_query_suggestion"),
                    "uncovered_queries": coverage_report.get("uncovered_queries") or [],
                    "section_coverage": coverage_report.get("section_coverage"),
                },
                "draft": {topic: placeholder},
                "review": None,
                "iteration_index": iteration_index,
                "extra_hints": self._merge_retry_hints(feedback_packet),
            }
        finally:
            self.state_controller.set_tier("check_data", 2)

    def _normalize_iteration(self, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 1
        return min(max(parsed, 1), self.MAX_RETRIES)

    def _normalize_max_retries(self, value: Any) -> int:
        if value is None:
            parsed = self.DEFAULT_RETRY_BUDGET + 1
            return min(max(parsed, 1), self.MAX_RETRIES)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = self.DEFAULT_RETRY_BUDGET + 1
        return min(max(parsed, 1), self.MAX_RETRIES)

    def _extract_target_entries(self, scraping_packet: dict) -> List[dict]:
        """Extract (target, passages_text) pairs from search_log."""
        entries = []
        for item in scraping_packet.get("search_log", []):
            if not isinstance(item, dict):
                continue
            target = str(item.get("target") or "").strip()
            if not target:
                continue
            passages_parts = []
            for passage in item.get("top_10_passages", []):
                if not isinstance(passage, dict):
                    continue
                content = str(passage.get("content") or "").strip()
                if content:
                    passages_parts.append(content)
            passages_text = " ".join(passages_parts)[: self.MAX_PASSAGES_CHARS]
            entries.append({"target": target, "passages_text": passages_text})
        return entries

    def _constraint_guard(self, target_entries: List[dict]) -> dict:
        """Pre-screen for empty passages and suspicious forecast language."""
        all_passages = " ".join(e["passages_text"] for e in target_entries).lower()
        empty_targets = [e["target"] for e in target_entries if not e["passages_text"].strip()]

        suspicious_hit = any(token in all_passages for token in self.SUSPICIOUS_TOKENS)
        actual_hit = any(token in all_passages for token in self.ACTUAL_TOKENS)
        suspicious = suspicious_hit and not actual_hit

        hard_fail_reasons = []
        if empty_targets:
            hard_fail_reasons.append("empty_passages")
        if not target_entries:
            hard_fail_reasons.append("no_targets")

        return {
            "hard_fail": bool(hard_fail_reasons),
            "hard_fail_reasons": hard_fail_reasons,
            "empty_targets": empty_targets,
            "suspicious": suspicious,
            "suspicious_reason": "projection_or_forecast_without_actual" if suspicious else "",
        }

    async def _llm_eval_targets(self, topic: str, target_entries: List[dict], precheck: dict) -> dict:
        """Batch LLM evaluation: judge whether each target is answered by its passages."""
        if not target_entries:
            return {
                "final_score": 0.0,
                "checked_count": 0,
                "total_targets": 0,
                "target_results": [],
                "failed_targets": [],
            }

        if precheck.get("hard_fail") and "empty_passages" in precheck.get("hard_fail_reasons", []):
            results = [
                {"target": e["target"], "checked": False, "reason": "no passages retrieved"}
                for e in target_entries
            ]
            return self._build_llm_eval_report(results, short_circuit=True)

        blocks = []
        for i, entry in enumerate(target_entries, start=1):
            evidence = entry["passages_text"] if entry["passages_text"].strip() else "(no evidence retrieved)"
            blocks.append(
                f"## Target {i}\n"
                f"Search goal: {entry['target']}\n"
                f"Evidence:\n{evidence}"
            )

        targets_section = "\n\n".join(blocks)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a research quality evaluator. "
                    "For each search target below, judge whether the provided evidence directly and specifically answers the search goal. "
                    "'checked' is true only if the evidence contains a concrete, on-topic answer — not a vague mention or tangential reference. "
                    "Return a JSON array only, no extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{targets_section}\n\n"
                    "---\n"
                    "Return a JSON array with one object per target:\n"
                    '[{"target": "...", "checked": true/false, "reason": "one sentence"}, ...]'
                ),
            },
        ]

        try:
            raw = await call_model(
                prompt=prompt,
                model="",
                response_format="json",
                route_context={
                    "shared_agent_class": "check_data_agent",
                    "agent_role": "check_data",
                    "stage_name": "check_data",
                    "task": (
                        "Task: Evaluate whether each target below is directly answered by "
                        f"the provided evidence for the section topic '{topic}'."
                    ),
                },
            )
            results = raw if isinstance(raw, list) else []
        except Exception as exc:
            self.logger.warning("LLM eval failed, falling back to unchecked: %s", exc)
            results = [
                {"target": e["target"], "checked": False, "reason": f"llm_error: {exc}"}
                for e in target_entries
            ]

        return self._build_llm_eval_report(results)

    def _build_llm_eval_report(self, results: list, short_circuit: bool = False) -> dict:
        checked_count = sum(1 for r in results if r.get("checked"))
        total = len(results)
        final_score = round(checked_count / total, 4) if total else 0.0
        failed_targets = [r for r in results if not r.get("checked")]
        return {
            "final_score": final_score,
            "checked_count": checked_count,
            "total_targets": total,
            "target_results": results,
            "failed_targets": failed_targets,
            "short_circuit": short_circuit,
        }

    def _resolve_status(
        self,
        final_score: float,
        hard_fail: bool,
        iteration_index: int,
        max_retries: int,
        coverage_failed: bool,
    ) -> str:
        if coverage_failed:
            if iteration_index < max_retries:
                return "RETRY"
            return "BLOCKED"
        if final_score >= self.PASS_THRESHOLD and not hard_fail:
            return "ACCEPT"
        if iteration_index < max_retries:
            return "RETRY"
        return "BLOCKED"

    def _build_feedback_packet(
        self,
        topic: str,
        precheck: dict,
        llm_eval_report: dict,
        coverage_report: dict,
    ) -> dict:
        missing_clauses = []

        if "empty_passages" in (precheck.get("hard_fail_reasons") or []):
            missing_clauses.append("some search targets returned no evidence at all")
        if "no_targets" in (precheck.get("hard_fail_reasons") or []):
            missing_clauses.append("no search targets were found in the scraping packet")
        if precheck.get("suspicious"):
            missing_clauses.append("exclude projection/forecast language and require actual or audited values")
        if float(coverage_report.get("section_coverage") or 0.0) < self.COVERAGE_THRESHOLD:
            missing_clauses.append(
                f"increase section evidence coverage to at least {self.COVERAGE_THRESHOLD:.2f}"
            )

        failed_targets = llm_eval_report.get("failed_targets") or []
        failed_target_texts = [str(r.get("target") or "") for r in failed_targets if r.get("target")]

        query_parts = failed_target_texts[:3]
        if precheck.get("suspicious"):
            query_parts.append("-projection -forecast")
        query = " ".join(query_parts).strip() or f"{topic} actual audited report"
        query = re.sub(r"\s+", " ", query).strip()

        uncovered_queries = coverage_report.get("uncovered_queries") or []
        if uncovered_queries:
            query = f"{query} {' '.join(uncovered_queries[:2])}".strip()

        if not missing_clauses and failed_targets:
            missing_clauses.append(
                f"re-search the following unanswered targets: {', '.join(failed_target_texts[:3])}"
            )
        if not missing_clauses:
            missing_clauses.append("improve relevance and evidence quality")

        return {
            "instruction": "Previous results are insufficient: " + "; ".join(missing_clauses) + ".",
            "new_query_suggestion": query,
            "failed_targets": failed_targets,
            "uncovered_queries": uncovered_queries,
        }

    def _merge_retry_hints(self, feedback_packet: dict) -> str:
        parts = []
        instruction = str(feedback_packet.get("instruction") or "").strip()
        suggestion = str(feedback_packet.get("new_query_suggestion") or "").strip()
        uncovered_queries = feedback_packet.get("uncovered_queries") or []
        if instruction:
            parts.append(instruction)
        if suggestion:
            parts.append(f"Search expression: {suggestion}")
        if uncovered_queries:
            parts.append(f"Prioritize unresolved queries: {', '.join(uncovered_queries[:3])}")
        return " ; ".join(parts)

    def _blocked_placeholder(
        self,
        topic: str,
        llm_eval_report: dict,
        feedback_packet: dict,
        max_retries: int,
    ) -> str:
        score = llm_eval_report.get("final_score")
        suggestion = feedback_packet.get("new_query_suggestion")
        return (
            f"### {topic}\n\n"
            f"证据不足，需人工复核（final_score={score}）。\n"
            f"系统在 {max_retries} 轮验证后未能达到最低置信阈值（0.7）。\n"
            f"建议补充搜索：{suggestion}\n"
        ).strip()

    async def _emit_log(self, event: str, message: str) -> None:
        if self.websocket and self.stream_output:
            await self.stream_output("logs", event, message, self.websocket)
        else:
            print_agent_output(message, agent="CHECK_DATA")
        self.logger.info(message)

    def _contains_term(self, corpus: str, term: str) -> bool:
        if not term:
            return False
        if " " not in term:
            return term in corpus
        return re.search(rf"\b{re.escape(term)}\b", corpus) is not None

    def _entity_density(self, corpus: str, subject: str) -> float:
        if not corpus or not subject:
            return 0.0
        occurrences = corpus.count(subject)
        words = max(len(corpus.split()), 1)
        return round((occurrences / words) * 100, 4)

    def _build_coverage_report(self, scraping_packet: dict, research_context: dict) -> dict:
        snapshot = scraping_packet.get("coverage_snapshot")
        if isinstance(snapshot, dict):
            query_coverage = float(snapshot.get("query_coverage") or 0.0)
            keypoint_coverage = float(snapshot.get("keypoint_coverage") or 0.0)
            section_coverage = float(snapshot.get("section_coverage") or min(query_coverage, keypoint_coverage))
            query_total = int(snapshot.get("query_total") or 0)
            keypoint_total = int(snapshot.get("keypoint_total") or 0)
            if query_total == 0 and keypoint_total == 0:
                query_coverage = 1.0
                keypoint_coverage = 1.0
                section_coverage = 1.0
            return {
                "query_coverage": round(query_coverage, 4),
                "keypoint_coverage": round(keypoint_coverage, 4),
                "section_coverage": round(section_coverage, 4),
                "uncovered_queries": snapshot.get("uncovered_queries") or [],
                "uncovered_key_points": snapshot.get("uncovered_key_points") or [],
                "coverage_threshold": float(snapshot.get("coverage_threshold") or self.COVERAGE_THRESHOLD),
            }

        source_queries = [
            str(item or "").strip()
            for item in (research_context.get("research_queries") or [])
            if str(item or "").strip()
        ]
        key_points = [
            str(item or "").strip()
            for item in (research_context.get("key_points") or [])
            if str(item or "").strip()
        ]
        search_log = scraping_packet.get("search_log") or []
        covered_queries_set = set()
        corpus = []
        for row in search_log:
            source_query = str(row.get("source_query") or "").strip()
            passages = row.get("top_10_passages") or []
            if source_query and passages:
                covered_queries_set.add(source_query)
            corpus.append(str(row.get("target") or ""))
            for passage in passages:
                corpus.append(str(passage.get("content") or ""))
        corpus_text = " ".join(corpus).lower()

        query_total = len(source_queries)
        query_covered = sum(1 for query in source_queries if query in covered_queries_set)
        query_coverage = (query_covered / query_total) if query_total else 0.0

        if not key_points:
            keypoint_coverage = 1.0
            uncovered_key_points = []
        else:
            corpus_tokens = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", corpus_text))
            uncovered_key_points = []
            for point in key_points:
                point_tokens = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", point.lower()))
                if not point_tokens:
                    continue
                overlap = len(point_tokens & corpus_tokens) / len(point_tokens)
                if overlap < 0.4:
                    uncovered_key_points.append(point)
            keypoint_coverage = (len(key_points) - len(uncovered_key_points)) / len(key_points)

        if query_total == 0 and len(key_points) == 0:
            query_coverage = 1.0
            keypoint_coverage = 1.0
        section_coverage = min(query_coverage, keypoint_coverage)
        uncovered_queries = [query for query in source_queries if query not in covered_queries_set]
        return {
            "query_coverage": round(query_coverage, 4),
            "keypoint_coverage": round(keypoint_coverage, 4),
            "section_coverage": round(section_coverage, 4),
            "uncovered_queries": uncovered_queries,
            "uncovered_key_points": uncovered_key_points,
            "coverage_threshold": self.COVERAGE_THRESHOLD,
        }
