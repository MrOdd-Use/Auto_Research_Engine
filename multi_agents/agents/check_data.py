import logging
import os
import re
from typing import Any, Dict, List, Optional

from .state_controller import StateController
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
    SUSPICIOUS_TOKENS = {
        "projection",
        "projected",
        "forecast",
        "estimated",
        "预计",
        "预测",
        "预估",
    }
    ACTUAL_TOKENS = {
        "actual",
        "audited",
        "reported",
        "official filing",
        "actuals",
        "审计",
        "实绩",
        "财报",
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
        checkpoint_note = (
            str(task.get("checkpoint_note") or "").strip()
            if task.get("checkpoint_target") == "check_data"
            else ""
        )
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
            segments = self._extract_segments(scraping_packet)
            claims = self._atomic_deconstruct(topic, research_context, checkpoint_note=checkpoint_note)
            precheck = self._constraint_guard(claims, segments)
            deep_eval_report = self._run_deep_eval(claims, segments, precheck, coverage_report)

            final_score = float(deep_eval_report.get("final_score") or 0.0)
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
                claims,
                precheck,
                deep_eval_report,
                coverage_report,
            )
            verdict = {
                "status": status,
                "deep_eval_report": deep_eval_report,
                "feedback_packet": feedback_packet,
                "atomic_claims": claims,
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

            placeholder = self._blocked_placeholder(topic, deep_eval_report, feedback_packet)
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

    def _extract_segments(self, scraping_packet: dict) -> List[str]:
        segments: List[str] = []
        for item in scraping_packet.get("search_log", []):
            if not isinstance(item, dict):
                continue
            for passage in item.get("top_10_passages", []):
                if not isinstance(passage, dict):
                    continue
                content = str(passage.get("content") or "").strip()
                if content:
                    segments.append(content)
        return segments

    def _atomic_deconstruct(self, topic: str, research_context: dict, checkpoint_note: str = "") -> dict:
        subject = self._extract_subject(topic, research_context, checkpoint_note)
        time_constraint = self._extract_time_constraint(topic)
        metric = self._extract_metric(topic, research_context, checkpoint_note)
        negative_constraints = self._extract_negative_constraints(topic)
        return {
            "subject": subject,
            "time_constraint": time_constraint,
            "metric": metric,
            "negative_constraints": negative_constraints,
        }

    def _extract_subject(self, topic: str, research_context: dict, checkpoint_note: str = "") -> str:
        key_points = research_context.get("key_points") or []
        candidates: List[str] = []
        for point in key_points:
            text = str(point or "").strip()
            if text:
                candidates.append(text)
        candidates.append(topic)
        if checkpoint_note:
            candidates.append(checkpoint_note)

        for text in candidates:
            match = re.search(r"\b([A-Z][A-Za-z0-9&\-.]{1,})\b", text)
            if match:
                return match.group(1)

        lowered = topic.lower()
        for marker in ("company", "firm", "vendor", "manufacturer"):
            idx = lowered.find(marker)
            if idx > 0:
                return topic[:idx].strip()
        return ""

    def _extract_time_constraint(self, topic: str) -> str:
        q_match = re.search(r"(20\d{2}\s*[-/]?\s*Q[1-4])", topic, flags=re.IGNORECASE)
        if q_match:
            return re.sub(r"\s+", "", q_match.group(1).upper())
        y_match = re.search(r"(20\d{2})", topic)
        if y_match:
            return y_match.group(1)
        return ""

    def _extract_metric(self, topic: str, research_context: dict, checkpoint_note: str = "") -> str:
        description = str(research_context.get("description") or "")
        key_points = " ".join(str(x or "") for x in (research_context.get("key_points") or []))
        source_text = f"{topic} {description} {key_points} {checkpoint_note}".lower()
        metric_candidates = [
            "revenue",
            "profit",
            "margin",
            "shipment",
            "market share",
            "guidance",
            "growth",
            "valuation",
            "营收",
            "利润",
            "增速",
            "份额",
        ]
        for metric in metric_candidates:
            if metric in source_text:
                return metric
        return ""

    def _extract_negative_constraints(self, topic: str) -> List[str]:
        negatives = []
        for year in ("2024", "2025"):
            if year not in topic:
                negatives.append(year)
        negatives.extend(["projection", "forecast"])
        return negatives

    def _constraint_guard(self, claims: dict, segments: List[str]) -> dict:
        corpus = " ".join(segments).lower()
        subject = str(claims.get("subject") or "").lower()
        time_constraint = str(claims.get("time_constraint") or "").lower()

        subject_present = bool(subject) and self._contains_term(corpus, subject)
        time_present = bool(time_constraint) and self._contains_term(corpus, time_constraint)

        hard_fail_reasons = []
        if subject and not subject_present:
            hard_fail_reasons.append("missing_subject")
        if time_constraint and not time_present:
            hard_fail_reasons.append("missing_time_constraint")

        suspicious_hit = any(token in corpus for token in self.SUSPICIOUS_TOKENS)
        actual_hit = any(token in corpus for token in self.ACTUAL_TOKENS)
        suspicious = suspicious_hit and not actual_hit

        density = self._entity_density(corpus, subject)
        return {
            "hard_fail": bool(hard_fail_reasons),
            "hard_fail_reasons": hard_fail_reasons,
            "subject_present": subject_present,
            "time_present": time_present,
            "entity_density": density,
            "suspicious": suspicious,
            "suspicious_reason": "projection_or_forecast_without_actual" if suspicious else "",
        }

    def _run_deep_eval(self, claims: dict, segments: List[str], precheck: dict, coverage_report: dict = None) -> dict:
        required_claims = [x for x in [claims.get("subject"), claims.get("time_constraint"), claims.get("metric")] if x]
        corpus = " ".join(segments).lower()
        verified_claims = 0
        failed_claims: List[str] = []

        for claim in required_claims:
            normalized = str(claim).lower()
            if self._contains_term(corpus, normalized):
                verified_claims += 1
            else:
                failed_claims.append(f"Claim not verified: {claim}")

        total_required = len(required_claims)
        if total_required == 0:
            # No structural claims extracted (e.g. pure Chinese topic with no year/metric).
            # Fall back to section_coverage so the agent doesn't hard-penalize valid content.
            section_cov = float((coverage_report or {}).get("section_coverage") or 0.0)
            final_score = round(section_cov, 4)
            if final_score == 0.0:
                failed_claims.append("No required claims extracted and coverage is zero.")
            else:
                failed_claims.append("No required claims extracted; score derived from section coverage.")
        else:
            final_score = verified_claims / total_required

        if precheck.get("hard_fail"):
            final_score = min(final_score, 0.69)
            failed_claims.extend(precheck.get("hard_fail_reasons") or [])

        if precheck.get("suspicious"):
            final_score = min(final_score, 0.69)
            failed_claims.append("Suspicious forecast/projection wording without actual or audited evidence.")

        final_score = max(0.0, min(1.0, round(final_score, 4)))
        verdict = "Pass" if final_score >= self.PASS_THRESHOLD and not precheck.get("hard_fail") else "Partial Match"

        return {
            "final_score": final_score,
            "verified_claims": verified_claims,
            "total_required_claims": total_required,
            "failed_claims": failed_claims,
            "verdict": verdict,
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
        claims: dict,
        precheck: dict,
        deep_eval_report: dict,
        coverage_report: dict,
    ) -> dict:
        failed_claims = deep_eval_report.get("failed_claims") or []
        missing_clauses = []

        if "missing_time_constraint" in (precheck.get("hard_fail_reasons") or []):
            missing_clauses.append("enforce the exact target year/quarter and exclude old years")
        if "missing_subject" in (precheck.get("hard_fail_reasons") or []):
            missing_clauses.append("focus strictly on the target entity")
        if precheck.get("suspicious"):
            missing_clauses.append("exclude projection/forecast language and require actual or audited values")
        if float(coverage_report.get("section_coverage") or 0.0) < self.COVERAGE_THRESHOLD:
            missing_clauses.append(
                f"increase section evidence coverage to at least {self.COVERAGE_THRESHOLD:.2f}"
            )

        subject = str(claims.get("subject") or "").strip()
        time_constraint = str(claims.get("time_constraint") or "").strip()
        metric = str(claims.get("metric") or "").strip()

        query_parts = [x for x in [subject, time_constraint, metric, "actual", "audited"] if x]
        query = " ".join(query_parts).strip()
        if "2026" in time_constraint:
            query = f'{query} -"2024" -"2025"'
        if precheck.get("suspicious"):
            query = f"{query} -projection -forecast"
        query = re.sub(r"\s+", " ", query).strip()

        if not query:
            query = f"{topic} actual audited report"

        if not missing_clauses and failed_claims:
            missing_clauses.append("address missing claims from the previous round")
        if not missing_clauses:
            missing_clauses.append("improve relevance and evidence quality")

        uncovered_queries = coverage_report.get("uncovered_queries") or []
        if uncovered_queries:
            query = f"{query} {' '.join(uncovered_queries[:2])}".strip()

        return {
            "instruction": "Previous results are insufficient: " + "; ".join(missing_clauses) + ".",
            "new_query_suggestion": query,
            "failed_claims": failed_claims,
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

    def _blocked_placeholder(self, topic: str, deep_eval_report: dict, feedback_packet: dict) -> str:
        score = deep_eval_report.get("final_score")
        suggestion = feedback_packet.get("new_query_suggestion")
        return (
            f"### {topic}\n\n"
            f"Insufficient evidence for this section, requires manual review (final_score={score}).\n"
            "System failed to meet the minimum confidence threshold (0.7) after 3 verification rounds.\n"
            f"Suggested additional search: {suggestion}\n"
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
            uncovered_key_points = [point for point in key_points if point.lower() not in corpus_text]
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
