import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from multi_agents.route_agent import build_route_context
from .utils.llms import call_model
from .utils.views import print_agent_output


class ClaimVerifierAgent:
    """Verifies writer claims against evidence sources and assigns confidence levels.

    Confidence levels:
    - HIGH: 2+ distinct source domains support the claim
    - MEDIUM: 1 source domain supports the claim
    - SUSPICIOUS: 2+ domains with conflicting content
    - HALLUCINATION: No matching evidence in any source
    """

    MAX_REFLEXION = 3
    CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
    FALLBACK_THRESHOLD = 0.5  # trigger fallback if < 50% sentences have citations

    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.logger = logging.getLogger(__name__)

    # ── Source Index ──────────────────────────────────────────────────────

    def build_source_index(
        self,
        scraping_packets: List[dict],
        start_id: int = 1,
        section_contexts: Optional[List[dict]] = None,
        existing_index: Optional[dict] = None,
    ) -> Tuple[dict, str]:
        """Assign [S1], [S2]... to each evidence passage.

        Args:
            scraping_packets: List of scraping packet dicts with search_log.
            start_id: Starting ID number (for append-only rebuilds).
            section_contexts: Optional section metadata aligned with scraping_packets.
            existing_index: Existing source index used to skip duplicate append-only entries.

        Returns:
            (source_index, formatted_string) where source_index maps
            "S1" -> {content, source_url, domain} and formatted_string
            is the indexed research data for the writer prompt.
        """
        index: Dict[str, dict] = {}
        current_id = start_id
        seen_fingerprints = self._collect_existing_fingerprints(existing_index)
        normalized_contexts = list(section_contexts or [])

        for packet_idx, packet in enumerate(scraping_packets or []):
            if not isinstance(packet, dict):
                continue
            section_context = (
                normalized_contexts[packet_idx]
                if packet_idx < len(normalized_contexts) and isinstance(normalized_contexts[packet_idx], dict)
                else {}
            )
            section_key = str(section_context.get("section_key") or "").strip()
            section_title = str(section_context.get("section_title") or "").strip()
            section_index = section_context.get("section_index")
            for log_entry in packet.get("search_log", []):
                if not isinstance(log_entry, dict):
                    continue
                for passage in log_entry.get("top_10_passages", []):
                    if not isinstance(passage, dict):
                        continue
                    content = str(passage.get("content") or "").strip()
                    source_url = str(passage.get("source_url") or "").strip()
                    if not content:
                        continue

                    fingerprint = self._build_passage_fingerprint(
                        content=content,
                        source_url=source_url,
                        section_key=section_key,
                    )
                    if fingerprint in seen_fingerprints:
                        continue
                    seen_fingerprints.add(fingerprint)

                    domain = self._extract_domain(source_url)
                    key = f"S{current_id}"
                    index[key] = {
                        "content": content,
                        "source_url": source_url,
                        "domain": domain,
                        "section_key": section_key,
                        "section_title": section_title,
                        "section_index": section_index,
                    }
                    current_id += 1

        return index, self.format_source_index(index)

    def format_source_index(self, source_index: dict) -> str:
        """Render a source index into the writer-facing indexed evidence format."""
        if not isinstance(source_index, dict) or not source_index:
            return ""

        lines: List[str] = []
        for key, info in sorted(source_index.items(), key=self._source_sort_key):
            if not isinstance(info, dict):
                continue
            domain = str(info.get("domain") or "").strip()
            content = str(info.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"[{key}]({domain}) {content}")
        return "\n\n".join(lines)

    # ── Citation Parsing ─────────────────────────────────────────────────

    def parse_citations(
        self, writer_output: dict, source_index: dict
    ) -> List[dict]:
        """Parse [S*] citations from writer introduction + conclusion.

        Returns list of claim dicts with source_ids and domains resolved.
        """
        claims: List[dict] = []

        # Structured claim_annotations are useful, but not sufficient on their own.
        # Merge them with inline parsing so omitted annotations do not bypass verification.
        annotations = writer_output.get("claim_annotations")
        if isinstance(annotations, list) and annotations:
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                source_ids = ann.get("source_ids", [])
                domains = self._resolve_domains(source_ids, source_index)
                claims.append({
                    "claim_text": str(ann.get("sentence") or "").strip(),
                    "source_section": str(ann.get("section") or "").strip(),
                    "source_ids": source_ids,
                    "domains": domains,
                    "original_sentence": str(ann.get("sentence") or "").strip(),
                })

        # Parse inline [S*] from the rendered text as well.
        for section_name in ("introduction", "conclusion"):
            text = str(writer_output.get(section_name) or "")
            if not text.strip():
                continue
            section_claims = self._extract_inline_citations(text, section_name, source_index)
            claims.extend(section_claims)

        return self._merge_claim_lists(claims)

    def _extract_inline_citations(
        self, text: str, section_name: str, source_index: dict
    ) -> List[dict]:
        """Extract claims with [S*] citations from a text block."""
        claims: List[dict] = []
        sentences = self._split_sentences(text)

        for sentence in sentences:
            raw = sentence.strip()
            if not raw:
                continue
            ids_found = self.CITATION_PATTERN.findall(raw)
            source_ids = [f"S{sid}" for sid in ids_found]
            # Clean sentence text (remove citation markers)
            clean_text = self.CITATION_PATTERN.sub("", raw).strip()
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            clean_text = re.sub(r"\s+([,.;:!?])", r"\1", clean_text)
            clean_text = clean_text.rstrip(" .;:!?")
            if not clean_text:
                continue

            domains = self._resolve_domains(source_ids, source_index)
            claims.append({
                "claim_text": clean_text,
                "source_section": section_name,
                "source_ids": source_ids,
                "domains": domains,
                "original_sentence": raw,
            })

        return claims

    def _check_citation_coverage(self, claims: List[dict]) -> float:
        """Return fraction of claims that have at least one citation."""
        if not claims:
            return 0.0
        cited = sum(1 for c in claims if c.get("source_ids"))
        return cited / len(claims)

    # ── Fallback: Post-hoc LLM Matching ─────────────────────────────────

    async def fallback_match_claims(
        self, writer_output: dict, source_index: dict, model: str
    ) -> List[dict]:
        """LLM-based post-hoc matching when writer didn't cite [S*] properly."""
        intro = str(writer_output.get("introduction") or "")
        conclusion = str(writer_output.get("conclusion") or "")
        combined_text = f"Introduction:\n{intro}\n\nConclusion:\n{conclusion}"

        # Build a condensed source list for the prompt
        source_summary = []
        for key, info in sorted(source_index.items(), key=lambda x: int(x[0][1:])):
            snippet = info["content"][:200]
            source_summary.append(f"[{key}]({info['domain']}) {snippet}")
        sources_text = "\n".join(source_summary[:50])  # limit to 50 sources

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a fact-checking assistant. Your task is to match "
                    "factual claims in the text to their supporting evidence sources."
                ),
            },
            {
                "role": "user",
                "content": f"""Analyze the following text and identify all factual claims.
For each claim, determine which source(s) from the evidence list support it.

Text:
{combined_text}

Evidence Sources:
{sources_text}

Return a JSON array where each element has:
- "sentence": the original sentence containing the claim
- "claim_text": the specific factual claim
- "source_ids": list of matching source IDs (e.g., ["S1", "S3"]), or [] if no match
- "section": "introduction" or "conclusion"

Return ONLY the JSON array, no markdown fences.""",
            },
        ]

        route_context = build_route_context(
            application_name=str(writer_output.get("application_name") or "auto_research_engine"),
            shared_agent_class="claim_verifier_agent",
            agent_role="claim_verifier",
            stage_name="fallback_claim_matching",
            system_prompt="You are a fact-checking assistant.",
            task=combined_text,
            extra={},
        )
        result = await call_model(prompt, model=model, response_format="json", route_context=route_context)
        if not isinstance(result, list):
            return []

        claims: List[dict] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            source_ids = item.get("source_ids", [])
            if not isinstance(source_ids, list):
                source_ids = []
            # Validate source IDs exist
            valid_ids = [sid for sid in source_ids if sid in source_index]
            domains = self._resolve_domains(valid_ids, source_index)
            claims.append({
                "claim_text": str(item.get("claim_text") or item.get("sentence") or "").strip(),
                "source_section": str(item.get("section") or "").strip(),
                "source_ids": valid_ids,
                "domains": domains,
                "original_sentence": str(item.get("sentence") or "").strip(),
            })

        return claims

    # ── Conflict Detection ───────────────────────────────────────────────

    async def detect_conflicts(
        self, claim: dict, source_index: dict, model: str
    ) -> dict:
        """For claims with 2+ domains, check if sources contradict each other."""
        source_ids = claim.get("source_ids", [])
        if len(set(claim.get("domains", []))) < 2:
            return {"has_conflict": False, "conflict_detail": ""}

        # Gather passages from different domains
        passages_by_domain: Dict[str, List[str]] = {}
        for sid in source_ids:
            info = source_index.get(sid)
            if not info:
                continue
            domain = info["domain"]
            passages_by_domain.setdefault(domain, []).append(
                info["content"][:300]
            )

        if len(passages_by_domain) < 2:
            return {"has_conflict": False, "conflict_detail": ""}

        passages_text = ""
        for domain, texts in passages_by_domain.items():
            for text in texts:
                passages_text += f"[{domain}]: {text}\n\n"

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a fact-checking assistant that detects contradictions "
                    "between different sources about the same claim."
                ),
            },
            {
                "role": "user",
                "content": f"""Claim: "{claim['claim_text']}"

Sources:
{passages_text}

Do these sources contradict each other about this claim?
Return JSON with:
- "has_conflict": true/false
- "conflict_detail": brief description of the contradiction (or "" if no conflict)

Return ONLY the JSON, no markdown fences.""",
            },
        ]

        route_context = build_route_context(
            application_name="auto_research_engine",
            shared_agent_class="claim_verifier_agent",
            agent_role="claim_verifier",
            stage_name="conflict_detection",
            system_prompt="You are a fact-checking assistant that detects contradictions between sources.",
            task=str(claim.get("claim_text") or ""),
        )
        result = await call_model(prompt, model=model, response_format="json", route_context=route_context)
        if not isinstance(result, dict):
            return {"has_conflict": False, "conflict_detail": ""}

        return {
            "has_conflict": bool(result.get("has_conflict")),
            "conflict_detail": str(result.get("conflict_detail") or ""),
        }

    # ── Classification ───────────────────────────────────────────────────

    async def classify_claims(
        self, parsed_claims: List[dict], source_index: dict, model: str
    ) -> List[dict]:
        """Assign confidence level to each claim."""
        classified: List[dict] = []

        for claim in parsed_claims:
            source_ids = claim.get("source_ids", [])
            domains = set(claim.get("domains", []))
            valid_ids = [sid for sid in source_ids if sid in source_index]

            if not valid_ids:
                confidence = "HALLUCINATION"
                note = "No source support"
            elif len(domains) == 1:
                confidence = "MEDIUM"
                note = f"Only {list(domains)[0]}"
            else:
                # Check for conflicts
                conflict = await self.detect_conflicts(claim, source_index, model)
                if conflict["has_conflict"]:
                    confidence = "SUSPICIOUS"
                    note = conflict["conflict_detail"]
                else:
                    confidence = "HIGH"
                    note = ""

            classified.append({
                **claim,
                "confidence": confidence,
                "note": note,
                "source_urls": [
                    source_index[sid]["source_url"]
                    for sid in valid_ids
                    if sid in source_index
                ],
            })

        return classified

    # ── Reflexion Helpers ────────────────────────────────────────────────

    def group_by_section(
        self,
        suspicious_claims: List[dict],
        source_index: Optional[dict] = None,
    ) -> Dict[str, List[dict]]:
        """Group suspicious claims by section_key for parallel checkpoint rerun."""
        groups: Dict[str, List[dict]] = {}
        for claim in suspicious_claims:
            section_keys = self._resolve_section_keys_for_claim(claim, source_index or {})
            if not section_keys:
                fallback_section = str(claim.get("source_section") or "").strip()
                if fallback_section:
                    section_keys = [fallback_section]
            for section_key in section_keys:
                groups.setdefault(section_key, []).append(claim)
        return groups

    def build_reflexion_note(self, claims: List[dict], source_index: dict) -> str:
        """Build a note string with conflict excerpts + disambiguation queries."""
        parts: List[str] = []
        parts.append("The following claims have source conflicts requiring additional searches to resolve contradictions:\n")

        for i, claim in enumerate(claims, 1):
            parts.append(f"{i}. Claim: \"{claim['claim_text']}\"")
            parts.append(f"   Conflict detail: {claim.get('note', '')}")

            # Include conflicting excerpts
            for sid in claim.get("source_ids", [])[:4]:
                info = source_index.get(sid)
                if info:
                    parts.append(f"   [{sid}]({info['domain']}): {info['content'][:150]}...")

            parts.append(f"   Suggested search: \"{claim['claim_text']}\" actual verified official")
            parts.append("")

        return "\n".join(parts)

    # ── Draft Annotation ─────────────────────────────────────────────────

    def annotate_draft(self, draft: str, claim_report: List[dict]) -> str:
        """Add inline confidence tags + append summary table to draft.

        Called AFTER review cycle, before publisher.
        """
        if not draft or not claim_report:
            return draft

        annotated = draft

        # Apply annotations in reverse order of appearance to preserve positions
        for claim in reversed(claim_report):
            confidence = claim.get("confidence", "")
            original = claim.get("original_sentence", "")
            if not original:
                continue

            # Strip [S*] citations from the text for clean output
            clean_original = self.CITATION_PATTERN.sub("", original).strip()
            clean_original = re.sub(r"\s+", " ", clean_original)

            if confidence == "HALLUCINATION":
                replacement = "[Knowledge unavailable — no reliable source supports this claim]"
            elif confidence == "SUSPICIOUS":
                replacement = f"{clean_original} [Source conflict: different sources contradict each other on this point]"
            elif confidence == "MEDIUM":
                replacement = f"{clean_original} [Single source]"
            else:
                replacement = clean_original

            # Replace original sentence (with citations) in the draft
            annotated = annotated.replace(original, replacement, 1)

        # Also clean any remaining [S*] citations not covered by claims
        annotated = self.CITATION_PATTERN.sub("", annotated)
        annotated = re.sub(r"\s{2,}", " ", annotated)

        # Append summary table
        annotated += self._build_summary_table(claim_report)

        return annotated

    def _build_summary_table(self, claim_report: List[dict]) -> str:
        """Build the confidence summary table in markdown."""
        if not claim_report:
            return ""

        confidence_map = {
            "HIGH": "High",
            "MEDIUM": "Medium",
            "SUSPICIOUS": "Suspicious",
            "HALLUCINATION": "Hallucination",
        }

        rows: List[str] = []
        counts = {"HIGH": 0, "MEDIUM": 0, "SUSPICIOUS": 0, "HALLUCINATION": 0}

        for i, claim in enumerate(claim_report, 1):
            conf = claim.get("confidence", "")
            counts[conf] = counts.get(conf, 0) + 1
            domains = claim.get("domains", [])
            domain_str = ", ".join(sorted(set(domains))) if domains else "—"
            note = claim.get("note", "") or "—"
            claim_text = claim.get("claim_text", "")[:60]
            rows.append(
                f"| {i} | {claim_text} | {confidence_map.get(conf, conf)} "
                f"| {len(set(domains))} | {domain_str} | {note} |"
            )

        table = "\n\n## Claim Confidence Report\n\n"
        table += "| # | Claim Summary | Confidence | Supporting Sources | Domain | Notes |\n"
        table += "|---|---------------|------------|-------------------|--------|-------|\n"
        table += "\n".join(rows)
        table += f"\n\nStats: High {counts['HIGH']} | Medium {counts['MEDIUM']}"
        table += f" | Suspicious {counts['SUSPICIOUS']} | Hallucination {counts['HALLUCINATION']}"
        reflexion_note = ""
        table += reflexion_note
        table += "\n"

        return table

    # ── Annotated Report (academic-style) ────────────────────────────

    def build_annotated_report_text(self, draft: str, claim_report: List[dict]) -> str:
        """Build report text with [HIGH]/[MEDIUM]/... tags, preserving [S*] citations.

        Unlike annotate_draft() which strips citations for end-user display,
        this method keeps [S*] references intact for academic-style output.
        """
        if not draft or not claim_report:
            return draft or ""

        result = draft
        # Map original_sentence -> confidence (first-write wins)
        sentence_map: Dict[str, str] = {}
        for claim in claim_report:
            sentence = str(claim.get("original_sentence") or "").strip()
            confidence = str(claim.get("confidence") or "").strip()
            if sentence and confidence and sentence not in sentence_map:
                sentence_map[sentence] = confidence

        # Apply longer matches first to avoid partial overlaps
        for sentence, confidence in sorted(
            sentence_map.items(), key=lambda x: -len(x[0])
        ):
            tagged = f"[{confidence}] {sentence}"
            result = result.replace(sentence, tagged, 1)

        return result

    # ── Main Entry ───────────────────────────────────────────────────────

    async def run(self, research_state: dict) -> dict:
        """Extract → trace → classify claims from writer output.

        Returns dict with claim_confidence_report, suspicious_claims,
        hallucinated_claims.
        """
        task = research_state.get("task") or {}
        model = task.get("model")
        source_index = research_state.get("source_index") or {}

        writer_output = {
            "introduction": research_state.get("introduction") or "",
            "conclusion": research_state.get("conclusion") or "",
            "claim_annotations": research_state.get("claim_annotations"),
        }

        await self._emit_log(
            "claim_verify_start",
            "Extracting and verifying claims from writer output...",
        )

        # Step 1: Parse citations
        claims = self.parse_citations(writer_output, source_index)

        # Step 2: Check coverage — fallback to LLM matching if needed
        coverage = self._check_citation_coverage(claims)
        if coverage < self.FALLBACK_THRESHOLD:
            await self._emit_log(
                "claim_verify_fallback",
                f"Citation coverage {coverage:.0%} below threshold, using LLM matching...",
            )
            fallback_claims = await self.fallback_match_claims(writer_output, source_index, model)
            if fallback_claims:
                claims = self._merge_claim_lists(claims, fallback_claims)
            else:
                await self._emit_log(
                    "claim_verify_fallback_failed",
                    "Fallback matching returned no usable claims; keeping the parsed claims for conservative verification.",
                )

        # Step 3: Classify all claims
        classified = await self.classify_claims(claims, source_index, model)

        suspicious = [c for c in classified if c["confidence"] == "SUSPICIOUS"]
        hallucinated = [c for c in classified if c["confidence"] == "HALLUCINATION"]

        await self._emit_log(
            "claim_verify_done",
            f"Verified {len(classified)} claims: "
            f"{len(classified) - len(suspicious) - len(hallucinated)} clean, "
            f"{len(suspicious)} suspicious, {len(hallucinated)} hallucinated.",
        )

        return {
            "claim_confidence_report": classified,
            "suspicious_claims": suspicious,
            "hallucinated_claims": hallucinated,
        }

    # ── Utilities ────────────────────────────────────────────────────────

    def _extract_domain(self, url: str) -> str:
        """Extract domain from a URL."""
        if not url:
            return ""
        try:
            parsed = urlparse(url if "://" in url else f"https://{url}")
            domain = parsed.netloc or parsed.path.split("/")[0]
            return domain.lower().removeprefix("www.")
        except Exception:
            return ""

    def _resolve_domains(self, source_ids: List[str], source_index: dict) -> List[str]:
        """Resolve source IDs to their domains."""
        domains: List[str] = []
        for sid in source_ids:
            info = source_index.get(sid)
            if info and info.get("domain"):
                domains.append(info["domain"])
        return domains

    def _merge_claim_lists(self, *claim_lists: List[dict]) -> List[dict]:
        merged: Dict[Tuple[str, str], dict] = {}

        for claim_list in claim_lists:
            for claim in claim_list or []:
                if not isinstance(claim, dict):
                    continue
                claim_text = str(claim.get("claim_text") or "").strip()
                source_section = str(claim.get("source_section") or "").strip()
                if not claim_text:
                    continue

                key = (self._normalize_claim_key(claim_text), source_section.lower())
                source_ids = self._dedupe_strings(claim.get("source_ids") or [])
                domains = self._dedupe_strings(claim.get("domains") or [])
                original_sentence = str(claim.get("original_sentence") or "").strip() or claim_text

                existing = merged.get(key)
                if existing is None:
                    merged[key] = {
                        "claim_text": claim_text,
                        "source_section": source_section,
                        "source_ids": source_ids,
                        "domains": domains,
                        "original_sentence": original_sentence,
                    }
                    continue

                existing["source_ids"] = self._dedupe_strings(existing.get("source_ids", []) + source_ids)
                existing["domains"] = self._dedupe_strings(existing.get("domains", []) + domains)
                if len(original_sentence) > len(str(existing.get("original_sentence") or "")):
                    existing["original_sentence"] = original_sentence
                if not existing.get("claim_text"):
                    existing["claim_text"] = claim_text
                if not existing.get("source_section"):
                    existing["source_section"] = source_section

        return list(merged.values())

    @staticmethod
    def _dedupe_strings(values: List[str]) -> List[str]:
        deduped: List[str] = []
        seen: Set[str] = set()
        for value in values or []:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    @staticmethod
    def _normalize_claim_key(value: str) -> str:
        normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
        return normalized.rstrip(" .;:!?")

    def _resolve_section_keys_for_claim(self, claim: dict, source_index: dict) -> List[str]:
        section_keys: List[str] = []
        seen: Set[str] = set()
        for sid in claim.get("source_ids", []) or []:
            info = source_index.get(sid)
            if not isinstance(info, dict):
                continue
            section_key = str(info.get("section_key") or "").strip()
            if not section_key or section_key in seen:
                continue
            seen.add(section_key)
            section_keys.append(section_key)
        return section_keys

    def _collect_existing_fingerprints(self, source_index: Optional[dict]) -> Set[str]:
        fingerprints: Set[str] = set()
        if not isinstance(source_index, dict):
            return fingerprints
        for info in source_index.values():
            if not isinstance(info, dict):
                continue
            content = str(info.get("content") or "").strip()
            source_url = str(info.get("source_url") or "").strip()
            section_key = str(info.get("section_key") or "").strip()
            if not content:
                continue
            fingerprints.add(
                self._build_passage_fingerprint(
                    content=content,
                    source_url=source_url,
                    section_key=section_key,
                )
            )
        return fingerprints

    def _build_passage_fingerprint(self, content: str, source_url: str, section_key: str) -> str:
        normalized_content = re.sub(r"\s+", " ", str(content or "").strip().lower())
        normalized_url = str(source_url or "").strip().lower()
        normalized_section = str(section_key or "").strip().lower()
        return "||".join([normalized_section, normalized_url, normalized_content])

    def _source_sort_key(self, item: Tuple[str, Any]) -> int:
        source_id = str(item[0] or "")
        if source_id.startswith("S"):
            try:
                return int(source_id[1:])
            except ValueError:
                return 0
        return 0

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving citation markers."""
        # Split on period, question mark, exclamation mark followed by space or EOL
        # But not on periods in URLs or abbreviations
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    async def _emit_log(self, event: str, message: str) -> None:
        """Emit a log message via websocket or print."""
        if self.websocket and self.stream_output:
            await self.stream_output("logs", event, message, self.websocket)
        else:
            try:
                print_agent_output(message, agent="CLAIM_VERIFIER")
            except KeyError:
                print(message)
        self.logger.info(message)
