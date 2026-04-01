"""Structured output writers for Auto_Research_Engine.

Produces three output file groups:
1. model_decisions.json + model_decisions.md  — LLM routing decision log
2. report.md                                  — confidence-tagged report with [S*] citations
3. evidence_base.md                           — standalone evidence database
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import aiofiles

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


# ── Model Decisions ──────────────────────────────────────────────────────


def format_model_decisions_json(decisions: List[Dict[str, Any]]) -> str:
    """Return pretty-printed JSON with per-call records + summary."""
    summary = _build_decisions_summary(decisions)
    payload = {"decisions": decisions, "summary": summary}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_model_decisions_md(decisions: List[Dict[str, Any]]) -> str:
    """Return a human-readable Markdown table of routing decisions."""
    lines: List[str] = ["# Model Routing Decisions\n"]

    lines.append(
        "| # | Agent Role | Stage | Selected Model | Provider "
        "| Reason | Candidates | Latency (ms) |"
    )
    lines.append(
        "|---|-----------|-------|----------------|----------"
        "|--------|------------|--------------|"
    )

    for idx, d in enumerate(decisions, 1):
        candidates_raw = d.get("candidates") or []
        candidate_names = ", ".join(
            str(c.get("model") or c) if isinstance(c, dict) else str(c)
            for c in candidates_raw
        )
        lines.append(
            f"| {idx} "
            f"| {d.get('agent_role', '')} "
            f"| {d.get('stage_name', '')} "
            f"| {d.get('selected_model', '')} "
            f"| {d.get('selected_provider', '')} "
            f"| {d.get('routing_reason', '')} "
            f"| {candidate_names} "
            f"| {d.get('route_latency_ms', '')} |"
        )

    # Append summary
    summary = _build_decisions_summary(decisions)
    lines.append("")
    lines.append("## Summary\n")
    lines.append(f"- **Total calls**: {summary['total_calls']}")
    for model, count in sorted(summary["models_used"].items(), key=lambda x: -x[1]):
        lines.append(f"  - `{model}`: {count}")
    lines.append(f"- **Total route latency**: {summary['total_route_latency_ms']:.1f} ms")
    lines.append(f"- **Fallback count**: {summary['fallback_count']}")

    return "\n".join(lines)


def _build_decisions_summary(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    model_counter: Counter[str] = Counter()
    total_latency = 0.0
    fallback_count = 0

    for d in decisions:
        model = str(d.get("selected_model") or "unknown")
        model_counter[model] += 1
        try:
            total_latency += float(d.get("route_latency_ms") or 0)
        except (TypeError, ValueError):
            pass
        if d.get("type") == "quota_fallback":
            fallback_count += 1

    return {
        "total_calls": len(decisions),
        "models_used": dict(model_counter),
        "total_route_latency_ms": round(total_latency, 2),
        "fallback_count": fallback_count,
    }


# ── Evidence Base ────────────────────────────────────────────────────────


def format_evidence_base_md(
    source_index: Dict[str, Any],
    cited_ids: Optional[set[str]] = None,
) -> str:
    """Render every evidence entry as a Markdown section.

    If *cited_ids* is provided, each heading shows the citation count
    or '(unused)' for entries not referenced by the report.
    """
    if not source_index:
        return "# Evidence Base\n\n_No evidence collected for this run._\n"

    lines: List[str] = ["# Evidence Base\n"]

    for key in sorted(source_index, key=_source_sort_key):
        info = source_index[key]
        if not isinstance(info, dict):
            continue

        # Citation frequency annotation
        tag = ""
        if cited_ids is not None:
            count = _count_citations_for_id(key, cited_ids)
            tag = f" (cited {count}×)" if count > 0 else " (unused)"

        lines.append(f"### {key}{tag}")
        lines.append(f"- **URL**: {info.get('source_url', '')}")
        lines.append(f"- **Domain**: {info.get('domain', '')}")
        section_title = info.get("section_title") or info.get("section_key") or ""
        if section_title:
            lines.append(f"- **Section**: {section_title}")
        lines.append(f"- **Content**: {info.get('content', '')}")
        lines.append("")

    return "\n".join(lines)


def _count_citations_for_id(source_id: str, cited_ids: set[str]) -> int:
    """Return how many times *source_id* appears in the cited set.

    cited_ids can be a Counter for accurate frequency counting.
    """
    if isinstance(cited_ids, Counter):
        return cited_ids.get(source_id, 0)
    return 1 if source_id in cited_ids else 0


# ── Annotated Report ─────────────────────────────────────────────────────


def format_annotated_report_md(
    report_layout: str,
    claim_report: List[Dict[str, Any]],
    source_index: Dict[str, Any],
) -> str:
    """Insert [HIGH]/[MEDIUM]/... tags before cited sentences, keep [S*] refs.

    Also appends a citation statistics summary at the end.
    """
    if not report_layout:
        return ""

    annotated = report_layout

    if claim_report:
        # Build a map: original_sentence -> confidence
        sentence_map = _build_sentence_confidence_map(claim_report)
        # Apply tags — process longer matches first to avoid partial overlaps
        for sentence, confidence in sorted(
            sentence_map.items(), key=lambda x: -len(x[0])
        ):
            if not sentence:
                continue
            tagged = f"[{confidence}] {sentence}"
            annotated = annotated.replace(sentence, tagged, 1)

    # Append citation statistics
    stats = _build_citation_statistics(claim_report, source_index, annotated)
    annotated += "\n\n" + stats

    return annotated


def _build_sentence_confidence_map(
    claim_report: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Map original_sentence -> confidence, first-write wins."""
    result: Dict[str, str] = {}
    for claim in claim_report:
        sentence = str(claim.get("original_sentence") or "").strip()
        confidence = str(claim.get("confidence") or "").strip()
        if sentence and confidence and sentence not in result:
            result[sentence] = confidence
    return result


def _build_citation_statistics(
    claim_report: List[Dict[str, Any]],
    source_index: Dict[str, Any],
    annotated_text: str,
) -> str:
    """Generate a Citation Statistics section for the report footer."""
    total_claims = len(claim_report)
    if total_claims == 0:
        return "## Citation Statistics\n\n_No claims verified._\n"

    counts: Counter[str] = Counter()
    for c in claim_report:
        counts[str(c.get("confidence") or "UNKNOWN")] += 1

    # Count actually cited source IDs in annotated text
    cited_ids = set(f"S{m}" for m in CITATION_PATTERN.findall(annotated_text))
    total_sources = len(source_index)
    cited_count = len(cited_ids & set(source_index.keys()))

    lines = ["## Citation Statistics\n"]
    lines.append(f"- **Total claims**: {total_claims}")

    parts = []
    for level in ("HIGH", "MEDIUM", "SUSPICIOUS", "HALLUCINATION"):
        c = counts.get(level, 0)
        pct = (c / total_claims * 100) if total_claims else 0
        parts.append(f"{level}: {c} ({pct:.1f}%)")
    lines.append(f"- {' | '.join(parts)}")

    if total_sources > 0:
        usage_pct = cited_count / total_sources * 100
        lines.append(
            f"- **Evidence sources cited**: {cited_count} / {total_sources} ({usage_pct:.1f}%)"
        )
    else:
        lines.append("- **Evidence sources cited**: 0 / 0")

    return "\n".join(lines) + "\n"


def collect_cited_ids_from_text(text: str) -> Counter[str]:
    """Extract [S*] citation IDs from text with frequency counts."""
    return Counter(f"S{m}" for m in CITATION_PATTERN.findall(text))


# ── Async File Writers ───────────────────────────────────────────────────


async def write_model_decisions(output_dir: str, decisions: List[Dict[str, Any]]) -> None:
    """Write model_decisions.json and model_decisions.md to output_dir."""
    json_path = f"{output_dir}/model_decisions.json"
    md_path = f"{output_dir}/model_decisions.md"

    json_content = format_model_decisions_json(decisions)
    md_content = format_model_decisions_md(decisions)

    async with aiofiles.open(json_path, "w", encoding="utf-8") as f:
        await f.write(json_content)
    async with aiofiles.open(md_path, "w", encoding="utf-8") as f:
        await f.write(md_content)


async def write_evidence_base(
    output_dir: str,
    source_index: Dict[str, Any],
    cited_ids: Optional[set[str]] = None,
) -> None:
    """Write evidence_base.md to output_dir."""
    path = f"{output_dir}/evidence_base.md"
    content = format_evidence_base_md(source_index, cited_ids)
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)


async def write_annotated_report(
    output_dir: str,
    report_layout: str,
    claim_report: List[Dict[str, Any]],
    source_index: Dict[str, Any],
) -> None:
    """Write report.md (confidence-tagged, citation-bearing) to output_dir."""
    path = f"{output_dir}/report.md"
    content = format_annotated_report_md(report_layout, claim_report, source_index)
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)


# ── Utilities ────────────────────────────────────────────────────────────


def _source_sort_key(key: str) -> int:
    if key.startswith("S"):
        try:
            return int(key[1:])
        except ValueError:
            return 0
    return 0