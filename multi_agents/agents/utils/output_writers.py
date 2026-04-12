"""Structured output writers for Auto_Research_Engine.

Produces three output file groups:
1. model_decisions.json + model_decisions.md  — LLM routing decision log
2. report.md                                  — confidence-tagged report with [S*] citations
3. evidence_base.md                           — standalone evidence database
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import aiofiles

CITATION_PATTERN = re.compile(r"\[(\d+\.\d+)\]")


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
    """Extract [chapter.index] citation IDs from text with frequency counts."""
    return Counter(m for m in CITATION_PATTERN.findall(text))


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


def _source_sort_key(key: str) -> tuple:
    parts = key.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (0, 0)


# ── Writer Draft Snapshot ────────────────────────────────────────────────


def format_writer_draft_md(
    layout: str,
    claim_annotations: List[Dict[str, Any]],
) -> str:
    """Format initial writer draft with claim annotations summary."""
    lines = [
        "# Writer Initial Draft\n",
        "_Snapshot taken immediately after writer node, before review/revision._\n",
        layout,
    ]

    if claim_annotations:
        cited = [c for c in claim_annotations if c.get("source_ids")]
        uncited = [c for c in claim_annotations if not c.get("source_ids")]
        lines.append("\n\n---\n\n## Claim Annotations\n")
        lines.append(
            f"**Total**: {len(claim_annotations)} | "
            f"**Cited**: {len(cited)} | "
            f"**Uncited**: {len(uncited)}\n"
        )
        for item in claim_annotations:
            ids = "".join(f"[{s}]" for s in (item.get("source_ids") or []))
            section = item.get("section") or ""
            sentence = str(item.get("sentence") or "").strip()
            tag = ids if ids else "[no source]"
            lines.append(f"- {tag} _{section}_ — {sentence[:120]}")

    return "\n".join(lines)


async def write_writer_draft_snapshot(
    output_dir: str,
    layout: str,
    claim_annotations: List[Dict[str, Any]],
) -> None:
    """Write drafts/writer_initial_draft.md to output_dir."""
    drafts_dir = os.path.join(output_dir, "drafts")
    os.makedirs(drafts_dir, exist_ok=True)
    path = os.path.join(drafts_dir, "writer_initial_draft.md")
    content = format_writer_draft_md(layout, claim_annotations)
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)



# ── Section Draft Files ──────────────────────────────────────────────────


def _extract_section_body_text(research_data_item: Any) -> str:
    """Extract plain body text from a research_data dict item."""
    if isinstance(research_data_item, dict):
        return "\n\n".join(str(v) for v in research_data_item.values() if v)
    return str(research_data_item or "")


def extract_section_summary_fallback(section_body: str) -> str:
    """Fallback: skip ### header line, return first non-empty paragraph."""
    paragraphs = re.split(r"\n{2,}", section_body.strip())
    for para in paragraphs:
        stripped = para.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:400]
    return ""


def format_section_content_draft(
    section_title: str,
    section_body: str,
) -> str:
    """Render content.md for a section draft."""
    return f"# {section_title}\n\n{section_body}\n"


def format_section_summary_draft(section_title: str, summary: str) -> str:
    """Render summary.md for a section draft."""
    return f"# Summary: {section_title}\n\n{summary}\n"


def format_section_evidence_draft(
    chapter_num: int,
    section_title: str,
    section_evidence: List[dict],
) -> str:
    """Render evidence.md with [chapter.index] citation labels.

    Uses entry['global_id'] as the label when available (set by remap step).
    Falls back to chapter_num.idx for entries that were not remapped.
    """
    if not section_evidence:
        return f"# Evidence: {section_title}\n\n_No evidence collected for this section._\n"
    lines = [f"# Evidence: {section_title}\n"]

    def _ev_sort_key(e: dict) -> tuple:
        gid = str(e.get("global_id") or "")
        parts = gid.split(".")
        try:
            return (int(parts[0]), int(parts[1]))
        except (IndexError, ValueError):
            return (9999, 9999)

    sorted_evidence = sorted(section_evidence, key=_ev_sort_key)
    for idx, entry in enumerate(sorted_evidence, start=1):
        if not isinstance(entry, dict):
            continue
        used = entry.get("used_in_draft", False)
        source_url = entry.get("source_url") or ""
        domain = entry.get("domain") or ""
        content = entry.get("content") or ""
        # 优先用 remap 后写入的 global_id，保证与 content.md 引用一致
        global_id = str(entry.get("global_id") or "").strip()
        label = global_id if global_id else f"{chapter_num}.{idx}"
        lines.append(f"### [{label}] ({'cited' if used else 'unused'})")
        lines.append(f"- **URL**: {source_url}")
        lines.append(f"- **Domain**: {domain}")
        lines.append(f"- **Content**: {content}")
        lines.append("")
    return "\n".join(lines)


def _merge_evidence(existing: List[dict], incoming: List[dict]) -> List[dict]:
    """Merge incoming evidence into existing, deduplicating by URL+content fingerprint."""
    seen: set = set()
    merged: List[dict] = []
    for entry in existing:
        if not isinstance(entry, dict):
            continue
        fp = f"{entry.get('source_url', '')}||{entry.get('content', '')}"
        if fp not in seen:
            seen.add(fp)
            merged.append(entry)
    for entry in incoming:
        if not isinstance(entry, dict):
            continue
        fp = f"{entry.get('source_url', '')}||{entry.get('content', '')}"
        if fp not in seen:
            seen.add(fp)
            merged.append(entry)
    return merged


def _parse_evidence_from_md(md_text: str) -> List[dict]:
    """Parse existing evidence.md back into a list of dicts (URL + content only)."""
    entries: List[dict] = []
    current: Dict[str, str] = {}
    for line in md_text.splitlines():
        if line.startswith("### ["):
            if current:
                entries.append(current)
            current = {}
        elif line.startswith("- **URL**:"):
            current["source_url"] = line[len("- **URL**:"):].strip()
        elif line.startswith("- **Domain**:"):
            current["domain"] = line[len("- **Domain**:"):].strip()
        elif line.startswith("- **Content**:"):
            current["content"] = line[len("- **Content**:"):].strip()
    if current:
        entries.append(current)
    return entries


def _resolve_section_dir(output_dir: str, section_key: str) -> str:
    """Return existing section dir with same index prefix, or canonical path if none found."""
    drafts_dir = os.path.join(output_dir, "drafts")
    canonical = os.path.join(drafts_dir, section_key)
    if os.path.isdir(canonical):
        return canonical
    m = re.match(r"^(section_\d+)_", section_key)
    if m and os.path.isdir(drafts_dir):
        prefix = m.group(1) + "_"
        for name in sorted(os.listdir(drafts_dir)):
            if name.startswith(prefix) and os.path.isdir(os.path.join(drafts_dir, name)):
                return os.path.join(drafts_dir, name)
    return canonical


async def write_section_draft_files(
    output_dir: str,
    chapter_num: int,
    section_key: str,
    section_title: str,
    section_body: str,
    summary: str,
    section_evidence: Optional[List[dict]],
    incremental_evidence: bool = False,
) -> None:
    """Write drafts/{section_key}/content.md, summary.md, and evidence.md.

    Args:
        section_evidence: Pass None to skip rewriting evidence.md.
        incremental_evidence: If True and evidence.md exists, merge new entries
            with existing ones (dedup by URL+content) before writing.
    """
    section_dir = _resolve_section_dir(output_dir, section_key)
    os.makedirs(section_dir, exist_ok=True)

    # content.md
    content_text = format_section_content_draft(section_title, section_body)
    async with aiofiles.open(os.path.join(section_dir, "content.md"), "w", encoding="utf-8") as f:
        await f.write(content_text)

    # summary.md
    if not summary:
        summary = extract_section_summary_fallback(section_body)
    summary_text = format_section_summary_draft(section_title, summary)
    async with aiofiles.open(os.path.join(section_dir, "summary.md"), "w", encoding="utf-8") as f:
        await f.write(summary_text)

    # evidence.md — skip if section_evidence is None
    if section_evidence is None:
        return

    evidence_path = os.path.join(section_dir, "evidence.md")
    final_evidence = section_evidence
    if incremental_evidence and os.path.exists(evidence_path):
        async with aiofiles.open(evidence_path, "r", encoding="utf-8") as f:
            existing_md = await f.read()
        existing = _parse_evidence_from_md(existing_md)
        final_evidence = _merge_evidence(existing, section_evidence)

    evidence_text = format_section_evidence_draft(chapter_num, section_title, final_evidence)
    async with aiofiles.open(evidence_path, "w", encoding="utf-8") as f:
        await f.write(evidence_text)