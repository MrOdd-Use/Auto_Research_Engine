"""Opinions store: accumulates reviewer-agent and human opinions across review rounds."""
from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Dict, List, Optional

# Valid rerun-node values
RERUN_RESEARCHER = "researcher"
RERUN_WRITER = "writer"
RERUN_NONE = "none"
_VALID_RERUN_NODES = {RERUN_RESEARCHER, RERUN_WRITER, RERUN_NONE}

# Opinion status values
STATUS_PENDING = "pending"
STATUS_UNRESOLVED = "unresolved"
STATUS_PARTIALLY_RESOLVED = "partially_resolved"
STATUS_RESOLVED = "resolved"
_VALID_STATUSES = {
    STATUS_PENDING,
    STATUS_UNRESOLVED,
    STATUS_PARTIALLY_RESOLVED,
    STATUS_RESOLVED,
}
_OPEN_STATUSES = {STATUS_PENDING, STATUS_UNRESOLVED, STATUS_PARTIALLY_RESOLVED}

_AUDIT_STATUS_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?\[(Resolved|Unresolved|Partially resolved)\]\s*Item#(\d+)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NEW_ISSUE_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(\[(?:New issue|HALLUCINATION)\]\s*.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NEW_ISSUE_PREFIX_RE = re.compile(r"^\s*(?:[-*•]\s*)?\[New issue\]\s*", re.IGNORECASE)


class OpinionItem:
    def __init__(
        self,
        source: str,
        content: str,
        status: str = "pending",
        rerun_node: Optional[str] = None,
    ) -> None:
        self.source = source            # "Agent" | "Human"
        self.content = content
        self.status = status if status in _VALID_STATUSES else STATUS_PENDING
        self.rerun_node = rerun_node    # "researcher" | "writer" | "none" | None (not yet annotated)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "content": self.content,
            "status": self.status,
            "rerun_node": self.rerun_node,
        }

    def mark_rerun_done(self) -> None:
        """Clear rerun tag after the rerun has been executed."""
        self.rerun_node = RERUN_NONE

    def set_status(self, status: str) -> None:
        """Update the item status, defaulting to unresolved for invalid input."""
        self.status = status if status in _VALID_STATUSES else STATUS_UNRESOLVED


class ReviewRound:
    def __init__(self, round_num: int, timestamp: str | None = None) -> None:
        self.round_num = round_num
        self.timestamp = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.items: List[OpinionItem] = []

    def add_item(self, source: str, content: str, rerun_node: Optional[str] = None) -> OpinionItem:
        item = OpinionItem(source=source, content=content, rerun_node=rerun_node)
        self.items.append(item)
        return item

    def mark_resolved(self, index: int) -> None:
        if 0 <= index < len(self.items):
            self.items[index].status = "resolved"

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "timestamp": self.timestamp,
            "items": [item.to_dict() for item in self.items],
        }

    def to_markdown(self) -> str:
        lines = [f"## Review Round {self.round_num} — {self.timestamp}\n"]
        if not self.items:
            lines.append("_No opinions recorded._\n")
            return "\n".join(lines)
        lines.append("| # | Source | Opinion | Rerun Node | Status |")
        lines.append("|---|--------|---------|-----------|--------|")
        for i, item in enumerate(self.items, 1):
            status_label = _status_label(item.status)
            rerun_label = _rerun_label(item.rerun_node)
            content = item.content.replace("\n", " ").replace("|", "｜")
            lines.append(f"| {i} | {item.source} | {content} | {rerun_label} | {status_label} |")
        lines.append("")
        return "\n".join(lines)


def _rerun_label(rerun_node: Optional[str]) -> str:
    if rerun_node == RERUN_RESEARCHER:
        return "[Rerun: Data Collection]"
    if rerun_node == RERUN_WRITER:
        return "[Rerun: Writing]"
    if rerun_node == RERUN_NONE:
        return "[Text Edit Only]"
    return "[Pending Annotation]"


def _status_label(status: str) -> str:
    if status == STATUS_RESOLVED:
        return "[Resolved]"
    if status == STATUS_PARTIALLY_RESOLVED:
        return "[Partially Resolved]"
    if status == STATUS_UNRESOLVED:
        return "[Unresolved]"
    return "[Pending]"


def _normalize_audit_status(raw_status: str) -> str:
    normalized = raw_status.strip().lower()
    if normalized == "resolved":
        return STATUS_RESOLVED
    if normalized == "partially resolved":
        return STATUS_PARTIALLY_RESOLVED
    return STATUS_UNRESOLVED


class OpinionsStore:
    """Accumulate review opinions across rounds; persist to Markdown."""

    def __init__(self) -> None:
        self._rounds: List[ReviewRound] = []

    # ── write ────────────────────────────────────────────────────────────

    def append_round(
        self,
        round_num: int,
        agent_opinions: List[str],
        human_opinions: Optional[List[str]] = None,
    ) -> ReviewRound:
        """Append a new review round; returns the round so callers can annotate rerun_node."""
        round_ = ReviewRound(round_num)
        for item in agent_opinions:
            if item and item.strip():
                round_.add_item("Agent", item.strip())
        for item in (human_opinions or []):
            if item and item.strip():
                round_.add_item("Human", item.strip())
        self._rounds.append(round_)
        return round_

    def mark_reruns_done(self) -> None:
        """After executing pending reruns, clear the rerun_node tag so they won't re-trigger."""
        for round_ in self._rounds:
            for item in round_.items:
                if item.rerun_node in {RERUN_RESEARCHER, RERUN_WRITER}:
                    item.mark_rerun_done()

    def mark_items_resolved(self, items: Optional[List[dict]] = None) -> None:
        """Mark the provided open items as resolved; when omitted, resolves all open items."""
        targets = items if items is not None else self.pending_items()
        for item_ref in targets:
            self._set_item_status(item_ref["round"], item_ref["index"], STATUS_RESOLVED)

    def apply_audit_results(self, review_text: str | None, audited_items: List[dict]) -> None:
        """Apply reviewer audit statuses to the provided open-item list."""
        if not audited_items:
            return

        if not review_text or not str(review_text).strip():
            self.mark_items_resolved(audited_items)
            return

        updates: dict[int, str] = {}
        for raw_status, item_num, _detail in _AUDIT_STATUS_RE.findall(str(review_text)):
            try:
                display_index = int(item_num)
            except ValueError:
                continue
            updates[display_index] = _normalize_audit_status(raw_status)

        for display_index, item_ref in enumerate(audited_items, 1):
            status = updates.get(display_index, STATUS_UNRESOLVED)
            self._set_item_status(item_ref["round"], item_ref["index"], status)

    # ── read ─────────────────────────────────────────────────────────────

    def pending_items(self) -> List[dict]:
        """Return opinion items whose status is still open."""
        result = []
        for round_ in self._rounds:
            for i, item in enumerate(round_.items):
                if item.status in _OPEN_STATUSES:
                    result.append({
                        "round": round_.round_num,
                        "index": i,
                        "source": item.source,
                        "content": item.content,
                        "status": item.status,
                        "rerun_node": item.rerun_node,
                    })
        return result

    def tracked_items(self) -> List[dict]:
        """Return all opinion items in stable round/index order for re-audit."""
        result = []
        for round_ in self._rounds:
            for i, item in enumerate(round_.items):
                result.append({
                    "round": round_.round_num,
                    "index": i,
                    "source": item.source,
                    "content": item.content,
                    "status": item.status,
                    "rerun_node": item.rerun_node,
                })
        return result

    def pending_as_numbered_list(self) -> str:
        """Format open opinion items as a numbered list for prompting."""
        items = self.pending_items()
        if not items:
            return ""
        return self._items_as_numbered_list(items)

    def tracked_as_numbered_list(self) -> str:
        """Format all opinion items as a numbered list for reviewer re-audit."""
        items = self.tracked_items()
        if not items:
            return ""
        return self._items_as_numbered_list(items)

    def resolved_as_numbered_list(self) -> str:
        """Format resolved opinion items as a numbered list for regression guards."""
        items = [item for item in self.tracked_items() if item["status"] == STATUS_RESOLVED]
        if not items:
            return ""
        return self._items_as_numbered_list(items)

    def _items_as_numbered_list(self, items: List[dict]) -> str:
        """Render opinion item references consistently for model prompts."""
        lines = [
            f"{i}. [Round {it['round']} / {it['source']} / {it['status']}] {it['content']}"
            for i, it in enumerate(items, 1)
        ]
        return "\n".join(lines)

    def pending_rerun_nodes(self) -> Dict[str, List[str]]:
        """
        Return pending items that require a node rerun, grouped by node name.
        Returns e.g. {"researcher": ["missing citation for X", ...], "writer": [...]}
        Only items with rerun_node in {RERUN_RESEARCHER, RERUN_WRITER} are included.
        """
        result: Dict[str, List[str]] = {}
        for round_ in self._rounds:
            for item in round_.items:
                if item.status in _OPEN_STATUSES and item.rerun_node in {RERUN_RESEARCHER, RERUN_WRITER}:
                    result.setdefault(item.rerun_node, []).append(item.content)
        return result

    def to_records(self) -> List[dict]:
        return [r.to_dict() for r in self._rounds]

    def to_markdown(self) -> str:
        if not self._rounds:
            return "# Review Opinions Document\n\n_No opinions recorded yet._\n"
        parts = ["# Review Opinions Document\n"]
        for round_ in self._rounds:
            parts.append(round_.to_markdown())
        return "\n".join(parts)

    # ── serialization ────────────────────────────────────────────────────

    @classmethod
    def from_records(cls, records: Optional[List[dict]]) -> "OpinionsStore":
        store = cls()
        for r in (records or []):
            round_ = ReviewRound(r["round"], timestamp=r.get("timestamp"))
            for item_data in r.get("items", []):
                round_.items.append(OpinionItem(
                    source=item_data["source"],
                    content=item_data["content"],
                    status=item_data.get("status", "pending"),
                    rerun_node=item_data.get("rerun_node"),
                ))
            store._rounds.append(round_)
        return store

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, output_dir: str) -> str:
        """Write opinions.md to output_dir; return the file path."""
        path = Path(output_dir) / "opinions.md"
        path.write_text(self.to_markdown(), encoding="utf-8")
        return str(path)

    def _set_item_status(self, round_num: int, item_index: int, status: str) -> None:
        for round_ in self._rounds:
            if round_.round_num != round_num:
                continue
            if 0 <= item_index < len(round_.items):
                round_.items[item_index].set_status(status)
            return


# ── Rerun annotation ──────────────────────────────────────────────────────────

_ANNOTATE_SYSTEM = (
    "You are a research workflow advisor. "
    "Given a list of review opinion items, classify each one into a rerun category:\n\n"
    '- "researcher": the opinion requires fetching new or additional data '
    "(missing citations, outdated statistics, need more evidence, insufficient sources, scraping gap)\n"
    '- "writer": the opinion requires regenerating the draft structure '
    "(major section reorganization, rewrite tone/style throughout, structural overhaul)\n"
    '- "none": the opinion can be addressed by text editing only '
    "(grammar, minor rewrites, adding transitions, small factual corrections inline)\n\n"
    "Return ONLY a JSON array: "
    '[{"index": 0, "rerun_node": "researcher"}, {"index": 1, "rerun_node": "none"}, ...]\n'
    "One entry per item. No explanation."
)


async def annotate_rerun_nodes(
    round_: "ReviewRound",
    draft_text: str,
    task: dict,
) -> None:
    """
    Classify each OpinionItem in round_ with a rerun_node via a single LLM call.
    Mutates items in-place; safe to call even if the round is empty.
    """
    items = round_.items
    if not items:
        return

    # Lazy import to avoid circular dependency at module load time
    import json as _json
    from multi_agents.agents.utils.llms import call_model
    from gpt_researcher.config.config import Config

    cfg = Config()
    model = task.get("model") or getattr(cfg, "fast_llm_model", None) or cfg.smart_llm_model

    numbered = "\n".join(
        f'{i}. [{item.source}] {item.content[:200]}'
        for i, item in enumerate(items)
    )
    draft_excerpt = (draft_text or "")[:600]

    prompt = [
        {"role": "system", "content": _ANNOTATE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Draft excerpt:\n{draft_excerpt}\n\n"
                f"Opinion items:\n{numbered}\n\n"
                "Classify each item."
            ),
        },
    ]

    try:
        raw = await call_model(
            prompt,
            model=model,
            response_format="json",
            route_context={
                "application_name": "auto_research_engine",
                "agent_role": "rerun_annotator",
                "stage_name": "opinion_rerun_classification",
            },
        )
        annotations = raw if isinstance(raw, list) else _json.loads(str(raw))
        for entry in annotations:
            idx = entry.get("index")
            node = entry.get("rerun_node", RERUN_NONE)
            if isinstance(idx, int) and 0 <= idx < len(items):
                items[idx].rerun_node = node if node in _VALID_RERUN_NODES else RERUN_NONE
    except Exception as exc:
        # Fall back: mark everything as text-only
        print(f"[rerun_annotator] classification failed ({exc}), defaulting to none", flush=True)
        for item in items:
            if item.rerun_node is None:
                item.rerun_node = RERUN_NONE


def parse_review_to_items(review_text: str) -> List[str]:
    """Split free-form review text into individual opinion items."""
    if not review_text or not review_text.strip():
        return []
    sanitized_lines: List[str] = []
    for raw_line in str(review_text).splitlines():
        if _AUDIT_STATUS_RE.match(raw_line):
            continue
        sanitized_lines.append(_NEW_ISSUE_PREFIX_RE.sub("", raw_line))
    review_text = "\n".join(sanitized_lines).strip()
    if not review_text:
        return []
    _leading_marker = re.compile(r"^\s*\d+[\.\)]\s+")
    # Numbered list: "1. ..." or "1) ..."
    numbered = re.split(r"\n\s*\d+[\.\)]\s+", review_text)
    if len(numbered) > 1:
        return [_leading_marker.sub("", item).strip() for item in numbered if item.strip()]
    # Bullet list: "- ..." or "* ..."
    bulleted = re.split(r"\n\s*[-*•]\s+", review_text)
    if len(bulleted) > 1:
        return [re.sub(r"^\s*[-*•]\s+", "", item).strip() for item in bulleted if item.strip()]
    # Paragraph split
    paragraphs = [p.strip() for p in review_text.split("\n\n") if p.strip()]
    return paragraphs if paragraphs else [review_text.strip()]


def extract_new_issue_items(review_text: str) -> List[str]:
    """Extract explicitly marked new issues from a reviewer response."""
    if not review_text or not str(review_text).strip():
        return []
    return [item.strip() for item in _NEW_ISSUE_RE.findall(str(review_text)) if item.strip()]
