import difflib
import re
from typing import Any, List, Optional, Tuple

from .utils.views import print_agent_output
from .utils.llms import call_model
from multi_agents.route_agent import build_route_context

TEMPLATE = """You are an expert research article reviewer. \
Your goal is to review research drafts and provide feedback to the reviser based on guidelines and factual accuracy. \
"""

SOURCE_VERIFY_INSTRUCTION = """
## Source Verification

The following paragraphs were modified or added by the reviser compared to the previous draft.
Check each factual claim against the evidence sources below. If a claim has NO supporting evidence
in the sources, flag it as: [HALLUCINATION] "<the claim>" — no source supports this, please remove or rewrite.

Modified paragraphs:
{changed_paragraphs}

Evidence Sources (condensed):
{sources_text}
"""

AUDIT_INSTRUCTION = """
## Opinion Item Audit

The following opinion items have been accumulated over previous rounds, including
items that were previously marked as resolved.
Please re-audit every listed item against the current draft on every review pass:

{tracked_items}

For each item, note:
- [Resolved] Item#N: draft has addressed this
- [Unresolved] Item#N: issue still present
- [Partially resolved] Item#N: draft improved it, but important parts are still missing
- If a previously resolved item regressed in the latest draft, mark it as [Unresolved] or [Partially resolved]
- Audit every listed Item#N in order and provide exactly one status line for each item before listing new issues

After the audit, append any new issues found this round (if any), in the format:
[New issue] Issue description
"""

FULL_DRAFT_SOURCE_VERIFY_INSTRUCTION = """
## Source Verification

Review the full draft above against the evidence sources below.
For every factual claim in the draft, check whether at least one evidence source supports it.
If a claim has NO supporting evidence in the sources, flag it as:
[HALLUCINATION] "<the claim>" — no source supports this, please remove or rewrite.

Evidence Sources (condensed):
{sources_text}
"""


def _extract_changed_paragraphs(previous: str, current: str) -> List[str]:
    """Extract paragraphs that were added or modified in the current version."""
    previous_lines = previous.splitlines()
    current_lines = current.splitlines()
    differ = difflib.unified_diff(previous_lines, current_lines, lineterm="")
    changed: List[str] = []
    for line in differ:
        if line.startswith("+") and not line.startswith("+++"):
            content = line[1:].strip()
            if content:
                changed.append(content)
    return changed


def _source_sort_key(item: Tuple[str, Any]) -> Tuple[int, Tuple[int, ...], str]:
    """Sort source ids safely across legacy and chapter-based key formats."""
    source_id = str(item[0] or "").strip()
    numeric_parts = tuple(int(part) for part in re.findall(r"\d+", source_id))
    if re.fullmatch(r"\d+(?:\.\d+)*", source_id):
        return (0, numeric_parts, source_id)
    if numeric_parts:
        return (1, numeric_parts, source_id)
    return (2, tuple(), source_id)


def _build_condensed_sources(source_index: dict, limit: int = 50) -> str:
    """Build a condensed source list for the review prompt."""
    entries = sorted(source_index.items(), key=_source_sort_key)[:limit]
    lines = []
    for key, info in entries:
        snippet = info.get("content", "")[:200]
        domain = info.get("domain", "")
        lines.append(f"[{key}]({domain}) {snippet}")
    return "\n".join(lines)


class ReviewerAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def review_draft(self, draft_state: dict):
        """Review a draft article for guideline compliance and factual accuracy."""
        task = draft_state.get("task") or {}
        guidelines = "- ".join(guideline for guideline in task.get("guidelines", []))
        revision_notes = draft_state.get("revision_notes")
        source_index = draft_state.get("source_index") or {}
        previous_draft = draft_state.get("previous_draft") or ""
        current_draft = draft_state.get("draft") or ""
        checkpoint_note = (
            str(task.get("checkpoint_note") or "").strip()
            if task.get("checkpoint_target") == "reviewer"
            else ""
        )
        note_block = (
            f"Additional rerun instruction for this review pass: {checkpoint_note}\n"
            if checkpoint_note
            else ""
        )

        revise_prompt = f"""The reviser has already revised the draft based on your previous review notes with the following feedback:
{revision_notes}\n
Please provide additional feedback ONLY if critical since the reviser has already made changes based on your previous feedback.
If you think the article is sufficient or that non critical revisions are required, please aim to return None.
"""

        # Build per-item audit block when previous opinions exist.
        pending_items_text = draft_state.get("pending_opinions") or ""
        audit_block = (
            AUDIT_INSTRUCTION.format(tracked_items=pending_items_text)
            if pending_items_text
            else ""
        )

        # Build source verification block if source evidence is available.
        source_verify_block = ""
        if source_index:
            sources_text = _build_condensed_sources(source_index)
            changed = _extract_changed_paragraphs(previous_draft, current_draft) if previous_draft else []
            if changed:
                source_verify_block = SOURCE_VERIFY_INSTRUCTION.format(
                    changed_paragraphs="\n".join(f"- {p}" for p in changed),
                    sources_text=sources_text,
                )
            else:
                source_verify_block = FULL_DRAFT_SOURCE_VERIFY_INSTRUCTION.format(
                    sources_text=sources_text,
                )

        review_prompt = f"""You have been tasked with reviewing the draft which was written by a non-expert based on specific guidelines.
Please accept the draft if it is good enough to publish, or send it for revision, along with your notes to guide the revision.
If not all of the guideline criteria are met, you should send appropriate revision notes.
If the draft meets all the guidelines and all factual claims are supported, please return None.
{revise_prompt if revision_notes else ""}

Guidelines: {guidelines}\nDraft: {current_draft}\n
{source_verify_block}
{audit_block}
{note_block}
"""
        prompt = [
            {"role": "system", "content": TEMPLATE},
            {"role": "user", "content": review_prompt},
        ]

        route_context = build_route_context(
            application_name=str(task.get("application_name") or "auto_research_engine"),
            shared_agent_class="review_agent",
            agent_role="reviewer",
            stage_name="final_review",
            system_prompt=TEMPLATE,
            task=str(task.get("query") or ""),
            state=draft_state,
            task_payload=task,
        )
        response = await call_model(prompt, model=task.get("model"), route_context=route_context)

        if task.get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "review_feedback",
                    f"Review feedback is: {response}...",
                    self.websocket,
                )
            else:
                print_agent_output(
                    f"Review feedback is: {response}...", agent="REVIEWER"
                )

        if self._is_empty_review_response(response):
            return None
        return response

    async def run(self, draft_state: dict):
        task = draft_state.get("task") or {}
        to_follow_guidelines = task.get("follow_guidelines")
        source_index = draft_state.get("source_index") or {}

        # Run review if guidelines are enabled or source evidence is available for factual auditing.
        should_review = bool(to_follow_guidelines) or bool(source_index)

        if should_review:
            print_agent_output("Reviewing draft...", agent="REVIEWER")
            if task.get("verbose") and to_follow_guidelines:
                guidelines = task.get("guidelines")
                print_agent_output(
                    f"Following guidelines {guidelines}...", agent="REVIEWER"
                )
            review = await self.review_draft(draft_state)
        else:
            print_agent_output(
                "Skipping review (no guidelines and no source evidence to audit)...",
                agent="REVIEWER",
            )
            review = None

        return {"review": review}

    @staticmethod
    def _is_empty_review_response(response) -> bool:
        if response is None:
            return True
        if isinstance(response, dict):
            candidate = response.get("review")
            if candidate is None:
                candidate = response.get("feedback")
            return ReviewerAgent._is_empty_review_response(candidate)
        if isinstance(response, list):
            return len(response) == 0

        normalized = str(response).strip().lower()
        normalized = normalized.strip("`'\" \n\t")
        normalized = normalized.rstrip(".!?:;")
        return normalized in {"none", "null", ""}
