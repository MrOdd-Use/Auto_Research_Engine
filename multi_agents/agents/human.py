import asyncio
import json
from typing import Any, Optional

from .intent_recognizer import IntentRecognizer

_STOP_WORDS = frozenset({
    "stop", "skip", "done", "finish", "force publish",
})


class HumanAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def review_plan(self, research_state: dict):
        task = research_state.get("task") or {}
        section_details = research_state.get("section_details") or []
        layout = research_state.get("sections")

        display = self._format_outline(section_details) if section_details else str(layout)

        user_feedback = None

        if task.get("include_human_feedback"):
            if self.websocket and self.stream_output:
                try:
                    await self.stream_output(
                        "human_feedback",
                        "request",
                        (
                            "Planner generated this research outline. "
                            "Please provide any revisions you'd like.\n\n"
                            f"{display}\n\n"
                            "If no changes are needed, reply with 'no'."
                        ),
                        self.websocket,
                    )
                    raw = await self._receive_feedback()
                    user_feedback = await self._expand_and_confirm_ws(raw, display)
                except Exception as e:
                    print(f"Error receiving human feedback: {e}", flush=True)
            else:
                print(
                    (
                        "Any feedback on this research outline?\n\n"
                        f"{display}\n\n"
                        "If not, please reply with 'no'."
                    ),
                    flush=True,
                )
                raw = input(">> ")
                user_feedback = await self._expand_and_confirm_console(raw, display)

        user_feedback = self._normalize_feedback(user_feedback)

        feedback_log = "None" if user_feedback is None else "[provided]"
        print(f"User feedback before return: {feedback_log}", flush=True)

        return {"human_feedback": user_feedback}

    async def _expand_and_confirm_console(self, raw: str | None, outline: str) -> str | None:
        """Expand raw feedback via LLM and confirm with user in console mode."""
        normalized = self._normalize_feedback(raw)
        if normalized is None:
            return raw  # "no" or empty — skip expansion

        print("\n[Intent Recognition] Analyzing your revision feedback...\n", flush=True)
        try:
            expanded = await IntentRecognizer().expand(normalized, outline)
        except Exception as e:
            print(f"[Intent Recognition] Expansion failed, using original input: {e}", flush=True)
            return raw

        print(f"[Expanded Revision Instruction]\n{'-' * 60}\n{expanded}\n{'-' * 60}\n", flush=True)
        confirm = input("Confirm the revision above? [Enter to confirm / type to override]\n>> ").strip()
        return confirm if confirm else expanded

    async def _expand_and_confirm_ws(self, raw: str | None, outline: str) -> str | None:
        """Expand raw feedback via LLM and confirm with user in websocket mode."""
        normalized = self._normalize_feedback(raw)
        if normalized is None:
            return raw

        try:
            expanded = await IntentRecognizer().expand(normalized, outline)
        except Exception as e:
            print(f"[Intent Recognition] Expansion failed, using original input: {e}", flush=True)
            return raw

        await self.stream_output(
            "human_feedback",
            "intent_expanded",
            (
                f"I've expanded your feedback into a detailed revision instruction:\n\n"
                f"{expanded}\n\n"
                "Reply 'ok' to confirm, or type new content to override."
            ),
            self.websocket,
        )
        confirm_raw = await self._receive_feedback()
        confirm = self._normalize_feedback(confirm_raw)
        if confirm is None or confirm.lower() in {"ok", "yes", "confirm"}:
            return expanded
        return confirm

    async def _receive_feedback(self) -> Optional[str]:
        """Receive human feedback from queue (preferred) or websocket text."""
        feedback_queue = self._get_feedback_queue()
        if feedback_queue is not None:
            while True:
                queued_feedback = await feedback_queue.get()
                if queued_feedback is None:
                    return None
                parsed = self._extract_feedback_content(queued_feedback)
                return parsed

        ws = self._get_raw_websocket()
        if ws is None or not hasattr(ws, "receive_text"):
            return None

        while True:
            response = await ws.receive_text()
            parsed = self._extract_feedback_content(response)
            if parsed is not None:
                return parsed

    def _get_raw_websocket(self):
        if self.websocket is None:
            return None
        return getattr(self.websocket, "websocket", self.websocket)

    def _get_feedback_queue(self) -> Optional[asyncio.Queue]:
        ws = self._get_raw_websocket()
        queue = getattr(ws, "human_feedback_queue", None) if ws is not None else None
        if queue is None and ws is not None:
            state = getattr(ws, "state", None)
            queue = getattr(state, "human_feedback_queue", None) if state is not None else None
        return queue if isinstance(queue, asyncio.Queue) else None

    def _extract_feedback_content(self, payload: Any) -> Optional[str]:
        if payload is None:
            return None
        if isinstance(payload, dict):
            if payload.get("type") == "human_feedback":
                return self._clean_feedback(payload.get("content"))
            if "content" in payload:
                return self._clean_feedback(payload.get("content"))
            return self._clean_feedback(str(payload))
        if not isinstance(payload, str):
            return self._clean_feedback(str(payload))
        text = payload.strip()
        if not text:
            return None
        if text == "ping":
            return None
        if text.startswith("human_feedback"):
            remaining = text[len("human_feedback"):].strip()
            if not remaining:
                return None
            text = remaining
        try:
            parsed = json.loads(text)
            return self._extract_feedback_content(parsed)
        except json.JSONDecodeError:
            return self._clean_feedback(text)

    @staticmethod
    def _clean_feedback(feedback: Any) -> Optional[str]:
        if feedback is None:
            return None
        text = str(feedback).strip()
        return text or None

    @staticmethod
    def _normalize_feedback(feedback: Optional[str]) -> Optional[str]:
        if feedback is None:
            return None
        cleaned = feedback.strip()
        if not cleaned:
            return None
        if cleaned.lower() in {"no", "n", "none", "null", "skip"}:
            return None
        return cleaned

    async def review_writer_draft(
        self,
        draft_layout: str,
        claim_annotations: list,
        task: dict,
    ) -> dict:
        """Interactive breakpoint after writer node.

        Returns:
            {"force_publish": bool, "reviewer_note": str | None}
        """
        if not task.get("include_writer_review"):
            return {"force_publish": False, "reviewer_note": None}

        cited = sum(1 for c in claim_annotations if c.get("source_ids"))
        total = len(claim_annotations)
        summary = f"{total} claims — {cited} cited, {total - cited} uncited."

        preview = draft_layout[:3000] + ("\n...[truncated]" if len(draft_layout) > 3000 else "")
        display = (
            f"[Writer Breakpoint] Draft complete. {summary}\n\n"
            f"{'-' * 60}\n{preview}\n{'-' * 60}\n\n"
            "Enter / 'no' → continue to review\n"
            "'stop'       → skip review, publish now\n"
            "Any text     → pass as instructions to reviewer\n"
        )

        if self.websocket and self.stream_output:
            try:
                await self.stream_output(
                    "human_feedback", "writer_draft_review", display, self.websocket
                )
                raw = await self._receive_feedback()
            except Exception as e:
                print(f"Error receiving writer draft feedback: {e}", flush=True)
                return {"force_publish": False, "reviewer_note": None}
        else:
            print(f"\n{display}\n", flush=True)
            raw = input(">> ").strip()

        normalized = self._normalize_feedback(raw)
        if normalized is None:
            return {"force_publish": False, "reviewer_note": None}

        if normalized.lower() in _STOP_WORDS:
            msg = "[Writer Breakpoint] Stop signal — skipping review cycle."
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "human_feedback", "writer_force_publish", msg, self.websocket
                )
            else:
                print(f"\n{msg}\n", flush=True)
            return {"force_publish": True, "reviewer_note": None}

        return {"force_publish": False, "reviewer_note": normalized}

    async def collect_review_feedback(
        self, review_text: str, draft_text: str, task: dict
    ) -> tuple:
        """
        Show reviewer feedback, collect human opinions with intent expansion + confirmation.

        Returns (list[str] | None, force_stop: bool).
        force_stop=True means skip remaining review rounds and publish immediately.
        """
        if not task.get("include_human_feedback"):
            return None, False

        display = (
            f"[Reviewer feedback]\n{'-' * 60}\n{review_text}\n{'-' * 60}\n\n"
            "Please enter your additional opinions (one per line for multiple).\n"
            "Enter 'no' to skip; enter 'stop' to force end revisions and publish immediately."
        )

        if self.websocket and self.stream_output:
            try:
                await self.stream_output(
                    "human_feedback", "review_request", display, self.websocket
                )
                raw = await self._receive_feedback()
            except Exception as e:
                print(f"Error receiving review feedback: {e}", flush=True)
                return None, False
            return await self._process_review_input_ws(raw, review_text, draft_text)
        else:
            print(f"\n{display}\n", flush=True)
            raw = input(">> ").strip()
            return await self._process_review_input_console(raw, review_text, draft_text)

    async def _process_review_input_console(
        self, raw: str, review_text: str, draft_text: str
    ) -> tuple:
        normalized = self._normalize_feedback(raw)
        if normalized is None:
            return None, False
        if normalized.lower() in _STOP_WORDS:
            print("\n[Checkpoint] Received stop signal, skipping remaining review rounds, entering publish stage.\n", flush=True)
            return None, True

        lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
        expanded_items = []
        recognizer = IntentRecognizer()
        for line in lines:
            print(f"\n[Intent Recognition] Analyzing: {line[:80]}...\n", flush=True)
            try:
                expanded = await recognizer.expand_review_feedback(line, review_text, draft_text)
            except Exception as e:
                print(f"[Intent Recognition] Failed, using original input: {e}", flush=True)
                expanded = line
            print(f"[Expanded]\n{'-' * 60}\n{expanded}\n{'-' * 60}\n", flush=True)
            confirm = input("Confirm this opinion? [Enter to confirm / type to override]\n>> ").strip()
            expanded_items.append(confirm if confirm else expanded)

        return (expanded_items if expanded_items else None), False

    async def _process_review_input_ws(
        self, raw: Optional[str], review_text: str, draft_text: str
    ) -> tuple:
        normalized = self._normalize_feedback(raw)
        if normalized is None:
            return None, False
        if normalized.lower() in _STOP_WORDS:
            await self.stream_output(
                "human_feedback", "force_publish",
                "Received stop signal, skipping remaining review rounds, entering publish stage.",
                self.websocket,
            )
            return None, True

        lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
        expanded_items = []
        recognizer = IntentRecognizer()
        for line in lines:
            try:
                expanded = await recognizer.expand_review_feedback(line, review_text, draft_text)
            except Exception:
                expanded = line
            await self.stream_output(
                "human_feedback", "review_intent_expanded",
                (
                    f"Expanded review opinion:\n\n{expanded}\n\n"
                    "Reply 'ok' to confirm, or type new content to override."
                ),
                self.websocket,
            )
            confirm_raw = await self._receive_feedback()
            confirm = self._normalize_feedback(confirm_raw)
            if confirm is None or confirm.lower() in {"ok", "yes", "confirm"}:
                expanded_items.append(expanded)
            else:
                expanded_items.append(confirm)

        return (expanded_items if expanded_items else None), False

    @staticmethod
    def _format_outline(section_details: list) -> str:
        lines = []
        for i, section in enumerate(section_details, 1):
            lines.append(f"{i}. {section.get('header', 'Untitled')}")
            description = section.get("description", "")
            if description:
                lines.append(f"   {description}")
            for kp in section.get("key_points", []):
                if isinstance(kp, dict):
                    point_text = kp.get("point") or kp.get("text") or str(kp)
                else:
                    point_text = str(kp)
                lines.append(f"   - {point_text}")
        return "\n".join(lines)
