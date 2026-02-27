import asyncio
import json
from typing import Any, Optional


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
            # Stream response to the user if a websocket is provided (such as from web app)
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
                    user_feedback = await self._receive_feedback()
                except Exception as e:
                    print(f"Error receiving human feedback: {e}", flush=True)
            # Otherwise, prompt the user for feedback in the console
            else:
                user_feedback = input(
                    f"Any feedback on this research outline?\n\n{display}\n\nIf not, please reply with 'no'.\n>> "
                )

        user_feedback = self._normalize_feedback(user_feedback)

        print(f"User feedback before return: {user_feedback}", flush=True)

        return {"human_feedback": user_feedback}

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
        """
        Resolve the underlying websocket object.
        In server mode, self.websocket is a CustomLogsHandler that wraps .websocket.
        """
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
        """Extract human feedback content from a websocket payload."""
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

    @staticmethod
    def _format_outline(section_details: list) -> str:
        """Format enriched section details into a readable numbered outline."""
        lines = []
        for i, section in enumerate(section_details, 1):
            lines.append(f"{i}. {section.get('header', 'Untitled')}")
            description = section.get("description", "")
            if description:
                lines.append(f"   {description}")
            for kp in section.get("key_points", []):
                lines.append(f"   - {kp}")
        return "\n".join(lines)
