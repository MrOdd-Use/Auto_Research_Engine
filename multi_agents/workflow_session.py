import copy
import uuid
import asyncio
from typing import Any, Dict, Optional

from backend.server.workflow_store import WorkflowStore, make_json_safe


def _section_slug(section_index: int, section_title: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in str(section_title or "section"))
    safe = "_".join(part for part in safe.split("_") if part)
    return f"section_{section_index + 1}_{safe[:48] or 'section'}"


class WorkflowSessionRecorder:
    def __init__(self, workflow_store: WorkflowStore, report_id: str, session: Dict[str, Any]):
        self.workflow_store = workflow_store
        self.report_id = report_id
        self.session = copy.deepcopy(make_json_safe(session))
        self._step_order = len(self.session.get("checkpoints") or [])
        self._order_lock = asyncio.Lock()

    @property
    def session_id(self) -> str:
        return str(self.session.get("session_id"))

    @property
    def note(self) -> Optional[str]:
        value = self.session.get("note")
        return str(value) if value else None

    async def append_event(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict) or not event.get("type"):
            return
        ordered_data = self.session.setdefault("ordered_data", [])
        ordered_data.append(make_json_safe(event))
        await self.flush()

    async def append_path_event(self, output: Dict[str, Any]) -> None:
        ordered_data = self.session.setdefault("ordered_data", [])
        ordered_data.append({"type": "path", "output": make_json_safe(output)})
        await self.flush()

    async def record_global_checkpoint(
        self,
        node_name: str,
        *,
        state_before: Dict[str, Any],
        output_delta: Dict[str, Any],
        state_after: Dict[str, Any],
        summary: Optional[Dict[str, Any]] = None,
        rerunnable: bool = True,
    ) -> Dict[str, Any]:
        checkpoint = {
            "checkpoint_id": uuid.uuid4().hex,
            "scope": "global",
            "node_name": node_name,
            "display_name": node_name,
            "status": "completed",
            "rerunnable": rerunnable,
            "step_order": await self._next_step_order(),
            "state_before": make_json_safe(state_before),
            "output_delta": make_json_safe(output_delta),
            "state_after": make_json_safe(state_after),
            "summary": make_json_safe(summary or {}),
        }
        self.session.setdefault("checkpoints", []).append(checkpoint)
        await self.flush()
        return checkpoint

    async def record_section_checkpoint(
        self,
        *,
        section_index: int,
        section_title: str,
        node_name: str,
        state_before: Dict[str, Any],
        output_delta: Dict[str, Any],
        state_after: Dict[str, Any],
        summary: Optional[Dict[str, Any]] = None,
        rerunnable: bool = True,
    ) -> Dict[str, Any]:
        checkpoint = {
            "checkpoint_id": uuid.uuid4().hex,
            "scope": "section_node",
            "node_name": node_name,
            "display_name": node_name,
            "section_index": section_index,
            "section_key": _section_slug(section_index, section_title),
            "section_title": section_title or f"Section {section_index + 1}",
            "status": "completed",
            "rerunnable": rerunnable,
            "step_order": await self._next_step_order(),
            "state_before": make_json_safe(state_before),
            "output_delta": make_json_safe(output_delta),
            "state_after": make_json_safe(state_after),
            "summary": make_json_safe(summary or {}),
        }
        self.session.setdefault("checkpoints", []).append(checkpoint)
        await self.flush()
        return checkpoint

    async def mark_completed(self, final_state: Dict[str, Any], answer: str) -> None:
        self.session["status"] = "completed"
        self.session["final_state"] = make_json_safe(final_state)
        self.session["answer"] = answer or ""
        await self.flush()

    async def mark_failed(self, error: str, final_state: Optional[Dict[str, Any]] = None) -> None:
        self.session["status"] = "failed"
        self.session["error"] = error
        if final_state is not None:
            self.session["final_state"] = make_json_safe(final_state)
        await self.flush()

    async def flush(self) -> None:
        await self.workflow_store.save_session(self.report_id, self.session)

    async def _next_step_order(self) -> int:
        async with self._order_lock:
            self._step_order += 1
            return self._step_order
