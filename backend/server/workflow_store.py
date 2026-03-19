import asyncio
import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return str(value)


def build_session_summary(session: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_id": session.get("session_id"),
        "parent_session_id": session.get("parent_session_id"),
        "root_session_id": session.get("root_session_id"),
        "rerun_from_checkpoint_id": session.get("rerun_from_checkpoint_id"),
        "round_index": session.get("round_index", 1),
        "status": session.get("status", "running"),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "note": session.get("note"),
        "target": make_json_safe(session.get("target") or {}),
    }


def build_checkpoint_tree(checkpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    global_nodes: List[Dict[str, Any]] = []
    sections: Dict[str, Dict[str, Any]] = {}

    for checkpoint in checkpoints:
        scope = checkpoint.get("scope")
        if scope == "global":
            global_nodes.append(
                {
                    "checkpoint_id": checkpoint.get("checkpoint_id"),
                    "node_name": checkpoint.get("node_name"),
                    "display_name": checkpoint.get("display_name"),
                    "status": checkpoint.get("status", "completed"),
                    "step_order": checkpoint.get("step_order", 0),
                    "rerunnable": checkpoint.get("rerunnable", True),
                    "summary": make_json_safe(checkpoint.get("summary") or {}),
                    "scope": "global",
                }
            )
            continue

        section_key = str(checkpoint.get("section_key") or "")
        if not section_key:
            continue
        section_entry = sections.setdefault(
            section_key,
            {
                "section_key": section_key,
                "section_index": checkpoint.get("section_index", 0),
                "section_title": checkpoint.get("section_title") or checkpoint.get("section_key") or "Section",
                "scope": "section",
                "checkpoints": [],
            },
        )
        section_entry["checkpoints"].append(
            {
                "checkpoint_id": checkpoint.get("checkpoint_id"),
                "node_name": checkpoint.get("node_name"),
                "display_name": checkpoint.get("display_name"),
                "status": checkpoint.get("status", "completed"),
                "step_order": checkpoint.get("step_order", 0),
                "rerunnable": checkpoint.get("rerunnable", True),
                "summary": make_json_safe(checkpoint.get("summary") or {}),
                "scope": checkpoint.get("scope", "section"),
                "section_key": section_key,
            }
        )

    sorted_globals = sorted(global_nodes, key=lambda item: item.get("step_order", 0))
    sorted_sections = sorted(
        sections.values(),
        key=lambda item: (item.get("section_index", 0), item.get("section_title", "")),
    )
    for section in sorted_sections:
        section["checkpoints"] = sorted(
            section.get("checkpoints") or [],
            key=lambda item: item.get("step_order", 0),
        )

    return {
        "global_nodes": sorted_globals,
        "sections": sorted_sections,
    }


class WorkflowStore:
    def __init__(self, root: Path):
        self._root = root
        self._lock = asyncio.Lock()

    def _report_dir(self, report_id: str) -> Path:
        return self._root / report_id

    def _index_path(self, report_id: str) -> Path:
        return self._report_dir(report_id) / "index.json"

    def _session_path(self, report_id: str, session_id: str) -> Path:
        return self._report_dir(report_id) / f"session_{session_id}.json"

    def _default_index(self, report_id: str) -> Dict[str, Any]:
        return {
            "report_id": report_id,
            "workflow_available": False,
            "current_session_id": None,
            "last_successful_session_id": None,
            "sessions": [],
            "updated_at": _utc_now_iso(),
        }

    async def _ensure_report_dir(self, report_id: str) -> None:
        self._report_dir(report_id).mkdir(parents=True, exist_ok=True)

    async def _read_json_unlocked(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return copy.deepcopy(default)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return copy.deepcopy(default)

    async def _write_json_unlocked(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(make_json_safe(value), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    async def get_index(self, report_id: str) -> Dict[str, Any]:
        async with self._lock:
            return await self._read_json_unlocked(
                self._index_path(report_id),
                self._default_index(report_id),
            )

    async def get_session(self, report_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            session = await self._read_json_unlocked(self._session_path(report_id, session_id), None)
            return session if isinstance(session, dict) else None

    async def create_session(
        self,
        report_id: str,
        *,
        parent_session_id: Optional[str] = None,
        rerun_from_checkpoint_id: Optional[str] = None,
        note: Optional[str] = None,
        target: Optional[Dict[str, Any]] = None,
        task_query: str = "",
    ) -> Dict[str, Any]:
        async with self._lock:
            await self._ensure_report_dir(report_id)
            index = await self._read_json_unlocked(
                self._index_path(report_id),
                self._default_index(report_id),
            )
            parent_session = None
            if parent_session_id:
                parent_session = await self._read_json_unlocked(
                    self._session_path(report_id, parent_session_id),
                    None,
                )

            session_id = uuid.uuid4().hex
            round_index = 1
            root_session_id = session_id
            if isinstance(parent_session, dict):
                round_index = int(parent_session.get("round_index") or 1) + 1
                root_session_id = parent_session.get("root_session_id") or parent_session_id or session_id

            session = {
                "report_id": report_id,
                "session_id": session_id,
                "parent_session_id": parent_session_id,
                "root_session_id": root_session_id,
                "rerun_from_checkpoint_id": rerun_from_checkpoint_id,
                "round_index": round_index,
                "status": "running",
                "note": note,
                "target": make_json_safe(target or {}),
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "checkpoints": [],
                "ordered_data": (
                    [{"type": "question", "content": task_query}]
                    if task_query
                    else []
                ),
                "answer": "",
                "final_state": {},
                "error": None,
            }

            sessions = [item for item in index.get("sessions") or [] if item.get("session_id") != session_id]
            sessions.append(build_session_summary(session))
            index.update(
                {
                    "workflow_available": True,
                    "current_session_id": session_id,
                    "sessions": sessions,
                    "updated_at": _utc_now_iso(),
                }
            )
            await self._write_json_unlocked(self._session_path(report_id, session_id), session)
            await self._write_json_unlocked(self._index_path(report_id), index)
            return session

    async def save_session(self, report_id: str, session: Dict[str, Any]) -> None:
        async with self._lock:
            await self._ensure_report_dir(report_id)
            session = make_json_safe(session)
            session["updated_at"] = _utc_now_iso()

            index = await self._read_json_unlocked(
                self._index_path(report_id),
                self._default_index(report_id),
            )
            summaries = []
            found = False
            for item in index.get("sessions") or []:
                if item.get("session_id") == session.get("session_id"):
                    summaries.append(build_session_summary(session))
                    found = True
                else:
                    summaries.append(item)
            if not found:
                summaries.append(build_session_summary(session))

            index["workflow_available"] = True
            index["current_session_id"] = session.get("session_id")
            if session.get("status") == "completed":
                index["last_successful_session_id"] = session.get("session_id")
            index["sessions"] = summaries
            index["updated_at"] = _utc_now_iso()

            await self._write_json_unlocked(
                self._session_path(report_id, str(session.get("session_id"))),
                session,
            )
            await self._write_json_unlocked(self._index_path(report_id), index)

    async def mark_session_failed(self, report_id: str, session_id: str, error: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            session = await self._read_json_unlocked(self._session_path(report_id, session_id), None)
            if not isinstance(session, dict):
                return None
            session["status"] = "failed"
            session["error"] = error
            session["updated_at"] = _utc_now_iso()

            index = await self._read_json_unlocked(
                self._index_path(report_id),
                self._default_index(report_id),
            )
            summaries = []
            for item in index.get("sessions") or []:
                if item.get("session_id") == session_id:
                    summaries.append(build_session_summary(session))
                else:
                    summaries.append(item)
            index["workflow_available"] = True
            index["current_session_id"] = session_id
            index["sessions"] = summaries
            index["updated_at"] = _utc_now_iso()

            await self._write_json_unlocked(self._session_path(report_id, session_id), session)
            await self._write_json_unlocked(self._index_path(report_id), index)
            return session

    async def get_latest_successful_session(self, report_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            index = await self._read_json_unlocked(
                self._index_path(report_id),
                self._default_index(report_id),
            )
            session_id = index.get("last_successful_session_id")
            if not session_id:
                return None
            session = await self._read_json_unlocked(self._session_path(report_id, session_id), None)
            return session if isinstance(session, dict) else None

    async def find_checkpoint(self, report_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            index = await self._read_json_unlocked(
                self._index_path(report_id),
                self._default_index(report_id),
            )
            for summary in reversed(index.get("sessions") or []):
                session_id = summary.get("session_id")
                if not session_id:
                    continue
                session = await self._read_json_unlocked(self._session_path(report_id, session_id), None)
                if not isinstance(session, dict):
                    continue
                for checkpoint in session.get("checkpoints") or []:
                    if checkpoint.get("checkpoint_id") == checkpoint_id:
                        return {
                            "session": session,
                            "checkpoint": checkpoint,
                        }
            return None

    async def build_workflow_response(
        self,
        report_id: str,
        *,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            index = await self._read_json_unlocked(
                self._index_path(report_id),
                self._default_index(report_id),
            )
            selected_session_id = session_id or index.get("current_session_id")
            selected_session = None
            if selected_session_id:
                selected_session = await self._read_json_unlocked(
                    self._session_path(report_id, selected_session_id),
                    None,
                )
            if not isinstance(selected_session, dict):
                selected_session = None

            if selected_session is None:
                return {
                    "report_id": report_id,
                    "workflow_available": False,
                    "legacy_reason": "This report was created before Rerun from Checkpoint was enabled.",
                    "current_session_id": index.get("current_session_id"),
                    "last_successful_session_id": index.get("last_successful_session_id"),
                    "sessions": index.get("sessions") or [],
                    "selected_session": None,
                }

            return {
                "report_id": report_id,
                "workflow_available": True,
                "legacy_reason": None,
                "current_session_id": index.get("current_session_id"),
                "last_successful_session_id": index.get("last_successful_session_id"),
                "sessions": index.get("sessions") or [],
                "selected_session": {
                    **build_session_summary(selected_session),
                    "answer": selected_session.get("answer") or "",
                    "ordered_data": make_json_safe(selected_session.get("ordered_data") or []),
                    "checkpoints_tree": build_checkpoint_tree(selected_session.get("checkpoints") or []),
                },
            }
