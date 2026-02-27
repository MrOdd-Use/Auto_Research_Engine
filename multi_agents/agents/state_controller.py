import json
import os
import tempfile
from threading import Lock
from typing import Any, Dict


DEFAULT_AGENT_STATES: Dict[str, Dict[str, Any]] = {
    "planner": {
        "current_idx": 3,
        "tiers": ["o1-preview", "gpt-4o", "gpt-4o-mini", "gpt-4o-mini"],
    },
    "scrap": {
        "current_idx": 3,
        "tiers": ["o1-preview", "o1-preview", "gpt-4o", "gpt-4o-mini"],
    },
    "check_data": {
        "current_idx": 2,
        "tiers": ["o1-preview", "gpt-4o", "gpt-4o-mini", "gpt-4o-mini"],
    },
    "writer": {
        "current_idx": 1,
        "tiers": ["o1-preview", "gpt-4o", "gpt-4o-mini", "gpt-4o-mini"],
    },
    "review": {
        "current_idx": 0,
        "tiers": ["o1-preview", "gpt-4o", "gpt-4o-mini", "gpt-4o-mini"],
    },
    "revise": {
        "current_idx": 2,
        "tiers": ["o1-preview", "gpt-4o", "gpt-4o-mini", "gpt-4o-mini"],
    },
}


class StateController:
    """Persistent tier state manager for multi-agent model switching."""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self._lock = Lock()
        self.states = self._load_states()

    def _load_states(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.state_file):
            self._atomic_write(DEFAULT_AGENT_STATES)
            return json.loads(json.dumps(DEFAULT_AGENT_STATES))

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                content = json.load(f)
            if not isinstance(content, dict):
                raise ValueError("state file must be a JSON object")
        except Exception:
            self._atomic_write(DEFAULT_AGENT_STATES)
            return json.loads(json.dumps(DEFAULT_AGENT_STATES))

        merged = json.loads(json.dumps(DEFAULT_AGENT_STATES))
        for agent_id, data in content.items():
            if isinstance(data, dict):
                merged.setdefault(agent_id, {})
                merged[agent_id].update(data)
        self._atomic_write(merged)
        return merged

    def _atomic_write(self, payload: Dict[str, Dict[str, Any]]) -> None:
        directory = os.path.dirname(self.state_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=directory or None,
            prefix="agent_state_",
            suffix=".tmp",
        ) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            temp_name = tmp.name
        os.replace(temp_name, self.state_file)

    def _save_state(self) -> None:
        self._atomic_write(self.states)

    def _ensure_agent(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self.states:
            self.states[agent_id] = {"current_idx": 3, "tiers": ["gpt-4o-mini"] * 4}
        tiers = self.states[agent_id].get("tiers")
        if not isinstance(tiers, list) or len(tiers) < 4:
            self.states[agent_id]["tiers"] = ["gpt-4o-mini"] * 4
        if "current_idx" not in self.states[agent_id]:
            self.states[agent_id]["current_idx"] = 3
        return self.states[agent_id]

    def get_current_idx(self, agent_id: str) -> int:
        state = self._ensure_agent(agent_id)
        idx = int(state.get("current_idx", 3))
        return min(max(idx, 0), 3)

    def get_current_model(self, agent_id: str, fallback_model: str = "gpt-4o-mini") -> str:
        state = self._ensure_agent(agent_id)
        idx = self.get_current_idx(agent_id)
        tiers = state.get("tiers", [])
        if idx < len(tiers) and isinstance(tiers[idx], str) and tiers[idx].strip():
            return tiers[idx].strip()
        return fallback_model

    def promote(self, agent_id: str) -> None:
        with self._lock:
            state = self._ensure_agent(agent_id)
            current = self.get_current_idx(agent_id)
            state["current_idx"] = max(current - 1, 0)
            self._save_state()

    def demote(self, agent_id: str) -> None:
        with self._lock:
            state = self._ensure_agent(agent_id)
            current = self.get_current_idx(agent_id)
            state["current_idx"] = min(current + 1, 3)
            self._save_state()

    def set_tier(self, agent_id: str, tier_idx: int) -> None:
        with self._lock:
            state = self._ensure_agent(agent_id)
            state["current_idx"] = min(max(int(tier_idx), 0), 3)
            self._save_state()
