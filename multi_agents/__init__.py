from __future__ import annotations

from importlib import import_module


_EXPORT_MAP = {
    "ResearchAgent": "multi_agents.agents",
    "ScrapingAgent": "multi_agents.agents",
    "WriterAgent": "multi_agents.agents",
    "PublisherAgent": "multi_agents.agents",
    "ReviserAgent": "multi_agents.agents",
    "ReviewerAgent": "multi_agents.agents",
    "EditorAgent": "multi_agents.agents",
    "ChiefEditorAgent": "multi_agents.agents",
    "DraftState": "multi_agents.memory",
    "ResearchState": "multi_agents.memory",
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module 'multi_agents' has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
