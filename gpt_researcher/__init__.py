from __future__ import annotations

from importlib import import_module

__all__ = ["GPTResearcher"]


def __getattr__(name: str):
    if name != "GPTResearcher":
        raise AttributeError(f"module 'gpt_researcher' has no attribute {name!r}")
    module = import_module("gpt_researcher.agent")
    return getattr(module, name)
