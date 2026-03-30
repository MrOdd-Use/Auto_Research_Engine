from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple


@dataclass
class GlobalHealthStat:
    exec_failures: int = 0
    provider_failures: int = 0
    unavailable: bool = False

    @property
    def penalty(self) -> float:
        penalty = (self.exec_failures * 1.5) + (self.provider_failures * 2.0)
        if self.unavailable:
            penalty += 100.0
        return penalty


@dataclass
class SharedClassStat:
    successes: int = 0
    failures: int = 0
    applications: Set[str] = field(default_factory=set)

    @property
    def bonus(self) -> float:
        if not self.applications:
            return 0.0
        total = self.successes + self.failures
        if total <= 0:
            return 0.0
        ratio = self.successes / total
        return round(max(0.0, ratio - 0.5) * 0.8, 4)


@dataclass
class AppClassStat:
    quality_successes: int = 0
    quality_failures: int = 0
    exec_failures: int = 0

    @property
    def bonus(self) -> float:
        return round(self.quality_successes * 0.6, 4)

    @property
    def penalty(self) -> float:
        return round((self.quality_failures * 0.9) + (self.exec_failures * 0.4), 4)


class LayeredRoutingStore:
    """In-memory layered learning store for Route_Agent integration tests and demos."""

    def __init__(self) -> None:
        self.global_health: Dict[str, GlobalHealthStat] = defaultdict(GlobalHealthStat)
        self.shared_class_stats: Dict[Tuple[str, str], SharedClassStat] = defaultdict(SharedClassStat)
        self.app_class_stats: Dict[Tuple[str, str, str], AppClassStat] = defaultdict(AppClassStat)
        self.app_defaults: Dict[Tuple[str, str], str] = {}

    def mark_success(self, application_name: str, shared_agent_class: str, model_id: str) -> None:
        shared_key = (shared_agent_class, model_id)
        app_key = (application_name, shared_agent_class, model_id)
        self.shared_class_stats[shared_key].successes += 1
        self.shared_class_stats[shared_key].applications.add(application_name)
        self.app_class_stats[app_key].quality_successes += 1
        current_default = self.app_defaults.get((application_name, shared_agent_class))
        if current_default is None:
            self.app_defaults[(application_name, shared_agent_class)] = model_id
            return
        incumbent = self.app_class_stats[(application_name, shared_agent_class, current_default)]
        challenger = self.app_class_stats[app_key]
        incumbent_score = incumbent.bonus - incumbent.penalty
        challenger_score = challenger.bonus - challenger.penalty
        if challenger_score >= incumbent_score:
            self.app_defaults[(application_name, shared_agent_class)] = model_id

    def mark_quality_failure(self, application_name: str, shared_agent_class: str, model_id: str) -> None:
        app_key = (application_name, shared_agent_class, model_id)
        self.app_class_stats[app_key].quality_failures += 1

    def mark_exec_failure(
        self,
        application_name: str,
        shared_agent_class: str,
        model_id: str,
        *,
        provider_failure: bool = False,
        unavailable: bool = False,
    ) -> None:
        app_key = (application_name, shared_agent_class, model_id)
        self.app_class_stats[app_key].exec_failures += 1
        health = self.global_health[model_id]
        health.exec_failures += 1
        if provider_failure:
            health.provider_failures += 1
        if unavailable:
            health.unavailable = True

    def recover_model(self, model_id: str) -> None:
        self.global_health[model_id].unavailable = False

    def get_global_penalty(self, model_id: str) -> float:
        return self.global_health[model_id].penalty

    def get_shared_bonus(self, shared_agent_class: str, model_id: str) -> float:
        return self.shared_class_stats[(shared_agent_class, model_id)].bonus

    def get_app_bonus(self, application_name: str, shared_agent_class: str, model_id: str) -> float:
        return self.app_class_stats[(application_name, shared_agent_class, model_id)].bonus

    def get_app_penalty(self, application_name: str, shared_agent_class: str, model_id: str) -> float:
        return self.app_class_stats[(application_name, shared_agent_class, model_id)].penalty

    def get_default_model(self, application_name: str, shared_agent_class: str) -> str:
        return self.app_defaults.get((application_name, shared_agent_class), "")

    def snapshot(self) -> Dict[str, Any]:
        def serialize_defaultdict(source: Dict[Any, Any]) -> Dict[str, Any]:
            payload: Dict[str, Any] = {}
            for key, value in source.items():
                payload[str(key)] = dict(vars(value))
            return payload

        return {
            "global_health": serialize_defaultdict(self.global_health),
            "shared_class_stats": serialize_defaultdict(self.shared_class_stats),
            "app_class_stats": serialize_defaultdict(self.app_class_stats),
            "app_defaults": dict(self.app_defaults),
        }

    @staticmethod
    def iter_unique_models(pools: Iterable[Iterable[str]]) -> List[str]:
        ordered: List[str] = []
        seen: Set[str] = set()
        for pool in pools:
            for model_id in pool:
                value = str(model_id or "").strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                ordered.append(value)
        return ordered
