from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class RouteExecutionContext:
    workflow_id: str = ""
    run_id: str = ""
    step_id: str = ""
    parent_step_id: str = ""
    rollback_of_step_id: str = ""
    rollback_reason: str = ""
    seed: int | None = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "parent_step_id": self.parent_step_id,
            "rollback_of_step_id": self.rollback_of_step_id,
            "rollback_reason": self.rollback_reason,
            "seed": self.seed,
            "tags": list(self.tags),
        }


@dataclass(slots=True)
class RouteRequest:
    application_name: str
    shared_agent_class: str = ""
    agent_role: str = ""
    stage_name: str = ""
    system_prompt: str = ""
    task: str = ""
    requested_model: str | None = None
    llm_provider: str = ""
    execution_context: RouteExecutionContext = field(default_factory=RouteExecutionContext)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application_name": self.application_name,
            "shared_agent_class": self.shared_agent_class,
            "agent_role": self.agent_role,
            "stage_name": self.stage_name,
            "system_prompt": self.system_prompt,
            "task": self.task,
            "requested_model": self.requested_model,
            "llm_provider": self.llm_provider,
            "execution_context": self.execution_context.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RouteDecision:
    selected_model: str
    selected_provider: str = ""
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    resolved_shared_agent_class: str = ""
    class_resolution_source: str = ""
    routing_reason: str = ""
    route_latency_ms: float = 0.0
    analysis_latency_ms: float = 0.0
    selection_latency_ms: float = 0.0
    registry_latency_ms: float = 0.0
    score_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    trace_context: Dict[str, Any] = field(default_factory=dict)
    runtime_context: Any | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "selected_provider": self.selected_provider,
            "candidates": list(self.candidates),
            "resolved_shared_agent_class": self.resolved_shared_agent_class,
            "class_resolution_source": self.class_resolution_source,
            "routing_reason": self.routing_reason,
            "route_latency_ms": self.route_latency_ms,
            "analysis_latency_ms": self.analysis_latency_ms,
            "selection_latency_ms": self.selection_latency_ms,
            "registry_latency_ms": self.registry_latency_ms,
            "score_breakdown": dict(self.score_breakdown),
            "trace_context": dict(self.trace_context),
        }


@dataclass(slots=True)
class RouteScope:
    application_name: str
    shared_agent_class: str = ""
    agent_role: str = ""
    stage_name: str = ""
    system_prompt: str = ""
    task: str = ""
    execution_context: RouteExecutionContext = field(default_factory=RouteExecutionContext)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def build_request(
        self,
        *,
        task: str = "",
        system_prompt: str = "",
        requested_model: str | None = None,
        llm_provider: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RouteRequest:
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return RouteRequest(
            application_name=self.application_name,
            shared_agent_class=self.shared_agent_class,
            agent_role=self.agent_role,
            stage_name=self.stage_name,
            system_prompt=system_prompt or self.system_prompt,
            task=task or self.task,
            requested_model=requested_model,
            llm_provider=llm_provider,
            execution_context=self.execution_context,
            metadata=merged_metadata,
        )
