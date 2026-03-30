from __future__ import annotations

import asyncio
import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..client import RouteAgentClient
from ..invoker import RoutedLLMInvoker
from ..models import RouteExecutionContext, RouteRequest
from ..storage.operation_logger import OperationLogger


QUERY = "ai对就业市场的影响"
APPLICATION_NAME = "auto_research_engine"


@dataclass
class ScenarioArtifacts:
    initial_draft_path: Path
    final_draft_path: Path
    operation_log_path: Path
    initial_draft: str
    final_draft: str


class ReferenceRouteAgentScenario:
    def __init__(
        self,
        *,
        output_dir: str | Path = "research/route_agent_test",
        seed: int = 20260329,
        client: RouteAgentClient | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.random = random.Random(seed)
        self.workflow_id = "route-agent-reference"
        self.run_id = uuid.uuid4().hex
        operation_log_path = self.output_dir / "operation_log.jsonl"
        if operation_log_path.exists():
            operation_log_path.unlink()
        self.logger = OperationLogger(self.output_dir / "operation_log.jsonl")
        self.client = client or RouteAgentClient(application_name=APPLICATION_NAME)
        self.invoker = RoutedLLMInvoker(self.client, event_logger=self.logger.log)
        self._step_counter = 0
        self.requested_model = ""

    async def run(self) -> ScenarioArtifacts:
        self.logger.log(
            {
                "type": "scenario_start",
                "application_name": APPLICATION_NAME,
                "workflow_id": self.workflow_id,
                "run_id": self.run_id,
                "query": QUERY,
                "seed": self.seed,
            }
        )

        planner = await self._invoke_step(
            shared_agent_class="planner_agent",
            agent_role="planner",
            stage_name="outline_planning",
            system_prompt="You are a planner who decomposes a research query into non-overlapping section titles.",
            task=QUERY,
            payload={"query": QUERY},
        )
        sections = list(planner["sections"])
        revision_index = self.random.randrange(len(sections))
        revision_request = await self._invoke_step(
            shared_agent_class="planner_agent",
            agent_role="planner_revision",
            stage_name="outline_revision",
            system_prompt="You revise one planner section title while preserving the rest of the outline.",
            task=f"Revise section: {sections[revision_index]}",
            payload={"sections": sections, "revision_index": revision_index},
        )
        sections[revision_index] = revision_request["revised_section"]
        self.logger.log(
            {
                "type": "planner_revision_applied",
                "application_name": APPLICATION_NAME,
                "workflow_id": self.workflow_id,
                "run_id": self.run_id,
                "revised_index": revision_index,
                "revised_section": sections[revision_index],
            }
        )

        section_bodies: Dict[str, str] = {}
        for section in sections:
            body = await self._invoke_step(
                shared_agent_class="research_agent",
                agent_role="section_researcher",
                stage_name="section_research",
                system_prompt="You are a section researcher who drafts evidence-backed findings for a report section.",
                task=f"{QUERY} - {section}",
                payload={"query": QUERY, "section": section},
            )
            section_bodies[section] = body["content"]

        writer_initial = await self._invoke_step(
            shared_agent_class="writer_agent",
            agent_role="writer",
            stage_name="initial_draft",
            system_prompt="You are a report writer who assembles section findings into markdown.",
            task=QUERY,
            payload={"query": QUERY, "sections": sections, "section_bodies": section_bodies, "revision_round": 0},
        )
        initial_draft = str(writer_initial["markdown"])
        initial_path = self.output_dir / "initial_draft.md"
        initial_path.write_text(initial_draft, encoding="utf-8")
        self.logger.log(
            {
                "type": "file_written",
                "application_name": APPLICATION_NAME,
                "workflow_id": self.workflow_id,
                "run_id": self.run_id,
                "path": str(initial_path),
                "kind": "initial_draft",
            }
        )

        await self._invoke_step(
            shared_agent_class="review_agent",
            agent_role="reviewer",
            stage_name="initial_review",
            system_prompt="You are a reviewer who critiques a draft for factual rigor and structure.",
            task=QUERY,
            payload={"draft": initial_draft, "phase": "initial"},
        )

        challenger = await self._invoke_step(
            shared_agent_class="review_agent",
            agent_role="challenger",
            stage_name="challenge_generation",
            system_prompt="You are a challenger who raises targeted doubts against a draft.",
            task=QUERY,
            payload={"sections": sections, "draft": initial_draft},
        )
        challenges = list(challenger["challenges"])

        updated_sections = dict(section_bodies)
        first = challenges[0]
        self.logger.log(
            {
                "type": "rollback_chain",
                "application_name": APPLICATION_NAME,
                "workflow_id": self.workflow_id,
                "run_id": self.run_id,
                "challenge_id": first["challenge_id"],
                "nodes": ["section_researcher", "writer", "reviewer"],
            }
        )
        revised_section = await self._invoke_step(
            shared_agent_class="research_agent",
            agent_role="section_researcher",
            stage_name="rollback_research",
            system_prompt="You are a section researcher addressing a challenged section with stronger evidence.",
            task=first["question"],
            payload={"query": QUERY, "section": first["section"], "challenge": first},
            rollback_reason=first["question"],
        )
        updated_sections[first["section"]] = revised_section["content"]
        rewritten_after_first = await self._invoke_step(
            shared_agent_class="writer_agent",
            agent_role="writer",
            stage_name="rollback_rewrite",
            system_prompt="You are a report writer who rewrites sections after a challenge.",
            task=first["question"],
            payload={
                "query": QUERY,
                "sections": sections,
                "section_bodies": updated_sections,
                "revision_round": 1,
                "challenge": first,
            },
            rollback_reason=first["question"],
        )
        mid_draft = str(rewritten_after_first["markdown"])
        await self._invoke_step(
            shared_agent_class="review_agent",
            agent_role="reviewer",
            stage_name="rollback_review",
            system_prompt="You are a reviewer validating a rewritten draft after challenge remediation.",
            task=first["question"],
            payload={"draft": mid_draft, "phase": "rollback_1"},
            rollback_reason=first["question"],
        )

        second = challenges[1]
        self.logger.log(
            {
                "type": "rollback_chain",
                "application_name": APPLICATION_NAME,
                "workflow_id": self.workflow_id,
                "run_id": self.run_id,
                "challenge_id": second["challenge_id"],
                "nodes": ["writer", "reviewer"],
            }
        )
        rewritten_after_second = await self._invoke_step(
            shared_agent_class="writer_agent",
            agent_role="writer",
            stage_name="rollback_rewrite",
            system_prompt="You are a report writer who revises the narrative to answer a challenge.",
            task=second["question"],
            payload={
                "query": QUERY,
                "sections": sections,
                "section_bodies": updated_sections,
                "revision_round": 2,
                "challenge": second,
            },
            rollback_reason=second["question"],
        )
        reviewed_after_second = await self._invoke_step(
            shared_agent_class="review_agent",
            agent_role="reviewer",
            stage_name="rollback_review",
            system_prompt="You are a reviewer validating the second rollback pass.",
            task=second["question"],
            payload={"draft": rewritten_after_second["markdown"], "phase": "rollback_2"},
            rollback_reason=second["question"],
        )
        final_revision = await self._invoke_step(
            shared_agent_class="reviser_agent",
            agent_role="reviser",
            stage_name="final_polish",
            system_prompt="You are a reviser who applies final editorial polish after challenges are resolved.",
            task=QUERY,
            payload={
                "draft": rewritten_after_second["markdown"],
                "review": reviewed_after_second["review"],
                "revision_round": 3,
            },
        )
        final_draft = str(final_revision["markdown"])
        final_path = self.output_dir / "final_draft.md"
        final_path.write_text(final_draft, encoding="utf-8")
        self.logger.log(
            {
                "type": "file_written",
                "application_name": APPLICATION_NAME,
                "workflow_id": self.workflow_id,
                "run_id": self.run_id,
                "path": str(final_path),
                "kind": "final_draft",
            }
        )
        self.logger.log(
            {
                "type": "scenario_completed",
                "application_name": APPLICATION_NAME,
                "workflow_id": self.workflow_id,
                "run_id": self.run_id,
                "initial_draft_length": len(initial_draft),
                "final_draft_length": len(final_draft),
            }
        )
        return ScenarioArtifacts(
            initial_draft_path=initial_path,
            final_draft_path=final_path,
            operation_log_path=self.output_dir / "operation_log.jsonl",
            initial_draft=initial_draft,
            final_draft=final_draft,
        )

    async def _invoke_step(
        self,
        *,
        shared_agent_class: str,
        agent_role: str,
        stage_name: str,
        system_prompt: str,
        task: str,
        payload: Dict[str, Any],
        rollback_reason: str = "",
    ) -> Dict[str, Any]:
        step_id = self._next_step_id(agent_role)
        execution_context = RouteExecutionContext(
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            step_id=step_id,
            rollback_reason=rollback_reason,
            rollback_of_step_id=payload.get("rollback_of_step_id") or "",
            seed=self.seed,
            tags=[agent_role, stage_name],
        )
        route_request = RouteRequest(
            application_name=APPLICATION_NAME,
            shared_agent_class=shared_agent_class,
            agent_role=agent_role,
            stage_name=stage_name,
            system_prompt=system_prompt,
            task=task,
            requested_model=self.requested_model,
            llm_provider="",
            execution_context=execution_context,
            metadata={"payload": payload},
        )

        async def provider_call(selected_model: str, selected_provider: str = "") -> Dict[str, Any]:
            await asyncio.sleep(0)
            return self._synthetic_response(
                agent_role=agent_role,
                stage_name=stage_name,
                selected_model=selected_model,
                payload=payload,
            )

        result = await self.invoker.invoke(
            provider_call=provider_call,
            requested_model=self.requested_model,
            llm_provider="",
            route_request=route_request,
            metadata={"payload": payload},
        )
        self.logger.log(
            {
                "type": "step_completed",
                "application_name": APPLICATION_NAME,
                "workflow_id": self.workflow_id,
                "run_id": self.run_id,
                "step_id": step_id,
                "agent_role": agent_role,
                "stage_name": stage_name,
                "output_summary": self._summarize_output(result),
            }
        )
        return result

    def _synthetic_response(
        self,
        *,
        agent_role: str,
        stage_name: str,
        selected_model: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if agent_role == "planner":
            return {
                "sections": [
                    "就业结构重塑",
                    "岗位两极分化",
                    "技能迁移与再培训",
                    "区域与行业分化",
                ],
                "model_used": selected_model,
            }
        if agent_role == "planner_revision":
            sections = list(payload.get("sections") or [])
            index = int(payload.get("revision_index") or 0)
            section = sections[index]
            return {
                "revised_section": f"{section}与政策响应",
                "model_used": selected_model,
            }
        if agent_role == "section_researcher":
            section = str(payload.get("section") or "未命名章节")
            challenge = payload.get("challenge")
            suffix = "补充了更强的证据链与反例说明。" if challenge else "给出趋势、风险和岗位样本。"
            return {
                "content": (
                    f"## {section}\n"
                    f"- AI 正在重塑 {section} 对应的岗位任务边界，企业更偏好能协同使用自动化工具的人才。\n"
                    f"- 在高重复工作中，岗位会被拆解并重组；在复杂岗位中，人机协作能力成为增值项。\n"
                    f"- {suffix}\n"
                ),
                "model_used": selected_model,
            }
        if agent_role == "writer":
            sections = list(payload.get("sections") or [])
            section_bodies = dict(payload.get("section_bodies") or {})
            revision_round = int(payload.get("revision_round") or 0)
            challenge = payload.get("challenge") or {}
            challenge_note = ""
            if challenge:
                challenge_note = f"\n> 已响应质疑：{challenge.get('question')}\n"
            intro = (
                "# AI对就业市场的影响\n\n"
                "## 引言\n"
                "AI 对就业市场的影响并不是简单的岗位替代，而是岗位重组、技能迁移和组织流程再设计的叠加效应。"
            )
            body = "\n\n".join(section_bodies.get(section, f"## {section}\n- 待补充") for section in sections)
            conclusion = (
                "\n\n## 结论\n"
                "更稳健的判断是：AI 会扩大岗位分化，抬高技能迁移速度，并迫使企业与政策系统同步调整培训和安全网。"
            )
            if revision_round >= 1:
                conclusion += "\n修订版进一步强调了证据边界、行业差异与政策响应。"
            if revision_round >= 2:
                conclusion += "\n同时补充了对中短期冲击与长期生产率收益之间张力的讨论。"
            return {
                "markdown": f"{intro}\n\n{body}{challenge_note}{conclusion}\n",
                "model_used": selected_model,
            }
        if agent_role == "reviewer":
            phase = str(payload.get("phase") or "")
            if phase == "initial":
                review = "需要加强对岗位两极分化与政策响应的证据衔接。"
            else:
                review = None
            return {"review": review, "model_used": selected_model}
        if agent_role == "challenger":
            sections = list(payload.get("sections") or [])
            picks = self.random.sample(range(len(sections)), 2)
            challenges = []
            for idx, pick in enumerate(picks, start=1):
                challenges.append(
                    {
                        "challenge_id": f"challenge_{idx}",
                        "section": sections[pick],
                        "question": f"请质疑章节“{sections[pick]}”中关于 AI 对就业影响的证据是否足够具体。",
                    }
                )
            self.logger.log(
                {
                    "type": "challenges_generated",
                    "application_name": APPLICATION_NAME,
                    "workflow_id": self.workflow_id,
                    "run_id": self.run_id,
                    "challenges": challenges,
                }
            )
            return {"challenges": challenges, "model_used": selected_model}
        if agent_role == "reviser":
            draft = str(payload.get("draft") or "")
            review = str(payload.get("review") or "").strip()
            addition = "\n\n## 修订说明\n- 已根据质疑补强证据链。\n- 已对存在歧义的判断补充边界条件。"
            if review:
                addition += f"\n- 审阅备注：{review}"
            return {"markdown": f"{draft}{addition}\n", "model_used": selected_model}
        return {"text": json.dumps(payload, ensure_ascii=False), "model_used": selected_model}

    def _summarize_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for key, value in result.items():
            if isinstance(value, str):
                summary[key] = value[:120]
            elif isinstance(value, list):
                summary[key] = f"list[{len(value)}]"
            elif isinstance(value, dict):
                summary[key] = f"dict[{len(value)}]"
            else:
                summary[key] = value
        return summary

    def _next_step_id(self, agent_role: str) -> str:
        self._step_counter += 1
        return f"{self._step_counter:03d}_{agent_role}"


async def run_reference_application(
    *,
    output_dir: str | Path = "research/route_agent_test",
    seed: int = 20260329,
) -> ScenarioArtifacts:
    scenario = ReferenceRouteAgentScenario(output_dir=output_dir, seed=seed)
    return await scenario.run()
