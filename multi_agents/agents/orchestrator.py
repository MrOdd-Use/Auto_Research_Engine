import asyncio
import os
import time
import datetime
import json
import copy
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
from .utils.views import print_agent_output
from ..memory.research import ResearchState
from .utils.utils import sanitize_filename

# Import agent classes
from . import \
    WriterAgent, \
    EditorAgent, \
    PublisherAgent, \
    ResearchAgent, \
    HumanAgent, \
    ReviewerAgent, \
    ReviserAgent
from .claim_verifier import ClaimVerifierAgent

from multi_agents.route_agent.invoker import get_global_invoker
from multi_agents.workflow_session import WorkflowSessionRecorder
from .utils.output_writers import write_model_decisions


class ChiefEditorAgent:
    """Agent responsible for managing and coordinating editing tasks."""

    def __init__(self, task: dict, websocket=None, stream_output=None, tone=None, headers=None, output_dir=None):
        self.task = task
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tone = tone
        self.task_id = self._generate_task_id()
        self.output_dir = str(output_dir) if output_dir else self._create_output_directory()
        self._workflow_agents = {}

    def _generate_task_id(self):
        # Currently time based, but can be any unique identifier
        return int(time.time())

    def _create_output_directory(self):
        output_dir = "./outputs/" + \
            sanitize_filename(
                f"run_{self.task_id}_{self.task.get('query')[0:40]}")

        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _initialize_agents(self):
        return {
            "writer": WriterAgent(self.websocket, self.stream_output, self.headers),
            "editor": EditorAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "research": ResearchAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "publisher": PublisherAgent(self.output_dir, self.websocket, self.stream_output, self.headers),
            "human": HumanAgent(self.websocket, self.stream_output, self.headers),
            "reviewer": ReviewerAgent(self.websocket, self.stream_output, self.headers),
            "reviser": ReviserAgent(self.websocket, self.stream_output, self.headers),
            "claim_verifier": ClaimVerifierAgent(self.websocket, self.stream_output, self.headers),
        }

    def _create_workflow(self, agents):
        self._workflow_agents = agents
        workflow = StateGraph(ResearchState)

        # Add nodes for each agent
        workflow.add_node("browser", agents["research"].run_initial_research)
        workflow.add_node("planner", agents["editor"].plan_research)
        workflow.add_node("researcher", agents["editor"].run_parallel_research)
        workflow.add_node("writer", agents["writer"].run)
        workflow.add_node("reviewer", self._run_final_reviewer)
        workflow.add_node("reviser", self._run_final_reviser)
        workflow.add_node("publisher", agents["publisher"].run)
        workflow.add_node("human", agents["human"].review_plan)

        # Add edges
        self._add_workflow_edges(workflow)

        return workflow

    def _add_workflow_edges(self, workflow):
        workflow.add_edge('browser', 'planner')
        workflow.add_edge('planner', 'human')
        workflow.add_edge('researcher', 'writer')
        workflow.add_edge('writer', 'reviewer')
        workflow.add_conditional_edges(
            'reviewer',
            lambda state: "accept" if state.get("review") is None else "revise",
            {"accept": "publisher", "revise": "reviser"}
        )
        workflow.add_conditional_edges(
            'reviser',
            lambda state: "force_publish" if self._is_review_cap_reached(state) else "continue_review",
            {"force_publish": "publisher", "continue_review": "reviewer"},
        )
        workflow.set_entry_point("browser")
        workflow.add_edge('publisher', END)

        # Add human in the loop
        workflow.add_conditional_edges(
            'human',
            lambda review: "accept" if review['human_feedback'] is None else "revise",
            {"accept": "researcher", "revise": "planner"}
        )

    @staticmethod
    def _normalize_max_review_rounds(value: object) -> int:
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            parsed = 3
        return max(parsed, 1)

    def _is_review_cap_reached(self, research_state: dict) -> bool:
        task = research_state.get("task") or {}
        max_rounds = self._normalize_max_review_rounds(task.get("max_review_rounds", 3))
        iterations = research_state.get("review_iterations", 0)
        try:
            iterations = int(iterations)
        except (TypeError, ValueError):
            iterations = 0
        return iterations >= max_rounds

    async def _run_final_reviewer(self, research_state: dict):
        reviewer = self._workflow_agents["reviewer"]
        publisher = self._workflow_agents["publisher"]
        draft_text = self._normalize_draft_text(research_state.get("final_draft"))
        if not draft_text:
            draft_text = publisher.generate_layout(research_state)

        review_result = await reviewer.run(
            {
                "task": research_state.get("task"),
                "draft": draft_text,
                "revision_notes": research_state.get("revision_notes"),
                "source_index": research_state.get("source_index") or {},
                "previous_draft": research_state.get("_draft_before_revision") or "",
            }
        )
        return {
            "review": review_result.get("review"),
            "final_draft": draft_text,
        }

    async def _run_final_reviser(self, research_state: dict):
        reviser = self._workflow_agents["reviser"]
        publisher = self._workflow_agents["publisher"]

        current_draft = self._normalize_draft_text(research_state.get("final_draft"))
        if not current_draft:
            current_draft = publisher.generate_layout(research_state)

        revision_result = await reviser.run(
            {
                "task": research_state.get("task"),
                "draft": current_draft,
                "review": research_state.get("review"),
                "revision_notes": research_state.get("revision_notes"),
            }
        )

        revised_text = self._normalize_draft_text(revision_result.get("draft")) or current_draft
        iterations = research_state.get("review_iterations", 0)
        try:
            iterations = int(iterations)
        except (TypeError, ValueError):
            iterations = 0
        return {
            "final_draft": revised_text,
            "revision_notes": revision_result.get("revision_notes"),
            "review_iterations": iterations + 1,
            "claim_annotations": None,
            "claim_confidence_report": [],
            "suspicious_claims": [],
            "hallucinated_claims": [],
        }

    async def _annotate_if_needed(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh claim verification if needed, then annotate the final draft."""
        state = await self._refresh_claim_report_from_final_draft(state)

        verifier = self._workflow_agents.get("claim_verifier")
        if verifier is not None:
            report = state.get("claim_confidence_report") or []
            if report:
                draft = self._normalize_draft_text(state.get("final_draft"))
                if draft:
                    state["final_draft"] = verifier.annotate_draft(draft, report)
        return state

    def _normalize_draft_text(self, draft: object) -> str:
        if draft is None:
            return ""
        if isinstance(draft, str):
            return draft.strip()
        if isinstance(draft, dict):
            return json.dumps(draft, ensure_ascii=False, indent=2)
        if isinstance(draft, list):
            return "\n".join(str(item) for item in draft)
        return str(draft).strip()

    def _prepare_for_writer_pass(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Drop stale downstream artifacts before recomputing the narrative."""
        for key in (
            "final_draft",
            "report",
            "review",
            "revision_notes",
            "_draft_before_revision",
            "claim_annotations",
            "claim_confidence_report",
            "suspicious_claims",
            "hallucinated_claims",
        ):
            state.pop(key, None)
        state["review_iterations"] = 0
        state["claim_reflexion_iterations"] = 0
        return state

    @staticmethod
    def _normalize_heading_label(heading: str) -> str:
        normalized = re.sub(r"\s+", " ", str(heading or "").strip().lower())
        return normalized.rstrip(":：")

    @classmethod
    def _parse_markdown_sections(cls, draft: str) -> List[Dict[str, str]]:
        sections: List[Dict[str, str]] = []
        current_heading: str | None = None
        current_lines: List[str] = []

        for raw_line in str(draft or "").splitlines():
            heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*$", raw_line)
            if heading_match:
                if current_heading is not None:
                    sections.append(
                        {
                            "heading": current_heading,
                            "body": "\n".join(current_lines).strip(),
                        }
                    )
                current_heading = str(heading_match.group(1) or "").strip()
                current_lines = []
                continue

            if current_heading is not None:
                current_lines.append(raw_line)

        if current_heading is not None:
            sections.append(
                {
                    "heading": current_heading,
                    "body": "\n".join(current_lines).strip(),
                }
            )

        return sections

    @classmethod
    def _extract_report_section(cls, draft: str, heading: str) -> str:
        normalized_heading = cls._normalize_heading_label(heading)
        if not str(draft or "").strip() or not normalized_heading:
            return ""

        for section in cls._parse_markdown_sections(draft):
            candidate_heading = cls._normalize_heading_label(section.get("heading") or "")
            if candidate_heading == normalized_heading:
                return str(section.get("body") or "").strip()
        return ""

    @classmethod
    def _build_claim_review_fallback_text(cls, draft: str, headers: Dict[str, Any]) -> str:
        excluded = {
            cls._normalize_heading_label(headers.get("title") or ""),
            cls._normalize_heading_label(headers.get("date") or ""),
            cls._normalize_heading_label(headers.get("table_of_contents") or "Table of Contents"),
            cls._normalize_heading_label(headers.get("references") or "References"),
        }
        bodies: List[str] = []

        for section in cls._parse_markdown_sections(draft):
            heading = cls._normalize_heading_label(section.get("heading") or "")
            if heading in excluded:
                continue
            body = str(section.get("body") or "").strip()
            if body:
                bodies.append(body)

        if bodies:
            return "\n\n".join(bodies).strip()

        cleaned_lines: List[str] = []
        for line in str(draft or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^\s{0,3}#{1,6}\s+", stripped):
                continue
            if stripped.lower().startswith(("source:", "target:")):
                continue
            cleaned_lines.append(stripped)
        return "\n".join(cleaned_lines).strip()

    async def _refresh_claim_report_from_final_draft(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Re-verify intro/conclusion claims after the Reviser changes the final draft."""
        verifier = self._workflow_agents.get("claim_verifier")
        if verifier is None:
            return state

        draft = self._normalize_draft_text(state.get("final_draft"))
        if not draft:
            return state

        headers = state.get("headers") or {}
        introduction = self._extract_report_section(
            draft,
            str(headers.get("introduction") or "Introduction"),
        )
        conclusion = self._extract_report_section(
            draft,
            str(headers.get("conclusion") or "Conclusion"),
        )

        if not introduction and not conclusion:
            fallback_text = self._build_claim_review_fallback_text(draft, headers)
            if not fallback_text:
                state["claim_annotations"] = None
                state["claim_confidence_report"] = []
                state["suspicious_claims"] = []
                state["hallucinated_claims"] = []
                return state
            introduction = fallback_text

        verify_state = copy.deepcopy(state)
        verify_state["introduction"] = introduction or str(state.get("introduction") or "")
        verify_state["conclusion"] = conclusion or str(state.get("conclusion") or "")
        verify_state["claim_annotations"] = None

        result = await verifier.run(verify_state)
        state.update(result)
        if introduction:
            state["introduction"] = introduction
        if conclusion:
            state["conclusion"] = conclusion
        state["claim_annotations"] = None
        return state

    def init_research_team(self):
        """Initialize and create a workflow for the research team."""
        agents = self._initialize_agents()
        return self._create_workflow(agents)

    async def _log_research_start(self):
        message = f"Starting the research process for query '{self.task.get('query')}'..."
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "starting_research", message, self.websocket)
        else:
            print_agent_output(message, "MASTER")

    async def _run_global_node(
        self,
        node_name: str,
        runner: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        state: Dict[str, Any],
        recorder: WorkflowSessionRecorder | None,
        *,
        note: str | None = None,
        rerunnable: bool = True,
    ) -> Dict[str, Any]:
        state_before = copy.deepcopy(state)
        self._inject_global_note(node_name, state, note)
        output_delta = await runner(state)
        state.update(output_delta or {})
        self._clear_ephemeral_global_note(state)
        if recorder is not None:
            await recorder.record_global_checkpoint(
                node_name,
                state_before=state_before,
                output_delta=output_delta or {},
                state_after=state,
                summary=self._summarize_global_output(node_name, output_delta or {}, state),
                rerunnable=rerunnable,
            )
        return state

    async def _run_browser(self, state: Dict[str, Any], recorder: WorkflowSessionRecorder | None, note: str | None = None) -> Dict[str, Any]:
        return await self._run_global_node(
            "browser",
            self._workflow_agents["research"].run_initial_research,
            state,
            recorder,
            note=note,
        )

    async def _run_planner_loop(
        self,
        state: Dict[str, Any],
        recorder: WorkflowSessionRecorder | None,
        *,
        include_human_feedback: bool,
        planner_note: str | None = None,
    ) -> Dict[str, Any]:
        while True:
            state = await self._run_global_node(
                "planner",
                self._workflow_agents["editor"].plan_research,
                state,
                recorder,
                note=planner_note,
            )
            planner_note = None
            if not include_human_feedback:
                state["human_feedback"] = None
                break
            state = await self._run_global_node(
                "human",
                self._workflow_agents["human"].review_plan,
                state,
                recorder,
                rerunnable=False,
            )
            if state.get("human_feedback") is None:
                break
        return state

    async def _run_researcher(
        self,
        state: Dict[str, Any],
        recorder: WorkflowSessionRecorder | None,
        *,
        note: str | None = None,
        selected_section_key: str | None = None,
        section_start_node: str | None = None,
        section_state_before: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        state_before = copy.deepcopy(state)
        editor = self._workflow_agents["editor"]
        run_parallel_research = editor.run_parallel_research
        try:
            output_delta = await run_parallel_research(
                state,
                session_recorder=recorder,
                selected_section_key=selected_section_key,
                start_from_section_node=section_start_node,
                section_state_before=section_state_before,
                note=note,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            output_delta = await run_parallel_research(state)
        state.update(output_delta or {})
        if recorder is not None:
            await recorder.record_global_checkpoint(
                "researcher",
                state_before=state_before,
                output_delta=output_delta or {},
                state_after=state,
                summary=self._summarize_global_output("researcher", output_delta or {}, state),
            )
        return state

    async def _run_review_cycle(
        self,
        state: Dict[str, Any],
        recorder: WorkflowSessionRecorder | None,
        *,
        start_node: str = "reviewer",
        reviewer_note: str | None = None,
        reviser_note: str | None = None,
    ) -> Dict[str, Any]:
        current_node = start_node
        pending_reviewer_note = reviewer_note
        pending_reviser_note = reviser_note

        while True:
            if current_node == "reviewer":
                state = await self._run_global_node(
                    "reviewer",
                    self._run_final_reviewer,
                    state,
                    recorder,
                    note=pending_reviewer_note,
                )
                pending_reviewer_note = None
                if state.get("review") is None:
                    break
                current_node = "reviser"
                continue

            # Snapshot draft before revision so Reviewer can diff on next round
            state["_draft_before_revision"] = str(state.get("final_draft") or "")

            state = await self._run_global_node(
                "reviser",
                self._run_final_reviser,
                state,
                recorder,
                note=pending_reviser_note,
            )
            pending_reviser_note = None
            if self._is_review_cap_reached(state):
                break
            current_node = "reviewer"

        return state

    @staticmethod
    def _fallback_section_key(section_index: int, header: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(header or "").strip().lower()).strip("_")
        return f"section_{section_index}_{cleaned[:48] or 'section'}"

    def _build_section_contexts(
        self,
        state: Dict[str, Any],
        section_keys: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        section_details = list(state.get("section_details") or [])
        editor = self._workflow_agents.get("editor")
        contexts: List[Dict[str, Any]] = []

        for index, detail in enumerate(section_details):
            if not isinstance(detail, dict):
                continue
            header = str(detail.get("header") or "").strip()
            if editor is not None and hasattr(editor, "_make_section_key"):
                try:
                    section_key = str(editor._make_section_key(index, header))
                except Exception:
                    section_key = self._fallback_section_key(index, header)
            else:
                section_key = self._fallback_section_key(index, header)

            if section_keys is not None and section_key not in section_keys:
                continue

            contexts.append(
                {
                    "section_index": index,
                    "section_key": section_key,
                    "section_title": header or f"Section {index + 1}",
                }
            )

        return contexts

    @staticmethod
    def _merge_section_rerun_output(
        state: Dict[str, Any],
        rerun_result: Dict[str, Any],
        *,
        section_index: int,
    ) -> None:
        for key in ("research_data", "scraping_packets", "check_data_reports"):
            new_values = rerun_result.get(key)
            if not isinstance(new_values, list) or section_index >= len(new_values):
                continue
            merged = list(state.get(key) or [])
            while len(merged) <= section_index:
                merged.append(None)
            merged[section_index] = new_values[section_index]
            state[key] = merged

    @staticmethod
    def _next_source_id(source_index: Dict[str, Any]) -> int:
        max_id = 0
        for key in (source_index or {}).keys():
            try:
                max_id = max(max_id, int(str(key).lstrip("S")))
            except ValueError:
                continue
        return max_id + 1

    @staticmethod
    def _build_writer_reflexion_note(suspicious_claims: List[dict]) -> str:
        lines = [
            "Refresh the introduction and conclusion using the updated evidence index.",
            "Resolve the suspicious or conflicting claims below and regenerate citations and claim_annotations.",
            "Prefer claims backed by non-conflicting evidence; if still unresolved, omit the claim or use the unavailable marker.",
        ]
        for claim in suspicious_claims[:6]:
            claim_text = str(claim.get("claim_text") or "").strip()
            note = str(claim.get("note") or "").strip()
            if claim_text:
                detail = f'- "{claim_text}"'
                if note:
                    detail = f"{detail} ({note})"
                lines.append(detail)
        return "\n".join(lines)

    async def _inject_source_index(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build source index from scraping_packets and inject into state for writer."""
        verifier = self._workflow_agents.get("claim_verifier")
        if verifier is None:
            return state
        scraping_packets = state.get("scraping_packets") or []
        if not scraping_packets:
            return state
        section_contexts = self._build_section_contexts(state)
        source_index, _ = verifier.build_source_index(
            scraping_packets,
            section_contexts=section_contexts,
        )
        state["source_index"] = source_index
        state["indexed_research_data"] = verifier.format_source_index(source_index)
        return state

    async def _run_claim_review(
        self,
        state: Dict[str, Any],
        recorder: WorkflowSessionRecorder | None,
    ) -> Dict[str, Any]:
        """Verify writer claims and trigger reflexion for suspicious ones."""
        verifier = self._workflow_agents.get("claim_verifier")
        if verifier is None:
            return state

        state_before = copy.deepcopy(state)

        for iteration in range(verifier.MAX_REFLEXION):
            result = await verifier.run(state)
            state.update(result)
            state["claim_reflexion_iterations"] = iteration + 1

            suspicious = result.get("suspicious_claims", [])
            if not suspicious:
                break

            # Group by section, parallel rerun via existing checkpoint mechanism
            sections = verifier.group_by_section(suspicious, state.get("source_index", {}))
            if not sections:
                break

            section_contexts = self._build_section_contexts(state, set(sections.keys()))
            context_by_key = {context["section_key"]: context for context in section_contexts}
            rerun_tasks = []
            for section_key, claims in sections.items():
                if section_key not in context_by_key:
                    continue
                note = verifier.build_reflexion_note(claims, state.get("source_index", {}))
                rerun_tasks.append(
                    (
                        section_key,
                        self._run_researcher(
                            copy.deepcopy(state),
                            recorder,
                            selected_section_key=section_key,
                            note=note,
                        ),
                    )
                )
            if not rerun_tasks:
                break
            if rerun_tasks:
                results = await asyncio.gather(*[task for _, task in rerun_tasks])
                for (section_key, _), rerun_result in zip(rerun_tasks, results):
                    context = context_by_key.get(section_key)
                    if context is None:
                        continue
                    self._merge_section_rerun_output(
                        state,
                        rerun_result,
                        section_index=int(context["section_index"]),
                    )

            # Incrementally append only newly fetched evidence for the rerun sections.
            existing_index = dict(state.get("source_index") or {})
            selected_packets = []
            selected_contexts = []
            for context in section_contexts:
                section_index = int(context["section_index"])
                packets = state.get("scraping_packets") or []
                if section_index >= len(packets):
                    continue
                selected_packets.append(packets[section_index])
                selected_contexts.append(context)

            new_index, _ = verifier.build_source_index(
                selected_packets,
                start_id=self._next_source_id(existing_index),
                section_contexts=selected_contexts,
                existing_index=existing_index,
            )
            existing_index.update(new_index)
            state["source_index"] = existing_index
            state["indexed_research_data"] = verifier.format_source_index(existing_index)

            # Rewrite against the refreshed index so citations and claim annotations can move to new evidence.
            state = await self._run_global_node(
                "writer",
                self._workflow_agents["writer"].run,
                state,
                recorder,
                note=self._build_writer_reflexion_note(suspicious),
            )

        # Record checkpoint
        if recorder is not None:
            await recorder.record_global_checkpoint(
                "claim_review",
                state_before=state_before,
                output_delta={
                    "claim_confidence_report": state.get("claim_confidence_report"),
                    "claim_reflexion_iterations": state.get("claim_reflexion_iterations"),
                },
                state_after=state,
                summary=self._summarize_global_output("claim_review", {}, state),
            )

        return state

    async def _run_writer_and_verify(
        self,
        state: Dict[str, Any],
        recorder: WorkflowSessionRecorder | None,
        *,
        writer_note: str | None = None,
    ) -> Dict[str, Any]:
        """Run writer → claim_review → review_cycle (with source-aware Reviewer) → annotate."""
        # Inject source index before writer
        state = self._prepare_for_writer_pass(state)
        state = await self._inject_source_index(state)

        # Run writer
        state = await self._run_global_node(
            "writer",
            self._workflow_agents["writer"].run,
            state,
            recorder,
            note=writer_note,
        )

        # Claim verification + reflexion
        state = await self._run_claim_review(state, recorder)

        # Review cycle — Reviewer now handles hallucination detection via diff + source_index
        state = await self._run_review_cycle(state, recorder)

        # Annotate AFTER review cycle (clean text during review)
        return await self._annotate_if_needed(state)

    async def _execute_workflow(
        self,
        *,
        initial_state: Dict[str, Any],
        recorder: WorkflowSessionRecorder | None = None,
        start_node: str = "browser",
        include_human_feedback: bool = True,
        note: str | None = None,
        selected_section_key: str | None = None,
        section_start_node: str | None = None,
        section_state_before: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        state = copy.deepcopy(initial_state)

        if start_node == "browser":
            state = await self._run_browser(state, recorder, note=note)
            state = await self._run_planner_loop(
                state,
                recorder,
                include_human_feedback=include_human_feedback,
            )
            state = await self._run_researcher(state, recorder)
            state = await self._run_writer_and_verify(state, recorder)
            state = await self._run_global_node("publisher", self._workflow_agents["publisher"].run, state, recorder)
            return state

        if start_node in {"planner", "human"}:
            state = await self._run_planner_loop(
                state,
                recorder,
                include_human_feedback=include_human_feedback,
                planner_note=note if start_node == "planner" else None,
            )
            state = await self._run_researcher(state, recorder)
            state = await self._run_writer_and_verify(state, recorder)
            state = await self._run_global_node("publisher", self._workflow_agents["publisher"].run, state, recorder)
            return state

        if start_node == "researcher":
            state = await self._run_researcher(
                state,
                recorder,
                note=note,
                selected_section_key=selected_section_key,
                section_start_node=section_start_node,
                section_state_before=section_state_before,
            )
            state = await self._run_writer_and_verify(state, recorder)
            state = await self._run_global_node("publisher", self._workflow_agents["publisher"].run, state, recorder)
            return state

        if start_node == "writer":
            state = await self._run_writer_and_verify(state, recorder, writer_note=note)
            state = await self._run_global_node("publisher", self._workflow_agents["publisher"].run, state, recorder)
            return state

        if start_node == "reviewer":
            state = await self._run_review_cycle(state, recorder, start_node="reviewer", reviewer_note=note)
            state = await self._annotate_if_needed(state)
            state = await self._run_global_node("publisher", self._workflow_agents["publisher"].run, state, recorder)
            return state

        if start_node == "reviser":
            state = await self._run_review_cycle(state, recorder, start_node="reviser", reviser_note=note)
            state = await self._annotate_if_needed(state)
            state = await self._run_global_node("publisher", self._workflow_agents["publisher"].run, state, recorder)
            return state

        if start_node == "publisher":
            state = await self._run_global_node(
                "publisher",
                self._workflow_agents["publisher"].run,
                state,
                recorder,
                note=note,
            )
            return state

        raise ValueError(f"Unsupported workflow start node: {start_node}")

    def _inject_global_note(self, node_name: str, state: Dict[str, Any], note: str | None) -> None:
        if not note:
            return
        task = state.setdefault("task", {})
        if node_name == "planner":
            task["include_human_feedback"] = True
            state["human_feedback"] = note
            return
        if node_name == "researcher":
            existing = str(state.get("extra_hints") or "").strip()
            state["extra_hints"] = f"{existing}\n{note}".strip() if existing else note
            return
        if node_name == "browser":
            query = str(task.get("query") or "").strip()
            task["query"] = f"{query}\nAdditional rerun instruction: {note}".strip()
            return
        task["checkpoint_note"] = note
        task["checkpoint_target"] = node_name

    def _clear_ephemeral_global_note(self, state: Dict[str, Any]) -> None:
        task = state.get("task")
        if isinstance(task, dict):
            task.pop("checkpoint_note", None)
            task.pop("checkpoint_target", None)

    def _summarize_global_output(
        self,
        node_name: str,
        output_delta: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        if node_name == "planner":
            return {
                "sections": list(state_after.get("sections") or []),
                "section_count": len(state_after.get("section_details") or []),
            }
        if node_name == "researcher":
            return {
                "research_items": len(output_delta.get("research_data") or []),
                "scraping_packets": len(output_delta.get("scraping_packets") or []),
                "check_data_reports": len(output_delta.get("check_data_reports") or []),
            }
        if node_name == "writer":
            return {
                "sources": len(output_delta.get("sources") or []),
                "has_introduction": bool(output_delta.get("introduction")),
                "has_conclusion": bool(output_delta.get("conclusion")),
            }
        if node_name == "reviewer":
            return {
                "accepted": state_after.get("review") is None,
                "review_iterations": state_after.get("review_iterations", 0),
            }
        if node_name == "reviser":
            return {
                "review_iterations": state_after.get("review_iterations", 0),
                "has_revision_notes": bool(output_delta.get("revision_notes")),
            }
        if node_name == "publisher":
            report = str(output_delta.get("report") or "")
            return {"report_length": len(report)}
        if node_name == "browser":
            report = str(output_delta.get("initial_research") or "")
            return {"initial_research_length": len(report)}
        if node_name == "claim_review":
            claim_report = state_after.get("claim_confidence_report") or []
            return {
                "total_claims": len(claim_report),
                "high": sum(1 for c in claim_report if c.get("confidence") == "HIGH"),
                "medium": sum(1 for c in claim_report if c.get("confidence") == "MEDIUM"),
                "suspicious": sum(1 for c in claim_report if c.get("confidence") == "SUSPICIOUS"),
                "hallucination": sum(1 for c in claim_report if c.get("confidence") == "HALLUCINATION"),
                "reflexion_rounds": state_after.get("claim_reflexion_iterations", 0),
            }
        return {"keys": sorted((output_delta or {}).keys())}

    async def run_research_task(
        self,
        task_id=None,
        *,
        session_recorder: WorkflowSessionRecorder | None = None,
        start_node: str = "browser",
        initial_state: Optional[Dict[str, Any]] = None,
        include_human_feedback: bool = True,
        note: str | None = None,
        selected_section_key: str | None = None,
        section_start_node: str | None = None,
        section_state_before: Dict[str, Any] | None = None,
    ):
        """
        Run a research task with the initialized research team.

        Args:
            task_id (optional): The ID of the task to run.

        Returns:
            The result of the research task.
        """
        agents = self._initialize_agents()
        self._workflow_agents = agents

        await self._log_research_start()

        # Collect route decisions via event_logger wrapper
        decisions: List[Dict[str, Any]] = []
        invoker = get_global_invoker()
        original_logger = invoker.event_logger

        def _collecting_logger(payload: Dict[str, Any]) -> None:
            if payload.get("type") == "route_decision":
                decisions.append(payload)
            if original_logger is not None:
                original_logger(payload)

        invoker.event_logger = _collecting_logger

        base_state = initial_state or {"task": copy.deepcopy(self.task)}
        try:
            result = await self._execute_workflow(
                initial_state=base_state,
                recorder=session_recorder,
                start_node=start_node,
                include_human_feedback=include_human_feedback,
                note=note,
                selected_section_key=selected_section_key,
                section_start_node=section_start_node,
                section_state_before=section_state_before,
            )
            if session_recorder is not None:
                await session_recorder.mark_completed(
                    final_state=result,
                    answer=str(result.get("report") or ""),
                )
            # Write model decisions log
            await write_model_decisions(self.output_dir, decisions)
            return result
        except Exception as exc:
            if session_recorder is not None:
                await session_recorder.mark_failed(str(exc))
            # Write partial decisions even on failure
            if decisions:
                await write_model_decisions(self.output_dir, decisions)
            raise
        finally:
            invoker.event_logger = original_logger
