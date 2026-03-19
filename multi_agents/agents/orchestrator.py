import os
import time
import datetime
import json
import copy
from typing import Any, Awaitable, Callable, Dict, Optional
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

from multi_agents.workflow_session import WorkflowSessionRecorder


class ChiefEditorAgent:
    """Agent responsible for managing and coordinating editing tasks."""

    def __init__(self, task: dict, websocket=None, stream_output=None, tone=None, headers=None):
        self.task = task
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tone = tone
        self.task_id = self._generate_task_id()
        self.output_dir = self._create_output_directory()
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
        }

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
            state = await self._run_global_node("writer", self._workflow_agents["writer"].run, state, recorder)
            state = await self._run_review_cycle(state, recorder)
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
            state = await self._run_global_node("writer", self._workflow_agents["writer"].run, state, recorder)
            state = await self._run_review_cycle(state, recorder)
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
            state = await self._run_global_node("writer", self._workflow_agents["writer"].run, state, recorder)
            state = await self._run_review_cycle(state, recorder)
            state = await self._run_global_node("publisher", self._workflow_agents["publisher"].run, state, recorder)
            return state

        if start_node == "writer":
            state = await self._run_global_node(
                "writer",
                self._workflow_agents["writer"].run,
                state,
                recorder,
                note=note,
            )
            state = await self._run_review_cycle(state, recorder)
            state = await self._run_global_node("publisher", self._workflow_agents["publisher"].run, state, recorder)
            return state

        if start_node == "reviewer":
            state = await self._run_review_cycle(state, recorder, start_node="reviewer", reviewer_note=note)
            state = await self._run_global_node("publisher", self._workflow_agents["publisher"].run, state, recorder)
            return state

        if start_node == "reviser":
            state = await self._run_review_cycle(state, recorder, start_node="reviser", reviser_note=note)
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
                "scrap_packets": len(output_delta.get("scrap_packets") or []),
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
            return result
        except Exception as exc:
            if session_recorder is not None:
                await session_recorder.mark_failed(str(exc))
            raise
