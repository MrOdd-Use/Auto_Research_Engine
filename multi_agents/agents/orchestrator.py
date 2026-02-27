import os
import time
import datetime
import json
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

    async def run_research_task(self, task_id=None):
        """
        Run a research task with the initialized research team.

        Args:
            task_id (optional): The ID of the task to run.

        Returns:
            The result of the research task.
        """
        research_team = self.init_research_team()
        chain = research_team.compile()

        await self._log_research_start()

        config = {
            "configurable": {
                "thread_id": task_id,
                "thread_ts": datetime.datetime.utcnow()
            }
        }

        result = await chain.ainvoke({"task": self.task}, config=config)
        return result
