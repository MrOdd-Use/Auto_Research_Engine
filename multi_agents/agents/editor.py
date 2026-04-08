from datetime import datetime
import asyncio
import copy
import logging
import re
import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from langgraph.graph import StateGraph, END

from gpt_researcher.utils.validators import ResearchOutline
from .utils.views import print_agent_output
from .utils.llms import call_model
from multi_agents.route_agent import build_route_context
from ..memory.draft import DraftState
from .researcher import ResearchAgent
from .scraping import ScrapingAgent
from .check_data import CheckDataAgent
from .reviewer import ReviewerAgent
from .reviser import ReviserAgent
from multi_agents.workflow_session import WorkflowSessionRecorder


_PLANNING_SYSTEM_PROMPT = (
    "You are a senior research editor specializing in structuring comprehensive "
    "research reports. Your goal is to analyze a research summary and create a "
    "well-organized outline that ensures thorough, non-overlapping coverage of "
    "the topic.\n\n"
    "You approach this task methodically:\n"
    "1. First, identify the user's query intent "
    "(analytical, descriptive, comparative, or exploratory)\n"
    "2. Decompose the query into distinct sub-aspects or dimensions\n"
    "3. Design sections that map to these sub-aspects with logical ordering\n"
    "4. For each section, identify key investigation points and craft targeted "
    "search queries\n\n"
    "Your sections must be:\n"
    "- Mutually exclusive (no overlapping content between sections)\n"
    "- Collectively exhaustive (cover all important aspects of the topic)\n"
    "- Logically ordered (build upon each other or follow a natural progression)\n"
    "- Focused on substantive research topics "
    "(never include Introduction, Conclusion, or References)"
)


class EditorAgent:
    """Agent responsible for planning research outlines and managing parallel research."""

    _DEFAULT_MAX_SECTIONS = 3
    _FORBIDDEN_SECTION_TITLES = {
        "introduction",
        "conclusion",
        "references",
        "reference",
        "summary",
    }

    def __init__(self, websocket=None, stream_output=None, tone=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.tone = tone
        self.headers = headers or {}
        self.enable_scraping = os.getenv("ASA_ENABLE_SCRAPING", "true").strip().lower() != "false"

    # ── Planning ──────────────────────────────────────────────────────────

    async def plan_research(self, research_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the research outline based on initial research and task parameters.

        Uses chain-of-thought prompting to:
        1. Classify the query intent
        2. Decompose the query into sub-aspects
        3. Generate structured sections with descriptions, key points, and search queries

        Returns both flat `sections` (backward compat) and enriched `section_details`.
        """
        initial_research = research_state.get("initial_research") or ""
        task = research_state.get("task") or {}
        include_human_feedback = task.get("include_human_feedback")
        human_feedback = research_state.get("human_feedback")
        max_sections = self._normalize_max_sections(task.get("max_sections"))
        query = task.get("query", "")
        query_intent = str(task.get("query_intent") or "").strip().lower()

        prompt = self._create_planning_prompt(
            query,
            initial_research,
            include_human_feedback,
            human_feedback,
            max_sections,
            query_intent,
        )

        print_agent_output(
            "Planning a structured outline based on initial research...", agent="EDITOR")
        route_context = build_route_context(
            application_name=str(task.get("application_name") or "auto_research_engine"),
            shared_agent_class="planner_agent",
            agent_role="planner",
            stage_name="outline_planning",
            system_prompt=_PLANNING_SYSTEM_PROMPT,
            task=query,
            state=research_state,
            task_payload=task,
        )
        plan = await call_model(
            prompt=prompt,
            model=task.get("model"),
            response_format="json",
            route_context=route_context,
        ) or {}

        outline = self._parse_outline(plan, max_sections, task)

        sections = [s["header"] for s in outline.section_details]
        if not sections:
            sections = [self._fallback_section_title(task)]

        return {
            "title": outline.title,
            "date": outline.date,
            "sections": sections,
            "section_details": outline.section_details,
        }

    # ── Parallel Research ─────────────────────────────────────────────────

    async def run_parallel_research(
        self,
        research_state: Dict[str, Any],
        session_recorder: WorkflowSessionRecorder | None = None,
        selected_section_key: str | None = None,
        start_from_section_node: str | None = None,
        section_state_before: Dict[str, Any] | None = None,
        note: str | None = None,
    ) -> Dict[str, Any]:
        """
        Execute parallel research tasks for each section.

        Reads enriched `section_details` when available, falling back to flat `sections`.
        """
        agents = self._initialize_agents()

        task = research_state.get("task") or {}
        max_sections = self._normalize_max_sections(task.get("max_sections"))
        title = self._normalize_title(research_state.get("title"), task)

        section_details = research_state.get("section_details") or []
        if not section_details:
            queries = self._normalize_sections(research_state.get("sections"), max_sections)
            if not queries:
                queries = [self._fallback_section_title(task)]
            section_details = [
                {"header": q, "description": "", "key_points": [], "research_queries": []}
                for q in queries
        ]

        self._log_parallel_research([s["header"] for s in section_details])

        audit_feedback_queue = research_state.get("audit_feedback_queue") or task.get("audit_feedback_queue") or []
        shared_extra_hints = research_state.get("extra_hints") or task.get("extra_hints")
        existing_research_results = list(research_state.get("research_data") or [])
        existing_scraping_packets = list(research_state.get("scraping_packets") or [])
        existing_check_data_reports = list(research_state.get("check_data_reports") or [])

        async def run_one_section(idx: int, section_detail: Dict[str, Any]) -> Dict[str, Any]:
            section_key = self._make_section_key(idx, section_detail.get("header", ""))
            if selected_section_key and section_key != selected_section_key:
                return {
                    "draft": existing_research_results[idx] if idx < len(existing_research_results) else {},
                    "scraping_packet": existing_scraping_packets[idx] if idx < len(existing_scraping_packets) else None,
                    "check_data_verdict": (
                        existing_check_data_reports[idx] if idx < len(existing_check_data_reports) else None
                    ),
                }

            task_input = (
                copy.deepcopy(section_state_before)
                if selected_section_key and section_key == selected_section_key and section_state_before
                else self._create_task_input(
                    research_state=research_state,
                    section_detail={**section_detail, "_section_index": idx},
                    title=title,
                    audit_feedback=(
                        audit_feedback_queue[idx]
                        if idx < len(audit_feedback_queue)
                        else (audit_feedback_queue[0] if audit_feedback_queue else None)
                    ),
                    extra_hints=shared_extra_hints,
                )
            )
            if note and selected_section_key and section_key == selected_section_key:
                task_input = self._inject_section_note(task_input, start_from_section_node or "researcher", note)

            return await self._run_section_workflow(
                draft_state=task_input,
                agents=agents,
                section_index=idx,
                section_title=section_detail.get("header", ""),
                session_recorder=session_recorder,
                start_node=start_from_section_node if selected_section_key and section_key == selected_section_key else None,
            )

        gathered_results = await asyncio.gather(
            *[run_one_section(idx, section_detail) for idx, section_detail in enumerate(section_details)]
        )
        research_results = [result.get("draft") for result in gathered_results]
        scraping_packets = [result.get("scraping_packet") for result in gathered_results]
        check_data_reports = [result.get("check_data_verdict") for result in gathered_results]

        return {
            "research_data": research_results,
            "scraping_packets": scraping_packets,
            "check_data_reports": check_data_reports,
        }

    # ── Prompt Construction ───────────────────────────────────────────────

    def _create_planning_prompt(
        self,
        query: str,
        initial_research: str,
        include_human_feedback: bool,
        human_feedback: Optional[str],
        max_sections: int,
        query_intent: str = "",
    ) -> List[Dict[str, str]]:
        """Create the prompt for structured research planning."""
        return [
            {"role": "system", "content": _PLANNING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._format_planning_instructions(
                    query,
                    initial_research,
                    include_human_feedback,
                    human_feedback,
                    max_sections,
                    query_intent,
                ),
            },
        ]

    def _format_planning_instructions(
        self,
        query: str,
        initial_research: str,
        include_human_feedback: bool,
        human_feedback: Optional[str],
        max_sections: int,
        query_intent: str = "",
    ) -> str:
        """Format chain-of-thought planning instructions."""
        today = datetime.now().strftime('%d/%m/%Y')

        feedback_block = ""
        if include_human_feedback and human_feedback and human_feedback.strip().lower() != "no":
            feedback_block = (
                f"\n## Human Feedback\n{human_feedback}\n"
                "You must incorporate this feedback into your outline.\n"
            )

        intent_block = ""
        if query_intent:
            intent_block = (
                f"\n## Preclassified Query Intent\n{query_intent}\n"
                "Reuse this intent classification when designing the outline unless the query text clearly contradicts it.\n"
            )

        return f"""Today's date is {today}.

## Original Research Query
"{query}"
{intent_block}

## Initial Research Summary
{initial_research}
{feedback_block}
## Your Task
Generate a structured research outline following these steps:

**Step 1 - Classify Intent**: Determine if this query is analytical (why/how), \
descriptive (what/who), comparative (vs/differences), or exploratory (overview/landscape).

**Step 2 - Decompose**: Break the query into {max_sections} distinct sub-aspects \
or dimensions that together provide comprehensive coverage. Each sub-aspect must be \
mutually exclusive (no overlapping content) and collectively exhaustive (cover all \
important angles).

**Step 3 - Structure Sections**: For each sub-aspect, create a section with:
- A clear, specific header (not generic like "Overview" or "Background")
- A one-sentence description of its scope and boundaries
- 2-3 key points that a researcher should investigate
- 2-3 targeted search queries a researcher should use to find relevant information

**Step 4 - Verify**: Ensure sections don't overlap, follow a logical order, and \
together fully address the original query.

You must generate exactly {max_sections} sections (no more, no fewer).
Do NOT include Introduction, Conclusion, Summary, or References sections.

Return valid JSON only (no markdown fences) with this exact shape:
{{
  "title": "Descriptive research report title",
  "date": "{today}",
  "query_intent": "analytical|descriptive|comparative|exploratory",
  "sections": [
    {{
      "header": "Specific Section Title",
      "description": "One sentence describing what this section covers and its boundaries",
      "key_points": ["Point to investigate 1", "Point to investigate 2"],
      "research_queries": ["targeted search query 1", "targeted search query 2"]
    }}
  ]
}}"""

    # ── Outline Parsing ───────────────────────────────────────────────────

    def _parse_outline(self, plan: dict, max_sections: int, task: Dict[str, Any]) -> SimpleNamespace:
        """
        Parse LLM output into a structured outline.

        Attempts Pydantic validation first, then falls back to flat-string parsing
        for backward compatibility with models that return the old format.
        """
        try:
            outline = ResearchOutline(**plan)
            valid_sections = [
                s.model_dump()
                for s in outline.sections
                if s.header.strip().lower() not in self._FORBIDDEN_SECTION_TITLES
            ][:max_sections]

            if valid_sections:
                return SimpleNamespace(
                    title=outline.title or self._fallback_section_title(task),
                    date=outline.date or datetime.now().strftime('%d/%m/%Y'),
                    section_details=valid_sections,
                )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                "Pydantic outline validation failed, falling back to flat parsing: %s", exc
            )

        flat = self._normalize_sections(plan.get("sections"), max_sections)
        return SimpleNamespace(
            title=self._normalize_title(plan.get("title"), task),
            date=self._normalize_date(plan.get("date")),
            section_details=[
                {"header": h, "description": "", "key_points": [], "research_queries": []}
                for h in flat
            ],
        )

    # ── Workflow ──────────────────────────────────────────────────────────

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize the research, reviewer, and reviser agents."""
        return {
            "research": ResearchAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "scraping": ScrapingAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "check_data": CheckDataAgent(self.websocket, self.stream_output, self.headers),
            "reviewer": ReviewerAgent(self.websocket, self.stream_output, self.headers),
            "reviser": ReviserAgent(self.websocket, self.stream_output, self.headers),
        }

    def _create_workflow(self, agents: Optional[Dict[str, Any]] = None) -> StateGraph:
        """Create the workflow for the research process."""
        agents = agents or self._initialize_agents()
        workflow = StateGraph(DraftState)

        researcher_node = (
            agents["scraping"].run_depth_scraping
            if self.enable_scraping
            else agents["research"].run_depth_research
        )
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("reviewer", agents["reviewer"].run)
        workflow.add_node("reviser", agents["reviser"].run)

        workflow.set_entry_point("researcher")
        if self.enable_scraping:
            workflow.add_node("check_data", agents["check_data"].run)
            workflow.add_edge("researcher", "check_data")
            workflow.add_conditional_edges(
                "check_data",
                self._route_check_data,
                {
                    "retry": "researcher",
                    "accept": END,
                    "blocked": END,
                },
            )
        else:
            workflow.add_edge("researcher", "reviewer")
        workflow.add_edge("reviser", "reviewer")
        workflow.add_conditional_edges(
            "reviewer",
            lambda draft: "accept" if draft["review"] is None else "revise",
            {"accept": END, "revise": "reviser"},
        )

        return workflow

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log_parallel_research(self, queries: List[str]) -> None:
        """Log the start of parallel research tasks."""
        if self.websocket and self.stream_output:
            asyncio.create_task(self.stream_output(
                "logs",
                "parallel_research",
                f"Running parallel research for the following queries: {queries}",
                self.websocket,
            ))
        else:
            print_agent_output(
                f"Running the following research tasks in parallel: {queries}...",
                agent="EDITOR",
            )

    def _create_task_input(
        self,
        research_state: Dict[str, Any],
        section_detail: dict,
        title: str,
        audit_feedback: Optional[dict],
        extra_hints: Optional[str],
    ) -> Dict[str, Any]:
        """Create the input for a single research task with enriched context."""
        research_queries = self._normalize_research_queries(
            section_detail.get("research_queries")
        )
        planning_incomplete = len(research_queries) == 0
        if planning_incomplete:
            logging.getLogger(__name__).warning(
                "Planning incomplete for section '%s': missing research_queries.",
                section_detail.get("header"),
            )
        return {
            "task": research_state.get("task"),
            "topic": section_detail["header"],
            "iteration_index": 1,
            "section_key": self._make_section_key(
                int(section_detail.get("_section_index", 0)),
                section_detail["header"],
            ),
            "research_context": {
                "description": section_detail.get("description", ""),
                "key_points": section_detail.get("key_points", []),
                "research_queries": research_queries,
                "planning_incomplete": planning_incomplete,
                "planning_issue": "missing_research_queries" if planning_incomplete else "",
            },
            "title": title,
            "audit_feedback": audit_feedback,
            "extra_hints": extra_hints,
            "headers": self.headers,
        }

    async def _run_section_workflow(
        self,
        *,
        draft_state: Dict[str, Any],
        agents: Dict[str, Any],
        section_index: int,
        section_title: str,
        session_recorder: WorkflowSessionRecorder | None = None,
        start_node: str | None = None,
    ) -> Dict[str, Any]:
        state = copy.deepcopy(draft_state)
        node = start_node or "researcher"

        while True:
            if node in {"researcher", "scraping"}:
                state = await self._run_section_node(
                    "scraping" if self.enable_scraping else "researcher",
                    agents["scraping"].run_depth_scraping if self.enable_scraping else agents["research"].run_depth_research,
                    state,
                    section_index=section_index,
                    section_title=section_title,
                    session_recorder=session_recorder,
                )
                node = "check_data" if self.enable_scraping else "reviewer"
                if not self.enable_scraping:
                    continue

            if self.enable_scraping:
                state = await self._run_section_node(
                    "check_data",
                    agents["check_data"].run,
                    state,
                    section_index=section_index,
                    section_title=section_title,
                    session_recorder=session_recorder,
                )
                action = self._route_check_data(state)
                if action == "retry":
                    node = "researcher"
                    continue
                return state

            if node == "reviewer":
                state = await self._run_section_node(
                    "reviewer",
                    agents["reviewer"].run,
                    state,
                    section_index=section_index,
                    section_title=section_title,
                    session_recorder=session_recorder,
                )
                if state.get("review") is None:
                    return state
                node = "reviser"
                continue

            state = await self._run_section_node(
                "reviser",
                agents["reviser"].run,
                state,
                section_index=section_index,
                section_title=section_title,
                session_recorder=session_recorder,
            )
            node = "reviewer"

    async def _run_section_node(
        self,
        node_name: str,
        runner,
        state: Dict[str, Any],
        *,
        section_index: int,
        section_title: str,
        session_recorder: WorkflowSessionRecorder | None = None,
    ) -> Dict[str, Any]:
        state_before = copy.deepcopy(state)
        output_delta = await runner(state)
        state.update(output_delta or {})
        self._clear_section_note(state)

        if session_recorder is not None:
            await session_recorder.record_section_checkpoint(
                section_index=section_index,
                section_title=section_title,
                node_name=node_name,
                state_before=state_before,
                output_delta=output_delta or {},
                state_after=state,
                summary=self._summarize_section_output(node_name, output_delta or {}, state),
            )
        return state

    def _inject_section_note(self, draft_state: Dict[str, Any], node_name: str, note: str) -> Dict[str, Any]:
        state = copy.deepcopy(draft_state)
        task = state.setdefault("task", {})
        task["checkpoint_note"] = note
        task["checkpoint_target"] = node_name
        if node_name in {"researcher", "scraping"}:
            existing = str(state.get("extra_hints") or "").strip()
            state["extra_hints"] = f"{existing}\n{note}".strip() if existing else note
        elif node_name == "check_data":
            context = state.setdefault("research_context", {})
            description = str(context.get("description") or "").strip()
            context["description"] = f"{description}\nOperator note: {note}".strip() if description else note
        return state

    def _clear_section_note(self, draft_state: Dict[str, Any]) -> None:
        task = draft_state.get("task")
        if isinstance(task, dict):
            task.pop("checkpoint_note", None)
            task.pop("checkpoint_target", None)

    def _summarize_section_output(
        self,
        node_name: str,
        output_delta: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        if node_name in {"researcher", "scraping"}:
            scraping_packet = output_delta.get("scraping_packet") or {}
            return {
                "iteration_index": output_delta.get("iteration_index") or state_after.get("iteration_index"),
                "model_level": scraping_packet.get("model_level"),
                "active_engines": scraping_packet.get("active_engines") or [],
            }
        if node_name == "check_data":
            verdict = output_delta.get("check_data_verdict") or {}
            deep_eval = verdict.get("deep_eval_report") or {}
            return {
                "action": output_delta.get("check_data_action"),
                "status": verdict.get("status"),
                "final_score": deep_eval.get("final_score"),
            }
        if node_name == "reviewer":
            return {"accepted": state_after.get("review") is None}
        if node_name == "reviser":
            return {"has_revision_notes": bool(output_delta.get("revision_notes"))}
        return {"keys": sorted(output_delta.keys())}

    def _make_section_key(self, section_index: int, header: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(header or "").strip().lower()).strip("_")
        return f"section_{section_index}_{cleaned[:48] or 'section'}"

    @staticmethod
    def _route_check_data(draft_state: Dict[str, Any]) -> str:
        action = str(draft_state.get("check_data_action") or "").strip().lower()
        if action == "retry":
            return "retry"
        if action == "blocked":
            return "blocked"
        return "accept"

    def _normalize_max_sections(self, max_sections: Any) -> int:
        """Normalize max_sections to a safe integer range."""
        try:
            parsed = int(max_sections)
        except (TypeError, ValueError):
            return self._DEFAULT_MAX_SECTIONS
        return min(max(parsed, 1), 12)

    def _fallback_section_title(self, task: Dict[str, Any]) -> str:
        """Return a safe fallback section title."""
        query = str(task.get("query") or "").strip()
        if not query:
            return "Main Research Focus"
        return query[:180]

    def _normalize_title(self, title: Any, task: Dict[str, Any]) -> str:
        """Ensure we always have a usable title."""
        cleaned = self._clean_text(title)
        return cleaned or self._fallback_section_title(task)

    def _normalize_date(self, date_value: Any) -> str:
        """Normalize date value from model output."""
        cleaned = self._clean_text(date_value)
        return cleaned or datetime.now().strftime('%d/%m/%Y')

    def _normalize_sections(self, sections: Any, max_sections: int) -> List[str]:
        """Normalize planner sections from model output into a clean list."""
        normalized: List[str] = []
        seen = set()

        for raw_section in self._iter_section_candidates(sections):
            cleaned = self._clean_text(raw_section)
            if not cleaned:
                continue
            if cleaned.lower() in self._FORBIDDEN_SECTION_TITLES:
                continue
            dedupe_key = cleaned.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(cleaned)
            if len(normalized) >= max_sections:
                break

        return normalized

    def _iter_section_candidates(self, sections: Any) -> Iterable[Any]:
        """Yield section candidates from a variety of possible model outputs."""
        if sections is None:
            return

        if isinstance(sections, list):
            yield from sections
            return

        if isinstance(sections, str):
            for part in re.split(r"[\n;,]+", sections):
                yield part
            return

        if isinstance(sections, dict):
            yield from sections.values()

    def _clean_text(self, value: Any) -> str:
        """Normalize text output from model results."""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)

        text = value.strip().strip("-*").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _normalize_research_queries(self, research_queries: Any) -> List[str]:
        """Normalize planner research queries for downstream deterministic use."""
        if not isinstance(research_queries, list):
            return []
        normalized: List[str] = []
        seen = set()
        for item in research_queries:
            cleaned = self._clean_text(item)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
        return normalized
