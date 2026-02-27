from datetime import datetime
import asyncio
import re
import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from langgraph.graph import StateGraph, END

from gpt_researcher.utils.validators import ResearchOutline
from .utils.views import print_agent_output
from .utils.llms import call_model
from ..memory.draft import DraftState
from .researcher import ResearchAgent
from .scrap import ScrapAgent
from .check_data import CheckDataAgent
from .reviewer import ReviewerAgent
from .reviser import ReviserAgent


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
        self.enable_scrap = os.getenv("ASA_ENABLE_SCRAP", "true").strip().lower() != "false"

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

        prompt = self._create_planning_prompt(
            query, initial_research, include_human_feedback, human_feedback, max_sections,
        )

        print_agent_output(
            "Planning a structured outline based on initial research...", agent="EDITOR")
        plan = await call_model(
            prompt=prompt,
            model=task.get("model"),
            response_format="json",
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

    async def run_parallel_research(self, research_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute parallel research tasks for each section.

        Reads enriched `section_details` when available, falling back to flat `sections`.
        """
        agents = self._initialize_agents()
        workflow = self._create_workflow(agents)
        chain = workflow.compile()

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

        final_drafts = [
            chain.ainvoke(
                self._create_task_input(
                    research_state=research_state,
                    section_detail=section_detail,
                    title=title,
                    audit_feedback=(
                        audit_feedback_queue[idx]
                        if idx < len(audit_feedback_queue)
                        else (audit_feedback_queue[0] if audit_feedback_queue else None)
                    ),
                    extra_hints=shared_extra_hints,
                ),
                config={"tags": ["Auto_Research_Engine"]},
            )
            for idx, section_detail in enumerate(section_details)
        ]
        gathered_results = await asyncio.gather(*final_drafts)
        research_results = [result["draft"] for result in gathered_results]
        scrap_packets = [result.get("scrap_packet") for result in gathered_results if result.get("scrap_packet")]
        check_data_reports = [
            result.get("check_data_verdict")
            for result in gathered_results
            if result.get("check_data_verdict")
        ]

        return {
            "research_data": research_results,
            "scrap_packets": scrap_packets,
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
    ) -> List[Dict[str, str]]:
        """Create the prompt for structured research planning."""
        return [
            {"role": "system", "content": _PLANNING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._format_planning_instructions(
                    query, initial_research, include_human_feedback,
                    human_feedback, max_sections,
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
    ) -> str:
        """Format chain-of-thought planning instructions."""
        today = datetime.now().strftime('%d/%m/%Y')

        feedback_block = ""
        if include_human_feedback and human_feedback and human_feedback.strip().lower() != "no":
            feedback_block = (
                f"\n## Human Feedback\n{human_feedback}\n"
                "You must incorporate this feedback into your outline.\n"
            )

        return f"""Today's date is {today}.

## Original Research Query
"{query}"

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
            "scrap": ScrapAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "check_data": CheckDataAgent(self.websocket, self.stream_output, self.headers),
            "reviewer": ReviewerAgent(self.websocket, self.stream_output, self.headers),
            "reviser": ReviserAgent(self.websocket, self.stream_output, self.headers),
        }

    def _create_workflow(self, agents: Optional[Dict[str, Any]] = None) -> StateGraph:
        """Create the workflow for the research process."""
        agents = agents or self._initialize_agents()
        workflow = StateGraph(DraftState)

        researcher_node = (
            agents["scrap"].run_depth_scrap
            if self.enable_scrap
            else agents["research"].run_depth_research
        )
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("reviewer", agents["reviewer"].run)
        workflow.add_node("reviser", agents["reviser"].run)

        workflow.set_entry_point("researcher")
        if self.enable_scrap:
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
        return {
            "task": research_state.get("task"),
            "topic": section_detail["header"],
            "iteration_index": 1,
            "research_context": {
                "description": section_detail.get("description", ""),
                "key_points": section_detail.get("key_points", []),
                "research_queries": section_detail.get("research_queries", []),
            },
            "title": title,
            "audit_feedback": audit_feedback,
            "extra_hints": extra_hints,
            "headers": self.headers,
        }

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
