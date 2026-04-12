from datetime import datetime
import re
import json5 as json
from .utils.views import print_agent_output
from .utils.llms import call_model
from multi_agents.route_agent import build_route_context
from gpt_researcher.context.compression import truncate_research_data

sample_json = """
{
  "introduction": An in-depth introduction in markdown syntax. Cite every factual claim immediately after the sentence using [chapter.index] source IDs, e.g. 'Jobs displaced by 2030[1.1][1.2].',
  "conclusion": A conclusion in markdown syntax. Cite every factual claim using [chapter.index] source IDs, e.g. 'Productivity gains reached 40%[2.3].',
  "sources": A list with strings of all used source links in APA citation format with markdown hyperlinks. For example: ['-  Title, year, Author [source url](source)', ...],
  "claim_annotations": A list of objects for each factual claim: [{"sentence": "the factual claim sentence", "source_ids": ["1.1", "1.2"], "section": "introduction"}, ...]
}
"""

TOC_SAMPLE = """
- Chapter Title One
  - Sub-topic A
  - Sub-topic B
- Chapter Title Two
  - Sub-topic C
"""


class WriterAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers

    def get_headers(self, research_state: dict):
        return {
            "title": research_state.get("title"),
            "date": "Date",
            "introduction": "Introduction",
            "table_of_contents": "Table of Contents",
            "conclusion": "Conclusion",
            "references": "References",
        }

    def _format_research_data(self, data) -> str:
        parts = []
        for item in (data or []):
            if isinstance(item, dict):
                for title, body in item.items():
                    parts.append(f"## {title}\n\n{body}")
            elif isinstance(item, str) and item.strip():
                parts.append(item)
        return "\n\n---\n\n".join(parts)

    def _format_summaries(self, research_state: dict) -> str:
        """Format per-section summaries for the TOC generation prompt."""
        section_details = research_state.get("section_details") or []
        section_summaries = research_state.get("section_summaries") or []
        lines = []
        for i, detail in enumerate(section_details):
            title = str(detail.get("header") or f"Section {i + 1}").strip()
            summary = section_summaries[i] if i < len(section_summaries) else ""
            lines.append(f"Chapter {i + 1}: {title}")
            if summary:
                lines.append(f"  Summary: {summary}")
        return "\n".join(lines)

    async def _write_toc_from_summaries(self, research_state: dict) -> str:
        """Generate TOC via LLM using per-section summaries."""
        query = research_state.get("title") or ""
        task = research_state.get("task") or {}
        summaries_text = self._format_summaries(research_state)

        if not summaries_text.strip():
            return ""

        prompt = [
            {
                "role": "system",
                "content": "You are a research editor. Generate a markdown table of contents.",
            },
            {
                "role": "user",
                "content": (
                    f"Report title: {query}\n\n"
                    f"Chapter summaries:\n{summaries_text}\n\n"
                    "Generate a markdown table of contents based strictly on the chapters above.\n"
                    "Rules:\n"
                    "- Top-level items use '- ' and match the chapter title exactly\n"
                    "- Sub-items use '  - ' and reflect key sub-topics from the summary\n"
                    "- Do NOT invent chapters not listed above\n"
                    "- Return ONLY the markdown list, no other text\n\n"
                    f"Example format:\n{TOC_SAMPLE}"
                ),
            },
        ]

        route_context = build_route_context(
            application_name=str(task.get("application_name") or "auto_research_engine"),
            shared_agent_class="writer_agent",
            agent_role="writer",
            stage_name="toc_generation",
            system_prompt="You are a research editor.",
            task=query,
            state=research_state,
            task_payload=task,
        )
        result = await call_model(prompt, task.get("model"), route_context=route_context)
        if isinstance(result, dict):
            result = result.get("content") or result.get("text") or ""
        return str(result or "").strip()

    @staticmethod
    def _order_sections_by_toc(toc: str, research_data: list) -> list:
        """Reorder research_data sections to match TOC top-level entry order.

        Matches by lowercased, stripped title. Unmatched sections are appended at the end.
        """
        if not toc or not research_data:
            return list(research_data or [])

        toc_titles = []
        for line in toc.splitlines():
            m = re.match(r"^-\s+(.+)", line.strip())
            if m:
                toc_titles.append(m.group(1).strip().lower())

        if not toc_titles:
            return list(research_data)

        # Build lookup: normalised title -> section item
        lookup: dict = {}
        for item in research_data:
            if isinstance(item, dict):
                for key in item:
                    lookup[key.strip().lower()] = item
            elif isinstance(item, str):
                lookup[item.strip()[:80].lower()] = item

        ordered = []
        used = set()
        for title in toc_titles:
            match = lookup.get(title)
            if match is not None:
                item_id = id(match)
                if item_id not in used:
                    ordered.append(match)
                    used.add(item_id)

        # Append any sections not matched by TOC
        for item in research_data:
            if id(item) not in used:
                ordered.append(item)

        return ordered

    async def write_sections(self, research_state: dict):
        query = research_state.get("title")
        data = truncate_research_data(
            research_state.get("indexed_research_data") or research_state.get("research_data")
        )
        has_source_index = bool(research_state.get("indexed_research_data"))
        task = research_state.get("task")
        follow_guidelines = task.get("follow_guidelines")
        guidelines = task.get("guidelines")
        checkpoint_note = (
            str(task.get("checkpoint_note") or "").strip()
            if task.get("checkpoint_target") == "writer"
            else ""
        )
        note_block = (
            f"Additional rerun instruction for this writing pass: {checkpoint_note}\n"
            if checkpoint_note
            else ""
        )
        guidelines_block = (
            f"You must follow the guidelines provided: {guidelines}\n"
            "Additionally, apply these guidelines to the header values in the JSON output "
            "(introduction, conclusion, sources keys must follow the guidelines). "
            "Header string values must be plain text without markdown syntax."
            if follow_guidelines
            else ""
        )

        citation_instruction = ""
        if has_source_index:
            citation_instruction = (
                "IMPORTANT: The research data below includes source IDs like [1.1], [1.2], etc.\n"
                "You MUST cite these source IDs after every factual claim. "
                "For example: 'Revenue reached $128B [1.1][1.2].'\n"
                "Only use source IDs that appear in the research data. "
                "Do not invent facts not present in the research data.\n"
                "If information for a topic is unavailable in the sources, "
                "write: [Knowledge on this topic unavailable from known sources]\n\n"
            )

        prompt = [
            {
                "role": "system",
                "content": "You are a research writer. Your sole purpose is to write a well-written "
                "research reports about a "
                "topic based on research findings and information.\n ",
            },
            {
                "role": "user",
                "content": f"Today's date is {datetime.now().strftime('%d/%m/%Y')}\n."
                f"Query or Topic: {query}\n"
                f"Research data:\n{self._format_research_data(data)}\n"
                f"{citation_instruction}"
                f"Your task is to write an in depth, well written and detailed "
                f"introduction and conclusion to the research report based on the provided research data. "
                f"Do not include headers in the results.\n"
                f"Cite all sources using [chapter.index] IDs placed directly after each claim, e.g. 'Jobs displaced by 2030[1.1][2.3].'\n"
                f"Do NOT use inline markdown hyperlinks. All URLs belong in the 'sources' list only.\n\n"
                f"{note_block}"
                f"{guidelines_block}\n"
                f"You MUST return nothing but a JSON in the following format (without json markdown):\n"
                f"{sample_json}\n\n",
            },
        ]

        route_context = build_route_context(
            application_name=str(task.get("application_name") or "auto_research_engine"),
            shared_agent_class="writer_agent",
            agent_role="writer",
            stage_name="draft_composition",
            system_prompt="You are a research writer.",
            task=query,
            state=research_state,
            task_payload=task,
        )
        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
            route_context=route_context,
        )
        return response

    async def revise_headers(self, task: dict, headers: dict):
        checkpoint_note = (
            str(task.get("checkpoint_note") or "").strip()
            if task.get("checkpoint_target") == "writer"
            else ""
        )
        note_block = f"Additional rerun instruction: {checkpoint_note}\n" if checkpoint_note else ""
        prompt = [
            {
                "role": "system",
                "content": """You are a research writer.
Your sole purpose is to revise the headers data based on the given guidelines.""",
            },
            {
                "role": "user",
                "content": f"""Your task is to revise the given headers JSON based on the guidelines given.
You are to follow the guidelines but the values should be in simple strings, ignoring all markdown syntax.
You must return nothing but a JSON in the same format as given in headers data.
Guidelines: {task.get("guidelines")}\n
{note_block}
Headers Data: {headers}\n
""",
            },
        ]

        route_context = build_route_context(
            application_name=str(task.get("application_name") or "auto_research_engine"),
            shared_agent_class="writer_agent",
            agent_role="writer",
            stage_name="header_revision",
            system_prompt="You are a research writer.",
            task=str(task.get("query") or ""),
            task_payload=task,
        )
        response = await call_model(
            prompt,
            task.get("model"),
            response_format="json",
            route_context=route_context,
        )
        return {"headers": response}

    async def run(self, research_state: dict):
        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "writing_report",
                f"Writing final research report based on research data...",
                self.websocket,
            )
        else:
            print_agent_output(
                f"Writing final research report based on research data...",
                agent="WRITER",
            )

        # Step 1: Generate TOC from per-section summaries
        toc = await self._write_toc_from_summaries(research_state)

        # Step 2: Order sections by TOC, build final_draft body
        research_data = research_state.get("research_data") or []
        ordered_sections = self._order_sections_by_toc(toc, research_data)
        sections_body = "\n\n".join(
            "\n\n".join(str(v) for v in item.values() if v)
            if isinstance(item, dict)
            else str(item or "")
            for item in ordered_sections
        ).strip()

        # Step 3: Generate intro + conclusion
        layout_content = await self.write_sections(research_state)

        if research_state.get("task", {}).get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "research_layout_content",
                    json.dumps(layout_content, indent=2),
                    self.websocket,
                )
            else:
                print_agent_output(layout_content, agent="WRITER")

        headers = self.get_headers(research_state)
        return {
            **layout_content,
            "table_of_contents": toc,
            "sections_body": sections_body,
            "headers": headers,
        }
