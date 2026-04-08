from datetime import datetime
import json5 as json
from .utils.views import print_agent_output
from .utils.llms import call_model
from multi_agents.route_agent import build_route_context

sample_json = """
{
  "table_of_contents": A table of contents in markdown syntax (using '-') based on the research headers and subheaders,
  "introduction": An indepth introduction to the topic in markdown syntax and hyperlink references to relevant sources. Each factual claim MUST cite source IDs e.g. 'Revenue reached $128B [S1][S2].',
  "conclusion": A conclusion to the entire research based on all research data in markdown syntax and hyperlink references to relevant sources. Each factual claim MUST cite source IDs.,
  "sources": A list with strings of all used source links in the entire research data in markdown syntax and apa citation format. For example: ['-  Title, year, Author [source url](source)', ...],
  "claim_annotations": A list of objects for each factual claim: [{"sentence": "the factual claim sentence", "source_ids": ["S1", "S2"], "section": "introduction"}, ...]
}
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

    async def write_sections(self, research_state: dict):
        query = research_state.get("title")
        # Use indexed research data (with source IDs) if available, fallback to raw
        data = research_state.get("indexed_research_data") or research_state.get("research_data")
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
            f"You must follow the guidelines provided: {guidelines}"
            if follow_guidelines
            else ""
        )

        citation_instruction = ""
        if has_source_index:
            citation_instruction = (
                "IMPORTANT: The research data below includes source IDs like [S1], [S2], etc.\n"
                "You MUST cite these source IDs after every factual claim. "
                "For example: 'Revenue reached $128B [S1][S2].'\n"
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
                f"Research data: {str(data)}\n"
                f"{citation_instruction}"
                f"Your task is to write an in depth, well written and detailed "
                f"introduction and conclusion to the research report based on the provided research data. "
                f"Do not include headers in the results.\n"
                f"You MUST include any relevant sources to the introduction and conclusion as markdown hyperlinks -"
                f"For example: 'This is a sample text. ([url website](url))'\n\n"
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

        research_layout_content = await self.write_sections(research_state)

        if research_state.get("task").get("verbose"):
            if self.websocket and self.stream_output:
                research_layout_content_str = json.dumps(
                    research_layout_content, indent=2
                )
                await self.stream_output(
                    "logs",
                    "research_layout_content",
                    research_layout_content_str,
                    self.websocket,
                )
            else:
                print_agent_output(research_layout_content, agent="WRITER")

        headers = self.get_headers(research_state)
        if research_state.get("task").get("follow_guidelines"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "rewriting_layout",
                    "Rewriting layout based on guidelines...",
                    self.websocket,
                )
            else:
                print_agent_output(
                    "Rewriting layout based on guidelines...", agent="WRITER"
                )
            headers = await self.revise_headers(
                task=research_state.get("task"), headers=headers
            )
            headers = headers.get("headers")

        return {**research_layout_content, "headers": headers}
