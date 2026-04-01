from .utils.file_formats import \
    write_md_to_pdf, \
    write_md_to_word, \
    write_text_to_md
from .utils.output_writers import (
    collect_cited_ids_from_text,
    write_annotated_report,
    write_evidence_base,
)

from .utils.views import print_agent_output


class PublisherAgent:
    def __init__(self, output_dir: str, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.output_dir = output_dir.strip()
        self.headers = headers or {}
        
    async def publish_research_report(self, research_state: dict, publish_formats: dict):
        layout = self.generate_layout(research_state)
        await self.write_report_by_formats(layout, publish_formats)

        # Write structured output files (evidence_base.md + report.md)
        await self._write_structured_outputs(layout, research_state)

        return layout

    async def _write_structured_outputs(self, layout: str, research_state: dict) -> None:
        """Write evidence_base.md and report.md alongside the standard report."""
        source_index = research_state.get("source_index") or {}
        claim_report = research_state.get("claim_confidence_report") or []

        # Collect citation frequencies from the layout for evidence_base annotations
        cited_ids = collect_cited_ids_from_text(layout) if source_index else None

        await write_evidence_base(self.output_dir, source_index, cited_ids)
        await write_annotated_report(
            self.output_dir, layout, claim_report, source_index,
        )

    def generate_layout(self, research_state: dict):
        final_draft = research_state.get("final_draft")
        if isinstance(final_draft, str) and final_draft.strip():
            return final_draft.strip()

        sections = []
        for subheader in research_state.get("research_data", []):
            if isinstance(subheader, dict):
                # Handle dictionary case
                for key, value in subheader.items():
                    sections.append(f"{value}")
            else:
                # Handle string case
                sections.append(f"{subheader}")
        
        sections_text = '\n\n'.join(sections)
        references = '\n'.join(f"{reference}" for reference in research_state.get("sources", []))
        headers = research_state.get("headers", {})
        layout = f"""# {headers.get('title')}
#### {headers.get("date")}: {research_state.get('date')}

## {headers.get("introduction")}
{research_state.get('introduction')}

## {headers.get("table_of_contents")}
{research_state.get('table_of_contents')}

{sections_text}

## {headers.get("conclusion")}
{research_state.get('conclusion')}

## {headers.get("references")}
{references}
"""
        return layout

    async def write_report_by_formats(self, layout:str, publish_formats: dict):
        if publish_formats.get("pdf"):
            await write_md_to_pdf(layout, self.output_dir)
        if publish_formats.get("docx"):
            await write_md_to_word(layout, self.output_dir)
        if publish_formats.get("markdown"):
            await write_text_to_md(layout, self.output_dir)

    async def run(self, research_state: dict):
        task = research_state.get("task")
        publish_formats = task.get("publish_formats")
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "publishing", f"Publishing final research report based on retrieved data...", self.websocket)
        else:
            print_agent_output(output="Publishing final research report based on retrieved data...", agent="PUBLISHER")
        final_research_report = await self.publish_research_report(research_state, publish_formats)
        return {"report": final_research_report}
