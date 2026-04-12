"""Publishing helpers for research report artifacts."""

from .utils.file_formats import \
    write_md_to_pdf, \
    write_md_to_word
from .utils.output_writers import (
    collect_cited_ids_from_text,
    write_annotated_report,
    write_evidence_base,
)

from .utils.views import print_agent_output


class PublisherAgent:
    """Generate final report artifacts for a research session."""

    def __init__(self, output_dir: str, websocket=None, stream_output=None, headers=None):
        """Initialize the publisher with an output directory and stream hooks."""
        self.websocket = websocket
        self.stream_output = stream_output
        self.output_dir = output_dir.strip()
        self.headers = headers or {}
        
    async def publish_research_report(self, research_state: dict, publish_formats: dict):
        """Render the final layout and persist all report artifacts."""
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

    @staticmethod
    def _format_named_block(heading: str | None, body: object) -> str:
        """Render a named markdown section when it has non-empty content."""
        heading_text = str(heading or "").strip()
        body_text = str(body or "").strip()

        if not body_text:
            return ""
        if not heading_text:
            return body_text
        return f"## {heading_text}\n{body_text}"

    @staticmethod
    def _collect_sections_text(research_state: dict) -> str:
        """Flatten the section payloads into the published report body."""
        sections = []
        for subheader in research_state.get("research_data", []):
            if isinstance(subheader, dict):
                for value in subheader.values():
                    value_text = str(value or "").strip()
                    if value_text:
                        sections.append(value_text)
            else:
                value_text = str(subheader or "").strip()
                if value_text:
                    sections.append(value_text)
        return "\n\n".join(sections)

    def generate_layout(self, research_state: dict):
        """Build the markdown layout for the final report."""
        final_draft = research_state.get("final_draft")
        if isinstance(final_draft, str) and final_draft.strip():
            return final_draft.strip()

        # Prefer TOC-ordered sections_body from WriterAgent; fall back to raw research_data
        sections_body = research_state.get("sections_body")
        if not sections_body:
            sections_body = self._collect_sections_text(research_state)
        references = "\n".join(
            str(reference).strip()
            for reference in research_state.get("sources", [])
            if str(reference).strip()
        )
        headers = research_state.get("headers", {})
        title = str(headers.get("title") or "").strip()
        date_label = str(headers.get("date") or "").strip()
        date_value = str(research_state.get("date") or "").strip()

        blocks = []
        if title:
            blocks.append(f"# {title}")
        if date_label and date_value:
            blocks.append(f"#### {date_label}: {date_value}")

        table_of_contents_block = self._format_named_block(
            headers.get("table_of_contents"),
            research_state.get("table_of_contents"),
        )
        if table_of_contents_block:
            blocks.append(table_of_contents_block)

        introduction_block = self._format_named_block(
            headers.get("introduction"),
            research_state.get("introduction"),
        )
        if introduction_block:
            blocks.append(introduction_block)

        if sections_body:
            blocks.append(sections_body)

        conclusion_block = self._format_named_block(
            headers.get("conclusion"),
            research_state.get("conclusion"),
        )
        if conclusion_block:
            blocks.append(conclusion_block)

        references_block = self._format_named_block(
            headers.get("references"),
            references,
        )
        if references_block:
            blocks.append(references_block)

        return "\n\n".join(blocks)

    async def write_report_by_formats(self, layout: str, publish_formats: dict):
        """Write optional non-markdown export formats.

        Markdown is persisted separately as the fixed `report.md` artifact via
        `_write_structured_outputs`, so this method intentionally skips the
        legacy random-stem markdown export.
        """
        if publish_formats.get("pdf"):
            await write_md_to_pdf(layout, self.output_dir)
        if publish_formats.get("docx"):
            await write_md_to_word(layout, self.output_dir)

    async def run(self, research_state: dict):
        """Publish the current research state and return the report text."""
        task = research_state.get("task")
        publish_formats = task.get("publish_formats")
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "publishing", f"Publishing final research report based on retrieved data...", self.websocket)
        else:
            print_agent_output(output="Publishing final research report based on retrieved data...", agent="PUBLISHER")
        final_research_report = await self.publish_research_report(research_state, publish_formats)
        return {"report": final_research_report}
