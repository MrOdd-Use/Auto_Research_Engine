"""
Command line interface for multi-agent research only.

Usage:
python cli.py "<query>" --tone objective
"""
import argparse
import asyncio
import os
from uuid import uuid4

from dotenv import load_dotenv

from backend.utils import write_md_to_pdf, write_md_to_word
from gpt_researcher.utils.enum import Tone
from multi_agents.main import run_research_task


cli = argparse.ArgumentParser(
    description="Generate a research report using multi_agents mode.",
)

cli.add_argument("query", type=str, help="The query to conduct research on.")

cli.add_argument(
    "--tone",
    type=str,
    choices=[
        "objective",
        "formal",
        "analytical",
        "persuasive",
        "informative",
        "explanatory",
        "descriptive",
        "critical",
        "comparative",
        "speculative",
        "reflective",
        "narrative",
        "humorous",
        "optimistic",
        "pessimistic",
    ],
    default="objective",
    help="The tone of the report.",
)

cli.add_argument(
    "--no-pdf",
    action="store_true",
    help="Skip PDF generation (generate markdown and DOCX only).",
)

cli.add_argument(
    "--no-docx",
    action="store_true",
    help="Skip DOCX generation (generate markdown and PDF only).",
)


async def main(query: str, tone: str, no_pdf: bool, no_docx: bool):
    tone_map = {
        "objective": Tone.Objective,
        "formal": Tone.Formal,
        "analytical": Tone.Analytical,
        "persuasive": Tone.Persuasive,
        "informative": Tone.Informative,
        "explanatory": Tone.Explanatory,
        "descriptive": Tone.Descriptive,
        "critical": Tone.Critical,
        "comparative": Tone.Comparative,
        "speculative": Tone.Speculative,
        "reflective": Tone.Reflective,
        "narrative": Tone.Narrative,
        "humorous": Tone.Humorous,
        "optimistic": Tone.Optimistic,
        "pessimistic": Tone.Pessimistic,
    }

    federation_client = None
    if os.getenv("ROUTE_AGENT_BACKEND") == "federation":
        try:
            from multi_agents.route_agent import RouteAgentClient, RoutedLLMInvoker, set_global_invoker
            client = RouteAgentClient(backend="federation")
            await client.astart()
            set_global_invoker(RoutedLLMInvoker(client))
            federation_client = client
            print(f"Federation routing initialized (app_id={client.application_name})")
        except Exception as exc:
            print(f"Warning: Federation routing init failed: {exc}")

    try:
        result = await run_research_task(
            query=query,
            websocket=None,
            stream_output=None,
            tone=tone_map[tone],
            headers=None,
        )

        if isinstance(result, dict):
            report = str(result.get("report", ""))
        else:
            report = str(result)

        task_id = str(uuid4())
        os.makedirs("outputs", exist_ok=True)

        md_path = f"outputs/{task_id}.md"
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"Report written to '{md_path}'")

        if not no_pdf:
            try:
                pdf_path = await write_md_to_pdf(report, task_id)
                if pdf_path:
                    print(f"PDF written to '{pdf_path}'")
            except Exception as exc:
                print(f"Warning: PDF generation failed: {exc}")

        if not no_docx:
            try:
                docx_path = await write_md_to_word(report, task_id)
                if docx_path:
                    print(f"DOCX written to '{docx_path}'")
            except Exception as exc:
                print(f"Warning: DOCX generation failed: {exc}")
    finally:
        if federation_client is not None:
            await federation_client.astop()


if __name__ == "__main__":
    load_dotenv()
    args = cli.parse_args()
    asyncio.run(main(args.query, args.tone, args.no_pdf, args.no_docx))
