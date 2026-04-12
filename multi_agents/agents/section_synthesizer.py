import logging
from typing import Dict, List
from urllib.parse import urlparse

from .utils.llms import call_model
from .utils.views import print_agent_output
from multi_agents.route_agent import build_route_context

_SYNTHESIS_SYSTEM = (
    "You are an expert research analyst. "
    "Your task is to synthesize verified evidence passages into a well-structured "
    "research report section. Write authoritative, analytical prose. "
    "Cite every factual claim immediately after the sentence using numeric source IDs "
    "from the provided evidence list, e.g. 'Revenue reached $128B [S1][S2].'. "
    "Only use source IDs that appear in the evidence list. "
    "Do not invent facts not present in the evidence."
)

_SYNTHESIS_SCHEMA = """{
  "section_body": "Full section content in markdown. Start with ### <section title>. Use #### sub-headers for distinct sub-topics where appropriate. Cite every factual claim with [Sx] immediately after the sentence. Write analytical prose, not a list of passages. If information on a sub-topic is unavailable, write: [Knowledge on this topic unavailable from known sources]",
  "section_summary": "2-3 sentence summary of this section's key findings and scope. Plain text, no markdown.",
  "used_source_ids": ["S1", "S3"]
}"""


class SectionSynthesizerAgent:
    """Synthesizes accepted scraping passages into a formal section body.

    Runs after check_data ACCEPT. Builds a per-section local source index,
    calls LLM to write structured markdown prose with sub-headers, and emits
    section_evidence for later remapping to the global source_index.
    """

    MAX_PASSAGES = 20
    MAX_PASSAGE_CHARS = 500

    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.logger = logging.getLogger(__name__)

    async def run(self, draft_state: dict) -> dict:
        topic = str(draft_state.get("topic") or "").strip() or "Section"
        task = draft_state.get("task") or {}
        research_context = draft_state.get("research_context") or {}
        scraping_packet = draft_state.get("scraping_packet") or {}

        local_index = self._build_local_index(scraping_packet)
        if not local_index:
            await self._emit_log(
                "section_synthesis_skip",
                f"No evidence passages for '{topic}'; keeping raw draft.",
            )
            return {}

        indexed_evidence_text = self._format_local_index(local_index)
        context_block = self._build_context_block(research_context)

        prompt = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Section title: {topic}\n\n"
                    f"{context_block}"
                    f"Evidence passages (cite using the IDs shown):\n{indexed_evidence_text}\n\n"
                    "Write a comprehensive, well-structured section based on the evidence above.\n"
                    "Requirements:\n"
                    "- Start with ### {section title}\n"
                    "- Use #### sub-headers for distinct sub-topics where appropriate\n"
                    "- Cite every factual claim with [Sx] immediately after the sentence\n"
                    "- Write analytical prose, not a list of passages\n"
                    "- If information on a sub-topic is unavailable, write: "
                    "[Knowledge on this topic unavailable from known sources]\n\n"
                    f"Return ONLY valid JSON (no markdown fences):\n{_SYNTHESIS_SCHEMA}"
                ),
            },
        ]

        route_context = build_route_context(
            application_name=str(task.get("application_name") or "auto_research_engine"),
            shared_agent_class="section_synthesizer_agent",
            agent_role="section_synthesizer",
            stage_name="section_synthesis",
            system_prompt=_SYNTHESIS_SYSTEM,
            task=topic,
            state=draft_state,
            task_payload=task,
        )

        await self._emit_log(
            "section_synthesis_start",
            f"Synthesizing section '{topic}' from {len(local_index)} evidence passages.",
        )

        result = await call_model(
            prompt,
            model=task.get("model"),
            response_format="json",
            route_context=route_context,
        )

        section_body = ""
        section_summary = ""
        used_ids: List[str] = []
        if isinstance(result, dict):
            section_body = str(result.get("section_body") or "").strip()
            section_summary = str(result.get("section_summary") or "").strip()
            used_ids = [
                str(x) for x in (result.get("used_source_ids") or [])
                if str(x) in local_index
            ]

        if not section_body:
            await self._emit_log(
                "section_synthesis_fallback",
                f"LLM returned empty body for '{topic}'; keeping raw draft.",
            )
            return {}

        section_evidence = [
            {
                "local_id": sid,
                "source_url": info["source_url"],
                "content": info["content"],
                "domain": info["domain"],
                "used_in_draft": sid in used_ids,
            }
            for sid, info in local_index.items()
        ]

        await self._emit_log(
            "section_synthesis_done",
            f"Section '{topic}' synthesized: {len(section_body)} chars, "
            f"{len(used_ids)}/{len(local_index)} passages cited.",
        )

        return {
            "draft": {topic: section_body},
            "section_evidence": section_evidence,
            "section_summary": section_summary,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_local_index(self, scraping_packet: dict) -> Dict[str, dict]:
        # Collect all passages across search_log entries
        all_passages = []
        for log_entry in scraping_packet.get("search_log", []):
            if not isinstance(log_entry, dict):
                continue
            for passage in log_entry.get("top_10_passages", []):
                if not isinstance(passage, dict):
                    continue
                content = str(passage.get("content") or "").strip()
                source_url = str(passage.get("source_url") or "").strip()
                if not content:
                    continue
                all_passages.append({
                    "content": content,
                    "source_url": source_url,
                    "relevance_score": float(passage.get("relevance_score") or 0.0),
                })

        # Sort by relevance score descending, then cap at MAX_PASSAGES
        all_passages.sort(key=lambda p: p["relevance_score"], reverse=True)

        index: Dict[str, dict] = {}
        for i, passage in enumerate(all_passages[:self.MAX_PASSAGES], start=1):
            source_url = passage["source_url"]
            index[f"S{i}"] = {
                "content": passage["content"],
                "source_url": source_url,
                "domain": self._extract_domain(source_url),
            }
        return index

    def _format_local_index(self, local_index: Dict[str, dict]) -> str:
        lines = []
        for key, info in local_index.items():
            content = (info.get("content") or "")[:self.MAX_PASSAGE_CHARS]
            lines.append(f"[{key}]({info.get('domain') or ''}) {content}")
        return "\n\n".join(lines)

    def _build_context_block(self, research_context: dict) -> str:
        parts = []
        description = str(research_context.get("description") or "").strip()
        key_points = research_context.get("key_points") or []
        if description:
            parts.append(f"Section scope: {description}")
        if key_points:
            kp_lines = "\n".join(f"  - {pt}" for pt in key_points if str(pt).strip())
            parts.append(f"Key investigation points:\n{kp_lines}")
        return ("\n".join(parts) + "\n\n") if parts else ""

    def _extract_domain(self, url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url if "://" in url else f"https://{url}")
            domain = parsed.netloc or parsed.path.split("/")[0]
            return domain.lower().removeprefix("www.")
        except Exception:
            return ""

    async def _emit_log(self, event: str, message: str) -> None:
        if self.websocket and self.stream_output:
            await self.stream_output("logs", event, message, self.websocket)
        else:
            print_agent_output(message, agent="SECTION_SYNTHESIZER")
        self.logger.info(message)
