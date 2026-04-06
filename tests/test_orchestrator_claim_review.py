import pytest

from multi_agents.agents.claim_verifier import ClaimVerifierAgent
from multi_agents.agents.orchestrator import ChiefEditorAgent


def _make_packet(passages):
    return {
        "search_log": [
            {
                "target": "test query",
                "top_10_passages": [
                    {"content": content, "source_url": url}
                    for content, url in passages
                ],
            }
        ]
    }


class _ReflexionEditor:
    def __init__(self):
        self.calls = []
        self.section_a_key = "section_0_section_a"

    def _make_section_key(self, section_index, header):
        if section_index == 0:
            return self.section_a_key
        return "section_1_section_b"

    async def run_parallel_research(
        self,
        state,
        session_recorder=None,
        selected_section_key=None,
        start_from_section_node=None,
        section_state_before=None,
        note=None,
    ):
        self.calls.append(
            {
                "selected_section_key": selected_section_key,
                "note": note,
            }
        )
        research_data = list(state.get("research_data") or [])
        scraping_packets = list(state.get("scraping_packets") or [])
        check_data_reports = list(state.get("check_data_reports") or [])

        if selected_section_key == self.section_a_key:
            research_data[0] = {"Section A": "updated evidence"}
            scraping_packets[0] = _make_packet(
                [
                    ("Section A metric is 23% actual", "https://c.com/evidence"),
                    ("Section A metric is 23% verified", "https://d.com/evidence"),
                ]
            )
            check_data_reports[0] = {"status": "ACCEPT"}

        return {
            "research_data": research_data,
            "scraping_packets": scraping_packets,
            "check_data_reports": check_data_reports,
        }


class _ReflexionWriter:
    def __init__(self):
        self.calls = 0
        self.notes = []

    async def run(self, state):
        self.calls += 1
        self.notes.append((state.get("task") or {}).get("checkpoint_note"))
        source_ids = sorted(
            (state.get("source_index") or {}).keys(),
            key=lambda key: int(str(key).lstrip("S")),
        )
        refreshed_ids = [sid for sid in source_ids if sid in {"S3", "S4"}]
        chosen_ids = refreshed_ids[:2] if len(refreshed_ids) >= 2 else source_ids[-2:]
        sentence = f"Section A metric is 23% [{chosen_ids[0]}][{chosen_ids[1]}]."
        return {
            "headers": {},
            "introduction": sentence,
            "conclusion": "",
            "sources": [],
            "claim_annotations": [
                {
                    "sentence": "Section A metric is 23%",
                    "source_ids": chosen_ids,
                    "section": "introduction",
                }
            ],
        }


@pytest.mark.asyncio
async def test_claim_review_reruns_section_then_rewrites_with_new_sources():
    chief = ChiefEditorAgent({"query": "claim reflexion"})
    verifier = ClaimVerifierAgent()
    editor = _ReflexionEditor()
    writer = _ReflexionWriter()

    async def mock_detect_conflicts(claim, source_index, model):
        source_ids = set(claim.get("source_ids") or [])
        if {"S1", "S2"}.issubset(source_ids):
            return {"has_conflict": True, "conflict_detail": "Old sources disagree"}
        return {"has_conflict": False, "conflict_detail": ""}

    verifier.detect_conflicts = mock_detect_conflicts
    chief._workflow_agents = {
        "claim_verifier": verifier,
        "editor": editor,
        "writer": writer,
    }

    state = {
        "task": {"model": "gpt-4o-mini"},
        "section_details": [
            {"header": "Section A"},
            {"header": "Section B"},
        ],
        "research_data": [
            {"Section A": "old evidence"},
            {"Section B": "unchanged"},
        ],
        "scraping_packets": [
            _make_packet(
                [
                    ("Section A metric is 23%", "https://a.com/evidence"),
                    ("Section A metric is 21.5%", "https://b.com/evidence"),
                ]
            ),
            _make_packet([]),
        ],
        "check_data_reports": [
            {"status": "ACCEPT"},
            {"status": "ACCEPT"},
        ],
        "introduction": "Section A metric is 23% [S1][S2].",
        "conclusion": "",
        "claim_annotations": [
            {
                "sentence": "Section A metric is 23%",
                "source_ids": ["S1", "S2"],
                "section": "introduction",
            }
        ],
    }

    state = await chief._inject_source_index(state)
    result = await chief._run_claim_review(state, recorder=None)

    assert editor.calls
    assert editor.calls[0]["selected_section_key"] == editor.section_a_key
    assert "Section A metric is 23%" in str(editor.calls[0]["note"] or "")
    assert writer.calls == 1
    assert "regenerate citations and claim_annotations" in str(writer.notes[0] or "")
    assert sorted(result["source_index"].keys(), key=lambda key: int(key[1:])) == ["S1", "S2", "S3", "S4"]
    assert "[S3]" in result["indexed_research_data"]
    assert "[S4]" in result["indexed_research_data"]
    assert result["claim_reflexion_iterations"] == 2
    assert result["claim_confidence_report"][0]["confidence"] == "HIGH"
    assert result["claim_confidence_report"][0]["source_ids"] == ["S3", "S4"]
