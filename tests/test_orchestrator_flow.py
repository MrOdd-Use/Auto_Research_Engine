import json
from pathlib import Path

import pytest

from multi_agents.agents.orchestrator import ChiefEditorAgent


@pytest.fixture(autouse=True)
def isolate_output_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class _FakeResearchAgent:
    async def run_initial_research(self, state):
        return {"task": state.get("task"), "initial_research": "initial"}


class _FakeHumanAgent:
    async def review_plan(self, state):
        return {"human_feedback": None}


class _FakeEditorAgent:
    async def plan_research(self, state):
        return {
            "title": "My Report",
            "date": "01/01/2026",
            "sections": ["Section A"],
            "section_details": [{"header": "Section A", "description": "", "key_points": [], "research_queries": []}],
        }

    async def run_parallel_research(self, state):
        return {
            "research_data": [{"Section A": "Body A"}],
            "scraping_packets": [],
            "check_data_reports": [],
        }


class _FakeWriterAgent:
    async def run(self, state):
        return {
            "headers": {
                "title": "Title",
                "date": "Date",
                "introduction": "Introduction",
                "table_of_contents": "Table of Contents",
                "conclusion": "Conclusion",
                "references": "References",
            },
            "date": "01/01/2026",
            "introduction": "Intro",
            "table_of_contents": "- A",
            "conclusion": "Conc",
            "sources": ["- src"],
        }


class _FakePublisherAgent:
    def __init__(self):
        self.run_calls = 0

    def generate_layout(self, state):
        final_draft = state.get("final_draft")
        if isinstance(final_draft, str) and final_draft.strip():
            return final_draft
        return "BASE REPORT"

    async def run(self, state):
        self.run_calls += 1
        return {"report": self.generate_layout(state)}


class _LayoutPublisherAgent:
    def __init__(self):
        self.run_calls = 0

    def generate_layout(self, state):
        final_draft = state.get("final_draft")
        if isinstance(final_draft, str) and final_draft.strip():
            return final_draft

        headers = state.get("headers") or {}
        return (
            f"# {headers.get('title', 'Title')}\n"
            f"## {headers.get('introduction', 'Introduction')}\n"
            f"{state.get('introduction', '')}\n\n"
            f"## {headers.get('conclusion', 'Conclusion')}\n"
            f"{state.get('conclusion', '')}\n"
        )

    async def run(self, state):
        self.run_calls += 1
        return {"report": self.generate_layout(state)}


class _FakeReviewerAccept:
    def __init__(self):
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        return {"review": None}


class _CapturingReviewerAccept:
    def __init__(self):
        self.calls = 0
        self.seen_drafts = []

    async def run(self, state):
        self.calls += 1
        self.seen_drafts.append(state.get("draft"))
        return {"review": None}


class _FakeReviewerReviseThenAccept:
    def __init__(self):
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        if self.calls == 1:
            return {"review": "Needs improvement"}
        return {"review": None}


class _FakeReviser:
    def __init__(self):
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        return {"draft": "REVISED REPORT", "revision_notes": "Applied fixes"}

class _FakeReviewerAlwaysRevise:
    def __init__(self):
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        return {"review": "Needs improvement"}


class _FakeReviserIncremental:
    def __init__(self):
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        return {
            "draft": f"REVISED REPORT v{self.calls}",
            "revision_notes": f"Applied fixes v{self.calls}",
        }


class _FreshWriterAgent:
    async def run(self, state):
        return {
            "headers": {
                "title": "Fresh Title",
                "date": "Date",
                "introduction": "Introduction",
                "table_of_contents": "Table of Contents",
                "conclusion": "Conclusion",
                "references": "References",
            },
            "date": "01/01/2026",
            "introduction": "Fresh intro",
            "table_of_contents": "- A",
            "conclusion": "Fresh conclusion",
            "sources": ["- src"],
        }


class _LayoutReviser:
    async def run(self, state):
        return {
            "draft": (
                "# Fresh Title\n"
                "### Introduction:\n"
                "Revised intro\n\n"
                "### Conclusion:\n"
                "Revised conclusion\n"
            ),
            "revision_notes": "Updated the narrative",
        }


class _EchoClaimVerifier:
    MAX_REFLEXION = 3

    def __init__(self):
        self.run_inputs = []

    async def run(self, state):
        introduction = state.get("introduction") or ""
        self.run_inputs.append(introduction)
        return {
            "claim_confidence_report": [
                {
                    "claim_text": introduction,
                    "confidence": "HIGH",
                    "original_sentence": introduction,
                    "domains": [],
                    "note": "",
                    "source_ids": [],
                }
            ] if introduction else [],
            "suspicious_claims": [],
            "hallucinated_claims": [],
        }

    def annotate_draft(self, draft, report):
        claim_text = report[0]["claim_text"] if report else ""
        return f"{draft}\n\nANNOTATED:{claim_text}"


def test_default_output_dir_uses_single_session_folder():
    task = {"query": "workflow rerun support"}

    chief = ChiefEditorAgent(task)

    output_dir = Path(chief.output_dir)
    assert output_dir.parent == Path("outputs")
    assert output_dir.name.endswith("_workflow rerun support")
    assert (output_dir / "task.json").exists()
    assert json.loads((output_dir / "task.json").read_text(encoding="utf-8")) == task


@pytest.mark.asyncio
async def test_orchestrator_main_flow_reviewer_accepts_directly():
    task = {
        "query": "test",
        "publish_formats": {"markdown": False, "pdf": False, "docx": False},
        "follow_guidelines": True,
        "guidelines": ["must be clear"],
        "verbose": False,
    }
    chief = ChiefEditorAgent(task)

    reviewer = _FakeReviewerAccept()
    reviser = _FakeReviser()
    publisher = _FakePublisherAgent()

    chief._initialize_agents = lambda: {
        "writer": _FakeWriterAgent(),
        "editor": _FakeEditorAgent(),
        "research": _FakeResearchAgent(),
        "publisher": publisher,
        "human": _FakeHumanAgent(),
        "reviewer": reviewer,
        "reviser": reviser,
    }

    result = await chief.run_research_task(task_id="t1")

    assert reviewer.calls == 1
    assert reviser.calls == 0
    assert publisher.run_calls == 1
    assert result["report"] == "BASE REPORT"


@pytest.mark.asyncio
async def test_orchestrator_main_flow_revise_then_publish():
    task = {
        "query": "test",
        "publish_formats": {"markdown": False, "pdf": False, "docx": False},
        "follow_guidelines": True,
        "guidelines": ["must be clear"],
        "verbose": False,
    }
    chief = ChiefEditorAgent(task)

    reviewer = _FakeReviewerReviseThenAccept()
    reviser = _FakeReviser()
    publisher = _FakePublisherAgent()

    chief._initialize_agents = lambda: {
        "writer": _FakeWriterAgent(),
        "editor": _FakeEditorAgent(),
        "research": _FakeResearchAgent(),
        "publisher": publisher,
        "human": _FakeHumanAgent(),
        "reviewer": reviewer,
        "reviser": reviser,
    }

    result = await chief.run_research_task(task_id="t2")

    assert reviewer.calls == 2
    assert reviser.calls == 1
    assert publisher.run_calls == 1
    assert result["report"] == "REVISED REPORT"


@pytest.mark.asyncio
async def test_orchestrator_main_flow_publishes_after_max_review_rounds():
    task = {
        "query": "test",
        "publish_formats": {"markdown": False, "pdf": False, "docx": False},
        "follow_guidelines": True,
        "guidelines": ["must be clear"],
        "verbose": False,
        "max_review_rounds": 3,
    }
    chief = ChiefEditorAgent(task)

    reviewer = _FakeReviewerAlwaysRevise()
    reviser = _FakeReviserIncremental()
    publisher = _FakePublisherAgent()

    chief._initialize_agents = lambda: {
        "writer": _FakeWriterAgent(),
        "editor": _FakeEditorAgent(),
        "research": _FakeResearchAgent(),
        "publisher": publisher,
        "human": _FakeHumanAgent(),
        "reviewer": reviewer,
        "reviser": reviser,
    }

    result = await chief.run_research_task(task_id="t3")

    assert reviewer.calls == 3
    assert reviser.calls == 3
    assert publisher.run_calls == 1
    assert result["report"] == "REVISED REPORT v3"


@pytest.mark.asyncio
async def test_writer_rerun_discards_stale_final_draft_and_publishes_fresh_layout():
    task = {
        "query": "test",
        "publish_formats": {"markdown": False, "pdf": False, "docx": False},
        "follow_guidelines": True,
        "guidelines": ["must be clear"],
        "verbose": False,
    }
    chief = ChiefEditorAgent(task)

    reviewer = _CapturingReviewerAccept()
    publisher = _LayoutPublisherAgent()

    chief._initialize_agents = lambda: {
        "writer": _FreshWriterAgent(),
        "editor": _FakeEditorAgent(),
        "research": _FakeResearchAgent(),
        "publisher": publisher,
        "human": _FakeHumanAgent(),
        "reviewer": reviewer,
        "reviser": _FakeReviser(),
    }

    initial_state = {
        "task": task,
        "final_draft": "STALE REPORT",
        "report": "STALE REPORT",
        "review_iterations": 2,
        "claim_confidence_report": [{"claim_text": "old"}],
    }
    result = await chief.run_research_task(
        task_id="writer-rerun",
        start_node="writer",
        initial_state=initial_state,
        include_human_feedback=False,
    )

    assert reviewer.calls == 1
    assert reviewer.seen_drafts[0] != "STALE REPORT"
    assert "Fresh intro" in reviewer.seen_drafts[0]
    assert "Fresh intro" in result["report"]
    assert "STALE REPORT" not in result["report"]


@pytest.mark.asyncio
async def test_reviser_output_refreshes_claim_report_before_annotation():
    task = {
        "query": "test",
        "publish_formats": {"markdown": False, "pdf": False, "docx": False},
        "follow_guidelines": True,
        "guidelines": ["must be clear"],
        "verbose": False,
    }
    chief = ChiefEditorAgent(task)

    verifier = _EchoClaimVerifier()
    publisher = _LayoutPublisherAgent()
    reviewer = _FakeReviewerReviseThenAccept()

    chief._initialize_agents = lambda: {
        "writer": _FreshWriterAgent(),
        "editor": _FakeEditorAgent(),
        "research": _FakeResearchAgent(),
        "publisher": publisher,
        "human": _FakeHumanAgent(),
        "reviewer": reviewer,
        "reviser": _LayoutReviser(),
        "claim_verifier": verifier,
    }

    result = await chief.run_research_task(task_id="refresh-claim-report")

    assert reviewer.calls == 2
    assert verifier.run_inputs[0] == "Fresh intro"
    assert verifier.run_inputs[-1] == "Revised intro"
    assert "ANNOTATED:Revised intro" in result["report"]
