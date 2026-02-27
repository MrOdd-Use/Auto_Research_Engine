import pytest

from multi_agents.agents.orchestrator import ChiefEditorAgent


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
            "scrap_packets": [],
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


class _FakeReviewerAccept:
    def __init__(self):
        self.calls = 0

    async def run(self, state):
        self.calls += 1
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
