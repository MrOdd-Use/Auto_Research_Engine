import pytest

from multi_agents.agents.editor import EditorAgent


class _FakeScrapingAgent:
    def __init__(self, counter):
        self.counter = counter

    async def run_depth_scraping(self, draft_state):
        self.counter["research"] += 1
        current_iteration = int(draft_state.get("iteration_index") or 1)
        return {
            "draft": {draft_state["topic"]: f"draft-at-{current_iteration}"},
            "scraping_packet": {
                "iteration_index": current_iteration,
                "model_level": "Level_1_Base",
                "active_engines": ["tavily"],
                "search_log": [],
            },
            "iteration_index": current_iteration,
        }


class _FakeCheckDataRetryThenAccept:
    def __init__(self, counter):
        self.counter = counter

    async def run(self, draft_state):
        self.counter["check_data"] += 1
        call_number = self.counter["check_data"]
        if call_number == 1:
            return {
                "check_data_action": "retry",
                "check_data_verdict": {
                    "status": "RETRY",
                    "deep_eval_report": {"final_score": 0.4},
                    "feedback_packet": {
                        "instruction": "Need audited actual value.",
                        "new_query_suggestion": "Nvidia 2026 revenue actual audited -projection -forecast",
                    },
                },
                "audit_feedback": {
                    "is_satisfied": False,
                    "confidence_score": 0.4,
                    "instruction": "Need audited actual value.",
                    "new_query_suggestion": "Nvidia 2026 revenue actual audited -projection -forecast",
                },
                "iteration_index": 2,
                "extra_hints": "Need audited actual value.",
            }

        return {
            "check_data_action": "accept",
            "check_data_verdict": {
                "status": "ACCEPT",
                "deep_eval_report": {"final_score": 0.9},
                "feedback_packet": {
                    "instruction": "",
                    "new_query_suggestion": "",
                },
            },
            "audit_feedback": {
                "is_satisfied": True,
                "confidence_score": 0.9,
            },
            "iteration_index": int(draft_state.get("iteration_index") or 2),
        }


class _FakeCheckDataBlocked:
    def __init__(self, counter):
        self.counter = counter

    async def run(self, draft_state):
        self.counter["check_data"] += 1
        topic = draft_state["topic"]
        return {
            "check_data_action": "blocked",
            "check_data_verdict": {
                "status": "BLOCKED",
                "deep_eval_report": {"final_score": 0.3},
                "feedback_packet": {
                    "instruction": "Evidence insufficient after retries.",
                    "new_query_suggestion": "retry with audited filing",
                },
            },
            "draft": {topic: "该章节证据不足，需人工复核。"},
            "review": None,
            "iteration_index": 3,
        }


class _FakeReviewer:
    def __init__(self, counter):
        self.counter = counter

    async def run(self, draft_state):
        self.counter["reviewer"] += 1
        return {"review": None}


class _FakeReviser:
    def __init__(self, counter):
        self.counter = counter

    async def run(self, draft_state):
        self.counter["reviser"] += 1
        return {
            "draft": draft_state.get("draft"),
            "revision_notes": "n/a",
        }


@pytest.mark.asyncio
async def test_editor_workflow_retries_then_accepts():
    counter = {"research": 0, "check_data": 0, "reviewer": 0, "reviser": 0}
    agent = EditorAgent()
    agent.enable_scraping = True

    workflow = agent._create_workflow(
        {
            "scraping": _FakeScrapingAgent(counter),
            "research": None,
            "check_data": _FakeCheckDataRetryThenAccept(counter),
            "reviewer": _FakeReviewer(counter),
            "reviser": _FakeReviser(counter),
        }
    )
    chain = workflow.compile()

    result = await chain.ainvoke(
        {
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 1,
        }
    )

    assert counter["research"] == 2
    assert counter["check_data"] == 2
    assert counter["reviewer"] == 0
    assert counter["reviser"] == 0
    assert result["check_data_action"] == "accept"
    assert result["check_data_verdict"]["status"] == "ACCEPT"


@pytest.mark.asyncio
async def test_editor_workflow_blocked_branch_skips_reviewer():
    counter = {"research": 0, "check_data": 0, "reviewer": 0, "reviser": 0}
    agent = EditorAgent()
    agent.enable_scraping = True

    workflow = agent._create_workflow(
        {
            "scraping": _FakeScrapingAgent(counter),
            "research": None,
            "check_data": _FakeCheckDataBlocked(counter),
            "reviewer": _FakeReviewer(counter),
            "reviser": _FakeReviser(counter),
        }
    )
    chain = workflow.compile()

    result = await chain.ainvoke(
        {
            "task": {},
            "topic": "2026 Nvidia AI GPU revenue",
            "research_context": {},
            "iteration_index": 1,
        }
    )

    assert counter["research"] == 1
    assert counter["check_data"] == 1
    assert counter["reviewer"] == 0
    assert result["check_data_action"] == "blocked"
    assert result["check_data_verdict"]["status"] == "BLOCKED"
    assert "需人工复核" in result["draft"]["2026 Nvidia AI GPU revenue"]
