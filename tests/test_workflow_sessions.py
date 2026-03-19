import pytest

from backend.server.workflow_store import WorkflowStore
from multi_agents.agents.editor import EditorAgent
from multi_agents.agents.orchestrator import ChiefEditorAgent
from multi_agents.workflow_session import WorkflowSessionRecorder


class _FakeInitialResearchAgent:
    async def run_initial_research(self, state):
        return {"task": state.get("task"), "initial_research": "initial research"}


class _FakeHumanAgent:
    async def review_plan(self, state):
        return {"human_feedback": None}


class _CountingScrapAgent:
    def __init__(self):
        self.calls_by_topic = {}
        self.note_log = []

    async def run_depth_scrap(self, draft_state):
        topic = draft_state["topic"]
        self.calls_by_topic[topic] = self.calls_by_topic.get(topic, 0) + 1
        self.note_log.append(
            {
                "topic": topic,
                "checkpoint_note": (draft_state.get("task") or {}).get("checkpoint_note"),
                "extra_hints": draft_state.get("extra_hints"),
            }
        )
        iteration = int(draft_state.get("iteration_index") or 1)
        return {
            "draft": {topic: f"{topic} draft v{self.calls_by_topic[topic]}"},
            "scrap_packet": {
                "iteration_index": iteration,
                "model_level": "Level_1_Base",
                "active_engines": ["tavily"],
                "search_log": [],
            },
            "iteration_index": iteration,
        }


class _CountingCheckDataAgent:
    def __init__(self):
        self.calls_by_topic = {}

    async def run(self, draft_state):
        topic = draft_state["topic"]
        self.calls_by_topic[topic] = self.calls_by_topic.get(topic, 0) + 1
        return {
            "check_data_action": "accept",
            "check_data_verdict": {
                "status": "ACCEPT",
                "deep_eval_report": {"final_score": 0.95},
                "feedback_packet": {
                    "instruction": "",
                    "new_query_suggestion": "",
                },
            },
            "iteration_index": int(draft_state.get("iteration_index") or 1),
        }


class _NoopSectionReviewer:
    async def run(self, draft_state):
        return {"review": None}


class _NoopSectionReviser:
    async def run(self, draft_state):
        return {"draft": draft_state.get("draft"), "revision_notes": "n/a"}


class _CheckpointEditorAgent(EditorAgent):
    def __init__(self, section_titles):
        super().__init__(None, None, None, {})
        self.enable_scrap = True
        self.section_titles = section_titles
        self.scrap_agent = _CountingScrapAgent()
        self.check_data_agent = _CountingCheckDataAgent()

    async def plan_research(self, state):
        section_details = [
            {
                "header": title,
                "description": f"{title} scope",
                "key_points": [f"{title} key point"],
                "research_queries": [f"{title} research query"],
            }
            for title in self.section_titles
        ]
        return {
            "title": "Workflow Report",
            "date": "01/01/2026",
            "sections": list(self.section_titles),
            "section_details": section_details,
        }

    def _initialize_agents(self):
        return {
            "research": None,
            "scrap": self.scrap_agent,
            "check_data": self.check_data_agent,
            "reviewer": _NoopSectionReviewer(),
            "reviser": _NoopSectionReviser(),
        }


class _CountingWriterAgent:
    def __init__(self):
        self.calls = 0
        self.notes = []

    async def run(self, state):
        self.calls += 1
        task = state.get("task") or {}
        self.notes.append(task.get("checkpoint_note"))
        return {
            "headers": {
                "title": "Workflow Title",
                "date": "01/01/2026",
                "introduction": "Introduction",
                "table_of_contents": "Table of Contents",
                "conclusion": "Conclusion",
                "references": "References",
            },
            "date": "01/01/2026",
            "introduction": "Intro",
            "table_of_contents": "- A\n- B",
            "conclusion": "Conc",
            "sources": ["- src"],
        }


class _FailingWriterAgent:
    async def run(self, state):
        raise RuntimeError("writer failed on rerun")


class _ScheduledReviewerAgent:
    def __init__(self, schedule):
        self.schedule = list(schedule)
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        review = self.schedule.pop(0) if self.schedule else None
        return {"review": review}


class _CountingReviserAgent:
    def __init__(self):
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        return {
            "draft": f"REVISED REPORT v{self.calls}",
            "revision_notes": f"Applied fixes v{self.calls}",
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


def _build_task():
    return {
        "query": "workflow rerun support",
        "report_source": "web",
        "tone": "Objective",
        "publish_formats": {"markdown": False, "pdf": False, "docx": False},
        "follow_guidelines": True,
        "guidelines": ["be clear"],
        "verbose": False,
        "max_review_rounds": 3,
    }


def _build_chief(task, *, editor, writer, reviewer, reviser, publisher):
    chief = ChiefEditorAgent(task)
    chief._initialize_agents = lambda: {
        "writer": writer,
        "editor": editor,
        "research": _FakeInitialResearchAgent(),
        "publisher": publisher,
        "human": _FakeHumanAgent(),
        "reviewer": reviewer,
        "reviser": reviser,
    }
    return chief


async def _create_recorder(
    workflow_store,
    report_id,
    task_query,
    *,
    parent_session_id=None,
    rerun_from_checkpoint_id=None,
    note=None,
    target=None,
):
    session = await workflow_store.create_session(
        report_id,
        parent_session_id=parent_session_id,
        rerun_from_checkpoint_id=rerun_from_checkpoint_id,
        note=note,
        target=target,
        task_query=task_query,
    )
    return WorkflowSessionRecorder(workflow_store, report_id, session)


def _find_checkpoint(session, *, node_name, scope=None, section_key=None):
    for checkpoint in session.get("checkpoints") or []:
        if checkpoint.get("node_name") != node_name:
            continue
        if scope is not None and checkpoint.get("scope") != scope:
            continue
        if section_key is not None and checkpoint.get("section_key") != section_key:
            continue
        return checkpoint
    raise AssertionError(f"checkpoint not found: node={node_name}, scope={scope}, section={section_key}")


@pytest.mark.asyncio
async def test_initial_run_persists_global_and_section_checkpoints(tmp_path):
    workflow_store = WorkflowStore(tmp_path / "workflows")
    task = _build_task()
    report_id = "report-initial"

    editor = _CheckpointEditorAgent(["Section A", "Section B"])
    writer = _CountingWriterAgent()
    reviewer = _ScheduledReviewerAgent([None])
    reviser = _CountingReviserAgent()
    publisher = _FakePublisherAgent()
    chief = _build_chief(
        task,
        editor=editor,
        writer=writer,
        reviewer=reviewer,
        reviser=reviser,
        publisher=publisher,
    )

    recorder = await _create_recorder(
        workflow_store,
        report_id,
        task["query"],
        target={"scope": "global", "node_name": "browser"},
    )
    result = await chief.run_research_task(task_id="initial", session_recorder=recorder)

    assert result["report"] == "BASE REPORT"

    saved_session = await workflow_store.get_session(report_id, recorder.session_id)
    assert saved_session["status"] == "completed"
    assert saved_session["round_index"] == 1

    global_nodes = [
        checkpoint["node_name"]
        for checkpoint in saved_session["checkpoints"]
        if checkpoint["scope"] == "global"
    ]
    assert global_nodes == ["browser", "planner", "human", "researcher", "writer", "reviewer", "publisher"]

    section_nodes = [
        checkpoint["node_name"]
        for checkpoint in saved_session["checkpoints"]
        if checkpoint["scope"] == "section_node"
    ]
    assert section_nodes.count("scrap") == 2
    assert section_nodes.count("check_data") == 2

    workflow_response = await workflow_store.build_workflow_response(report_id)
    assert workflow_response["workflow_available"] is True
    assert workflow_response["current_session_id"] == recorder.session_id
    assert workflow_response["last_successful_session_id"] == recorder.session_id
    assert len(workflow_response["selected_session"]["checkpoints_tree"]["sections"]) == 2


@pytest.mark.asyncio
async def test_section_rerun_only_reexecutes_selected_section_and_creates_new_round(tmp_path):
    workflow_store = WorkflowStore(tmp_path / "workflows")
    task = _build_task()
    report_id = "report-section-rerun"

    editor = _CheckpointEditorAgent(["Section A", "Section B"])
    writer = _CountingWriterAgent()
    reviewer = _ScheduledReviewerAgent([None, None])
    reviser = _CountingReviserAgent()
    publisher = _FakePublisherAgent()
    chief = _build_chief(
        task,
        editor=editor,
        writer=writer,
        reviewer=reviewer,
        reviser=reviser,
        publisher=publisher,
    )

    recorder_1 = await _create_recorder(
        workflow_store,
        report_id,
        task["query"],
        target={"scope": "global", "node_name": "browser"},
    )
    await chief.run_research_task(task_id="round-1", session_recorder=recorder_1)
    session_1 = await workflow_store.get_session(report_id, recorder_1.session_id)

    section_checkpoint = _find_checkpoint(session_1, node_name="scrap", scope="section_node")
    selected_section_key = section_checkpoint["section_key"]
    selected_topic = section_checkpoint["section_title"]
    other_topic = next(title for title in editor.section_titles if title != selected_topic)

    recorder_2 = await _create_recorder(
        workflow_store,
        report_id,
        task["query"],
        parent_session_id=recorder_1.session_id,
        rerun_from_checkpoint_id=section_checkpoint["checkpoint_id"],
        note="Use audited sources only",
        target={
            "scope": "section_node",
            "node_name": section_checkpoint["node_name"],
            "section_key": selected_section_key,
            "section_title": section_checkpoint["section_title"],
        },
    )
    await chief.run_research_task(
        task_id="round-2",
        session_recorder=recorder_2,
        start_node="researcher",
        initial_state=session_1["final_state"],
        include_human_feedback=False,
        note="Use audited sources only",
        selected_section_key=selected_section_key,
        section_start_node=section_checkpoint["node_name"],
        section_state_before=section_checkpoint["state_before"],
    )
    session_2 = await workflow_store.get_session(report_id, recorder_2.session_id)

    assert session_2["parent_session_id"] == recorder_1.session_id
    assert session_2["round_index"] == 2
    assert session_2["rerun_from_checkpoint_id"] == section_checkpoint["checkpoint_id"]
    assert editor.scrap_agent.calls_by_topic[selected_topic] == 2
    assert editor.scrap_agent.calls_by_topic[other_topic] == 1
    assert editor.check_data_agent.calls_by_topic[selected_topic] == 2
    assert editor.check_data_agent.calls_by_topic[other_topic] == 1
    assert writer.calls == 2
    assert reviewer.calls == 2
    assert publisher.run_calls == 2

    session_2_section_keys = {
        checkpoint["section_key"]
        for checkpoint in session_2["checkpoints"]
        if checkpoint["scope"] == "section_node"
    }
    assert session_2_section_keys == {selected_section_key}
    assert any(
        entry["topic"] == selected_topic and "Use audited sources only" in str(entry["extra_hints"] or "")
        for entry in editor.scrap_agent.note_log
    )


@pytest.mark.asyncio
async def test_multi_round_reruns_track_parentage_and_preserve_last_successful_after_failure(tmp_path):
    workflow_store = WorkflowStore(tmp_path / "workflows")
    task = _build_task()
    report_id = "report-multi-round"

    editor = _CheckpointEditorAgent(["Section A", "Section B"])
    writer = _CountingWriterAgent()
    reviewer = _ScheduledReviewerAgent([None, None, "Needs rewrite", None])
    reviser = _CountingReviserAgent()
    publisher = _FakePublisherAgent()

    chief = _build_chief(
        task,
        editor=editor,
        writer=writer,
        reviewer=reviewer,
        reviser=reviser,
        publisher=publisher,
    )

    recorder_1 = await _create_recorder(
        workflow_store,
        report_id,
        task["query"],
        target={"scope": "global", "node_name": "browser"},
    )
    await chief.run_research_task(task_id="round-1", session_recorder=recorder_1)
    session_1 = await workflow_store.get_session(report_id, recorder_1.session_id)

    section_checkpoint = _find_checkpoint(session_1, node_name="scrap", scope="section_node")
    recorder_2 = await _create_recorder(
        workflow_store,
        report_id,
        task["query"],
        parent_session_id=recorder_1.session_id,
        rerun_from_checkpoint_id=section_checkpoint["checkpoint_id"],
        note="Refresh section evidence",
        target={
            "scope": "section_node",
            "node_name": section_checkpoint["node_name"],
            "section_key": section_checkpoint["section_key"],
            "section_title": section_checkpoint["section_title"],
        },
    )
    await chief.run_research_task(
        task_id="round-2",
        session_recorder=recorder_2,
        start_node="researcher",
        initial_state=session_1["final_state"],
        include_human_feedback=False,
        note="Refresh section evidence",
        selected_section_key=section_checkpoint["section_key"],
        section_start_node=section_checkpoint["node_name"],
        section_state_before=section_checkpoint["state_before"],
    )
    session_2 = await workflow_store.get_session(report_id, recorder_2.session_id)

    writer_checkpoint = _find_checkpoint(session_2, node_name="writer", scope="global")
    recorder_3 = await _create_recorder(
        workflow_store,
        report_id,
        task["query"],
        parent_session_id=recorder_2.session_id,
        rerun_from_checkpoint_id=writer_checkpoint["checkpoint_id"],
        note="Rewrite the narrative",
        target={"scope": "global", "node_name": "writer"},
    )
    await chief.run_research_task(
        task_id="round-3",
        session_recorder=recorder_3,
        start_node="writer",
        initial_state=writer_checkpoint["state_before"],
        include_human_feedback=False,
        note="Rewrite the narrative",
    )
    session_3 = await workflow_store.get_session(report_id, recorder_3.session_id)

    assert session_3["parent_session_id"] == recorder_2.session_id
    assert session_3["round_index"] == 3
    assert session_2["final_state"].get("review_iterations", 0) == 0
    assert session_3["final_state"].get("review_iterations", 0) == 1
    assert reviewer.calls == 4
    assert reviser.calls == 1

    writer_checkpoint_round_3 = _find_checkpoint(session_3, node_name="writer", scope="global")
    failing_chief = _build_chief(
        task,
        editor=editor,
        writer=_FailingWriterAgent(),
        reviewer=reviewer,
        reviser=reviser,
        publisher=publisher,
    )
    recorder_4 = await _create_recorder(
        workflow_store,
        report_id,
        task["query"],
        parent_session_id=recorder_3.session_id,
        rerun_from_checkpoint_id=writer_checkpoint_round_3["checkpoint_id"],
        note="This rerun should fail",
        target={"scope": "global", "node_name": "writer"},
    )

    with pytest.raises(RuntimeError, match="writer failed on rerun"):
        await failing_chief.run_research_task(
            task_id="round-4",
            session_recorder=recorder_4,
            start_node="writer",
            initial_state=writer_checkpoint_round_3["state_before"],
            include_human_feedback=False,
            note="This rerun should fail",
        )

    failed_session = await workflow_store.get_session(report_id, recorder_4.session_id)
    index = await workflow_store.get_index(report_id)

    assert failed_session["status"] == "failed"
    assert index["current_session_id"] == recorder_4.session_id
    assert index["last_successful_session_id"] == recorder_3.session_id

    last_successful = await workflow_store.get_session(report_id, recorder_3.session_id)
    assert last_successful["answer"] == session_3["answer"]
