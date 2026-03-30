import json

import pytest

from multi_agents.route_agent.tools.reference_app import run_reference_application


@pytest.mark.asyncio
async def test_reference_application_generates_artifacts_and_logs_rollbacks(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTE_AGENT_BACKEND", "local")
    output_dir = tmp_path / "route_agent_test"
    artifacts = await run_reference_application(output_dir=output_dir, seed=123)

    assert artifacts.initial_draft_path.exists()
    assert artifacts.final_draft_path.exists()
    assert artifacts.operation_log_path.exists()
    assert artifacts.initial_draft != artifacts.final_draft

    entries = [
        json.loads(line)
        for line in artifacts.operation_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    route_entries = [entry for entry in entries if entry.get("type") == "route_decision"]
    rollback_entries = [entry for entry in entries if entry.get("type") == "rollback_chain"]
    challenge_entries = [entry for entry in entries if entry.get("type") == "challenges_generated"]

    assert route_entries
    assert all("route_latency_ms" in entry for entry in route_entries)
    assert any(entry.get("agent_role") == "planner" for entry in route_entries)
    assert any(entry.get("agent_role") == "planner_revision" for entry in route_entries)
    assert any(entry.get("agent_role") == "section_researcher" for entry in route_entries)
    assert any(entry.get("agent_role") == "writer" for entry in route_entries)
    assert any(entry.get("agent_role") == "reviewer" for entry in route_entries)
    assert any(entry.get("agent_role") == "challenger" for entry in route_entries)
    assert any(entry.get("agent_role") == "reviser" for entry in route_entries)
    assert len(challenge_entries) == 1
    assert len(challenge_entries[0]["challenges"]) == 2
    assert {len(entry["nodes"]) for entry in rollback_entries} == {2, 3}
