import os
from pathlib import Path

import pytest

from backend.server.server_utils import Researcher


@pytest.mark.asyncio
async def test_researcher_logging(monkeypatch):
    async def fake_run_research_task(*args, **kwargs):
        return {"report": "# Synthetic report\n\nThis is a test report."}

    monkeypatch.setattr(
        "backend.server.server_utils.run_research_task",
        fake_run_research_task,
    )

    researcher = Researcher(
        query="Test query for logging verification",
        report_type="multi_agents",
    )
    result = await researcher.research()

    assert "output" in result
    assert "json" in result["output"]

    json_path = Path(result["output"]["json"])
    assert json_path.exists()

    os.remove(json_path)
