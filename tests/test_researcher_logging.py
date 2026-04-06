from pathlib import Path
from urllib.parse import unquote

import pytest

from backend.server.server_utils import Researcher


@pytest.mark.asyncio
async def test_researcher_logging(tmp_path, monkeypatch):
    async def fake_run_research_task(*args, **kwargs):
        return {"report": "# Synthetic report\n\nThis is a test report."}

    monkeypatch.chdir(tmp_path)
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
    md_path = Path(unquote(result["output"]["md"]))
    docx_path = Path(unquote(result["output"]["docx"]))
    pdf_output = unquote(result["output"]["pdf"])

    assert json_path.exists()
    assert md_path.exists()
    assert docx_path.exists()
    assert json_path.parent == md_path.parent == docx_path.parent
    assert json_path.name == "session.json"
    assert md_path.name == "report.md"

    if pdf_output:
        pdf_path = Path(pdf_output)
        assert pdf_path.exists()
        assert pdf_path.parent == json_path.parent
