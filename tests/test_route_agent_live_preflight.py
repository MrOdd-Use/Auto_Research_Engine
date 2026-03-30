from multi_agents.route_agent.client import RouteAgentClient
from multi_agents.route_agent.tools.live_preflight import run_live_preflight


def test_default_model_pool_prefers_configured_models(monkeypatch):
    monkeypatch.setenv("ROUTE_AGENT_BACKEND", "local")
    monkeypatch.delenv("ROUTE_AGENT_MODEL_POOL", raising=False)
    monkeypatch.setenv("ROUTE_AGENT_MODEL_POOL", "deepseek:deepseek-chat,openai:gpt-4o")

    client = RouteAgentClient()

    assert client.model_pool == ["deepseek-chat", "gpt-4o"]


def test_live_preflight_writes_json_report(monkeypatch, tmp_path):
    monkeypatch.setenv("ROUTE_AGENT_BACKEND", "local")
    monkeypatch.setenv("ROUTE_AGENT_MODEL_POOL", "deepseek:deepseek-chat")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "present")
    monkeypatch.setenv("RETRIEVER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "present")
    monkeypatch.setenv("EMBEDDING", "openai:text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "present")

    output_path = tmp_path / "live_preflight.json"
    report = run_live_preflight(output_path=output_path)

    assert output_path.exists()
    assert report["filesystem"]["status"] == "ok"
    assert report["llms"][0]["status"] == "ok"
    assert report["llms"][0]["provider"] == "deepseek"
