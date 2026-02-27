import pytest

from multi_agents.agents.scrap import ScrapAgent
from multi_agents.agents.state_controller import StateController


def _make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRAP_MMR_USE_EMBEDDINGS", "0")
    agent = ScrapAgent()
    agent.state_controller = StateController(str(tmp_path / "agent_state.json"))

    async def fake_decompose_targets(topic, research_context, extra_hints, model_name):
        return [f"{topic} target A", f"{topic} target B", f"{topic} target C"]

    async def fake_scrape_urls(urls):
        return [
            {
                "url": url,
                "raw_content": (f"{url} objective data point with conflicting claims. " * 40).strip(),
                "title": "Stub",
            }
            for url in urls
        ]

    monkeypatch.setattr(agent, "_decompose_targets", fake_decompose_targets)
    monkeypatch.setattr(agent, "_scrape_urls", fake_scrape_urls)
    return agent


@pytest.mark.asyncio
async def test_scrap_round1_uses_tavily(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    seen_engines = set()

    async def fake_search_with_engine(engine, query, query_domains, max_results):
        seen_engines.add(engine)
        return (
            [{"href": f"https://{engine}.example.com/{query.replace(' ', '-')}", "body": "body", "title": "t"}],
            False,
        )

    monkeypatch.setattr(agent, "_search_with_engine", fake_search_with_engine)
    result = await agent.run_depth_scrap({"task": {}, "topic": "AI Chips", "research_context": {}})

    assert seen_engines == {"tavily"}
    assert result["scrap_packet"]["iteration_index"] == 1


@pytest.mark.asyncio
async def test_scrap_round2_domain_routing(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)

    async def fake_search_with_engine(engine, query, query_domains, max_results):
        return (
            [{"href": f"https://{engine}.example.com/{hash(query)}", "body": "body", "title": "t"}],
            False,
        )

    monkeypatch.setattr(agent, "_search_with_engine", fake_search_with_engine)
    result = await agent.run_depth_scrap(
        {
            "task": {},
            "topic": "AI model scaling limits",
            "research_context": {},
            "audit_feedback": {"is_satisfied": False, "confidence_score": 0.8},
        }
    )

    assert result["scrap_packet"]["iteration_index"] == 2
    assert result["scrap_packet"]["active_engines"] == ["arxiv", "google", "semantic_scholar"]


def test_url_set_union_dedup(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    deduped = agent._dedupe_search_results(
        [
            {"href": "https://example.com/a?b=2&a=1", "body": "x", "title": "1", "source_engine": "tavily"},
            {"href": "https://example.com/a/?a=1&b=2", "body": "y", "title": "2", "source_engine": "google"},
        ]
    )
    assert len(deduped) == 1


@pytest.mark.asyncio
async def test_passage_mmr_top10_diversity(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    passages = []
    for i in range(20):
        passages.append(
            {
                "content": f"passage {i} on revenue growth risk contradiction data point {'x' * 220}",
                "source_url": f"https://example.com/{i % 5}",
                "metadata": {},
            }
        )
    top = await agent._select_top_passages_with_mmr("revenue risk contradiction", passages, top_k=10)
    assert len(top) == 10
    assert len({item["content"] for item in top}) == 10


@pytest.mark.asyncio
async def test_external_feedback_gate(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)

    async def fake_search_with_engine(engine, query, query_domains, max_results):
        return (
            [{"href": f"https://{engine}.example.com/{hash(query)}", "body": "body", "title": "t"}],
            False,
        )

    monkeypatch.setattr(agent, "_search_with_engine", fake_search_with_engine)
    result = await agent.run_depth_scrap({"task": {}, "topic": "Nvidia revenue", "research_context": {}})
    assert result["scrap_packet"]["iteration_index"] == 1


@pytest.mark.asyncio
async def test_fallback_on_vertical_engine_failure(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)

    async def fake_search_with_engine(engine, query, query_domains, max_results):
        if engine == "arxiv":
            return [], True
        return (
            [{"href": f"https://{engine}.example.com/{hash(query)}", "body": "body", "title": "t"}],
            False,
        )

    monkeypatch.setattr(agent, "_search_with_engine", fake_search_with_engine)
    _, used_engines = await agent._collect_results_for_target(
        target="AI chips",
        engines=["arxiv"],
        query_domains=[],
        max_results=3,
    )
    assert "google" in used_engines
    assert "bing" in used_engines


@pytest.mark.asyncio
async def test_output_contract_prd_json(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)

    async def fake_search_with_engine(engine, query, query_domains, max_results):
        return (
            [{"href": f"https://{engine}.example.com/{hash(query)}", "body": "body", "title": "t"}],
            False,
        )

    monkeypatch.setattr(agent, "_search_with_engine", fake_search_with_engine)
    result = await agent.run_depth_scrap({"task": {}, "topic": "AI export control", "research_context": {}})
    packet = result["scrap_packet"]

    assert {"iteration_index", "model_level", "active_engines", "search_log"} <= set(packet.keys())
    assert isinstance(packet["search_log"], list)
    assert "confidence_score" not in packet
    assert "truthfulness_rating" not in packet


@pytest.mark.asyncio
async def test_writer_compat_research_data_present(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)

    async def fake_search_with_engine(engine, query, query_domains, max_results):
        return (
            [{"href": f"https://{engine}.example.com/{hash(query)}", "body": "body", "title": "t"}],
            False,
        )

    monkeypatch.setattr(agent, "_search_with_engine", fake_search_with_engine)
    result = await agent.run_depth_scrap({"task": {}, "topic": "Semiconductor policy", "research_context": {}})

    assert isinstance(result["draft"], dict)
    assert "Semiconductor policy" in result["draft"]
    assert isinstance(result["draft"]["Semiconductor policy"], str)
    assert result["draft"]["Semiconductor policy"].strip()
