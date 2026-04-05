import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from gpt_researcher.llm_provider.generic import base as llm_base
from multi_agents.route_agent.client import RouteAgentClient
from multi_agents.route_agent.models import RouteExecutionContext, RouteRequest
from multi_agents.route_agent.tools import live_preflight as live_preflight_module
from multi_agents.route_agent.tools.external_bridge import ExternalRouteAgentBridge
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
    original_find_spec = live_preflight_module.importlib.util.find_spec
    monkeypatch.setattr(
        live_preflight_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "langchain_openai" else original_find_spec(name),
    )

    output_path = tmp_path / "live_preflight.json"
    report = run_live_preflight(output_path=output_path)

    assert output_path.exists()
    assert report["filesystem"]["status"] == "ok"
    assert report["llms"][0]["status"] == "ok"
    assert report["llms"][0]["provider"] == "deepseek"


def test_live_preflight_merges_external_live_probe_results(monkeypatch, tmp_path):
    class _FakeBridge:
        project_path = "D:/agent/Route_Agent"
        app_env_path = "D:/agent/Auto_Research_Engine/.env"

        def is_available(self):
            return True

        def probe_global_pool(self, *, force=False, limit=None, mark_unavailable=True):
            assert force is False
            assert limit is None
            assert mark_unavailable is True
            return [
                {
                    "model_id": "relay:good-model",
                    "provider": "relay",
                    "model": "good-model",
                    "ok": True,
                    "skipped": False,
                    "cached": False,
                    "message": "live probe ok",
                },
                {
                    "model_id": "relay:bad-model",
                    "provider": "relay",
                    "model": "bad-model",
                    "ok": False,
                    "skipped": False,
                    "cached": False,
                    "message": "no available channel",
                },
            ]

    class _FakeConfig:
        retrievers = []
        embedding_provider = ""
        embedding_model = ""

    class _FakeClient:
        backend = "external"
        is_external_backend = True
        external_error = ""
        external_bridge = _FakeBridge()
        _external_bridge = external_bridge

        def describe_model_pool(self):
            return [
                {"model_id": "relay:good-model", "provider": "relay", "model": "good-model"},
                {"model_id": "relay:bad-model", "provider": "relay", "model": "bad-model"},
            ]

    monkeypatch.setattr(live_preflight_module, "Config", lambda: _FakeConfig())
    monkeypatch.setattr(live_preflight_module, "RouteAgentClient", lambda: _FakeClient())
    monkeypatch.setenv("RELAY_API_KEY", "present")
    monkeypatch.setenv("RELAY_BASE_URL", "https://relay.example")
    original_find_spec = live_preflight_module.importlib.util.find_spec
    monkeypatch.setattr(
        live_preflight_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "langchain_openai" else original_find_spec(name),
    )

    output_path = tmp_path / "live_preflight.json"
    report = run_live_preflight(
        output_path=output_path,
        include_live_model_probe=True,
    )

    assert report["route_agent"]["status"] == "ok"
    assert report["route_agent"]["live_probe"]["filtered_count"] == 1
    bad_check = next(item for item in report["llms"] if item["model"] == "bad-model")
    assert bad_check["status"] == "warning"
    assert bad_check["live_probe"]["status"] == "warning"


def test_live_preflight_marks_external_backend_not_ready_when_no_model_is_reachable(monkeypatch, tmp_path):
    class _FakeBridge:
        project_path = "D:/agent/Route_Agent"
        app_env_path = "D:/agent/Auto_Research_Engine/.env"

        def is_available(self):
            return True

        def probe_global_pool(self, *, force=False, limit=None, mark_unavailable=True):
            return [
                {
                    "model_id": "relay:bad-model",
                    "provider": "relay",
                    "model": "bad-model",
                    "ok": False,
                    "skipped": False,
                    "cached": False,
                    "message": "no available channel",
                }
            ]

    class _FakeConfig:
        retrievers = []
        embedding_provider = ""
        embedding_model = ""

    class _FakeClient:
        backend = "external"
        is_external_backend = True
        external_error = ""
        external_bridge = _FakeBridge()
        _external_bridge = external_bridge

        def describe_model_pool(self):
            return [
                {"model_id": "relay:bad-model", "provider": "relay", "model": "bad-model"},
            ]

    monkeypatch.setattr(live_preflight_module, "Config", lambda: _FakeConfig())
    monkeypatch.setattr(live_preflight_module, "RouteAgentClient", lambda: _FakeClient())
    monkeypatch.setenv("RELAY_API_KEY", "present")
    monkeypatch.setenv("RELAY_BASE_URL", "https://relay.example")

    output_path = tmp_path / "live_preflight.json"
    report = run_live_preflight(
        output_path=output_path,
        include_live_model_probe=True,
    )

    assert report["ready"] is False
    assert report["route_agent"]["status"] == "error"
    assert report["route_agent"]["reachable_model_count"] == 0


def test_grouped_relay_provider_reuses_global_api_key(monkeypatch):
    monkeypatch.setenv("RELAY_API_KEY", "shared-key")
    monkeypatch.setenv("RELAY_CC_GLM_BASE_URL", "https://nexus.itssx.com/api/claude_code/cc_glm")
    original_find_spec = live_preflight_module.importlib.util.find_spec
    monkeypatch.setattr(
        live_preflight_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "langchain_openai" else original_find_spec(name),
    )

    check = live_preflight_module._build_llm_check(
        {
            "model_id": "relay_cc_glm:glm-4.7",
            "provider": "relay_cc_glm",
            "model": "glm-4.7",
        },
        source="route_agent_global_pool",
    )

    assert check["status"] == "ok"
    assert check["provider"] == "relay_cc_glm"
    assert check["required_envs"] == [
        "RELAY_CC_GLM_BASE_URL or RELAY_BASE_URL",
        "RELAY_CC_GLM_API_KEY or RELAY_API_KEY",
    ]
    assert check["present_envs"] == ["RELAY_CC_GLM_BASE_URL", "RELAY_API_KEY"]


def test_external_bridge_normalizes_fully_qualified_requested_model():
    bridge = ExternalRouteAgentBridge(project_path="D:/agent/Route_Agent")

    payload = bridge._build_route_payload(
        RouteRequest(
            application_name="auto_research_engine",
            shared_agent_class="planner_agent",
            agent_role="planner",
            stage_name="outline_planning",
            system_prompt="You are a planner.",
            task="Plan a report.",
            requested_model="openai:gpt-4.1",
            llm_provider="openai",
            execution_context=RouteExecutionContext(),
        )
    )

    assert payload["constraints"] == {"preferred_model": "openai:gpt-4.1"}


class _FakeHTTPResponse:
    """Minimal requests.Response test double for relay wrapper tests."""

    def __init__(self, payload, status_code=200):
        """Store the payload returned by json()."""
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        """Return the canned JSON payload."""
        return self._payload


@pytest.mark.asyncio
async def test_grouped_relay_messages_wrapper_uses_messages_endpoint_and_tool_replay(monkeypatch):
    monkeypatch.setenv("RELAY_API_KEY", "shared-key")
    monkeypatch.setenv("RELAY_CC_GLM_BASE_URL", "https://relay.example/cc_glm")

    posts = []
    responses = [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "glm-5",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_echo",
                    "name": "echo",
                    "input": {"text": "hello"},
                }
            ],
            "stop_reason": "tool_use",
        },
        {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "glm-5",
            "content": [{"type": "text", "text": "The echo tool returned: hello"}],
            "stop_reason": "end_turn",
        },
    ]

    def fake_post(url, headers=None, json=None, timeout=None):
        posts.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return _FakeHTTPResponse(responses[len(posts) - 1])

    monkeypatch.setattr(llm_base.requests, "post", fake_post)

    provider = llm_base.GenericLLMProvider.from_provider(
        "relay_cc_glm",
        model="glm-5",
        max_tokens=64,
        temperature=0.1,
    )
    llm = provider.llm.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo one string.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ]
    )

    first = await llm.ainvoke(
        [
            SystemMessage(content="Be concise."),
            HumanMessage(content="Use echo on hello."),
        ]
    )

    assert posts[0]["url"] == "https://relay.example/cc_glm/messages"
    assert posts[0]["headers"]["Authorization"] == "Bearer shared-key"
    assert posts[0]["json"]["system"] == "Be concise."
    assert posts[0]["json"]["messages"] == [
        {"role": "user", "content": "Use echo on hello."},
    ]
    assert posts[0]["json"]["tools"] == [
        {
            "name": "echo",
            "description": "Echo one string.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    ]
    assert first.tool_calls == [
        {
            "id": "call_echo",
            "name": "echo",
            "args": {"text": "hello"},
            "type": "tool_call",
        }
    ]

    final = await llm.ainvoke(
        [
            SystemMessage(content="Be concise."),
            HumanMessage(content="Use echo on hello."),
            first,
            ToolMessage(content="hello", tool_call_id="call_echo"),
        ]
    )

    assert posts[1]["json"]["messages"][1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_echo",
                "name": "echo",
                "input": {"text": "hello"},
            }
        ],
    }
    assert posts[1]["json"]["messages"][2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_echo",
                "content": "hello",
            }
        ],
    }
    assert final.content == "The echo tool returned: hello"


@pytest.mark.asyncio
async def test_grouped_relay_responses_wrapper_uses_responses_endpoint_and_replays_tools(monkeypatch):
    monkeypatch.setenv("RELAY_API_KEY", "shared-key")
    monkeypatch.setenv("RELAY_CODEX_BASE_URL", "https://relay.example/codex")

    posts = []
    responses = [
        {
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_echo",
                    "name": "echo",
                    "arguments": "{\"text\":\"hello\"}",
                }
            ],
        },
        {
            "id": "resp_2",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
        },
    ]

    def fake_post(url, headers=None, json=None, timeout=None):
        posts.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return _FakeHTTPResponse(responses[len(posts) - 1])

    monkeypatch.setattr(llm_base.requests, "post", fake_post)

    provider = llm_base.GenericLLMProvider.from_provider(
        "relay_codex",
        model="gpt-5.3-codex",
        max_tokens=64,
    )
    llm = provider.llm.bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo one string.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ]
    )

    first = await llm.ainvoke(
        [
            SystemMessage(content="Be concise."),
            HumanMessage(content="Use echo on hello."),
        ]
    )

    assert posts[0]["url"] == "https://relay.example/codex/responses"
    assert posts[0]["headers"]["Authorization"] == "Bearer shared-key"
    assert posts[0]["json"]["instructions"] == "Be concise."
    assert "previous_response_id" not in posts[0]["json"]
    assert posts[0]["json"]["tools"] == [
        {
            "type": "function",
            "name": "echo",
            "description": "Echo one string.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    ]
    assert first.tool_calls == [
        {
            "id": "call_echo",
            "name": "echo",
            "args": {"text": "hello"},
            "type": "tool_call",
        }
    ]

    final = await llm.ainvoke(
        [
            SystemMessage(content="Be concise."),
            HumanMessage(content="Use echo on hello."),
            first,
            ToolMessage(content="hello", tool_call_id="call_echo"),
        ]
    )

    input_items = posts[1]["json"]["input"]
    assert "previous_response_id" not in posts[1]["json"]
    assert any(item.get("type") == "function_call" for item in input_items)
    assert any(item.get("type") == "function_call_output" for item in input_items)
    assert final.content == "hello"


@pytest.mark.asyncio
async def test_generic_provider_normalizes_structured_non_stream_response():
    class _StructuredLLM:
        async def ainvoke(self, messages, **kwargs):
            assert messages
            return type(
                "StructuredResponse",
                (),
                {
                    "content": [
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "hidden"}],
                        },
                        {"type": "output_text", "text": '{"ok": true}'},
                    ]
                },
            )()

    provider = llm_base.GenericLLMProvider(_StructuredLLM(), verbose=False)

    response = await provider.get_chat_response(
        [HumanMessage(content="Return JSON.")],
        stream=False,
    )

    assert response == '{"ok": true}'


@pytest.mark.asyncio
async def test_generic_provider_normalizes_structured_stream_chunks():
    class _Chunk:
        def __init__(self, content):
            self.content = content

    class _StructuredStreamLLM:
        async def astream(self, messages, **kwargs):
            assert messages
            yield _Chunk([{"type": "output_text", "text": "he"}])
            yield _Chunk([{"type": "reasoning", "summary": "hidden"}])
            yield _Chunk([{"type": "output_text", "text": "llo"}])

    provider = llm_base.GenericLLMProvider(_StructuredStreamLLM(), verbose=False)

    response = await provider.get_chat_response(
        [HumanMessage(content="Say hello.")],
        stream=True,
    )

    assert response == "hello"
