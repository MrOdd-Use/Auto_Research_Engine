"""Tests for federation adapter integration in Auto_Research_Engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_agents.route_agent.models import RouteDecision, RouteExecutionContext, RouteRequest
from multi_agents.route_agent.client import RouteAgentClient
from multi_agents.route_agent.invoker import RoutedLLMInvoker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeAnalysis:
    task_class: str = "general"
    domain: str = "general"
    domain_description: str = ""
    relevant_dimensions: tuple = ()


@dataclass(frozen=True)
class _FakeRouteResult:
    model: str | None = "gpt-4o-mini"
    lease_id: str | None = "lease-abc"
    mode: str = "local"
    analysis: _FakeAnalysis = _FakeAnalysis()
    candidates: list = ()  # type: ignore[assignment]
    local_reason: str = "test"


def _make_request(**overrides: Any) -> RouteRequest:
    defaults = {
        "application_name": "auto_research_engine",
        "agent_role": "writer",
        "stage_name": "draft",
        "task": "Write a summary",
        "requested_model": "gpt-4o-mini",
        "execution_context": RouteExecutionContext(
            workflow_id="wf-1", run_id="run-1", step_id="step-1"
        ),
    }
    defaults.update(overrides)
    return RouteRequest(**defaults)


# ---------------------------------------------------------------------------
# FederationAdapter unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_federation_adapter_route_returns_decision_and_lease():
    """FederationAdapter.route() converts RouteResult → (RouteDecision, lease_id)."""
    from multi_agents.route_agent.federation_adapter import FederationAdapter

    mock_client = AsyncMock()
    mock_client.route = AsyncMock(return_value=_FakeRouteResult())

    adapter = FederationAdapter.__new__(FederationAdapter)
    adapter._app_id = "test_app"
    adapter._client = mock_client
    adapter._started = True

    request = _make_request()
    decision, lease_id = await adapter.route(request)

    assert isinstance(decision, RouteDecision)
    assert decision.selected_model == "gpt-4o-mini"
    assert lease_id == "lease-abc"
    assert decision.resolved_shared_agent_class == "general"
    assert "federation_local" in decision.class_resolution_source


@pytest.mark.asyncio
async def test_federation_adapter_release_calls_client():
    """release() delegates to RouteClient.release()."""
    mock_client = AsyncMock()

    from multi_agents.route_agent.federation_adapter import FederationAdapter

    adapter = FederationAdapter.__new__(FederationAdapter)
    adapter._app_id = "test_app"
    adapter._client = mock_client
    adapter._started = True

    await adapter.release("lease-123")
    mock_client.release.assert_awaited_once_with("lease-123")


@pytest.mark.asyncio
async def test_federation_adapter_release_skips_empty_lease():
    """release() is a no-op when lease_id is empty."""
    mock_client = AsyncMock()

    from multi_agents.route_agent.federation_adapter import FederationAdapter

    adapter = FederationAdapter.__new__(FederationAdapter)
    adapter._app_id = "test_app"
    adapter._client = mock_client
    adapter._started = True

    await adapter.release("")
    mock_client.release.assert_not_awaited()


@pytest.mark.asyncio
async def test_federation_adapter_report_outcome_dual_write():
    """report_outcome() delegates to RouteClient for dual-write."""
    mock_client = AsyncMock()

    from multi_agents.route_agent.federation_adapter import FederationAdapter

    adapter = FederationAdapter.__new__(FederationAdapter)
    adapter._app_id = "test_app"
    adapter._client = mock_client
    adapter._started = True

    await adapter.report_outcome(
        lease_id="lease-xyz",
        model_id="gpt-4o-mini",
        agent_class="general",
        outcome_type="success",
        duration_ms=150,
    )
    mock_client.report_outcome.assert_awaited_once_with(
        lease_id="lease-xyz",
        model_id="gpt-4o-mini",
        agent_class="general",
        outcome_type="success",
        duration_ms=150,
        quality_score=None,
    )


# ---------------------------------------------------------------------------
# Client federation backend tests
# ---------------------------------------------------------------------------


def test_client_federation_backend_flag():
    """RouteAgentClient with backend='federation' sets is_federation=True."""
    with patch(
        "multi_agents.route_agent.federation_adapter.FederationAdapter"
    ) as MockAdapter:
        MockAdapter.return_value = MagicMock()
        client = RouteAgentClient(
            backend="federation",
            model_pool=["gpt-4o-mini"],
        )
        assert client.is_federation is True
        assert client.federation is not None
        assert client.backend == "federation"


def test_client_local_backend_no_federation():
    """RouteAgentClient with backend='local' has no federation."""
    client = RouteAgentClient(
        backend="local",
        model_pool=["gpt-4o-mini"],
    )
    assert client.is_federation is False
    assert client.federation is None


@pytest.mark.asyncio
async def test_client_astart_astop_delegates():
    """astart/astop delegate to FederationAdapter."""
    with patch(
        "multi_agents.route_agent.federation_adapter.FederationAdapter"
    ) as MockAdapter:
        mock_adapter = AsyncMock()
        MockAdapter.return_value = mock_adapter
        client = RouteAgentClient(
            backend="federation",
            model_pool=["gpt-4o-mini"],
        )
        await client.astart()
        mock_adapter.start.assert_awaited_once()

        await client.astop()
        mock_adapter.stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# Invoker federation dispatch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoker_federation_success_path():
    """Invoker dispatches to federation and calls release + report_outcome on success."""
    mock_federation = AsyncMock()
    mock_federation.route = AsyncMock(
        return_value=(
            RouteDecision(
                selected_model="gpt-4o-mini",
                selected_provider="openai",
                resolved_shared_agent_class="general",
                routing_reason="test",
                trace_context={},
            ),
            "lease-001",
        )
    )

    mock_client = MagicMock()
    mock_client.is_federation = True
    mock_client.federation = mock_federation

    events = []
    invoker = RoutedLLMInvoker(mock_client, event_logger=events.append)

    async def fake_provider(model: str, provider: str = "") -> str:
        return "response_text"

    request = _make_request()
    result = await invoker.invoke(
        provider_call=fake_provider,
        requested_model="gpt-4o-mini",
        llm_provider="openai",
        route_request=request,
    )

    assert result == "response_text"
    mock_federation.release.assert_awaited_once_with("lease-001")
    mock_federation.report_outcome.assert_awaited_once()

    outcome_kwargs = mock_federation.report_outcome.call_args.kwargs
    assert outcome_kwargs["lease_id"] == "lease-001"
    assert outcome_kwargs["outcome_type"] == "success"
    assert outcome_kwargs["model_id"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_invoker_federation_failure_path():
    """Invoker dispatches to federation and calls release + report failure on error."""
    mock_federation = AsyncMock()
    mock_federation.route = AsyncMock(
        return_value=(
            RouteDecision(
                selected_model="gpt-4o-mini",
                selected_provider="openai",
                resolved_shared_agent_class="general",
                routing_reason="test",
                trace_context={},
            ),
            "lease-002",
        )
    )

    mock_client = MagicMock()
    mock_client.is_federation = True
    mock_client.federation = mock_federation

    invoker = RoutedLLMInvoker(mock_client)

    async def failing_provider(model: str, provider: str = "") -> str:
        raise RuntimeError("provider down")

    request = _make_request()
    with pytest.raises(RuntimeError, match="provider down"):
        await invoker.invoke(
            provider_call=failing_provider,
            requested_model="gpt-4o-mini",
            llm_provider="openai",
            route_request=request,
        )

    mock_federation.release.assert_awaited_once_with("lease-002")
    outcome_kwargs = mock_federation.report_outcome.call_args.kwargs
    assert outcome_kwargs["outcome_type"] == "failure"


@pytest.mark.asyncio
async def test_invoker_federation_no_lease_skips_release():
    """When federation route returns no lease, release/report_outcome are skipped."""
    mock_federation = AsyncMock()
    mock_federation.route = AsyncMock(
        return_value=(
            RouteDecision(
                selected_model="gpt-4o-mini",
                selected_provider="openai",
                resolved_shared_agent_class="general",
                routing_reason="central_unavailable_fallback",
                trace_context={},
            ),
            None,  # no lease
        )
    )

    mock_client = MagicMock()
    mock_client.is_federation = True
    mock_client.federation = mock_federation

    invoker = RoutedLLMInvoker(mock_client)

    async def fake_provider(model: str, provider: str = "") -> str:
        return "ok"

    request = _make_request()
    result = await invoker.invoke(
        provider_call=fake_provider,
        requested_model="gpt-4o-mini",
        llm_provider="openai",
        route_request=request,
    )

    assert result == "ok"
    mock_federation.release.assert_not_awaited()
    mock_federation.report_outcome.assert_not_awaited()
