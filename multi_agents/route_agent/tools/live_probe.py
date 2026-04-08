"""Live model probing capability for ExternalRouteAgentBridge.

Extracted from external_bridge.py to keep that module under the 800-line limit.
The canonical _provider_env_groups and _provider_package_names helpers are
defined here and imported by live_preflight.py to avoid duplication.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import threading
import time
from typing import Any, Dict, List, Sequence

from gpt_researcher.llm_provider.generic.relay import _relay_group_suffix


_LIVE_PROBE_PROMPT = (
    {"role": "system", "content": "Reply with OK."},
    {"role": "user", "content": "OK"},
)
_LIVE_PROBE_OK_TTL_S = 900.0
_LIVE_PROBE_FAIL_TTL_S = 300.0
_LIVE_PROBE_DEFAULT_CONCURRENCY = 4
_LIVE_PROBE_DEFAULT_TIMEOUT_S = 3.0

# Canonical env-var map shared with live_preflight.py
_PROVIDER_ENV_MAP: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY",),
    "google_vertexai": ("GOOGLE_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "ollama": ("OLLAMA_BASE_URL", "OLLAMA_HOST"),
    "relay": ("RELAY_API_KEY", "RELAY_BASE_URL"),
}

# Canonical package map shared with live_preflight.py
_PROVIDER_PACKAGE_MAP: dict[str, tuple[str, ...]] = {
    "openai": ("langchain_openai",),
    "deepseek": ("langchain_openai",),
    "anthropic": ("langchain_anthropic",),
    "google_genai": ("langchain_google_genai",),
    "google_vertexai": ("langchain_google_vertexai",),
    "xai": ("langchain_xai",),
    "ollama": ("langchain_community", "langchain_ollama"),
    "relay": ("langchain_openai",),
}

# Relay-unavailable error string (hard-failure pattern for Chinese relay services)
_NO_CHANNEL_MSG = "No available channel supports this model"


def _provider_env_groups(provider: str) -> list[list[str]]:
    """Return env-var groups required for one provider (canonical, shared)."""
    if provider == "ollama":
        return [["OLLAMA_BASE_URL", "OLLAMA_HOST"]]
    if provider == "relay" or provider.startswith("relay_"):
        suffix = _relay_group_suffix(provider)
        api_names: list[str] = [f"RELAY_{suffix}_API_KEY"] if suffix else []
        base_names: list[str] = [f"RELAY_{suffix}_BASE_URL"] if suffix else []
        api_names.append("RELAY_API_KEY")
        base_names.append("RELAY_BASE_URL")
        return [base_names, api_names]
    return [[name] for name in _PROVIDER_ENV_MAP.get(provider, ())]


def _provider_package_names(provider: str) -> list[str]:
    """Return SDK package names required for one provider (canonical, shared)."""
    if provider == "relay":
        return list(_PROVIDER_PACKAGE_MAP["relay"])
    if provider.startswith("relay_"):
        from gpt_researcher.llm_provider.generic.relay import _relay_endpoint_mode

        if _relay_endpoint_mode(provider) == "chat_completions":
            return list(_PROVIDER_PACKAGE_MAP["relay"])
        return []
    return list(_PROVIDER_PACKAGE_MAP.get(provider, ()))


class LiveProbeMixin:
    """Mixin providing live model probing for ExternalRouteAgentBridge."""

    def _init_probe_cache(self) -> None:
        """Initialize probe cache state — call from the host class __init__."""
        self._probe_cache: Dict[str, Dict[str, Any]] = {}
        self._probe_cache_lock = threading.Lock()

    def probe_global_pool(
        self,
        *,
        force: bool = False,
        limit: int | None = None,
        mark_unavailable: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run a lightweight live probe across the global pool."""
        entries = self.describe_global_pool()  # type: ignore[attr-defined]
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        if not entries:
            return []

        router_storage = None
        if mark_unavailable:
            _, engine = self._build_route_agent_runtime_support()  # type: ignore[attr-defined]
            router_storage = getattr(engine, "_router_storage", None)

        return self._run_async(
            self._probe_entries_async(entries, force=force, router_storage=router_storage)
        )

    async def probe_global_pool_async(
        self,
        *,
        force: bool = False,
        limit: int | None = None,
        mark_unavailable: bool = True,
    ) -> List[Dict[str, Any]]:
        """Async version of probe_global_pool — no thread wrapping needed."""
        entries = self.describe_global_pool()  # type: ignore[attr-defined]
        if limit is not None:
            entries = entries[: max(0, int(limit))]
        if not entries:
            return []

        router_storage = None
        if mark_unavailable:
            _, engine = self._build_route_agent_runtime_support()  # type: ignore[attr-defined]
            router_storage = getattr(engine, "_router_storage", None)

        return await self._probe_entries_async(entries, force=force, router_storage=router_storage)

    async def _probe_entries_async(
        self,
        entries: Sequence[Dict[str, str]],
        *,
        force: bool,
        router_storage: Any | None,
    ) -> List[Dict[str, Any]]:
        """Probe several model entries concurrently with a small concurrency cap."""
        if not entries:
            return []

        semaphore = asyncio.Semaphore(self._probe_concurrency())

        async def _guarded(entry: Dict[str, str]) -> Dict[str, Any]:
            async with semaphore:
                return await self._probe_entry_async(
                    entry,
                    force=force,
                    router_storage=router_storage,
                )

        return list(await asyncio.gather(*[_guarded(entry) for entry in entries]))

    async def _probe_entry_async(
        self,
        entry: Dict[str, str],
        *,
        force: bool,
        router_storage: Any | None,
    ) -> Dict[str, Any]:
        """Probe one candidate model with cache, env/package checks, and live call."""
        model_id = str(entry.get("model_id") or "").strip()
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        cached = self._read_cached_probe(model_id, force=force)
        if cached is not None:
            return cached

        env_error = self._missing_env_error(provider)
        if env_error is not None:
            result = self._new_probe_result(
                model_id=model_id,
                provider=provider,
                model=model,
                ok=False,
                skipped=False,
                hard_failure=True,
                cached=False,
                message=env_error,
            )
            self._write_cached_probe(model_id, result)
            if router_storage is not None:
                await router_storage.mark_unable_async(model_id)
            return result

        package_error = self._missing_package_error(provider)
        if package_error is not None:
            result = self._new_probe_result(
                model_id=model_id,
                provider=provider,
                model=model,
                ok=False,
                skipped=False,
                hard_failure=True,
                cached=False,
                message=package_error,
            )
            self._write_cached_probe(model_id, result)
            if router_storage is not None:
                await router_storage.mark_unable_async(model_id)
            return result

        timeout_s = self._probe_timeout_s()
        try:
            await asyncio.wait_for(self._live_probe_model_async(provider, model), timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001
            hard_failure, skipped = self._classify_probe_error(exc)
            result = self._new_probe_result(
                model_id=model_id,
                provider=provider,
                model=model,
                ok=False,
                skipped=skipped,
                hard_failure=hard_failure,
                cached=False,
                message=self._probe_error_message(exc, timeout_s=timeout_s),
            )
            self._write_cached_probe(model_id, result)
            if router_storage is not None and hard_failure and not skipped:
                await router_storage.mark_unable_async(model_id)
            return result

        result = self._new_probe_result(
            model_id=model_id,
            provider=provider,
            model=model,
            ok=True,
            skipped=False,
            hard_failure=False,
            cached=False,
            message="live probe ok",
        )
        self._write_cached_probe(model_id, result)
        return result

    async def _live_probe_model_async(self, provider: str, model: str) -> str:
        """Execute one minimal completion against the actual provider stack."""
        from gpt_researcher.llm_provider.generic.base import GenericLLMProvider

        last_error: Exception | None = None
        for kwargs in ({"model": model, "max_tokens": 1}, {"model": model}):
            try:
                llm = GenericLLMProvider.from_provider(
                    provider,
                    verbose=False,
                    **kwargs,
                )
                return str(
                    await llm.get_chat_response(
                        list(_LIVE_PROBE_PROMPT),
                        stream=False,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is None:
            raise RuntimeError("Live probe exhausted all attempts without capturing an exception")
        raise last_error

    def _classify_probe_error(self, exc: Exception) -> tuple[bool, bool]:
        """Classify a live-probe error into (hard_failure, skipped)."""
        if self._is_timeout_probe_error(exc):
            return False, True
        message = str(exc).lower()
        if "unsupported" in message and "provider" in message:
            return False, True
        hard_patterns = (
            _NO_CHANNEL_MSG,
            "no available channel",
            "unsupported model",
            "model not found",
            "does not exist",
            "invalid_request_error",
            "authentication",
            "unauthorized",
            "api key",
            "permission denied",
        )
        if any(pattern in message for pattern in hard_patterns):
            return True, False
        return False, False

    def _is_timeout_probe_error(self, exc: Exception) -> bool:
        """Return whether the probe error indicates the probe timed out."""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True
        message = str(exc).lower()
        return "timed out" in message or "timeout" in message

    def _probe_error_message(self, exc: Exception, *, timeout_s: float) -> str:
        """Return a stable, non-empty message for one probe failure."""
        message = str(exc).strip()
        if message:
            return message
        if self._is_timeout_probe_error(exc):
            return f"live probe timed out after {timeout_s:g}s"
        return exc.__class__.__name__

    def _missing_env_error(self, provider: str) -> str | None:
        """Return a message when the provider's required env vars are absent."""
        required_groups = _provider_env_groups(provider)
        if not required_groups:
            return None
        if all(any(str(os.getenv(name) or "").strip() for name in group) for group in required_groups):
            return None
        required = [(" or ".join(group)) for group in required_groups]
        return f"missing provider credentials for {provider}: {', '.join(required)}"

    def _missing_package_error(self, provider: str) -> str | None:
        """Return a message when the provider's SDK packages are missing."""
        required = _provider_package_names(provider)
        if not required:
            return None
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if not missing:
            return None
        return f"missing provider SDK packages for {provider}: {', '.join(missing)}"

    def _probe_concurrency(self) -> int:
        """Return the concurrency used by startup live probes."""
        raw = str(os.getenv("ROUTE_AGENT_LIVE_PROBE_CONCURRENCY") or "").strip()
        if raw.isdigit():
            return max(1, int(raw))
        return _LIVE_PROBE_DEFAULT_CONCURRENCY

    def _probe_timeout_s(self) -> float:
        """Return the timeout used by startup live probes."""
        raw = str(os.getenv("ROUTE_AGENT_LIVE_PROBE_TIMEOUT_S") or "").strip()
        if not raw:
            return _LIVE_PROBE_DEFAULT_TIMEOUT_S
        try:
            value = float(raw)
        except ValueError:
            return _LIVE_PROBE_DEFAULT_TIMEOUT_S
        if value <= 0:
            return _LIVE_PROBE_DEFAULT_TIMEOUT_S
        return value

    def _new_probe_result(
        self,
        *,
        model_id: str,
        provider: str,
        model: str,
        ok: bool,
        skipped: bool,
        hard_failure: bool,
        cached: bool,
        message: str,
    ) -> Dict[str, Any]:
        """Build one cached probe-result payload."""
        return {
            "model_id": model_id,
            "provider": provider,
            "model": model,
            "ok": ok,
            "skipped": skipped,
            "hard_failure": hard_failure,
            "cached": cached,
            "message": message,
            "checked_at": time.monotonic(),
        }

    def _read_cached_probe(self, model_id: str, *, force: bool) -> Dict[str, Any] | None:
        """Read one cached probe result when it is still fresh."""
        if force or not model_id:
            return None
        with self._probe_cache_lock:
            cached = self._probe_cache.get(model_id)
        if cached is None:
            return None

        checked_at = float(cached.get("checked_at") or 0.0)
        ttl = _LIVE_PROBE_OK_TTL_S if cached.get("ok") else _LIVE_PROBE_FAIL_TTL_S
        if time.monotonic() - checked_at > ttl:
            return None

        return {
            **cached,
            "cached": True,
        }

    def _write_cached_probe(self, model_id: str, result: Dict[str, Any]) -> None:
        """Write one fresh probe result into the in-memory cache."""
        with self._probe_cache_lock:
            self._probe_cache[model_id] = dict(result)

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine from sync code, even inside a live event loop."""
        from concurrent.futures import ThreadPoolExecutor

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)
