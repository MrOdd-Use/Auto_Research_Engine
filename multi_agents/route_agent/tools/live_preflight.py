from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv

from gpt_researcher.config.config import Config
from gpt_researcher.mcp import HAS_MCP_ADAPTERS

from ..client import RouteAgentClient
from ..utils.model_utils import normalize_app_provider


PROVIDER_ENV_MAP = {
    "openai": ["OPENAI_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google_genai": ["GOOGLE_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "ollama": ["OLLAMA_BASE_URL", "OLLAMA_HOST"],
}

PROVIDER_PACKAGE_MAP = {
    "openai": ["langchain_openai"],
    "deepseek": ["langchain_openai"],
    "anthropic": ["langchain_anthropic"],
    "google_genai": ["langchain_google_genai"],
    "ollama": ["langchain_community", "langchain_ollama"],
    "xai": ["langchain_xai"],
}


def run_live_preflight(
    *,
    env_path: str | Path = ".env",
    output_path: str | Path = "outputs/route_agent_test/live_preflight.json",
) -> Dict[str, Any]:
    load_dotenv(dotenv_path=env_path, override=False)
    cfg = Config()
    client = RouteAgentClient()

    route_agent_check = _collect_route_agent_check(client)
    llm_checks = _collect_llm_checks(client)
    retriever_checks = _collect_retriever_checks(cfg)
    embedding_check = _collect_embedding_check(cfg)
    filesystem_check = _collect_filesystem_check(output_path)

    issues = []
    for bucket in ([route_agent_check], llm_checks, retriever_checks, [embedding_check], [filesystem_check]):
        for item in bucket:
            if item.get("status") in {"warning", "error"}:
                issues.append(item)

    summary = {
        "ready": not any(item.get("status") == "error" for item in issues),
        "route_agent": route_agent_check,
        "llms": llm_checks,
        "retrievers": retriever_checks,
        "embedding": embedding_check,
        "filesystem": filesystem_check,
        "issues": issues,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _collect_route_agent_check(client: RouteAgentClient) -> Dict[str, Any]:
    if client.is_external_backend:
        bridge = getattr(client, "_external_bridge", None)
        project_path = str(getattr(bridge, "project_path", ""))
        app_env_path = str(getattr(bridge, "app_env_path", ""))
        if not bridge or not bridge.is_available():
            return {
                "backend": client.backend,
                "project_path": project_path,
                "app_env_path": app_env_path,
                "status": "error",
                "message": "external Route_Agent project is not available",
            }

        entries = client.describe_model_pool()
        if client.external_error:
            return {
                "backend": client.backend,
                "project_path": project_path,
                "app_env_path": app_env_path,
                "status": "error",
                "message": f"external Route_Agent inspection failed: {client.external_error}",
            }

        return {
            "backend": client.backend,
            "project_path": project_path,
            "app_env_path": app_env_path,
            "status": "ok" if entries else "warning",
            "message": (
                f"external Route_Agent global pool loaded with {len(entries)} available models"
                if entries
                else "external Route_Agent returned an empty available-model pool"
            ),
            "available_model_count": len(entries),
        }

    return {
        "backend": client.backend,
        "status": "ok",
        "message": "local routing backend will use ROUTE_AGENT_MODEL_POOL / ROUTE_AGENT_DEFAULT_MODEL",
        "available_model_count": len(client.model_pool),
    }


def _collect_llm_checks(client: RouteAgentClient) -> List[Dict[str, Any]]:
    entries = client.describe_model_pool()
    if client.is_external_backend:
        if client.external_error:
            return [
                {
                    "entry": "",
                    "provider": "",
                    "model": "",
                    "status": "error",
                    "message": f"failed to inspect Route_Agent global pool: {client.external_error}",
                    "source": "route_agent_global_pool",
                    "required_envs": [],
                    "present_envs": [],
                    "required_packages": [],
                    "missing_packages": [],
                }
            ]
        if not entries:
            return [
                {
                    "entry": "",
                    "provider": "",
                    "model": "",
                    "status": "warning",
                    "message": "Route_Agent global pool is empty",
                    "source": "route_agent_global_pool",
                    "required_envs": [],
                    "present_envs": [],
                    "required_packages": [],
                    "missing_packages": [],
                }
            ]
        return [_build_llm_check(entry, source="route_agent_global_pool") for entry in entries]

    if not entries:
        return [
            {
                "entry": "",
                "provider": "",
                "model": "",
                "status": "warning",
                "message": "ROUTE_AGENT_MODEL_POOL is not set; router will only use per-call requested models",
                "source": "local_config",
                "required_envs": [],
                "present_envs": [],
                "required_packages": [],
                "missing_packages": [],
            }
        ]
    return [_build_llm_check(entry, source="local_config") for entry in entries]


def _build_llm_check(entry: Dict[str, str], *, source: str) -> Dict[str, Any]:
    model_id = str(entry.get("model_id") or "").strip()
    provider = normalize_app_provider(entry.get("provider") or "")
    model = str(entry.get("model") or "").strip()
    required_envs = PROVIDER_ENV_MAP.get(provider, [])
    present_envs = [name for name in required_envs if str(os.getenv(name) or "").strip()]
    required_packages = PROVIDER_PACKAGE_MAP.get(provider, [])
    missing_packages = [name for name in required_packages if importlib.util.find_spec(name) is None]
    status = "ok"
    message = f"available via {source}"

    if not provider or not model:
        status = "error"
        message = "missing provider/model configuration"
    elif required_envs and not present_envs:
        status = "error"
        message = f"missing provider credentials: {', '.join(required_envs)}"
    elif missing_packages:
        status = "error"
        message = f"missing provider SDK packages: {', '.join(missing_packages)}"

    return {
        "entry": model_id or model,
        "provider": provider,
        "model": model,
        "status": status,
        "message": message,
        "source": source,
        "required_envs": required_envs,
        "present_envs": present_envs,
        "required_packages": required_packages,
        "missing_packages": missing_packages,
    }


def _collect_retriever_checks(cfg: Config) -> List[Dict[str, Any]]:
    checks = []
    retrievers = list(getattr(cfg, "retrievers", []) or [])
    for retriever in retrievers:
        name = str(retriever or "").strip()
        status = "ok"
        message = "configured"
        details: Dict[str, Any] = {}
        if name == "tavily":
            if not str(os.getenv("TAVILY_API_KEY") or "").strip():
                status = "error"
                message = "missing TAVILY_API_KEY"
        elif name == "mcp":
            if not HAS_MCP_ADAPTERS:
                status = "error"
                message = "langchain-mcp-adapters is not installed"
            else:
                raw_servers = str(os.getenv("MCP_SERVERS") or "").strip()
                if not raw_servers:
                    status = "warning"
                    message = "MCP retriever is enabled but no MCP_SERVERS config is present"
                details["has_mcp_adapters"] = HAS_MCP_ADAPTERS
        checks.append(
            {
                "retriever": name,
                "status": status,
                "message": message,
                **details,
            }
        )
    return checks


def _collect_embedding_check(cfg: Config) -> Dict[str, Any]:
    provider = normalize_app_provider(str(getattr(cfg, "embedding_provider", "") or "").strip())
    model = str(getattr(cfg, "embedding_model", "") or "").strip()
    payload: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "status": "ok",
        "message": "configured",
    }
    if provider == "ollama":
        base_url = str(os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip().rstrip("/")
        payload["base_url"] = base_url
        ok, message = _probe_ollama(base_url)
        payload["status"] = "ok" if ok else "warning"
        payload["message"] = message
    else:
        required_envs = PROVIDER_ENV_MAP.get(provider, [])
        present_envs = [name for name in required_envs if str(os.getenv(name) or "").strip()]
        required_packages = PROVIDER_PACKAGE_MAP.get(provider, [])
        missing_packages = [name for name in required_packages if importlib.util.find_spec(name) is None]
        payload["required_packages"] = required_packages
        payload["missing_packages"] = missing_packages
        if required_envs and not present_envs:
            payload["status"] = "warning"
            payload["message"] = f"embedding provider credentials not found: {', '.join(required_envs)}"
        elif missing_packages:
            payload["status"] = "warning"
            payload["message"] = f"embedding provider SDK packages missing: {', '.join(missing_packages)}"
    return payload


def _collect_filesystem_check(output_path: str | Path) -> Dict[str, Any]:
    target = Path(output_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        target.unlink()
        return {
            "path": str(target.parent),
            "status": "ok",
            "message": "output directory is writable",
        }
    except OSError as exc:
        return {
            "path": str(target.parent),
            "status": "error",
            "message": f"output directory is not writable: {exc}",
        }


def _probe_ollama(base_url: str) -> Tuple[bool, str]:
    try:
        with urlopen(f"{base_url}/api/tags", timeout=2.0) as response:
            if 200 <= getattr(response, "status", 0) < 300:
                return True, "Ollama endpoint is reachable"
            return False, f"Ollama probe returned status {getattr(response, 'status', 'unknown')}"
    except URLError as exc:
        return False, f"Ollama endpoint is unreachable: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama probe failed: {exc}"
