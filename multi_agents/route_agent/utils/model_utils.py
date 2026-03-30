from __future__ import annotations

from typing import Tuple


_APP_PROVIDER_ALIASES = {
    "gemini": "google_genai",
    "google": "google_genai",
}

_ROUTE_AGENT_PROVIDER_ALIASES = {
    "google_genai": "google",
    "gemini": "google",
}


def normalize_app_provider(provider: str) -> str:
    raw = str(provider or "").strip().lower()
    if not raw:
        return ""
    return _APP_PROVIDER_ALIASES.get(raw, raw)


def normalize_route_agent_provider(provider: str) -> str:
    raw = str(provider or "").strip().lower()
    if not raw:
        return ""
    return _ROUTE_AGENT_PROVIDER_ALIASES.get(raw, raw)


def build_model_identifier(provider: str, model: str, *, target: str = "app") -> str:
    model_name = str(model or "").strip()
    if not model_name:
        return ""

    if target == "route_agent":
        provider_name = normalize_route_agent_provider(provider)
    else:
        provider_name = normalize_app_provider(provider)

    if not provider_name:
        return model_name
    return f"{provider_name}:{model_name}"


def split_model_identifier(model_id: str, *, fallback_provider: str = "") -> Tuple[str, str, str]:
    raw = str(model_id or "").strip()
    fallback = normalize_app_provider(fallback_provider)
    if not raw:
        return "", fallback, ""

    if ":" in raw:
        provider, model = raw.split(":", 1)
        normalized_provider = normalize_app_provider(provider)
        model_name = model.strip()
        return build_model_identifier(normalized_provider, model_name), normalized_provider, model_name

    return build_model_identifier(fallback, raw), fallback, raw
