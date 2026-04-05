"""Relay endpoint helpers and _RelayEndpointChatModel for non-chat relay transports.

Extracted from base.py so that base.py stays under the 800-line limit.
The canonical _relay_group_suffix and related helpers live here and are
imported by live_preflight.py and external_bridge.py to avoid duplication.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


_SUPPORTED_PROVIDERS = {
    "openai",
    "anthropic",
    "azure_openai",
    "cohere",
    "google_vertexai",
    "google_genai",
    "fireworks",
    "ollama",
    "together",
    "mistralai",
    "huggingface",
    "groq",
    "bedrock",
    "dashscope",
    "xai",
    "deepseek",
    "litellm",
    "gigachat",
    "openrouter",
    "vllm_openai",
    "aimlapi",
    "netmind",
    "relay",
}


def normalize_response_text(value: Any) -> str:
    """Flatten one structured model response payload into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [normalize_response_text(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").strip().lower()
        if block_type in {
            "tool_use",
            "tool_result",
            "function_call",
            "server_tool_call",
            "computer_call",
            "reasoning",
            "reasoning_content",
            "thinking",
        }:
            return ""
        for key in ("text", "output_text", "content", "summary", "refusal"):
            if value.get(key) is not None:
                return normalize_response_text(value.get(key))
        if block_type:
            return ""
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def is_supported_provider(provider: str) -> bool:
    """Return whether one provider name is supported, including grouped relays."""
    normalized = str(provider or "").strip().lower()
    return normalized in _SUPPORTED_PROVIDERS or normalized.startswith("relay_")


def _is_relay_provider(provider: str) -> bool:
    """Return whether the provider uses the OpenAI-compatible relay transport."""
    normalized = str(provider or "").strip().lower()
    return normalized == "relay" or normalized.startswith("relay_")


def _relay_group_suffix(provider: str) -> str:
    """Return the normalized env-var suffix for one grouped relay provider.

    This is the canonical implementation imported by live_preflight.py and
    live_probe.py to avoid triplication.
    """
    normalized = str(provider or "").strip().lower()
    if not normalized.startswith("relay_"):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", normalized[len("relay_"):]).strip("_").upper()


def _relay_env_candidates(provider: str, suffix: str) -> list[str]:
    """Return env-var candidates for one relay config field."""
    group_suffix = _relay_group_suffix(provider)
    candidates: list[str] = []
    if group_suffix:
        candidates.append(f"RELAY_{group_suffix}_{suffix}")
    candidates.append(f"RELAY_{suffix}")
    return candidates


def _resolve_relay_env(provider: str, suffix: str) -> str:
    """Resolve one relay env value, preferring group-specific vars."""
    for name in _relay_env_candidates(provider, suffix):
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    names = ", ".join(_relay_env_candidates(provider, suffix))
    raise KeyError(f"Missing relay configuration for {provider}: {names}")


def _try_resolve_relay_env(provider: str, suffix: str) -> str:
    """Resolve one relay env value when present, otherwise return an empty string."""
    for name in _relay_env_candidates(provider, suffix):
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _relay_endpoint_mode(provider: str) -> str:
    """Return the transport mode used by one relay provider."""
    override = (
        _try_resolve_relay_env(provider, "ENDPOINT_MODE")
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )
    aliases = {
        "chat": "chat_completions",
        "chat_completion": "chat_completions",
        "chat_completions": "chat_completions",
        "messages": "messages",
        "responses": "responses",
    }
    if override:
        if override in aliases:
            return aliases[override]
        raise ValueError(
            f"Unsupported relay endpoint mode for {provider}: {override}. "
            "Expected chat_completions, messages, or responses."
        )

    normalized = str(provider or "").strip().lower()
    if normalized in {"relay_cc_glm", "relay_cc_kimi25", "relay_cc_minimax21"}:
        return "messages"
    if normalized == "relay_codex":
        return "responses"
    return "chat_completions"


class _RelayEndpointChatModel:
    """Minimal async LangChain-compatible wrapper for non-chat relay endpoints."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        endpoint_mode: str,
        default_options: dict[str, Any] | None = None,
        bound_tools: list[Any] | None = None,
    ) -> None:
        """Initialize one relay endpoint wrapper."""
        if not str(model or "").strip():
            raise ValueError(f"Missing model name for relay provider {provider}")
        self.provider = str(provider or "").strip()
        self.model = str(model or "").strip()
        self.model_name = self.model
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.endpoint_mode = str(endpoint_mode or "").strip().lower()
        self.default_options = dict(default_options or {})
        self.bound_tools = list(bound_tools or [])

    def bind_tools(self, tools: list[Any], **_: Any) -> "_RelayEndpointChatModel":
        """Return a copy of the wrapper with tool schemas attached."""
        return _RelayEndpointChatModel(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            endpoint_mode=self.endpoint_mode,
            default_options=self.default_options,
            bound_tools=list(tools or []),
        )

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        """Invoke the relay endpoint and return an AIMessage."""
        options = self._merged_options(kwargs)
        if self.endpoint_mode == "messages":
            payload = self._build_messages_payload(messages, options)
            response = await self._post_json("messages", payload, options)
            return self._parse_messages_response(response)
        if self.endpoint_mode == "responses":
            payload = self._build_responses_payload(messages, options)
            response = await self._post_json("responses", payload, options)
            return self._parse_responses_response(response)
        raise ValueError(f"Unsupported relay endpoint mode: {self.endpoint_mode}")

    async def astream(self, messages: list[Any], **kwargs: Any):
        """Yield a single final chunk for streaming-compatible callers."""
        from langchain_core.messages import AIMessageChunk

        response = await self.ainvoke(messages, **kwargs)
        yield AIMessageChunk(
            content=response.content or "",
            additional_kwargs=dict(getattr(response, "additional_kwargs", {}) or {}),
            response_metadata=dict(getattr(response, "response_metadata", {}) or {}),
            tool_calls=list(getattr(response, "tool_calls", []) or []),
            id=getattr(response, "id", None),
            chunk_position="last",
        )

    def _merged_options(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge invocation overrides with the wrapper defaults."""
        merged = dict(self.default_options)
        merged.update(dict(overrides or {}))
        for key in ("config", "callbacks", "metadata", "run_name", "tags", "stop"):
            merged.pop(key, None)
        return merged

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """POST JSON to one relay endpoint using an async HTTP client."""
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = float(options.get("request_timeout") or options.get("timeout") or 180)
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=timeout)
        try:
            data = response.json()
        except ValueError:
            data = None
        if response.status_code >= 400:
            raise RuntimeError(self._format_error(response.status_code, data, response.text))
        if not isinstance(data, dict):
            raise RuntimeError(
                f"{self.provider} {self.endpoint_mode} endpoint returned a non-JSON response"
            )
        error_payload = data.get("error")
        if error_payload:
            raise RuntimeError(self._format_error(response.status_code, data, response.text))
        return data

    def _format_error(self, status_code: int, data: Any, body: str) -> str:
        """Build one concise relay HTTP error message."""
        if isinstance(data, dict):
            error_payload = data.get("error")
            if isinstance(error_payload, dict):
                error_type = str(error_payload.get("type") or "error").strip()
                error_message = str(error_payload.get("message") or body or "").strip()
                return (
                    f"{self.provider} {self.endpoint_mode} endpoint HTTP {status_code}: "
                    f"{error_type}: {error_message}"
                )
            if error_payload:
                return (
                    f"{self.provider} {self.endpoint_mode} endpoint HTTP {status_code}: "
                    f"{error_payload}"
                )
        snippet = str(body or "").strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = f"{snippet[:397]}..."
        return f"{self.provider} {self.endpoint_mode} endpoint HTTP {status_code}: {snippet}"

    def _build_messages_payload(
        self,
        messages: list[Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one Anthropic-style messages payload."""
        system_parts: list[str] = []
        payload_messages: list[dict[str, Any]] = []
        for message in list(messages or []):
            role = self._message_role(message)
            if role == "system":
                text = self._stringify_content(self._message_content(message))
                if text:
                    system_parts.append(text)
                continue
            if role == "assistant":
                payload_messages.append(
                    {
                        "role": "assistant",
                        "content": self._assistant_messages_content(message),
                    }
                )
                continue
            if role == "tool":
                payload_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": self._tool_call_id(message),
                                "content": self._stringify_content(self._message_content(message)),
                            }
                        ],
                    }
                )
                continue
            payload_messages.append(
                {
                    "role": "user",
                    "content": self._stringify_content(self._message_content(message)),
                }
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": int(options.get("max_tokens") or options.get("max_output_tokens") or 1024),
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if options.get("temperature") is not None:
            payload["temperature"] = options["temperature"]
        tools = self._messages_tools()
        if tools:
            payload["tools"] = tools
        return payload

    def _build_responses_payload(
        self,
        messages: list[Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one OpenAI Responses-style payload."""
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []
        for message in list(messages or []):
            role = self._message_role(message)
            if role == "system":
                text = self._stringify_content(self._message_content(message))
                if text:
                    instructions.append(text)
                continue
            if role == "assistant":
                input_items.extend(self._assistant_responses_input(message))
                continue
            if role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": self._tool_call_id(message),
                        "output": self._stringify_content(self._message_content(message)),
                    }
                )
                continue
            input_items.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._stringify_content(self._message_content(message)),
                        }
                    ],
                }
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": int(
                options.get("max_output_tokens") or options.get("max_tokens") or 1024
            ),
        }
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)
        if options.get("temperature") is not None:
            payload["temperature"] = options["temperature"]
        tools = self._responses_tools()
        if tools:
            payload["tools"] = tools
        return payload

    def _messages_tools(self) -> list[dict[str, Any]]:
        """Convert bound tools to Anthropic-compatible schemas."""
        if not self.bound_tools:
            return []
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tools: list[dict[str, Any]] = []
        for tool in self.bound_tools:
            converted = convert_to_openai_tool(tool)
            function_payload = dict(converted.get("function") or converted)
            tools.append(
                {
                    "name": str(function_payload.get("name") or ""),
                    "description": str(function_payload.get("description") or ""),
                    "input_schema": dict(
                        function_payload.get("parameters")
                        or {"type": "object", "properties": {}}
                    ),
                }
            )
        return tools

    def _responses_tools(self) -> list[dict[str, Any]]:
        """Convert bound tools to OpenAI Responses-compatible schemas."""
        if not self.bound_tools:
            return []
        from langchain_core.utils.function_calling import convert_to_openai_tool

        tools: list[dict[str, Any]] = []
        for tool in self.bound_tools:
            converted = convert_to_openai_tool(tool)
            function_payload = dict(converted.get("function") or converted)
            tools.append(
                {
                    "type": "function",
                    "name": str(function_payload.get("name") or ""),
                    "description": str(function_payload.get("description") or ""),
                    "parameters": dict(
                        function_payload.get("parameters")
                        or {"type": "object", "properties": {}}
                    ),
                }
            )
        return tools

    def _assistant_messages_content(self, message: Any) -> Any:
        """Convert one assistant message into Anthropic content blocks."""
        raw_content = self._message_additional_kwargs(message).get("relay_content")
        if isinstance(raw_content, list):
            return raw_content

        blocks: list[dict[str, Any]] = []
        text = self._stringify_content(self._message_content(message))
        if text:
            blocks.append({"type": "text", "text": text})
        for index, tool_call in enumerate(self._message_tool_calls(message), start=1):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or f"tool_call_{index}"),
                    "name": str(tool_call.get("name") or ""),
                    "input": self._coerce_tool_args(tool_call.get("args")),
                }
            )
        if not blocks:
            return ""
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            return blocks[0]["text"]
        return blocks

    def _assistant_responses_input(self, message: Any) -> list[dict[str, Any]]:
        """Convert one assistant message into Responses API input items."""
        raw_output = self._message_additional_kwargs(message).get("relay_response_output")
        if isinstance(raw_output, list):
            return list(raw_output)

        items: list[dict[str, Any]] = []
        for index, tool_call in enumerate(self._message_tool_calls(message), start=1):
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(tool_call.get("id") or f"call_{index}"),
                    "name": str(tool_call.get("name") or ""),
                    "arguments": json.dumps(
                        self._coerce_tool_args(tool_call.get("args")),
                        ensure_ascii=False,
                    ),
                }
            )

        text = self._stringify_content(self._message_content(message))
        if text:
            items.append(
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            )
        return items

    def _parse_messages_response(self, payload: dict[str, Any]) -> Any:
        """Parse one messages endpoint response into an AIMessage."""
        from langchain_core.messages import AIMessage

        content_blocks = list(payload.get("content") or [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            block_type = str(block.get("type") or "").strip().lower()
            if block_type == "text" and block.get("text") is not None:
                text_parts.append(str(block.get("text") or ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                        "args": self._coerce_tool_args(block.get("input")),
                        "type": "tool_call",
                    }
                )

        return AIMessage(
            id=payload.get("id"),
            content="".join(text_parts),
            tool_calls=tool_calls,
            additional_kwargs={
                "relay_content": content_blocks,
                "relay_response": payload,
            },
            response_metadata={
                "model": payload.get("model"),
                "stop_reason": payload.get("stop_reason"),
                "usage": payload.get("usage"),
            },
        )

    def _parse_responses_response(self, payload: dict[str, Any]) -> Any:
        """Parse one responses endpoint payload into an AIMessage."""
        from langchain_core.messages import AIMessage

        output_items = list(payload.get("output") or [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in output_items:
            item_type = str(item.get("type") or "").strip().lower()
            if item_type == "message":
                for block in list(item.get("content") or []):
                    block_type = str(block.get("type") or "").strip().lower()
                    if block_type in {"output_text", "text"} and block.get("text") is not None:
                        text_parts.append(str(block.get("text") or ""))
            elif item_type == "function_call":
                tool_calls.append(
                    {
                        "id": str(item.get("call_id") or item.get("id") or ""),
                        "name": str(item.get("name") or ""),
                        "args": self._coerce_tool_args(item.get("arguments")),
                        "type": "tool_call",
                    }
                )

        return AIMessage(
            id=payload.get("id"),
            content="".join(text_parts),
            tool_calls=tool_calls,
            additional_kwargs={
                "relay_response_output": output_items,
                "relay_response": payload,
            },
            response_metadata={
                "status": payload.get("status"),
                "usage": payload.get("usage"),
                "model": payload.get("model"),
            },
        )

    def _message_role(self, message: Any) -> str:
        """Normalize a message object or dict into one of the core roles."""
        if isinstance(message, dict):
            role = str(message.get("role") or message.get("type") or "").strip().lower()
        else:
            role = str(getattr(message, "type", "") or getattr(message, "role", "")).strip().lower()
        aliases = {
            "human": "user",
            "ai": "assistant",
        }
        return aliases.get(role, role)

    def _message_content(self, message: Any) -> Any:
        """Return the content payload for one message-like object."""
        if isinstance(message, dict):
            return message.get("content")
        return getattr(message, "content", None)

    def _message_additional_kwargs(self, message: Any) -> dict[str, Any]:
        """Return additional kwargs carried by one message-like object."""
        if isinstance(message, dict):
            return dict(message.get("additional_kwargs") or {})
        return dict(getattr(message, "additional_kwargs", {}) or {})

    def _message_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        """Return tool calls stored on one message-like object."""
        if isinstance(message, dict):
            return list(message.get("tool_calls") or [])
        return list(getattr(message, "tool_calls", []) or [])

    def _tool_call_id(self, message: Any) -> str:
        """Return the tool call id associated with one tool result message."""
        if isinstance(message, dict):
            return str(
                message.get("tool_call_id")
                or self._message_additional_kwargs(message).get("tool_call_id")
                or ""
            )
        return str(
            getattr(message, "tool_call_id", "")
            or self._message_additional_kwargs(message).get("tool_call_id")
            or ""
        )

    def _coerce_tool_args(self, value: Any) -> dict[str, Any]:
        """Normalize raw tool-call arguments into a dict."""
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return {}
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"input": raw}
            if isinstance(parsed, dict):
                return parsed
            return {"input": parsed}
        if value is None:
            return {}
        return {"input": value}

    def _stringify_content(self, value: Any) -> str:
        """Flatten one message content payload into plain text."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [self._stringify_content(item) for item in value]
            return "".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in ("text", "content", "output_text", "thinking"):
                if value.get(key) is not None:
                    return self._stringify_content(value.get(key))
            return json.dumps(value, ensure_ascii=False)
        return str(value)
