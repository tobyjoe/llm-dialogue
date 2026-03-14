from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


class LLMClientError(RuntimeError):
    """Raised when a model call fails."""


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatResult:
    content: str
    raw_response: dict[str, Any]


@dataclass(slots=True)
class OpenAICompatibleClient:
    model: str
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 1.0
    max_tokens: int | None = None
    timeout_seconds: int = 120
    extra_headers: dict[str, str] | None = None

    def generate(self, messages: list[ChatMessage]) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.extra_headers:
            headers.update(self.extra_headers)
        req = request.Request(endpoint, data=body, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise LLMClientError(
                f"Timeout calling model '{self.model}' via '{endpoint}' after {self.timeout_seconds}s"
            ) from exc
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(
                f"HTTP {exc.code} calling model '{self.model}': {detail}"
            ) from exc
        except error.URLError as exc:
            raise LLMClientError(
                f"Network error calling model '{self.model}': {exc.reason}"
            ) from exc

        try:
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(
                f"Unexpected response shape from model '{self.model}': {parsed}"
            ) from exc

        if not isinstance(content, str):
            raise LLMClientError(
                f"Expected string content from model '{self.model}', got: {type(content)}"
            )

        return ChatResult(content=content, raw_response=parsed)


def api_key_from_env(env_var: str) -> str:
    value = os.getenv(env_var)
    if not value:
        raise LLMClientError(
            f"Environment variable '{env_var}' is required but not set."
        )
    return value


def optional_env(env_var: str) -> str | None:
    value = os.getenv(env_var)
    if not value:
        return None
    return value
