from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only exercised on Python 3.10
    import tomli as tomllib


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_X_TITLE = "llm-dialogue-experiments"
DEFAULT_OPENING_INSTRUCTION = (
    "Begin now. Speak directly to the other model and start the interaction."
)


class ConfigError(ValueError):
    """Raised when an experiment config is invalid."""


@dataclass(slots=True)
class AgentSettings:
    name: str
    model: str
    prompt: str
    base_url: str
    api_key_env: str


@dataclass(slots=True)
class ExperimentConfig:
    path: Path
    output_dir: str
    opening_instruction: str
    turns: int
    temperature: float
    max_tokens: int | None
    verbose_mode: bool
    early_stop: bool
    http_referer: str | None
    x_title: str | None
    agent_a: AgentSettings
    agent_b: AgentSettings


def load_experiment(path: Path) -> ExperimentConfig:
    resolved_path = path.resolve()
    raw = tomllib.loads(resolved_path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ConfigError("Experiment config must be a TOML table.")

    agent_a = _parse_agent_settings(raw, "agent_a", default_name="Agent A")
    agent_b = _parse_agent_settings(raw, "agent_b", default_name="Agent B")
    openrouter = _optional_table(raw, "openrouter")

    turns = _required_positive_int(raw, "turns")
    max_tokens = _optional_int(raw, "max_tokens")

    return ExperimentConfig(
        path=resolved_path,
        output_dir=_optional_str(raw, "output_dir") or "transcripts",
        opening_instruction=_optional_str(raw, "opening_instruction")
        or DEFAULT_OPENING_INSTRUCTION,
        turns=turns,
        temperature=_optional_float(raw, "temperature") or 1.0,
        max_tokens=max_tokens,
        verbose_mode=_optional_bool(raw, "verbose_mode", default=False),
        early_stop=_optional_bool(raw, "early_stop", default=True),
        http_referer=_optional_str(openrouter, "http_referer"),
        x_title=_optional_str(openrouter, "x_title") or DEFAULT_X_TITLE,
        agent_a=agent_a,
        agent_b=agent_b,
    )


def _parse_agent_settings(
    raw: dict[str, Any], key: str, default_name: str
) -> AgentSettings:
    table = _required_table(raw, key)
    return AgentSettings(
        name=_optional_str(table, "name") or default_name,
        model=_required_str(table, "model"),
        prompt=_required_str(table, "prompt"),
        base_url=_optional_str(table, "base_url") or DEFAULT_BASE_URL,
        api_key_env=_optional_str(table, "api_key_env") or DEFAULT_API_KEY_ENV,
    )


def _required_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid table: {key}")
    return value


def _optional_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Invalid table: {key}")
    return value


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing or invalid string value: {key}")
    return value


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Invalid string value: {key}")
    return value


def _required_positive_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"Missing or invalid positive integer value: {key}")
    return value


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ConfigError(f"Invalid integer value: {key}")
    return value


def _optional_float(raw: dict[str, Any], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ConfigError(f"Invalid numeric value: {key}")
    return float(value)


def _optional_bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"Invalid boolean value: {key}")
    return value
