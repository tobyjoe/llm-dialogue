from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from .client import (
    LLMClientError,
    OpenAICompatibleClient,
    api_key_from_env,
)
from .config import ConfigError, ExperimentConfig, load_experiment
from .conversation import AgentConfig, ConversationResult, TranscriptTurn, run_conversation, utc_now
from .markdown import timestamp_slug, write_transcript


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tracked experiment file describing a turn-based direct conversation between two LLM agents."
        )
    )
    parser.add_argument(
        "experiment_file",
        help="Path to the experiment TOML file to run.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional override for the transcript/log output directory. Defaults to the experiment file's output_dir.",
    )
    return parser


def build_openrouter_headers(
    http_referer: str | None, x_title: str | None, base_urls: list[str]
) -> dict[str, str] | None:
    if not any("openrouter.ai" in url for url in base_urls):
        return None

    headers: dict[str, str] = {}
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title
    return headers or None


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or key in os.environ:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        os.environ[key] = value


def find_dotenv(start_dir: Path) -> Path | None:
    for directory in [start_dir, *start_dir.parents]:
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def resolve_output_dir(output_dir: str, anchor_dir: Path) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return (anchor_dir / path).resolve()


def render_progress(
    message_number: int,
    total_messages: int,
    speaker_turn_number: int,
    turns_per_agent: int,
    speaker: str,
    model: str,
    width: int = 24,
) -> str:
    complete = int(width * message_number / total_messages)
    bar = "#" * complete + "-" * (width - complete)
    return (
        f"[{bar}] message {message_number}/{total_messages} "
        f"{speaker} turn {speaker_turn_number}/{turns_per_agent} ({model})"
    )


def emit_progress(turn: TranscriptTurn, total_turns: int) -> None:
    message = render_progress(
        message_number=turn.message_number,
        total_messages=total_turns,
        speaker_turn_number=turn.speaker_turn_number,
        turns_per_agent=total_turns // 2,
        speaker=turn.speaker,
        model=turn.model,
    )
    if sys.stderr.isatty():
        end = "\n" if turn.message_number == total_turns else ""
        print(f"\r{message}", file=sys.stderr, end=end, flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


class TranscriptRecorder:
    def __init__(self, initial_result: ConversationResult, output_dir: Path) -> None:
        self.current_result = initial_result
        self.path = write_transcript(initial_result, output_dir)

    def record_turn(self, turn: TranscriptTurn, total_turns: int) -> None:
        del total_turns
        self.current_result.turns.append(turn)
        write_transcript(self.current_result, self.path.parent)

    def finalize(self, result: ConversationResult) -> None:
        self.current_result = result
        write_transcript(result, self.path.parent)

    def fail(self, error_message: str) -> None:
        self.current_result = replace(
            self.current_result,
            finished_at_utc=utc_now(),
            status="failed",
            error_message=error_message,
        )
        write_transcript(self.current_result, self.path.parent)


class RunLogger:
    def __init__(self, output_dir: Path, started_at_utc: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / f"conversation_{timestamp_slug(started_at_utc)}.log"
        self._append(f"run started at {started_at_utc}")

    def _append(self, message: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now()}] {message}\n")

    def log_configuration(
        self,
        experiment: ExperimentConfig,
        model_a: str,
        model_b: str,
        output_dir: Path,
    ) -> None:
        self._append(
            "configuration "
            f"experiment={experiment.path} "
            f"model_a={model_a} model_b={model_b} "
            f"turns_per_agent={experiment.turns} total_messages={experiment.turns * 2} "
            f"verbose_mode={experiment.verbose_mode} allow_early_stop={experiment.early_stop} "
            f"output_dir={output_dir}"
        )

    def log_turn_start(
        self, message_number: int, speaker_turn_number: int, speaker: str, model: str
    ) -> None:
        self._append(
            f"message {message_number} starting speaker={speaker} "
            f"speaker_turn={speaker_turn_number} model={model}"
        )

    def log_turn_complete(self, turn: TranscriptTurn, total_turns: int) -> None:
        self._append(
            f"message {turn.message_number}/{total_turns} completed "
            f"speaker={turn.speaker} speaker_turn={turn.speaker_turn_number} "
            f"model={turn.model} "
            f"chars={len(turn.content)}"
        )

    def log_success(self, transcript_path: Path) -> None:
        self._append(f"run completed transcript={transcript_path}")

    def log_failure(self, exc: BaseException, transcript_path: Path | None) -> None:
        self._append(f"run failed error={exc!r}")
        if transcript_path is not None:
            self._append(f"partial transcript={transcript_path}")
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        for line in tb.rstrip().splitlines():
            self._append(line)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    experiment_path = Path(args.experiment_file).resolve()

    dotenv_path = find_dotenv(Path.cwd()) or find_dotenv(experiment_path.parent)
    if dotenv_path is not None:
        load_dotenv(dotenv_path)

    recorder: TranscriptRecorder | None = None
    logger: RunLogger | None = None
    started_at_utc = utc_now()

    try:
        experiment = load_experiment(experiment_path)
        output_dir = resolve_output_dir(
            args.output_dir or experiment.output_dir,
            experiment.path.parent,
        )
        extra_headers = build_openrouter_headers(
            http_referer=experiment.http_referer,
            x_title=experiment.x_title,
            base_urls=[
                experiment.agent_a.base_url,
                experiment.agent_b.base_url,
            ],
        )
        logger = RunLogger(output_dir=output_dir, started_at_utc=started_at_utc)
        logger.log_configuration(
            experiment=experiment,
            model_a=experiment.agent_a.model,
            model_b=experiment.agent_b.model,
            output_dir=output_dir,
        )
        agent_a = AgentConfig(
            name=experiment.agent_a.name,
            prompt=experiment.agent_a.prompt,
            client=OpenAICompatibleClient(
                model=experiment.agent_a.model,
                api_key=api_key_from_env(experiment.agent_a.api_key_env),
                base_url=experiment.agent_a.base_url,
                temperature=experiment.temperature,
                max_tokens=experiment.max_tokens,
                extra_headers=extra_headers,
            ),
        )
        agent_b = AgentConfig(
            name=experiment.agent_b.name,
            prompt=experiment.agent_b.prompt,
            client=OpenAICompatibleClient(
                model=experiment.agent_b.model,
                api_key=api_key_from_env(experiment.agent_b.api_key_env),
                base_url=experiment.agent_b.base_url,
                temperature=experiment.temperature,
                max_tokens=experiment.max_tokens,
                extra_headers=extra_headers,
            ),
        )
        recorder = TranscriptRecorder(
            initial_result=ConversationResult(
                input_file_path=str(experiment.path),
                started_at_utc=started_at_utc,
                finished_at_utc="",
                status="running",
                error_message=None,
                turn_limit=experiment.turns,
                terminated_early=False,
                termination_reason="running",
                verbose_mode=experiment.verbose_mode,
                temperature=experiment.temperature,
                max_tokens=experiment.max_tokens,
                agent_a_name=experiment.agent_a.name,
                agent_b_name=experiment.agent_b.name,
                agent_a_model=experiment.agent_a.model,
                agent_b_model=experiment.agent_b.model,
                agent_a_base_url=experiment.agent_a.base_url,
                agent_b_base_url=experiment.agent_b.base_url,
                agent_a_prompt=experiment.agent_a.prompt,
                agent_b_prompt=experiment.agent_b.prompt,
                opening_instruction=experiment.opening_instruction,
                turns=[],
            ),
            output_dir=output_dir,
        )

        def on_turn_start(
            message_number: int, speaker_turn_number: int, speaker: str, model: str
        ) -> None:
            if logger is not None:
                logger.log_turn_start(
                    message_number, speaker_turn_number, speaker, model
                )

        def on_turn(turn: TranscriptTurn, total_turns: int) -> None:
            emit_progress(turn, total_turns)
            if logger is not None:
                logger.log_turn_complete(turn, total_turns)
            if recorder is not None:
                recorder.record_turn(turn, total_turns)

        result = run_conversation(
            agent_a=agent_a,
            agent_b=agent_b,
            turn_limit=experiment.turns,
            verbose_mode=experiment.verbose_mode,
            allow_early_stop=experiment.early_stop,
            opening_instruction=experiment.opening_instruction,
            on_turn_start=on_turn_start,
            on_turn=on_turn,
            started_at_utc=started_at_utc,
        )
        result = replace(result, input_file_path=str(experiment.path))
        recorder.finalize(result)
        if logger is not None:
            logger.log_success(recorder.path)
    except (ConfigError, LLMClientError) as exc:
        if recorder is not None:
            recorder.fail(str(exc))
            print(f"Partial transcript: {recorder.path}", file=sys.stderr)
        if logger is not None:
            logger.log_failure(exc, recorder.path if recorder is not None else None)
            print(f"Debug log: {logger.path}", file=sys.stderr)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if recorder is not None:
            recorder.fail(str(exc))
            print(f"Partial transcript: {recorder.path}", file=sys.stderr)
        if logger is not None:
            logger.log_failure(exc, recorder.path if recorder is not None else None)
            print(f"Debug log: {logger.path}", file=sys.stderr)
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if recorder is None:
        print("Error: transcript recorder was not initialized.", file=sys.stderr)
        return 1

    print(recorder.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
