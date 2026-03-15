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
    optional_env,
)
from .conversation import AgentConfig, ConversationResult, TranscriptTurn, run_conversation, utc_now
from .markdown import timestamp_slug, write_transcript


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a turn-based direct conversation between two LLM agents and save the transcript as Markdown."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Shared base prompt for both agents. Override per agent with --prompt-a and --prompt-b.",
    )
    parser.add_argument("--prompt-a", help="Base prompt for agent A.")
    parser.add_argument("--prompt-b", help="Base prompt for agent B.")
    parser.add_argument(
        "--opening-instruction",
        default="Begin now. Speak directly to the other model and start the interaction.",
        help="Initial instruction used to kick off the conversation.",
    )
    parser.add_argument("--model-a", required=True, help="Model name for agent A.")
    parser.add_argument(
        "--model-b",
        help="Model name for agent B. Defaults to --model-a if omitted.",
    )
    parser.add_argument(
        "--base-url-a",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible API base URL for agent A. Defaults to OpenRouter.",
    )
    parser.add_argument(
        "--base-url-b",
        help="OpenAI-compatible API base URL for agent B. Defaults to --base-url-a.",
    )
    parser.add_argument(
        "--api-key-env-a",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the API key for agent A. Defaults to OpenRouter.",
    )
    parser.add_argument(
        "--api-key-env-b",
        help="Environment variable containing the API key for agent B. Defaults to --api-key-env-a.",
    )
    parser.add_argument(
        "--http-referer",
        default=optional_env("OPENROUTER_HTTP_REFERER"),
        help=(
            "Optional OpenRouter attribution header. "
            "Defaults to OPENROUTER_HTTP_REFERER when set."
        ),
    )
    parser.add_argument(
        "--x-title",
        default=optional_env("OPENROUTER_X_TITLE") or "llm-dialogue-experiments",
        help=(
            "Optional OpenRouter attribution title. "
            "Defaults to OPENROUTER_X_TITLE or 'llm-dialogue-experiments'."
        ),
    )
    parser.add_argument(
        "--name-a",
        default="Agent A",
        help="Display name for agent A in the transcript.",
    )
    parser.add_argument(
        "--name-b",
        default="Agent B",
        help="Display name for agent B in the transcript.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=20,
        help="Number of turns per agent. Total messages will be double this value.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature passed to both models.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional max_tokens value for each model response.",
    )
    parser.add_argument(
        "--verbose-mode",
        action="store_true",
        help=(
            "Ask the agents to make their reasoning visible in the text they send. "
            "This does not guarantee access to hidden provider-side reasoning."
        ),
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help=(
            "Disable the explicit conclusion protocol and always run the full configured turns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="transcripts",
        help="Directory where Markdown transcripts will be written.",
    )
    return parser


def parse_prompts(args: argparse.Namespace) -> tuple[str, str]:
    prompt_a = args.prompt_a or args.prompt
    prompt_b = args.prompt_b or args.prompt
    if not prompt_a or not prompt_b:
        raise SystemExit(
            "You must provide a shared prompt positional argument or both --prompt-a and --prompt-b."
        )
    return prompt_a, prompt_b


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
        model_a: str,
        model_b: str,
        turn_limit: int,
        verbose_mode: bool,
        allow_early_stop: bool,
        output_dir: Path,
    ) -> None:
        self._append(
            "configuration "
            f"model_a={model_a} model_b={model_b} "
            f"turns_per_agent={turn_limit} total_messages={turn_limit * 2} "
            f"verbose_mode={verbose_mode} allow_early_stop={allow_early_stop} "
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
    dotenv_path = find_dotenv(Path.cwd())
    if dotenv_path is not None:
        load_dotenv(dotenv_path)

    parser = build_parser()
    args = parser.parse_args()

    prompt_a, prompt_b = parse_prompts(args)
    model_b = args.model_b or args.model_a
    base_url_b = args.base_url_b or args.base_url_a
    api_key_env_b = args.api_key_env_b or args.api_key_env_a
    anchor_dir = dotenv_path.parent if dotenv_path is not None else Path.cwd()
    output_dir = resolve_output_dir(args.output_dir, anchor_dir)
    extra_headers = build_openrouter_headers(
        http_referer=args.http_referer,
        x_title=args.x_title,
        base_urls=[args.base_url_a, base_url_b],
    )
    allow_early_stop = not args.no_early_stop
    recorder: TranscriptRecorder | None = None
    logger: RunLogger | None = None
    started_at_utc = utc_now()

    try:
        logger = RunLogger(output_dir=output_dir, started_at_utc=started_at_utc)
        logger.log_configuration(
            model_a=args.model_a,
            model_b=model_b,
            turn_limit=args.turns,
            verbose_mode=args.verbose_mode,
            allow_early_stop=allow_early_stop,
            output_dir=output_dir,
        )
        agent_a = AgentConfig(
            name=args.name_a,
            prompt=prompt_a,
            client=OpenAICompatibleClient(
                model=args.model_a,
                api_key=api_key_from_env(args.api_key_env_a),
                base_url=args.base_url_a,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_headers=extra_headers,
            ),
        )
        agent_b = AgentConfig(
            name=args.name_b,
            prompt=prompt_b,
            client=OpenAICompatibleClient(
                model=model_b,
                api_key=api_key_from_env(api_key_env_b),
                base_url=base_url_b,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_headers=extra_headers,
            ),
        )
        recorder = TranscriptRecorder(
            initial_result=ConversationResult(
                started_at_utc=started_at_utc,
                finished_at_utc="",
                status="running",
                error_message=None,
                turn_limit=args.turns,
                terminated_early=False,
                termination_reason="running",
                verbose_mode=args.verbose_mode,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                agent_a_name=args.name_a,
                agent_b_name=args.name_b,
                agent_a_model=args.model_a,
                agent_b_model=model_b,
                agent_a_base_url=args.base_url_a,
                agent_b_base_url=base_url_b,
                agent_a_prompt=prompt_a,
                agent_b_prompt=prompt_b,
                opening_instruction=args.opening_instruction,
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
            turn_limit=args.turns,
            verbose_mode=args.verbose_mode,
            allow_early_stop=allow_early_stop,
            opening_instruction=args.opening_instruction,
            on_turn_start=on_turn_start,
            on_turn=on_turn,
            started_at_utc=started_at_utc,
        )
        recorder.finalize(result)
        if logger is not None:
            logger.log_success(recorder.path)
    except LLMClientError as exc:
        if recorder is not None:
            recorder.fail(str(exc))
            print(f"Partial transcript: {recorder.path}", file=sys.stderr)
        if logger is not None:
            logger.log_failure(exc, recorder.path if recorder is not None else None)
            print(f"Debug log: {logger.path}", file=sys.stderr)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
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
