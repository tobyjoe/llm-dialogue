from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm_dialogue.client import ChatResult
from llm_dialogue.cli import (
    RunLogger,
    build_openrouter_headers,
    find_dotenv,
    load_dotenv,
    render_progress,
    resolve_output_dir,
)
from llm_dialogue.conversation import AgentConfig, ConversationResult, run_conversation
from llm_dialogue.markdown import write_transcript


class FakeClient:
    def __init__(self, model: str, prefix: str) -> None:
        self.model = model
        self.prefix = prefix
        self.calls = 0
        self.history_lengths = []
        self.base_url = "https://openrouter.ai/api/v1"
        self.temperature = 1.0
        self.max_tokens = None

    def generate(self, messages):
        self.calls += 1
        self.history_lengths.append(len(messages))
        latest = messages[-1].content
        return ChatResult(
            content=f"{self.prefix} reply {self.calls} to: {latest}",
            raw_response={},
        )


class ConversationTests(unittest.TestCase):
    def test_conversation_alternates_agents_and_respects_limit(self) -> None:
        agent_a = AgentConfig(
            name="Alpha",
            prompt="You are Alpha.",
            client=FakeClient(model="model-alpha", prefix="Alpha"),
        )
        agent_b = AgentConfig(
            name="Beta",
            prompt="You are Beta.",
            client=FakeClient(model="model-beta", prefix="Beta"),
        )

        result = run_conversation(
            agent_a=agent_a,
            agent_b=agent_b,
            turn_limit=2,
            verbose_mode=False,
            opening_instruction="Start talking.",
        )

        self.assertEqual(len(result.turns), 4)
        self.assertEqual(
            [turn.speaker for turn in result.turns],
            ["Alpha", "Beta", "Alpha", "Beta"],
        )
        self.assertEqual(
            [turn.speaker_turn_number for turn in result.turns],
            [1, 1, 2, 2],
        )
        self.assertEqual(
            [turn.message_number for turn in result.turns],
            [1, 2, 3, 4],
        )
        self.assertIn("Start talking.", result.turns[0].content)
        self.assertIn("Alpha reply 1", result.turns[1].content)
        self.assertEqual(agent_a.client.history_lengths, [2, 4])
        self.assertEqual(agent_b.client.history_lengths, [2, 4])
        self.assertEqual(result.turn_limit, 2)

    def test_turn_limit_must_be_positive(self) -> None:
        agent = AgentConfig(
            name="Solo",
            prompt="Prompt",
            client=FakeClient(model="fake", prefix="Solo"),
        )

        with self.assertRaises(ValueError):
            run_conversation(
                agent_a=agent,
                agent_b=agent,
                turn_limit=0,
                verbose_mode=False,
            )

    def test_on_turn_callback_receives_each_turn(self) -> None:
        agent_a = AgentConfig(
            name="Alpha",
            prompt="You are Alpha.",
            client=FakeClient(model="model-alpha", prefix="Alpha"),
        )
        agent_b = AgentConfig(
            name="Beta",
            prompt="You are Beta.",
            client=FakeClient(model="model-beta", prefix="Beta"),
        )
        seen = []
        starts = []

        run_conversation(
            agent_a=agent_a,
            agent_b=agent_b,
            turn_limit=2,
            verbose_mode=False,
            on_turn_start=lambda message_number, speaker_turn_number, speaker, model: starts.append(
                (message_number, speaker_turn_number, speaker, model)
            ),
            on_turn=lambda turn, total: seen.append(
                (turn.message_number, turn.speaker_turn_number, turn.speaker, total)
            ),
        )

        self.assertEqual(
            starts,
            [
                (1, 1, "Alpha", "model-alpha"),
                (2, 1, "Beta", "model-beta"),
                (3, 2, "Alpha", "model-alpha"),
                (4, 2, "Beta", "model-beta"),
            ],
        )
        self.assertEqual(
            seen,
            [
                (1, 1, "Alpha", 4),
                (2, 1, "Beta", 4),
                (3, 2, "Alpha", 4),
                (4, 2, "Beta", 4),
            ],
        )

    def test_openrouter_headers_are_only_added_for_openrouter_urls(self) -> None:
        headers = build_openrouter_headers(
            http_referer="http://localhost:3000",
            x_title="Experiment Runner",
            base_urls=["https://openrouter.ai/api/v1"],
        )
        self.assertEqual(
            headers,
            {
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Experiment Runner",
            },
        )

        self.assertIsNone(
            build_openrouter_headers(
                http_referer="http://localhost:3000",
                x_title="Experiment Runner",
                base_urls=["https://api.example.com/v1"],
            )
        )

    def test_load_dotenv_sets_missing_env_without_overriding_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text(
                "OPENROUTER_API_KEY=from-file\nEXISTING=from-file\nQUOTED='quoted value'\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                os.environ,
                {"EXISTING": "from-env"},
                clear=False,
            ):
                os.environ.pop("OPENROUTER_API_KEY", None)
                os.environ.pop("QUOTED", None)

                load_dotenv(dotenv_path)

                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "from-file")
                self.assertEqual(os.environ["EXISTING"], "from-env")
                self.assertEqual(os.environ["QUOTED"], "quoted value")

    def test_render_progress_includes_counter_and_identity(self) -> None:
        self.assertEqual(
            render_progress(3, 10, 2, 5, "Agent A", "openai/gpt-4o"),
            "[#######-----------------] message 3/10 Agent A turn 2/5 (openai/gpt-4o)",
        )

    def test_find_dotenv_walks_upward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            dotenv_path = root / ".env"
            dotenv_path.write_text("OPENROUTER_API_KEY=test\n", encoding="utf-8")

            self.assertEqual(find_dotenv(nested), dotenv_path)

    def test_resolve_output_dir_anchors_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = Path(tmpdir)
            self.assertEqual(
                resolve_output_dir("transcripts", anchor),
                (anchor / "transcripts").resolve(),
            )
            self.assertEqual(
                resolve_output_dir("/tmp/out", anchor),
                Path("/tmp/out"),
            )

    def test_write_transcript_renders_running_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ConversationResult(
                started_at_utc="2026-03-14T20:00:00+00:00",
                finished_at_utc="",
                status="running",
                error_message=None,
                turn_limit=2,
                verbose_mode=False,
                temperature=1.0,
                max_tokens=None,
                agent_a_name="A",
                agent_b_name="B",
                agent_a_model="m1",
                agent_b_model="m2",
                agent_a_base_url="https://openrouter.ai/api/v1",
                agent_b_base_url="https://openrouter.ai/api/v1",
                agent_a_prompt="pa",
                agent_b_prompt="pb",
                opening_instruction="start",
                turns=[],
            )

            path = write_transcript(result, Path(tmpdir))
            text = path.read_text(encoding="utf-8")
            self.assertIn("- Status: `running`", text)
            self.assertIn("- Finished (UTC): `in-progress`", text)

    def test_run_logger_writes_turn_and_failure_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = RunLogger(
                output_dir=Path(tmpdir),
                started_at_utc="2026-03-14T20:00:00+00:00",
            )
            logger.log_configuration(
                model_a="openai/gpt-4o",
                model_b="anthropic/claude-3.5-sonnet",
                turn_limit=20,
                verbose_mode=True,
                output_dir=Path(tmpdir),
            )
            logger.log_turn_start(13, 7, "Agent A", "openai/gpt-4o")
            logger.log_failure(
                TimeoutError("timed out"),
                Path(tmpdir) / "conversation_2026-03-14T20-00-00+00-00.md",
            )

            text = logger.path.read_text(encoding="utf-8")
            self.assertIn("turns_per_agent=20 total_messages=40", text)
            self.assertIn(
                "message 13 starting speaker=Agent A speaker_turn=7 model=openai/gpt-4o",
                text,
            )
            self.assertIn("run failed error=TimeoutError('timed out')", text)
            self.assertIn("partial transcript=", text)


if __name__ == "__main__":
    unittest.main()
