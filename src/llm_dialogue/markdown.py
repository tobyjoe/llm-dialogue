from __future__ import annotations

from pathlib import Path

from .conversation import ConversationResult


def timestamp_slug(started_at_utc: str) -> str:
    return started_at_utc.replace(":", "-")


def transcript_path(started_at_utc: str, output_dir: Path) -> Path:
    return output_dir / f"conversation_{timestamp_slug(started_at_utc)}.md"


def write_transcript(result: ConversationResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_path(result.started_at_utc, output_dir)
    path.write_text(render_transcript(result), encoding="utf-8")
    return path


def render_transcript(result: ConversationResult) -> str:
    lines = [
        "# LLM Conversation Transcript",
        "",
        f"Input file: `{result.input_file_path}`",
        "",
        "## Run Metadata",
        "",
        f"- Started (UTC): `{result.started_at_utc}`",
        f"- Finished (UTC): `{result.finished_at_utc or 'in-progress'}`",
        f"- Status: `{result.status}`",
        f"- Turns per agent: `{result.turn_limit}`",
        f"- Total messages: `{result.turn_limit * 2}`",
        f"- Messages used: `{len(result.turns)}`",
        f"- Terminated early: `{result.terminated_early}`",
        f"- Termination reason: `{result.termination_reason}`",
        f"- Verbose mode: `{result.verbose_mode}`",
        f"- Temperature: `{result.temperature}`",
        f"- Max tokens: `{result.max_tokens}`",
        f"- Agent A: `{result.agent_a_name}` using `{result.agent_a_model}`",
        f"- Agent B: `{result.agent_b_name}` using `{result.agent_b_model}`",
        f"- Agent A base URL: `{result.agent_a_base_url}`",
        f"- Agent B base URL: `{result.agent_b_base_url}`",
        "",
    ]

    if result.error_message:
        lines.extend(
            [
                "## Error",
                "",
                "```text",
                result.error_message,
                "```",
                "",
            ]
        )

    lines.extend(
        [
        "## Opening Setup",
        "",
        "### Agent A Prompt",
        "",
        "```text",
        result.agent_a_prompt,
        "```",
        "",
        "### Agent B Prompt",
        "",
        "```text",
        result.agent_b_prompt,
        "```",
        "",
        "### Opening Instruction",
        "",
        "```text",
        result.opening_instruction,
        "```",
        "",
        "## Transcript",
        "",
        ]
    )

    for turn in result.turns:
        lines.extend(
            [
                f"### {turn.speaker} Turn {turn.speaker_turn_number}",
                "",
                f"- Message number: `{turn.message_number}`",
                f"- Model: `{turn.model}`",
                f"- Timestamp (UTC): `{turn.timestamp_utc}`",
                "",
                "```text",
                turn.content,
                "```",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"
