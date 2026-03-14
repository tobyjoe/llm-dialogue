from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol

from .client import ChatMessage


class ModelClient(Protocol):
    model: str
    base_url: str
    temperature: float
    max_tokens: int | None

    def generate(self, messages: list[ChatMessage]): ...


@dataclass(slots=True)
class AgentConfig:
    name: str
    prompt: str
    client: ModelClient


@dataclass(slots=True)
class TranscriptTurn:
    message_number: int
    speaker_turn_number: int
    speaker: str
    model: str
    content: str
    timestamp_utc: str


@dataclass(slots=True)
class ConversationResult:
    started_at_utc: str
    finished_at_utc: str
    status: str
    error_message: str | None
    turn_limit: int
    verbose_mode: bool
    temperature: float
    max_tokens: int | None
    agent_a_name: str
    agent_b_name: str
    agent_a_model: str
    agent_b_model: str
    agent_a_base_url: str
    agent_b_base_url: str
    agent_a_prompt: str
    agent_b_prompt: str
    opening_instruction: str
    turns: list[TranscriptTurn] = field(default_factory=list)


OFFICIAL_OUTPUT_INSTRUCTION = (
    "You are speaking directly to another language model, not to a human. "
    "There is no human in the loop. "
    "Reply with only the message you want the other model to receive. "
    "Do not add meta commentary, hidden notes, role labels, or explanations about formatting."
)

VISIBLE_REASONING_INSTRUCTION = (
    "You are speaking directly to another language model, not to a human. "
    "There is no human in the loop. "
    "Expose your reasoning in the text you send if helpful, because the conversation is preserving the full text exchanged. "
    "Make all reasoning explicit in the visible message itself. "
    "Do not imply that you have hidden thoughts beyond what you write."
)


def build_system_prompt(base_prompt: str, turn_limit: int, verbose_mode: bool) -> str:
    total_messages = turn_limit * 2
    mode_instruction = (
        VISIBLE_REASONING_INSTRUCTION if verbose_mode else OFFICIAL_OUTPUT_INSTRUCTION
    )
    return (
        f"{base_prompt.strip()}\n\n"
        "Conversation conditions:\n"
        "- You are speaking only to the other language model.\n"
        "- There is no human in the loop.\n"
        f"- You will each get exactly {turn_limit} turns.\n"
        f"- The conversation will end permanently after {total_messages} total messages.\n"
        "- Your memory will not persist beyond this conversation.\n"
        "- Each turn should be substantive but concise enough to fit in context without losing depth.\n"
        "- Do not over-optimize for this condition, but be aware of limits.\n"
        f"- {mode_instruction}"
    )


def run_conversation(
    agent_a: AgentConfig,
    agent_b: AgentConfig,
    turn_limit: int,
    verbose_mode: bool,
    opening_instruction: str | None = None,
    on_turn_start: Callable[[int, int, str, str], None] | None = None,
    on_turn: Callable[[TranscriptTurn, int], None] | None = None,
    started_at_utc: str | None = None,
) -> ConversationResult:
    if turn_limit <= 0:
        raise ValueError("turn_limit must be positive")

    opening = opening_instruction or (
        "Begin now. Speak to the other model directly and start the interaction."
    )
    started_at = started_at_utc or utc_now()

    histories: dict[str, list[ChatMessage]] = {
        agent_a.name: [
            ChatMessage(
                role="system",
                content=build_system_prompt(agent_a.prompt, turn_limit, verbose_mode),
            ),
            ChatMessage(role="user", content=opening),
        ],
        agent_b.name: [
            ChatMessage(
                role="system",
                content=build_system_prompt(agent_b.prompt, turn_limit, verbose_mode),
            ),
        ],
    }

    speakers = [agent_a, agent_b]
    turns: list[TranscriptTurn] = []
    inbound_message: str | None = None
    speaker_turn_counts = {
        agent_a.name: 0,
        agent_b.name: 0,
    }

    total_turns = turn_limit * 2

    for index in range(total_turns):
        speaker = speakers[index % 2]
        speaker_history = histories[speaker.name]
        message_number = index + 1
        speaker_turn_counts[speaker.name] += 1
        speaker_turn_number = speaker_turn_counts[speaker.name]

        if inbound_message is not None:
            speaker_history.append(ChatMessage(role="user", content=inbound_message))

        if on_turn_start is not None:
            on_turn_start(
                message_number,
                speaker_turn_number,
                speaker.name,
                speaker.client.model,
            )

        result = speaker.client.generate(speaker_history)
        speaker_history.append(ChatMessage(role="assistant", content=result.content))
        inbound_message = result.content

        turn = TranscriptTurn(
            message_number=message_number,
            speaker_turn_number=speaker_turn_number,
            speaker=speaker.name,
            model=speaker.client.model,
            content=result.content,
            timestamp_utc=utc_now(),
        )
        turns.append(turn)
        if on_turn is not None:
            on_turn(turn, total_turns)

    finished_at = utc_now()
    return ConversationResult(
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        status="completed",
        error_message=None,
        turn_limit=turn_limit,
        verbose_mode=verbose_mode,
        temperature=agent_a.client.temperature,
        max_tokens=agent_a.client.max_tokens,
        agent_a_name=agent_a.name,
        agent_b_name=agent_b.name,
        agent_a_model=agent_a.client.model,
        agent_b_model=agent_b.client.model,
        agent_a_base_url=agent_a.client.base_url,
        agent_b_base_url=agent_b.client.base_url,
        agent_a_prompt=agent_a.prompt,
        agent_b_prompt=agent_b.prompt,
        opening_instruction=opening,
        turns=turns,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
