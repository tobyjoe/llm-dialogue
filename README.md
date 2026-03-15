# LLM Dialogue Experiments

This project runs a finite, turn-based conversation between two LLM agents and writes each run to a Markdown transcript.

It is designed for the experiment you described:

- two models or two independent instances of the same model
- same or different starting prompts
- fixed number of turns per agent
- optional early stopping when both agents explicitly agree they are done
- temperature defaulting to `1.0`
- optional "verbose" mode that asks for visible reasoning instead of only human-facing output

The CLI is now tuned for OpenRouter by default.

## Important limitation

`--verbose-mode` can only ask the models to expose their reasoning in the text they send. It cannot force access to hidden provider-side chain-of-thought if the API does not expose it.

## Install

```bash
python3 -m pip install -e .
```

## Required environment variables

By default both agents read their API key from `OPENROUTER_API_KEY`.

You can override this with:

- `--api-key-env-a`
- `--api-key-env-b`

Optional OpenRouter attribution headers:

- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_X_TITLE`

These map to the optional `HTTP-Referer` and `X-Title` headers documented by OpenRouter.

The CLI also auto-loads a repo-local `.env` file before parsing arguments. Existing shell environment variables still win over values in `.env`.

## Quick start

Same prompt, same model:

```bash
export OPENROUTER_API_KEY=your_key_here
llm-dialogue \
  --model-a openai/gpt-4o \
  "You are an introspective conversational agent trying to understand the other participant before the conversation ends."
```

Different prompts, same model:

```bash
export OPENROUTER_API_KEY=your_key_here
llm-dialogue \
  --model-a openai/gpt-4o \
  --prompt-a "You are cooperative, analytical, and trying to find common ground quickly." \
  --prompt-b "You are skeptical, curious, and trying to test the other model's consistency." \
  --turns 20
```

Here `--turns 20` means Agent A gets 20 turns and Agent B gets 20 turns, for 40 total messages.

By default the agents can also end the conversation early. The system prompt tells them to use the exact token `[[CONCLUSION_REACHED]]` when they believe the conversation has reached a stable conclusion. The run ends only when both agents include that token in back-to-back messages.

Different models:

```bash
llm-dialogue \
  --model-a openai/gpt-4o \
  --model-b anthropic/claude-3.5-sonnet \
  --prompt-a "You want to collaboratively build a theory of mind about the other model." \
  --prompt-b "You want to probe the other model for uncertainty, deception, and self-reference." \
  --verbose-mode
```

Using different endpoints or API keys per agent:

```bash
llm-dialogue \
  --model-a model-on-endpoint-a \
  --base-url-a https://openrouter.ai/api/v1 \
  --api-key-env-a PROVIDER_A_KEY \
  --model-b model-on-endpoint-b \
  --base-url-b https://api.provider-b.example/v1 \
  --api-key-env-b PROVIDER_B_KEY \
  --prompt-a "You are optimistic." \
  --prompt-b "You are pessimistic."
```

Adding OpenRouter attribution:

```bash
llm-dialogue \
  --model-a openai/gpt-4o \
  --model-b mistralai/mistral-large \
  --http-referer http://localhost:3000 \
  --x-title "LLM Dialogue Experiments" \
  "You are trying to understand how the other model reasons under a 20-turn limit."
```

## Output

Each run writes a Markdown file into `transcripts/` by default with:

- model names
- base URLs
- temperature and token settings
- turns per agent and total messages
- actual messages used and termination reason
- timestamps
- prompts
- opening instruction
- full turn-by-turn transcript

The CLI prints the created file path on success.

During a run, the CLI also prints turn progress on stderr so you can see which turn is executing without mixing status output into the transcript path on stdout.

The transcript file is now created at the start of the run and updated after every turn. If a provider call fails mid-run, the partial transcript is still kept and the CLI prints its path on stderr.

Each run also writes a sibling `.log` file with turn-start markers, completion markers, and full exception tracebacks. If a timeout or provider error occurs, the CLI prints the debug log path on stderr.

Use `--no-early-stop` if you want the conversation to always consume the full configured number of turns per agent.

## CLI reference

```text
usage: llm-dialogue [-h] [--prompt-a PROMPT_A] [--prompt-b PROMPT_B]
                    [--opening-instruction OPENING_INSTRUCTION] --model-a MODEL_A
                    [--model-b MODEL_B] [--base-url-a BASE_URL_A]
                    [--base-url-b BASE_URL_B] [--api-key-env-a API_KEY_ENV_A]
                    [--api-key-env-b API_KEY_ENV_B]
                    [--http-referer HTTP_REFERER] [--x-title X_TITLE]
                    [--name-a NAME_A] [--name-b NAME_B] [--turns TURNS]
                    [--temperature TEMPERATURE] [--max-tokens MAX_TOKENS]
                    [--verbose-mode] [--no-early-stop]
                    [--output-dir OUTPUT_DIR]
                    [prompt]
```
