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
Experiment inputs live as tracked TOML files under [experiments/example.toml](/tmp/convos-config-worktree/experiments/example.toml).

## Important limitation

`--verbose-mode` can only ask the models to expose their reasoning in the text they send. It cannot force access to hidden provider-side chain-of-thought if the API does not expose it.

## Install

```bash
python3 -m pip install -e .
```

## Required environment variables

By default both agents read their API key from `OPENROUTER_API_KEY`.

You can override this per agent inside the experiment file.

Optional OpenRouter attribution headers:

- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_X_TITLE`

These map to the optional `HTTP-Referer` and `X-Title` headers documented by OpenRouter.

The CLI auto-loads a repo-local `.env` file before running an experiment. Existing shell environment variables still win over values in `.env`.

## Experiment files

The CLI now takes a single tracked TOML file instead of many prompt/model flags. Each experiment file stores:

- prompts
- model IDs
- per-agent names, base URLs, and API key env vars
- turn count
- temperature and max token settings
- opening instruction
- verbose mode and early-stop behavior
- output directory
- optional OpenRouter attribution headers

See [experiments/example.toml](/tmp/convos-config-worktree/experiments/example.toml) for the canonical shape.

## Quick start

Run the sample experiment:

```bash
export OPENROUTER_API_KEY=your_key_here
llm-dialogue \
  experiments/example.toml
```

Override only the output directory:

```bash
llm-dialogue \
  experiments/example.toml \
  --output-dir transcripts/scratch
```

In the sample config, `turns = 20` means Agent A gets 20 turns and Agent B gets 20 turns, for 40 total messages.
By default the agents can also end the conversation early. The system prompt tells them to use the exact token `[[CONCLUSION_REACHED]]` when they believe the conversation has reached a stable conclusion. The run ends only when both agents include that token in back-to-back messages.

## Output

Each run writes a Markdown file into `transcripts/` by default with:

- the absolute path to the input experiment file at the top
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

## CLI reference

```text
usage: llm-dialogue [-h] [--output-dir OUTPUT_DIR] experiment_file
```
