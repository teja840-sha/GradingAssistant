# Grading Assistant (Agentic, A2A, Two Apps)

This repository scaffolds an agentic grading assistant with:
- **A2A protocol** (typed, versioned) for agent handoffs
- **Prompts** externalized as YAML (system/task/constraints/schema)
- Two apps (single-file each):
  - `app_grading_assistant.py`: static talking avatar + SRT (no heavy ML)
  - `app_avatar_viseme.py`: creative viseme-based lipsync + SRT (no heavy ML)
- Strict **no-secrets** policy: configuration comes from environment variables.

## Quick start (scaffold check)
```bash
# using uv
uv run python apps/app_grading_assistant.py --help
uv run python apps/app_avatar_viseme.py --help
```

## Layout
```
apps/                 # single-file apps
configs/              # model config and env examples
prompts/              # agent prompts (YAML)
protocols/            # A2A envelope models (pydantic skeleton)
assets/               # put avatar images here (kept out of git if private)
samples/              # sample inputs/outputs (text-only placeholders)
outputs/              # generated artifacts (gitignored)
```
