# DIRECT:ON playground

Offline environment for comparing tool-calling strategies (free LLM, decision-tree prompt, rule guard, your decision model) on Korean customer-service scenarios, by accuracy and latency.
The rest of this folder is the original starter kit (commit `02cb796`), unchanged.

## Setup

```bash
cp env.example .env   # fill BAZE_API_KEY (or SONIOX_API_KEY), GEMINI_API_KEY, OPENAI_API_KEY (+ DECISION_* later)
uv sync
uv run python -m playground.check   # shows which keys are set and which STT route is used
uv run pytest -q      # offline checks, no keys needed
```

## Layout

| Path | Role |
|---|---|
| `playground/domain/` | Practice manual (`manual.md`), tool schemas (`tools.json`), mock DB (`data.json`). Swap for the official ones. |
| `playground/data/scenarios.jsonl` | Evaluation scenarios: caller turns + `expect` (calls, one_of, allow, forbid, handoff, say_not). |
| `playground/mock.py` | Stateful mock backend per session (cancel/return change order state). |
| `playground/policy.py` | Manual rules checked on every call (verification first, consent, cancel/return eligibility). |
| `playground/tools.py` | `ToolRunner`: in-process mock or `TOOLS_URL`, call log, optional guard that blocks violating calls. |
| `playground/strategies/` | `llm_tools.py` baseline, `decision_stub.py` template for your decision model, `base.py` interface. |
| `playground/prompts/` | Prompt variants (`manual.txt`, `tree.txt`); add files and reference them in `variants.json`. |
| `playground/variants.json` | Experiment grid: strategy + options (prompt, guard, reasoning_effort, model, prefix). |
| `playground/evaluate.py` | Runs variants x scenarios, writes `runs/playground/<ts>/results.jsonl`, `summary.md`. |
| `playground/chat.py` | Interactive text chat with one variant, showing tool calls, policy hits and latency. |
| `playground/dataset/` | Dataset building: import real dialogs, LLM-drafted labels, caller TTS audio, Soniox STT. |

## Running comparisons

```bash
uv run python -m playground.evaluate --repeat 3                      # all variants, all scenarios
uv run python -m playground.evaluate --variants manual,tree --scenarios S02,S03,S15
uv run python -m playground.evaluate --input stt                     # use Soniox transcripts instead of clean text
uv run python -m playground.chat --variant tree+guard
```

Summary columns: `pass_rate`, `call_recall`, `violations/run` (executed rule breaks), `blocked/run` (guard stops),
`forbidden/run`, `extra/run` (unneeded or duplicate calls), `first_text_p50/p90_ms`, `turn_p50_ms`, `llm_calls/turn`, `tokens/turn`.

## Adding your decision model

1. Copy `strategies/decision_stub.py`, implement `decide` and `phrase`; call the model with `llm.chat(..., prefix="DECISION")` so it reads `DECISION_API_KEY`, `DECISION_BASE_URL` and `DECISION_MODEL`.
2. Register the class in `strategies/__init__.py` and add a variant to `variants.json`.
3. Every call goes through `self.s.tools.run`, so scoring and policy checks apply unchanged.

## Building the dataset from real dialogs

Sources found (details and access steps in the session notes):
AI Hub 민원(콜센터) 질의-응답 (K쇼핑 text, intents), AI Hub 상담 음성 (통신판매 audio + transcripts), AI Hub 저음질 전화망 음성인식 (8 kHz phone audio).
All AI Hub sets need an approved application; HF `khaimk/ecommerce-function-calling-datasets-korean` is open but unlicensed (local reference only).

```bash
uv run python -m playground.dataset.import_aihub --src ~/aihub/<unzipped> --inspect     # see the JSON keys
uv run python -m playground.dataset.import_aihub --src ~/aihub/<unzipped> --out playground/data/candidates_aihub.jsonl
uv run --group hf python -m playground.dataset.import_hf --out playground/data/candidates_hf.jsonl
uv run python -m playground.dataset.label --in playground/data/candidates_aihub.jsonl   # drafts -> scenarios_draft.jsonl
# review drafts (set "review": "ok", fix expect), then append accepted rows to scenarios.jsonl
uv run python -m playground.dataset.audio synth --out playground/data/scenarios_audio.jsonl
uv run python -m playground.dataset.audio transcribe --in playground/data/scenarios_audio.jsonl
uv run python -m playground.evaluate --file playground/data/scenarios_audio.jsonl --input stt
```

Do not commit downloaded datasets or official competition data; keep them outside git or under `runs/`.
