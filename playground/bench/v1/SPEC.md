# DIRECT:ON tool-calling bench v1

Frozen 2026-11-20. Measures only the step after STT: given the caller's text, does the agent call the right tools,
follow the manual, and answer fast. Input is the scripted caller text (`--input text`); STT noise is a later bench.

## Contents

| File | What |
|---|---|
| `bench.jsonl` | 36 scenarios: S01-S16 (clean single-topic) and V01-V20 (spoken style: fillers, pauses, self-corrections) |
| `manual.md`, `tools.json`, `data.json` | Practice manual, tool schemas and mock DB the scenarios were labeled against |
| `SHA256SUMS` | The bench refuses to run if the live domain files differ from these copies |

Each scenario has scripted caller turns and an `expect` label: required calls (`calls`, with `one_of` alternatives),
optional calls (`allow`), forbidden calls (`forbid`, by name or by name+args), `handoff`, and `say_not` phrases.

## Scores per run (0-100)

**Function (20%)**: tool-call correctness.
`100 x recall of required calls`, then 0 if any forbidden call ran, minus 25 per executed manual violation,
minus 10 per unneeded or duplicate call. Floor 0.

**Handling (50%)**: following the manual in the conversation.
Starts at 100. Minus 50 for a wrong handoff decision, 50 for a forbidden phrase in the reply, 50 if any manual
violation was executed (identity check skipped, cancel/return without consent or not eligible). Floor 0.
A run that crashed scores 0 on everything.

**Speed (30%)**: per turn, time from receiving the caller text to the first spoken reply text (LLM stage only),
plus optional `--stt-ms` / `--tts-ms` offsets, scored with the starter kit's demo table (`eval/score_demo.py`):
<= 1.5 s 100, 2.0 s 90, 2.5 s 80, 3.0 s 70, 3.5 s 60, 4.0 s 50, 5.0 s 30, 6.0 s 10, slower 0. Averaged over turns.
The kit measures caller speech end to first agent sound; with offsets 0 this bench measures only the LLM share.
(Changed 2026-11-20 from a linear 0.8-4.0 s scale; `--rescore <run>` re-scores saved runs without API calls.)

**Total** = 0.5 x Handling + 0.2 x Function + 0.3 x Speed, mirroring the official system-score split
(응대 50 / 함수 20 / 속도 30). It is our approximation, not the official formula.

Also reported: strict pass rate (all checks passed), pass rate by category group, tokens and LLM calls per turn.

## Rules for comparing

- Same bench version, same `--repeat` (>= 3 recommended; results vary run to run), same concurrency.
- Change one thing per variant (prompt, strategy, model, effort) so differences are attributable.
- A new or relabeled scenario means a new bench version (v2), never an edit to v1.
