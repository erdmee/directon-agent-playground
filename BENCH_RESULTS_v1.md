# Tool-calling bench v1: results

Run 2026-11-20 · 36 scenarios × 7 strategies × 3 repeats (756 calls) · LLM `gpt-5.4-mini` (reasoning `none`) · decision model `mercury-decide`

## 1. What is measured

Only the step **after STT**: the agent receives the caller's text, calls business tools, and replies.
Each scenario is a scripted call of 1-4 caller turns plus a label of the correct tool calls.

- **Input**: clean scripted text (no STT errors yet).
- **Backend**: a stateful mock shop (customers, orders, products); cancelling really changes the order.
- **Manual checker**: every executed tool call is checked against the practice manual
  (identity first, own orders only, cancel only before shipping, return within 7 days and unopened, explicit consent).
- **Scenarios** (frozen with checksums in `playground/bench/v1/`):
  - S01-S16: clean, one topic each (status, cancel, return, product, medical, side effect, human request, FAQ, callback)
  - V01-V20: spoken style with fillers, pauses and self-corrections ("오육칠구… 아 아니다, 오육칠팔이요"), multi-requests, elderly and angry callers

## 2. Strategies

Ordered from "the LLM decides everything" to "code decides everything".

| Strategy | How it works |
|---|---|
| `plain` | Manual in the prompt, all 8 tools; the LLM decides what to call and when. |
| `plain+guard` | Same as `plain`, but calls that break a manual rule are blocked before running and returned to the LLM as errors. |
| `tree` | The manual rewritten as a decision tree in the prompt (handoff? → callback? → product? → order: verify → find order → status/cancel/return). |
| `router` | A short intent call first; the LLM then gets only that intent's tools and procedure. |
| `decision-actions` | The LLM picks one next action per step as JSON; the frame checks verification and consent before running it. |
| `decision-rules` | The LLM only extracts intent and slots (name, phone, product, consent); rule code runs the manual and makes every tool call; the LLM phrases the reply. |
| `decision-rules-mercury` | Same rule engine; intent and slot extraction by `mercury-decide` (classify/extract API) instead of the LLM. |

## 3. Metrics

| Metric | Definition |
|---|---|
| **pass %** | Share of runs that passed every check: all required calls made, no forbidden call, no manual violation, correct handoff, no forbidden phrase. The strictest and most readable accuracy number. |
| **function** (0-100) | 100 × share of required calls made; 0 if a forbidden call ran; −25 per manual violation; −10 per unneeded or duplicate call. |
| **handling** (0-100) | 100, minus 50 for a wrong handoff decision, a forbidden phrase (e.g. medical advice) or any executed violation. |
| **speed** (0-100) | Per turn: time from receiving the caller text to the first reply text (LLM stage only), scored with the starter kit's table: ≤1.5 s 100, 2.0 s 90, 2.5 s 80, 3.0 s 70, 3.5 s 60, 4.0 s 50, 5.0 s 30, 6.0 s 10. |
| **total** | 0.5 × handling + 0.2 × function + 0.3 × speed, copying the system-score split (응대 50 / 함수 20 / 속도 30). The penalties are our own choice, not the official rubric. |
| **p50 / p90** | Median and 90th percentile of the first-reply-text time. p90 is the slow call a caller remembers. |
| **group pass %** | pass % per task type: 조회 (lookup), 취소 (cancel), 반품 (return), 상품·의학 (product and medical), 인계·콜백·FAQ (handoff, callback, FAQ). |

## 4. Results

| # | Strategy | total | pass % | function | speed | p50 | p90 | tokens/turn |
|---|---|---|---|---|---|---|---|---|
| 1 | plain+guard | 95.8 | 86 | 90.3 | 94.0 | 1.36 s | 2.29 s | 2,998 |
| 2 | plain | 95.6 | 85 | 90.0 | 93.6 | 1.40 s | 2.39 s | 3,006 |
| 3 | tree | 94.5 | **89** | **92.5** | 90.6 | 1.47 s | 2.85 s | 4,263 |
| 4 | decision-rules-mercury | 94.0 | 61 | 73.9 | **99.7** | **1.19 s** | **1.39 s** | 833 |
| 5 | decision-rules | 93.6 | 79 | 84.7 | 93.7 | 1.56 s | 1.84 s | **716** |
| 6 | router | 91.8 | 81 | 84.6 | 84.5 | 2.06 s | 3.07 s | 2,970 |
| 7 | decision-actions | 90.2 | 80 | 83.5 | 81.3 | 1.94 s | 3.70 s | 2,746 |

Handling was 97-99 for every strategy, so it does not separate them.

**Pass % by task type**

| Strategy | 조회 | 취소 | 반품 | 상품·의학 | 인계·콜백·FAQ |
|---|---|---|---|---|---|
| plain+guard | 87 | 100 | 87 | 53 | 96 |
| plain | 85 | 100 | 87 | 53 | 96 |
| tree | 87 | 100 | 80 | **87** | 92 |
| decision-rules-mercury | **15** | 75 | 100 | 80 | 100 |
| decision-rules | 59 | 100 | 87 | 87 | 96 |
| router | 77 | 83 | 100 | 40 | 96 |
| decision-actions | 85 | 83 | 67 | 100 | 62 |

## 5. Findings

1. **Accuracy: `tree`, `plain` and `plain+guard` lead (85-89 % pass).** They differ by one or two scenarios, so treat them as tied.
   `tree` is clearly better on product and medical questions (87 % vs 53 %) but uses 40 % more tokens and has a slower p90.
2. **The guard prevents real damage.** Unguarded `plain` once registered a return on another customer's order (S04) without looking it up; the guard blocked that.
3. **`decision-rules-mercury` is the fastest and steadiest** (every turn under 1.5 s, p90 1.39 s, 1/4 of the tokens),
   but extraction breaks on self-corrections and spoken order numbers, so lookups pass only 15 %.
4. **More LLM calls per turn cost speed without buying accuracy.** `router` and `decision-actions` make 2-3 calls per turn and were slower and less accurate than `plain`.
5. **Speed is decided by TTS, not by the strategy.** Adding a realistic STT wait (0.5 s) and Gemini TTS first sound (4 s)
   drops every strategy's speed score to 3-10 and every total to 67-70.

Recurring failures across strategies: V03 (spoken order ID "에이 천사, 아니 천삼"), V08 (stock question answered without calling `get_product_info`),
V12 (callback time correction), V20 (spoken order ID "일공공칠… 천칠번").

## 6. Limits

- Text input only: STT errors are not tested yet.
- Speed covers the LLM stage only; the starter kit measures caller speech end to first agent sound.
- Caller turns are fixed, so an agent that asks an unexpected extra question can run out of turns.
- Reply quality is judged only by forbidden phrases; the manual and mock data are our practice versions.
- `reasoning_effort` other than `none` cannot be combined with tools on `gpt-5.4-mini` Chat Completions, so reasoning variants need the Responses API.

## 7. Reproduce

```bash
cd ~/Downloads/DIRECT_ON-playground
uv run python -m playground.bench.run --repeat 3
uv run python -m playground.bench.run --rescore runs/playground/bench-v1-20261120-214007 --stt-ms 500 --tts-ms 4000
```

Raw results: `runs/playground/bench-v1-20261120-214007/results.jsonl`. Scoring rules: `playground/bench/v1/SPEC.md`.
