# Frozen protocol

This document states what is fixed before any model is evaluated, what each model sees, and
which decisions are left to the models rather than to the program.

## Cases

| Benchmark | Cases | Action cap | Planner targets per episode |
|---|---:|---:|---:|
| BabyAI | 5 tasks × 25 seeds = 125 | 64 (goto, pickup), 128 (open, putnext, pick_up_seq_go_to) | 4 |
| MiniHack | 8 tasks × 5 seeds = 40 | 100 | 4 |
| Baba Is AI | 40 levels × 5 seeds = 200 | 100 | 4 |
| ALFWorld | 134 valid-unseen tasks | 50 | 4 |

Seeds are `20271014 + i` per task. Order, ids, groups and hashes are frozen by `prepare`
before evaluation; cases are never resampled after seeing results. The three grid benchmarks
use the pinned BALROG text wrappers (`blade/upstream.lock.json`); ALFWorld uses its text
setting. Native termination and action caps are kept.

## Methods

| Method | Query per action | Planner |
|---|---|---|
| `*-flat-h0` (System I) | `P(a | s)` | none |
| `*-history-h4` (History) | `P(a | s, h)` | none |
| `*-joint-h4` (Joint) | `P(a | s, h)` for 8 actions, then one query `P(a | s, h, p)` | yes |
| `*-w05`, `*-w2` (BLaDe) | `P(a | s, h)` for 8 actions, then `pi_0`, `pi_h`, `pi_p` | yes |

BLaDe selects `argmax_a  log pi_h(a) + w · (log pi_p(a) − log pi_0(a))` with additive
smoothing `1e-6`. Exact ties are broken by the fixed menu order. Each of the three
distributions comes from an independent request; none sees another request's output.

## Inputs

All methods share the public task text, the complete current native observation, the native
manual and action descriptions, the complete action menu, and the remaining-action counters.

- **History (H4)** is the last four real actions with their complete native feedback. It is
  not summarized, deduplicated or carried across episodes. Rejected actions and failure
  messages stay in the history.
- **Menus.** BabyAI, MiniHack and Baba Is AI use the full static native menu. ALFWorld uses
  the official full list of currently admissible commands, identically for every method.
  Nothing is filtered by state or by target, and there is no top-k truncation.
- **Excluded from prompts:** task ids, seeds, reward and score dictionaries, hidden engine
  state, expert plans, future observations, and other methods' outputs. Evaluation fields are
  recorded only after the environment has executed the action.

## Planner

GPT (`gpt-6-sol`, medium reasoning effort, default service tier, at most 4,096 output tokens
including reasoning) receives the current public state, H4, the action descriptions and the
counters, and returns one JSON string: `{"goal": "<a desired environment state>"}`. Its
reasoning is not passed to the actor.

Targets are requested after `8 + floor(j · (T − 8) / 4)` completed actions for `j = 0..3`,
where `T` is the action cap: `[8, 22, 36, 50]` for T = 64, `[8, 38, 68, 98]` for T = 128,
`[8, 31, 54, 77]` for T = 100. An episode that ends early gets no extra targets. Each method
generates targets along its own trajectory.

## What the program does not do

- It does not check whether a target has been reached and does not refresh targets when the
  agent is stuck.
- It does not predict next states, roll out, search or path-plan.
- It does not test whether an action is executable, mask actions, replace actions or detect
  loops.
- It does not repair prompts or retry after seeing an outcome.

## What the program does do

Shared infrastructure, identical for all methods: native text wrappers and manuals;
deterministic serialization of the last four interactions; the fixed target clock;
normalization and argmax of a complete distribution; exactly one `step` per decision; native
termination; logging and post-hoc scoring. Upstream behaviours are kept as pinned: MiniHack
`AutoMore`/`skip_more`, and Baba Is AI's automatic reset when no object is YOU. This is a
disclosed common interface, not a harness-free setting.

Structural validation only: a response must give finite probabilities over 1–255 distinct
menu options. A malformed response is stored and the episode stops; it is never converted
into a different action.

## Models

- **Jev**: hosted `jev-1.13.0`.
- **d3**: `vllm-sr/d3` at a pinned revision, 27B, BF16 weights, original readout, full
  non-causal attention, temperature 1, no quantization.
- **Kev**: `jaredpalmer/kev-27b` at a pinned revision, full BF16 weights, original decision
  head.

Local servers verify checkpoint hashes, compute settings and library versions at health
check. Silent model substitution is refused.

## Ledger, budgets and failures

Every request reserves budget and is persisted with its raw response. An identical successful
request may be shared between methods; each method's logical cost still counts all of its
dependencies. Cloud cost is computed from reported usage and the configured prices. Unknown
costs and local GPU cost are reported as `null`, not zero.

A provider failure ends only its own episode. The episode stays in the denominator and is
not retried; the run stops after five consecutive failed jobs. Planner dispatch is limited to
45 requests per 60 seconds across processes. On resume, stored actions are replayed and
compared with the stored public observations; nothing is re-decided.

## Metrics

Per benchmark and actor: successes over the frozen denominator, failures, native score,
progress, and mean actions over all completed episodes and over successful ones. Paired
comparisons report repairs and damages and bootstrap intervals over frozen case groups
(2,000 resamples, seed 20261001). A cohort with missing or failed episodes is marked
provisional.

Physical requests (what was actually sent) and logical requests (what each method depends
on) are reported separately, by provider, model, credential slot, task and method, with
tokens, cost and latency mean / p50 / p95.

## Interpretation limits

- System I (no history) versus BLaDe changes both history and target; the difference cannot
  be attributed to the target alone. The History and Joint methods are the controls.
- Closed-loop trajectories diverge, so method comparisons are not single-step comparisons at
  identical states.
- Offline tests check interfaces and scientific identity. They are not evidence of model
  performance.
