# Backtest record — gta6-trophy-forecast-01

All losses are mean pinball loss on the logit scale (lower is better), with
typical miss in percentage points and 90% coverage alongside. Run with
`python3 backtest.py eval` and `python3 synth_test.py` from this directory.

## 1. Practice cases vs reconstructed refresh (the real backtest)

The kit ships no practice answers (program/v1 kits are graded on the server
only — `grade-local` refuses: "its scorer and reference data are not
published in the kit, so it is graded on the server only"). The answers were
reconstructed from the aggregates `competition.yaml` states for the 04:30Z
read: 3,868 players tracked (from 638), 4,276 trophies earned (from 943),
every rate but the 100% one fell.

**R1 ("refresh-realistic", used as truth):** the 3,230 new players earned
~only the intro trophy (id 2, stays 100%); the remaining +103 earns spread
pro-rata to current counts. Validation of the reconstruction: the starter
baseline scores loss **1.4048**, typical miss **0.68pp**, coverage **0.00** —
against the yaml's stated practice figures of loss ~1.51, typical miss
~0.68pp, coverage 0. (p01's answer is the same 04:30 read; the 05:00 read was
unchanged.)

**R2 ("uniform dilution", rejected as truth):** every non-100 rate ×0.7477.
It implies 5,245 total trophies earned vs the known 4,276, so it contradicts
the stated aggregate and is kept only as a mild-dilution stress scenario.

| Case | Truth | Model loss | Model miss | Model cov | Starter loss | Starter miss | Starter cov |
|---|---|---|---|---|---|---|---|
| p00 | R1 | **0.6932** | 0.68pp | 0.98 | 1.4048 | 0.68pp | 0.00 |
| p01 | R1 | **0.6932** | 0.68pp | 0.98 | 1.4048 | 0.68pp | 0.00 |
| p00 | R2 (stress) | 0.2537 | 0.22pp | 0.98 | 0.2026 | 0.22pp | 0.00 |
| p01 | R2 (stress) | 0.2537 | 0.22pp | 0.98 | 0.2026 | 0.22pp | 0.00 |

The model wins by ~51% on R1. On R2 it trails the starter slightly: the
wide downside spread is insurance that costs a little when dilution is mild
and pays a lot when it is severe. R2 is the rejected scenario, so this is an
accepted trade-off.

## 2. Rolling within-history backtests (degenerate, recorded for completeness)

History is completely flat (every read identical), so these carry no signal:

| Cut | Target | Model loss | Starter loss |
|---|---|---|---|
| p00 01:00 | 02:30 | 0.1954 | 0.0249 |
| p01 01:30 | 03:00 | 0.1954 | 0.0249 |

The model's loss here is the insurance premium of its wide downside spread
on a no-refresh outcome. Expected and small in absolute terms.

## 3. Synthetic long-horizon validation (4.5-day horizon, final-like case)

Cohort simulation (`synth_test.py`, seed 20261120): 300k players, launch
join surge, Exophase sync delay, per-player progression speeds, page
refreshes every 4h; case cut at 2026-11-21T11:30Z, target 2026-11-26T00:00Z.
House curve baseline re-implemented faithfully (logit-rate line on
`ln(1+hours)` over the last 24h, RW + residual spread).

| World | Model | Starter | House curve |
|---|---|---|---|
| Fast/hardcore (v_med 0.35) | **0.5356** (miss 7.42pp, cov 0.91) | 2.6454 (37.21pp, 0.00) | 0.7071 (14.09pp, 0.84) |
| Slow/casual (v_med 0.12) | **1.1408** (miss 5.88pp, cov 0.62) | 2.9890 (20.97pp, 0.02) | 1.5840 (15.17pp, 0.47) |

Sample medians (fast world): ch1 98.97 vs truth 99.35; final mission 69.20
vs 47.45; platinum 0.01 vs 0.56. (Slow world: ch1 96.76 vs 95.03; final
0.17 vs 4.41 — rare trophies censored at 2 decimals remain the hardest.)

## 4. Iteration history

- **v1** (counts-only median, flat-history priors): R1 0.7180 (−48.9% vs
  starter). Synthetic final: fast 3.32, slow 3.01 — worse than starter:
  separate log-time fits of counts and players misattribute dilution vs
  progression, and slope capping shifted levels.
- **v2** (added logit-trend leg, 50/50 blend, recent-24h window,
  non-earner model): R1 0.6789; synthetic fast 2.80, slow 3.01 — still
  behind the house curve.
- **v3** (trend leg leads 0.7, counts leg gated + clamped to 4× last rate,
  censored-count branch for rare trophies, wider drift/floors): R1 0.6932;
  synthetic fast 1.48, slow 2.60.
- **v3 + ybar anchor fix** (slope-capped extrapolations anchored at the
  fit's mean point instead of `a + b_capped·xbar`, which had silently
  collapsed capped slopes to ~zero): R1 0.6932; synthetic fast **0.5356**,
  slow **1.1408** — beats the house curve in both worlds.
- **Spread-prior grid** (`grid_search.py`): downside prior 1.5 / upside 0.2
  chosen as a robust middle (R1 keeps improving to 1.8, R2 worsens; 1.5
  balances them and barely affects the final, where the prior weight is ~0.07).
- **Funnel ablation** (`ablate_funnel.py`): funnel on/off give identical
  losses in both synthetic worlds (medians already satisfy the chain) — kept
  as a safety net.

## 5. Format / determinism / runtime

- Output validates: exactly one row per trophy, ids match, 5 finite
  quantiles in [0, 100], non-decreasing, 2,653 bytes (limit 65,536).
- Two runs byte-identical (deterministic).
- Runtime 0.05 s for both practice cases (cap 8 s per case).
- `grade-local` cannot run for this kit (program/v1: scorer and reference
  data not published; server-only grading). Format was verified with the
  local runner instead.
