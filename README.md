[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xiacai520/gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xiacai520/gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast — `gta6-trophy-rarity-forecast-01`

Forecasts every GTA VI (PS5) trophy's Exophase earn rate at 2026-11-26T00:00:00Z
as five quantiles, from the house's recorded reads up to the case checkpoint.
Entry for [make.fun competition `gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250, 20 places).

## Method

**Counts, not rates.** `earned_i(t) = rate_i(t)/100 × players(t)` is
non-decreasing; `players(t)` only jumps when Exophase refreshes the page. A
rate can fall sharply at a refresh while every count grows, so forecasting
the rate directly confuses dilution with decay. We fit separate log-time
trends to each trophy's earned count and to the player count, extrapolate
both to the target, and divide.

**Median = logit-blend of two legs** (both fit on the last 24h of *distinct*
reads only — consecutive identical reads carry no information, the page is
frozen between Exophase refreshes; regressor is `ln(1 + hours since release)`):

- *Counts leg* (weight 0.3, gated): log-linear fit of `ln(earned + 0.5)` and
  of `ln(players)`; valid only with ≥4 uncensored points (count ≥ 10) and a
  clean fit. Its median is clamped to within 4× of the last rate, so a slope
  error cannot blow the forecast up.
- *Trend leg* (weight 0.7): direct least-squares line of `logit(rate)` on
  log-time — the shape the house curve baseline uses, which tracks the
  story-progression wave well. For trophies already above 80% it is averaged
  with a non-earner-fraction model (`ln(100 − rate)`) so the forecast
  saturates toward 100% instead of overshooting it. For heavily censored
  trophies (last rate < 0.05%, the page prints 2 decimals) the trend is fit
  on `ln(count + 1)` instead of a flat logit series.

All slope-capped extrapolations are anchored at the fit's mean point
(`ybar + b_capped·(x_target − xbar)`), so capping a slope never shifts the level.

**Funnel constraints on the medians** (pool-adjacent-violators): the story
chain Chapter 1 → 2 → 3 → 4 → final mission is non-increasing, and the
platinum / 100%-completion trophies cannot exceed any other trophy.

**Asymmetric spreads.** Downside (refresh dilution) is much wider than
upside. Width = fit residual volatility (which includes past refresh jumps)
× a leverage factor for the extrapolation, plus a prior that fades as
distinct reads accumulate (a flat launch-night history still expects a
refresh — the launch-night refresh diluted most rates ~6×), plus a horizon
drift term. The 100% trophy (held by every tracked player even after a 6×
refresh) keeps tight quantiles just under 100%.

## Backtest results (see `backtest_results.md` for detail)

| Test | Model | Starter (persistence) | House curve |
|---|---|---|---|
| Practice p00/p01 vs reconstructed refresh (R1) | **0.693** | 1.405 | — |
| Synthetic final, fast progression | **0.536** | 2.645 | 0.707 |
| Synthetic final, slow progression | **1.141** | 2.989 | 1.584 |

Pinball loss on the logit scale (lower is better). The model beats both
house baselines in both synthetic worlds and the starter by ~51% on the
practice reconstruction. Rolling within-history backtests on the flat
practice data are degenerate (no refresh inside the window) and are recorded
in `backtest_results.md`.

## Files

- `main.py` — the entry (Python 3.13 stdlib only, no network, deterministic,
  ~0.03 s per case; reads `$MAKEFUN_INPUT_DIR`, writes `$MAKEFUN_OUTPUT_DIR`).
- `backtest.py` — exact scorer (logit pinball loss), practice-answer
  reconstruction, rolling backtests, format/determinism checks.
- `synth_test.py` — synthetic long-horizon validation: cohort simulation of
  a GTA VI launch (join surge, sync delay, per-player progression speeds,
  4-hour Exophase refreshes), cut as a final-like case.
- `grid_search.py`, `ablate_funnel.py` — spread-prior grid search and funnel
  ablation (funnel never binds in the sims; kept as a safety net).

## Reproduce

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 backtest.py eval        # practice backtest + format/determinism checks
python3 synth_test.py           # synthetic 4.5-day-horizon validation
```

## License

MIT (see `LICENSE`).
