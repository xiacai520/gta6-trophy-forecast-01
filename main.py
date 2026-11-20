"""gta6-trophy-rarity-forecast-01 entry: counts-based trophy rarity forecaster.

Method (see README.md for the full write-up):
  * Work in COUNTS, not rates.  earned_i(t) = rate_i(t)/100 * players(t) is
    non-decreasing; players(t) only jumps when Exophase refreshes the page.
    A rate can fall sharply at a refresh even while every count grows, so
    forecasting the rate directly confuses dilution with decay.  We fit a
    separate log-time trend to each trophy's earned count and to the player
    count, extrapolate both to the target, and divide: rate = count/players.
  * The median is a logit-space blend of that counts model and a direct
    logit(rate) trend (the house curve's shape), both fit on the last 24
    hours of DISTINCT reads only (consecutive identical reads carry no
    information: the page is frozen between Exophase refreshes).  The blend
    hedges the two failure modes: dilution (counts model wins) and the
    story-progression wave arriving late (trend model wins).  For trophies
    already above 80%, a non-earner-fraction model replaces the trend leg
    so the forecast saturates toward 100% instead of overshooting it.
  * Slopes are clamped >= 0 for counts/players (they never decrease).
  * Funnel constraints on the medians: the story chain (Chapter 1..4, final
    mission) is non-increasing, and the platinum / 100%-completion trophies
    cannot exceed any other trophy.
  * Quantile spreads are asymmetric: downside (refresh dilution) is much
    wider than upside.  Width comes from the fits' residual volatility
    (which includes past refresh jumps) times a leverage factor for the
    extrapolation, plus a prior that fades as more distinct reads arrive
    (so a flat launch-night history still expects a refresh, as the
    practice cases demand) and a horizon floor.
  * The 100% trophy (every tracked player holds it, even after a 6x
    refresh) keeps tight quantiles just under 100.

Python 3.13 standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Spread priors (logit units).  The launch-night refresh diluted most rates
# ~6x (ln 6 = 1.79); the downside prior sits below that because a refresh is
# a possibility, not a certainty, inside a short horizon.
PRIOR_DOWN = 1.50
PRIOR_UP = 0.20
# Horizon floors: never be overconfident about a multi-day forecast.
FLOOR_DOWN = 0.60
FLOOR_UP = 0.45
# Random-walk drift added per sqrt(day) of horizon.
DRIFT = 0.50
# Slope caps in log-time space (guards against explosive extrapolation).
MAX_SLOPE_N = 3.0
MAX_SLOPE_C = 4.0
MAX_SLOPE_LOGIT = 4.0
MIN_SLOPE_LOGIT = -2.0
# Fits use the last 24h of distinct reads (recent dynamics dominate).
FIT_WINDOW_HOURS = 24.0
# Above this last rate, the non-earner-fraction model replaces the trend leg.
HIGH_RATE = 80.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt: datetime) -> float:
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


def ols(xs, ys):
    """Least squares y ~ a + b*x.  Returns (a, b, resid_std, xbar, sxx, ybar)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return ybar, 0.0, 0.0, xbar, sxx, ybar
    b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    a = ybar - b * xbar
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, xbar, sxx, ybar


def pava_decreasing(vals):
    """Pool-adjacent-violators: the non-increasing fit minimising squared error."""
    out = []
    for v in vals:
        out.append([v, 1])
        while len(out) >= 2 and out[-2][0] < out[-1][0]:
            m2, w2 = out.pop()
            m1, w1 = out.pop()
            out.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    res = []
    for mean, w in out:
        res.extend([mean] * w)
    return res


def distinct_reads(history):
    """Collapse consecutive identical (players, rates) reads to change points."""
    pts = []
    prev = None
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev:
            pts.append(h)
            prev = key
    return pts


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    target_dt = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target_dt))
    at_dt = parse_time(case["at"])

    reads = distinct_reads(case["history"])
    n_distinct = len(reads)
    # Recent window: distinct reads from the last FIT_WINDOW_HOURS (fall back to all).
    recent = [h for h in reads
              if (at_dt - parse_time(h["at"])).total_seconds() <= FIT_WINDOW_HOURS * 3600]
    if len(recent) < 3:
        recent = reads
    xs_all = [math.log1p(hours_since_release(parse_time(h["at"]))) for h in reads]
    xs = [math.log1p(hours_since_release(parse_time(h["at"]))) for h in recent]
    players = [float(h["players_tracked"]) for h in recent]
    last = case["history"][-1]
    last_rates = [last["rates"][i] if last["rates"][i] is not None else 0.0 for i in range(n_tr)]
    last_n = float(last["players_tracked"])

    # --- players-tracked trend (log counts on log time), recent window ---
    if len(recent) >= 3 and max(players) > min(players):
        a_n, b_n, s_n, xbar_n, sxx_n, ybar_n = ols(xs, [math.log(p) for p in players])
        b_n = min(max(b_n, 0.0), MAX_SLOPE_N)
        lev_n = (x_target - xbar_n) ** 2 / sxx_n if sxx_n > 1e-12 else 0.0
        # Anchor at the fit's mean point: capping the slope must not move the level.
        log_n_pred = ybar_n + b_n * (x_target - xbar_n)
        log_n_pred = min(max(log_n_pred, math.log(last_n)), math.log(last_n) + math.log(20.0))
    else:
        b_n, s_n, lev_n = 0.0, 0.0, 0.0
        log_n_pred = math.log(last_n)
    n_pred = math.exp(log_n_pred)

    # Prior weight: fades as distinct reads accumulate (1.0 when flat, ~0.07 at 15).
    prior_w = 1.0 / (1.0 + max(0, n_distinct - 2))
    drift = DRIFT * math.sqrt(horizon_days)

    medians = []
    sig_down = []
    sig_up = []
    for i, tr in enumerate(trophies):
        counts = []
        use_x = []
        for k, h in enumerate(recent):
            r = h["rates"][i]
            if r is None:
                continue
            counts.append(r / 100.0 * h["players_tracked"])
            use_x.append(xs[k])
        last_c = last_rates[i] / 100.0 * last_n

        if last_rates[i] >= 99.995:
            # Held by every tracked player, through a 6x refresh: stays ~100%.
            medians.append(99.99)
            sig_down.append(0.05)
            sig_up.append(0.0)
            continue

        # --- leg 1: counts model (gated: needs uncensored signal) ---
        # A count is 'uncensored' when the 2-decimal page print can see it:
        # rate >= 0.01% means count >= ~10 at the player counts involved.
        uncensored = sum(1 for c in counts if c >= 10.0)
        counts_valid = False
        m_counts = last_rates[i]
        base_sig = 0.0
        s_c = 0.0
        if len(counts) >= 4 and uncensored >= 4 and max(counts) > min(counts) and max(counts) > 0:
            ys = [math.log(c + 0.5) for c in counts]
            a_c, b_c, s_c, xbar_c, sxx_c, ybar_c = ols(use_x, ys)
            if s_c < 0.8 and 0.0 <= b_c <= MAX_SLOPE_C:
                counts_valid = True
                b_c = min(max(b_c, 0.0), MAX_SLOPE_C)
                lev_c = (x_target - xbar_c) ** 2 / sxx_c if sxx_c > 1e-12 else 0.0
                log_c_pred = ybar_c + b_c * (x_target - xbar_c)
                log_c_pred = min(max(log_c_pred, math.log(last_c + 0.5)), math.log(n_pred))
                fit_sig_c = math.sqrt(s_c ** 2 + s_n ** 2)
                extrap_c = math.sqrt(1.0 + min(lev_c + lev_n, 5.0))
                c_pred = math.exp(log_c_pred) - 0.5
                c_pred = min(max(c_pred, 0.0), n_pred)
                m_counts = 100.0 * c_pred / n_pred if n_pred > 0 else last_rates[i]
                # Guard against slope-error blow-up: the counts leg may not move
                # the median more than 4x away from the last rate on its own.
                m_counts = min(max(m_counts, last_rates[i] / 4.0 if last_rates[i] > 0 else 0.01),
                               min(last_rates[i] * 4.0, 99.5) if last_rates[i] > 0 else 99.5)
                base_sig = fit_sig_c * extrap_c

        # --- leg 2: direct trend ---
        rates_recent = [h["rates"][i] for h in recent if h["rates"][i] is not None]
        if last_rates[i] < 0.05 and len(counts) >= 3:
            # Heavily censored (page prints 2 decimals): fit count growth in
            # log space instead of a flat logit series.
            ys = [math.log(c + 1.0) for c in counts]
            a_r, b_r, s_r, xbar_r, _, ybar_r = ols(use_x, ys)
            b_r = min(max(b_r, 0.0), MAX_SLOPE_C)
            c_pred = math.exp(ybar_r + b_r * (x_target - xbar_r)) - 1.0
            c_pred = min(max(c_pred, last_c), n_pred)
            m_trend = 100.0 * c_pred / n_pred if n_pred > 0 else last_rates[i]
            s_trend = s_r
        elif len(rates_recent) >= 3 and max(rates_recent) > min(rates_recent):
            if last_rates[i] > HIGH_RATE:
                # Non-earner fraction, averaged with the plain logit trend so
                # saturation is approached, not jumped at.
                ys_u = [math.log(max(100.0 - r, 0.05)) for r in rates_recent]
                a_u, b_u, s_u, xbar_u, _, ybar_u = ols(use_x, ys_u)
                b_u = min(b_u, 0.0)
                m_u = 100.0 - math.exp(ybar_u + b_u * (x_target - xbar_u))
                ys_t = [logit(r) for r in rates_recent]
                a_t, b_t, s_t, xbar_t, _, ybar_t = ols(use_x, ys_t)
                b_t = min(max(b_t, MIN_SLOPE_LOGIT), MAX_SLOPE_LOGIT)
                m_t = inv_logit(ybar_t + b_t * (x_target - xbar_t))
                m_trend = inv_logit(0.5 * logit(clamp(m_u)) + 0.5 * logit(clamp(m_t)))
                s_trend = max(s_u, s_t)
            else:
                ys = [logit(r) for r in rates_recent]
                a_t, b_t, s_t, xbar_t, _, ybar_t = ols(use_x, ys)
                b_t = min(max(b_t, MIN_SLOPE_LOGIT), MAX_SLOPE_LOGIT)
                m_trend = inv_logit(ybar_t + b_t * (x_target - xbar_t))
                s_trend = s_t
        else:
            m_trend = last_rates[i]
            s_trend = 0.0

        # Logit-space blend: trend leg leads (0.7); counts leg (0.3) only when valid.
        if counts_valid:
            med = inv_logit(0.7 * logit(clamp(m_trend)) + 0.3 * logit(clamp(m_counts)))
        else:
            med = m_trend
        medians.append(clamp(med))
        sig = max(base_sig, s_trend)
        sd = math.sqrt(sig ** 2 + (PRIOR_DOWN * prior_w) ** 2 + drift ** 2)
        su = math.sqrt(sig ** 2 + (PRIOR_UP * prior_w) ** 2 + drift ** 2)
        sig_down.append(max(sd, FLOOR_DOWN))
        sig_up.append(max(su, FLOOR_UP))

    # --- funnel constraints on medians ---
    # Story chain ids 3..7 (Chapter 1, 2, 3, 4, final mission): non-increasing.
    chain_idx = [i for i, t in enumerate(trophies) if t["id"] in (3, 4, 5, 6, 7)]
    if len(chain_idx) >= 2:
        fixed = pava_decreasing([medians[i] for i in chain_idx])
        for j, i in enumerate(chain_idx):
            medians[i] = fixed[j]
    # Platinum (id 1) and 100%-completion (id 14) cannot exceed any other trophy.
    others_min = min(m for j, m in enumerate(medians) if trophies[j]["id"] not in (1, 14))
    for i, t in enumerate(trophies):
        if t["id"] in (1, 14):
            medians[i] = min(medians[i], others_min)

    # --- build quantile rows ---
    rows = []
    for i, tr in enumerate(trophies):
        if last_rates[i] >= 99.995:
            rows.append({"id": tr["id"], "q": [99.9, 99.97, 99.99, 99.99, 99.99]})
            continue
        m = medians[i]
        lg = logit(m)
        qs = []
        for tau, z in zip(LEVELS, Z):
            s = sig_down[i] if z < 0 else (sig_up[i] if z > 0 else 0.0)
            qs.append(clamp(inv_logit(lg + z * s)))
        qs = [round4(q) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
