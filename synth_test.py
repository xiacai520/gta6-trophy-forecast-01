"""Synthetic long-horizon validation (round 2, attempt 2).

Simulates a GTA VI launch with cohort dynamics: players join on a launch
surge, sync to Exophase after a delay, progress through story stages at
individual speeds; Exophase refreshes the page every ~4h (frozen between).
Cuts a final-like case at 2026-11-21T11:30Z (horizon 6,510 min to
2026-11-26T00:00Z) and scores this model vs the starter persistence
baseline and a faithful re-implementation of the house curve baseline.
Seeded RNG: deterministic.
"""
import json, math, random, subprocess, sys, os
from datetime import datetime, timedelta, timezone
sys.path.insert(0, ".")
import backtest as bt

random.seed(20261120)
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)

# Trophy stage times in nominal play-hours (from the practice trophy list).
STAGES = {1: 220, 2: 0.5, 3: 4, 4: 10, 5: 18, 6: 28, 7: 45, 8: 12, 9: 20, 10: 14,
          11: 24, 12: 16, 13: 30, 14: 260, 15: 120, 16: 8, 17: 90, 18: 70, 19: 60,
          20: 6, 21: 5, 22: 40, 23: 42, 24: 38, 25: 55, 26: 44, 27: 36, 28: 50,
          29: 26, 30: 100, 31: 130, 32: 80, 33: 22, 34: 18, 35: 48, 36: 15, 37: 9,
          38: 25, 39: 19, 40: 75, 41: 17, 42: 65, 43: 58, 44: 3, 45: 11, 46: 13,
          47: 33, 48: 30, 49: 30, 50: 30, 51: 7, 52: 85, 53: 95, 54: 110, 55: 35}

J_INF = 300_000  # total eventual joiners (Exophase-tracked universe proxy)

def simulate(v_med=0.35):
    """Return (reads, truth_rates): reads = list of dicts {slot, at, players, rates} every 30 min;
    truth = rates at target as of last refresh before target."""
    # Player join times: cumulative J(t) = J_INF*(1-exp(-t/48h)); sample via inverse CDF.
    players = []
    for _ in range(J_INF):
        u = random.random()
        join_h = -48.0 * math.log(1 - u * (1 - math.exp(-(7 * 24) / 48.0)))
        sync_delay = random.expovariate(1 / 10.0)  # hours
        v = random.lognormvariate(math.log(v_med), 0.6)
        players.append((join_h, join_h + sync_delay, v))
    players.sort()
    # Precompute per-trophy SORTED earn wall-times, and sorted sync times.
    # (earn time of trophy i for a player = join_h + STAGES[i]/v; a player only
    #  counts once synced, so at time t the earners are those with sync<=t AND earn<=t.)
    import bisect as _b
    sync_times = sorted(s for _, s, _ in players)
    earn_sorted = []
    for i in range(1, 56):
        earn_sorted.append(sorted(j + STAGES[i] / v for j, s, v in players))
    # For correct counts we need players with sync<=t AND earn<=t; approximate
    # with earn<=t among synced by using joint order: build per-trophy list of
    # (max(sync, earn)) times -- a player counts once both have happened.
    both_sorted = []
    for i in range(1, 56):
        both_sorted.append(sorted(max(s, j + STAGES[i] / v) for j, s, v in players))
    # Walk time in 30-min steps; page refreshes every 4h.
    reads = []
    page = None  # frozen page values
    truth = None
    t = 0.0  # hours since release
    end_h = 7 * 24
    import bisect
    while t <= end_h:
        now = RELEASE + timedelta(hours=t)
        # refresh at multiples of 4h (and at t=0)
        if abs(t % 4.0) < 1e-9 or page is None:
            n_tr = bisect.bisect_right(sync_times, t)
            if n_tr > 0:
                rates = []
                for i in range(55):
                    c = bisect.bisect_right(both_sorted[i], t)
                    rates.append(round(100.0 * c / n_tr, 2))
                page = (n_tr, rates)
        if page is not None:
            reads.append({"slot": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "at": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                          "players_tracked": page[0], "trophies_earned": None,
                          "rates": list(page[1])})
        if now >= TARGET and truth is None:
            truth = list(page[1])
        t += 0.5
    return reads, truth

def house_curve_forecast(case):
    """Faithful re-implementation of baseline-trend/main.ts (median + combined spread)."""
    import math as m
    Z = [-1.645, -0.674, 0, 0.674, 1.645]
    d = case["horizon_minutes"] / 1440
    last24 = [h for h in case["history"]
              if (datetime.fromisoformat(case["at"].replace("Z", "+00:00")) -
                  datetime.fromisoformat(h["at"].replace("Z", "+00:00"))).total_seconds() <= 24 * 3600]
    rows = []
    for i, tr in enumerate(case["trophies"]):
        xs, ys = [], []
        for h in last24:
            r = h["rates"][i]
            if r is None: continue
            hrs = (datetime.fromisoformat(h["at"].replace("Z", "+00:00")) - RELEASE).total_seconds() / 3600
            xs.append(m.log1p(hrs)); ys.append(bt.logit(r))
        if len(xs) >= 3:
            n = len(xs); xb = sum(xs) / n; yb = sum(ys) / n
            sxx = sum((x - xb) ** 2 for x in xs)
            b = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0
            a = yb - b * xb
            resid = m.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)) if n > 2 else 0
            xt = m.log1p((datetime.fromisoformat(case["target"].replace("Z", "+00:00")) - RELEASE).total_seconds() / 3600)
            med = a + b * xt
            sig = m.sqrt((0.5 * m.sqrt(d)) ** 2 + resid ** 2)
        else:
            med = bt.logit(case["history"][-1]["rates"][i])
            sig = 0.5 * m.sqrt(d)
        rows.append({"id": tr["id"], "q": [round(100 / (1 + m.exp(-(med + z * sig))), 4) for z in Z]})
    return {"trophies": rows}

def run_one(v_med, tag):
    print(f"--- sim {tag}: v_med={v_med} ---")
    random.seed(20261120)
    reads, truth = simulate(v_med)
    print(f"reads: {len(reads)}  N at deadline: {[r['players_tracked'] for r in reads if r['slot']=='2026-11-21T11:30:00Z']}")
    print(f"N at target: {[r['players_tracked'] for r in reads if r['slot']=='2026-11-26T00:00:00Z']}  truth ch1/final/plat: {truth[2]:.2f}/{truth[6]:.2f}/{truth[0]:.2f}")
    case = json.loads(open("/home/hatch/workspace/makefun/kits/gta6-trophy-rarity-forecast-01/gta6-trophy-rarity-forecast-01-kit/input/p00.in").read())
    hist = [r for r in reads if r["slot"] < "2026-11-21T11:30:00Z"]
    case = dict(case, id="f00", at="2026-11-21T11:30:00Z", horizon_minutes=6510,
                target="2026-11-26T00:00:00Z", history=hist)
    for r in case["history"]:
        r["trophies_earned"] = sum(1 for x in r["rates"] if x and x > 0)
    d = "/tmp/synthcase"; os.makedirs(d + "/in", exist_ok=True); os.makedirs(d + "/out", exist_ok=True)
    json.dump(case, open(d + "/in/f00.in", "w"))
    env = dict(os.environ, MAKEFUN_INPUT_DIR=d + "/in", MAKEFUN_OUTPUT_DIR=d + "/out")
    subprocess.run([sys.executable, "main.py"], env=env, check=True)
    mine = json.loads(open(d + "/out/f00.out").read())
    st = bt.starter_forecast(case)
    cu = house_curve_forecast(case)
    for name, fc in [("MODEL", mine), ("STARTER-persistence", st), ("HOUSE-curve", cu)]:
        loss, miss, cov = bt.pinball_loss(fc, truth)
        print(f"  {name}: loss={loss:.4f} typical_miss={miss:.2f}pp coverage90={cov:.2f}")
    print(f"  model medians: ch1={mine['trophies'][2]['q'][2]} (truth {truth[2]}), final={mine['trophies'][6]['q'][2]} (truth {truth[6]}), plat={mine['trophies'][0]['q'][2]} (truth {truth[0]})")
    print(f"  starter medians: ch1={st['trophies'][2]['q'][2]}, final={st['trophies'][6]['q'][2]}, plat={st['trophies'][0]['q'][2]}")

if __name__ == "__main__":
    run_one(0.35, "fast/hardcore")
    run_one(0.12, "slow/casual")
