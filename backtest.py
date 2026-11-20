"""Backtest harness for gta6-trophy-rarity-forecast-01.

- Implements the exact scorer from competition.yaml (pinball loss on logit scale).
- Reconstructs practice answers from the yaml's stated aggregates:
  04:30Z read: 3,868 players tracked, 4,276 trophies earned, every rate but the
  100% one fell. Baseline practice loss ~1.51, typical miss ~0.68pp (yaml).
  R1 ("refresh-realistic"): new players (3,230) earned ~only the intro trophy
  (id 2, stays 100%); the remaining +103 earns spread pro-rata to current counts.
  R2 ("uniform dilution"): every non-100 rate x0.7477 (aggregate-consistent).
- Rolling within-history backtests (degenerate on flat practice data, recorded anyway).
"""
import json, math, subprocess, sys, os
from pathlib import Path

KIT = Path.home() / "workspace/makefun/kits/gta6-trophy-rarity-forecast-01/gta6-trophy-rarity-forecast-01-kit"
WORK = Path.home() / "workspace/makefun/gta6-trophy-forecast-01"

def clamp(x): return min(99.99, max(0.01, x))
def logit(pct): return math.log(clamp(pct) / (100 - clamp(pct)))

def pinball_loss(forecast, actual_rates):
    """forecast: {"trophies":[{"id":n,"q":[...]}]} ; actual_rates: list in trophy order."""
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = {r["id"]: r["q"] for r in forecast["trophies"]}
    tot, n = 0.0, 0
    miss, cov = 0.0, 0
    for i, y in enumerate(actual_rates):
        q = rows[i + 1]
        ly = logit(y)
        for tau, qv in zip(levels, q):
            d = ly - logit(qv)
            tot += 2 * max(tau * d, (tau - 1) * d)
            n += 1
        miss += abs(y - q[2])
        if q[0] <= y <= q[4]:
            cov += 1
    return tot / n, miss / len(actual_rates), cov / len(actual_rates)

def load_case(cid):
    return json.loads((KIT / f"input/{cid}.in").read_text())

def starter_forecast(case):
    """Replicate starter/main.py persistence baseline."""
    Z = [-1.645, -0.674, 0, 0.674, 1.645]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    last = case["history"][-1]
    rows = []
    for i, t in enumerate(case["trophies"]):
        r = last["rates"][i]
        lg = logit(r if r is not None else 0)
        rows.append({"id": t["id"], "q": [round(100 / (1 + math.exp(-(lg + z * spread))), 4) for z in Z]})
    return {"trophies": rows}

def reconstruct_answers():
    """Return {case_id: actual_rates_list} for p00 (target 04:30) and p01 (target 05:00, read unchanged)."""
    c = load_case("p00")
    last = c["history"][-1]
    rates = last["rates"]
    N_old, N_new = 638, 3868
    counts = [r / 100 * N_old for r in rates]
    total_old = 943
    # R1: trophy 2 (id 2, 100%) takes all 3230 new players; remaining +103 earns pro-rata to counts excl. trophy 2
    new_players = N_new - N_old
    total_new_earns = 4276 - total_old  # 3333
    e2 = new_players  # every new player did Gone Fishing
    rest = total_new_earns - e2  # 103
    base = sum(counts) - counts[1]
    new_counts = []
    for i, cnt in enumerate(counts):
        if i == 1:
            new_counts.append(N_new)  # 100%
        else:
            new_counts.append(cnt + rest * cnt / base)
    r1 = [round(100 * nc / N_new, 2) for nc in new_counts]
    # R2: uniform dilution x0.7477, trophy 2 stays 100
    r2 = [100.0 if i == 1 else round(r * 0.7477, 2) for i, r in enumerate(rates)]
    return {"R1": r1, "R2": r2}

if __name__ == "__main__":
    ans = reconstruct_answers()
    for name, actual in ans.items():
        print(f"--- reconstruction {name}: total earned check R1-only ---")
        for cid in ["p00", "p01"]:
            case = load_case(cid)
            bl = starter_forecast(case)
            loss, miss, cov = pinball_loss(bl, actual)
            print(f"  {cid} starter baseline: loss={loss:.4f} typical_miss={miss:.2f}pp coverage90={cov:.2f}")
        print(f"  R1 sample rates: id1={actual[0]} id3={actual[2]} id20={actual[19]} (last: 0.63/1.25/2.04)")
        print(f"  R1 implied total earned: {sum(a/100*3868 for a in actual):.0f} (target 4276)")

# ---------------- model evaluation ----------------
import importlib.util
def load_model():
    spec = importlib.util.spec_from_file_location("model_main", str(WORK / "main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def run_model_on_cases(model, case_ids, outdir):
    import shutil
    outdir = Path(outdir)
    if outdir.exists(): shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(KIT / "input"), MAKEFUN_OUTPUT_DIR=str(outdir))
    # run via subprocess so env vars apply cleanly
    for cid in case_ids:
        pass
    subprocess.run([sys.executable, str(WORK / "main.py")], env=env, check=True, cwd=str(WORK))
    return {cid: json.loads((outdir / f"{cid}.out").read_text()) for cid in case_ids}

def validate_output(fc, case):
    assert set(fc.keys()) == {"trophies"}, fc.keys()
    assert len(fc["trophies"]) == len(case["trophies"])
    ids = sorted(r["id"] for r in fc["trophies"])
    assert ids == sorted(t["id"] for t in case["trophies"]), ids
    for r in fc["trophies"]:
        assert set(r.keys()) == {"id", "q"}
        q = r["q"]
        assert len(q) == 5 and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 100 for v in q), q
        assert all(q[k] <= q[k+1] for k in range(4)), q
    return True

def evaluate(tag="v1"):
    model = load_model()
    print(f"===== backtest {tag} =====")
    ans = reconstruct_answers()
    outs = run_model_on_cases(model, ["p00", "p01"], WORK / "out")
    for cid in ["p00", "p01"]:
        case = load_case(cid)
        validate_output(outs[cid], case)
        for name, actual in ans.items():
            loss, miss, cov = pinball_loss(outs[cid], actual)
            bl = starter_forecast(case)
            bloss, bmiss, bcov = pinball_loss(bl, actual)
            print(f"  {cid} vs {name}: MODEL loss={loss:.4f} miss={miss:.2f}pp cov={cov:.2f} | "
                  f"STARTER loss={bloss:.4f} miss={bmiss:.2f}pp cov={bcov:.2f} | improvement={100*(bloss-loss)/bloss:.1f}%")
    # determinism: run twice, compare bytes
    b1 = (WORK / "out" / "p00.out").read_bytes()
    outs2 = run_model_on_cases(model, ["p00", "p01"], WORK / "out2")
    b2 = (WORK / "out2" / "p00.out").read_bytes()
    print(f"  determinism p00: {'IDENTICAL' if b1 == b2 else 'DIFFERENT!!!'}  size={len(b1)} bytes")
    # timing
    import time
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(KIT / "input"), MAKEFUN_OUTPUT_DIR=str(WORK / "out3"))
    t0 = time.time()
    subprocess.run([sys.executable, str(WORK / "main.py")], env=env, check=True, cwd=str(WORK))
    print(f"  runtime both cases: {time.time()-t0:.2f}s (cap 8s/case)")
    return outs

def rolling_backtest(model):
    """Within-history: truncate history at 01:00, forecast 02:30 actual (flat data)."""
    print("===== rolling within-history backtest (flat data, expected degenerate) =====")
    for cid, cut_at, tgt_at in [("p00", "2026-11-19T01:00:00Z", "2026-11-19T02:30:00Z"),
                                ("p01", "2026-11-19T01:30:00Z", "2026-11-19T03:00:00Z")]:
        case = json.loads((KIT / f"input/{cid}.in").read_text())
        hist = [h for h in case["history"] if h["slot"] <= cut_at]
        actual_read = [h for h in case["history"] if h["slot"] == tgt_at][0]
        sub = dict(case)
        sub["history"] = hist
        sub["at"] = cut_at
        sub["target"] = tgt_at
        sub["horizon_minutes"] = 90
        fc = model.forecast_case(sub, cid)
        bl = starter_forecast(sub)
        print(f"  {cid} cut {cut_at} -> {tgt_at}: MODEL loss={pinball_loss(fc, actual_read['rates'])[0]:.4f} "
              f"STARTER loss={pinball_loss(bl, actual_read['rates'])[0]:.4f}")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "eval":
    evaluate()
    rolling_backtest(load_model())
