"""Ablation: funnel constraints on/off, scored on both synthetic worlds."""
import json, math, random, subprocess, sys, os
sys.path.insert(0, ".")
import backtest as bt
import synth_test as st
import importlib.util

def load_main(patch_funnel=False):
    src = open("main.py").read()
    if patch_funnel:
        src = src.replace("fixed = pava_decreasing([medians[i] for i in chain_idx])",
                          "fixed = [medians[i] for i in chain_idx]")
        src = src.replace('''    for i, t in enumerate(trophies):
        if t["id"] in (1, 14):
            medians[i] = min(medians[i], others_min)''',
'''    for i, t in enumerate(trophies):
        if t["id"] in (1, 14):
            pass''')
    ns = {"__file__": "main.py", "__name__": "m"}
    exec(compile(src, "main_abl", "exec"), ns)
    return ns["forecast_case"]

def run_world(v_med, fc_fn):
    random.seed(20261120)
    reads, truth = st.simulate(v_med)
    case = json.loads(open("/home/hatch/workspace/makefun/kits/gta6-trophy-rarity-forecast-01/gta6-trophy-rarity-forecast-01-kit/input/p00.in").read())
    hist = [r for r in reads if r["slot"] < "2026-11-21T11:30:00Z"]
    case = dict(case, id="f00", at="2026-11-21T11:30:00Z", horizon_minutes=6510, target="2026-11-26T00:00:00Z", history=hist)
    fc = fc_fn(case, "f00")
    return bt.pinball_loss(fc, truth)

for tag, patch in [("funnel ON", False), ("funnel OFF", True)]:
    fc_fn = load_main(patch)
    l_fast = run_world(0.35, fc_fn)
    l_slow = run_world(0.12, fc_fn)
    print(f"{tag}: fast loss={l_fast[0]:.4f} | slow loss={l_slow[0]:.4f}")
