"""Round-2 tuning: grid over downside/upside spread priors, scored on R1 (validated)
and R2 (mild-dilution stress). Pick a robust middle, not the argmin."""
import json, math, importlib.util, sys
sys.path.insert(0, ".")
import backtest as bt

spec = importlib.util.spec_from_file_location("mm", "main.py")
mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
ans = bt.reconstruct_answers()
cases = {cid: bt.load_case(cid) for cid in ["p00", "p01"]}

print("sd\\su | R1 loss (p00) | R2 loss (p00) | R1+R2 mean")
best = None
for sd in [0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8]:
    row = []
    for su in [0.2, 0.35, 0.5]:
        mm.PRIOR_DOWN, mm.PRIOR_UP = sd, su
        fc = mm.forecast_case(cases["p00"], "p00")
        l1 = bt.pinball_loss(fc, ans["R1"])[0]
        l2 = bt.pinball_loss(fc, ans["R2"])[0]
        row.append((su, l1, l2))
        print(f"  sd={sd:.1f} su={su:.2f}: R1={l1:.4f} R2={l2:.4f} mean={(l1+l2)/2:.4f}")
    print()
