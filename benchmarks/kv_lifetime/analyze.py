"""`serve.py`'s events and steps as the frontier's inputs: for each phase, `lifetimes.analyze` over the phase after its
warm-up, at carrier stocks on a grid of the pool (`STOCKS`, shares of `pool_bytes`), and the step model's six
quantities (T, X, Q, K, n, n0) summed over 1 s windows, each step's spread over its span, so any linear units model
gives its units a window (`frontier.py`), and the token rows a second of the steps with at least `PREFILL_ROWS` rows
(the rate a recompute from fresh inputs re-prefills at). Writes analysis.json beside the inputs.

    python analyze.py OUT_DIR [--warmup S] [--phases I,J] [--delays 0,10,30] [--time-scale S --json PATH]

`--time-scale S` stretches the run's clock by S, for a slower model on the same schedule (70B TP8 from an 8B run:
with `frontier.py --kv-scale`); the warm-up is on the stretched clock.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lifetimes  # noqa: E402

STOCKS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
MODEL = ("T", "X", "Q", "K", "n", "n0")
PREFILL_ROWS = 2048  # a step of at least this many token rows is mostly prefill: its rows a second are the re-prefill rate


def binned(steps: dict[str, np.ndarray], t0: float, t1: float, width: float = 1.0) -> dict[str, list[float]]:
    """Each quantity a window of `width` seconds over [t0, t1), a step's spread pro rata over its span."""
    n = max(1, int(np.ceil((t1 - t0) / width)))
    out = {k: np.zeros(n) for k in MODEL}
    for i in range(len(steps["t0"])):
        a, b = steps["t0"][i], steps["t1"][i]
        if b <= t0 or a >= t1:
            continue
        lo, hi = int((max(a, t0) - t0) // width), int((min(b, t1) - t0) // width)
        for w in range(lo, min(n - 1, hi) + 1):
            ov = min(b, t0 + (w + 1) * width) - max(a, t0 + w * width)
            frac = ov / (b - a) if b > a else 1.0
            if ov > 0 or b <= a:
                for k in MODEL:
                    out[k][w] += steps[k][i] * frac
    return {k: v.tolist() for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--warmup", type=float, default=120.0)
    ap.add_argument("--phases", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--delays", type=lambda s: [float(x) for x in s.split(",")], default=[0.0, 10.0, 30.0],
                    help="append delays (s) to compute the frontier at")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="stretch every time by this factor (a slower model on the run's schedule)")
    ap.add_argument("--json", type=Path, default=None, help="where to write (default OUT_DIR/analysis.json)")
    a = ap.parse_args()
    serve = json.loads((a.out / "serve.json").read_text())
    z = np.load(a.out / "events.npz")
    ev = {k: z[k] for k in ("t", "kind", "block", "origin") if k in z.files}
    steps = lifetimes.steps_arrays(z["steps"])
    s = a.time_scale
    if s != 1.0:
        for k in ("t", "origin"):
            if k in ev:
                ev[k] = ev[k] * s
        steps["t0"], steps["t1"] = steps["t0"] * s, steps["t1"] * s
        for ph in serve["phases"]:
            ph["t0"], ph["t1"] = ph["t0"] * s, ph["t1"] * s
    pool = serve["pool_bytes"]
    res = {"format": "verity-kv-lifetime/analysis/v0", "serve": {k: v for k, v in serve.items() if k != "phases"},
           "stocks_pool_share": list(STOCKS), "time_scale": s, "phases": []}
    for pi, ph in enumerate(serve["phases"]):
        if a.phases is not None and pi not in a.phases:
            continue
        t0, t1 = ph["t0"], ph["t1"]
        warm = min(a.warmup, (t1 - t0) / 3)
        sel = (ev["t"] >= t0) & (ev["t"] <= t1)
        sub = {k: v[sel] for k, v in ev.items()}
        r = lifetimes.analyze(sub, steps, t1, serve["block_bytes"], t_start=t0 + warm,
                              stocks=[s * pool for s in STOCKS], delays=a.delays)
        busy = (steps["t0"] >= t0 + warm) & (steps["t1"] <= t1)
        r["serve"] = ph
        r["warmup_s"] = warm
        r["model_windows_1s"] = binned(steps, t0 + warm, t1)
        r["model_totals"] = {k: float(steps[k][busy].sum()) for k in MODEL}
        r["busy_s"] = float((steps["t1"][busy] - steps["t0"][busy]).sum())
        big = busy & (steps["T"] >= PREFILL_ROWS)
        r["prefill_rows_per_s"] = float(steps["T"][big].sum()) / max(
            1e-12, float((steps["t1"][big] - steps["t0"][big]).sum())) if big.any() else None
        res["phases"].append(r)
        print(f"phase {pi} rate {ph['rate']}: held mean {r['occupancy']['held_bytes_mean'] / 1e9:.2f} GB of "
              f"{pool / 1e9:.1f}, held life p50 {r['held_life_s']['quantiles_byte_weighted']['0.5']:.1f} s p90 "
              f"{r['held_life_s']['quantiles_byte_weighted']['0.9']:.1f} s, stock older than 60 s "
              f"{r['stock_age_tail']['60']:.3f}, written {r['written_bytes_per_s'] / 1e9:.2f} GB/s", flush=True)
    (a.json or a.out / "analysis.json").write_text(json.dumps(res) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
