"""Prices the page conversion over each serving run of `serve.py` (`serve-<pool>.json`) with the encode cells of
`bench.py` (`bench.json`, at each k it measured), and writes `conversion.json`. Torch-free.

Two arms, each from `lag.py`:
  fused   the encode in the KV write: no lag; a prefill step adds its tokens' encode at the alone rate, a decode step
          at the rate per-step encode at grid 128 costs it (the encode's bytes over the time it added, at 64 × 2048)
  lazy-G  the encode on a least-priority stream, held to grid G beside decode (the backlog's measured rate and
          slowdown at 64 × 2048), none beside prefill, the alone rate in idle time; until a token is encoded, each
          step reading it reads plain bytes, γ_lag per window = those bytes over R·T

    python analyze.py --bench bench.json --serve serve-large.json serve-sized.json --out conversion.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lag  # noqa: E402

DECODE_CELL = (64, 2048)


def arms(enc: dict, grids=(64, 128)) -> dict:
    cells = enc["beside_decode"]

    def one(form, grid):
        return next(r for r in cells if r["form"] == form and r["grid"] == grid
                    and (r["batch"], r["ctx"]) == DECODE_CELL)

    per = one("per-step tokens", 128)
    added_ms = per["step_ms_beside"] - per["step_ms_alone"]
    out = {"fused": {"alone_GBps": enc["alone_GBps"],
                     "decode_GBps": DECODE_CELL[0] * lag.KV_TOKEN_BYTES / (added_ms / 1e3) / 1e9}}
    for g in grids:
        b = one("backlog", g)
        out[f"lazy-{g}"] = lag.Rates(b["encode_GBps"], 0.0, enc["alone_GBps"], b["slowdown"], 0.0)
    return out


def phases(steps: list[dict]) -> list[dict]:
    """A step's prefill tokens as its tokens beyond one a request: a decode request schedules one. Records from before
    serve.py counted it so (vLLM's batch queue puts `num_computed_tokens` a step ahead) miss most prefill chunks."""
    return [{**s, "prefill_tokens": max(s["prefill_tokens"], s["tokens"] - s["reqs"])} for s in steps]


def shortfall(steps: list[dict], seconds: float, weight_bytes: int, filler_bytes: int, R: float,
              token_bytes: int = lag.KV_TOKEN_BYTES) -> dict:
    """η, the share of R·T over [0, seconds) that no registered read covers, by where it falls: each phase's seconds,
    its steps' own fraction of R (the weights and each scheduled request's context, as `serve.py` counts a window),
    and its share of η before filler (decode, prefill, empty steps that schedule nothing; gaps, the engine's host time
    between two steps that both schedule requests; idle, the rest of the time between steps). The filler the run's
    top-up read comes off the total, wherever it fell."""
    cap = R * 1e9 * seconds
    out, busy = {}, 0.0
    for name, keep in (("decode", lambda s: s["tokens"] > 0 and s["prefill_tokens"] == 0),
                       ("prefill", lambda s: s["prefill_tokens"] > 0), ("empty", lambda s: s["tokens"] == 0)):
        sel = [s for s in steps if keep(s)]
        t = sum(s["t1"] - s["t0"] for s in sel)
        reads = sum(weight_bytes + s["kv_tokens"] * token_bytes for s in sel if s["tokens"] > 0)
        busy += t
        out[name] = {"s": t, "of_R": reads / (R * 1e9 * t) if t else 0.0, "eta": (R * 1e9 * t - reads) / cap}
    gaps = sum(max(0.0, b["t0"] - a["t1"]) for a, b in zip(steps, steps[1:]) if a["reqs"] > 0 and b["reqs"] > 0)
    idle = max(0.0, seconds - busy - gaps)
    out["gaps"] = {"s": gaps, "of_R": 0.0, "eta": gaps / seconds}
    out["idle"] = {"s": idle, "of_R": 0.0, "eta": idle / seconds}
    out["eta_before_filler"] = sum(v["eta"] for v in out.values())
    out["filler"] = filler_bytes / cap
    out["eta"] = out["eta_before_filler"] - out["filler"]
    return out


def price(run: dict, a: dict, R: float) -> dict:
    steps = phases(run["step_records"])
    t1 = run["seconds"]
    res = {}
    for name, p in a.items():
        if name == "fused":
            f = lag.fused(steps, p["alone_GBps"], p["decode_GBps"])
            res[name] = {"overhead": f["overhead"], "added_s": f["added_s"], "gamma_lag_mean": 0.0,
                         "gamma_lag_max_1s": 0.0}
            continue
        r = lag.replay(steps, p)
        out = {"overhead": r["added_s"] / r["busy_s"] if r["busy_s"] else 0.0, "added_s": r["added_s"],
               "queue_peak_GB": r["queue_peak_bytes"] / 1e9, "lag_s_max": r["lag_s_max"],
               "plain_GB": r["plain_total"] / 1e9}
        for w in (1.0, 5.0):
            g = lag.windows(steps, r["plain_bytes"], 0.0, t1, w, R)
            out[f"gamma_lag_max_{w:g}s"] = max(g, default=0.0)
            out[f"gamma_lag_mean_{w:g}s"] = statistics.fmean(g) if g else 0.0
        out["gamma_lag_mean"] = out["gamma_lag_mean_1s"]
        res[name] = out
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--serve", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--R", type=float, default=None)
    a = ap.parse_args()
    bench = json.loads(a.bench.read_text())
    R = a.R or bench["read"]["R_GBps"]
    by_k = {e["k"]: arms(e) for e in bench["encode"].values()}
    out = {"format": "verity-kv-filler/conversion/v0", "R_GBps": R,
           "arms": {k: {n: (vars(p) if isinstance(p, lag.Rates) else p) for n, p in v.items()} for k, v in by_k.items()},
           "runs": []}
    for path in a.serve:
        s = json.loads(path.read_text())
        pool = s["args"]["pool"]
        for run in s["runs"]:
            eta = shortfall(phases(run["step_records"]), run["seconds"], s["weight_bytes_per_step"],
                            run["filler_bytes"], R)
            print(f"{pool:5s} {run['policy']:8s} {run['mode']:5s} η {eta['eta']:.3f} (before filler "
                  f"{eta['eta_before_filler']:.3f}: " + ", ".join(
                      f"{n} {eta[n]['s']:.1f} s at {eta[n]['of_R']:.3f} R, {eta[n]['eta']:.3f}"
                      for n in ("decode", "prefill", "empty", "gaps", "idle")) + f"; filler {eta['filler']:.3f})")
            for k, ar in by_k.items():
                row = {"pool": pool, "policy": run["policy"], "mode": run["mode"], "k": k, "shortfall": eta,
                       "arms": price(run, ar, R)}
                out["runs"].append(row)
                print(f"{pool:5s} {run['policy']:8s} {run['mode']:5s} k={k}: " + "; ".join(
                    f"{n} +{v['overhead']:.1%} γ_lag {v['gamma_lag_mean']:.3%} (1 s max {v['gamma_lag_max_1s']:.3%})"
                    for n, v in row["arms"].items()))
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
