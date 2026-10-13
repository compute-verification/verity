"""η over `serve.py`'s runs (`serve-<pool>.json`), the leaderboard's rows: each run's η and its 1 s windows' mean, its
TPOT against the same engine's `none` run, its token gate, and η before filler by phase (`analyze.shortfall`), and
after it where the run placed its filler (`filler_by_phase`).
Torch-free; writes `eta.json`.

    python eta.py --serve serve-large.json [...] --out eta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze  # noqa: E402


def rows(serve: dict, R: float) -> list[dict]:
    base = next((r for r in serve["runs"] if r["policy"] == "none"), None)
    out = []
    for r in serve["runs"]:
        steps = analyze.phases(r["step_records"])
        sh = analyze.shortfall(steps, r["seconds"], serve["weight_bytes_per_step"], r["filler_bytes"], R,
                               gpu_steps=r.get("gpu_steps"))
        out.append({
            "policy": r["policy"], "mode": r["mode"], "token_gate": r.get("token_gate"), "eta": sh["eta"],
            "windows_mean": next((w["mean"] for w in r["windows"] if w["width_s"] == 1.0), None),
            "tpot_ms": r["tpot_ms_median"], "decode_step_ms": r["decode_step_ms_median"],
            "tpot_x": r["tpot_ms_median"] / base["tpot_ms_median"] if base else None,
            "filler": sh["filler"], "before_filler": {k: sh[k]["eta"] for k in ("decode", "prefill", "empty", "gaps", "idle")},
        })
        fp = r.get("filler_by_phase")
        if fp:
            cap = R * 1e9 * r["seconds"]
            bf = out[-1]["before_filler"]
            out[-1]["after_filler"] = {"decode": bf["decode"] - fp["decode"] / cap,
                                       "prefill": bf["prefill"] - fp["prefill"] / cap,
                                       "between": bf["gaps"] + bf["idle"] + bf["empty"] - fp["between"] / cap}
        fg, g = r.get("filler_by_gpu_phase"), sh.get("gpu")
        if fg and g:
            cap = R * 1e9 * r["seconds"]
            out[-1]["after_filler_gpu"] = {"decode": g["decode"]["eta"] + g["empty"]["eta"] - fg["decode"] / cap,
                                           "prefill": g["prefill"]["eta"] - fg["prefill"] / cap,
                                           "idle": g["eta_idle"] - fg["between"] / cap}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--R", type=float, default=None, help="GB/s (default: each serve file's own)")
    a = ap.parse_args()
    res = {}
    for p in a.serve:
        s = json.loads(p.read_text())
        res[p.name] = rows(s, a.R or s["args"]["R"])
        for r in res[p.name]:
            b = r["before_filler"]
            print(f"{p.name} {r['policy']:>12}/{r['mode']:<6} η {r['eta']:.3f} (1 s windows {1 - (r['windows_mean'] or 0):.3f})"
                  f"  TPOT {r['tpot_ms']:.2f} ms ×{r['tpot_x'] or 0:.2f}  gate {r['token_gate']}  filler {r['filler']:.3f}"
                  f"  before: decode {b['decode']:.3f} prefill {b['prefill']:.3f} gaps {b['gaps']:.3f}"
                  f" idle {b['idle']:.3f} empty {b['empty']:.3f}" + (
                      "  after: " + " ".join(f"{k} {v:.3f}" for k, v in r["after_filler"].items())
                      if "after_filler" in r else "") + (
                      "  after, GPU clock: " + " ".join(f"{k} {v:.3f}" for k, v in r["after_filler_gpu"].items())
                      if "after_filler_gpu" in r else ""))
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
