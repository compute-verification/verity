#!/usr/bin/env python3
"""A timed window's numbers from its OUT (timed.sh): every timed row's median over the harness's plain-FP8 divisor and over
the run's own interleaved plain GEMM, the spread of its reps and of their baselines, the forming's price (rates.py), the
pinning and the per-core load by band.

The divisors are the harness's plain FP8 GEMM: 1.446 ms at 8,192³ (p) and 0.052 ms at m = 32 (d).  The harness has none
for l (Llama-3.1-8B's down_proj), so l's is p's scaled by MACs (1.446 x 8,192·4,096·14,336 / 8,192³), marked as such.

  python3 summary.py OUT [--json]
"""

from __future__ import annotations

import json
import os
import statistics
import sys

DIVISOR = {"p": 1.446, "d": 0.052, "l": 1.446 * (8192 * 4096 * 14336) / 8192 ** 3}
SCALED = {"l"}
BANDS = {"0-47": (0, 48), "48-91": (48, 92), "92-95": (92, 96), "96-123": (96, 124), "124-191": (124, 192)}


def spread(reps: list[float]) -> float:
    m = statistics.median(reps)
    return (max(reps) - min(reps)) / m if m else float("nan")


def row(kind: str, name: str, t: dict) -> dict:
    return {"kind": kind, "arm": name, "ms": round(t["ms"], 4), "x_divisor": round(t["ms"] / DIVISOR[kind], 2),
            "x_own_plain": round(t["ms"] / t["base_ms"], 2), "base_ms": round(t["base_ms"], 4),
            "spread": round(spread(t["reps"]), 4), "base_spread": round(spread(t["base_reps"]), 4),
            "sm_mhz": sorted({*t.get("sm_mhz_arm", []), *t.get("sm_mhz_base", [])})}


def gemm_rows(path: str, label: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    for sh in json.load(open(path))["shapes"]:
        for a in sh["arms"]:
            if a["arm"] != "control" and "reps" in a and a["arm"] in ("plain", "core", "chash", "hash_ir"):
                out.append({**row(sh["kind"], f"{label}{a['arm']}:S{a['S']}:K{a['K']}", a)})
    return out


def form_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    for sh in json.load(open(path))["shapes"]:
        for t in sh["timing"]:
            out.append({**row(sh["kind"], t["arm"] + (f":S{t['S']}" if "S" in t else ""), t), "xof": sh["xof"]})
    return out


def breakdown(res: dict, S: int = 4) -> list[dict]:
    """The bound unit's time split along its arms: plain GEMM at k, the stacked 3k, the chain, the hashing, the forming."""
    out = []
    for kind in sorted({r["kind"] for r in res["gemm_mix1"]}):
        g = {r["arm"]: r["ms"] for r in res["gemm_mix1"] if r["kind"] == kind}
        plain = sorted((int(a.rsplit(":K", 1)[1]), v) for a, v in g.items() if a.startswith("plain:"))
        core = next((v for a, v in g.items() if a.startswith("core:")), None)
        chash = next((v for a, v in g.items() if a.startswith(f"chash:S{S}:")), None)
        if len(plain) < 2 or core is None or chash is None:
            continue
        for xof in sorted({r["xof"] for r in res["unit"] if r["kind"] == kind}):
            u = {r["arm"]: r["ms"] for r in res["unit"] if r["kind"] == kind and r["xof"] == xof}
            unit = u.get(f"unit_chash:S{S}")
            if unit is None:
                continue
            parts = {"plain_k": plain[0][1], "stacked_3k": plain[-1][1] - plain[0][1], "chain": core - plain[-1][1],
                     "hashing": chash - core, "forming": unit - chash}
            out.append({"kind": kind, "xof": xof, "S": S, "unit_ms": unit, "x_divisor": round(unit / DIVISOR[kind], 2),
                        "ms": {k: round(v, 4) for k, v in parts.items()},
                        "share": {k: round(v / unit, 3) for k, v in parts.items()}, "form_alone_ms": u.get("form")})
    return out


def load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    lines = [json.loads(x) for x in open(path)]
    head, rows = lines[0], [x for x in lines[1:] if "busy" in x]
    res = {"pinning": head.get("pinning"), "samples": len(rows), "interval_s": head.get("interval_s")}
    for name, (a, b) in BANDS.items():
        v = [statistics.mean(x["busy"][a:b]) for x in rows]
        res[name] = {"mean": round(statistics.mean(v), 3), "max": round(max(v), 3)} if v else None
    return res


def main(argv: list[str]) -> int:
    out = argv[1]
    res = {"pinning": json.load(open(os.path.join(out, "pinning.json"))) if os.path.exists(os.path.join(out, "pinning.json")) else None,
           "gemm_mix1": gemm_rows(os.path.join(out, "arms.json"), ""),
           "gemm_mix0": gemm_rows(os.path.join(out, "mix0", "arms.json"), "mix0 "),
           "unit": form_rows(os.path.join(out, "form", "arms.json")),
           "load": load(os.path.join(out, "coreload.jsonl"))}
    rates = os.path.join(out, "rates", "rates.json")
    if os.path.exists(rates):
        r = json.load(open(rates))
        res["rates"] = {"floor": {k: v["floor"] for k, v in r["floor"].items()}, "achieved": r["achieved"]}
    res["breakdown"] = breakdown(res)
    if "--json" in argv:
        print(json.dumps(res, indent=1))
        return 0
    print("pinning", res["pinning"], "\nload", json.dumps(res["load"]), "\nrates", json.dumps(res.get("rates")))
    print(f"{'shape':5s} {'xof':6s} {'arm':34s} {'ms':>9s} {'x div':>7s} {'x own':>7s} {'spread':>7s} {'base sp':>7s}")
    for r in res["gemm_mix1"] + res["gemm_mix0"] + res["unit"]:
        mark = "*" if r["kind"] in SCALED else " "
        print(f"{r['kind']:5s} {r.get('xof', '-'):6s} {r['arm']:34s} {r['ms']:9.4f} {r['x_divisor']:6.2f}{mark} "
              f"{r['x_own_plain']:7.2f} {r['spread']:7.3f} {r['base_spread']:7.3f}")
    for b in res["breakdown"]:
        print(f"breakdown {b['kind']} {b['xof']:6s} unit chash S={b['S']} {b['unit_ms']:.4f} ms ({b['x_divisor']}x): "
              + ", ".join(f"{k} {v:.4f} ({b['share'][k]:.0%})" for k, v in b["ms"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
