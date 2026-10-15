#!/usr/bin/env python3
"""Round 2's tables (CMP-16) from a lease_fast.sh run's out/ directory: the gates (run.py's fixture with the fast and the
reference kernels, eta.py's `same`), the SHA-512 kernels by launch-bound variant (`sha2`), η of the served call with the
reference and the fast kernels and the what-if schedules (`time`), and the power holds with each schedule's energy a call
and the 600 W energy bound T >= E_dyn / (600 W - P_idle).

  python3 analyze2.py <run dir>/out [P_IDLE_W]   (P_IDLE_W: the board at the lock with nothing running; default 84)
"""
from __future__ import annotations

import json
import os
import sys

LIMIT_W = 600.0


def rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.startswith("{")]


def gates(out: str) -> None:
    print("## gates\n")
    for sha in ("fast", "ref"):
        for r in rows(f"{out}/run_py_{sha}.jsonl"):
            if r["kind"] == "check":
                v = r["variants"]
                print(f"- run.py fixture ({sha}): " + "; ".join(f"{k} {x['buffers']} buffers, failed {x['failed'] or 'none'}"
                                                               for k, x in v.items()))
    for r in rows(f"{out}/eta_same.jsonl"):
        if r["kind"] == "same":
            bad = [f"{st}:{nm}" for st in ("steps", "graph") for nm, ok in r[st].items() if not ok]
            print(f"- same {'x'.join(map(str, r['shape']))} (tiles {r['tiles']}): identical={r['identical']} "
                  f"{'differ: ' + ', '.join(bad) if bad else ''} rowdig {r['sha256']['rowdig_a']} leaves {r['sha256']['leaves']}")
    print()


def sha2(out: str) -> None:
    rs = [r for r in rows(f"{out}/eta_sha2.jsonl") if r["kind"] == "sha2"]
    if not rs:
        return
    print("## SHA-512 kernels by variant (ms; compressions/s: the format's count, and those executed)\n")
    print("| m | unit | variant | same | segs ms | folds ms | both ms | format /s | executed /s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        print(f"| {r['shape'][0]} | {r['unit']} | {r['variant']} | {r['same']} | {r.get('segs_ms', '')} | {r.get('folds_ms', '')} | "
              f"{r['both_ms']} | {r['format_per_s']:.3g} | {r['executed_per_s']:.3g} |")
    print()
    regs = {}
    for f in sorted(os.listdir(out)):
        if f.startswith("probe_ptxas_"):
            v, cur = f[len("probe_ptxas_"):-4], None
            for line in open(f"{out}/{f}"):
                if "Compiling entry function" in line:
                    cur = line.split("'")[1]
                elif cur in ("s512_row_segs", "s512_tile_segs", "s512_folds") and "Used" in line and "registers" in line:
                    regs.setdefault(v, {})[cur] = line.split("Used")[1].split("registers")[0].strip()
                elif cur in ("s512_row_segs", "s512_tile_segs") and "spill" in line:
                    regs.setdefault(v, {})[cur + "_spill"] = line.strip()
    for v, d in regs.items():
        print(f"- {v}: " + ", ".join(f"{k} {x}" for k, x in d.items()))
    print()


def eta(out: str) -> dict:
    table: dict = {}
    for tag in ("ref", "fast", "fast_curve"):
        for r in rows(f"{out}/eta_{tag}.jsonl"):
            if r["kind"] == "eta" and "per_call_ms" in r:
                table[(tag.split("_")[0], r["shape"][0], r["schedule"])] = r
    if not table:
        return table
    print("## η (W = 3, graphs of 6 calls)\n")
    print("| m | schedule | ref ms | ref η | fast ms | fast η | fast η_cap | MHz |")
    print("|---|---|---|---|---|---|---|---|")
    ms = sorted({k[1] for k in table})
    scheds = []
    for k in table:
        if k[2] not in scheds:
            scheds.append(k[2])
    for m in ms:
        for s in scheds:
            a, b = table.get(("ref", m, s)), table.get(("fast", m, s))
            if not (a or b):
                continue
            f = lambda r, key: "" if r is None else r.get(key, "")  # noqa: E731
            print(f"| {m} | {s} | {f(a, 'per_call_ms')} | {f(a, 'eta')} | {f(b, 'per_call_ms')} | {f(b, 'eta')} | "
                  f"{f(b, 'eta_cap')} | {round((b or a).get('sm_mhz', 0))} |")
    print()
    return table


def power(out: str, p_idle: float) -> None:
    rs = [r for r in rows(f"{out}/eta_power.jsonl") if r["kind"] == "power"]
    if not rs:
        return
    print(f"## power holds (m = {rs[0]['shape'][0]}, W = 3) and the 600 W energy bound (P_idle {p_idle} W)\n")
    print("| schedule | ms / call | MHz | W | throttle | J / call | dynamic J | T at 600 W, all overlapped (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rs:
        e = r.get("energy_mj_per_call", 0) / 1e3
        dyn = (r["power_w_median"] - p_idle) * r["per_call_ms"] / 1e3
        print(f"| {r['schedule']} | {r['per_call_ms']} | {r['sm_mhz_median']} | {r['power_w_median']} | "
              f"{','.join(r['throttle_seen'])} | {e:.3f} | {dyn:.3f} | {dyn / (LIMIT_W - p_idle) * 1e3:.3f} |")
    print()


def main(argv: list[str]) -> int:
    out = argv[1]
    p_idle = float(argv[2]) if len(argv) > 2 else 84.0
    gates(out)
    sha2(out)
    eta(out)
    power(out, p_idle)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
