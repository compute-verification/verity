#!/usr/bin/env python3
"""Round 2's tables (CMP-16) from lease_fast.sh's and lease_cores.sh's out/ directories: the gates (run.py's fixture with the
fast and the reference kernels, eta.py's `same`), the SHA-512 kernels by launch-bound variant (`sha2`), η of the served call
with the reference and the fast kernels and the what-if schedules at either tile (`time`), the power holds with each
schedule's energy a call, and the path to the ε < 1 threshold: each combination's time bound and its bound under the 600 W
cap, the clock solved from P(f) = P_idle (f/f0)^(α-1) + P_dyn (f/f0)^α at α = 1, 2, 3.

  python3 analyze2.py <run dir>/out [<run dir>/out ...] [P_IDLE_W]   (P_IDLE_W: the board at the lock with nothing running;
                                                                      default 84)
"""
from __future__ import annotations

import json
import os
import sys

LIMIT_W = 600.0
MACS, SMS, MACS_PER_SM_CLOCK, REF_MHZ = 8192 ** 3, 188, 1024, 2077
#: af + pa at m = 8192, round 1's ncu pass (note:20261015T0608Z-draft-pouw-eta-study): what main-a-nf-ct keeps beside the
#: floor and gemm_ct
CLEANUP_MS = 0.07


def rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.startswith("{")]


def gates(out: str) -> None:
    print(f"## gates ({out})\n")
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


def eta(outs: list[str]) -> dict:
    """(kernels, tile, m, schedule) -> the eta row, over every out dir."""
    table: dict = {}
    for out in outs:
        for tag in ("ref", "fast", "fast_curve", "d64"):
            for r in rows(f"{out}/eta_{tag}.jsonl"):
                if r["kind"] == "eta" and "per_call_ms" in r:
                    table[("ref" if tag == "ref" else "fast", r["tile"][0], r["shape"][0], r["schedule"])] = r
    if not table:
        return table
    print("## η (W = 3, graphs of 6 calls)\n")
    print("| tile | m | schedule | ref ms | ref η | fast ms | fast η | fast η_cap | MHz |")
    print("|---|---|---|---|---|---|---|---|---|")
    keys = sorted({k[1:3] for k in table}, key=lambda t: (-t[0], t[1]))
    scheds = []
    for k in table:
        if k[3] not in scheds:
            scheds.append(k[3])
    for tile, m in keys:
        for s in scheds:
            a, b = table.get(("ref", tile, m, s)), table.get(("fast", tile, m, s))
            if not (a or b):
                continue
            f = lambda r, key: "" if r is None else r.get(key, "")  # noqa: E731
            print(f"| {tile} | {m} | {s} | {f(a, 'per_call_ms')} | {f(a, 'eta')} | {f(b, 'per_call_ms')} | {f(b, 'eta')} | "
                  f"{f(b, 'eta_cap')} | {round((b or a).get('sm_mhz', 0))} |")
    print()
    return table


def power(outs: list[str], p_idle: float) -> dict:
    """(tile, schedule) -> (ms a call, dynamic J), over every out dir's holds at m = 8192."""
    rs = [r for out in outs for f in ("eta_power.jsonl", "eta_power_d64.jsonl") for r in rows(f"{out}/{f}")
          if r["kind"] == "power" and r["shape"][0] == 8192]
    if not rs:
        return {}
    print(f"## power holds (m = 8192, W = 3) and the 600 W energy bound (P_idle {p_idle} W)\n")
    print("| tile | schedule | ms / call | MHz | W | throttle | J / call | dynamic J | T at 600 W, all overlapped (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    held = {}
    for r in rs:
        e = r.get("energy_mj_per_call", 0) / 1e3
        dyn = (r["power_w_median"] - p_idle) * r["per_call_ms"] / 1e3
        held[(r["tile"][0], r["schedule"])] = (r["per_call_ms"], dyn)
        print(f"| {r['tile'][0]} | {r['schedule']} | {r['per_call_ms']} | {r['sm_mhz_median']} | {r['power_w_median']} | "
              f"{','.join(r['throttle_seen'])} | {e:.3f} | {dyn:.3f} | {dyn / (LIMIT_W - p_idle) * 1e3:.3f} |")
    print()
    return held


def capped_ms(t0: float, e0: float, alpha: int, p_idle: float) -> float:
    """The call's ms once the clock drops until P(f) fits under the cap (t0 at the reference clock, e0 dynamic J)."""
    p = e0 / (t0 * 1e-3)
    if p_idle + p <= LIMIT_W:
        return t0
    lo, hi = 0.3, 1.0
    for _ in range(60):
        x = (lo + hi) / 2
        lo, hi = (lo, x) if p_idle * x ** (alpha - 1) + p * x ** alpha > LIMIT_W else (x, hi)
    return t0 / lo


def path(table: dict, held: dict, p_idle: float) -> None:
    """Each combination's time bound (the measured schedule with its hidden steps left out) and energy (all the work the
    call still does), at m = 8192, 128 x 128 tiles."""
    t = lambda s: table[("fast", 128, 8192, s)]["per_call_ms"]  # noqa: E731
    e = lambda s: held[(128, s)][1]  # noqa: E731
    need = ("serial", "main-a", "main-a-nf", "main-a-nf-ct", "gemm_ct", "hashonly")
    if any(("fast", 128, 8192, s) not in table for s in need) or any((128, s) not in held for s in need[:4]):
        return
    ref_ms = MACS / (SMS * MACS_PER_SM_CLOCK * REF_MHZ * 1e6) * 1e3
    hidden = e("serial") - e("main-a")
    kept = (0.0, (e("main-a") - e("main-a-nf")) / 2)
    rows_j = e("hashonly") if (128, "hashonly") in held else 0.0
    rows_share = table[("fast", 128, 8192, "hashonly")]["per_call_ms"]
    sha = {r["unit"]: r["both_ms"] for out in OUTS for r in rows(f"{out}/eta_sha2.jsonl")
           if r["kind"] == "sha2" and r["shape"][0] == 8192 and r["variant"] == "ship"}
    rows_ms, tiles_ms = sha.get("rows", 0.457), sha.get("tiles", 0.798)
    rows_e, tiles_e = (rows_j * rows_ms / rows_share, rows_j * tiles_ms / rows_share)
    floor_ms = t("main-a-nf-ct") - t("gemm_ct") - CLEANUP_MS
    e_e = [e("main-a-nf-ct") + hidden + k for k in kept]
    combos = [
        ("A. (1) only", t("serial"), [e("serial")] * 2),
        ("B. + tile leaves hidden", t("serial") - tiles_ms, [e("serial")] * 2),
        ("C. + A's tree hidden (main-a)", t("main-a"), [e("serial")] * 2),
        ("D. + fused forming (main-a-nf)", t("main-a-nf"), [e("main-a-nf") + hidden + k for k in kept]),
        ("E. + epilogue without U (main-a-nf-ct)", t("main-a-nf-ct"), e_e),
        ("F. + floor behind another GEMM", t("main-a-nf-ct") - floor_ms, e_e),
        ("G. E + tile leaf over C~ only", t("main-a-nf-ct"), [x - tiles_e / 2 for x in e_e]),
        ("G'. F + tile leaf over C~ only", t("main-a-nf-ct") - floor_ms, [x - tiles_e / 2 for x in e_e]),
        ("H. G + rows over 1-byte A", t("main-a-nf-ct") - 0.75 * rows_ms, [x - tiles_e / 2 - 0.75 * rows_e for x in e_e]),
    ]
    print(f"## path to ε < 1 (η over {ref_ms:.3f} ms; thresholds: T < "
          f"{MACS / (0.58 * SMS * MACS_PER_SM_CLOCK * 2430e6) * 1e3:.3f} ms at a 2,430 MHz cap, "
          f"{MACS / (0.58 * SMS * MACS_PER_SM_CLOCK * 2092e6) * 1e3:.3f} ms at the 2,092 MHz lock)\n")
    print("| combination | time-bound ms | η | dynamic J | η at α = 1 | α = 2 | α = 3 |")
    print("|---|---|---|---|---|---|---|")
    for name, t0, es in combos:
        cells = []
        for a in (1, 2, 3):
            lo, hi = sorted(ref_ms / capped_ms(t0, x, a, p_idle) for x in es)
            cells.append(f"{lo:.2f}" if hi - lo < 0.005 else f"{lo:.2f}–{hi:.2f}")
        print(f"| {name} | {t0:.3f} | {ref_ms / t0:.3f} | {min(es):.3f}–{max(es):.3f} | " + " | ".join(cells) + " |")
    print()


OUTS: list[str] = []


def main(argv: list[str]) -> int:
    args = argv[1:]
    p_idle = float(args.pop()) if args and not os.path.isdir(args[-1]) else 84.0
    OUTS[:] = args
    for out in OUTS:
        gates(out)
        sha2(out)
    table = eta(OUTS)
    held = power(OUTS, p_idle)
    path(table, held, p_idle)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
