"""The H200 step from `v31 bench`'s JSON lines (h200.sh): the frozen shape's 52 layers (q and o, kv, up, down) plus the
head at M = 16,384, each arm's seconds a step, gamma_u against wgmma.linear's step, the split and the filler's residual.

A step's calls by (N, K): q and o (16,384, 16,384) twice a layer, kv (128, 16,384), up (65,536, 16,384) and down
(16,384, 65,536) once, and the head (256, 16,384) once a step. The split is the protocol's GEMM (`gemm`: X W'^T, the
peel's tail stages, the T_s stores and the epilogue: the tensor side), its hashing (`coins`, the SHA-512 coin streams, and
`commit`, the leaves and the tree) and its other CUDA-core work (`form` and `wrap`). The residual is 1 - W / (step θ),
W the step's MACs and θ the H200's sheet rate (4,096 MACs an SM a clock, 132 SMs, 1,830 MHz; #4940).

    python3 h200_step.py BENCH.jsonl ... [--wgmma 3.06]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS = 52
CALLS = {(16384, 16384): 2 * LAYERS, (128, 16384): LAYERS, (65536, 16384): LAYERS, (16384, 65536): LAYERS,
         (256, 16384): 1}
THETA = 4096 * 132 * 1.83e9
ARMS = ("lt", "plain", "coins", "form", "wrap", "gemm", "commit", "call", "ovl")


def rows(paths: list[str]) -> list[dict]:
    """Each bench line, with its variant: the file's name past `bench-` (run.sh writes a variant's lines to one file)."""
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.startswith("{") and '"gamma_ovl"' in line:
                out.append(json.loads(line) | {"variant": Path(p).stem.removeprefix("bench-")})
    return out


def steps(lines: list[dict], m: int = 16384) -> dict[tuple[int, int, int, str], dict]:
    """Per (L', t, ROW, variant): each arm's seconds a step and the step's MACs, for groups that timed every call."""
    groups: dict[tuple[int, int, int, str], dict[tuple[int, int], dict]] = {}
    for r in lines:
        if r["m"] == m and (r["n"], r["k"]) in CALLS:
            groups.setdefault((r["lp"], r["t"], r["row"], r.get("variant", "")), {})[(r["n"], r["k"])] = r
    out = {}
    for key, by in sorted(groups.items()):
        if set(by) != set(CALLS):
            continue
        sec = {a: sum(CALLS[nk] * by[nk]["arms"][a]["ms"] for nk in CALLS) / 1e3 for a in ARMS}
        macs = sum(CALLS[nk] * m * nk[0] * nk[1] for nk in CALLS)
        mhz = [by[nk]["arms"]["ovl"]["mhz"] for nk in CALLS if by[nk]["arms"]["ovl"]["mhz"]]
        out[key] = {"s": sec, "macs": macs, "mhz": (min(mhz), max(mhz)) if mhz else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--wgmma", type=float, default=3.06, help="wgmma.linear's step at M = 16,384, s (HASH-22)")
    a = ap.parse_args()
    st = steps(rows(a.files))
    print("| L' | t | L | ROW | variant | ovl s | gamma_u ovl | call s | gamma_u call | gemm s | coins + commit s | "
          "form + wrap s | plain s | cuBLASLt s | residual (ovl) | SM MHz (ovl) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (lp, t, row, var), v in st.items():
        s = v["s"]
        mhz = f"{v['mhz'][0]:.0f}-{v['mhz'][1]:.0f}" if v["mhz"] else "-"
        print(f"| {lp} | {t} | {lp * t} | {row} | {var} | {s['ovl']:.3f} | {s['ovl'] / a.wgmma:.3f} | {s['call']:.3f} | "
              f"{s['call'] / a.wgmma:.3f} | {s['gemm']:.3f} | {s['coins'] + s['commit']:.3f} | {s['form'] + s['wrap']:.3f} | "
              f"{s['plain']:.3f} | {s['lt']:.3f} | {1 - v['macs'] / (s['ovl'] * THETA):.3f} | {mhz} |")
    for (lp, t, row, var), v in st.items():
        s = v["s"]
        print(f"summary: L {lp * t} row {row} {var}: step {s['ovl']:.3f} s overlapped, gamma_u {s['ovl'] / a.wgmma:.3f} against "
              f"wgmma's {a.wgmma} s; serial {s['call']:.3f} s; gemm {s['gemm']:.3f}, hashing {s['coins'] + s['commit']:.3f}, "
              f"other {s['form'] + s['wrap']:.3f} s")
    if not st:
        print("summary: no complete step (a bench shape is missing)")
    return 0 if st else 1


if __name__ == "__main__":
    raise SystemExit(main())
