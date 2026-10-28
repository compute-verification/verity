"""The PoUW tile shape R×C as a planner knob, on the served B4 window (4,32,4): `optimize.py`'s work target over the
window's int8 GEMM cells (`window_b4_census.py`'s census), every tile shape an alternative of one op a K.

A tile is R rows of activations by C weight columns at depth K: R·C·K MACs, credited as its work, at `ANDS_A_MAC` in
circuit (the inbox's calibration, an estimate) and opening R activation rows and C weight rows of K bytes and an output
row of 4·R·C bytes (`row_ands`). A tile past `MAX_PROVED_ANDS` is proved as its R lanes (1×C each), one proof block a
lane, the tile's ANDs still a draw. Mode 2 adds each tile's serve-time receipt, an estimate: H1's CP-W receipt,
2(K/32 + m) u32 a tile whatever its shape (`transformer.serve`), at `class_rows`' per-row cost, h at 398 GB/s. Costs
are `window_b4.costs` (C-Flock's fit, statements shared); `--cap-log 30` models Flock's cap at k_log 30 (a bigger cap's
extra queries and proof bytes are not priced).

    uv run python benchmarks/sampling_plan/pouw_tile.py CENSUS.json --out DIR [--cap-log 27]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import transformer as X
import window_b4 as W

from verity.core.protocols.sampled_proofs.optimize import Candidate, Objective, for_target, row_ands

ANDS_A_MAC = 300
SIDES = (1, 2, 4, 8, 16, 32)
COMPRESSION = 128


def receipt_compressions(K: int) -> int:
    b = 2 * (K // 32 + (2 if K <= 4096 else 3)) * 4
    return -(-(128 + b + 17) // 128) + 6


def tiles(census: dict, cap: int) -> tuple[list[Candidate], list[dict]]:
    """One op a K (the window's int8 linears' cells), one alternative a shape that divides its cells."""
    cells: dict[int, int] = {}
    for e in census["templates"]:
        if e["label"].startswith("ScaledMmS8_v1"):
            K = int(e["label"].split("K=")[1].split(",")[0])
            cells[K] = cells.get(K, 0) + e["units"] * e["N"]
    out, table = [], []
    for K, n in sorted(cells.items()):
        for R in SIDES:
            for C in SIDES:
                if n % (R * C):
                    continue
                rows = (K,) * R + (K,) * C + (4 * R * C,)
                ands = ANDS_A_MAC * R * C * K
                proved = ands + sum(row_ands(r) for r in rows)
                lane = ANDS_A_MAC * C * K + sum(row_ands(r) for r in (K,) * (1 + C) + (4 * C,))
                lanes = proved > cap
                if lanes and lane > cap:
                    continue
                c = Candidate(f"pouw-tile{{K={K}}}", f"PouwTile{{K={K},R={R},C={C}}}", n // (R * C), R * C * K, 16,
                              cut=f"{R}x{C}", ands=ands, rows=rows, commit_rows=float(receipt_compressions(K)),
                              cap=X._cap(lane if lanes else proved))
                out.append(c)
                table.append({"K": K, "R": R, "C": C, "tiles": c.units, "proved_ands": proved, "lanes": lanes,
                              "block_ands": lane if lanes else proved, "openings_over_compute": (proved - ands) / ands})
    return out, table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("census")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-log", type=int, default=27)
    ap.add_argument("--grid", type=int, default=61)
    args = ap.parse_args()
    census = json.loads(Path(args.census).read_text())
    cands, table = tiles(census, 1 << args.cap_log)
    F = W.window_work(cands)
    doc = {"cap_log": args.cap_log, "ands_a_mac": ANDS_A_MAC, "window_macs": F, "table": table, "runs": {}}
    h = 1 / W.SHA512_BPS["card 4 KiB"]
    for name, cm in (("mode1", W.costs()), ("mode2", W.costs(h=h, w=1.0))):
        p = for_target(cands, cm, Objective(W.DELTA, work=1.0), W.EPS * F, by="free_compute", exact=False,
                       grid=args.grid)
        chosen = [c for c in cands if c.template in p.sizes]
        r = W.report(p, chosen, cm, F, W.parts(W.fitted()))
        r["shapes"] = {c.op: c.cut for c in chosen}
        r["block_ands"] = {c.op: next(t["block_ands"] for t in table if f"{t['R']}x{t['C']}" == c.cut
                                      and c.op.endswith(f"K={t['K']}}}")) for c in chosen}
        doc["runs"][name] = r
        print(name, r["shapes"], "draws", r["draws"], "proving ANDs %.4g" % r["proving_ands"],
              "fit-GPU-s %.1f" % r["total_gpu_s"], "commit %.4g" % r["commit_gpu_s"], "eps %.4g" % r["eps"],
              "block ANDs", r["block_ands"])
    cm = W.costs()
    for shape in sorted({c.cut for c in cands}, key=lambda s: tuple(map(int, s.split("x")))):
        fixed = [c for c in cands if c.cut == shape]
        if len({c.op for c in fixed}) < len({c.op for c in cands}):
            continue
        p = for_target(fixed, cm, Objective(W.DELTA, work=1.0), W.EPS * F, by="free_compute", exact=False,
                       grid=args.grid)
        doc["runs"][f"shape {shape}"] = {"draws": sum(p.sizes.values()), "total_gpu_s": p.cost["total"],
                                         "proving_ands": sum(p.sizes[c.template] * cm.ands(c) for c in fixed),
                                         "lanes": any(t["lanes"] for t in table if f"{t['R']}x{t['C']}" == shape)}
        print("shape", shape, doc["runs"][f"shape {shape}"])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"pouw_tile_cap{args.cap_log}.json").write_text(json.dumps(doc, indent=1, default=str))


if __name__ == "__main__":
    main()
