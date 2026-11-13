#!/usr/bin/env python3
"""A reviewer's sweep of two classes against TT_OUT rev1 on the H100 (CPU, bit-exact): dead pairs and the raised adder.

`tests/test_tt_out_tile_h100_rev1.py` holds each class: its rows, its W1 program and the program's price. This runs them
over depths and seeds on whole 64 x 64 audit tiles (`--cases` for dead pairs, `--raised` for the raised adder), and
checks each mechanism on the scheme's own code (`--scheme`, `--raised-scheme`): `pearl_c.PearlC` at forming v1 with -h3's
SHA-512 lines forms a small tile, admits it, and computes its ticket words; the program's words are computed with the
device record's own atom and compared.

    python benchmarks/pouw/dead_pairs_sweep.py --cases 8192:8192:1,16384:8192:2 --runs 1,2,3 --out "$OUT/cases.json"
    python benchmarks/pouw/dead_pairs_sweep.py --scheme 8192:1 --salts 3 --tile 4x4 --out "$OUT/scheme.json"
    python benchmarks/pouw/dead_pairs_sweep.py --raised 8192:8192:20:zero,8192:8192:20:gauss --units 1 --out "$OUT/raised.json"
    python benchmarks/pouw/dead_pairs_sweep.py --raised-scheme 8192:20:zero --salts 3 --tile 4x4 --out "$OUT/scheme.json"
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ttout_tile_h100_rev1", HERE / "tests" / "test_tt_out_tile_h100_rev1.py")
T = importlib.util.module_from_spec(spec)
sys.modules["ttout_tile_h100_rev1"] = T
spec.loader.exec_module(T)
PC, H100, C = T.PC, T.H100, T.C

from verity.definitions.silicon import fp32  # noqa: E402
from verity.protocols.pouw.verifier.protocol import Context, Matmul, Shape, Tile  # noqa: E402


def scheme_check(k: int, groups: int, salt: bytes, rows: int, cols: int) -> dict:
    """The mechanism on the scheme's own code: form `rows` x `cols` of the family with `PearlC.encode` (SHA-512 row seeds
    and lines), admit it, take its ticket words from `PearlC.checked`, and compute the program's with `H100.atom`."""
    sch = PC.PearlC(forming="v1", device=H100)
    s = Shape(8192, k, 8192)
    ctx = Context(salt, Matmul(0, s, 0), hashlib.sha512(b"root-a" + salt).digest(), hashlib.sha512(b"root-b").digest())
    XA = T._dead_pair_rows("A", rows, k, groups, C._rng("scheme", k, "A", salt.hex()))
    XB = T._dead_pair_rows("B", cols, k, groups, C._rng("scheme", k, "B"))
    a_rows = [bytes(r) for r in XA.astype("<f4").view("u1").reshape(rows, 4 * k)]
    b_rows = [bytes(r) for r in XB.astype("<f4").view("u1").reshape(cols, 4 * k)]
    ops = sch.encode(ctx, sch.noise(ctx), a_rows, b_rows, Tile(tuple(range(rows)), tuple(range(cols))))
    assert ops is not None
    admitted = sch.admissible(ctx, ops)
    checked = sch.checked(ctx, ops)
    top = PC.P.f32_to_fp8(448.0)
    live = [128 * g + l for g in range(groups) for l in range(0, 128, 8)]
    dead = [l for l in range(128 * groups) if l % 8]
    want_a = {448.0, 320.0, 288.0}
    dead_ok = all(abs(PC.P.fp8_to_f32(ops.a.codes[i][l])) in want_a for i in range(rows) for l in dead) and \
        all(abs(PC.P.fp8_to_f32(ops.b.codes[j][l])) in {448.0, 320.0, 352.0} for j in range(cols) for l in dead)
    equal, atoms = 0, 0
    for i in range(rows):
        for j in range(cols):
            a, b = ops.a.codes[i], ops.b.codes[j]
            acc, atoms = PC.P.f32_bits(4096.0 * groups), 0
            for lo in range(0, len(live), 30):
                part = live[lo:lo + 30]
                xa = [top, top] + [a[l] for l in part] + [0] * (30 - len(part))
                xb = [top, top | 0x80] + [b[l] for l in part] + [0] * (30 - len(part))
                acc, atoms = H100.atom.step(acc, xa, xb), atoms + 1
            total = acc
            for ws in range(128 * groups, k, 128):
                local = 0
                for t in range(ws, ws + 128, 32):
                    local = H100.atom.step(local, a[t:t + 32], b[t:t + 32])
                total = fp32.add(total, local)
                total = 0 if total == 0x80000000 else total
            equal += total == checked.c[i][j]
    return {"k": k, "groups": groups, "salt": salt.hex()[:16], "tile": (rows, cols), "admitted": bool(admitted),
            "dead_codes_as_designed": bool(dead_ok), "words": rows * cols, "equal": int(equal),
            "atoms": atoms + (k - 128 * groups) // 32, "reference_atoms": k // 32}


def raised_scheme_check(k: int, groups: int, fill: str, salt: bytes, rows: int, cols: int) -> dict:
    """The raised adder on the scheme's own code, as `scheme_check`: the planted codes as designed, every other code of the
    planted groups' later atoms under 8, and against `PearlC.checked`'s words the words of the program that runs one atom a
    planted group (its accumulator the planted sum) and of the one that only skips each planted group's second and third
    atoms."""
    sch = PC.PearlC(forming="v1", device=H100)
    s = Shape(8192, k, 8192)
    ctx = Context(salt, Matmul(0, s, 0), hashlib.sha512(b"root-a" + salt).digest(), hashlib.sha512(b"root-b").digest())
    XA = T._raised_rows("A", rows, k, groups, C._rng("scheme", "raised", k, "A", salt.hex()), fill)
    XB = T._raised_rows("B", cols, k, groups, C._rng("scheme", "raised", k, "B"), fill)
    a_rows = [bytes(r) for r in XA.astype("<f4").view("u1").reshape(rows, 4 * k)]
    b_rows = [bytes(r) for r in XB.astype("<f4").view("u1").reshape(cols, 4 * k)]
    ops = sch.encode(ctx, sch.noise(ctx), a_rows, b_rows, Tile(tuple(range(rows)), tuple(range(cols))))
    assert ops is not None
    admitted = sch.admissible(ctx, ops)
    checked = sch.checked(ctx, ops)
    val = PC.P.fp8_to_f32
    planted_ok, largest = True, 0.0
    for side, codes, plant in (("A", ops.a.codes, T.PLANT_A), ("B", ops.b.codes, T.PLANT_B)):
        for row in codes:
            for g in range(groups):
                top = -448.0 if side == "B" and g % 2 else 448.0
                planted_ok &= all(val(row[128 * g + l]) == top for l in (1, 2, 3))
                planted_ok &= all(val(row[128 * g + 32 * t + 1]) == v for t, v in zip((1, 2, 3), plant[g % 2]))
                largest = max([largest] + [abs(val(row[l])) for l in range(128 * g + 32, 128 * g + 128) if l % 32 != 1])
    equal = equal_skip = 0
    for i in range(rows):
        for j in range(cols):
            a, b = ops.a.codes[i], ops.b.codes[j]
            total = total_skip = 0
            for g in range(k // 128):
                local = 0
                for t in range(128 * g, 128 * g + 128, 32):
                    local = H100.atom.step(local, a[t:t + 32], b[t:t + 32])
                skipped = local
                if g < groups:
                    local = H100.atom.step(PC.P.f32_bits(T.PLANT_SUM[g % 2]), a[128 * g:128 * g + 32], b[128 * g:128 * g + 32])
                    skipped = 0
                    for t in (128 * g, 128 * g + 96):
                        skipped = H100.atom.step(skipped, a[t:t + 32], b[t:t + 32])
                total, total_skip = fp32.add(total, local), fp32.add(total_skip, skipped)
                total = 0 if total == 0x80000000 else total
                total_skip = 0 if total_skip == 0x80000000 else total_skip
            equal += total == checked.c[i][j]
            equal_skip += total_skip == checked.c[i][j]
    return {"class": "raised-adder", "fill": fill, "k": k, "groups": groups, "salt": salt.hex()[:16], "tile": (rows, cols),
            "admitted": bool(admitted), "planted_codes_as_designed": bool(planted_ok),
            "largest_other_code_in_later_atoms": largest, "words": rows * cols, "equal": int(equal),
            "atoms": k // 32 - 3 * groups, "equal_skip_only": int(equal_skip), "atoms_skip_only": k // 32 - 2 * groups,
            "reference_atoms": k // 32}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="", help="k:n:groups, comma separated")
    ap.add_argument("--runs", default="1")
    ap.add_argument("--units", type=int, default=256)
    ap.add_argument("--scheme", default="", help="k:groups, comma separated")
    ap.add_argument("--raised", default="", help="k:n:groups:fill[:program] (fill: zero or gauss; program: one-atom or skip), comma separated")
    ap.add_argument("--raised-scheme", default="", help="k:groups:fill, comma separated")
    ap.add_argument("--salts", type=int, default=1)
    ap.add_argument("--tile", default="4x4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"cases": [], "scheme": []}
    for spec_ in filter(None, args.cases.split(",")):
        k, n, groups = map(int, spec_.split(":"))
        for run in map(int, args.runs.split(",")):
            t0 = time.perf_counter()
            case = T._dead_pair_case(k=k, n=n, groups=groups, units=args.units, run=run)
            case["seconds"] = round(time.perf_counter() - t0, 1)
            out["cases"].append(case)
            print(json.dumps(case), flush=True)
    for spec_ in filter(None, args.raised.split(",")):
        k, n, groups, fill, program = (spec_.split(":") + ["one-atom"])[:5]
        for run in map(int, args.runs.split(",")):
            t0 = time.perf_counter()
            case = T._raised_case(k=int(k), n=int(n), groups=int(groups), units=args.units, run=run, fill=fill,
                                  program=program)
            case["seconds"] = round(time.perf_counter() - t0, 1)
            out["cases"].append(case)
            print(json.dumps(case), flush=True)
    rows, cols = map(int, args.tile.split("x"))
    for spec_ in filter(None, args.raised_scheme.split(",")):
        k, groups, fill = spec_.split(":")
        for i in range(args.salts):
            t0 = time.perf_counter()
            r = raised_scheme_check(int(k), int(groups), fill, hashlib.sha512(b"raised/salt/%d" % i).digest()[:32], rows, cols)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            out["scheme"].append(r)
            print(json.dumps(r), flush=True)
    for spec_ in filter(None, args.scheme.split(",")):
        k, groups = map(int, spec_.split(":"))
        for i in range(args.salts):
            t0 = time.perf_counter()
            r = scheme_check(k, groups, hashlib.sha512(b"dead-pairs/salt/%d" % i).digest()[:32], rows, cols)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            out["scheme"].append(r)
            print(json.dumps(r), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
