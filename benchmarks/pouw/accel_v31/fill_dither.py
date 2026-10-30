"""The GPU's no-fold call (v31lib's Prover over libv31.so) checked against the reference itself,
`verity/core/protocols/pouw/circuit/dither.py`, rather than v31.py's numpy forms of it, on filler calls (the pad,
`dither.Shape` with `o` False: T_1 .. T_S a tile) and served calls (an operand A, `o` True: T_1 .. T_{S-1} and then O a
tile). On each case: the call's seeds (`dither.fill_seeds` for a filler, `dither.seeds` for a served call) as the device's
seed buffer holds them (v31's `fill_seeds_bytes`), read back stream by stream against dither's byte for byte, the
device's pad (`dither.pad_rows`), every committed
value of sampled cells (`dither.cell_values`: the T_s the layout commits, and O), sampled leaves rebuilt from only the
cells they hold (`dither.leaf_row`) and opened against the GPU's root on the GPU's own paths (`dither.opens`, salted
`dither.salts(Ks, off)` at the call's offset), and, on the small cases, the whole call (`dither.values`,
`dither.values_buffer`, `dither.commit`) against the GPU's values and root. The device runs on the seeds read back from
its buffer and the reference on dither's. A cheating call must fail them.

    PYTHONPATH=. python3 benchmarks/pouw/accel_v31/fill_dither.py --lib LIB [--cells 64] [--leaves 8] [--out DIR]

dither.py is pure Python, so only the small cases rebuild the whole call. One line a case, then the verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31
import v31lib

from verity.core.protocols.pouw.circuit import dither

# NAME M N K LP L ROW KIND CHEAT FULL: KIND fill (the pad, from fill seeds) or served (an operand A, from served seeds;
# operands in -127 .. 126, as a served root refuses a row of A + d_A or B + d_B spanning -128 .. 128); FULL 1: the whole
# call through dither.py
CASES = """
f-small 128 256 768 256 512 1024 fill none 1
f-qkv-L4096 256 6144 4096 512 4096 1024 fill none 0
f-o-L1024-row16k 256 4096 4096 128 1024 16384 fill none 0
f-up-L4096 128 28672 8192 512 4096 1024 fill none 0
s-small 256 128 1024 128 512 2048 served none 1
s-small-S1 128 256 768 256 768 1024 served none 1
s-qkv-L4096-row16k 256 6144 4096 512 4096 16384 served none 0
s-o-L1024-row16k 256 4096 4096 128 1024 16384 served none 0
s-down-L4096 128 384 14336 512 4096 1024 served none 0
s-up-L8192 128 28672 4096 512 8192 1024 served none 0
f-qkv-L4096-row8k 256 6144 4096 512 4096 8192 fill none 0
s-down-L4096-row8k 128 384 14336 512 4096 8192 served none 0
f-small-row32k 128 256 1024 256 512 32768 fill none 1
s-small-row32k 256 128 1024 128 512 32768 served none 1
f-o-L4096-row32k 128 512 4096 512 4096 32768 fill none 0
s-down-L4096-row32k 128 384 14336 512 4096 32768 served none 0
c-skip 128 256 2048 256 512 1024 fill skip:3 1
c-forge 256 256 4096 256 1024 1024 served forge:2:0.5 0
c-forgeO 256 256 4096 256 1024 1024 served forgeO:0.5 0
"""


def case_seeds(served: bool, rng: np.random.Generator) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """(the seeds read back from the device's seed buffer, dither's) for one call."""
    if served:
        ref = dither.seeds(os.urandom(64), os.urandom(64), int(rng.integers(1 << 62)))
    else:
        ref = dither.fill_seeds(os.urandom(64), os.urandom(64))
    return v31.fill_seeds_from(v31.fill_seeds_bytes(ref), served), ref


def committed(sh: dither.Shape, buf: bytes, i: int, j: int) -> list[int]:
    """Cell (i, j)'s committed values in `buf`: its T_s (s <= W), then O when the call commits it."""
    out = []
    for s in range(1, sh.W + 1):
        pos, off = sh.word_leaf(i, j, s)
        out.append(int.from_bytes(buf[pos * sh.row + off:pos * sh.row + off + 2], "little"))
    if sh.o:
        pos, off = sh.o_leaf(i, j)
        out.append(int.from_bytes(buf[pos * sh.row + off:pos * sh.row + off + 4], "little"))
    return out


def run_case(lib: v31lib.Lib, line: str, cells: int, leaves: int, rng: np.random.Generator) -> dict:
    name, m, n, k, lp, L, row, kind, cheat, full = line.split()
    m, n, k, lp, L, row = map(int, (m, n, k, lp, L, row))
    served = kind == "served"
    sh = dither.Shape(m, n, k, lp, L, row, o=served)
    lo, hi = (-127, 127) if served else (-128, 128)
    B = rng.integers(lo, hi, size=(n, k), dtype=np.int8)
    A = rng.integers(lo, hi, size=(m, k), dtype=np.int8) if served else None
    seeds, ref = case_seeds(served, rng)
    rep: dict = {"name": name, "shape": sh.label(), "L": L, "S": sh.S, "W": sh.W, "kind": kind, "cheat": cheat,
                 "seeds_equal": seeds == ref}
    Aref = A if served else np.array(dither.pad_rows(ref, k, range(m)), dtype=np.int8)
    with v31lib.Prover(m, n, k, lp, L, row, served=served, lib=lib) as p:
        p.set_weights(B)
        c = p.prove(seeds, p.load_a(A) if served else None, cheat)
        root, off = p.root(c), p.calls[c]["off"]
        buf = p.values(c)
        if not served:
            rep["pad_equal"] = p.pad() == Aref.tobytes()
        bad_cells = []
        for _ in range(cells):
            i, j = int(rng.integers(m)), int(rng.integers(n))
            T, O = dither.cell_values(sh, ref, Aref[i], B[j], i, j)
            want = T[:sh.W] + ([O] if served else [])
            got = committed(sh, buf, i, j)
            if got != want:
                bad_cells.append((i, j, next(x for x in range(len(want)) if got[x] != want[x])))
        rep["cells"], rep["cells_bad"] = cells, bad_cells[:8]
        rep["cells_equal"] = cells - len(bad_cells)
        picks = [int(x) for x in rng.choice(sh.leaves, size=min(leaves, sh.leaves), replace=False)]
        rows_equal = opens = 0
        dom, salt = sh.domain(), dither.salts(p.ks, off)
        for pos, (g_row, _, g_path) in zip(picks, p.open(c, picks)):
            ref_row = dither.leaf_row(sh, ref, Aref, B, pos)
            rows_equal += g_row == ref_row
            opens += dither.opens(dom, root, sh.leaves, pos, ref_row, salt(pos), g_path)
        rep["leaves"], rep["rows_equal"], rep["leaves_open"] = len(picks), rows_equal, opens
        if full == "1":
            T, O = dither.values(sh, ref, Aref, B)
            want = dither.values_buffer(sh, T, O)
            rep["values_equal"] = want == buf
            rep["root_equal"] = dither.commit(sh, want, salt).root == root
        rep["root"] = root.hex()[:16]
    honest = (rep["seeds_equal"] and rep.get("pad_equal", True) and rep["cells_equal"] == cells
              and rep["rows_equal"] == rep["leaves"] and rep["leaves_open"] == rep["leaves"]
              and rep.get("values_equal", True) and rep.get("root_equal", True))
    rep["accept"] = bool(honest)
    rep["good"] = rep["accept"] == (cheat == "none")
    return rep


def describe(r: dict) -> str:
    sh = r["shape"]
    parts = [f"{r['kind']} {sh['m']}x{sh['n']}x{sh['k']} L' {sh['lp']} L {r['L']} row {sh['row']} S {r['S']} W {r['W']}",
             f"seeds {'equal' if r['seeds_equal'] else 'DIFFER'}"]
    if "pad_equal" in r:
        parts.append(f"pad {'equal' if r['pad_equal'] else 'DIFFERS'}")
    parts.append(f"cells {r['cells_equal']}/{r['cells']} equal"
                 + (f" (first off: (i, j, value) {tuple(r['cells_bad'][0])})" if r["cells_bad"] else ""))
    parts.append(f"leaves {r['rows_equal']}/{r['leaves']} rows equal, {r['leaves_open']}/{r['leaves']} open the GPU's root")
    if "values_equal" in r:
        parts.append(f"whole call: values {'equal' if r['values_equal'] else 'DIFFER'}, root "
                     f"{'equal' if r['root_equal'] else 'DIFFERS'}")
    return (f"{r['name']}: {'ACCEPT' if r['accept'] else 'REFUSE'} ({r['cheat']}) " + "; ".join(parts)
            + f"; root {r['root']}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cells", type=int, default=64)
    ap.add_argument("--leaves", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv[1:])
    lib, rng = v31lib.Lib(a.lib), np.random.default_rng(a.seed)
    reps = []
    for line in CASES.strip().splitlines():
        r = run_case(lib, line, a.cells, a.leaves, rng)
        reps.append(r)
        print(describe(r), flush=True)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "fill-dither.json").write_text(json.dumps(reps, indent=1))
    honest = [r for r in reps if r["cheat"] == "none"]
    cheats = [r for r in reps if r["cheat"] != "none"]
    good = all(r["good"] for r in reps)
    print(f"dither.py recheck: {'ACCEPT' if good else 'REFUSE'}: {sum(r['accept'] for r in honest)}/{len(honest)} honest "
          f"calls exact against dither.py ({sum(r['kind'] == 'served' for r in honest)} served), "
          f"{sum(not r['accept'] for r in cheats)}/{len(cheats)} cheating calls refused", flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
