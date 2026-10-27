"""The GPU's no-fold call (v31lib's Prover over libv31.so) checked against DITHER-FILL's reference itself,
`verity/core/protocols/pouw/circuit/dither.py` (#4974), rather than v31.py's mirror of it. On each case: the pad
(`dither.pad_rows`), every word of sampled cells (`dither.cell_words`), sampled leaves rebuilt from only the cells they hold
(`dither.leaf_row`) and opened against the GPU's root on the GPU's own paths (`dither.opens`, salted `dither.prf_salt` under
the call's key), and, on the small cases, the whole call (`dither.words`, `dither.commit`) against the GPU's words and root.
A cheating call must fail them.

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
import v31lib  # noqa: E402

from verity.core.protocols.pouw.circuit import dither  # noqa: E402

# NAME M N K LP L ROW A CHEAT FULL: A pad (the filler) or llm (a served operand); FULL 1: the whole call through dither.py
CASES = """
d-small-pad 128 256 768 256 512 1024 pad none 1
d-small-llm 256 128 1024 128 512 2048 llm none 1
d-qkv-L4096 256 6144 4096 256 4096 1024 pad none 0
d-o-L1024-row16k 256 4096 4096 128 1024 16384 pad none 0
d-down-L4096 128 384 14336 256 4096 1024 llm none 0
d-up-L8192 128 28672 8192 256 8192 1024 pad none 0
c-skip 128 256 2048 256 512 1024 pad skip:3 1
c-forge 256 256 4096 256 1024 1024 llm forge:2:0.5 0
"""


def run_case(lib: v31lib.Lib, line: str, cells: int, leaves: int, rng: np.random.Generator) -> dict:
    name, m, n, k, lp, L, row, a_kind, cheat, full = line.split()
    m, n, k, lp, L, row = map(int, (m, n, k, lp, L, row))
    sh = dither.Shape(m, n, k, lp, L, row)
    B = rng.integers(-128, 128, size=(n, k), dtype=np.int8)
    A = None if a_kind == "pad" else rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    seeds = dither.call_seeds(os.urandom(64), os.urandom(64))
    rep: dict = {"name": name, "shape": sh.label(), "S": sh.S, "A": a_kind, "cheat": cheat}
    with v31lib.Prover(m, n, k, lp, L, row, lib=lib) as p:
        p.set_weights(B)
        c = p.prove(seeds, None if A is None else p.load_a(A), cheat)
        root, key = p.root(c), p.calls[c]["key"]
        words = p.words(c)
        if A is None:
            rep["pad_equal"] = p.pad() == np.array(dither.pad_rows(seeds, k, range(m)), dtype=np.int8).tobytes()
        bad_cells = []
        for _ in range(cells):
            i, j = int(rng.integers(m)), int(rng.integers(n))
            ref = dither.cell_words(sh, seeds, B[j], i, j, None if A is None else A[i])
            got = []
            for s in range(1, sh.S + 1):
                pos, off = sh.word_leaf(i, j, s)
                got.append(int.from_bytes(words[pos * row + off:pos * row + off + 2], "little"))
            if got != ref:
                bad_cells.append((i, j, next(s + 1 for s in range(sh.S) if got[s] != ref[s])))
        rep["cells"], rep["cells_bad"] = cells, bad_cells[:8]
        rep["cells_equal"] = cells - len(bad_cells)
        picks = [int(x) for x in rng.choice(sh.leaves, size=min(leaves, sh.leaves), replace=False)]
        opened = p.open(c, picks)
        rows_equal = opens = 0
        for pos, (g_row, _, g_path) in zip(picks, opened):
            ref_row = dither.leaf_row(sh, seeds, B, pos, A)
            rows_equal += g_row == ref_row
            opens += dither.opens(sh, root, pos, ref_row, dither.prf_salt(key, pos), g_path)
        rep["leaves"], rep["rows_equal"], rep["leaves_open"] = len(picks), rows_equal, opens
        if full == "1":
            T = dither.words(sh, seeds, B, A)
            rep["words_equal"] = dither.words_buffer(sh, T) == words
            rep["root_equal"] = dither.commit(sh, T, lambda q: dither.prf_salt(key, q)).root == root
        rep["root"] = root.hex()[:16]
    honest = (rep.get("pad_equal", True) and rep["cells_equal"] == cells and rep["rows_equal"] == rep["leaves"]
              and rep["leaves_open"] == rep["leaves"] and rep.get("words_equal", True) and rep.get("root_equal", True))
    rep["accept"] = bool(honest)
    rep["good"] = rep["accept"] == (cheat == "none")
    return rep


def describe(r: dict) -> str:
    sh = r["shape"]
    parts = [f"{sh['m']}x{sh['n']}x{sh['k']} L' {sh['lp']} L {sh['L']} row {sh['row']} S {r['S']} A {r['A']}"]
    if "pad_equal" in r:
        parts.append(f"pad {'equal' if r['pad_equal'] else 'DIFFERS'}")
    parts.append(f"cells {r['cells_equal']}/{r['cells']} equal"
                 + (f" (first off: (i, j, s) {tuple(r['cells_bad'][0])})" if r["cells_bad"] else ""))
    parts.append(f"leaves {r['rows_equal']}/{r['leaves']} rows equal, {r['leaves_open']}/{r['leaves']} open the GPU's root")
    if "words_equal" in r:
        parts.append(f"whole call: words {'equal' if r['words_equal'] else 'DIFFER'}, root "
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
          f"calls exact against dither.py, {sum(not r['accept'] for r in cheats)}/{len(cheats)} cheating calls refused",
          flush=True)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
