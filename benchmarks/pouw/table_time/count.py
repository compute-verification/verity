"""Instructions a word, from each arm's timed loop: every opcode of one iteration (gpu_kit's SASS reader), the words a
thread makes an iteration, and the duo fold's share (LOP3, IMAD, SHF.L.W, IADD3 a word) taken out. For the x arms (design
A's depth-k bit product, no word a segment) a "word" is a lane-word, 32 bit-products, and x = 4 × its instructions is
the warp instructions per 4,096 bit-products.

  python count.py BINARY [--json OUT]           # the table, and every loop's full histogram
  python count.py BINARY --pin expect.json      # also gate every kernel with its whole histogram as --expect

A thread's instruction is one issue slot of a warp that serves 32 lanes, so the counts a thread are lane-instructions a
word as the brief counts them (q*).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gpu_kit_sass_gate", HERE.parent / "gpu_kit" / "sass_gate.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["gpu_kit_sass_gate"] = G
_spec.loader.exec_module(G)

#: words a thread makes in one iteration of the timed loop, what makes them, and how many of them it folds
ARMS = {
    "k_imma_zero": (32, "IMMA m16n8k32, U resident (2 segments × 4 MMAs × 4 words a lane)", 32),
    "k_imma_gen": (32, "IMMA m16n8k32, generic (2 segments × 4 MMAs × 4 words a lane)", 32),
    "k_dp4a": (32, "IDP.4A, 4×8 register tile, LDS.128 operands", 32),
    "k_bits": (8, "bit planes: 8 LOP3 + 8 POPC + combine a word", 8),
    "k_fr8": (16, "Four Russians 8-bit, shared-memory tables, LDS.128 = 4 columns", 16),
    "k_fr16_c16": (16, "Four Russians 16-bit, L2 tables, LDG.256 = 8 columns, 16 columns a thread", 16),
    "k_fr16_c32": (32, "Four Russians 16-bit, L2 tables, LDG.256 = 8 columns, 32 columns a thread", 32),
    "k_bb": (32, "design A: LOP3 AND + POPC + IADD3 a word (4 rows × 4 columns, 2 segments)", 32),
    "k_bx": (128, "x: AND + POPC a lane-word, IADD3 a pair (4 × 4 outputs, 8 segments)", 0),
    "k_bx_hs": (128, "x: Harley-Seal, 8 AND + 7 CSAs + 1 POPC an output (4 × 4, 8 segments)", 0),
    "k_bx_mix": (128, "x: two rows AND + POPC, two Harley-Seal (4 × 4, 8 segments)", 0),
    "k_a_fr8": (16, "design A packed byte route (SPEC §4.2a): smem byte table rebuilt a segment, ¼ LDS.128 + packed "
                    "IADD3 + IDP.4A writer; the build loop's body is counted once, so its full cost is read from the time",
                16),
}
#: the duo fold, a word: u = y ^ d ^ k (LOP3), y = u a + z (IMAD), z = rotl(u, r) + y (SHF.L.W, IADD3)
FOLD = {"LOP3.LUT": 1, "IMAD": 1, "SHF.L.W.U32.HI": 1, "IADD3": 1}


def loop_body(fn) -> Counter:
    clocks = G.clock_reads(fn)
    lo, hi = clocks[0], clocks[-1]
    timed = [(j, k) for j, k in G.loops(fn) if lo < j and k < hi]
    if len(timed) != 1:
        raise SystemExit(f"count: {fn.name} has {len(timed)} timed loops, expected 1")
    j, k = timed[0]
    return Counter(ins.name for ins in fn.insns[j:k + 1])


def table(path: Path) -> dict:
    fns = G.functions(path)
    out = {}
    for name, (words, what, folded) in ARMS.items():
        picked, missing = G.select(fns, [name])
        if missing:
            continue
        body = loop_body(picked[name][0])
        total = sum(body.values())
        fold = sum(min(body.get(op, 0), n * folded) for op, n in FOLD.items())
        out[name] = {"what": what, "words": words, "total": total, "fold": fold, "per_word": total / words,
                     "production_per_word": (total - fold) / words, "histogram": dict(body.most_common())}
        if not folded:
            out[name]["x"] = 4 * total / words
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--pin", type=Path, help="expect.json: kernel -> histogram; gate each kernel against it")
    a = ap.parse_args(argv)
    t = table(a.binary)
    print(f"{'kernel':<12} {'words':>5} {'instr':>6} {'a word':>7} {'fold':>5} {'production a word':>18}  what")
    for name, r in t.items():
        x = f"  x = {r['x']:.2f}" if "x" in r else ""
        print(f"{name:<12} {r['words']:>5} {r['total']:>6} {r['per_word']:>7.3f} {r['fold']:>5} "
              f"{r['production_per_word']:>18.3f}  {r['what']}{x}")
    rc = 0
    if a.pin:
        pins = json.loads(a.pin.read_text())
        for name, hist in pins.items():
            v = G.gate(a.binary, [name], [hist])
            ok = v["ok"]
            print(f"GATE {name} {'pass' if ok else 'REFUSED'}")
            for why in G.refusals(v):
                print("  " + why)
            rc |= 0 if ok else G.EXIT_FAIL
            if name in t:
                t[name]["gate"] = v
    if a.json:
        a.json.write_text(json.dumps(t, indent=1, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
