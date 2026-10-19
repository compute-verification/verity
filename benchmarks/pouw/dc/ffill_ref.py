"""ffill: the H200's FFMA filler beside the LI-CP-W GEMM (machine_model's sm_90 `dither-2s-licpw-redux-l32` [filler]),
in Python: the receipt a thread's two chains reach from its committed words and the round's coins, every FMA exact.

A thread of the LI-CP-W warpgroup holds 128 s32 accumulator words of its m64n256 tile; after each MMA (a k32 segment)
they are that segment's committed prefix words. On three of them, `REGS`, it takes one filler step each, step n of the
tile being n = 3·segment + j:

    u = IMAD(w, one, BIAS)            `one` a runtime 1, so ptxas keeps it on the IMAD pipe (an IADD3 would load ALU's)
    f = u's bits read as binary32     1.5·2^23 + w exactly for |w| < 2^22, and a positive normal number for |w| < 2^29
    x = fma.rn.f32(f, ±s, x)          the sign alternates step by step, so each chain walks on differences of words
    y = fma.rn.f32(f, ±t, y)

Nine lane instructions a thread a MMA over its 128 words: 0.0703125 a committed word, two FFMA to one IMAD (`PER_WORD`,
`MIX`). s, t, x0 and y0 come from the round's coins and the tile (`coins`): s and t are distinct odd significands in
(1, 2), 24 significant bits each, so f·s needs about 48 and every step rounds; no chain is an affine map of the words.
For |w| < 2^22 (a uniform s8 dot product's spread at K = 8192 is about 2^19) f·s ≥ 2^23 while |x| < 2^47 over any
tile, so every step moves both chains: a step is absorbed only when f·s is under half an ulp of x. Past 2^22 f's binade
follows w's high bits and one lane's step can be absorbed, which saves an issue slot only if all 32 lanes of its warp
absorb that step at once. No chain of a tile overflows for |w| < 2^29 (`WORD_BOUND`).
At the tile's end the li fold absorbs bits(x) and bits(y) as two more words of its last segment, so LI-CP-W's last
checkpoint commits them and the check that recomputes a drawn tile recomputes them with it (PinnedFillBlocksDeficit).

Every FMA is `verity.ml.circuits.ops.fp32.F32FmaV2`, the registered FFMA: the exact rational a·b + c rounded once to
nearest even, subnormals kept (`fma.rn.f32` without `.ftz`, SASS `FFMA`), every NaN result 0x7FFFFFFF, the word sm_90
writes. It is never a float64 product summed and cast to binary32, which rounds twice (`DOUBLE_ROUNDING` is a case where
that differs). ffill.cu runs the same steps on the GPU through an explicit `fma.rn.f32`, which neither contraction nor
-fmad touches, and `check` holds its dumps to this file:

    python3 ffill_ref.py check DIR     DIR/fma.bin and DIR/chain.bin from `ffill fma` and `ffill chain`; DIR/check.json
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from verity.ml.circuits.ops.fp32 import F32_NAN_PTX, F32FmaV2

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
SIGN = 0x80000000

BIAS = 0x4B400000                 # 1.5·2^23's word: f = BIAS + w read as binary32
REGS = (0, 43, 86)                # the accumulator registers of IGMMA.64x256x32's 128 a thread the filler reads
WORDS_PER_MMA = 128               # a thread's committed words a MMA (k32 segment)
PER_WORD = 3 * len(REGS) / WORDS_PER_MMA   # lane instructions a committed word: an IMAD and two FFMA a step
MIX = {"FFMA": 2, "IMAD": 1}
FF_KEYS = 0x46464D4146494C4C      # "FFMAFILL"
WORD_BOUND = 1 << 29              # |w| below it: f is a positive normal number and no chain of a tile overflows
DOUBLE_ROUNDING = (0x40400000, 0x3F800001, 0xA1800000, 0x40400001)   # 3 · (1 + 2^-23) − 2^-60; float64 then cast: 0x40400002


def fma(a: int, b: int, c: int) -> int:
    """`fma.rn.f32` on words: RN(a·b + c) rounded once, subnormals kept, every NaN 0x7FFFFFFF."""
    return F32FmaV2.evaluate(a & M32, b & M32, c & M32)


def smix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def seed(c0: int, c1: int, tile: int) -> int:
    """The tile's key from the round's coins, as dfill's and lfill's (`dseed`)."""
    return smix(smix(c0 ^ smix(tile)) ^ c1)


def coins(c0: int, c1: int, tile: int) -> tuple[int, int, int, int]:
    """(s, t, x0, y0) as binary32 words: s and t odd significands in (1, 2), t ≠ s; x0 and y0 in [1, 2)."""
    k = seed(c0, c1, tile)
    r, q = smix(k ^ FF_KEYS), smix(k ^ FF_KEYS ^ 1)
    s = 0x3F800001 | (r & 0x7FFFFE)
    t = 0x3F800001 | ((r >> 32) & 0x7FFFFE)
    if t == s:
        t ^= 2
    return s, t, 0x3F800000 | (q & 0x7FFFFF), 0x3F800000 | ((q >> 32) & 0x7FFFFF)


def operand(w: int) -> int:
    """IMAD(w, 1, BIAS): the word's binary32 operand."""
    return (w + BIAS) & M32


def step(x: int, y: int, w: int, s: int, t: int, n: int) -> tuple[int, int]:
    """Filler step n on word w: s and t carry the sign (−1)^n, FFMA's operand negation."""
    f, sg = operand(w), (n & 1) << 31
    return fma(f, s ^ sg, x), fma(f, t ^ sg, y)


def chain(words, s: int, t: int, x0: int, y0: int) -> tuple[int, int]:
    """A thread's (x, y) after a step on each word in order (step n on the n-th)."""
    x, y = x0, y0
    for n, w in enumerate(words):
        x, y = step(x, y, w, s, t, n)
    return x, y


def thread_words(segments) -> list[int]:
    """The words a thread's chains take, in step order, from its accumulator words after each MMA (128 a segment)."""
    return [int(seg[r]) for seg in segments for r in REGS]


def receipt(c0: int, c1: int, tile: int, words) -> tuple[int, int]:
    """A thread's receipt words (bits(x), bits(y)) on its tile, from the round's coins."""
    return chain(words, *coins(c0, c1, tile))


# ---- the GPU's dumps (ffill.cu) ----------------------------------------------------------------------------------------

def check_fma(path: Path) -> dict:
    """fma.bin: little-endian u32 quads (a, b, c, d), d the GPU's fma.rn.f32. Every d must be `fma(a, b, c)`."""
    raw = path.read_bytes()
    n, bad, first, nan, sub = len(raw) // 16, 0, None, 0, 0
    for i in range(n):
        a, b, c, d = struct.unpack_from("<4I", raw, 16 * i)
        want = fma(a, b, c)
        nan += want == F32_NAN_PTX
        sub += (want & 0x7F800000) == 0 and (want & 0x7FFFFF) != 0
        if d != want:
            bad += 1
            first = first or {"a": f"{a:#010x}", "b": f"{b:#010x}", "c": f"{c:#010x}", "gpu": f"{d:#010x}",
                              "ref": f"{want:#010x}"}
    return {"cases": n, "bad": bad, "first": first, "nan_results": nan, "subnormal_results": sub}


def check_chain(path: Path) -> dict:
    """chain.bin: a header (u64 c0, c1, tile; u32 threads, steps; u32 s, t, x0, y0 as the GPU derived them), the words
    (threads × steps s32, thread-major), then each thread's (x, y). The coins and every receipt must be this file's."""
    raw = path.read_bytes()
    c0, c1, tile, threads, steps = struct.unpack_from("<3Q2I", raw, 0)
    gpu_coins = struct.unpack_from("<4I", raw, 32)
    off = 48
    words = struct.unpack_from(f"<{threads * steps}i", raw, off)
    off += 4 * threads * steps
    got = struct.unpack_from(f"<{2 * threads}I", raw, off)
    want_coins = coins(c0, c1, tile)
    bad, first = 0, None
    for u in range(threads):
        want = chain(words[u * steps:(u + 1) * steps], *want_coins)
        if (got[2 * u], got[2 * u + 1]) != want:
            bad += 1
            first = first or {"thread": u, "gpu": [f"{v:#010x}" for v in got[2 * u:2 * u + 2]],
                              "ref": [f"{v:#010x}" for v in want]}
    return {"threads": threads, "steps": steps, "coins_ok": tuple(gpu_coins) == want_coins, "bad": bad, "first": first,
            "max_word": max((abs(w) for w in words), default=0)}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "check":
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    d = Path(argv[1])
    out = {"fma": check_fma(d / "fma.bin"), "chain": check_chain(d / "chain.bin")}
    out["ok"] = (out["fma"]["cases"] > 0 and out["fma"]["bad"] == 0 and out["chain"]["bad"] == 0
                 and out["chain"]["coins_ok"] and out["chain"]["threads"] > 0)
    (d / "check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
