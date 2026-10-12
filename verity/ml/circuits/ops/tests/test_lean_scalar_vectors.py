"""The Lean transcriptions of `verity.ml.circuits.ops.scalar`'s primitives (`Security/Definitions/Core/Scalar.lean`) and
`verity.ml.circuits.ops.prims`' casts (`Verity/Protocol/Prims.lean`) are held to the Python by vectors: `Verity/SecurityProofs/Scalar/
Vectors.lean` and `Verity/SecurityProofs/Prims/Vectors.lean` state each one as a theorem the kernel checks against the
Lean, and this test checks the same vectors against the Python: each primitive's evaluator, and `evaluate_batch` (its numpy
kernel, where one is registered). A change to either side's semantics then fails a build."""
from __future__ import annotations

import re
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np

from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops import scalar as S
from verity.ml.circuits.silicon.cast import E2M1_UP

LEAN = Path(__file__).resolve().parents[5] / "Security" / "Proofs" / "Core"
FILES = (LEAN / "Scalar" / "Vectors.lean", LEAN / "Prims" / "Vectors.lean")
#: each Lean function, the primitive it transcribes, and the fewest vectors its theorems hold
PRIMITIVES = {
    "bitAnd": (S.BitAnd, 4), "f32Fabs": (S.F32Fabs, 20), "f32Neg": (S.F32Neg, 20), "f32Fmaxf": (S.F32Fmaxf, 40),
    "i32Le": (S.I32Le, 24), "i32Add": (S.I32Add, 12), "selectI32": (S.SelectI32, 8), "selectBf16": (S.SelectBf16, 8),
    "f2fpBf16": (P.F2fpBf16, 60), "f32ToE4m3Sat": (P.F32ToE4m3Sat, 1000), "f32ToE2m1Sat": (P.F32ToE2m1Sat, 80),
}
VECTOR = re.compile(r"^theorem \w+ : (\w+)((?: 0x[0-9a-f]+)+) = (0x[0-9a-f]+) := by decide \+kernel$", re.M)
E4M3_ALL = re.compile(r"^theorem e4m3ToF32_all : \(List\.range 256\)\.map e4m3ToF32 = \[([^\]]*)\] := by\n  decide \+kernel$",
                      re.M)
BF16_ALL = ("theorem bf16ToF32_all : (List.range 0x10000).all (fun a => bf16ToF32 a == a <<< 16) = true := by\n"
            "  decide +kernel\n")


def _vectors() -> dict[str, list[tuple[tuple[int, ...], int]]]:
    out: dict[str, list[tuple[tuple[int, ...], int]]] = {}
    for path in FILES:
        text = path.read_text()
        found = VECTOR.findall(text)
        exhaustive = len(E4M3_ALL.findall(text)) + text.count(BF16_ALL)
        # every theorem of the files is one this test reads: none is skipped by a form the patterns miss
        assert len(found) + exhaustive == len(re.findall(r"^theorem ", text, re.M)), path
        for fn, args, word in found:
            out.setdefault(fn, []).append((tuple(int(a, 16) for a in args.split()), int(word, 16)))
    return out


def _f32(word: int) -> Fraction:
    return Fraction(struct.unpack(">f", struct.pack(">I", word))[0])


def _word(value: Fraction) -> int:
    word = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    assert _f32(word) == value
    return word


def test_lean_scalar_vectors_are_the_python() -> None:
    vectors = _vectors()
    assert set(vectors) == set(PRIMITIVES)
    for fn, rows in vectors.items():
        prim, least = PRIMITIVES[fn]
        assert len(rows) >= least, (fn, len(rows))
        widths = [t.w for _, t in prim.params]
        for args, word in rows:
            assert all(0 <= a < 1 << w for a, w in zip(args, widths, strict=True)), (fn, args)
            assert 0 <= word < 1 << prim.ret.w, (fn, args)
            assert prim.evaluate(*args) == word, (fn, [hex(a) for a in args])
        columns = [np.array(col, dtype=np.uint64) for col in zip(*(args for args, _ in rows))]
        assert evaluate_batch(prim, *columns).tolist() == [word for _, word in rows], fn


def test_lean_exhaustive_casts_are_the_python() -> None:
    text = FILES[1].read_text()
    (table,) = E4M3_ALL.findall(text)
    words = [int(w, 16) for w in table.split(",")]
    assert words == [P.E4m3ToF32.evaluate(c) for c in range(256)]
    assert evaluate_batch(P.E4m3ToF32, np.arange(256)).tolist() == words
    assert text.count(BF16_ALL) == 1
    every = np.arange(1 << 16)
    assert [P.Bf16ToF32.evaluate(a) for a in range(1 << 16)] == (every << 16).tolist()
    assert evaluate_batch(P.Bf16ToF32, every).tolist() == (every << 16).tolist()


def test_lean_scalar_vectors_cover_the_boundaries() -> None:
    vectors = _vectors()
    inputs = {fn: {args for args, _ in rows} for fn, rows in vectors.items()}
    outputs = {fn: {word for _, word in rows} for fn, rows in vectors.items()}
    signs = (0, 0x80000000)

    # E4M3: the word below, at and above every tie between adjacent codes and the 448|480 tie, of both signs; every
    # code's own value; every code as a result but 0xFF (a NaN of either sign is 0x7F)
    codes = [_f32(P.E4m3ToF32.evaluate(c)) for c in range(0x7F)]
    ties = [(codes[c] + codes[c + 1]) / 2 for c in range(0x7E)] + [Fraction(464)]
    want = {((_word(t) + d) | s,) for t in ties for d in (-1, 0, 1) for s in signs}
    want |= {(_word(v) | s,) for v in codes for s in signs}
    assert want <= inputs["f32ToE4m3Sat"]
    assert outputs["f32ToE4m3Sat"] == set(range(0x100)) - {0xFF}

    # E2M1: the word below, at and above every boundary of E2M1_UP, of both signs; every code as a result
    assert {((b + d) | s,) for b, _ in E2M1_UP for d in (-1, 0, 1) for s in signs} <= inputs["f32ToE2m1Sat"]
    assert outputs["f32ToE2m1Sat"] == set(range(0x10))

    # BF16 rounding: below, on and above a tie, after an even and after an odd significand, of both signs, from a subnormal
    # and up to an infinity
    bf16 = inputs["f2fpBf16"]
    for high in (0x3F80, 0x3F81, 0xBF80, 0xBF81, 0x0000, 0x0001, 0x7F7F):
        assert {((high << 16) | low,) for low in (0x7FFF, 0x8000, 0x8001)} <= bf16, hex(high)
    assert {0x7FFF, 0x7F80, 0xFF80, 0x0000, 0x8000} <= outputs["f2fpBf16"]
    finite = [word for (a,), word in vectors["f2fpBf16"] if a & 0x7F800000 != 0x7F800000]
    assert {0x7F80, 0xFF80} <= set(finite)

    # NaN payloads of both signs, quiet and signalling, through every f32 operation
    nans = {0x7FC00000, 0x7F800001, 0xFFC00000, 0xFF800001, 0xFFFFFFFF}
    for fn in ("f32Fabs", "f32Neg", "f2fpBf16", "f32ToE4m3Sat", "f32ToE2m1Sat"):
        assert {(n,) for n in nans} <= inputs[fn], fn

    # fmaxf: every pair of signed zeros, NaN on either side and on both, both infinities
    fmax = inputs["f32Fmaxf"]
    assert {(a, b) for a in signs for b in signs} <= fmax
    assert {(0x7FC00000, 0x3F800000), (0x3F800000, 0x7FC00000), (0x7FC00000, 0x7FC00000)} <= fmax
    assert {(0xFF800000, 0x7F800000), (0x7F800000, 0xFF800000)} <= fmax

    # i32: the extremes, and both orders across the sign
    MIN, MAX = 0x80000000, 0x7FFFFFFF
    assert {(MIN, MAX), (MAX, MIN), (0xFFFFFFFF, 0), (0, 0xFFFFFFFF), (MIN, MIN), (MAX, MAX)} <= inputs["i32Le"]
    assert {(MAX, 1), (MIN, MIN), (0xFFFFFFFF, 0xFFFFFFFF), (MIN, 0xFFFFFFFF)} <= inputs["i32Add"]

    # bits and selects: both conditions
    assert inputs["bitAnd"] == {(a, b) for a in (0, 1) for b in (0, 1)}
    for fn in ("selectI32", "selectBf16"):
        assert {args[0] for args in inputs[fn]} == {0, 1}, fn
