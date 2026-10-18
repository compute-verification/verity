"""The primitives' output intervals (`Accounting.Guarantees.Gates.FmaxfRange`, `FminfRange`, `RzByteRange`), held to the
Python they state.

`Accounting.Gates.f32Fmaxf`, `f32Fminf` and `f32RzByte` (Security/Definitions/Accounting/Gates/Ranges.lean) are the
evaluators the theorems are over, and `fmaxfRule`, `fminfRule` and `rzByteRule` property/v1's interval rules for them
(`stacking.RANGES`). Security/Proofs/Accounting/Gates.lean carries rows of each that the kernel evaluates the Lean
functions on (`f32MaxMin_f32Rows`, `f32RzByte_rzRows`, `rules_ruleRows`, `rzByteRule_rzRuleRows`), and the tests here
recompute each row from the Python: `scalar.F32Fmaxf` and `F32Fminf`, `cvt.rzi.u32.f32`'s low byte by the float's
truncation (the evaluator of record, `dsv4_qnorm_insert.f32_to_u32_rz_byte`, is in `verity_vllm`, which `verity` doesn't
import), and property/v1's rules where `verity.ml.circuits.stacking` exists. An f32 is its encoding read as s32.
"""

from __future__ import annotations

import ast
import math
import struct
from pathlib import Path

import pytest

from verity.ml.circuits.ops.scalar import F32Fmaxf, F32Fminf

LEAN_GATES = Path(__file__).resolve().parents[5] / "Security" / "Proofs" / "Accounting" / "Gates.lean"
M32 = 0xFFFFFFFF


def lean_rows(name: str) -> list[tuple]:
    """The rows of `def name` as Python tuples, `none` as `None` and `some r` as `r`."""
    block = LEAN_GATES.read_text().split(f"\ndef {name} ", 1)[1]
    body = block[block.index(":= [") + 4:block.index("]\n")]
    rows = [ast.literal_eval(line.strip().rstrip(",").replace("some ", "").replace("none", "None"))
            for line in body.splitlines() if line.strip()]
    assert rows, f"no rows read from {name}"
    return rows


def s32(u: int) -> int:
    return u - (1 << 32) if u >> 31 else u


def f32(z: int) -> float:
    return struct.unpack("<f", struct.pack("<I", z & M32))[0]


def rz_byte(z: int) -> int:
    """The low byte of `cvt.rzi.u32.f32`: a NaN, a negative float and `−0` give 0, from `2^32` it saturates."""
    x = f32(z)
    if math.isnan(x) or (z & M32) >> 31:
        return 0
    return 0xFF if x >= 2.0 ** 32 else math.trunc(x) & 0xFF


def test_fmaxf_fminf_agree_with_lean() -> None:
    rows = lean_rows("f32Rows")
    arms = set()
    for a, b, mx, mn in rows:
        assert s32(F32Fmaxf.evaluate(a & M32, b & M32)) == mx, (a, b)
        assert s32(F32Fminf.evaluate(a & M32, b & M32)) == mn, (a, b)
        na, nb = math.isnan(f32(a)), math.isnan(f32(b))
        arms.add("two NaNs" if na and nb else "a NaN" if na or nb else
                 "±0" if f32(a) == f32(b) == 0 and a != b else "ordered")
    assert arms == {"two NaNs", "a NaN", "±0", "ordered"}, arms


def test_rz_byte_agrees_with_lean() -> None:
    rows = lean_rows("rzRows")
    exponents = set()
    for z, byte in rows:
        assert rz_byte(z) == byte, z
        exponents.add(-1 if z < 0 else (z >> 23) & 0xFF)
    assert {-1, 126, 127, 149, 150, 158, 159, 255} <= exponents, exponents


def test_rules_agree_with_property_v1() -> None:
    stacking = pytest.importorskip("verity.ml.circuits.stacking")
    fmaxf, fminf, rz = (stacking.RANGES[f"{g}_v1"] for g in ("F32Fmaxf", "F32Fminf", "F32ToU32RzByte"))
    for a, b, mx, mn in lean_rows("ruleRows"):
        assert fmaxf(a, b) == mx, (a, b)
        assert fminf(a, b) == mn, (a, b)
    for a, o in lean_rows("rzRuleRows"):
        assert rz(a) == o, a
