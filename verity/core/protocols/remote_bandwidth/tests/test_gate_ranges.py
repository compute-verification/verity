"""The primitives' output intervals (`Accounting.Guarantees.Gates.FmaxfRange`, `FminfRange`, `RzByteRange`), held to the
Python they state.

`Accounting.Gates.f32Fmaxf`, `f32Fminf` and `f32RzByte` (Security/Definitions/Accounting/Gates/Ranges.lean) are the
evaluators the theorems are over, and `fmaxfRule`, `fminfRule` and `rzByteRule` property/v1's interval rules for them
(`stacking.RANGES`). Security/Proofs/Accounting/Gates.lean carries rows of each that the kernel evaluates the Lean
functions on (`f32MaxMin_f32Rows`, `f32RzByte_rzRows`, `rules_ruleRows`, `rzByteRule_rzRuleRows`), and the tests here
recompute each row from the Python: `scalar.F32Fmaxf` and `F32Fminf`, `cvt.rzi.u32.f32`'s low byte by the float's
truncation (the evaluator of record, `dsv4_qnorm_insert.f32_to_u32_rz_byte`, is in `verity_vllm`, which `verity` doesn't
import), and property/v1's rules where `verity.ml.circuits.stacking` exists. An f32 is its encoding read as s32.

`Accounting.Gates.exp2Lut127` (Ranges.lean), the table `Exp2Lut127Range` and `LutPvWindow` are over, is held to vLLM's
LUT softmax table on all 256 entries: `fixtures/exp2_lut127.json` pins `engine.int8_attention.LUT` as vLLM
writes it, and its words' sha256 is the one `verity_flock.tail_pieces.EXP2_LUT127_SHA256` pins there.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.ops.scalar import F32Fmaxf, F32Fminf

ROOT = Path(__file__).resolve().parents[5]
LEAN_GATES = ROOT / "Security" / "Proofs" / "Accounting" / "Gates.lean"
LEAN_RANGES = ROOT / "Security" / "Definitions" / "Accounting" / "Gates" / "Ranges.lean"
LUT_FIXTURE = Path(__file__).parent / "fixtures" / "exp2_lut127.json"
#: `tail_pieces.EXP2_LUT127_SHA256`: the sha256 of the table's 256 little-endian u32 words
EXP2_LUT127_SHA256 = "6eb8e9b3478db58b4e5a711f31968394f7e93fa4e1df2e0d801eab92a5f66f84"
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


def lean_exp2_lut127() -> list[int]:
    """`exp2Lut127` as Ranges.lean writes it: a list literal, then `++ List.replicate n v`."""
    m = re.search(r"\ndef exp2Lut127 : List ℤ :=\s*\[([^\]]*)\]\s*\+\+\s*List\.replicate (\d+) (-?\d+)\n",
                  LEAN_RANGES.read_text())
    assert m, "exp2Lut127's form in Ranges.lean changed: update this reader"
    return [int(t) for t in m.group(1).replace("\n", " ").split(",")] + [int(m.group(3))] * int(m.group(2))


def test_exp2_lut127_agrees_with_vllm() -> None:
    fixture = json.loads(LUT_FIXTURE.read_text())
    vllm = fixture["entries"]
    assert hashlib.sha256(np.asarray(vllm, dtype="<u4").tobytes()).hexdigest() == EXP2_LUT127_SHA256
    assert vllm == np.rint(127.0 * np.exp2(-np.arange(256) / 16.0)).astype(np.int64).tolist()
    lean = lean_exp2_lut127()
    assert len(lean) == len(vllm) == 256
    assert [t for t in range(256) if lean[t] != vllm[t]] == []
    assert (min(lean), max(lean)) == (0, 127)


def test_rules_agree_with_property_v1() -> None:
    stacking = pytest.importorskip("verity.ml.circuits.stacking")
    fmaxf, fminf, rz = (stacking.RANGES[f"{g}_v1"] for g in ("F32Fmaxf", "F32Fminf", "F32ToU32RzByte"))
    for a, b, mx, mn in lean_rows("ruleRows"):
        assert fmaxf(a, b) == mx, (a, b)
        assert fminf(a, b) == mn, (a, b)
    for a, o in lean_rows("rzRuleRows"):
        assert rz(a) == o, a
