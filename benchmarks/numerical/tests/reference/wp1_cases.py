"""Case sources shared by the WP1 tests and the recorded exhaustive driver: fixture loaders, structured and random transition-unit
cases, boundary groups for ``group_sum``.  Stdlib only; deterministic (fixed seeds)."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import NamedTuple

from verity.ml.circuits.silicon.silicon import Term
from verity_numerical.reference.hawkeye_close import E_ZERO, Gfloat, is_finite_bf16, is_finite_fp32

ROOT = Path(__file__).resolve().parents[4]
GOLDEN = ROOT / "verity/ml/circuits/ops/tests/fixtures/golden/ada_bf16_m16n8k16.json"
SPECIAL = {
    device: ROOT / f"verity/ml/circuits/ops/tests/fixtures/tc-total-2026-09-07/{device}/mma_special.reduced.json.gz"
    for device in ("a100", "4090")
}


class Case(NamedTuple):
    """One transition-unit case; ``d`` is the hardware word when the case was measured, else ``None``."""

    source: str
    acc: int
    a: tuple[int, ...]
    b: tuple[int, ...]
    d: int | None


def words(hex_text: str, width: int = 4) -> tuple[int, ...]:
    return tuple(int(hex_text[i : i + width], 16) for i in range(0, len(hex_text), width))


def hexrec(case: Case) -> dict[str, str]:
    return {
        "acc": f"{case.acc:08x}",
        "a": "".join(f"{w:04x}" for w in case.a),
        "b": "".join(f"{w:04x}" for w in case.b),
        "d": "" if case.d is None else f"{case.d:08x}",
    }


def all_finite(case: Case) -> bool:
    return is_finite_fp32(case.acc) and all(is_finite_bf16(w) for w in (*case.a, *case.b))


# --- fixtures ----------------------------------------------------------------------------------------------------------------


def golden_records() -> list[Case]:
    """The 360 RTX 4090 golden records (every one is all-finite on input; 114 saturate to an infinity on output)."""

    data = json.loads(GOLDEN.read_text())
    assert data["pipeline"] == "ada_bf16_m16n8k16" and data["k"] == 16
    return [
        Case("golden/" + r["family"], int(r["acc"], 16), words(r["a"]), words(r["b"]), int(r["d"], 16))
        for r in data["records"]
    ]


def special_records(device: str) -> list[Case]:
    """Every all-finite record of the special-value campaign on ``device``: the ``cases`` list and the collapsed ``pos`` list."""

    data = json.loads(gzip.open(SPECIAL[device]).read())
    assert data["instruction"] == "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
    out = []
    for r in data["cases"]:
        case = Case(f"{device}/cases/{r['family']}", int(r["acc"], 16), words(r["a"]), words(r["b"]), int(r["d"], 16))
        if all_finite(case):
            out.append(case)
    for r in data["pos"]:
        assert len(r["d"]) == 1, r["name"]  # no position dependence was measured
        case = Case(f"{device}/pos", int(r["acc"], 16), words(r["a"]), words(r["b"]), int(r["d"][0], 16))
        if all_finite(case):
            out.append(case)
    return out


# --- structured transition-unit cases ------------------------------------------------------------------------------------------

BF16_WORDS = {
    "+0": 0x0000,
    "-0": 0x8000,
    "min_sub": 0x0001,
    "max_sub": 0x007F,
    "min_norm": 0x0080,
    "-min_norm": 0x8080,
    "tiny": 0x0400,  # 2^-119
    "2^-60": 0x2180,
    "0.5": 0x3F00,
    "1": 0x3F80,
    "-1": 0xBF80,
    "1+ulp": 0x3F81,
    "1.99": 0x3FFF,
    "2": 0x4000,
    "3": 0x4040,
    "2^23": 0x4B00,
    "2^24": 0x4B80,
    "2^60": 0x5D80,
    "big": 0x7E80,  # 2^126
    "max": 0x7F7F,
    "-max": 0xFF7F,
}

FP32_WORDS = {
    "+0": 0x00000000,
    "-0": 0x80000000,
    "min_sub": 0x00000001,
    "max_sub": 0x007FFFFF,
    "min_norm": 0x00800000,
    "-min_norm": 0x80800000,
    "2^-100": 0x0D800000,
    "2^-24": 0x33800000,
    "1": 0x3F800000,
    "-1": 0xBF800000,
    "1+ulp": 0x3F800001,
    "1.5": 0x3FC00000,
    "16": 0x41800000,
    "-16": 0xC1800000,
    "2^23": 0x4B000000,
    "2^23+1": 0x4B000001,
    "2^24": 0x4B800000,
    "2^60": 0x5D800000,
    "2^100": 0x71800000,
    "max": 0x7F7FFFFF,
    "-max": 0xFF7FFFFF,
    "0x3FFFFFFF": 0x3FFFFFFF,
    "0x7F000000": 0x7F000000,
}


def structured_cases() -> list[Case]:
    """Deterministic structured transition-unit cases from the special-word vocabularies.

    For every accumulator word and every pair of vocabulary words ``(x, y)``: all sixteen lanes ``x * y``; lane 0 only; lane 15 only
    (group 2 only); lane 0 = ``x*y`` and lane 8 = ``x*(-y)`` (cancellation across groups); lanes 0..7 = ``x*y`` and lanes 8..15
    = ``x*(-y)``; lane 0 = ``x*y`` with lane 1 = ``1*(-1)``.
    """

    out = []
    xs = list(BF16_WORDS.values())
    neg = lambda w: w ^ 0x8000
    one, minus_one = BF16_WORDS["1"], BF16_WORDS["-1"]
    zero = [0] * 16
    for acc in FP32_WORDS.values():
        for x in xs:
            for y in xs:
                variants = []
                variants.append(([x] * 16, [y] * 16))
                a = zero[:]
                b = zero[:]
                a[0], b[0] = x, y
                variants.append((a, b))
                a = zero[:]
                b = zero[:]
                a[15], b[15] = x, y
                variants.append((a, b))
                a = zero[:]
                b = zero[:]
                a[0], b[0], a[8], b[8] = x, y, x, neg(y)
                variants.append((a, b))
                variants.append(([x] * 16, [y] * 8 + [neg(y)] * 8))
                a = zero[:]
                b = zero[:]
                a[0], b[0], a[1], b[1] = x, y, one, minus_one
                variants.append((a, b))
                for a, b in variants:
                    out.append(Case("structured", acc, tuple(a), tuple(b), None))
    return out


# --- random transition-unit cases ------------------------------------------------------------------------------------------------


def _random_bf16(rng: random.Random, exp_lo: int, exp_hi: int) -> int:
    return (rng.getrandbits(1) << 15) | (rng.randint(exp_lo, exp_hi) << 7) | rng.getrandbits(7)


def _random_fp32(rng: random.Random, exp_lo: int, exp_hi: int) -> int:
    return (rng.getrandbits(1) << 31) | (rng.randint(exp_lo, exp_hi) << 23) | rng.getrandbits(23)


def random_cases(n: int, seed: int) -> list[Case]:
    """``n`` random all-finite cases in six equally weighted mixtures (uniform encodings; magnitudes clustered so the products and
    the accumulator interact; zero accumulator; subnormal-heavy; near overflow; cancellation-heavy with a random accumulator)."""

    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = i % 6
        if kind == 0:
            a = [_random_bf16(rng, 0, 254) for _ in range(16)]
            b = [_random_bf16(rng, 0, 254) for _ in range(16)]
            acc = _random_fp32(rng, 0, 254)
        elif kind == 1:
            centre = rng.randint(20, 235)
            a = [_random_bf16(rng, centre - 12, centre + 12) for _ in range(16)]
            b = [_random_bf16(rng, max(0, 254 - centre - 12), min(254, 254 - centre + 12)) for _ in range(16)]
            acc = _random_fp32(rng, 110, 145)
        elif kind == 2:
            a = [_random_bf16(rng, 100, 154) for _ in range(16)]
            b = [_random_bf16(rng, 100, 154) for _ in range(16)]
            acc = rng.choice((0x00000000, 0x80000000))
        elif kind == 3:
            a = [_random_bf16(rng, 0, 3) for _ in range(16)]
            b = [_random_bf16(rng, 0, 3) for _ in range(16)]
            acc = _random_fp32(rng, 0, 3)
        elif kind == 4:
            a = [_random_bf16(rng, 245, 254) for _ in range(16)]
            b = [_random_bf16(rng, 245, 254) for _ in range(16)]
            acc = _random_fp32(rng, 250, 254)
        else:
            a = [_random_bf16(rng, 120, 140) for _ in range(16)]
            b = [_random_bf16(rng, 120, 140) for _ in range(16)]
            for k in range(0, 16, 2):
                a[k + 1] = a[k]
                b[k + 1] = b[k] ^ 0x8000
            for k in rng.sample(range(16), 3):
                a[k] = _random_bf16(rng, 90, 130)
            acc = _random_fp32(rng, 100, 130)
        out.append(Case(f"random/{kind}", acc, tuple(a), tuple(b), None))
    return out


# --- boundary groups for group_sum ---------------------------------------------------------------------------------------------


def boundary_groups() -> list[tuple[str, list[Gfloat]]]:
    """Named ``group_sum`` inputs at the edges: maximal shifts, truncation to zero, exact cancellation, near overflow, subnormal
    accumulators, the all-zero group, the single-term group."""

    p = lambda sign, e, sig: Gfloat(sign, e, sig)
    top = 65025 << 9  # 255 * 255 << 9: the largest product significand
    groups = [
        ("all zero", [p(False, E_ZERO, 0)] * 9),
        ("single zero accumulator", [p(False, E_ZERO, 0)]),
        ("single normal", [p(False, 0, 1 << 23)]),
        ("single subnormal 1 bit", [p(True, -126, 1)]),
        ("single subnormal 23 bits", [p(False, -126, (1 << 23) - 1)]),
        ("max shift, small term vanishes", [p(False, 127, (1 << 24) - 1), p(True, -132, 1)]),
        ("shift 24 keeps one bit", [p(False, 0, 1 << 23), p(False, -24, 1 << 24)]),
        ("shift 25 keeps it in the window, the 24-bit truncation drops it", [p(False, 0, 1 << 23), p(False, -25, 1 << 24)]),
        ("shift 26 drops the bit", [p(False, 0, 1 << 23), p(False, -26, 1 << 24)]),
        ("shift 31 vs 32 boundary", [p(False, 0, 1 << 23), p(False, -31, top), p(False, -32, top)]),
        ("truncation to zero, positive", [p(False, -140, 1 << 9)]),
        ("truncation to zero, negative", [p(True, -140, 1 << 9)]),
        ("just above truncation", [p(False, -135, 1 << 9)]),
        ("exact cancellation", [p(False, 0, 1 << 23), p(True, 0, 1 << 23)]),
        ("cancellation to one ulp", [p(False, 0, (1 << 23) | 1), p(True, 0, 1 << 23)]),
        ("cancellation of products", [p(False, 10, top), p(True, 10, top), p(False, -100, 1 << 9)]),
        ("near overflow, exponent 127 exact", [p(False, 127, (1 << 24) - 1)]),
        ("overflow to 128, NaN word", [p(False, 127, (1 << 24) - 1), p(False, 127, (1 << 24) - 1)]),
        ("overflow to 128, infinity word", [p(False, 127, 1 << 23), p(False, 127, 1 << 23)]),
        ("overflow to 129 (masking)", [p(False, 127, (1 << 24) - 1)] * 4),
        ("nine maximal products", [p(False, 254, top)] * 9),
        ("nine maximal products, alternating signs", [p(i % 2 == 1, 254, top) for i in range(9)]),
        ("subnormal accumulator plus tiny products", [p(False, -126, 0x1234), p(False, -140, 1 << 9), p(True, -150, 1 << 9)]),
        ("subnormal result from normal terms", [p(False, -126, 1 << 23), p(True, -126, (1 << 23) - 3)]),
        ("floor: products below E_ZERO only", [p(False, -200, top), p(True, -252, 1 << 9)]),
        ("floor: term exactly at E_ZERO", [p(False, -132, top)]),
        ("carry out of the 25-bit window", [p(False, 0, (1 << 24) - 1), p(False, 0, (1 << 24) - 1), p(False, 0, 1 << 9)]),
        ("sum needs 29 bits", [p(False, 5, top)] * 8 + [p(False, 5, (1 << 24) - 1)]),
    ]
    return groups


def realise(sign: bool, max_exponent: int, magnitude: int) -> list[Gfloat]:
    """A group whose alignment yields exactly the signed sum ``+-magnitude`` at maximum exponent ``E = max_exponent`` (``E >= E_ZERO``):
    ``v >> 1`` at ``E`` when ``v`` is even; ``(v >> 1) + 1`` at ``E`` minus one unit at ``E - 1`` when ``v`` is odd (so a nonzero term
    at ``E`` fixes the maximum in both the simulator's convention and silicon's).  Lets the normalisation stage be tested through
    the real ``group_sum`` functions on any ``(E, v)``."""

    if magnitude & 1 == 0:
        return [Gfloat(sign, max_exponent, magnitude >> 1)]
    return [Gfloat(sign, max_exponent, (magnitude >> 1) + 1), Gfloat(not sign, max_exponent - 1, 1)]


# --- Term <-> Gfloat -------------------------------------------------------------------------------------------------------------


def term_of(g: Gfloat) -> Term:
    return Term(False, 0, 0) if g.significand == 0 else Term(g.sign, g.exponent, g.significand)


def gfloat_of(t: Term) -> Gfloat:
    return Gfloat(False, E_ZERO, 0) if t.magnitude == 0 else Gfloat(t.negative, t.exponent, t.magnitude)


def canon(g: Gfloat) -> tuple:
    """A zero is a zero (the internal sign of a zero never reaches a word); everything else is the triple."""

    return (0,) if g.significand == 0 else (g.sign, g.exponent, g.significand)


def canon_term(t: Term) -> tuple:
    return (0,) if t.magnitude == 0 else (t.negative, t.exponent, t.magnitude)
