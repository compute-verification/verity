"""The case list: the Ada golden records and structured cases that exercise the relation's corners.

A case is ``(name, (acc, a, b))`` for one k16 step: ``acc`` an FP32 word, ``a``/``b`` sixteen BF16 words.  The golden
records carry the measured ``d``; the structured cases are checked against ``tc_dot_total`` (equal to
``silicon.tc_dot`` on the finite domain).  Every structured case is finite unless its name says otherwise.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import verity
from verity.ml.circuits.silicon.silicon import Pipeline

__all__ = ["GOLDEN", "Case", "golden_cases", "structured_cases", "words"]

Case = tuple[str, tuple[int, tuple[int, ...], tuple[int, ...]]]
GOLDEN = Path(verity.__file__).parents[0] / "ml" / "circuits" / "ops" / "tests" / "fixtures" / "golden" / "ada_bf16_m16n8k16.json"


def words(packed: str) -> tuple[int, ...]:
    return tuple(int(packed[i:i + 4], 16) for i in range(0, len(packed), 4))


def golden_cases() -> tuple[list[Case], dict[str, int]]:
    """The 360 Ada records as cases, plus ``name -> measured d``."""
    doc = json.loads(GOLDEN.read_text())
    cases, measured = [], {}
    for i, rec in enumerate(doc["records"]):
        name = f"golden[{i}]:{rec['family']}"
        cases.append((name, (int(rec["acc"], 16), words(rec["a"]), words(rec["b"]))))
        measured[name] = int(rec["d"], 16)
    return cases, measured


def bf16(sign: int, e: int, m: int) -> int:
    return (sign << 15) | (e << 7) | m


def fp32(sign: int, E: int, m: int) -> int:
    return (sign << 31) | (E << 23) | m


MAXF = bf16(0, 254, 0x7F)      # largest finite BF16
ONE = bf16(0, 127, 0)
TINY = bf16(0, 0, 1)           # smallest subnormal


def _step(k: int, acc: int, pairs: dict[int, tuple[int, int]], fill: tuple[int, int] = (0, 0)) -> tuple[int, tuple, tuple]:
    a, b = [fill[0]] * k, [fill[1]] * k
    for j, (x, y) in pairs.items():
        a[j], b[j] = x, y
    return acc, tuple(a), tuple(b)


def structured_cases(pipe: Pipeline, *, n_random: int = 48, seed: int = 0) -> list[Case]:
    k = pipe.k
    g2 = pipe.groups[0] if len(pipe.groups) > 1 else 0    # first index of the second group (0 = single group)
    cases: list[Case] = []
    add = lambda name, acc, pairs, fill=(0, 0): cases.append((name, _step(k, acc, pairs, fill)))  # noqa: E731

    add("all_zero", 0, {})
    add("neg_zero_acc", 0x80000000, {})
    add("acc_only", 0x3F800000, {})
    add("max_shift_d506", 0, {0: (MAXF, MAXF), 1: (TINY, TINY)})
    add("max_shift_with_acc", 0x3F800000, {0: (MAXF, MAXF), 1: (TINY, TINY)})
    add("cancel_acc_vs_product", 0x3F800000, {0: (ONE, ONE | 0x8000)})
    add("cancel_two_products", 0, {0: (ONE, ONE), 1: (ONE, ONE | 0x8000)})
    add("cancel_huge_ovf_flag", 0, {0: (MAXF, MAXF), 1: (MAXF, MAXF | 0x8000)})
    add("subnormal_acc_min", fp32(0, 0, 1), {})
    add("subnormal_acc_max", fp32(0, 0, 0x7FFFFF), {})
    add("subnormal_acc_neg_plus_flushed_product", fp32(1, 0, 1), {0: (TINY, TINY)})
    add("subnormal_acc_plus_product_below", fp32(0, 0, 0x400000), {0: (bf16(0, 62, 0), bf16(0, 62, 0))})
    add("subnormal_acc_cancel", fp32(0, 0, 0x400000), {0: (bf16(0, 63, 0), bf16(1, 64, 0))})    # 2^-127 - 2^-127
    add("subnormal_products_only", 0, {0: (TINY, MAXF), 1: (bf16(0, 0, 0x7F), bf16(0, 0, 0x7F))})
    add("max_finite_acc_plus_one", fp32(0, 254, 0x7FFFFF), {0: (ONE, ONE)})
    add("overflow_group0", fp32(0, 254, 0x7FFFFF), {0: (MAXF, MAXF)})
    add("overflow_group0_then_negative", fp32(0, 254, 0x7FFFFF), {0: (MAXF, MAXF), k - 1: (MAXF, MAXF | 0x8000)})
    add("overflow_sum_of_max_products", 0, {j: (MAXF, MAXF) for j in range(k)})
    add("negative_overflow", fp32(1, 254, 0x7FFFFF), {0: (MAXF, MAXF | 0x8000)})
    add("L_above_W_all_max", 0, {j: (MAXF, MAXF) for j in range(pipe.groups[0])})
    add("L_above_W_with_max_acc", fp32(0, 254, 0), {j: (bf16(0, 127, 0x7F), bf16(0, 127, 0x7F)) for j in range(pipe.groups[0])})
    add("extreme_product_exponents", 0, {0: (MAXF, TINY), 1: (TINY, MAXF), 2: (bf16(0, 1, 0), bf16(0, 1, 0))})
    if g2:
        add("group1_only", 0, {g2: (ONE, ONE), g2 + 1: (bf16(0, 120, 3), bf16(1, 130, 0x55))})
        add("group0_result_cancels_in_group1", 0, {0: (ONE, ONE), g2: (ONE, ONE | 0x8000)})
        add("group0_zero_group1_tiny", 0, {g2: (TINY, bf16(0, 127, 0))})
        add("overflow_group1_only", fp32(0, 200, 0), {j: (MAXF, MAXF) for j in range(g2, k)})

    # truncation to zero: |total| = 1 at the adder's grid with E* = -126 flushes below 2^-149 (constructed per width)
    if pipe.width == 25:
        add("flush_total_1", fp32(0, 1, 0), {0: (bf16(0, 63, 0x59), bf16(1, 64, 0x17)), 1: (bf16(0, 0, 7), bf16(1, 116, 0x12))})
    elif pipe.width == 26:
        add("flush_total_1", fp32(0, 1, 0), {0: (bf16(0, 63, 0x59), bf16(1, 64, 0x17)), 1: (bf16(0, 0, 11), bf16(1, 115, 0x3A))})

    rng = random.Random(seed)

    def rword(lo_e: int, hi_e: int) -> int:
        return bf16(rng.getrandbits(1), rng.randint(lo_e, hi_e), rng.getrandbits(7))

    def racc(lo_E: int, hi_E: int) -> int:
        return fp32(rng.getrandbits(1), rng.randint(lo_E, hi_E), rng.getrandbits(23))

    for i in range(n_random):
        kind = i % 4
        if kind == 0:      # ordinary magnitudes
            acc, a, b = racc(100, 140), [rword(110, 140) for _ in range(k)], [rword(110, 140) for _ in range(k)]
        elif kind == 1:    # wide spread: many terms flushed by alignment
            acc, a, b = racc(1, 254), [rword(0, 254) for _ in range(k)], [rword(0, 254) for _ in range(k)]
        elif kind == 2:    # subnormal accumulator and tiny operands
            acc, a, b = racc(0, 2), [rword(0, 70) for _ in range(k)], [rword(55, 70) for _ in range(k)]
        else:              # near the top
            acc, a, b = racc(240, 254), [rword(120, 134) for _ in range(k)], [rword(240, 254) for _ in range(k)]
        cases.append((f"random[{i}]:kind{kind}", (acc, tuple(a), tuple(b))))
    return cases
