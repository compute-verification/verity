"""The refusal suite's cases compute what they claim."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import cases as C
import pytest

from verity.circuits.evaluation.reference import evaluate
from verity.circuits.silicon.integer import S32_MAX, S32_MIN

CASES_JSON = Path(__file__).resolve().parents[3] / "benchmarks" / "property_redteam" / "cases.json"


def _ids(cases):
    return [c.id for c in cases]


def _program(case):
    try:
        return case.program()
    except (TypeError, ValueError) as e:
        if case.kind == "honest":
            raise
        pytest.skip(f"the IR now refuses {case.id} at build ({e})")


@pytest.mark.parametrize("case", C.CASES, ids=_ids(C.CASES))
def test_samples_compute_the_claim(case):
    _program(case)
    samples = case.samples()
    assert samples
    for size, inputs, want in samples:
        program = case.build(**(case.size | size))
        assert evaluate(program.fn, *inputs) == want, (case.id, size)


def test_cases_name_a_verdict_a_charge_and_a_rule():
    for c in C.CASES:
        assert c.kind in ("attack", "honest") and c.computes and c.rule
        bound = c.charge if c.kind == "attack" else c.allow
        if c.kind == "honest":
            assert c.verdict == "pass" and c.allow is not None and c.charge is None and C.NEEDS_FAMILY in c.needs
        else:
            assert c.verdict in ("refuse", "budget") and c.allow is None
            assert c.charge is not None or c.verdict == "refuse"
        for cls, n in (bound or {}).items():
            assert (cls in C.CLASSES or cls == "any") and isinstance(n, int) and n > 0


def test_cases_json_is_the_metadata():
    assert CASES_JSON.read_text() == json.dumps([c.to_json() for c in C.CASES], indent=2) + "\n"


def test_the_ruling_numbers():
    """The reduction lengths the cases sit at: s8 x s8 over [-128, 127] is exact to K = 131,071, over [-127, 127] to 133,144;
    P in [0, 127] against V in [-128, 127] to 132,104, so 131,072 keys fit only with P's interval from its producer."""
    assert S32_MAX // (128 * 128) == 131_071 and S32_MAX // (127 * 127) == 133_144
    assert -S32_MIN // (127 * 128) == 132_104
    assert 127 * 127 * C.DITHER_K <= S32_MAX < 128 * 128 * C.DITHER_K
    assert 128 * 128 * C.SPLIT_HALF <= S32_MAX < 2 * 128 * 128 * C.SPLIT_HALF and C.SPLIT_HALF % 32 == 0
    assert 128 * 128 * 131_072 > S32_MAX >= 127 * 128 * 131_072
    clamped = [s for s in range(1, C.DITHER_K // 32 + 1) if 32 * 128 * 128 * s > S32_MAX]
    assert len(clamped) == C.BY_ID["imma-bound-bare"].charge["compare_select"] == C.BY_ID["fam-bound-dither"].charge["compare_select"]


def test_shift_lanes():
    for total, a_max, b_max in ((0, 127, 127), (1, 127, 127), (261_145, 127, 127), (30 * 127 * 127 + 5000, 127, 127),
                                (261_144, 255, 127), (32 * 127 * 127, 127, 127)):
        a, b = C.shift_lanes(total, a_max, b_max)
        assert len(a) == len(b) == 32 and sum(x * y for x, y in zip(a, b)) == total
        assert all(0 <= x <= a_max for x in a) and all(0 <= y <= b_max for y in b)
    with pytest.raises(ValueError, match="not a sum of 32 products"):
        C.shift_lanes(32 * 127 * 127 - 1)


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def test_pquant_keeps_p_in_its_interval():
    """PQUANT, the honest P's producer, maps every f32 encoding class into [0, 127]: the interval a check must derive."""
    edges = [0x7FC00000, 0xFFC00001, 0x7F800001, _f32(float("inf")), _f32(float("-inf")), _f32(-0.0), _f32(0.0), 1, 0x80000001,
             _f32(1e-45), _f32(0.999), _f32(1.0), _f32(126.6), _f32(127.0), _f32(127.9), _f32(128.0), _f32(3e38), _f32(-3e38)]
    for p in edges:
        (got,) = evaluate(C.PQUANT, p)
        assert got == C.pquant(p) and 0 <= got <= 127, hex(p)


@pytest.mark.parametrize("q, lo, hi, top", [(C.PQUANT_NO_FMAXF, None, 127.0, 127), (C.PQUANT_NO_FMINF, 0.0, None, 255),
                                            (C.PQUANT_128, 0.0, 128.0, 128)], ids=["no-fmaxf", "no-fminf", "clamp-128"])
def test_p_s_producer_variants_take_the_intervals_their_cases_claim(q, lo, hi, top):
    """The variants against `quant` on every edge class, and their ranges: RzByte saturates negatives to 0, so without
    fmaxf P stays in [0, 127]; without fminf it is the whole byte; clamped at 128 it reaches the byte the s8 step reads as
    -128."""
    edges = [0x7FC00000, 0xFFC00001, _f32(float("inf")), _f32(float("-inf")), _f32(-0.0), _f32(0.0), 1, 0x80000001,
             _f32(-3.5), _f32(126.6), _f32(127.9), _f32(128.0), _f32(128.5), _f32(200.0), _f32(255.9), _f32(256.0),
             _f32(300.0), _f32(2.0 ** 32), _f32(3e38), _f32(-3e38)]
    got = [evaluate(q, p)[0] for p in edges]
    assert got == [C.quant(p, lo, hi) for p in edges]
    assert min(got) == 0 and max(got) == top
