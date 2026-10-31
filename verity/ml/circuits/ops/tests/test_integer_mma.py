"""``verity.ml.circuits.silicon.integer``: IMMA m16n8k32 s8 x s8 -> s32, an exact integer step, against the PRO 6000 probes."""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon.instructions import EVIDENCE_IDS as INSTRUCTION_IDS
from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.silicon.integer import (
    BLACKWELL_SM100_S8_K32,
    BLACKWELL_SM120_S8_M16N8K32,
    BLACKWELL_SM120_S8_M16N8K32_WRAP,
    BLACKWELL_SM120_U8S8_M16N8K32,
    EVIDENCE_IDS,
    HOPPER_SM90_S8_K32,
    S32_MAX,
    S32_MIN,
    decode_int8,
    exact_k_limit,
    s32_value,
    s32_word,
)
from verity.ml.circuits.silicon.models import MODELS, tc_dot_chain
from verity.ml.circuits.silicon.term import term_value

SAT, WRAP, U8S8 = BLACKWELL_SM120_S8_M16N8K32, BLACKWELL_SM120_S8_M16N8K32_WRAP, BLACKWELL_SM120_U8S8_M16N8K32

# The library-kernel probe's edge case: every entry of A is 127 and every entry of B is
# sign * 127, zero incoming accumulator, the device word of one output: (k, sign) -> s32 value.
DEVICE_EDGE = {(133_120, 1): 2_147_092_480, (133_120, -1): -2_147_092_480, (133_152, 1): 2_147_483_647}

P, N = 0x7F, 0x81  # +127 and -127 as s8 codes; 0x81 is 129 as u8

# The IMMA probe's structured tiles (the .satfinite forms): one mma.sync whose products run
# past s32 and come back, B all 0x7F.  (a_dtype, A codes, C) -> the device's D.
DEVICE_CLAMP = {
    ("s8", (P,) + (N,) * 31, S32_MAX): 2_146_999_777,               # first+: a clamp after each product gives S32_MAX - 499,999
    ("s8", (N,) * 31 + (P,), S32_MAX): 2_146_999_777,               # last+: the same, with the k order reversed
    ("s8", (N,) + (P,) * 31, S32_MIN): -2_146_999_778,              # first-
    ("s8", (N,) * 16 + (P,) * 16, S32_MIN + 1000): S32_MIN + 1000,  # blocks16-: a clamp after 16 products gives S32_MIN + 258,064
    ("u8", (N,) + (P,) * 31, S32_MIN): -2_146_967_266,              # u8 x s8: 129 * 127 + 31 * 127 * 127 above S32_MIN
}


def code(v: int) -> int:
    return v & 0xFF


def test_step_is_the_exact_integer_dot() -> None:
    rng = random.Random(3)
    for model in (SAT, WRAP):
        for _ in range(500):
            a = [rng.randrange(-128, 128) for _ in range(32)]
            b = [rng.randrange(-128, 128) for _ in range(32)]
            c = rng.randrange(-(1 << 30), 1 << 30)
            d = model.step(s32_word(c), [code(x) for x in a], [code(x) for x in b])
            assert s32_value(d) == c + sum(x * y for x, y in zip(a, b))


def test_products_are_exact_terms() -> None:
    for a in (-128, -127, -1, 0, 1, 127):
        for b in (-128, -3, 0, 3, 127):
            n, e = term_value(SAT.product(code(a), code(b)))
            assert n * 2 ** e == a * b


def test_out_of_range_saturates_or_wraps() -> None:
    a, b = [code(127)] * 32, [code(127)] * 32
    top = s32_word(S32_MAX - 100)
    assert s32_value(SAT.step(top, a, b)) == S32_MAX
    assert s32_value(WRAP.step(top, a, b)) == S32_MAX - 100 + 32 * 127 * 127 - (1 << 32)
    bottom = s32_word(S32_MIN + 100)
    neg = [code(-127)] * 32
    assert s32_value(SAT.step(bottom, a, neg)) == S32_MIN
    assert s32_value(WRAP.step(bottom, a, neg)) == S32_MIN + 100 - 32 * 127 * 127 + (1 << 32)


def test_the_device_edge_words() -> None:
    """The saturating model reproduces the probe's three edge words as a chain of k/32 steps; the wrapping model
    reproduces the two in range and not the one past it, so the kernel the probe ran saturates."""
    for (k, sign), word in DEVICE_EDGE.items():
        a, b = [code(127)] * k, [code(sign * 127)] * k
        assert s32_value(tc_dot_chain(SAT, 0, a, b)) == word
        wrapped = s32_value(tc_dot_chain(WRAP, 0, a, b))
        assert (wrapped == word) == (abs(sign * k * 127 * 127) <= S32_MAX)


def test_the_device_clamps_once_at_d() -> None:
    """One instruction's sum leaves s32 and comes back: the device's D is the exact sum, so ``.satfinite`` clamps only at D."""
    for (a_dtype, a, c), word in DEVICE_CLAMP.items():
        model = replace(SAT, dtype=a_dtype)
        assert s32_value(model.step(s32_word(c), list(a), [P] * 32)) == word


@pytest.mark.parametrize("formats", [("s8", "s8"), ("s8", "u8"), ("u8", "s8"), ("u8", "u8")])
def test_the_exact_k_limit_is_the_least_k_whose_sum_can_leave_s32(formats) -> None:
    """Against every pair of codes: K - 1 of the extreme product stay in s32 and K of one of them leave it."""
    products = [decode_int8(formats[0], a) * decode_int8(formats[1], b) for a in range(256) for b in range(256)]
    lo, hi = min(products), max(products)
    k = exact_k_limit(*formats)
    assert S32_MIN <= (k - 1) * lo and (k - 1) * hi <= S32_MAX
    assert k * lo < S32_MIN or k * hi > S32_MAX


def test_the_s8_limit_is_2_17_and_tight() -> None:
    """2^17 - 32 codes -128 sum exactly and 2^17 clamp, in a chain of k/32 steps. At 2^17 only the last step can clamp, so
    the word is still the same in every k order; at 2^17 + 32 it is not: a clamp after 2^17 products of 2^14 and then 32
    of -128 * 127 gives one less than the reverse order, which never leaves s32."""
    k = exact_k_limit("s8", "s8")
    assert k == 1 << 17
    m = code(-128)
    assert s32_value(tc_dot_chain(SAT, 0, [m] * (k - 32), [m] * (k - 32))) == (k - 32) << 14
    assert s32_value(tc_dot_chain(SAT, 0, [m] * k, [m] * k)) == S32_MAX != k << 14
    a, b = [m] * (k + 32), [m] * k + [code(127)] * 32
    up, down = tc_dot_chain(SAT, 0, a, b), tc_dot_chain(SAT, 0, a[::-1], b[::-1])
    assert s32_value(down) == (k << 14) - 32 * 128 * 127 == s32_value(up) + 1


def test_registry_entry() -> None:
    ins = INSTRUCTIONS["sm120.mma.m16n8k32.s8"]
    assert ins.model is SAT and ins.status == "pinned" and ins.acc_dtype == "s32" and ins.shape == (16, 8, 32)
    ids = EVIDENCE_IDS[SAT.name]
    assert ins.evidence[0] == SAT.validated and "`run`" in SAT.validated and "`int_mm`" in SAT.validated
    assert ids["run"] != ids["int_mm"] and ins.dossier == ids["dossier"] == INSTRUCTION_IDS[ins.id]["dossier"]
    assert "HYPOTHESIS" not in SAT.validated and "HYPOTHESIS" not in WRAP.validated
    assert MODELS[SAT.name] is SAT and MODELS[WRAP.name] is WRAP


def test_u8s8_sibling_wraps_the_exact_dot() -> None:
    """`BlackwellU8S8ImmaDot32_v1` is the u8 x s8 form of the same instruction without ``.satfinite``: the exact sum modulo 2^32."""
    from verity.ml.circuits.ops import prims as P

    assert (U8S8.dtype, U8S8.b_dtype, U8S8.satfinite) == ("u8", "s8", False) and MODELS[U8S8.name] is U8S8
    assert EVIDENCE_IDS[U8S8.name]["run"] == EVIDENCE_IDS[SAT.name]["run"] and "`run`" in U8S8.validated
    assert "u8.s8" in U8S8.validated
    rng = random.Random(5)
    cases = [([255] * 32, [127] * 32, S32_MAX), ([255] * 32, [-128] * 32, S32_MIN), ([0] * 32, [0] * 32, -1)]
    cases += [([rng.randrange(256) for _ in range(32)], [rng.randrange(-128, 128) for _ in range(32)],
               rng.randrange(S32_MIN, S32_MAX + 1)) for _ in range(300)]
    for a, b, c in cases:
        exact = c + sum(x * y for x, y in zip(a, b))
        d = P.BlackwellU8S8ImmaDot32.evaluate(s32_word(c), *a, *[code(y) for y in b])
        assert d == U8S8.step(s32_word(c), a, [code(y) for y in b]) == exact & 0xFFFFFFFF


@pytest.mark.parametrize("iid, model", [("sm90.wgmma.m64n8k32.s8", HOPPER_SM90_S8_K32), ("sm90.mma.m16n8k32.s8", HOPPER_SM90_S8_K32),
                                        ("sm90.wgmma.m64n64k32.s8", HOPPER_SM90_S8_K32),
                                        ("sm90.wgmma.m64n128k32.s8", HOPPER_SM90_S8_K32),
                                        ("sm90.wgmma.m64n256k32.s8", HOPPER_SM90_S8_K32),
                                        ("sm100.tcgen05.mma.s8", BLACKWELL_SM100_S8_K32),
                                        ("sm100.mma.m16n8k32.s8", BLACKWELL_SM100_S8_K32)])
def test_the_datacenter_cards_compute_the_sm120_step(iid, model) -> None:
    """The H200's and B200's integer instructions are sm_120's step under their own arch's name: field for field, and on words
    `BlackwellS8ImmaDot32_v1` itself, so one Definition (and the Boolean step `step_s8_bits` builds) serves all three cards."""
    from dataclasses import fields

    from verity.ml.circuits.ops import prims as P

    ins = INSTRUCTIONS[iid]
    assert ins.model is model and ins.status == "pinned" and ins.dossier.startswith("art:") and MODELS[model.name] is model
    assert ins.sm == model.arch and ins.acc_dtype == "s32"
    for f in fields(model):
        if f.name not in ("name", "arch", "validated"):
            assert getattr(model, f.name) == getattr(SAT, f.name), f.name
    rng = random.Random(7)
    edges = [S32_MAX, S32_MIN, S32_MAX - 1000, S32_MIN + 1000, 0, -1]
    for _ in range(300):
        a = [rng.randrange(256) for _ in range(32)]
        b = [rng.randrange(256) for _ in range(32)]
        c = s32_word(rng.choice(edges) if rng.random() < 0.3 else rng.randrange(S32_MIN, S32_MAX + 1))
        assert model.step(c, a, b) == P.BlackwellS8ImmaDot32.evaluate(c, *a, *b) == SAT.step(c, a, b)


def test_refuses_words_outside_the_formats() -> None:
    with pytest.raises(InvalidArtifact):
        SAT.step(1 << 32, [0] * 32, [0] * 32)
    with pytest.raises(InvalidArtifact):
        SAT.step(0, [256] + [0] * 31, [0] * 32)
    with pytest.raises(InvalidArtifact):
        SAT.step(0, [0] * 31, [0] * 31)
