"""`verity.ml.circuits.ops.boolean.s8`: the Boolean s8 IMMA steps, `S8Dot_v2`, the limb steps and `LimbSplit_v2` agree with their
word Definitions (`word_of`) on sampled words and on the clamp, carry and wrap edges, and the steps carry exactly their
builders' live ANDs, pinned."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import s8 as S8
from verity.ml.circuits.ops.boolean import s8 as BS8

STEPS = [BS8.BlackwellS8ImmaDot32, BS8.BlackwellU8S8ImmaDot32]
LIMB_STEPS = [BS8.S32LowByte, BS8.S32ByteCarry, BS8.S32ClampS8, BS8.S32ClampRest]
BOOLEAN = {"DOT": BS8.BlackwellS8ImmaDot32, "TDOT": None}
DOTS = [bind(BS8.S8Dot, **S8.SMOKE_PLAIN | BOOLEAN),
        bind(BS8.S8Dot, **S8.SMOKE_PLAIN | {"T0": "input", "C": (-(1 << 30), 1 << 30)} | BOOLEAN),
        bind(BS8.S8Dot, **S8.SMOKE_EDITED | BOOLEAN | {"TDOT": BS8.BlackwellS8ImmaDot32})]
SPLITS = [bind(BS8.LimbSplit, FORM="bytes", N=3, Y=(-127 * 65793, 127 * 65793)),
          bind(BS8.LimbSplit, FORM="bytes", N=4, Y=(-(1 << 30), 1 << 30)),
          bind(BS8.LimbSplit, FORM="byteclamp", N=3, Y=(-65152, 65151))]
#: id -> And gates
PINS = {"BlackwellS8ImmaDot32_v2": 4157, "BlackwellU8S8ImmaDot32_v2": 4125, "S32LowByte_v2": 0, "S32ByteCarry_v2": 31,
        "S32ClampS8_v2": 59, "S32ClampRest_v2": 57}


def _agree(target, inputs) -> None:
    got = BS.evaluate_words(target, *inputs).astype(np.uint64)
    ref = E.evaluate_batch(TR.word_of(target), *inputs).astype(np.uint64)
    assert got.shape == ref.shape
    bad = np.flatnonzero(got.reshape(len(got), -1) != ref.reshape(len(ref), -1))
    assert not bad.size, f"{target.id}: {bad.size} instances differ from its word Definition, first {bad[:5]}"


@pytest.mark.parametrize("d", STEPS, ids=lambda d: d.name)
def test_the_steps_agree_with_their_word_steps(d) -> None:
    """Uniform words, and the corners: all-0x7F and all-0x80 operands against accumulators at the edges of s32, where the
    satfinite step clamps and the u8 x s8 step wraps."""
    rng = np.random.default_rng(23)
    n = 20_000
    acc = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    a = rng.integers(0, 256, (n, 32), dtype=np.uint64).astype(np.uint8)
    b = rng.integers(0, 256, (n, 32), dtype=np.uint64).astype(np.uint8)
    for i, (ca, cb, cacc) in enumerate([(0x7F, 0x7F, 0x7FFFFF00), (0x80, 0x80, 0x7FFFFF00), (0x80, 0x7F, 0x80000100),
                                        (0xFF, 0x7F, 0x7FFFFFFF), (0xFF, 0x80, 0x80000000), (0x00, 0x00, 0xFFFFFFFF)]):
        sl = slice(1000 * i, 1000 * i + 1000)
        a[sl], b[sl], acc[sl] = ca, cb, cacc
    _agree(bind(d), [acc, a, b])


@pytest.mark.parametrize("d", LIMB_STEPS, ids=lambda d: d.name)
def test_the_limb_steps_agree_with_their_word_steps(d) -> None:
    """Every word within 400 of 0, the s32 and 2^23 edges, the clamp's and carry's thresholds, and uniform words."""
    edges = list(range(-400, 400)) + [-(1 << 31), -(1 << 31) + 1, (1 << 31) - 2, (1 << 31) - 1, -(1 << 23) - 1, -(1 << 23),
                                      (1 << 23) - 1, 1 << 23, -32641, -32640, 32639, 32640, -65152, 65151]
    rng = np.random.default_rng(31)
    y = np.concatenate([np.array(edges, dtype=np.int64), rng.integers(-(1 << 31), 1 << 31, 20_000)]).astype(np.uint32)
    _agree(bind(d), [y])


@pytest.mark.parametrize("target", SPLITS, ids=lambda t: t.id)
def test_the_limb_split_on_bits_agrees_with_its_word_definition(target) -> None:
    lo, hi = target.bindings["Y"]
    rng = np.random.default_rng(37)
    y = np.concatenate([rng.integers(lo, hi + 1, 5_000), [lo, hi, -1, 0, 1]]).astype(np.int64).astype(np.uint32)
    _agree(target, [y])
    assert TR.word_of(target).id == target.id.replace("LimbSplit_v2", "LimbSplit_v1")


@pytest.mark.parametrize("d", STEPS + LIMB_STEPS, ids=lambda d: d.name)
def test_the_ands_are_the_builders_live_ands(d) -> None:
    T = d.traced()
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert T.counts["and"] == C.counts()["ands"] == PINS[d.id]


@pytest.mark.parametrize("target", DOTS, ids=lambda t: t.id[:60])
def test_the_family_on_bits_agrees_with_its_word_definition(target) -> None:
    rng = np.random.default_rng(29)
    params, _ = TR.word_of(target).signature()
    n = 2_000
    inputs = []
    for name, t in params:
        if name == "acc":
            inputs.append(rng.integers(-(1 << 30), (1 << 30) + 1, n).astype(np.int64).astype(np.uint32))
        else:
            lim = 64 if name == "tb" else 127
            inputs.append(rng.integers(-lim, lim + 1, (n, t.n)).astype(np.int64).astype(np.uint8))
    _agree(target, inputs)
    assert TR.word_of(target).id == target.id.replace("S8Dot_v2", "S8Dot_v1").replace("ImmaDot32_v2", "ImmaDot32_v1")


def test_a_word_step_is_refused_on_bits_and_a_boolean_one_on_words() -> None:
    from verity.ml.circuits.ops import prims as P

    with pytest.raises(ValueError, match="as a Boolean Definition"):
        bind(BS8.S8Dot, **S8.SMOKE_PLAIN).signature()
    with pytest.raises(ValueError, match="s8 family step"):
        bind(S8.S8Dot, **S8.SMOKE_PLAIN | BOOLEAN).signature()
    assert bind(S8.S8Dot, **S8.SMOKE_PLAIN).bindings["DOT"] is P.BlackwellS8ImmaDot32
