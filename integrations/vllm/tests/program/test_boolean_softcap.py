"""`boolean_softcap`: `AttnBlockSoftcap_v3` / `AttentionHeadSoftcap_v3` / `AttentionSoftcap_v3` with ``TANH`` = `MufuTanh_v2` and
``EX2`` = `MufuEx2Ftz_v2` (`verity.ml.circuits.ops.boolean.mufu`) are Boolean, agree with their word views (`fa2_softcap`'s v2 chain at the
same statics, over `MufuTanh_v1` and `MufuEx2Ftz_v1`), Call their blocks and heads as the word chain does, and are pinned by gate
count and program digest.

`_inputs` reaches every region of `MufuTanh_v1`: each key row has its own scale, so a capped score `s * softmax_scale / CAP` lands
below 2^-8 (the identity), on the measured table between 2^-8 and 8, or from 8 on (+-1), and from the second third on, special
words (signed zeros, subnormals, the normal range's ends, infinities, NaNs) are planted in every operand.
`test_the_instances_reach_every_tanh_region` holds the construction to it, and `test_a_wrong_tanh_on_the_table_is_seen` shows the
agreement would see a ``TANH`` that is `MufuTanh_v1` only on its closed regions."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.batch import type_shape
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops.boolean import attention as BA
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import NOT, const_bits, sub
from verity_vllm.program.registry import boolean_softcap as BSC
from verity_vllm.program.registry import fa2_softcap as FS
from verity_vllm.program.registry import prims as P

F32 = Array(32, BL.BIT)


def _set(bits, word: int, flag):
    return [x ^ flag if word >> j & 1 else x for j, x in enumerate(bits)]


def _closed_regions(C, x):
    """`MufuTanh_v1` on its closed regions and 0x7F7FFFFF on the table region."""
    a, s = list(x[:31]), x[31]
    _, ident = sub(C, a, const_bits(P.MUFU_TANH_RULES["identity_below"], 31), 31)
    _, below8 = sub(C, a, const_bits(P.MUFU_TANH_RULES["saturate_from"], 31), 31)
    _, not_nan = sub(C, a, const_bits(0x7F800001, 31), 31)
    table = C.AND(NOT(ident), below8)
    sat = C.AND(NOT(below8), not_nan)
    out = _set([C.AND(ident, b) for b in x], 0x3F800000, sat)
    out[31] = out[31] ^ C.AND(sat, s)
    out = _set(out, P.MUFU_TANH_RULES["nan_out"], NOT(not_nan))
    return _set(out, 0x7F7FFFFF, table)


_CLOSED_REGIONS = TR.definition("TanhClosedRegions", 1, _closed_regions, (("x", F32),), F32, register=False)

HOPPER = {"CAP": 50.0, "DOT": BG.HopperBF16WgmmaDot16, "TANH": BM.MufuTanh, "EX2": BM.MufuEx2Ftz}
AMPERE = {**HOPPER, "DOT": BG.AmpereBF16TcDot16}


def _block(**kw):
    return bind(BSC.AttnBlockSoftcap, D=16, **{**HOPPER, **kw})


def _head(**kw):
    return bind(BSC.AttentionHeadSoftcap, D=16, INV=BA.Fa2InvSum, **{**HOPPER, **kw})


def _attention(**kw):
    return bind(BSC.AttentionSoftcap, D=16, INV=BA.Fa2InvSum, **{**HOPPER, **kw})


#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "AttnBlockSoftcap_v3{D=16,NVIS=5,FIRST=True,BN=16,CAP=50.0,DOT=HopperBF16WgmmaDot16_v2,CHECK=True,TANH=MufuTanh_v2,"
    "EX2=MufuEx2Ftz_v2}": (216995, 438108, 19465, "6f2847e732be67644cfa4edce44537abc6979d3f5d59795a935acd4af1c0e7f204e08b154d2b33ef9eb9e4d99bf80d0735cbdde3268e49cc89b53724e199d60b"),
    "AttnBlockSoftcap_v3{D=16,NVIS=16,FIRST=False,BN=16,CAP=50.0,DOT=HopperBF16WgmmaDot16_v2,CHECK=True,TANH=MufuTanh_v2,"
    "EX2=MufuEx2Ftz_v2}": (480332, 987971, 39245, "84dbc72f3f9b264731c36109e8eef7c199197b0592da4127e8e93d5592d2a075477ced7e4b8e85afdb7bc0c41f25452e5d78bbe3d426bbb179b29e29432c234a"),
    "AttentionHeadSoftcap_v3{T=20,D=16,BN=16,CAP=50.0,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=1,"
    "TANH=MufuTanh_v2,EX2=MufuEx2Ftz_v2}": (724703, 1473314, 59765,
                                           "74781601637f97ffb5c13e429032ae65d804c743a2c83e24c9a0c9a75d45cfedd4dcb80c51c99e7dae48d946e86f27788d85402303beb5934618c5cb79c52e72"),
    "AttentionSoftcap_v3{T=20,NH=4,KVH=2,D=16,BN=16,CAP=50.0,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=1,"
    "TANH=MufuTanh_v2,EX2=MufuEx2Ftz_v2}": (2898812, 5893256, 239060,
                                           "c13d0649e3db71d63de1a218ea208ecc6f0531625ae14614c234614ee86d7aeb7c7b355c53ba15c5ab8cfb0fb502ebbd4d8437cab1f95b64fa0c0b8e4d3c0376"),
}

PINNED = [_block(NVIS=5, FIRST=True, BN=16, CHECK=True), _block(NVIS=16, FIRST=False, BN=16, CHECK=True),
          _head(T=20, BN=16, MASKED_FROM=1), _attention(T=20, NH=4, KVH=2, BN=16, MASKED_FROM=1)]

#: one block per mask shape, state and Check_inf placement, BN = 32, the Ampere step; one and two blocks, a partial last block,
#: either MASKED_FROM; GQA with two KV heads
AGREEMENT = PINNED + [_block(NVIS=16, FIRST=False, BN=16, CHECK=False), _block(NVIS=9, FIRST=False, BN=32, CHECK=False),
                      bind(BSC.AttnBlockSoftcap, D=16, NVIS=7, FIRST=True, BN=16, CHECK=True, **AMPERE),
                      _head(T=5, BN=16, MASKED_FROM=0), _head(T=20, BN=16, MASKED_FROM=0), _head(T=40, BN=32, MASKED_FROM=0)]

#: bf16 words that reach FA2's guards: signed zeros, subnormals, the normal range's ends, infinities, NaNs, 2^63
SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x5F00, 0xDF00],
                    np.uint16)
F32_SPECIALS = np.array([0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001], np.uint32)
#: a key row's scale: with queries ~ N(0, 1/4) and D = 16, softmax_scale / CAP = 1/200 takes its scores to about 4e-4 (identity),
#: 0.01, 0.3 and 3 (the table) and 40 (+-1)
KEY_SCALES = np.array([0.04, 1.0, 30.0, 300.0, 4096.0])
#: `MufuTanh_v1`'s regions on |x|: the identity below, the table between, +-1 from
IDENTITY_BELOW, SATURATE_FROM = (float(np.uint32(P.MUFU_TANH_RULES[k]).view(np.float32))
                                  for k in ("identity_below", "saturate_from"))


def _bf16(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _f32_of_bf16(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _inputs(word, n: int, seed: int) -> list[np.ndarray]:
    """``n`` instances per parameter of ``word``; from the second third on, special words planted at random positions."""
    rng = np.random.default_rng(seed)
    out = []
    for name, t in word.signature()[0]:
        shape = (n, *type_shape(t))
        if name == "q":
            a = _bf16(rng.standard_normal(shape) * 0.5)
        elif name in ("k", "kc"):
            a = _bf16(rng.standard_normal(shape) * rng.choice(KEY_SCALES, shape[:-1])[..., None])
        elif name in ("v", "vc"):
            a = _bf16(rng.standard_normal(shape))
        else:
            a = (rng.standard_normal(shape) * (0.5 if name == "m_prev" else 3.0)).astype(np.float32).view(np.uint32)
            if name == "l_prev":
                a &= np.uint32(0x7FFFFFFF)
            m = rng.random(shape) < 0.05
            m[: n // 3] = False
            a[m] = rng.choice(F32_SPECIALS if name != "l_prev" else F32_SPECIALS[:1], int(m.sum()))
            out.append(a)
            continue
        m = rng.random(shape) < 0.03
        m[: n // 3] = False
        a[m] = rng.choice(SPECIALS, int(m.sum()))
        out.append(a)
    return out


def _mismatches(target, inputs) -> int:
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64)
    ref = np.asarray(E.evaluate_batch(TR.word_of(target), *inputs)).astype(np.uint64)
    return int((got != ref).reshape(len(got), -1).any(axis=1).sum())


def test_the_word_views_and_word_statics() -> None:
    assert BSC.WORD_STATICS == {"TANH": P.MufuTanh, "EX2": P.MufuEx2Ftz}
    for d in (BSC.AttnBlockSoftcap, BSC.AttentionHeadSoftcap, BSC.AttentionSoftcap):
        assert d.word_statics is BSC.WORD_STATICS
    hopper = TR.word_of(BG.HopperBF16WgmmaDot16)
    blk = TR.word_of(PINNED[1])
    assert blk.id == bind(FS.AttnBlockSoftcapV2, D=16, NVIS=16, FIRST=False, BN=16, CAP=50.0, DOT=hopper, CHECK=True).id
    att = TR.word_of(PINNED[3])
    assert att.id == bind(FS.AttentionSoftcapV2, T=20, NH=4, KVH=2, D=16, BN=16, CAP=50.0, DOT=hopper, INV=P.Fa2InvSum,
                          MASKED_FROM=1).id


def _slow_past_a_block(targets: list) -> list:
    return [t if t.defn is BSC.AttnBlockSoftcap else pytest.param(t, marks=pytest.mark.slow) for t in targets]


def test_the_switch_binds_tanh_and_ex2_to_mufus_boolean_versions() -> None:
    """The switch (`verity_vllm.program.boolean`) takes the word chain to these: ``TANH`` and ``EX2`` bound through
    ``word_statics`` to `MufuTanh_v2` and `MufuEx2Ftz_v2`, and nothing left on words."""
    from verity_vllm.program import boolean as BB
    for target in PINNED:
        bits = BB.boolean_version(TR.word_of(target))
        assert bits.id == target.id and BL.is_boolean(bits)


@pytest.mark.parametrize("target", _slow_past_a_block(PINNED), ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)) == digest


@pytest.mark.parametrize("target", _slow_past_a_block(AGREEMENT), ids=lambda t: t.id)
def test_agrees_with_its_word_view(target) -> None:
    n = 48 if target.defn is BSC.AttnBlockSoftcap else 24
    assert not _mismatches(target, _inputs(TR.word_of(target), n, sum(map(ord, target.id))))


def test_the_instances_reach_every_tanh_region() -> None:
    """Most instances of a block have a capped score in each of `MufuTanh_v1`'s three regions on |x| (scores here in float64:
    the region bounds are powers of two, far from where the f32 rounding of a score could move it across one)."""
    blk = PINNED[1]
    q, k = _inputs(TR.word_of(blk), 48, 1)[:2]
    with np.errstate(all="ignore"):
        x = np.abs((_f32_of_bf16(q)[:, None, :] * _f32_of_bf16(k)).sum(-1) * (16 ** -0.5 / 50.0))
    regions = (x < IDENTITY_BELOW, (IDENTITY_BELOW <= x) & (x < SATURATE_FROM), x >= SATURATE_FROM)
    assert all(r.any(axis=1).sum() > 36 for r in regions)


@pytest.mark.slow
def test_a_wrong_tanh_on_the_table_is_seen() -> None:
    """The agreement sees a ``TANH`` that is `MufuTanh_v1` on its closed regions only (0x7F7FFFFF on the table), on the same
    instances `MufuTanh_v2` agrees on."""
    blk = PINNED[1]
    inputs = _inputs(TR.word_of(blk), 48, 1)
    assert not _mismatches(blk, inputs)
    assert _mismatches(bind(blk.defn, **{**blk.bindings, "TANH": _CLOSED_REGIONS}), inputs) > 24


def _composite_calls(fn) -> list:
    return [(nd.fn.name, nd.form, nd.n, nd.axes) for nd in fn.body.nodes if nd.fn.name in ("AttnBlockSoftcap", "AttentionHeadSoftcap")]


@pytest.mark.parametrize("target", [PINNED[2], AGREEMENT[-1], PINNED[3]], ids=lambda t: t.id)
def test_the_blocks_and_heads_are_the_words(target) -> None:
    """A head Calls its blocks, and the attention batches its heads, in the word chain's order and forms, each Call the Boolean
    version at the word Call's statics: what an instance-granular partition of the word Program cuts, the swap keeps."""
    word = TR.word_of(target)
    assert _composite_calls(target) == _composite_calls(word)
    calls = [nd.fn for nd in target.body.nodes if nd.fn.name in ("AttnBlockSoftcap", "AttentionHeadSoftcap")]
    words = [nd.fn for nd in word.body.nodes if nd.fn.name in ("AttnBlockSoftcap", "AttentionHeadSoftcap")]
    assert calls and [TR.word_of(c).id for c in calls] == [w.id for w in words]
