"""`boolean_softcap`: `AttnBlockSoftcap_v3` / `AttentionHeadSoftcap_v3` / `AttentionSoftcap_v3` are Boolean, agree with their
word views (`fa2_softcap`'s v2 chain at the same statics), Call their blocks and heads as the word chain does, and are pinned by
gate count and program digest.

No Boolean `MufuTanh_v1` exists, so ``TANH`` is bound here to `_TANH`: a Boolean Definition of this test, unregistered and with no
word, equal to `MufuTanh_v1` on its closed regions (`MUFU_TANH_RULES`: x for |x| <= 2^-8, +-1 from 8 with the infinities,
0x7FFFFFFF for NaN) and 0x7F7FFFFF on the table region between them.  `_inputs` puts every capped score in the closed regions:
queries in [0.1, 0.6], key rows either small (|s * softmax_scale / CAP| < 2^-8) or +-4096 (>= 8), with signed zeros,
subnormals, infinities and NaNs planted; `test_a_wrong_tanh_and_a_table_input_are_seen` holds the construction to it."""
from __future__ import annotations

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.batch import type_shape
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir.defs import bind
from verity.ir.types import Array
from verity.ml.boolean import attention as BA
from verity.ml.boolean import gemm as BG
from verity.ml.boolean import mufu as BM
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import NOT, const_bits, or_reduce, sub
from verity_vllm.program.registry import boolean_softcap as BSC
from verity_vllm.program.registry import fa2_softcap as FS
from verity_vllm.program.registry import prims as P

F32 = Array(32, BL.BIT)


def _set(bits, word: int, flag):
    return [x ^ flag if word >> j & 1 else x for j, x in enumerate(bits)]


def _closed_regions(saturate: bool):
    """`MufuTanh_v1` on its closed regions and 0x7F7FFFFF on the table region; without ``saturate``, x from 8 on too."""
    def tanh(C, x):
        a, s = list(x[:31]), x[31]
        _, ident = sub(C, a, const_bits(P.MUFU_TANH_RULES["identity_below"], 31), 31)
        _, below8 = sub(C, a, const_bits(P.MUFU_TANH_RULES["saturate_from"], 31), 31)
        _, not_nan = sub(C, a, const_bits(0x7F800001, 31), 31)
        table = C.AND(NOT(ident), below8)
        sat = C.AND(NOT(below8), not_nan)
        out = [C.AND(ident if saturate else or_reduce(C, [ident, sat]), b) for b in x]
        if saturate:
            out = _set(out, 0x3F800000, sat)
            out[31] = out[31] ^ C.AND(sat, s)
        out = _set(out, P.MUFU_TANH_RULES["nan_out"], NOT(not_nan))
        return _set(out, 0x7F7FFFFF, table)
    return tanh


_TANH = TR.definition("TanhClosedRegions", 1, _closed_regions(True), (("x", F32),), F32, register=False)
_NO_SATURATION = TR.definition("TanhClosedRegionsUnsaturated", 1, _closed_regions(False), (("x", F32),), F32, register=False)

HOPPER = {"CAP": 50.0, "DOT": BG.HopperBF16WgmmaDot16, "TANH": _TANH, "EX2": BM.MufuEx2Ftz}
AMPERE = {**HOPPER, "DOT": BG.AmpereBF16TcDot16}


def _block(**kw):
    return bind(BSC.AttnBlockSoftcap, D=16, **{**HOPPER, **kw})


def _head(**kw):
    return bind(BSC.AttentionHeadSoftcap, D=16, INV=BA.Fa2InvSum, **{**HOPPER, **kw})


def _attention(**kw):
    return bind(BSC.AttentionSoftcap, D=16, INV=BA.Fa2InvSum, **{**HOPPER, **kw})


#: id -> (And, Xor, Not gates, standalone program digest), with ``TANH`` = `_TANH`
PINS = {
    "AttnBlockSoftcap_v3{D=16,NVIS=5,FIRST=True,BN=16,CAP=50.0,DOT=HopperBF16WgmmaDot16_v2,CHECK=True,TANH=TanhClosedRegions_v1,"
    "EX2=MufuEx2Ftz_v2}": (212130, 411818, 19240, "eba0b1d8b0b19753df854ddb565069138f3af3d08d24e97b31bb83da8fbbd04b"),
    "AttnBlockSoftcap_v3{D=16,NVIS=16,FIRST=False,BN=16,CAP=50.0,DOT=HopperBF16WgmmaDot16_v2,CHECK=True,TANH=TanhClosedRegions_v1,"
    "EX2=MufuEx2Ftz_v2}": (464764, 903843, 38525, "ba4e729d9d0440837cc2c43852959de87a788625be93cae59abca28e82bd42e2"),
    "AttentionHeadSoftcap_v3{T=20,D=16,BN=16,CAP=50.0,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=1,"
    "TANH=TanhClosedRegions_v1,EX2=MufuEx2Ftz_v2}": (705243, 1368154, 58865,
                                                     "4a2fbf4a4919f5e5e6543ef84e5e6eadfb2603ab37e688b639e1b2171afd882a"),
    "AttentionSoftcap_v3{T=20,NH=4,KVH=2,D=16,BN=16,CAP=50.0,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=1,"
    "TANH=TanhClosedRegions_v1,EX2=MufuEx2Ftz_v2}": (2820972, 5472616, 235460,
                                                     "230f856335413af5c343e3f3397edcf813d5fe363f4415f775b6266ac148dc53"),
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
#: a query's: no word whose product with a key's special is moderate (0x0080 * 0x7F7F is about 4)
Q_SPECIALS = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1], np.uint16)
F32_SPECIALS = np.array([0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001], np.uint32)


def _bf16(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _inputs(word, n: int, seed: int) -> list[np.ndarray]:
    """``n`` instances per parameter of ``word``; from the second third on, special words planted at random positions."""
    rng = np.random.default_rng(seed)
    out = []
    for name, t in word.signature()[0]:
        shape = (n, *type_shape(t))
        if name in ("q",):
            a = _bf16(rng.uniform(0.1, 0.6, shape))
        elif name in ("k", "kc"):
            small = np.clip(rng.standard_normal(shape) * 0.04, -0.07, 0.07)
            big = np.where(rng.random(shape[:-1]) < 0.5, -4096.0, 4096.0)[..., None] * np.ones(shape[-1])
            a = _bf16(np.where((rng.random(shape[:-1]) < 0.4)[..., None], big, small))
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
        a[m] = rng.choice(Q_SPECIALS if name == "q" else SPECIALS, int(m.sum()))
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


@pytest.mark.parametrize("target", PINNED, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)) == digest


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_agrees_with_its_word_view(target) -> None:
    n = 48 if target.defn is BSC.AttnBlockSoftcap else 24
    assert not _mismatches(target, _inputs(TR.word_of(target), n, sum(map(ord, target.id))))


def test_a_wrong_tanh_and_a_table_input_are_seen() -> None:
    """The instances reach the saturating region (a ``TANH`` without it differs), and a capped score on the table region would
    differ from the word (`_TANH`'s 0x7F7FFFFF there is no tanh)."""
    blk = PINNED[1]
    inputs = _inputs(TR.word_of(blk), 48, 1)
    assert not _mismatches(blk, inputs)
    assert _mismatches(bind(blk.defn, **{**blk.bindings, "TANH": _NO_SATURATION}), inputs) > 24
    table = list(inputs)
    table[1] = _bf16(np.random.default_rng(2).standard_normal(inputs[1].shape) * 30)
    assert _mismatches(blk, table) > 24


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
