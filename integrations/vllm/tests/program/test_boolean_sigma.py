"""The Boolean-first fill (`verity_vllm.program.boolean`): the Boolean versions of the word Definitions the served rows reached
with none.

* each new Boolean leaf (`ops.boolean.scalar`'s I32/bit/f32 scalars, `ops.boolean.attention`'s GPU-NaN FTZ arithmetic,
  `registry.boolean_scalar`, `registry.boolean_marlin_moe`'s integer division) computes its word primitive on random words and
  on the f32 specials (zeros, subnormals, infinities, NaNs of both kinds), through bit-sliced evaluation against the word's;
* a word primitive made on demand (`Const<w>[0x..]_v1`, `GatherBf16x{V}_v1`) has a Boolean version made on demand;
* `sigma_lift`'s Definitions (`registry.boolean_sigma`) compute their word composites at small statics, are Boolean throughout,
  and are refused, naming every gap, when their word body Calls a Definition with no Boolean version.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch, evaluate_words
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import prims as MP
from verity.ml.circuits.ops import scalar as S
from verity.ml.circuits.types import Array, Value
from verity_vllm.program import boolean as BB
from verity_vllm.program.registry import b1, hopper, inductor as I, marlin_moe as MM, moe, prims as P, ref_prims as R

F32 = Value(32)
SPECIAL32 = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FFFFFFF, 0xFFC00001, 0x00000001, 0x807FFFFF, 0x3F800000,
             0x00800000, 0x7F7FFFFF, 0x7E800001, 0x00400000, 2, 0xFFFFFFFF, 0x7FFFFFFE]
SPECIAL16 = [0, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7FFF, 1, 0x3F80]


def _shape(t) -> tuple:
    if isinstance(t, Value):
        return ()
    if isinstance(t, Array):
        return (t.n, *_shape(t.elem))
    return (t.leaves,)


def _vectors(fn, n: int, seed: int) -> list[np.ndarray]:
    """``n`` argument rows of ``fn``'s signature, each word random or (for an f32 / bf16 word) one of the specials."""
    rng = np.random.default_rng(seed)
    out = []
    for _, t in fn.signature()[0]:
        w = t.homogeneous_width() or 16
        words = rng.integers(0, 1 << w, size=(n, t.leaves), dtype=np.uint64)
        specials = {32: SPECIAL32, 16: SPECIAL16}.get(w)
        if specials:
            pick = rng.random((n, t.leaves)) < 0.4
            words = np.where(pick, np.array(specials, np.uint64)[rng.integers(0, len(specials), size=(n, t.leaves))], words)
        out.append(words.reshape((n, *_shape(t))))
    return out


def _agrees(word, n: int = 64, seed: int = 0):
    """The Boolean version of ``word``, after checking it is Boolean and computes ``word`` on ``n`` rows."""
    got = BB.boolean_version(word)
    bits = bind(got) if isinstance(got, CompositeDefinition) else got
    assert BL.is_boolean(bits), bits.id
    args = _vectors(word, n, seed)
    want = np.asarray(evaluate_batch(word, *args), dtype=object).reshape(n, -1)
    have = np.asarray(evaluate_words(bits, *args), dtype=object).reshape(n, -1)
    bad = np.flatnonzero((want != have).any(axis=1))
    assert not len(bad), (bits.id, [a[bad[0]] for a in args], want[bad[0]], have[bad[0]])
    return bits


LEAVES = [S.I32Le, S.I32Eq, S.BitNot, S.BitOr, S.F32Neg, S.F32Sat, S.F32BitsShl23, S.F32IsFinite, W.F32AddFtzV2, W.F32SubFtzV2,
          W.F32MulFtzV2, W.F32FmaFtzV2, W.F32FmaSubFtzV2, P.DivFullScaleAV2, P.F32SubV2, R.F32Sub, R.F32GtStrict, moe.F32FmaRm,
          moe.F32FmaRmV2, MM.I32Div, MM.I32Rem]


@pytest.mark.parametrize("word", LEAVES, ids=lambda d: d.id)
def test_each_new_boolean_leaf_computes_its_word_primitive(word):
    bits = _agrees(word, n=256)
    assert bits.name == word.name and bits.version > word.version


def test_a_word_primitive_made_on_demand_has_a_boolean_version_made_on_demand():
    c = BB.boolean_version(MP.const(32, 0x3727C5AC))
    assert c.id == "Const32[0x3727c5ac]_v2" and BL.is_boolean(c)
    assert {n.fn.id for n in c.body.nodes} == {"Const1[0x0]_v1", "Const1[0x1]_v1"}
    assert BB.boolean_version(MP.const(32, 0x3727C5AC)) is c
    assert sum(b << i for i, b in enumerate(evaluate(c))) == 0x3727C5AC
    assert BB.boolean_version(MP.const(1, 1)) is MP.const(1, 1)
    _agrees(P.gather_bf16(16))


SIGMA = [bind(b1.ResidualAdd, N=4), bind(I.InductorRoPE, NHEADS=1, D=4), bind(I.InductorSiluMul, I=1),
         bind(moe.MoeRouterTopKNorm, E=4, TOPK=2, VPT=4), bind(moe.MoeExpertGemmWV2, E=2, K=16, N=2, DOT=hopper.HopperBF16WgmmaDot16),
         bind(MM.MoeAlignRanks, E=3, P=4, BM=2), bind(MM.MoeAlignBlocks, P=4, BM=2)]


@pytest.mark.parametrize("word", SIGMA, ids=lambda d: d.id)
def test_a_sigma_definition_computes_its_word_composite_on_bits(word):
    bits = _agrees(word)
    assert getattr(bits.defn, "sigma_of", None) is word.defn
    assert [BB._fid(n.fn) for n in bits.body.nodes] == [BB._fid(BB.boolean_version(n.fn)) for n in word.body.nodes]


def test_a_sigma_definition_whose_body_calls_a_definition_with_no_boolean_version_is_refused_by_name(monkeypatch):
    lonely = PrimitiveDefinition("LonelyWord", 1, (("a", F32), ("b", F32)), F32, lambda a, b: a, register=False)
    sig = lambda S: ((("x", Array(S.N, F32)),), Array(S.N, F32))  # noqa: E731
    word = CompositeDefinition("Twice", 1, ("N",), sig,
                               lambda B, S, x: B.batch(lonely, B.batch(W.F32AddFtz, x, x, axes=(0, 0)), x, axes=(0, 0)),
                               register=False)
    bits = BB.sigma_lift(word, 2, register=False)
    exact, by_name, errors = BB._load()
    monkeypatch.setattr(BB, "_TABLE", [(exact, {**by_name, "Twice": [bits]}, errors)])
    monkeypatch.setattr(BB, "_VERSIONS", {})
    with pytest.raises(BB.NoBooleanVersion, match=r"Twice_v2\{N=3\} is σ of it, and its body Calls LonelyWord_v1") as e:
        BB.boolean_version(bind(word, N=3))
    assert sorted(e.value.missing) == ["LonelyWord_v1", "Twice_v1{N=3}"]
