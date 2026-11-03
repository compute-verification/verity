"""The Boolean-first fill (`verity.ml.definitions.boolean`): the Boolean versions of the word Definitions the served rows reached
with none.

* each new Boolean leaf (`ops.boolean.scalar`'s I32/bit/f32 scalars, `ops.boolean.attention`'s GPU-NaN FTZ arithmetic,
  `registry.boolean_scalar`, `registry.boolean_marlin_moe`'s integer division) computes its word primitive on random words and
  on the f32 specials (zeros, subnormals, infinities, NaNs of both kinds), through bit-sliced evaluation against the word's;
* a word primitive made on demand (`Const<w>[0x..]_v1`, `GatherBf16x{V}_v1`, `GatherI32x{V}_v1`, whole and as a tree of
  1024-word chunks) has a Boolean version made on demand;
* `sigma_lift`'s Definitions (`registry.boolean_sigma`) compute their word composites at small statics, are Boolean throughout,
  and are refused, naming every gap, when their word body Calls a Definition with no Boolean version;
* GLM-5.3 TP8's embedding shard selects the rows of its own tokens and zeros the rest on bits, its fused all-reduce + RMSNorm
  computes its word composite on every branch the served descriptor takes, and the constants, residual add and fused norm its
  B1 request Calls with no Boolean version of their own resolve to Boolean ones at their served descriptors.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch, evaluate_words
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import prims as MP
from verity.ml.circuits.ops import scalar as S
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions import boolean as BB
from verity.ml.definitions.registry import b1, b1_tp2, dsv4_decode_indices as DI, hopper, inductor as I, marlin_moe as MM, moe
from verity.ml.definitions.registry import prims as P, ref_prims as R, tp_allreduce as TA

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


def _boolean_ids(word) -> list[str]:
    """The descriptor ids of the Boolean versions of ``word``'s Calls, in body order."""
    got = [BB.boolean_version(n.fn) for n in word.body.nodes]
    return [descriptor_id(bind(g) if isinstance(g, CompositeDefinition) else g) for g in got]


LEAVES = [S.I32Le, S.I32Eq, S.BitNot, S.BitOr, S.F32Neg, S.F32Sat, S.F32BitsShl23, S.F32IsFinite, W.F32AddFtzV2, W.F32SubFtzV2,
          W.F32MulFtzV2, W.F32FmaFtzV2, W.F32FmaSubFtzV2, P.DivFullScaleAV2, P.F32SubV2, R.F32Sub, R.F32GtStrict, moe.F32FmaRm,
          moe.F32FmaRmV2, MM.I32Div, MM.I32Rem]


@pytest.mark.parametrize("word", LEAVES, ids=lambda d: d.id)
def test_each_new_boolean_leaf_computes_its_word_primitive(word):
    bits = _agrees(word, n=256)
    assert bits.name == word.name and bits.version > word.version


@pytest.mark.slow
def test_a_word_primitive_made_on_demand_has_a_boolean_version_made_on_demand():
    c = BB.boolean_version(MP.const(32, 0x3727C5AC))
    assert c.id == "Const32[0x3727c5ac]_v2" and BL.is_boolean(c)
    assert {n.fn.id for n in c.body.nodes} == {"Const1[0x0]_v1", "Const1[0x1]_v1"}
    assert BB.boolean_version(MP.const(32, 0x3727C5AC)) is c
    assert sum(b << i for i, b in enumerate(evaluate(c))) == 0x3727C5AC
    assert BB.boolean_version(MP.const(1, 1)) is MP.const(1, 1)
    _agrees(P.gather_bf16(16))
    _agrees(DI.gather_i32(16))
    word = DI.gather_i32(1100)
    tree = BB.boolean_version(word)
    assert tree.id == "GatherI32x1100_v2" and BL.is_boolean(tree) and {n.fn.id for n in tree.body.nodes} == {
        "GatherI32x1024u10_v1", "GatherI32x76u10_v1", "GatherI32x2u22_v1"}
    rng = np.random.default_rng(3)
    tokens = np.array([0, 1, 1023, 1024, 1025, 1099, 1100, 2047, 2048, 0xFFFFFFFF, 0x80000000, *rng.integers(0, 1100, 21)], np.uint64)
    column = rng.integers(0, 1 << 32, size=(len(tokens), 1100), dtype=np.uint64)
    want = [c[t] if t < 1100 else 0xFFFFFFFF for t, c in zip(tokens.tolist(), column)]
    assert evaluate_batch(word, tokens, column).tolist() == want
    assert evaluate_words(tree, tokens, column).tolist() == want


SIGMA = [bind(b1.ResidualAdd, N=4), bind(b1.Gemm, K=32, N=2), bind(I.InductorRoPE, NHEADS=1, D=4), bind(I.InductorSiluMul, I=1),
         bind(moe.MoeRouterTopKNorm, E=4, TOPK=2, VPT=4), bind(moe.MoeExpertGemmWV2, E=2, K=16, N=2, DOT=hopper.HopperBF16WgmmaDot16),
         bind(MM.MoeAlignRanks, E=3, P=4, BM=2), bind(MM.MoeAlignBlocks, P=4, BM=2)]


@pytest.mark.parametrize("word", SIGMA, ids=lambda d: d.id)
def test_a_sigma_definition_computes_its_word_composite_on_bits(word):
    bits = _agrees(word)
    assert getattr(bits.defn, "sigma_of", None) is word.defn
    assert [descriptor_id(n.fn) for n in bits.body.nodes] == _boolean_ids(word)


@pytest.mark.parametrize("start", [0, 16])
def test_an_embedding_shard_on_bits_selects_its_own_tokens_rows_and_zeros_the_rest(start):
    """`EmbeddingShard_v2` over a 16-row shard starting at 0 (rank 0's) and at 16, on the tokens at and past either end of the
    shard, at the sign boundary, and inside it, where a random 32-bit token almost never is."""
    VS, H = 16, 2
    word = bind(b1_tp2.EmbeddingShard, VS=VS, H=H, START=start)
    bits = bind(BB.boolean_version(word))
    assert BL.is_boolean(bits) and bits.defn.sigma_of is word.defn
    rng = np.random.default_rng(start)
    tokens = [(start - 1) & 0xFFFFFFFF, start, start + 1, start + VS - 1, start + VS, 0, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
              *rng.integers(start, start + VS, 7).tolist()]
    table = rng.integers(0, 1 << 16, size=(len(tokens), VS, H), dtype=np.uint64)
    want = [table[i, t - start].tolist() if start <= t < start + VS else [0] * H for i, t in enumerate(tokens)]
    tok = np.array(tokens, np.uint64)
    assert np.asarray(evaluate_batch(word, tok, table)).tolist() == want
    assert np.asarray(evaluate_words(bits, tok, table)).tolist() == want


#: FlashInfer's fused all-reduce + RMSNorm on the branches GLM-5.3 TP8's served descriptor takes (eight ranks, an eight-CTA
#: cluster, the scale's FFMA, one-shot; at N = 2048, a warp a CTA), and two-shot over one CTA of two warps
MNNVL = [bind(TA.AllReduceRmsNormMnnvl, WORLD=2, N=512, SHOT=2, CLUSTER=1, EPS=1e-5, FMA=True),
         bind(TA.AllReduceRmsNormMnnvl, WORLD=8, N=2048, SHOT=1, CLUSTER=8, EPS=1e-5, FMA=True)]


@pytest.mark.slow
@pytest.mark.parametrize("word", MNNVL, ids=lambda d: d.id)
def test_the_fused_all_reduce_rms_norm_on_bits_computes_its_word_composite(word):
    """`AllReduceRmsNormMnnvl_v2` (17M gates two-shot, about 6 s; 79M at GLM-5.3's branches, about 17 s), on branches
    circuit-check's binding (one-shot over one warp, the scale an FMUL and an FADD) does not take."""
    bits = _agrees(word, n=16)
    assert bits.defn.sigma_of is word.defn


#: the word Definitions GLM-5.3 TP8 B1's request Calls as root Calls with no Boolean version of their own, at their descriptors
#: there, but the embedding shard (19360 x 6144, whose body views its table element by element), which is above at 16 x 2
GLM53_TP8 = [MP.const(32, 1), MP.const(32, 0x11F), bind(b1.ResidualAdd, N=6144),
             bind(TA.AllReduceRmsNormMnnvl, WORLD=8, N=6144, SHOT=1, CLUSTER=8, EPS=1e-5, FMA=True)]


@pytest.mark.parametrize("word", GLM53_TP8, ids=lambda d: d.id)
def test_glm53_tp8s_word_calls_resolve_to_boolean_versions_at_their_served_descriptors(word):
    """Each resolves to a Boolean Definition of its name: a constant to its bits (made on demand), a composite to its σ
    Definition at the same statics, whose body is its word body with each Call's Boolean version."""
    bits = BB.boolean_version(word)
    bits = bind(bits) if isinstance(bits, CompositeDefinition) else bits
    assert BL.is_boolean(bits) and bits.name == word.name and bits.version > word.version
    if isinstance(word, PrimitiveDefinition):
        assert (sum(b << i for i, b in enumerate(evaluate(bits))),) == evaluate(word)
        return
    assert bits.defn.sigma_of is word.defn and bits.bindings == word.bindings
    assert [descriptor_id(n.fn) for n in bits.body.nodes] == _boolean_ids(word)


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
