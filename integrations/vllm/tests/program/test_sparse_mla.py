"""`registry.sparse_mla` (FA3's sparse MLA over the indexer's selected rows) and `boolean_sparse_mla`: where it reduces, the head is
FA3's dense decode head (`fa3_check_inf.AttentionHead_v4` with K = [k_pe | kv_c], V = [kv_c | 0] and the scale D^-0.5), at the
served statics as well; at GLM-5.3's scale it is softmax attention within bf16 rounding; a token is its heads; the Boolean
versions agree with their word views and the switch maps each word Definition to its Boolean one."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.defs import bind
from verity.ml.circuits.boolean import trace as TR
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.registry import b1, hopper
from verity.ml.definitions.registry import boolean_sparse_mla as BM
from verity.ml.definitions.registry import fa3_check_inf as F3
from verity.ml.definitions.registry import sparse_mla as M
from tests.program.test_boolean_fp8 import _flat, _inputs

FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
GLM = M.SPARSE_MLA_GLM53


def _bf16(rng, shape, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(shape).astype(np.float32) * np.float32(scale)).view(np.uint32) >> 16).astype(np.uint16)


def _val(words) -> np.ndarray:
    return (np.asarray(words, dtype=np.uint32) << 16).view(np.float32).astype(np.float64)


def _case(seed: int, T: int, DR: int, DV: int, kv_scale: float = 1.0):
    rng = np.random.default_rng(seed)
    return _bf16(rng, DV), _bf16(rng, DR), _bf16(rng, (T, DV + DR), kv_scale)


def _run_head(T, DR, DV, BN, scale, qn, qr, kv) -> np.ndarray:
    head = bind(M.SparseMlaHead, T=T, DR=DR, DV=DV, BN=BN, SCALE=scale, **FA3)
    return np.asarray(evaluate(head, qn.tolist(), qr.tolist(), kv.reshape(-1).tolist()), dtype=np.uint32)


def _dense_fa3(T, DR, DV, BN, qn, qr, kv) -> np.ndarray:
    """`AttentionHead_v4` (FA3 decode, Check_inf on the first visited block only) over q = [q_rope | q_nope], k = [k_pe | kv_c],
    v = [kv_c | 0]: its first DV outputs."""
    ref = bind(F3.AttentionHeadV4, T=T, D=DR + DV, BN=BN, MASKED_FROM=(T - 1) // BN, **FA3)
    q = np.concatenate([qr, qn])
    k = np.concatenate([kv[:, DV:], kv[:, :DV]], axis=1)
    v = np.concatenate([kv[:, :DV], np.zeros((T, DR), np.uint16)], axis=1)
    out = evaluate(ref, q.tolist(), k.reshape(-1).tolist(), v.reshape(-1).tolist())
    return np.asarray(out[:DV], dtype=np.uint32)


@pytest.mark.parametrize("T", [1, 5, 16, 17, 37, 64])
def test_the_head_is_fa3s_decode_head_where_it_reduces(T: int) -> None:
    """At the scale (DR + DV)^-0.5 the sparse head is `AttentionHead_v4` over the rotated rows, word for word: one block or
    several, the first visited one partial and zero-padded or full."""
    DR, DV, BN = 16, 16, 16
    assert M.scale_log2_bits((DR + DV) ** -0.5) == b1.scale_log2_bits(DR + DV)
    qn, qr, kv = _case(T, T, DR, DV)
    assert np.array_equal(_run_head(T, DR, DV, BN, (DR + DV) ** -0.5, qn, qr, kv), _dense_fa3(T, DR, DV, BN, qn, qr, kv))


def test_the_head_at_the_served_statics_is_fa3s_decode_head_where_it_reduces() -> None:
    """DR = 64, DV = 512, BN = 64 and T = 130 (three blocks, the first visited one of 2 keys) at the scale 576^-0.5."""
    T, DR, DV, BN = 130, GLM["DR"], GLM["DV"], GLM["BN"]
    qn, qr, kv = _case(7, T, DR, DV, kv_scale=0.5)
    assert np.array_equal(_run_head(T, DR, DV, BN, (DR + DV) ** -0.5, qn, qr, kv), _dense_fa3(T, DR, DV, BN, qn, qr, kv))


def test_glm53_scale() -> None:
    """softmax_scale (192 + 64)^-0.5 = 1/16, and scale_log2 its f32 product with M_LOG2E taken in double."""
    assert GLM["SCALE"] == (192 + 64) ** -0.5
    assert M.scale_log2_bits(GLM["SCALE"]) == int(np.float32(0.0625 * 1.4426950408889634).view(np.uint32))


@pytest.mark.parametrize("T", [3, 130])
def test_the_head_is_softmax_attention_within_bf16_rounding(T: int) -> None:
    """At GLM-5.3's statics the head is softmax(q . k / 16) kv_c within the kernel's roundings (P to bf16, the output to bf16)."""
    DR, DV, BN, scale = GLM["DR"], GLM["DV"], GLM["BN"], GLM["SCALE"]
    qn, qr, kv = _case(11 + T, T, DR, DV, kv_scale=0.25)
    got = _val(_run_head(T, DR, DV, BN, scale, qn, qr, kv))
    s = (_val(kv[:, :DV]) @ _val(qn) + _val(kv[:, DV:]) @ _val(qr)) * scale
    p = np.exp(s - s.max())
    ref = (p / p.sum()) @ _val(kv[:, :DV])
    assert np.max(np.abs(got - ref)) <= 2 ** -7 * np.max(np.abs(_val(kv[:, :DV])))


def test_a_token_is_its_heads() -> None:
    T, NH, DR, DV, BN = 20, 3, 16, 16, 16
    rng = np.random.default_rng(3)
    qn, qr, kv = _bf16(rng, (NH, DV)), _bf16(rng, (NH, DR)), _bf16(rng, (T, DV + DR))
    tok = bind(M.SparseMla, T=T, NH=NH, DR=DR, DV=DV, BN=BN, SCALE=0.0625, **FA3)
    got = np.asarray(evaluate(tok, qn.reshape(-1).tolist(), qr.reshape(-1).tolist(), kv.reshape(-1).tolist()), dtype=np.uint32)
    heads = [_run_head(T, DR, DV, BN, 0.0625, qn[h], qr[h], kv) for h in range(NH)]
    assert np.array_equal(got.reshape(NH, DV), np.stack(heads))


@pytest.mark.parametrize("first", [True, False])
def test_only_a_later_block_takes_the_running_state(first: bool) -> None:
    blk = bind(M.SparseMlaBlock, DR=16, DV=16, NVIS=5, FIRST=first, BN=16, DOT=FA3["DOT"], SCALE=0.0625)
    names = [n for n, _ in blk.signature()[0]]
    assert names == ["q", "kv"] + ([] if first else ["m_prev", "l_prev", "o_prev"])


def test_statics_are_checked() -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        bind(M.SparseMlaBlock, DR=8, DV=16, NVIS=4, FIRST=True, BN=16, DOT=FA3["DOT"], SCALE=0.0625).body
    with pytest.raises(ValueError, match="selected keys"):
        bind(M.SparseMlaHead, T=0, DR=16, DV=16, BN=16, SCALE=0.0625, **FA3).body


BOOL = {"DOT": BM.SPARSE_MLA_GLM53["DOT"], "INV": BM.SPARSE_MLA_GLM53["INV"]}
AGREEMENT = [bind(BM.SparseMlaBlockV2, DR=16, DV=16, NVIS=5, FIRST=True, BN=16, DOT=BOOL["DOT"], SCALE=0.0625),
             bind(BM.SparseMlaHeadV2, T=5, DR=16, DV=16, BN=16, SCALE=0.0625, **BOOL)]


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    """Each traces a few hundred thousand ANDs (minutes): what the word-level tests can't see is the Boolean pieces' agreement on
    special words (zeros, infinities, NaNs, subnormals) through the online softmax."""
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k in ("DOT", "INV") else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
    assert SW.boolean_version(word).id == target.id
    inputs = _inputs(word, 96, sum(map(ord, target.id)))
    for g, r in zip(_flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"
