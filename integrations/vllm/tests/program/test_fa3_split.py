"""`registry.fa3_split` (FA3's split-KV dense prefill at vLLM's served `num_splits = 0`), `boolean_fa3_split` and the twins
`kernels.fa3_split_twins`: a row's splits are `get_n_block_min_max`'s cut at its keys; the first block is `AttnBlock_v4`'s first
iteration; the twin is the word Definition word for word; the fast path (the C++ row model per split) is the twin; a split head is
softmax attention within bf16 rounding; the target's binding maps to the split chain; the Boolean versions agree with their word
views."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import evaluate
from verity_vllm.program import boolean as SW
from verity_vllm.program.kernels import fa2_relation
from verity_vllm.program.kernels import fa3_model as FM
from verity_vllm.program.kernels import fa3_split_twins as FT
from verity_vllm.program.registry import attention_vocab, b1, hopper
from verity_vllm.program.registry import boolean_attention as BA
from verity_vllm.program.registry import boolean_fa3_split as BF
from verity_vllm.program.registry import fa3_check_inf as F3
from verity_vllm.program.registry import fa3_split as FS
from verity_vllm.program.registry import sparse_mla as M
from verity_vllm.program.registry import sparse_mla_split as W
from tests.program.test_boolean_fp8 import _flat, _inputs

FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
U32 = np.uint32


def _bf16(rng, shape, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(shape).astype(np.float32) * np.float32(scale)).view(U32) >> 16).astype(np.uint16)


def _val(words) -> np.ndarray:
    return (np.asarray(words, dtype=U32) << 16).view(np.float32).astype(np.float64)


def _kernel_splits(T: int, NBM: int, NS: int, BN: int) -> list[tuple[int, int]]:
    """`hopper/block.h::get_n_block_min_max` under `Split` for split s of NS over the tile's [0, NBM), then the blocks of those the
    row sees (columns below T): the non-empty ones in split order."""
    per = -(-NBM // NS) if NBM > 0 else 0
    out = []
    for s in range(NS):
        lo, hi = s * per, min(NBM, s * per + per)
        seen = [b for b in range(lo, hi) if b * BN < T]
        if seen:
            out.append((seen[0], seen[-1] + 1))
    return out


def test_split_ranges_are_the_kernels_cut_at_the_row() -> None:
    for BN in (16, 80):
        for NBM in range(1, 14):
            for NS in range(2, 9):
                for T in range(1, NBM * BN + 1, 7 if BN == 80 else 1):
                    assert FS.split_ranges(T, NBM, NS, BN) == _kernel_splits(T, NBM, NS, BN), (T, NBM, NS, BN)
                    if -(-T // BN) == NBM:
                        assert FS.split_ranges(T, NBM, NS, BN) == W.split_blocks(T, NS, BN)


def test_split_statics_refused() -> None:
    for T, NBM, NS, MF in ((0, 1, 2, 0), (40, 2, 2, 0), (40, 3, 1, 0), (40, 3, W.LANES + 1, 0), (40, 3, 2, 3), (40, 3, 2, -1)):
        with pytest.raises(ValueError):
            FS.check_split_statics("t", T, NBM, NS, 16, MF)
        with pytest.raises(ValueError):
            _ = bind(FS.AttentionHeadSplit, T=T, NBM=NBM, NS=NS, D=16, BN=16, MASKED_FROM=MF, **FA3).body
    with pytest.raises(ValueError, match="NS=1"):
        FT.split_call(np.zeros((1, 1, 256), np.uint16), np.zeros((80, 1, 256), np.uint16), np.zeros((80, 1, 256), np.uint16), [0],
                      80, 1, None)


@pytest.mark.parametrize("D", [16, 256])
def test_the_split_end_scale_is_the_blocks(D: int) -> None:
    assert M.scale_log2_bits(FS.scale(D)) == b1._scale_log2_bits(D)


def test_the_first_block_is_attn_block_v4s_first_iteration() -> None:
    rng = np.random.default_rng(1)
    for nvis in (1, 8, 16):
        q, k, v = _bf16(rng, 16), _bf16(rng, (16, 16)), _bf16(rng, (16, 16))
        k[nvis:], v[nvis:] = 0, 0
        first = bind(FS.AttnBlockFirst, D=16, NVIS=nvis, BN=16, DOT=FA3["DOT"])
        v4 = bind(F3.AttnBlockV4, D=16, NVIS=nvis, FIRST=True, BN=16, DOT=FA3["DOT"], CHECK=True)
        args = (q.tolist(), k.reshape(-1).tolist(), v.reshape(-1).tolist())
        got = [np.ravel(x).tolist() for x in evaluate(first, *args)]
        assert got == [np.ravel(x).tolist() for x in evaluate(v4, *args, 0, [0] * 4, [0] * 16)]


#: (T, NBM, NS, MASKED_FROM): two splits with an unguarded block; three of one block each; the tile's third split past the
#: row; a row in one of the tile's splits; a split of three blocks two of them masked iterations
CASES = [(40, 3, 2, 1), (33, 3, 2, 1), (20, 3, 3, 0), (40, 3, 3, 2), (17, 4, 2, 1), (64, 4, 2, 0), (90, 6, 2, 3)]


def _head_words(T, NBM, NS, MF, q, k, v) -> np.ndarray:
    head = bind(FS.AttentionHeadSplit, T=T, NBM=NBM, NS=NS, D=16, BN=16, MASKED_FROM=MF, **FA3)
    return np.asarray(evaluate(head, q.tolist(), k.reshape(-1).tolist(), v.reshape(-1).tolist()), dtype=U32)


@pytest.mark.parametrize("T,NBM,NS,MF", CASES)
def test_the_twin_is_the_definition(T: int, NBM: int, NS: int, MF: int) -> None:
    for seed in range(2):
        rng = np.random.default_rng(1000 * T + 10 * NS + seed)
        q, k, v = _bf16(rng, 16), _bf16(rng, (T, 16), 2.0), _bf16(rng, (T, 16))
        out, nf, opn = FT.head_split_rows(q[None], k, v, T=T, NBM=NBM, NS=NS, BN=16, MASKED_FROM=MF)
        assert not nf[0] and not opn[0]
        assert np.array_equal(out[0], _head_words(T, NBM, NS, MF, q, k, v))
        p = np.exp(_val(k) @ _val(q) / 4.0 - (_val(k) @ _val(q) / 4.0).max())
        assert np.allclose(_val(out[0]), p @ _val(v) / p.sum(), rtol=2 ** -6, atol=2 ** -7)


def test_attention_split_is_its_heads() -> None:
    T, NBM, NS, NH, KVH, MF = 40, 3, 2, 4, 2, 1
    rng = np.random.default_rng(7)
    q, kc, vc = _bf16(rng, (NH, 16)), _bf16(rng, (T, KVH, 16)), _bf16(rng, (T, KVH, 16))
    sp = bind(FS.AttentionSplit, T=T, NBM=NBM, NS=NS, NH=NH, KVH=KVH, D=16, BN=16, MASKED_FROM=MF, **FA3)
    got = np.asarray(evaluate(sp, q.reshape(-1).tolist(), kc.reshape(-1).tolist(), vc.reshape(-1).tolist()), dtype=U32)
    want = np.stack([FT.head_split_rows(q[h][None], kc[:, h // 2], vc[:, h // 2], T=T, NBM=NBM, NS=NS, BN=16, MASKED_FROM=MF)[0][0]
                     for h in range(NH)])
    assert np.array_equal(got.reshape(NH, 16), want)


def test_split_of_maps_the_targets_binding() -> None:
    spec = bind(F3.AttentionV4, T=40, NH=2, KVH=1, D=16, BN=16, MASKED_FROM=1, **FA3)
    assert FS.split_of(spec, 3, 2).id == bind(FS.AttentionSplit, T=40, NBM=3, NS=2, NH=2, KVH=1, D=16, BN=16, MASKED_FROM=1, **FA3).id
    assert FS.split_of(spec, 3, 1) is spec
    with pytest.raises(ValueError, match="not an Attention_v4 row"):
        FS.split_of(bind(FS.AttentionSplit, T=40, NBM=3, NS=2, NH=2, KVH=1, D=16, BN=16, MASKED_FROM=1, **FA3), 3, 2)


def test_the_served_rows_circuit_check_binds() -> None:
    """The sidecar's served rows: the last of a 321-token and of a 1000-token sequence on the 128 x 80 tile at H200's 132 SMs."""
    got = {}
    for L in (321, 1000):
        r = L - 1
        nbm = int(FM.kernel_geometry(L, L, [r], 80, kblock_m=128)[1][0])
        got[L] = (nbm, FM.split_plan([L], 8, 8, 256, num_sms=132, Dv=256)[1][0], attention_vocab.fa3_masked_from(L, 80, r, 128, 1))
    assert got == {321: (5, 5, 3), 1000: (13, 2, 11)}


def test_end_rows_flags_the_open_lg2() -> None:
    l = np.array([[0x3F000000, 0, 0, 0], [0x3F800000, 0, 0, 0], [0, 0, 0, 0]], U32)
    _, lse, opn = FT.end_rows(np.zeros(3, U32), l, np.zeros((3, 16), U32), D=16)
    assert opn.tolist() == [True, False, False] and lse[2] == W.NEG_INF


def _rows_case(L: int, NH: int, seed: int):
    rng = np.random.default_rng(seed)
    return _bf16(rng, (L, NH, 256)), _bf16(rng, (L, NH, 256), 2.0), _bf16(rng, (L, NH, 256))


def test_the_fast_path_is_the_twin() -> None:
    """The C++ row model per split, at head 256 on the 128 x 80 tile, against `head_split_rows` on rows at both sides of every
    block and tile boundary of a 300-token sequence in three splits (tiles whose second or third split is past the row)."""
    L, NS, NH = 300, 3, 2
    q, k, v = _rows_case(L, NH, 3)
    rows = np.array([0, 1, 79, 80, 81, 127, 128, 129, 159, 160, 200, 255, 256, 257, 299])
    out, nf, opn = FT.split_call(q[rows], k, v, rows, L, NS, fa2_relation.tables())
    assert not nf.any() and not opn.any()
    _, nbm = FM.kernel_geometry(L, L, rows, 80, kblock_m=128)
    for i, r in enumerate(rows):
        T = int(r) + 1
        mf = attention_vocab.fa3_masked_from(T, 80, int(r), 128, 1)
        for h in range(NH):
            want, _, _ = FT.head_split_rows(q[r, h][None], k[:T, h], v[:T, h], T=T, NBM=int(nbm[i]), NS=NS, BN=80, MASKED_FROM=mf)
            assert np.array_equal(out[i, h], want[0]), (int(r), h)


def test_the_fast_path_flags_a_non_finite_head() -> None:
    L, NS = 160, 2
    q, k, v = _rows_case(L, 2, 4)
    k[5, 1, 3] = 0x7F80
    rows = np.array([3, 100, 159])
    _, nf, _ = FT.split_call(q[rows], k, v, rows, L, NS, fa2_relation.tables())
    assert nf[1:, 1].all() and not nf[:, 0].any() and not nf[0, 1]


# --- the Boolean versions -------------------------------------------------------------------------------------------------------

BOOL = BA.ATTENTION_FA3
AGREEMENT = [bind(BF.AttnBlockFirstV2, D=16, NVIS=8, BN=16, DOT=BOOL["DOT"]),
             bind(BF.AttentionHeadSplitV2, T=33, NBM=3, NS=2, D=16, BN=16, MASKED_FROM=1, **BOOL)]


def test_the_boolean_versions_name_their_words() -> None:
    for target in AGREEMENT + [bind(BF.AttentionSplitV2, T=33, NBM=3, NS=2, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1, **BOOL)]:
        word = TR.word_of(target)
        lifted = {k: (TR.word_of(x) if k in ("DOT", "INV") else x) for k, x in target.bindings.items()}
        assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
        assert SW.boolean_version(word).id == target.id


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    """Seconds for the block, minutes for the head (three blocks, two ends and a combine): what the word-level tests can't see is
    the Boolean pieces' agreement on special words (zeros, infinities, NaNs, subnormals) through the blocks, the ends and the
    combine."""
    word = TR.word_of(target)
    inputs = _inputs(word, 96, sum(map(ord, target.id)))
    for g, r in zip(_flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"
