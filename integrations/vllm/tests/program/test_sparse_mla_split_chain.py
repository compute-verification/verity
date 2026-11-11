"""`sparse_mla_split.SparseMlaSplitEndCols_v1` and the cut serving stages a `SparseMlaSplit_v1` row as
(`sparse_mla_split.split_chain`): per head, each split's key blocks in FA3's order (`sparse_mla.SparseMlaHeadBlock_v1`), each
split's end in column blocks and the combine of each column block over the splits (at NS = 1 the head's end,
`sparse_mla.head_chain`'s cut) are the whole Call word for word, at small statics and at GLM-5.3's served ones; a column end is
those columns of the split's end; each unit fits the Flock block where the whole doesn't; the twin
(`sparse_mla_split_twins.split_chain_rows`) is each block's word; the Boolean column end agrees with its word view."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.numerics import sparse_mla_split_twins as TW
from verity.ml.definitions.registry import boolean_sparse_mla_split as BX
from verity.ml.definitions.registry import sparse_mla as M
from verity.ml.definitions.registry import sparse_mla_split as W
from tests.program.test_boolean_fp8 import _flat, _inputs
from tests.program.test_sparse_mla import FA3, _bf16, _case

GLM = M.SPARSE_MLA_GLM53
U32 = np.uint32


def _chain(T, NS, DR, DV, G, BN, scale, qn, qr, kv) -> tuple[np.ndarray, list[list[tuple]]]:
    """A head through its units, each evaluated on its own operands, its splits ended in column blocks of G: (its DV bf16
    words, per split each block's (m, l, o))."""
    blk = dict(DR=DR, DV=DV, BN=BN, DOT=FA3["DOT"], SCALE=scale)
    chains: list[list[tuple]] = []
    for b0, b1 in W.split_blocks(T, NS, BN):
        states: list[tuple] = []
        for b in range(b1 - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            unit = bind(M.SparseMlaHeadBlock, NVIS=hi - lo, FIRST=not states, **blk)
            out = evaluate(unit, qn.tolist(), qr.tolist(), kv[lo:hi].reshape(-1).tolist(), *(states[-1] if states else ()))
            states.append((out[0], list(out[1:5]), list(out[5:])))
        chains.append(states)
    if NS == 1:
        return np.asarray(evaluate(bind(M.SparseMlaHeadEnd, DV=DV, INV=FA3["INV"]), *chains[0][-1][1:]), dtype=U32), chains
    out = []
    for c in range(DV // G):
        ends = [evaluate(bind(W.SparseMlaSplitEndCols, DV=DV, G=G, C=c, INV=FA3["INV"], SCALE=scale), *st[-1]) for st in chains]
        comb = bind(W.SparseMlaCombine, NSPLIT=len(chains), DV=G, INV=FA3["INV"])
        out += evaluate(comb, [e[G] for e in ends], [x for e in ends for x in e[:G]])
    return np.asarray(out, dtype=U32), chains


def _head(T, NS, DR, DV, BN, scale, qn, qr, kv) -> np.ndarray:
    head = bind(W.SparseMlaHeadSplit, T=T, NS=NS, DR=DR, DV=DV, BN=BN, SCALE=scale, **FA3)
    return np.asarray(evaluate(head, qn.tolist(), qr.tolist(), kv.reshape(-1).tolist()), dtype=U32)


def test_a_column_end_is_those_columns_of_the_split_end() -> None:
    """`SparseMlaSplitEndCols_v1{DV, G, C}` is columns [C G, (C + 1) G) of `SparseMlaSplitEnd_v1{DV}`'s partial and its lse,
    word for word, on `_inputs`' words with specials; a column block that does not tile DV is refused."""
    end = bind(W.SparseMlaSplitEnd, DV=32, INV=FA3["INV"], SCALE=0.0625)
    m, l, o = _inputs(end, 400, 11)
    whole = np.hstack(_flat(E.evaluate_batch(end, m, l, o)))
    assert whole.shape == (400, 32 + 1)
    for G, C in ((8, 0), (8, 3), (16, 1), (32, 0)):
        got = np.hstack(_flat(E.evaluate_batch(bind(W.SparseMlaSplitEndCols, DV=32, G=G, C=C, INV=FA3["INV"], SCALE=0.0625), m, l, o)))
        assert np.array_equal(got, np.hstack([whole[:, C * G:(C + 1) * G], whole[:, 32:]]))
    for G, C in ((8, 4), (12, 0), (0, 0)):
        with pytest.raises(ValueError, match="not a block of DV=32"):
            _ = bind(W.SparseMlaSplitEndCols, DV=32, G=G, C=C, INV=FA3["INV"], SCALE=0.0625).body


@pytest.mark.parametrize("T,NS,G", [(1, 2, 16), (17, 2, 16), (37, 2, 8), (40, 3, 32), (65, 4, 16), (37, 1, 32)])
def test_a_split_head_is_its_chains(T: int, NS: int, G: int) -> None:
    """One split or several, of one block or more, some empty; any column blocks: the units' composition is the head."""
    DR, DV, BN = 16, 32, 16
    qn, qr, kv = _case(T, T, DR, DV, kv_scale=0.5)
    got, chains = _chain(T, NS, DR, DV, G, BN, 0.0625, qn, qr, kv)
    assert [len(c) for c in chains] == [b1 - b0 for b0, b1 in W.split_blocks(T, NS, BN)]
    assert np.array_equal(got, _head(T, NS, DR, DV, BN, 0.0625, qn, qr, kv))


def test_a_token_is_its_split_chains_at_the_served_statics() -> None:
    """GLM-5.3's statics at T = 130 in two splits: split 0 two full blocks, split 1 a block of 2 keys, each ended in eight
    column blocks of 64; the 8 heads' chains are the whole `SparseMlaSplit_v1` Call."""
    T, NS, NH, DR, DV, BN = 130, 2, GLM["NH"], GLM["DR"], GLM["DV"], GLM["BN"]
    rng = np.random.default_rng(130)
    qn, qr, kv = _bf16(rng, (NH, DV)), _bf16(rng, (NH, DR)), _bf16(rng, (T, DV + DR), 0.5)
    tok = bind(W.SparseMlaSplit, T=T, NS=NS, **GLM)
    want = np.asarray(evaluate(tok, qn.reshape(-1).tolist(), qr.reshape(-1).tolist(), kv.reshape(-1).tolist()), dtype=U32)
    G = W.combine_cols(DV)
    got = np.stack([_chain(T, NS, DR, DV, G, BN, GLM["SCALE"], qn[h], qr[h], kv)[0] for h in range(NH)])
    assert np.array_equal(got, want.reshape(NH, DV))


@pytest.mark.parametrize("T,NS", [(1, 2), (37, 2), (40, 3), (37, 1)])
def test_the_twin_is_each_block(T: int, NS: int) -> None:
    """`split_chain_rows`' state after each block of each split is that block's unit's, and its split ends' combine is the
    head's."""
    DR, DV, BN = 16, 32, 16
    qn, qr, kv = _case(T + 5, T, DR, DV, kv_scale=0.5)
    _got, chains = _chain(T, NS, DR, DV, 32, BN, 0.0625, qn, qr, kv)
    tw, nf = TW.split_chain_rows(qn[None], qr[None], kv, T=T, NS=NS, DR_=DR, DV=DV, BN=BN, scale=0.0625)
    assert not nf.any() and [len(c) for c in tw] == [len(c) for c in chains]
    for tc, wc in zip(tw, chains):
        for (m, l, o), (wm, wl, wo) in zip(tc, wc):
            assert int(m[0]) == wm and np.array_equal(l[0], np.asarray(wl, U32)) and np.array_equal(o[0], np.asarray(wo, U32))
    want = TW.head_split_rows(qn[None], qr[None], kv, T=T, NS=NS, DR_=DR, DV=DV, BN=BN, scale=0.0625)[0]
    if NS == 1:
        assert np.array_equal(TW.head_end_rows(tw[0][-1][1], tw[0][-1][2]), want)
    else:
        ends = [TW.split_end_rows(*c[-1], scale=0.0625) for c in tw]
        assert np.array_equal(TW.combine_rows(np.stack([e[1] for e in ends], 1), np.stack([e[0] for e in ends], 1)), want)


def test_split_chain_cuts_a_binding_at_its_statics() -> None:
    """The units of a row of T keys in NS splits: per head each split's first visited block, the rest full, each split's column
    ends and the combines; at NS = 1 `head_chain`'s."""
    blk = "DR=64,DV=512,NVIS={},FIRST={},BN=64,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625"
    got = [(u.id, n) for u, n in W.split_chain(bind(W.SparseMlaSplit, T=130, NS=2, **GLM))]
    assert got == [(f"SparseMlaHeadBlock_v1{{{blk.format(64, True)}}}", 8), (f"SparseMlaHeadBlock_v1{{{blk.format(2, True)}}}", 8),
                   (f"SparseMlaHeadBlock_v1{{{blk.format(64, False)}}}", 8),
                   *((f"SparseMlaSplitEndCols_v1{{DV=512,G=64,C={c},INV=Fa3InvSum_v1,SCALE=0.0625}}", 16) for c in range(8)),
                   ("SparseMlaCombine_v1{NSPLIT=2,DV=64,INV=Fa3InvSum_v1}", 64)]
    for T in (1, 64, 130, 2048):
        assert [(u.id, n) for u, n in W.split_chain(bind(W.SparseMlaSplit, T=T, NS=1, **GLM))] == \
            [(u.id, n) for u, n in M.head_chain(bind(M.SparseMla, T=T, **GLM))]
    one = {k: v for k, v in GLM.items() if k != "NH"}
    assert [n for _u, n in W.split_chain(bind(W.SparseMlaHeadSplit, T=272, NS=5, **one))] == [4, 1, *(5,) * 8, 8]
    with pytest.raises(ValueError, match="split_chain cuts a SparseMlaSplit_v1 or SparseMlaHeadSplit_v1 binding"):
        W.split_chain(bind(M.SparseMla, T=64, **GLM))


@pytest.mark.slow
def test_each_unit_fits_the_block_at_the_served_statics() -> None:
    """Every unit `split_chain` stages at GLM-5.3's statics past `head_chain`'s (whose tests hold the blocks and the end): each
    column end, and the combine of up to 32 splits (vllm-flash-attn's split cap), fits Flock's proved range where a split row
    of two blocks doesn't (`circuit_check.block_held` held it), each read off its Definition (seconds each); circuit-check's
    ``block`` checks a bound row's cut only."""
    from verity_flock import circuit as FC

    units = {u.id: u for NS in (2, 32) for u, _n in W.split_chain(bind(W.SparseMlaSplit, T=W.LANES * 64, NS=NS, **GLM))
             if u.defn in (W.SparseMlaSplitEndCols, W.SparseMlaCombine)}
    assert len(units) == 8 + 2
    for u in units.values():
        b = FC.unit_block(u)
        assert FC.past_block(b) is None, (u.id, b["k_log"])


AGREEMENT = [bind(BX.SparseMlaSplitEndColsV2, DV=16, G=8, C=1, INV=BX.SPARSE_MLA_GLM53["INV"], SCALE=0.0625)]


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    """What the word-level tests can't see is the Boolean column end's agreement on special words (zeros, infinities, NaNs,
    subnormals) through the split end's sum, reciprocal and log."""
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k == "INV" else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
    assert SW.boolean_version(word).id == target.id
    inputs = _inputs(word, 96, sum(map(ord, target.id)))
    for g, r in zip(_flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"
