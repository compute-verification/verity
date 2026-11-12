"""`sparse_mla.SparseMlaHeadBlock_v1` and `SparseMlaHeadEnd_v1`, the cut serving stages a `SparseMla_v1` row as
(`sparse_mla.head_chain`): per head its key blocks in FA3's order, each a unit carrying the online softmax's state to the next,
and its end are the whole Call word for word, at small statics and at GLM-5.3's served ones; each unit fits the Flock block where
the whole doesn't; the end is `dead_free.AttnHeadEnd_v1`'s function; the twin (`sparse_mla_split_twins.chain_rows`,
`head_end_rows`) is each unit's word; the Boolean versions agree with their word views."""
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
from verity.ml.definitions.registry import boolean_sparse_mla as BM
from verity.ml.definitions.registry import dead_free as DF
from verity.ml.definitions.registry import sparse_mla as M
from tests.program.test_boolean_fp8 import _flat, _inputs
from tests.program.test_sparse_mla import FA3, _bf16, _case, _run_head

GLM = M.SPARSE_MLA_GLM53
U32 = np.uint32


def _chain(T, DR, DV, BN, scale, qn, qr, kv) -> tuple[np.ndarray, list[tuple]]:
    """A head through its units, each evaluated on its own operands: (the end's DV bf16 words, each block's (m, l, o))."""
    blk = dict(DR=DR, DV=DV, BN=BN, DOT=FA3["DOT"], SCALE=scale)
    states: list[tuple] = []
    for j, (lo, hi) in enumerate(M.visits(T, BN)):
        unit = bind(M.SparseMlaHeadBlock, NVIS=hi - lo, FIRST=j == 0, **blk)
        state = [states[-1][0], states[-1][1], states[-1][2]] if j else []
        out = evaluate(unit, qn.tolist(), qr.tolist(), kv[lo:hi].reshape(-1).tolist(), *state)
        states.append((out[0], list(out[1:5]), list(out[5:])))
    end = evaluate(bind(M.SparseMlaHeadEnd, DV=DV, INV=FA3["INV"]), states[-1][1], states[-1][2])
    return np.asarray(end, dtype=U32), states


def test_the_end_is_the_dense_heads_end() -> None:
    """`SparseMlaHeadEnd_v1{DV, INV}` is `AttnHeadEnd_v1{D = DV, INV}` word for word, on `_inputs`' words with specials."""
    end = bind(M.SparseMlaHeadEnd, DV=16, INV=FA3["INV"])
    l, o = _inputs(end, 400, 7)
    want = _flat(E.evaluate_batch(bind(DF.AttnHeadEndV1, D=16, INV=FA3["INV"]), l, o))
    assert all(np.array_equal(g, w) for g, w in zip(_flat(E.evaluate_batch(end, l, o)), want, strict=True))


def test_a_heads_blocks_are_visited_last_to_first() -> None:
    assert M.visits(1, 16) == [(0, 1)]
    assert M.visits(16, 16) == [(0, 16)]
    assert M.visits(37, 16) == [(32, 37), (16, 32), (0, 16)]
    with pytest.raises(ValueError, match="T=0 selected keys"):
        M.visits(0, 16)


@pytest.mark.parametrize("T", [1, 5, 16, 17, 37, 64, 65])
def test_a_head_is_its_chain(T: int) -> None:
    """One block or several, the first visited one partial and zero-padded or full: the units' composition is the head."""
    DR, DV, BN = 16, 16, 16
    qn, qr, kv = _case(T, T, DR, DV, kv_scale=0.5)
    got, states = _chain(T, DR, DV, BN, 0.0625, qn, qr, kv)
    assert len(states) == -(-T // BN)
    assert np.array_equal(got, _run_head(T, DR, DV, BN, 0.0625, qn, qr, kv))


def test_a_token_is_its_chains_at_the_served_statics() -> None:
    """GLM-5.3's statics (NH = 8, DR = 64, DV = 512, BN = 64, the scale 1/16) at T = 130: three blocks a head, the first visited
    one of 2 keys; the 8 heads' chains are the whole `SparseMla_v1` Call."""
    T, NH, DR, DV, BN = 130, GLM["NH"], GLM["DR"], GLM["DV"], GLM["BN"]
    rng = np.random.default_rng(130)
    qn, qr, kv = _bf16(rng, (NH, DV)), _bf16(rng, (NH, DR)), _bf16(rng, (T, DV + DR), 0.5)
    tok = bind(M.SparseMla, T=T, **GLM)
    want = np.asarray(evaluate(tok, qn.reshape(-1).tolist(), qr.reshape(-1).tolist(), kv.reshape(-1).tolist()), dtype=U32)
    got = np.stack([_chain(T, DR, DV, BN, GLM["SCALE"], qn[h], qr[h], kv)[0] for h in range(NH)])
    assert np.array_equal(got, want.reshape(NH, DV))


@pytest.mark.slow
def test_a_token_is_its_chains_at_index_topk() -> None:
    """T = 2048, `index_topk`, the most keys a served row selects: 32 full blocks a head.  The whole Call is evaluated once
    (about a minute); the chains go through the twin, which the tests above hold to the units word for word.  What it catches
    that T = 130 can't: a state carried through 31 blocks rather than 2."""
    T, NH, DR, DV = 2048, GLM["NH"], GLM["DR"], GLM["DV"]
    rng = np.random.default_rng(2048)
    qn, qr, kv = _bf16(rng, (NH, DV)), _bf16(rng, (NH, DR)), _bf16(rng, (T, DV + DR), 0.5)
    tok = bind(M.SparseMla, T=T, **GLM)
    want = np.asarray(evaluate(tok, qn.reshape(-1).tolist(), qr.reshape(-1).tolist(), kv.reshape(-1).tolist()), dtype=U32)
    states, nf = TW.chain_rows(qn, qr, kv, T=T, scale=GLM["SCALE"])
    assert len(states) == 32 and not nf.any()
    assert np.array_equal(TW.head_end_rows(states[-1][1], states[-1][2]).astype(U32), want.reshape(NH, DV))


@pytest.mark.parametrize("T", [1, 16, 37, 130])
def test_the_twin_is_each_unit(T: int) -> None:
    """`chain_rows`' state after each block is that block's unit's, and `head_end_rows` of the last is the end's."""
    DR, DV, BN = (16, 16, 16) if T < 100 else (GLM["DR"], GLM["DV"], GLM["BN"])
    qn, qr, kv = _case(T + 3, T, DR, DV, kv_scale=0.5)
    got, states = _chain(T, DR, DV, BN, 0.0625, qn, qr, kv)
    tw, nf = TW.chain_rows(qn[None], qr[None], kv, T=T, DR_=DR, DV=DV, BN=BN, scale=0.0625)
    assert not nf.any() and len(tw) == len(states)
    for (m, l, o), (wm, wl, wo) in zip(tw, states):
        assert int(m[0]) == wm and np.array_equal(l[0], np.asarray(wl, U32)) and np.array_equal(o[0], np.asarray(wo, U32))
    assert np.array_equal(TW.head_end_rows(tw[-1][1], tw[-1][2])[0].astype(U32), got)


def test_head_chain_cuts_a_binding_at_its_statics() -> None:
    """The units of a row of T keys: per head the first visited block (NVIS = T mod BN, or BN), the rest full, and the end."""
    blk = "DR=64,DV=512,NVIS={},FIRST={},BN=64,DOT=HopperBF16WgmmaDot16_v1,SCALE=0.0625"
    end = "SparseMlaHeadEnd_v1{DV=512,INV=Fa3InvSum_v1}"
    got = [(u.id, n) for u, n in M.head_chain(bind(M.SparseMla, T=130, **GLM))]
    assert got == [(f"SparseMlaHeadBlock_v1{{{blk.format(2, True)}}}", 8), (f"SparseMlaHeadBlock_v1{{{blk.format(64, False)}}}", 16),
                   (end, 8)]
    got = [(u.id, n) for u, n in M.head_chain(bind(M.SparseMla, T=64, **GLM))]
    assert got == [(f"SparseMlaHeadBlock_v1{{{blk.format(64, True)}}}", 8), (end, 8)]
    with pytest.raises(ValueError, match="cuts a SparseMla_v1 binding"):
        M.head_chain(bind(M.SparseMlaHead, T=64, **{k: v for k, v in GLM.items() if k != "NH"}))


@pytest.mark.slow
def test_each_unit_fits_the_block_at_the_served_statics() -> None:
    """Every unit `head_chain` stages at GLM-5.3's statics, a first block of 1 to 64 keys, a full block and the end, fits Flock's
    proved range, where the whole row doesn't (`circuit_check.block_held` held it): each read off its Definition, a few
    seconds a block (about a minute); circuit-check's ``block`` checks the bound row's cut only, at T = 65."""
    from verity_flock import circuit as FC

    units = {u.id: u for T in range(1, 65) for u, _n in M.head_chain(bind(M.SparseMla, T=T + 64, **GLM))}
    assert len(units) == 64 + 1 + 1
    for u in units.values():
        b = FC.unit_block(u)
        assert FC.past_block(b) is None, (u.id, b["k_log"])


@pytest.mark.parametrize("first", [True, False])
def test_only_a_later_block_takes_the_running_state(first: bool) -> None:
    unit = bind(M.SparseMlaHeadBlock, DR=16, DV=16, NVIS=16, FIRST=first, BN=16, DOT=FA3["DOT"], SCALE=0.0625)
    names = [n for n, _ in unit.signature()[0]]
    assert names == ["q_nope", "q_rope", "kv"] + ([] if first else ["m_prev", "l_prev", "o_prev"])


def test_only_the_first_visited_block_is_partial() -> None:
    with pytest.raises(ValueError, match="only the first visited block is partial"):
        _ = bind(M.SparseMlaHeadBlock, DR=16, DV=16, NVIS=5, FIRST=False, BN=16, DOT=FA3["DOT"], SCALE=0.0625).body
    with pytest.raises(ValueError, match="multiples of 16"):
        _ = bind(M.SparseMlaHeadBlock, DR=8, DV=16, NVIS=4, FIRST=True, BN=16, DOT=FA3["DOT"], SCALE=0.0625).body


AGREEMENT = [bind(BM.SparseMlaHeadBlockV2, DR=16, DV=16, NVIS=5, FIRST=True, BN=16, SCALE=0.0625, DOT=BM.SPARSE_MLA_GLM53["DOT"]),
             bind(BM.SparseMlaHeadBlockV2, DR=16, DV=16, NVIS=16, FIRST=False, BN=16, SCALE=0.0625, DOT=BM.SPARSE_MLA_GLM53["DOT"]),
             bind(BM.SparseMlaHeadEndV2, DV=16, INV=BM.SPARSE_MLA_GLM53["INV"])]


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    """Each block traces a few hundred thousand ANDs (minutes): what the word-level tests can't see is the Boolean units'
    agreement on special words (zeros, infinities, NaNs, subnormals) through a block's padding, its carried state and the end."""
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k in ("DOT", "INV") else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
    assert SW.boolean_version(word).id == target.id
    inputs = _inputs(word, 96, sum(map(ord, target.id)))
    for g, r in zip(_flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"
