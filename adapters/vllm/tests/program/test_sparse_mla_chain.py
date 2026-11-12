"""`sparse_mla.SparseMlaHeadBlock_v1` and `SparseMlaHeadEnd_v1`, the cut serving stages a `SparseMla_v1` row as
(`sparse_mla.head_chain`): per head its key blocks in FA3's order, each a unit carrying the online softmax's state to the next,
and its end are the whole Call word for word, at small statics and at GLM-5.3's served ones; each unit fits the Flock block where
the whole doesn't; the end is `dead_free.AttnHeadEnd_v1`'s function; the twin (`sparse_mla_split_twins.chain_rows`,
`head_end_rows`) is each unit's word; the Boolean versions agree with their word views; the key block written as Calls
(`boolean_sparse_mla.SparseMlaHeadBlock_v3`) is the traced one, its step is FA3's and lowers by its builder to the circuit its
gates lower to, and at the served statics its description stays small and its Flock block under its word view's."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.codec import canonical_json, encode_program
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.program import Program
from verity.definitions import boolean as SW
from verity.definitions.numerics import sparse_mla_split_twins as TW
from verity.definitions.registry import boolean_attention as BA
from verity.definitions.registry import boolean_sparse_mla as BM
from verity.definitions.registry import dead_free as DF
from verity.definitions.registry import sparse_mla as M
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


#: a partial first block (its P V step an `AttnDotStep` on zero rows), a full later one (every step a `SparseMlaDotStep`), and
#: a first block of a full step and a partial one
CALLS = {"partial_first": dict(DR=16, DV=16, NVIS=5, FIRST=True, BN=16),
         "full_later": dict(DR=16, DV=16, NVIS=16, FIRST=False, BN=16),
         "full_then_partial_first": dict(DR=16, DV=32, NVIS=20, FIRST=True, BN=32)}


@pytest.mark.slow
@pytest.mark.parametrize("S", CALLS.values(), ids=list(CALLS))
def test_the_block_as_calls_is_the_traced_block(S: dict) -> None:
    """`SparseMlaHeadBlock_v3` has `_v2`'s word view and its function, special words included, and `boolean_version` still
    gives `_v2`.  Each binding traces a few hundred thousand ANDs twice (under a minute): only this holds the block written as
    Calls to the traced one bit for bit, with its key words' own gates shared by the score and P V steps."""
    calls, traced = (bind(d, SCALE=0.0625, DOT=BM.SPARSE_MLA_GLM53["DOT"], **S) for d in (BM.SparseMlaHeadBlockV3, BM.SparseMlaHeadBlockV2))
    word = TR.word_of(calls)
    assert word.id == TR.word_of(traced).id and SW.boolean_version(word).id == traced.id
    assert calls.signature() == traced.signature()
    inputs = _inputs(word, 96, sum(map(ord, calls.id)))
    want = _flat(E.evaluate_batch(word, *inputs))
    for unit in (calls, traced):
        for g, r in zip(_flat(BS.evaluate_words(unit, *inputs)), want, strict=True):
            bad = np.flatnonzero((g != r).any(axis=1))
            assert not bad.size, f"{unit.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"


def test_the_shared_step_is_fa3s_step() -> None:
    """`SparseMlaDotStep_v1{DOT}` over a's own gates (`AttnDotStepA`) and each word of b's (`SparseMlaDotStepB_v1{DOT}`) is
    DOT(acc, a, b) gate for gate and bit for bit, on random words and special ones."""
    dot = BM.SPARSE_MLA_GLM53["DOT"]
    own_a = bind(BA.AttnDotStepA, DOT=dot, NV=16, FROM_ZERO=False)
    own_b, step = bind(BM.SparseMlaDotStepB, DOT=dot), bind(BM.SparseMlaDotStep, DOT=dot)
    assert [n for n, _ in step.signature()[0]] == ["a", "b", "acc", "ad", "bd"]
    parts, whole = [BS.compile(x).counts for x in (own_a, own_b, step)], BS.compile(bind(dot)).counts
    assert {g: parts[0][g] + 16 * parts[1][g] + parts[2][g] for g in whole} == whole
    acc, a, b = _inputs(FA3["DOT"], 2000, 215)

    def bits(words: np.ndarray, width: int) -> np.ndarray:
        return (words.astype(np.uint64)[..., None] >> np.arange(width, dtype=np.uint64) & np.uint64(1)).astype(np.uint8)

    ab, bb = bits(a, 16), bits(b, 16)
    bd = BS.evaluate_bits(own_b, bb.reshape(-1, 16)).reshape(len(b), 16, -1)
    got = BS.evaluate_bits(step, ab, bb, bits(acc, 32), BS.evaluate_bits(own_a, ab), bd)
    assert np.array_equal(got, bits(np.asarray(E.evaluate_batch(FA3["DOT"], acc, a, b)).reshape(-1), 32))


def _lowered(unit) -> tuple[list, int]:
    """Flock's flat lowering of ``unit`` (`ir_lower.lower`, as a served unit's stager runs it): each committed bit and the
    output's forms, and the activations it lowered by a builder."""
    from verity.circuits.evaluation.reference import signature_of
    from verity_flock import ir_lower as IL
    from verity_flock.class_statement import _UnitCircuit

    C, built = _UnitCircuit(), []
    activation = IL._lower_activation

    def counted(C, s, *rest):
        built.append(s.spec.id.split("_v")[0])
        return activation(C, s, *rest)

    params, _ = signature_of(unit)
    ins = [C.input(f"in{j}", w) for j, w in enumerate(w for t in params for w in t.leaf_widths())]
    IL._lower_activation = counted
    try:
        out = [b for w in IL.lower(C, unit, ins) for b in w]
    finally:
        IL._lower_activation = activation
    bits = [(k, (tuple(d[0].s), d[0].c, tuple(d[1].s), d[1].c) if k == "and" else d) for k, d in zip(C.kind, C.data)]
    return [bits, [(tuple(x.s), x.c) for x in out]], sum(n in ("SparseMlaDotStep", "SparseMlaDotStepB") for n in built)


@pytest.mark.slow
def test_the_steps_builders_lower_to_their_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """`SparseMlaDotStep` and `SparseMlaDotStepB` carry a builder, so Flock's flat lowering runs each of their activations and
    keeps no word a gate of them. The circuit is the one their gates lower to, AND for AND: each part alone, and a block of
    5 keys (10 steps and 160 key words, each by its builder). About 10 s for the block's two lowerings; only the block has
    a builder read its operands from the other Calls' wires."""
    dot = BM.SPARSE_MLA_GLM53["DOT"]
    units = [bind(BM.SparseMlaDotStepB, DOT=dot), bind(BM.SparseMlaDotStep, DOT=dot),
             bind(BM.SparseMlaHeadBlockV3, SCALE=0.0625, DOT=dot, **CALLS["partial_first"])]
    with_builders = [_lowered(u) for u in units]
    assert [n for _, n in with_builders] == [1, 1, 5 * 32 + 2 * 5]
    monkeypatch.delattr(BM.SparseMlaDotStepB, "build")
    monkeypatch.delattr(BM.SparseMlaDotStep, "build")
    for unit, (circuit, _) in zip(units, with_builders, strict=True):
        by_gates, n = _lowered(unit)
        assert n == 0 and by_gates == circuit, unit.id


@pytest.mark.parametrize("nvis,first", [(64, False), (64, True), (16, True)])
def test_the_block_as_calls_stays_small_at_the_served_statics(nvis: int, first: bool) -> None:
    """At GLM-5.3's statics the block written as Calls is a few rows of Calls: its Program's description is under a byte for
    every 4 of its gates (a traced block's holds every gate), and its Flock block, read off the Definition as circuit-check's
    ``block`` reads it, is within the proved range and under its word view's, whose steps decode each operand again."""
    from verity_flock import circuit as FC

    S = {k: BM.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}
    unit = bind(BM.SparseMlaHeadBlockV3, NVIS=nvis, FIRST=first, **S)
    assert len(canonical_json(encode_program(Program(unit)))) * 4 < unit.gates
    b, word = FC.unit_block(unit), FC.unit_block(TR.word_of(unit))
    assert FC.past_block(b) is None, (unit.id, b["k_log"])
    assert b["bits"] < word["bits"], (unit.id, b["bits"], word["bits"])
