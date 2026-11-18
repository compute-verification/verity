"""The masked units of a decode step that read the key cache, `SparseMlaHeadBlockLive_v1` and `SparseMlaHeadKeysWireLive_v1`
and their Boolean versions, held to the units they mask: where the step's bit (bit 0 of ``live``) is 1 their outputs are
the unmasked Definition's word for word, at small statics and at GLM-5.3's served ones (Gaussian rows, specials planted),
and where it is 0 every output word is zero whatever ``kv`` and the carried state hold, NaNs and infinities included;
``live`` is a row of its own read at run time, not a constant; the mask is in each Definition's text and hash; and at the
served statics each masked unit is one Flock block, the unmasked unit's ANDs and one AND an output bit more, one row more."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.ops.boolean import versions as SW
from verity_vllm.definitions.registry import boolean_sparse_mla as BM
from verity_vllm.definitions.registry import sparse_mla as M
from tests.adapters.vllm.verity_vllm.definitions.test_sparse_mla_head_units import (BF16_SPECIALS, F32_SPECIALS, SERVED, SMALL, _bf16,
                                                                   _inputs, _planted)

_BITS = dict(DR=16, DV=16, BN=16, DOT=BM.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)
_SERVED_BITS = {k: BM.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}


def _live(n: int, on: int, rng) -> np.ndarray:
    """``n`` instances of ``live`` as the word Definition takes it, 16 bits each: bit 0 ``on``, the 15 above it random, which
    no unit reads (a row holds zeros there)."""
    bits = rng.integers(0, 2, (n, 16)).astype(np.uint8)
    bits[:, 0] = on
    return bits


def _word(live: np.ndarray) -> np.ndarray:
    """``live``'s 16 bits as the Boolean version takes the row: one u16 word an instance, bit 0 low."""
    return (live.astype(np.uint64) << np.arange(16, dtype=np.uint64)).sum(axis=1, keepdims=True).astype(np.uint16)


def _poisoned(rng, ins: list[np.ndarray]) -> list[np.ndarray]:
    """``ins`` with every bf16 word of the keys and a quarter of the queries' and every f32 word of the state a special:
    a dead step's cache rows and state, whatever they hold."""
    out = []
    for j, a in enumerate(ins):
        specials = BF16_SPECIALS if a.dtype == np.uint16 else F32_SPECIALS
        b = a.copy()
        at = np.ones(b.shape, bool) if j >= 2 else rng.random(b.shape) < 0.25
        b[at] = rng.choice(specials, int(at.sum()))
        out.append(b)
    return out


def _block_outputs(st: dict, NVIS: int, FIRST: bool, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    block = bind(M.SparseMlaHeadBlock, NVIS=NVIS, FIRST=FIRST, **st)
    live = bind(M.SparseMlaHeadBlockLive, NVIS=NVIS, FIRST=FIRST, **st)
    ins = _inputs(n, NVIS, FIRST, st["DR"], st["DV"], seed)
    want = np.asarray(E.evaluate_batch(block, *ins)).astype(np.uint64)
    got = np.asarray(E.evaluate_batch(live, *ins, _live(n, 1, rng))).astype(np.uint64)
    assert got.shape == want.shape == (n, 1 + 4 + st["DV"])
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not bad.size, f"{live.id} at a live step: differs from {block.id} in {bad.size} of {n}, first {bad[:5]}"
    for dead_ins in (ins, _poisoned(rng, ins)):
        dead = np.asarray(E.evaluate_batch(live, *dead_ins, _live(n, 0, rng))).astype(np.uint64)
        assert not dead.any(), f"{live.id} at a dead step: {np.count_nonzero(dead.any(axis=1))} of {n} return a nonzero word"


@pytest.mark.parametrize("NVIS,FIRST", [(16, False), (16, True), (5, True), (1, True)])
def test_the_masked_block_is_the_block_where_live_and_zeros_where_dead(NVIS: int, FIRST: bool) -> None:
    _block_outputs(SMALL, NVIS, FIRST, 32, 3 * NVIS + FIRST)


@pytest.mark.parametrize("NVIS,FIRST", [(64, False), (44, True)])
def test_the_masked_block_at_the_served_statics(NVIS: int, FIRST: bool) -> None:
    """GLM-5.3's DR = 64, DV = 512: O's 512 words and a later block's carried state, which the small statics' 16 don't reach."""
    _block_outputs(SERVED, NVIS, FIRST, 3, 17 + NVIS + FIRST)


@pytest.mark.parametrize("NVIS,W,C", [(16, 16, 0), (5, 8, 3), (64, 64, 8)])
def test_the_masked_writer_is_the_writer_where_live_and_zeros_where_dead(NVIS: int, W: int, C: int) -> None:
    rng = np.random.default_rng(NVIS + W + C)
    keys = bind(M.SparseMlaHeadKeysWire, NVIS=NVIS, W=W, C=C)
    live = bind(M.SparseMlaHeadKeysWireLive, NVIS=NVIS, W=W, C=C)
    n = 16
    kv = _planted(rng, _bf16(rng, (n, NVIS, W)), BF16_SPECIALS, 0.05)
    want = np.asarray(E.evaluate_batch(keys, kv)).astype(np.uint64)
    assert np.array_equal(np.asarray(E.evaluate_batch(live, kv, _live(n, 1, rng))).astype(np.uint64), want)
    nan = np.full_like(kv, 0x7FC0)
    for dead in (kv, nan):
        assert not np.asarray(E.evaluate_batch(live, dead, _live(n, 0, rng))).astype(np.uint64).any()


@pytest.mark.parametrize("target", [bind(BM.SparseMlaHeadBlockLiveV2, NVIS=5, FIRST=True, **_BITS),
                                    bind(BM.SparseMlaHeadBlockLiveV2, NVIS=16, FIRST=False, **_BITS),
                                    bind(BM.SparseMlaHeadKeysWireLiveV2, NVIS=16, W=16, C=2)], ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view_on_live_and_dead_steps(target) -> None:
    word = TR.word_of(target)
    assert SW.boolean_version(word).id == target.id
    n = 32
    rng = np.random.default_rng(sum(map(ord, target.id)))
    params = word.signature()[0]
    if "KeysWire" in word.id:
        ins = [_planted(rng, _bf16(rng, (n, word.bindings["NVIS"], word.bindings["W"])), BF16_SPECIALS, 0.05)]
    else:
        ins = _inputs(n, word.bindings["NVIS"], word.bindings["FIRST"], _BITS["DR"], _BITS["DV"], n)
    assert len(ins) == len(params) - 1 and params[-1][0] == M.LIVE
    live = _live(n, 0, rng)
    live[: n // 2, 0] = 1
    got = np.asarray(BS.evaluate_words(target, *ins, _word(live))).astype(np.uint64).reshape(n, -1)
    want = np.asarray(E.evaluate_batch(word, *ins, live)).astype(np.uint64).reshape(n, -1)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert got.shape == want.shape and not bad.size, f"{target.id}: {bad.size} of {n} instances differ from {word.id}"
    assert want[: n // 2].any() and not want[n // 2:].any()


def test_live_is_a_row_of_its_own_that_the_unit_reads_at_run_time() -> None:
    """``live`` is the masked unit's last input row, 16 bits as the select's ``o1`` is held, staged like its others
    (`interface.unit_rows`); its flat unit (`class_statement.unit_lowering`) on the same rows returns the block's words when
    the row's bit 0 is 1 and zeros when it is 0, so the bit is read from the row and not built into the unit."""
    from verity.commitments.periods import interface as IF
    from verity_flock import class_statement as CL
    from verity_flock import ir_lower as IL

    unit = bind(BM.SparseMlaHeadBlockLiveV2, NVIS=2, FIRST=True, **_BITS)
    ins, _outs = IF.unit_rows(unit)
    assert [(r, s, b) for r, s, b, _ps in ins] == [(p, IF.V2, b) for p, b in (("q_nope", 256), ("q_rope", 256), ("kv", 1024),
                                                                               (M.LIVE, 16))]
    low, out_words = CL.unit_lowering(unit, "live", [(r, ps) for r, _s, _b, ps in ins])
    rng = np.random.default_rng(2)
    qn, qr, kv = _inputs(1, 2, True, _BITS["DR"], _BITS["DV"], 9)
    want = [int(w) for w in np.asarray(BS.evaluate_words(bind(BM.SparseMlaHeadBlockV3, NVIS=2, FIRST=True, **_BITS), qn, qr, kv)
                                       ).astype(np.uint64).reshape(-1)]
    for on in (1, 0):
        words = [[int(x)] for a in (qn, qr, kv) for x in a.reshape(-1)] + [[int(_word(_live(1, on, rng))[0, 0])]]
        outs, ok = IL.evaluate_layout(low.layout, words, 1)
        got = [o[0] for o in outs]
        assert ok and len(got) == sum(out_words)
        # the running max and lanes are one f32 word each in two u16 words, O's the same
        halves = [(w >> (16 * h)) & 0xFFFF for w in want for h in (0, 1)]
        assert got == (halves if on else [0] * len(got)), on


@pytest.mark.parametrize("live,unmasked", [
    (bind(M.SparseMlaHeadBlockLive, NVIS=16, FIRST=False, **SMALL), bind(M.SparseMlaHeadBlock, NVIS=16, FIRST=False, **SMALL)),
    (bind(BM.SparseMlaHeadBlockLiveV2, NVIS=5, FIRST=True, **_BITS), bind(BM.SparseMlaHeadBlockV3, NVIS=5, FIRST=True, **_BITS)),
    (bind(M.SparseMlaHeadKeysWireLive, NVIS=16, W=16, C=1), bind(M.SparseMlaHeadKeysWire, NVIS=16, W=16, C=1)),
    (bind(BM.SparseMlaHeadKeysWireLiveV2, NVIS=16, W=16, C=1), bind(BM.SparseMlaHeadKeysWireV2, NVIS=16, W=16, C=1))],
    ids=lambda t: t.id)
def test_the_mask_is_in_the_definitions_text_and_hash(live, unmasked) -> None:
    """Each masked Definition is a family of its own (its id and its circuit digest not the unmasked unit's), and its body is
    one Call of the unmasked unit at the same statics on its other parameters, then selects whose bit is ``live``'s bit 0,
    the parameter itself, and whose other operands are the Call's words and a zero: no constant stands in for the bit."""
    from verity.circuits.codec import program_digest
    from verity.circuits.program import Program

    assert live.id != unmasked.id and live.defn.name == unmasked.defn.name + "Live"
    assert program_digest(Program(live)) != program_digest(Program(unmasked))
    body = live.body
    k = len(body.params) - 1
    assert body.params[k][0] == M.LIVE and body.params[:k] == tuple(unmasked.signature()[0])
    call, *rest = body.nodes
    assert call.form == "call" and call.fn.id == unmasked.id
    assert [(a.refs.space, a.refs.idx) for a in call.args] == [("p", j) for j in range(k)]
    consts = [n for n in rest if n.form == "primitive" and n.fn.id.startswith("Const")]
    assert len(consts) == 1 and "[0x0" in consts[0].fn.id
    selects = [n for n in rest if n not in consts]
    assert selects and all("Select" in n.fn.id for n in selects)
    for n in selects:
        bit = n.args[0].refs
        assert (bit.space, bit.idx, bit.base, bit.count) == ("p", k, 0, 1)
    S = live.bindings
    words = 1 + 4 + S["DV"] if "DV" in S else S["NVIS"] * S["W"]
    assert sum(n.n for n in selects) == words


@pytest.mark.parametrize("NVIS,FIRST", [(64, False), (64, True), (44, True), (16, True)])
def test_the_served_masked_blocks_are_one_flock_block_each(NVIS: int, FIRST: bool) -> None:
    """At GLM-5.3's statics, read off the Calls: the masked block takes `SparseMlaHeadBlock_v3`'s block, 2^27 at 64 keys (the
    range's top, `interface.K_MAX`), 2^27 at 44 and 2^25 at 16, with one AND an output bit (32 (1 + 4 + 512) = 16,544) and
    one row more than it."""
    from verity.commitments.periods import interface as IF
    from verity_flock import circuit as FC

    block = FC.unit_block(bind(BM.SparseMlaHeadBlockV3, NVIS=NVIS, FIRST=FIRST, **_SERVED_BITS))
    live = FC.unit_block(bind(BM.SparseMlaHeadBlockLiveV2, NVIS=NVIS, FIRST=FIRST, **_SERVED_BITS))
    assert FC.past_block(live) is None and live["k_log"] == block["k_log"] == {64: 27, 44: 27, 16: 25}[NVIS] <= IF.K_MAX
    assert (live["ands"] - block["ands"], live["rows"] - block["rows"]) == (32 * (1 + 4 + 512), 1)


@pytest.mark.parametrize("NVIS,k_log", [(64, 24), (16, 22)])
def test_the_served_masked_writers_are_one_flock_block_each(NVIS: int, k_log: int) -> None:
    """A writer of a column block of 64 at GLM-5.3's statics: the writer's block, with one AND a bit of its rows."""
    from verity_flock import circuit as FC

    keys = FC.unit_block(bind(BM.SparseMlaHeadKeysWireV2, NVIS=NVIS, W=64, C=0))
    live = FC.unit_block(bind(BM.SparseMlaHeadKeysWireLiveV2, NVIS=NVIS, W=64, C=0))
    assert FC.past_block(live) is None and live["k_log"] == keys["k_log"] == k_log
    assert (live["ands"] - keys["ands"], live["rows"] - keys["rows"]) == (16 * NVIS * 64, 1)
