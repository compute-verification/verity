"""Top-p's stage units on bits (`registry.boolean_topp_stages`), each σ of its word unit:

* the word gates that got a Boolean version for them (`BitXor_v2`, `U32Xor_v2`, `SelectBit_v2`, `TopPNumkeep_v2`,
  `NvLogfQuietNan_v1`, `GumbelNoiseLane_v3`) are their primitives on special and random words, gate by gate;
* every unit of a V = 64 row has a Boolean version whose word view is the unit, and its gates on the twin's inputs for the
  unit (`numerics.topp_stage_twins`) compute the twin's words, which the word unit computes (`test_topp_stages`);
* at the served vocabulary every unit's Boolean version, at every split count, has rows a proof block holds
  (`interface.check_proof_block`) and fits one Flock block, and every Definition its body Calls is Boolean;
* select chunks of a served row (the first, the last, and two that a slice's end falls inside) compute the twin's words on
  its inputs, as words and on bits;
* a served select chunk on bits, lowered flat, stays within the nonzeros of the largest unit proved under ``--zk``.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits import boolean as BL
from verity.circuits.boolean.trace import word_of
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate, evaluate_words
from verity.definitions import boolean as BB
from verity.definitions.numerics import topp_stage_twins as TW
from verity.definitions.registry import boolean_topp_stages as BT
from verity.definitions.registry import gumbel_boolean as GB
from verity.definitions.registry import sampling as SA
from verity.definitions.registry import topp_stages as T
from verity.definitions.registry import topp_word_gates as TG
from verity.commitments.windows import interface as IF

from test_topp_stages import ROWS, SERVED_V, f32, row, unit_args

#: the nonzeros of the largest unit proved under ``--zk``: a plain step block of the served row,
#: ``TopPStep_v1{V=154880,S=1,N=2048,FIRST=false,COMBINE=-1}``, whose proof Flock's verifier accepted.  The largest unit the
#: Lean verifier has accepted is a quarter of it: the 1B decode attention row with its table reads in lookup slots
#: (`verity_vllm.pipeline.chunk_rows.reads`).
PROVED_NNZ = 1_266_481_794

SPECIAL = [0, 1, 0x007FFFFF, 0x00800000, 0x34000000, 0x3F000000, 0x3F7FFFFF, 0x3F800000, 0x3F800001, 0x41200000, 0x4B000000,
           0x4E800000, 0x4F000000, 0x4F7FFFFF, 0x4F800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF,
           0x80000000, 0x80000001, 0x807FFFFF, 0x80800000, 0xBF800000, 0xFF7FFFFF, 0xFF800000, 0xFF800001, 0xFFC00000,
           0xFFC00001, 0xFFFFFFFF]


def words(rng: np.random.Generator, n: int) -> np.ndarray:
    """`SPECIAL`, then n random words and n with a random sign, exponent and mantissa each (every exponent as likely)."""
    fields = (rng.integers(0, 2, n, dtype=np.uint64) << 31) | (rng.integers(0, 256, n, dtype=np.uint64) << 23) \
        | rng.integers(0, 1 << 23, n, dtype=np.uint64)
    return np.concatenate([np.array(SPECIAL, np.uint64), rng.integers(0, 1 << 32, n, dtype=np.uint64), fields])


def on_bits(defn, *cols: np.ndarray) -> np.ndarray:
    return evaluate_words(defn, *(c.reshape(-1, 1) for c in cols)).reshape(-1).astype(np.uint64)


def reference(prim, *cols: np.ndarray) -> np.ndarray:
    return np.array([prim.evaluate(*(int(c[i]) for c in cols)) for i in range(len(cols[0]))], np.uint64)


def test_the_word_gates_on_bits_are_their_primitives():
    rng = np.random.default_rng(3)
    a = words(rng, 3000)
    quiet, gpu = on_bits(GB.NvLogfQuietNan, a), on_bits(GB.NvLogf, a)
    assert np.array_equal(quiet, reference(TG.NvLogf, a)) and np.array_equal(gpu, reference(TG.NvLogfV2, a))
    # the two logs differ only in their NaN results: NvLogf_v2's is one NaN, NvLogf_v1's the operand quieted or the invalid NaN
    nan = ((quiet & 0x7F800000) == 0x7F800000) & ((quiet & 0x7FFFFF) != 0)
    assert nan.sum() > 1000 and np.array_equal(quiet[~nan], gpu[~nan]) and set(gpu[nan].tolist()) == {SA.LOG_NAN}
    assert {SA.LOG_INVALID_NAN, 0x7FC00001, 0xFFC00001} <= set(quiet[nan].tolist())

    counts = np.concatenate([np.array([0, 1, 2, 3, 1 << 31, (1 << 32) - 1, SERVED_V], np.uint64),
                             rng.integers(0, 1 << 32, 2000, dtype=np.uint64), rng.integers(0, 2 * SERVED_V, 4000, dtype=np.uint64)])
    q = np.concatenate([words(rng, 1000), (rng.integers(0, 2 * SERVED_V, 4000) * rng.random(4000)).astype(np.float32)
                        .view(np.uint32).astype(np.uint64)])
    n = min(len(counts), len(q))
    assert np.array_equal(on_bits(BT.TopPNumkeep, counts[:n], q[:n]), reference(TG.TopPNumkeep, counts[:n], q[:n]))

    x, y = rng.integers(0, 1 << 32, (2, 500), dtype=np.uint64)
    assert np.array_equal(on_bits(BT.U32Xor, x, y), x ^ y) and np.array_equal(x ^ y, reference(TG.U32Xor, x, y))
    c, s, t = (np.array(v, np.uint64) for v in zip(*[(c, s, t) for c in (0, 1) for s in (0, 1) for t in (0, 1)]))
    assert np.array_equal(on_bits(BT.BitXor, s, t), reference(TG.BitXor, s, t))
    assert np.array_equal(on_bits(BT.SelectBit, c, s, t), reference(TG.SelectBit, c, s, t))


@pytest.mark.slow
def test_a_lane_s_noise_on_bits_is_its_primitive():
    rng = np.random.default_rng(4)
    keys = np.concatenate([np.array([0, 1, 0xFFFFFFFF, 0x80000000], np.uint64), rng.integers(0, 1 << 32, 508, dtype=np.uint64)])
    lanes = np.concatenate([np.array([0, 1, SERVED_V - 1, 0xFFFFFFFF], np.uint64), rng.integers(0, SERVED_V, 508, dtype=np.uint64)])
    assert np.array_equal(on_bits(GB.GumbelNoiseLane, keys, lanes), reference(SA.GumbelNoiseLane, keys, lanes))


def test_each_new_boolean_version_is_the_one_its_word_resolves_to():
    for b, w in ((BT.BitXor, TG.BitXor), (BT.U32Xor, TG.U32Xor), (BT.SelectBit, TG.SelectBit), (BT.TopPNumkeep, TG.TopPNumkeep),
                 (GB.NvLogfQuietNan, TG.NvLogf), (GB.NvLogf, TG.NvLogfV2), (GB.GumbelNoiseLane, SA.GumbelNoiseLane)):
        assert descriptor_id(BB.boolean_version(w)) == f"{b.name}_v{b.version}" and word_of(b) is w, w.id
    assert [(d.name, d.version) for d in BT.BOOLEAN_STAGE_DEFINITIONS] == [(d.name, 2) for d in T.STAGE_DEFINITIONS]


@pytest.mark.slow
def test_each_unit_on_bits_computes_the_twin_s_words_on_its_inputs():
    """Every unit of a V = 64 row, on bits: its Boolean version's word view is the unit, and its gates, evaluated bit-sliced on
    the twin's inputs for the unit on the four rows of `test_topp_stages` (ties, -inf and NaN lanes, temperature 0, 0.7 and 1,
    four split counts), compute the twin's words, which are the word unit's.  Most of a minute, all of it the gates of the
    row's 536 units of 61 Definitions: nothing short of evaluating them holds σ's Boolean versions to the words the row
    commits, and a smaller row has no select chunk of two lanes."""
    V = 64
    lgs, opss, outss = [], [], []
    for kind, temp, top_p, at in ROWS:
        lg = row(V, kind, seed=V + len(kind))
        ops = {"top_p": f32(top_p), "seed": 0x1234_5678_9ABC_DEF0, "pos": 41, "temp": f32(temp)}
        _tok, outs = TW.stage_rows(V, lg, splits=at, **ops)
        lgs.append(lg), opss.append({**ops, "splits": at}), outss.append(outs)
    bad = []
    for u in T.plan(V):
        b = BB.boolean_version(u.defn)
        assert BL.is_boolean(b) and descriptor_id(word_of(b)) == descriptor_id(u.defn), u.key
        assert b.defn in BT.BOOLEAN_STAGE_DEFINITIONS and b.bindings == u.defn.bindings, u.key
        args = [unit_args(u, lg, ops, outs) for lg, ops, outs in zip(lgs, opss, outss)]
        cols = [np.array([np.atleast_1d(np.asarray(a[g], dtype=np.uint64)) for a in args]) for g in range(len(args[0]))]
        got = evaluate_words(b, *cols).reshape(len(ROWS), -1).astype(np.uint64)
        want = np.array([[int(v) for f in outs[u.key] for v in f] for outs in outss], np.uint64)
        if got.shape != want.shape or not np.array_equal(got, want):
            bad.append(u.key)
    assert bad == []


@pytest.mark.slow
def test_each_served_unit_on_bits_fits_a_proof_block():
    """At GLM-5.3's vocabulary, each of the 85 unit Definitions (every split count's) has a Boolean version whose body Calls
    only Boolean Definitions, on the word unit's rows, which a proof block holds (`interface.check_proof_block`), and whose
    ANDs and ports, read off the Definition, leave its unit inside one Flock block (`verity_flock.circuit.unit_block`).  Most
    of a minute, all of it the served bodies: a block's size is its served statics', which no smaller binding has."""
    from verity_flock import circuit as FC

    units = {descriptor_id(u.defn): u.defn for u in T.plan(SERVED_V)}
    assert len(units) == 85
    for t, d in units.items():
        b = BB.boolean_version(d)
        assert descriptor_id(word_of(b)) == t and all(BL.is_boolean(n.fn) for n in b.body.nodes), t
        assert IF.check_proof_block(b) == IF.check_proof_block(d) <= IF.K_MAX and IF.unit_rows(b) == IF.unit_rows(d), t
        block = FC.unit_block(b)
        assert FC.past_block(block) is None and block["k_log"] <= IF.K_MAX, (t, block)


@pytest.mark.slow
def test_the_served_select_chunks_compute_the_twin_s_words():
    """At GLM-5.3's vocabulary, select chunks of a served row compute the twin's words on its inputs for them, as words and on
    bits (bit-sliced, so with every word `gumbel_boolean.philox_round` commits): the first chunk, the last, and the two forms
    of a middle one that a V = 64 row's chunks of two lanes never take, a chunk that a slice's end falls inside at S = 32
    and one at S = 16 and 32 both, which carry on the offsets `topp_stages.positions` leaves past the end.  The row is a
    plain one, on which no tie count decides a lane past the end: what those lanes read is held on rows of 1,000 lanes
    (`test_topp_stages`).  About a minute, most of it the twin over the row: these are the Definitions a served row's 4,840
    select units are, at their 32 lanes, and no smaller row has them."""
    V, n = SERVED_V, T.chunk(SERVED_V)
    lg = row(V, "plain", seed=V)
    ops = {"top_p": f32(0.9), "seed": 0x1234_5678_9ABC_DEF0, "pos": 41, "temp": f32(0.7), "splits": 32}
    _tok, outs = TW.stage_rows(V, lg, **ops)
    sel = [u for u in T.plan(V) if u.key[0] == "select"]
    inside = {S: [k for k, u in enumerate(sel) if any(u.lanes[0] < e < u.lanes[1] for e in range(T.Geo(V, S).L, V, T.Geo(V, S).L))]
              for S in (16, 32)}
    picked = [0, min(set(inside[32]) - set(inside[16])), min(inside[16]), len(sel) - 1]
    assert n == 32 and set(inside[16]) < set(inside[32]) and len(set(picked)) == 4
    assert {(sel[k].defn.bindings["FIRST"], sel[k].defn.bindings["LAST"]) for k in picked} == {(True, False), (False, False),
                                                                                           (False, True)}
    for k in picked:
        u = sel[k]
        args = unit_args(u, lg, ops, outs)
        want = [int(v) for f in outs[u.key] for v in f]
        assert [int(v) for v in evaluate(u.defn, *args)] == want, u.key
        b = BB.boolean_version(u.defn)
        assert descriptor_id(word_of(b)) == descriptor_id(u.defn) and u.defn.bindings["N"] == n, u.key
        got = evaluate_words(b, *(np.atleast_1d(np.asarray(a, dtype=np.uint64))[None] for a in args))
        assert got.reshape(-1).astype(np.uint64).tolist() == want, u.key


@pytest.mark.slow
def test_a_served_select_chunk_on_bits_is_within_the_largest_unit_proved():
    """A select chunk of the served row on bits, lowered flat as serving's stager lowers it (`class_statement.unit_lowering`,
    whose layout's nonzeros a staged unit reports), has no more nonzeros than the largest unit proved under ``--zk``
    (`PROVED_NNZ`).  The count is a one-lane and a two-lane chunk's at the served vocabulary, scaled to the chunk's lanes
    (`topp_stages.chunk`), not the chunk's own, whose lowering takes minutes and gigabytes: what grows faster than the lanes
    (the tie counts' scan, the running best) is left out of it.  It fails when a lane's nonzeros grow back
    (`gumbel_boolean.philox_round` commits each round's words, and without that a lane has ten times these and a chunk of 32
    more than twice the bound) and when `topp_stages.SELECT_N` is raised until a chunk passes the bound, a little over four
    times its 32 lanes, which are a quarter of it; the 32 lanes themselves are held by the served counts (`test_topp_stages`)."""
    from verity_flock.class_statement import unit_lowering

    def nnz(lanes: int) -> int:
        b = BB.boolean_version(bind(T.GumbelTopPSelect, V=SERVED_V, N=lanes, FIRST=False, LAST=False))
        low, _words = unit_lowering(b, "unit/select", [(r, ps) for r, _schema, _bits, ps in IF.unit_rows(b)[0]])
        return int(low.layout.counts["nnz_a"] + low.layout.counts["nnz_b"])

    n, one, two = T.chunk(SERVED_V), nnz(1), nnz(2)
    lane = two - one
    chunk = one + (n - 1) * lane
    assert chunk <= PROVED_NNZ, (
        f"a served select chunk on bits is about {chunk:,} nonzeros lowered flat ({n} lanes of {lane:,}), past the "
        f"{PROVED_NNZ:,} of the largest unit proved under --zk: set topp_stages.SELECT_N to at most "
        f"{1 + (PROVED_NNZ - one) // lane} lanes, or, if a lane's nonzeros grew, commit the form its ANDs now read (as "
        "gumbel_boolean.philox_round commits a round's words); raise PROVED_NNZ only to a larger unit proved under --zk")
