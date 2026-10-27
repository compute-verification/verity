"""The served rows' tie (`served_tie.ServedRowsTie_v2`, the routed form) on the compile slice's corpus-smallkv window: the
honest served rows against the honest batch's committed positions are accepted, and the six-pass run's rows, which
the slice accepts under their own registered row, are refused against the batch the server ran. Each negative is refused
by exactly the checks `check` names, in the circuit's verdict; the circuit is `reference` under any switches and accepts
only what `check` accepts; a window's blocks (G < P) are each the whole unit's; the port's shared read gives back the ids
and the positions; and a positions chunk is bound by its b‖c to its leaf of the step's tree."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS, io_tie as IT, served_rows as SR, served_tie as ST
from verity.ml.commitments import hm96
from verity.ml.commitments.frame_v3 import BitRowLeaf
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, bit_row, row_leaf, sha512_row_v2_digest

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"
COUNT, POS = 1 << ST.FLAGS.index("count"), 1 << ST.FLAGS.index("pos")


def _book() -> dict:
    """The registered book the compile slice's charge reads (as `test_network_certifier_compile_slice.book`)."""
    streams = json.loads((FIXTURE.parent / "arith_encode_vectors.json").read_text())["streams"]
    return next(s for s in streams if s["source"].startswith("async-light"))["decisions"]["book"]


BOOK = _book()


def _tokens(step):
    """The batch a step's rows ran: each slot's ids (a stand-in per request and position) and positions, in batch order."""
    ids, pos = [], []
    for _slot, req, _kind, start, n in step:
        ids += [1000 * req + start + t for t in range(n)]
        pos += [start + t for t in range(n)]
    return ids, pos


@pytest.fixture(scope="module")
def x():
    fx = json.loads(FIXTURE.read_text())
    S = SimpleNamespace(**CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"],
                                     K=fx["K"], CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"]))
    st = ST.statics(S)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"], tuple(map(tuple, fx["tokens"])))
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    batch = [_tokens(r) for r in rows]
    honest = ST.case(SimpleNamespace(**st), SR.window_bits(S, rows), batch)
    return SimpleNamespace(S=S, st=st, T=SimpleNamespace(**st), sl=sl, rows=rows, batch=batch, honest=honest)


def _srv(x, steps):
    return SR.window_bits(x.S, steps)


def test_the_honest_window_is_accepted(x):
    assert x.honest.verdict == ST.ACCEPT
    assert ST.evaluate(x.st, [x.honest]) == [ST.ACCEPT]


def test_six_passes_are_refused_against_the_batch_the_server_ran(x):
    """The six-pass run through the tie. The prover registers the six-pass run's rows (user-1 never admitted),
    consistently: their row is the registered one, so the slice accepts them at 0x1F with a 17-bit charge. The batch the
    server ran admitted user-1 at step 6, 29 prefill tokens beside user-0's decode, and those positions are committed;
    the cheap rows end step 6 at 1 token of its 30 and leave 29 uncovered, so the tie refuses on both bits."""
    S, sl = x.S, x.sl
    cheap = CS.case(S, CS.Slice(sl.arrivals, (0,) * 6, sl.steps, sl.tokens), BOOK)
    assert cheap.verdict == CS.ACCEPT and cheap.length == 17 and list(cheap.steps) != x.rows
    served = SR.register(S, [list(cheap.steps)], bytes(64))
    assert served.registration.rows[0] == SR._row(cheap.rows["srv"])[0]
    assert [u for u, (a, b) in enumerate(zip(cheap.steps, x.rows)) if a != b] == [6]
    tied = ST.case(x.T, _srv(x, list(cheap.steps)), x.batch)
    assert tied.flags == {"count": False, "pos": False}
    assert ST.evaluate(x.st, [tied]) == [tied.verdict] == [0]


def _with_step(x, u, step):
    steps = list(x.rows)
    steps[u] = step
    return steps


NEGATIVES = {
    # user-0's decode at step 2 one position on: its token isn't at the committed position
    "start": (lambda x: _with_step(x, 2, ((0, 0, 1, 373, 1),)), None, {"pos"}),
    # step 0's prefill claimed 200 tokens of the 256 it ran
    "short": (lambda x: _with_step(x, 0, ((0, 0, 0, 0, 200),)), None, {"count", "pos"}),
    # step 6's slots swapped: user-1's prefill claimed ahead of user-0's decode in the batch
    "order": (lambda x: _with_step(x, 6, ((0, 1, 0, 0, 29), (1, 0, 1, 376, 1))), None, {"pos"}),
    # step 1's prefill claimed one token longer, against the 116 committed
    "long": (lambda x: _with_step(x, 1, ((0, 0, 0, 256, 117),)), None, {"count"}),
}


@pytest.mark.parametrize("kind", sorted(NEGATIVES))
def test_each_negative_is_refused_by_the_checks_it_names(x, kind):
    steps, batch, refused = NEGATIVES[kind]
    c = ST.case(x.T, _srv(x, steps(x)), batch or x.batch)
    assert {f for f, ok in c.flags.items() if not ok} == refused
    assert ST.evaluate(x.st, [c]) == [c.verdict]


def test_padding_past_N_is_unread_and_a_high_bit_is_refused(x):
    T, G = x.T, ST.geometry(x.T)
    key = f"pos_0_{0}"
    off = 8 * G.off
    rows = dict(x.honest.rows)
    loud = list(rows["pos_2_0"])
    loud[off + 8 * T.PB * 5:off + 8 * T.PB * 6] = [1] * (8 * T.PB)   # step 2 has one token: position 5 is padding
    high = list(rows[key])
    high[off + 8 * T.PB * 3 + 40] = 1                                # step 0's token 3 with bit 40 set
    padded = replace(x.honest, rows={**rows, "pos_2_0": loud})
    wide = replace(x.honest, rows={**rows, key: high})
    assert ST.check(T, padded.rows) == {"count": True, "pos": True}
    assert ST.check(T, wide.rows) == {"count": True, "pos": False}
    assert ST.evaluate(x.st, [padded, wide]) == [ST.ACCEPT, COUNT]


def test_the_circuit_is_the_reference_on_random_rows(x):
    """Random edits of the honest rows: a slot's start, n or valid bit moved, a position moved; the circuit's verdict is
    `check`'s on each."""
    rng = random.Random(7)
    cases = []
    for _ in range(12):
        steps = [list(s) for s in x.rows]
        u = rng.randrange(len(steps))
        k = rng.randrange(len(steps[u]))
        slot, req, kind, start, n = steps[u][k]
        what = rng.choice(("start", "n", "drop"))
        if what == "start":
            start = max(0, start + rng.choice((-1, 1)))
        elif what == "n":
            n = max(1, n + rng.choice((-1, 1)))
        steps[u][k] = (slot, req, kind, start, n)
        if what == "drop":
            del steps[u][k]
        batch = [list(map(list, b)) for b in x.batch]
        if rng.random() < 0.5:
            v = rng.randrange(len(steps))
            if batch[v][1]:
                batch[v][1][rng.randrange(len(batch[v][1]))] += 1
        cases.append(ST.case(x.T, _srv(x, [tuple(s) for s in steps]), [tuple(b) for b in batch]))
    assert ST.evaluate(x.st, cases) == [c.verdict for c in cases]
    assert [ST.reference(x.T, c.rows) for c in cases] == [c.flags for c in cases]
    assert any(c.verdict != ST.ACCEPT for c in cases)


def _switched(x, c, rng, flips):
    rows = dict(c.rows)
    for _ in range(flips):
        g = rng.randrange(x.T.G)
        sw = list(rows[f"sw_{g}"])
        sw[rng.randrange(ST.widths(x.T).SW)] ^= 1
        rows[f"sw_{g}"] = sw
    return replace(c, rows=rows)


def test_under_any_switches_the_circuit_is_the_reference_and_accepts_only_the_statement(x):
    """The honest window, the six passes and the negatives under switches the prover moved: the circuit's verdict is
    `reference`'s (the routing in integers), and it never accepts a unit `check` refuses."""
    rng = random.Random(11)
    base = [x.honest] + [ST.case(x.T, _srv(x, steps(x)), x.batch) for steps, _b, _r in NEGATIVES.values()]
    cases = [_switched(x, c, rng, f) for c in base for f in (1, 3, 40)]
    refs = [ST.reference(x.T, c.rows) for c in cases]
    assert ST.evaluate(x.st, cases) == [ST.verdict(r) for r in refs]
    for c, r in zip(cases, refs):
        assert not (ST.verdict(r) == ST.ACCEPT and c.verdict != ST.ACCEPT)
        assert r["count"] == c.flags["count"]
    assert any(ST.verdict(r) != ST.ACCEPT for c, r in zip(cases, refs) if c.verdict == ST.ACCEPT)


def test_a_run_start_routed_away_is_refused(x):
    """Step 6's two runs (user-0's decode at 0, user-1's prefill at 1): its records routed to each other's places, then
    its prefill's record to the place past N. Each is a permutation, so the network takes it, and the scan refuses it."""
    T, Z = x.T, ST.widths(x.T)
    f = ST._fields(T, _srv(x, x.rows)[6 * Z.SB:7 * Z.SB])
    for perm in ([1, 0] + list(range(2, Z.M)), [0, 30] + [i for i in range(1, Z.M) if i != 30]):
        moved = replace(x.honest, rows={**x.honest.rows, "sw_6": IT.benes(perm) + [0] * (len(x.honest.rows["sw_6"]) - Z.SW)})
        assert ST.reference(T, moved.rows) == {"count": True, "pos": False}
        assert ST.evaluate(x.st, [moved]) == [COUNT]
    assert ST.route(T, f, 30) == x.honest.rows["sw_6"][:Z.SW]


def test_a_windows_blocks_are_its_units(x):
    """G = 1: the window is P units, block j each, the served row whole in each. Every block of the honest window is
    accepted; the short negative's step 0 refuses block 0 only; a unit naming another block, or none, is refused."""
    st = {**x.st, "G": 1}
    S1 = SimpleNamespace(**st)
    honest = [ST.case(S1, _srv(x, x.rows), x.batch, j) for j in range(x.T.P)]
    short = [ST.case(S1, _srv(x, NEGATIVES["short"][0](x)), x.batch, j) for j in range(x.T.P)]
    Z = ST.widths(S1)
    renamed = replace(honest[2], rows={**honest[2].rows, "pub": ST._bits_int(3, Z.JW) + honest[2].rows["pub"][Z.JW:]})
    outside = replace(honest[2], rows={**honest[2].rows, "pub": ST._bits_int(7, Z.JW) + honest[2].rows["pub"][Z.JW:]})
    got = ST.evaluate(st, honest + short + [renamed, outside])
    assert got[:x.T.P] == [ST.ACCEPT] * x.T.P
    assert got[x.T.P:2 * x.T.P] == [0] + [ST.ACCEPT] * (x.T.P - 1) == [c.verdict for c in short]
    assert ST.check(S1, renamed.rows) == {"count": True, "pos": False} and got[-2] == COUNT
    assert ST.check(S1, outside.rows) == {"count": False, "pos": False} and got[-1] == 0


def test_the_shared_read_gives_the_ids_and_positions(x):
    T = x.T
    ids, pos = x.batch[6]
    chunks = ST.port_chunks(T, ids, pos)
    G = ST.geometry(T)
    bits = [[(b >> j) & 1 for b in c for j in range(8)] for c in chunks]
    def ints(vs):
        return [sum(int(b) << j for j, b in enumerate(v)) for v in vs]
    assert ints(ST.positions(T, bits[G.lo:G.hi]))[:len(pos)] == pos
    assert ints(ST.ids(T, bits))[:len(ids)] == ids
    assert set(ints(ST.positions(T, bits[G.lo:G.hi]))[len(pos):]) == {0}


def test_a_positions_chunk_is_bound_by_bc_to_its_leaf(x):
    """The step's stream: the port's chunk, then two more of other values, under the rows-sha512 step's hiding leaves. The
    unit's row, committed as a proof commits a row (its sha512/row/v2 digest under the leaf's salt), has the leaf's b‖c;
    `chunk_ok` takes it to the tree's root at its index only, and refuses another row's b‖c."""
    T = x.T
    key, ks, n_bits = hm96.SHA512.default_key, bytes(range(64)), 8 * ST.CHUNK
    stream = ST.port_chunks(T, *x.batch[6]) + [bytes([7]) * ST.CHUNK, bytes([9]) * ST.CHUNK]
    dom = CommitmentDomain(hashlib.sha512(b"a step's binding").digest(), -1, RangeIndexedDomain(0, len(stream)), hash="sha512")
    salts = [b"".join(hashlib.sha512(ks + i.to_bytes(8, "big") + r.to_bytes(8, "big")).digest() for r in range(3))
             for i in range(len(stream))]
    bcs = [hm96.SHA512.commit_string(key, sha512_row_v2_digest(row, n_bits), y) for row, y in zip(stream, salts)]
    lv = tree_levels(dom, [row_leaf(dom, i, hm96.SHA512.tree_leaf(key, bc), SCHEMA_HM96_SHA512_ROW_V2)
                           for i, bc in enumerate(bcs)])
    root = lv[-1][0]
    G = ST.geometry(T)
    row = x.honest.rows[f"pos_6_{0}"]
    words = [sum(row[16 * k + j] << j for j in range(16)) for k in range(len(row) // 16)]
    proof_bc = hm96.SHA512.commit_string(key, BitRowLeaf(n_bits).layout().digest(bit_row((words, 16))[0]), salts[G.lo])
    assert proof_bc == bcs[G.lo]
    def path(i):
        return tuple(lv[d][(i >> d) ^ 1] for d in range(len(lv) - 1))
    assert ST.chunk_ok(dom, G.lo, proof_bc, path(G.lo), root)
    assert not ST.chunk_ok(dom, G.lo + 1, proof_bc, path(G.lo + 1), root)
    assert not ST.chunk_ok(dom, G.lo, bcs[1], path(G.lo), root)


def test_the_unit_size_is_pinned_by_circuit_check(x):
    pins = json.loads((Path(ST.__file__).with_suffix(".circuit-check.json")).read_text())["definitions"]
    key = "{" + ",".join(f"{k}={x.st[k]}" for k in ST.STATICS) + "}"
    assert pins[f"ServedRowsTie_v2{key}"]["and"] == ST.size(**x.st)["ands"]
    step = "{" + ",".join(f"{k}={x.st[k]}" for k in ST.STEP) + "}"
    assert pins[f"ServedTieStep_v1{step}"]["and"] == ST.step_size(**x.st)["ands"]
