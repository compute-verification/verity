"""REDTEAM phase 12, target 1 (#3745): can a prover register cheap served rows that ``ServedRowsTie_v1`` still accepts?

The lead's questions: rows that match the committed positions but (a) carry another request, (b) come from another step,
or (c) a step the binding map doesn't cover; and whether the shared reader with ``IoTieRequest`` gives one tie's freedom to
the other. The tie's circuit reads only ``valid``, ``n`` and ``start`` (the runs' lengths and offsets) and the given
positions chunks; it reads neither ``kind`` nor ``request`` (``id``), and the real step tree never enters the circuit. So
each of these cases accepts in the circuit alone — and each is backstopped outside it, which these tests pin so the
delegation is explicit and a regression that drops a backstop is caught:

- ``kind``/``request`` free in the circuit are held whole by ``leaf_ok`` (the ``srv`` row's b‖c to the registration) and by
  ``compile_slice``'s ``rows`` digest (the registration's digest is the replay's, which derives ``kind`` from n/start and
  ``request`` from the cert-bound certified arrivals). A flipped ``srv`` registered to pass ``leaf_ok`` publishes a digest
  the honest replay can't reproduce, so the slice's ``rows`` bit refuses it.
- the positions a cheap row claims are internal to the circuit: cheap rows beside a batch fabricated to match them accept,
  and only ``chunk_ok`` ties a positions chunk to its committed leaf of the real step tree (the PR's own
  ``test_a_positions_chunk_is_bound_by_bc_to_its_leaf`` is the native refusal of a wrong index or row).
- a step the binding map doesn't cover: an executed step declared idle (N = 0, zero chunks) accepts, because the circuit
  trusts ``pub``'s N and the chunks it is handed. N and the per-step chunk coverage are the Commit's binding map's, which
  the consumer (PROVE-PATH) must supply for every executed step; the circuit has no self-check for it.

The one exposure to carry: on this branch there is no consumer wiring these natives, and ``IoTieRequest`` is not present, so
end-to-end the tie binds nothing until PROVE-PATH wires ``chunk_ok`` over every executed step at the published index and N
and the ingress tie binds the request ids. Inside the unit the ties as written are sound.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS, served_rows as SR, served_tie as ST
from verity.ml.commitments import advice as ADV
from verity.ml.commitments.frame_v3 import BitRowLeaf
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import row_leaf

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"
SALT = bytes(range(32))
RUN_ROOT = bytes(64)


def _tokens(step):
    ids, pos = [], []
    for _slot, req, _kind, start, n in step:
        ids += [1000 * req + start + t for t in range(n)]
        pos += [start + t for t in range(n)]
    return ids, pos


@pytest.fixture(scope="module")
def x():
    fx = json.loads(FIXTURE.read_text())
    S = SimpleNamespace(**CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"],
                                     K=fx["K"], SEG=fx["SEG"]))
    st = ST.statics(S)
    T = SimpleNamespace(**st)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"])
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    batch = [_tokens(r) for r in rows]
    honest = ST.case(T, SR.window_bits(S, rows), batch)
    return SimpleNamespace(S=S, st=st, T=T, sl=sl, rows=rows, batch=batch, honest=honest)


def _path(served, w, run_root=RUN_ROOT):
    reg = served.registration
    b = SR.binding(run_root)
    dom = ADV.domain(b, len(reg.rows))
    leaves = [row_leaf(dom, i, ADV.INST.tree_leaf(ADV.INST.default_key, served.bc(i)),
                       BitRowLeaf(reg.n_bits).hiding_schema) for i in range(len(reg.rows))]
    lv = tree_levels(dom, leaves)
    return tuple(lv[d][(w >> d) ^ 1] for d in range(len(lv) - 1))


def test_the_honest_window_is_accepted(x):
    assert x.honest.verdict == ST.ACCEPT and x.honest.flags == {"count": True, "pos": True}


def test_the_circuit_reads_neither_kind_nor_request(x):
    """Flip a slot's ``kind`` and its ``request`` in ``srv`` alone, keeping start/n and the positions. The tie accepts:
    neither field is in its circuit (it reads valid, n, start). The request id is ``IoTieRequest``'s to bind, the kind the
    replay's; `check` is the circuit's reference, so the acceptance is the circuit's, not a spec artefact."""
    rows = [list(r) for r in x.rows]
    u = next(i for i, r in enumerate(rows) if r)
    slot, req, kind, start, n = rows[u][0]
    for mutated in ((slot, req, 1 - kind, start, n), (slot, (req + 1) % (1 << x.S.IW), kind, start, n)):
        steps = [tuple(r) for r in rows]
        steps[u] = (mutated, *steps[u][1:])
        c = ST.case(x.T, SR.window_bits(x.S, steps), x.batch)
        assert c.flags == {"count": True, "pos": True}
        assert ST.evaluate(x.st, [c]) == [ST.ACCEPT]


def test_a_flipped_srv_row_is_held_whole_by_leaf_ok(x):
    """``leaf_ok`` holds the whole ``srv`` row (its b‖c) to the registration, so the circuit's freedom over kind/request
    is not the prover's: a kind-flipped row's b‖c is refused against the honest registration. To pass ``leaf_ok`` the
    prover must register the flipped rows — which the next test closes."""
    honest = SR.register(x.S, [list(x.rows)], RUN_ROOT, digest_salts=[SALT])
    rec = honest.record()
    assert SR.leaf_ok(rec, RUN_ROOT, 0, honest.bc(0), _path(honest, 0))
    rows = [list(r) for r in x.rows]
    u = next(i for i, r in enumerate(rows) if r)
    slot, req, kind, start, n = rows[u][0]
    rows[u][0] = (slot, req, 1 - kind, start, n)
    flipped = SR.register(x.S, [[tuple(r) for r in rows]], RUN_ROOT, digest_salts=[SALT])
    assert not SR.leaf_ok(rec, RUN_ROOT, 0, flipped.bc(0), _path(honest, 0))


def test_a_registered_flip_publishes_a_digest_the_replay_cant_reproduce(x):
    """So the prover registers the flip (passing ``leaf_ok``). Its published digest is now the flip's, but
    ``compile_slice``'s ``rows`` bit holds the replay's rows to that digest, and the replay derives kind from n/start and
    request from the cert-bound arrivals — it reproduces the honest rows, whose digest differs. The slice refuses."""
    rows = [list(r) for r in x.rows]
    u = next(i for i, r in enumerate(rows) if r)
    slot, req, kind, start, n = rows[u][0]
    rows[u][0] = (slot, (req + 1) % (1 << x.S.IW), kind, start, n)
    flipped = SR.register(x.S, [[tuple(r) for r in rows]], RUN_ROOT, digest_salts=[SALT])
    assert flipped.digests[0] != CS.rows_digest(x.S, x.rows, SALT)
    honest_replay = CS.case(x.S, x.sl, SALT, digest=flipped.digests[0])
    assert honest_replay.flags["rows"] is False and honest_replay.verdict != CS.ACCEPT


def test_cheap_rows_pass_the_circuit_against_a_fabricated_batch(x):
    """The positions a cheap row claims are internal to the circuit: against the honest batch the six passes are refused
    (the PR's phase-10 test), but against a batch fabricated to match the cheap rows the circuit accepts. Only
    ``chunk_ok`` ties a positions chunk to its committed leaf of the real step tree — the circuit never sees the tree."""
    cheap = CS.case(x.S, CS.Slice(x.sl.arrivals, (0,) * 6, x.sl.steps), SALT)
    assert [u for u, (a, b) in enumerate(zip(cheap.steps, x.rows)) if a != b] == [6]
    against_honest = ST.case(x.T, SR.window_bits(x.S, list(cheap.steps)), x.batch)
    assert against_honest.flags == {"count": False, "pos": False}
    fabricated = [_tokens(r) for r in cheap.steps]
    against_fabricated = ST.case(x.T, SR.window_bits(x.S, list(cheap.steps)), fabricated)
    assert against_fabricated.flags == {"count": True, "pos": True}
    assert ST.evaluate(x.st, [against_honest, against_fabricated]) == [0, ST.ACCEPT]


def test_an_executed_step_declared_idle_is_accepted(x):
    """A step the binding map doesn't cover: declare step 0 (256 prefill tokens) idle — no slots, N = 0, zero chunks. The
    circuit accepts, trusting ``pub``'s N and the chunks it is handed. N and the per-step chunk coverage are the Commit's
    binding map's, which the consumer must supply for every executed step; the unit has no self-check that it did."""
    idle_steps = [()] + list(x.rows[1:])
    idle_batch = [([], [])] + x.batch[1:]
    idle = ST.case(x.T, SR.window_bits(x.S, idle_steps), idle_batch)
    assert idle.flags == {"count": True, "pos": True}
    assert ST.evaluate(x.st, [idle]) == [ST.ACCEPT]


def test_the_shared_read_is_one_chunk_at_the_slice_class(x):
    """At the slice class T·IB = 2048 < 4096, so the ids and the positions share chunk 0 (lo = 0, hi = 1): one
    ``chunk_ok`` binds both regions' bytes to the step tree. But the tie never reads the ids, so it does not connect
    ``srv``'s request to the pinned id bytes; that connection is ``IoTieRequest``'s. The shared reader therefore gives the
    tie IoTieRequest's chunk coverage, not its check — request binding still needs the ingress tie (absent here)."""
    T, G = x.T, ST.geometry(x.T)
    assert (T.T * T.IB, G.lo, G.hi, G.off) == (2048, 0, 1, 2048)
    ids, pos = x.batch[6]
    chunks = ST.port_chunks(T, ids, pos)
    bits = [[(b >> j) & 1 for b in c for j in range(8)] for c in chunks]

    def ints(vs):
        return [sum(int(b) << j for j, b in enumerate(v)) for v in vs]

    assert ints(ST.ids(T, bits))[:len(ids)] == ids
    assert ints(ST.positions(T, bits[G.lo:G.hi]))[:len(pos)] == pos
