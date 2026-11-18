"""A kv session's serving trace under refresh, driven as the prover's session loop drives it over the reference backend:
each window's due re-keys go out in their slots with their origins at its first step and are recorded as memory-map
re-keys, pages take a fresh slot when sent and wait while the window's are used, are committed and never encoded under
their own coin, and every pass's plan and opening keep the refresh rules."""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "verity/protocols/resource_accounting/memory/poub/prover/poub_session"))

from verity.commitments import Commitment, verify_opening  # noqa: E402
from verity.commitments import Opening as MerkleOpening  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.prover import audit_prover  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier.rota import Schedule  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier import audit, codec, receipts as rb  # noqa: E402
from verity.protocols.resource_accounting.memory.poub.verifier.session import wire  # noqa: E402

import prover  # noqa: E402

N, LIFETIME, WINDOWS, PASSES, PAGES, FRESH = 16, 2, 6, 3, 2, 2
NODE = b"refresh-node"


def backend() -> prover.CpuBackend:
    b = prover.CpuBackend(N, random.Random(5))
    reg = audit.register(b"source", NODE, N, b.payload_root(NODE), codec.Codec(codec.P512, 2, False), nonce=bytes(32))
    b.encode(reg)
    b.begin_session()
    return b


def run(refresh: bool, dropped: frozenset = frozenset()):
    """The session loop's moves for a kv:2 session at every pass: a step before each pass but the session's first,
    its pages only while the window has a fresh slot left, the plan, the step's appends sent, each in its slot, and
    each re-key encoded under its coin, then the pass and every opening.  A pass in `dropped`, (t, i), is dropped at
    its window's boundary after its step: it sends nothing and runs nothing."""
    b = backend()
    store, coins, joined = b.store, {}, {}
    r = audit_prover.Refresh(store, Schedule(N, LIFETIME, WINDOWS, FRESH)) if refresh else None
    trace = prover.ServingTrace(b, store.registration, PAGES, 8, random.Random(9), LIFETIME, 0, r)
    sent, n, out = 0, 0, []
    for t in range(WINDOWS):
        for i in range(PASSES):
            if n > 0:
                trace.step(n - 1, t, r is None or r.fresh_left(t))
            plan = trace.plan(t) if refresh else trace.plan()
            if (t, i) in dropped:
                n += 1
                continue
            if refresh:
                trace.appends[sent:] = [r.sent(a, t) for a in trace.appends[sent:]]
            for a in trace.appends[sent:]:
                coins[a.first] = hashlib.sha512(b"coin %d" % a.first).digest()[:codec.COIN_BYTES]
                joined[a.first] = t
                if not refresh or a.origin is not None:
                    b.encode_append(a.first, coins[a.first])
            sent = len(trace.appends)
            b.begin_window(bytes(32), bytes(64))
            root, handle = b.run_pass(hashlib.sha512(b"seed %d %d" % (t, i)).digest(), plan)
            assert b.misreads == ()
            out.append((t, plan, b.open(handle, list(range(plan.length)))))
            n += 1
    return b, trace, coins, joined, out


def append_of(store: audit_prover.Store, j: int) -> audit_prover.Append:
    return next(a for a in store.appends if a.first <= j < a.first + a.count)


def test_under_refresh_every_plan_and_opening_keeps_the_refresh_rules():
    """A registered block is held only while t < L, an appended one only under a key that joined fewer than L windows
    before t and after its payload's probation; each opening decodes under its block's key and its append's coin, and
    its payload's leaf sits at its origin's position under the origin's root."""
    b, trace, coins, joined, out = run(True)
    store, reg = b.store, b.store.registration
    for t, plan, fields in out:
        for q, j in enumerate(plan):
            block, _, _, _, salt, path, d, at = fields[q]
            if j < N:
                assert t < LIFETIME and at is None
                payload, at = codec.decode(block, j, reg.key), j
            else:
                a = append_of(store, j)
                assert a.origin is not None and joined[a.first] <= t < joined[a.first] + LIFETIME
                born = None if a.origin[0] == "registered" else joined[store.in_slot(a.origin[1]).first]
                assert born is None or born + LIFETIME <= t
                payload = codec.decode(block, j, reg.key, d, coins[a.first])
                assert at == store.origin_position(j)
            assert payload is not None
            if at < N:
                domain, root = audit.payload_domain(NODE, N), Commitment(reg.payload_root, N)
            else:
                o = append_of(store, at)
                domain, root = audit.append_domain(reg.digest, o.first, o.count), Commitment(o.root, o.count)
            leaf = MerkleOpening(at, rb.hidden(payload, salt), path)
            assert verify_opening(domain, root, leaf, audit.PAYLOAD_SCHEMA, rb.LIMITS)
    rekeys = [a for a in store.appends if a.origin is not None]
    assert {a.origin[0] for a in rekeys} == {"registered", "append"}
    assert all(a.root == audit.rekey_root(reg.digest, a.first, a.count, a.origin) for a in rekeys)
    schedule = trace.refresh.schedule
    assert all(schedule.slot_at(a.slot)[0] == joined[a.first] for a in store.appends)
    assert [a.slot for a in store.appends] == sorted({a.slot for a in store.appends})
    fresh = [schedule.slot_at(a.slot)[0] for a in store.appends if a.origin is None]
    assert max(fresh.count(t) for t in fresh) == FRESH and all(a.count <= 512 for a in store.appends if a.origin is None)
    assert all(store.blocks[j] is None for a in store.appends if a.origin is None for j in range(a.first, a.first + a.count))
    assert any(j >= N for _, plan, _ in out for j in plan)


def test_under_refresh_the_memory_map_records_each_rekey_once_in_its_chunk_s_slot():
    """A re-key is of its origin whole: the block of a chunk still kept is recorded re-keyed with its origin, and the
    block of one serving has since freed or overwritten is withdrawn, never planned."""
    b, trace, *_ = run(True)
    store = b.store
    blocks = [(j, store.origin_position(j)) for a in store.appends if a.origin is not None
              for j in range(a.first, a.first + a.count)]
    withdrawn = set(trace.withdrawn)
    for j, origin in blocks:
        assert (j in withdrawn) != (trace.replay.origin.get(j) == origin), j
    assert withdrawn and trace.rekeyed == len(blocks) - len(withdrawn)


def test_pages_of_passes_dropped_at_a_boundary_go_out_in_the_next_window_whose_rekey_slots_stay_empty():
    """Pages made at a step whose pass is dropped at window 2's boundary precede window 3's re-keys in index order but
    must follow them in slot order, so window 3 sends no re-key and the pages take its fresh slots."""
    b, trace, _, joined, out = run(True, frozenset({(2, PASSES - 2), (2, PASSES - 1)}))
    schedule = trace.refresh.schedule
    windows = [(schedule.slot_at(a.slot)[0], a.origin) for a in b.store.appends]
    assert trace.skipped == 1 and [o for w, o in windows if w == 3] == [None, None]
    assert all(schedule.slot_at(a.slot)[0] == joined[a.first] for a in b.store.appends)
    assert [a.slot for a in b.store.appends] == sorted({a.slot for a in b.store.appends})
    assert {w for w, o in windows if o is not None} == set(range(1, WINDOWS)) - {3}
    assert {t for t, _, _ in out} == set(range(WINDOWS))


def test_with_refresh_off_the_trace_rekeys_as_before_and_sends_no_origin():
    """Off, every append has a tree and no origin, the memory map records no re-key, and each pass plans every live
    block."""
    b, trace, _, _, out = run(False)
    assert all(a.origin is None and a.tree is not None for a in b.store.appends)
    assert not trace.replay.origin and all(len(f) == 8 and f[7] is None for _, _, fields in out for f in fields)
    assert out[-1][1] == rb.Plan(prover.runs_of(sorted(trace.live)), trace.replay.extent)


def test_a_refresh_session_is_refused_outside_an_honest_or_kv_session_over_a_backend_that_rekeys():
    p = prover.ProverSide.__new__(prover.ProverSide)
    p.backend, p.ahead = backend(), None
    with pytest.raises(ValueError, match="re-keys by appends"):
        prover.ProverSide.session(p, {"k": 0, "windows": 2, "lifetime": 2, "refresh": True}, "skip", 0.0)


def test_an_origin_is_named_on_the_wire_as_the_append_message_carries_it_and_read_back():
    for origin, named in ((("registered", 4), {"registered": 4}), (("append", 3), {"append": 3})):
        assert wire.origin_json(origin) == named and wire.origin_of(named) == origin
    assert wire.origin_of(None) is None
    for bad in ({"append": 3, "at": 0}, {"registered": 1, "at": 0}, [0, 1], "registered"):
        with pytest.raises(ValueError, match="an append's origin"):
            wire.origin_of(bad)
