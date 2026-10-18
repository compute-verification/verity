"""memory-map/v1 two ways on one session: the circuit's (`commit.memory_map.record`, the oracle) and the `Store`'s writer
(`store.memory_map`, the prover's map), each request's rows at its pages and its frees after its last step, on the store test's
pool (2 layers of 4 pages, 8 slots of 1 KiB a plane: a plane-page is two 4 KiB blocks; N = 37).  A request prefills its first
rows at its arrival step and decodes one row a step after, and its last step reads every row (`test_memory_map.decode`).  The
two maps agree line for line (steps, j, d_j and slots) on a session with pages out of order, blocks completed over several
steps, last blocks left part written, a block completed at its last read (in no record) and a page reused after its request,
whose untouched blocks stay held until a write first reaches them (memory accounting's rules, 18 Oct).  They agree as well
under deferred residency (#2461), the circuit's map taking the session's encodes (`commit.memory_map.Deferral`): appends landing
a step late as the GPU sessions' do (serve, one two steps late at a window's boundary and the last still out at the end;
servedrop, every one a step late), late enough that two in one line write one slot, and re-keyed under lifetime cap 2."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby

import pytest

from tests.commit.test_memory_map import _payload, decode
from tests.store.test_memory_map import SLOTS, store
from verity.core.protocols.remote_bandwidth import audit
from verity.core.protocols.remote_bandwidth.memory_map import Header, Step, parse
from verity_vllm.commit import memory_map as MM
from verity_vllm.store.interface import Append

REGISTRATION = bytes(range(64))
#: the session's first step, the engine's step 0: a binder on an engine that served before numbers on from there
FIRST = 64


@dataclass(frozen=True)
class Request:
    """A request from step `arrive`: `prefill` rows at that step, then one a step to `positions`, at `pages`."""

    arrive: int
    prefill: int
    positions: int
    pages: tuple[int, ...]


def circuit(reqs: dict[str, Request], deferral: MM.Deferral | None = None) -> tuple[Header, list[Step]]:
    s, _ = store()
    layout = MM.Layout(block_size=SLOTS, pages=s.pool.layout.pages, firsts=tuple(s.region(rn).first for rn in s.pool.regions),
                       blocks=s.layout().blocks)
    return MM.record({r: decode(x.prefill, x.positions) for r, x in reqs.items()}, {r: x.arrive for r, x in reqs.items()},
                     _payload, REGISTRATION, layout, {r: list(x.pages) for r, x in reqs.items()}, deferral)


def stored(reqs: dict[str, Request], late: Callable[[int], int | None] | None = None, windows: tuple[int, ...] = (),
           lifetime: int = audit.LIFETIME) -> tuple[Header, list[Step], MM.Deferral | None]:
    """The Store's map as the binder drives it from step `FIRST`, on the engine's steps: a step allocates the pages its writes
    first reach, writes, then frees the pages of the requests it ends.  With `late`, in a session (`store.remote.Serving`): once
    a step is done its completed blocks go out as one append (`MemoryMap.complete`), each append made at the step whose events
    it is in, and its encode lands `late(step made)` steps after (None: not by the end), in append order; a window opens at each
    step of `windows`.  The session's deferral beside it."""
    s, _ = store()
    sent: list[tuple[int, int]] = []
    m = s.record_memory_map(registration=REGISTRATION, lifetime=lifetime, on_append=None if late is None else
                            lambda first, digests, payloads: sent.append((first, len(digests))))
    rows = {r: decode(x.prefill, x.positions) for r, x in reqs.items()}
    held: dict[int, tuple[str, int]] = {}
    encodes: list[list] = []
    due: list[int | None] = []
    for t in range(max(x.arrive + rows[r].steps for r, x in reqs.items())):
        if t in windows:
            if s.window_open:
                s.end_window()
            s.begin_window(bytes(32), None if s.window >= 0 else REGISTRATION)
        appends, new = [], []
        for r, x in reqs.items():
            born = (q for q, row in enumerate(rows[r].layers[0][0]) if x.arrive + row.born == t)
            for k, qs in groupby(born, key=lambda q: q // SLOTS):
                qs = list(qs)
                if x.pages[k] not in held:
                    new.append(x.pages[k])
                    held[x.pages[k]] = (r, k)
                appends.append(Append(x.pages[k], qs[0] % SLOTS, len(qs)))
        if new:
            s.kv_alloc(new)
        s.kv_write_reference(appends, lambda layer, page, slot: b"".join(
            _payload(held[page][0], rows[held[page][0]].layers[layer][plane][held[page][1] * SLOTS + slot]) for plane in range(2)),
            FIRST + t)
        if done := [p for p, (r, _) in held.items() if reqs[r].arrive + rows[r].steps - 1 == t]:
            s.kv_free(done)
            for p in done:
                del held[p]
        if late is None:
            continue
        m.complete()
        for first, count in sent:
            n = late(t)
            due.append(None if n is None or (due and due[-1] is None) else max(t + n, due[-1] if due else 0))
            encodes.append([FIRST + t, None, first, count])
        sent.clear()
        for e, d in zip(encodes, due):
            if e[1] is None and d is not None and d <= t:
                m.encoded(e[2])
                e[1] = FIRST + t
    m.flush()
    head, steps = parse(m.text().splitlines())
    deferral = None if late is None else MM.Deferral(tuple(map(tuple, encodes)), FIRST, lifetime)
    return head, [Step(x.step - FIRST, x.freed, x.written) for x in steps], deferral


def agree(reqs: dict[str, Request], late: Callable[[int], int | None] | None = None, **kw) -> tuple[list[Step], MM.Deferral | None]:
    """The two maps over the session's steps (the GPU run's ``compare.py``'s): a block the session's last line writes after
    its last read is freed in the circuit's next line, a step the session never opens."""
    head, steps, deferral = stored(reqs, late, **kw)
    n = max(x.arrive + decode(x.prefill, x.positions).steps for x in reqs.values())
    got_head, got = circuit(reqs, deferral)
    assert (head, steps) == (got_head, [s for s in got if s.step < n])
    return steps, deferral


def _half(page: int, k: int) -> list[int]:
    """The slots of `page`'s k-th block of each plane (its slots 4k to 4k + 3), in both layers."""
    s, _ = store()
    return sorted(s.region(rn).first + int(s.pool.ids[page, c]) for rn in s.pool.regions for c in (k, 2 + k))


#: r0 at pages 2 then 1, its blocks completed over steps 0-7 and its last left part written; r1's last block part written,
#: its page 3 reused by r2 from step 4
SESSION = {"r0": Request(0, 5, 13, (2, 1)), "r1": Request(1, 6, 7, (3,)), "r2": Request(4, 3, 6, (3,))}
#: r1's last step, 3, completes its page's second blocks (position 7)
COMPLETED_AT_LAST_READ = SESSION | {"r1": Request(1, 6, 8, (3,))}
#: r1 never reaches its page's second blocks (positions 4-7), which r2 first reaches at step 6
UNTOUCHED = SESSION | {"r1": Request(1, 2, 3, (3,))}
#: the GPU sessions' deferral (r20261018-070848-81e2): serve's encodes a step late, the one made at step 3 two (its pass
#: discarded at a window's boundary), the last still out at the end; servedrop's every one a step late
SERVE = {"serve": lambda t: {3: 2, 7: None}.get(t, 1), "servedrop": lambda t: 1}


def test_the_circuits_map_is_the_stores():
    agree(SESSION)


@pytest.mark.parametrize("late", SERVE.values(), ids=SERVE.keys())
@pytest.mark.parametrize("reqs", [SESSION, COMPLETED_AT_LAST_READ, UNTOUCHED], ids=["session", "completed_at_last_read", "untouched"])
def test_the_circuits_map_is_the_stores_under_deferred_residency(reqs, late):
    steps, deferral = agree(reqs, late)
    assert steps != circuit(reqs)[1] and {m - r for m, r, _, _ in deferral.encodes if r is not None} <= {-1, -2}


def test_a_block_freed_before_its_encode_lands_is_written_there_and_freed_a_line_later():
    # r1's first blocks, completed at step 1 and read last at 2, land at 2: written there, freed at 3
    steps, _ = agree(SESSION, SERVE["servedrop"])
    by = {st.step: st for st in steps}
    assert [w.slot for w in by[2].written] == list(by[3].freed) == _half(3, 0)


def test_two_appends_landing_in_one_line_that_write_one_slot_are_written_a_line_apart():
    # r1's first blocks (step 1) land four steps late, at 5, with every append after them, among them step 5's, in which r2
    # completes page 3's first blocks again: that one waits for step 6's line, with the rest
    steps, deferral = agree(SESSION, lambda t: {1: 4, 5: 0}.get(t, 1))
    by = {st.step: {w.index for w in st.written} for st in steps}
    (made5,) = [(f, c) for m, _, f, c in deferral.encodes if m == FIRST + 5]
    assert [r for m, r, _, _ in deferral.encodes if m == FIRST + 5] == [FIRST + 5] and set(range(made5[0], sum(made5))) <= by[6]


def test_a_block_no_step_reads_while_resident_is_in_no_record():
    head, steps = circuit(COMPLETED_AT_LAST_READ)
    agree(COMPLETED_AT_LAST_READ)
    # never written; their filler freed where r1's prefill first reaches them
    second = set(_half(3, 1))
    assert not second & {w.slot for st in steps for w in st.written}
    assert [st.step for st in steps if second & set(st.freed)] == [1]


def test_a_freed_pages_untouched_filler_is_held_until_a_write_reaches_it():
    head, steps = circuit(UNTOUCHED)
    agree(UNTOUCHED)
    # not at step 2, where r1 ends: its end frees only the blocks it wrote
    second = set(_half(3, 1))
    assert [(st.step, sorted(second & set(st.freed))) for st in steps if second & set(st.freed)] == [(6, sorted(second))]


def test_a_re_key_frees_the_old_index_at_once_and_writes_the_new_one_where_its_encode_lands():
    # windows of 3 steps under lifetime cap 2: the line closing as window 2 opens (step 5's) frees r0's first blocks of page
    # 2, written in window 0, and the re-key's append, made at step 6, writes them again where its encode lands, at 7
    steps, deferral = agree(SESSION, SERVE["servedrop"], windows=(0, 3, 6, 9, 12), lifetime=2)
    by = {st.step: st for st in steps}
    assert [e for e in deferral.encodes if e[0] == FIRST + 6] == [(FIRST + 6, FIRST + 7, 53, 4)]
    assert list(by[5].freed) == [w.slot for w in by[1].written] == [w.slot for w in by[7].written] == _half(2, 0)
    assert [w.index for w in by[7].written] == [53, 54, 55, 56] and [w.digest for w in by[7].written] == [w.digest for w in by[1].written]
