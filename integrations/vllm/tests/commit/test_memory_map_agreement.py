"""memory-map/v1 two ways on one session: the circuit's (`commit.memory_map.record`, the oracle) and the `Store`'s writer
(`store.memory_map`, the prover's map), each request's rows at its pages and its frees after its last step, on the store test's
pool (2 layers of 4 pages, 8 slots of 1 KiB a plane: a plane-page is two 4 KiB blocks; N = 37).  A request prefills its first
rows at its arrival step and decodes one row a step after, and its last step reads every row (`test_memory_map.decode`).  The
two maps agree line for line (steps, j, d_j and slots) on a session with pages out of order, blocks completed over several
steps, last blocks left part written, a block completed at its last read (in no record) and a page reused after its request,
whose untouched blocks stay held until a write first reaches them (memory accounting's rules, 18 Oct)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from tests.commit.test_memory_map import _payload, decode
from tests.store.test_memory_map import SLOTS, store
from verity.core.protocols.remote_bandwidth.memory_map import Header, Step, dumps
from verity_vllm.commit import memory_map as MM
from verity_vllm.store.interface import Append

REGISTRATION = bytes(range(64))


@dataclass(frozen=True)
class Request:
    """A request from step `arrive`: `prefill` rows at that step, then one a step to `positions`, at `pages`."""

    arrive: int
    prefill: int
    positions: int
    pages: tuple[int, ...]


def circuit(reqs: dict[str, Request]) -> tuple[Header, list[Step]]:
    s, _ = store()
    layout = MM.Layout(block_size=SLOTS, pages=s.pool.layout.pages, firsts=tuple(s.region(rn).first for rn in s.pool.regions),
                       blocks=s.layout().blocks)
    return MM.record({r: decode(x.prefill, x.positions) for r, x in reqs.items()}, {r: x.arrive for r, x in reqs.items()},
                     _payload, REGISTRATION, layout, {r: list(x.pages) for r, x in reqs.items()})


def stored(reqs: dict[str, Request]) -> str:
    """The Store's map as the binder drives it: a step allocates the pages its writes first reach, writes, then frees the
    pages of the requests it ends."""
    s, _ = store()
    m = s.record_memory_map(registration=REGISTRATION)
    rows = {r: decode(x.prefill, x.positions) for r, x in reqs.items()}
    held: dict[int, tuple[str, int]] = {}
    for t in range(max(x.arrive + rows[r].steps for r, x in reqs.items())):
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
            _payload(held[page][0], rows[held[page][0]].layers[layer][plane][held[page][1] * SLOTS + slot]) for plane in range(2)), t)
        if done := [p for p, (r, _) in held.items() if reqs[r].arrive + rows[r].steps - 1 == t]:
            s.kv_free(done)
            for p in done:
                del held[p]
    m.flush()
    return m.text()


def _second_blocks(page: int) -> set[int]:
    """The slots of `page`'s second block of each plane (its slots 4-7), in both layers."""
    s, _ = store()
    return {s.region(rn).first + int(s.pool.ids[page, c]) for rn in s.pool.regions for c in (1, 3)}


#: r0 at pages 2 then 1, its blocks completed over steps 0-7 and its last left part written; r1's last block part written,
#: its page 3 reused by r2 from step 4
SESSION = {"r0": Request(0, 5, 13, (2, 1)), "r1": Request(1, 6, 7, (3,)), "r2": Request(4, 3, 6, (3,))}
#: r1's last step, 3, completes its page's second blocks (position 7)
COMPLETED_AT_LAST_READ = SESSION | {"r1": Request(1, 6, 8, (3,))}
#: r1 never reaches its page's second blocks (positions 4-7), which r2 first reaches at step 6
UNTOUCHED = SESSION | {"r1": Request(1, 2, 3, (3,))}


def test_the_circuits_map_is_the_stores():
    assert stored(SESSION).splitlines() == dumps(*circuit(SESSION)).splitlines()


def test_a_block_no_step_reads_while_resident_is_in_no_record():
    head, steps = circuit(COMPLETED_AT_LAST_READ)
    assert stored(COMPLETED_AT_LAST_READ).splitlines() == dumps(head, steps).splitlines()
    # never written; their filler freed where r1's prefill first reaches them
    second = _second_blocks(3)
    assert not second & {w.slot for st in steps for w in st.written}
    assert [st.step for st in steps if second & set(st.freed)] == [1]


def test_a_freed_pages_untouched_filler_is_held_until_a_write_reaches_it():
    head, steps = circuit(UNTOUCHED)
    assert stored(UNTOUCHED).splitlines() == dumps(head, steps).splitlines()
    # not at step 2, where r1 ends: its end frees only the blocks it wrote
    second = _second_blocks(3)
    assert [(st.step, sorted(second & set(st.freed))) for st in steps if second & set(st.freed)] == [(6, sorted(second))]
