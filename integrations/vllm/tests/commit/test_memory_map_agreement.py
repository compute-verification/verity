"""memory-map/v1 two ways on one session: the circuit's (`commit.memory_map.record`, the oracle) and the `Store`'s writer
(`store.memory_map`, the prover's map), each request's rows at its pages and its frees after its last step, on the store test's
pool (2 layers of 4 pages, 8 slots of 1 KiB a plane: a plane-page is two 4 KiB blocks; N = 37).  A request prefills its first
rows at its arrival step and decodes one row a step after, and its last step reads every row.  The two maps agree line for line
(steps, j, d_j and slots) on a session with pages out of order, blocks completed over several steps, last blocks left part
written and a page reused after its request.  They differ where the conventions do, until those are settled: a block completed
at its request's last step, and the untouched blocks of a freed page."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import groupby

import pytest

from tests.store.test_memory_map import LAYERS, SLOT_BYTES, SLOTS, store
from verity.core.protocols.remote_bandwidth.memory_map import dumps
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

    @property
    def steps(self) -> int:
        return 1 + self.positions - self.prefill

    def born(self, q: int) -> int:
        return max(0, q - self.prefill + 1)


def _row(r: str, layer: int, plane: int, q: int) -> bytes:
    return hashlib.shake_256(repr((r, layer, plane, q)).encode()).digest(SLOT_BYTES)


def circuit(reqs: dict[str, Request]) -> str:
    s, _ = store()
    layout = MM.Layout(block_size=SLOTS, pages=s.pool.layout.pages, firsts=tuple(s.region(rn).first for rn in s.pool.regions),
                       blocks=s.layout().blocks)
    rows = {r: MM.Rows(tuple(tuple(tuple(MM.Row((("kv", layer, plane, q),), x.born(q), x.steps - 1) for q in range(x.positions))
                                   for plane in range(2)) for layer in range(LAYERS)), SLOT_BYTES, 2, x.steps)
            for r, x in reqs.items()}
    head, steps = MM.record(rows, {r: x.arrive for r, x in reqs.items()}, lambda r, row: _row(r, *row.group[0][1:]),
                            REGISTRATION, layout, {r: list(x.pages) for r, x in reqs.items()})
    return dumps(head, steps)


def stored(reqs: dict[str, Request]) -> str:
    """The Store's map as the binder drives it: a step allocates the pages its writes first reach, writes, then frees the
    pages of the requests it ends."""
    s, _ = store()
    m = s.record_memory_map(registration=REGISTRATION)
    held: dict[int, tuple[str, int]] = {}
    for t in range(max(x.arrive + x.steps for x in reqs.values())):
        appends, new = [], []
        for r, x in reqs.items():
            for k, qs in groupby((q for q in range(x.positions) if x.arrive + x.born(q) == t), key=lambda q: q // SLOTS):
                qs = list(qs)
                if x.pages[k] not in held:
                    new.append(x.pages[k])
                    held[x.pages[k]] = (r, k)
                appends.append(Append(x.pages[k], qs[0] % SLOTS, len(qs)))
        if new:
            s.kv_alloc(new)
        s.kv_write_reference(appends, lambda layer, page, slot: b"".join(
            _row(held[page][0], layer, plane, held[page][1] * SLOTS + slot) for plane in range(2)), t)
        if done := [p for p, (r, _) in held.items() if reqs[r].arrive + reqs[r].steps - 1 == t]:
            s.kv_free(done)
            for p in done:
                del held[p]
    m.flush()
    return m.text()


#: r0 at pages 2 then 1, its blocks completed over steps 0-7 and its last left part written; r1's last block part written,
#: its page 3 reused by r2 from step 4
SESSION = {"r0": Request(0, 5, 13, (2, 1)), "r1": Request(1, 6, 7, (3,)), "r2": Request(4, 3, 6, (3,))}


@pytest.mark.parametrize("reqs", [
    SESSION,
    pytest.param(SESSION | {"r1": Request(1, 6, 8, (3,))}, marks=pytest.mark.xfail(strict=True, reason=(
        "a block completed at its request's last step (r1's position 7): the circuit writes it then and frees it the step "
        "after (#2378); the Store's line writes and frees it together, so it drops the write and frees the block the slot "
        "held before"))),
    pytest.param(SESSION | {"r1": Request(1, 2, 3, (3,))}, marks=pytest.mark.xfail(strict=True, reason=(
        "the untouched blocks of a freed page (r1's positions 4-7): the Store frees their filler at kv_free, step 2; the "
        "circuit frees a slot's registered block only when a write reaches it, r2's at step 6"))),
], ids=["agree", "block-completed-at-last-step", "untouched-block-of-a-freed-page"])
def test_the_circuits_map_is_the_stores(reqs):
    assert stored(reqs).splitlines() == circuit(reqs).splitlines()
