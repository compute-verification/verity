"""Red-team cases against the v3 I/O tie (`io_tie`, cursor/io-tie-061a) from the advice red team, added beside the
author's `test_network_certifier_io_tie{,_adversarial}.py`.

The author's cases already refuse, against the honest table, every single-sided tamper the lead named -- a duplicated,
dropped, reordered or wrong-stream cell, a cell on a stream with no request, a wrong id, an output moved to the next
bucket -- and show the in-bucket order carries nothing (of 5! orders of a bucket exactly one is accepted). These two
cases add what those don't reach:

- The reference ties the cells to the *table*, not to the certified outputs. Every author tamper changes the cells
  alone (against the honest table) or the table alone (against the honest cells); none forges both consistently. A
  prover controls its own table, so for any cells it emits there is a table that makes `reference` 1. The honest-table
  refusals are therefore load-bearing only because the table is pinned elsewhere to `window_slots(certified outputs,
  certified ids)`. `io_tie` does not bind the table to the certified I/O; the integrated certifier must, and this
  prototype on a fake upstream does not yet.
- The in-bucket order channel ADV-SPEC flagged (log2 n_t!): the residual freedom the canonical order leaves is cells
  of one stream in one bucket, and the id-to-slot check pins those whenever their ids differ. The only orders the map
  leaves free are byte-identical cells (a request's repeated token id), which commit to one value, so the channel is
  zero bits, not merely one canonical order among distinguishable ones."""

from __future__ import annotations

import itertools

from verity.core.protocols.certifier import io_tie as IT

from test_network_certifier_io_tie import BIG, WINDOW, commit, honest, serve, _cell_tampers


def _table_matching(shape: IT.Shape, base: int, cells) -> list[bytes]:
    """The table a prover builds from its own cells: one slot a data cell, its request its stream, its index the cell's
    order within that stream, its bucket and id the cell's. `reference` holds the cells to *this*, not to the certified
    outputs, so it is 1 for any cells the prover can parse."""
    jof: dict[int, int] = {}
    slots = []
    for i, c in enumerate(cells):
        p = IT.parse_cell(c)
        if p is not None and p.data:
            j = jof.get(p.stream, 0)
            jof[p.stream] = j + 1
            slots.append(IT.Slot(p.stream, j, base + shape.bucket(i), p.id))
    return IT.table_rows(sorted(slots), shape)


def test_the_reference_ties_cells_to_the_table_not_to_the_certified_outputs() -> None:
    """A wrong id is refused against the honest table (the author's case) but accepted once the table is forged to
    match, and a response from a request that never served is accepted with its matching table. So `io_tie.reference`
    is a cells-to-table consistency tie; its soundness rests entirely on the table being bound to the certified outputs
    (`window_slots`), which this prototype does not enforce -- the integrated certifier must, with the certified ids,
    not the prover's."""
    inst = honest(BIG, 11)
    assert IT.reference(inst)

    wrong = _cell_tampers(inst)["a wrong id"]
    assert not IT.reference(commit(BIG, wrong, inst.rows))
    assert IT.reference(commit(BIG, wrong, _table_matching(BIG, inst.base, wrong)))

    fabricated = [IT.pad_cell(BIG.B)] * BIG.N
    fabricated[0] = IT.data_cell(123_456, 424_242, BIG.B)
    fabricated[1] = IT.data_cell(222, 424_242, BIG.B)
    assert IT.reference(commit(BIG, fabricated, _table_matching(BIG, inst.base, fabricated)))


def test_the_only_free_in_bucket_order_is_byte_identical_cells() -> None:
    """ADV-SPEC's log2 n_t! channel: the canonical order (data first, streams non-decreasing) leaves one freedom,
    cells of one stream in one bucket. With distinct ids the id-to-slot check pins them (the author's case: exactly one
    order accepted). With a repeated id the map admits both orders, but the two cells are byte-identical, so they commit
    to one value: zero distinguishable strings, not one canonical among several."""
    # request 3 emits the same id twice in bucket 0; request 8 two distinct ids; request 20 one
    one = IT.Shape(BIG.B, 2, 8)
    cells, outputs, release, ids = serve(one, {3: (0, [10, 10]), 8: (4, [12, 13]), 20: (0, [14])}, 14, [5, 0])
    inst = commit(one, cells, IT.table_rows(IT.window_slots(outputs, release, ids, WINDOW, one.T), one))
    assert IT.reference(inst)
    assert [IT.parse_cell(c).stream for c in cells[:5]] == [3, 3, 8, 8, 20]

    accepted, distinct = 0, set()
    for order in itertools.permutations(range(5)):
        perm = [cells[k] for k in order] + list(inst.cells[5:])
        if IT.canonical_map(one, inst.base, perm, inst.rows) is not None:
            accepted += 1
            distinct.add(tuple(perm))
    # the two cells of stream 3 (same id) swap freely; every other swap is refused by the id-to-slot check
    assert accepted == 2 and len(distinct) == 1
