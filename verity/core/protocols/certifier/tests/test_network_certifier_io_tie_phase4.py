"""Red-team cases against the I/O tie (`io_tie`) from the advice red team, kept as regressions of the fix that closed
them, beside the author's `test_network_certifier_io_tie{,_adversarial}.py`.

v3 tied the cells to a table the prover supplied, and every author tamper was single-sided: the cells changed against
the honest table, or the table against the honest cells. A prover who forged the cells and a matching table passed, so
a wrong id and a response from a request that never served were both accepted. The tie now has no table: the verifier
computes the window's map from the advice, and each id is an entry of the Commit's rows, opened under the root the
Commit registered. These cases forge everything the prover controls together and are refused:

- A wrong id, with a map and a port of the prover's own that match it, or with the Commit's rows carrying it: the
  prover's map isn't the verifier's, and its rows open only under their own tree's root.
- A fabricated response, with a matching map and port, or with port rows of its ids: the verifier's map has no output
  there, whatever the port says.
- The in-bucket order channel ADV-SPEC flagged (log2 n_t!): the map fixes each cell's place, so the only orders left
  free are byte-identical cells (a request's repeated token id), which commit to one value: zero bits."""

from __future__ import annotations

import itertools
from dataclasses import replace

from verity.core.protocols.certifier import io_tie as IT

from test_network_certifier_io_tie import BIG, _cell_tampers, commit, serve, served_honestly, with_cells


def _witness_matching(inst: IT.Instance, cells) -> IT.Instance:
    """v3's matching table carried over: a map and a port the prover builds from its own cells, each data cell mapped
    in order to its stream and the next entry of a port whose rows carry the cells' ids, under the prover's own tree.
    `reference` is 1 on it for any cells the prover can parse; the verifier's map and the Commit's root refuse it."""
    sh = inst.shape
    cmap, ids = [], []
    for c in cells:
        p = IT.parse_cell(c)
        cmap.append(IT.MapEntry(p.stream, len(ids)) if p is not None and p.data else None)
        if cmap[-1] is not None:
            ids.append(p.id)
    rows = [IT.port_row(ids[s * sh.P:(s + 1) * sh.P], sh.P) for s in range(1 << sh.DS)]
    salts = [IT.fake_salt(b"prover", s) for s in range(len(rows))]
    levels = IT.rows_levels(inst.port_domain, rows, salts)
    steps = tuple(range(sh.S))
    port, port_salts, paths = IT.open_port(levels, rows, salts, steps)
    return replace(with_cells(inst, cells), map=tuple(cmap), steps=steps, port=port, port_salts=port_salts, paths=paths,
                   port_root=levels[-1][0])


def _verifiers(forged: IT.Instance, inst: IT.Instance) -> IT.Instance:
    """The prover's cells and openings under the window's public values: the verifier's map and steps and the Commit's
    root."""
    return replace(forged, map=inst.map, steps=inst.steps, port_root=inst.port_root)


def test_a_wrong_id_with_a_witness_forged_to_match_is_refused() -> None:
    """v3 accepted a wrong id once the table was forged to match. Its analogue now, the prover's own map and port, passes
    only against itself; and the prover's best port under the verifier's map, the Commit's rows with the wrong id in
    its entry, opens only under its own tree's root, not the Commit's."""
    served = served_honestly(BIG, 11)
    inst = commit(BIG, served)
    assert IT.reference(inst)

    wrong = _cell_tampers(inst)["a wrong id"]
    assert not IT.reference(with_cells(inst, wrong))
    forged = _witness_matching(inst, wrong)
    assert IT.reference(forged)
    assert not IT.reference(_verifiers(forged, inst))

    (i,) = [k for k, (a, b) in enumerate(zip(wrong, inst.cells)) if a != b]
    e = inst.map[i].e
    step, r = inst.steps[e // BIG.P], e % BIG.P
    rows = list(served.rows)
    rows[step] = rows[step][:4 * r] + IT.parse_cell(wrong[i]).id.to_bytes(4, "big") + rows[step][4 * r + 4:]
    spliced = commit(BIG, served._replace(cells=wrong, rows=rows))
    assert IT.reference_parts(spliced) == {"cells_root": True, "port": True, "content": True}
    assert spliced.port_root != inst.port_root
    assert IT.reference_parts(replace(spliced, port_root=inst.port_root)) == {
        "cells_root": True, "port": False, "content": True}


def test_a_fabricated_response_with_a_witness_forged_to_match_is_refused() -> None:
    """v3 accepted cells from a request that never served with their matching table. The verifier's map has pads where
    nothing was released, so they are refused under it with the prover's own port, with port rows of their ids, or with
    the Commit's."""
    served = served_honestly(BIG, 11)
    inst = commit(BIG, served)
    fabricated = [IT.pad_cell(BIG.B)] * BIG.N
    fabricated[0] = IT.data_cell(123_456, 424_242, BIG.B)
    fabricated[1] = IT.data_cell(222, 424_242, BIG.B)
    forged = _witness_matching(inst, fabricated)
    assert IT.reference(forged)
    assert not IT.reference(_verifiers(forged, inst))
    assert IT.reference_parts(with_cells(inst, fabricated))["content"] is False

    pads = [k for k, m in enumerate(inst.map) if m is None]
    beside = list(served.cells)
    for k, stream in zip(pads[:2], (424_242, next(m.q for m in inst.map if m is not None))):
        beside[k] = IT.data_cell(123_456, stream, BIG.B)
    rows = [IT.port_row([123_456] * BIG.P, BIG.P)] * len(served.rows)
    for name, s in {"beside the honest cells": served._replace(cells=beside),
                    "and a port of their ids": served._replace(cells=beside, rows=rows)}.items():
        assert IT.reference_parts(commit(BIG, s))["content"] is False, name


def test_the_only_free_in_bucket_order_is_byte_identical_cells() -> None:
    """ADV-SPEC's log2 n_t! channel: the map puts bucket t's outputs in (request, index) order, so a cell's place names
    its output. With distinct ids every other order is refused (the author's case: exactly one order accepted). With a
    repeated id the two cells swap freely, but they are byte-identical, so they commit to one value: zero
    distinguishable strings, not one canonical among several."""
    # request 3 emits the same id twice in bucket 0; request 8 two distinct ids; request 20 one
    one = IT.Shape(BIG.B, 2, 8, P=8, S=8, DS=4)
    served = serve(one, {3: (0, [10, 10]), 8: (4, [12, 13]), 20: (0, [14])}, 14, [5, 0])
    inst = commit(one, served)
    assert IT.reference(inst)
    assert [IT.parse_cell(c).stream for c in served.cells[:5]] == [3, 3, 8, 8, 20]

    accepted, distinct = 0, set()
    for order in itertools.permutations(range(5)):
        cells = [served.cells[k] for k in order] + list(inst.cells[5:])
        if IT.cells_match(one, cells, inst.map, inst.port):
            accepted += 1
            distinct.add(tuple(cells))
    # the two cells of stream 3 (same id) swap freely; every other swap is refused by the map
    assert accepted == 2 and len(distinct) == 1
