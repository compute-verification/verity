"""Adversarial cases against the I/O tie (`io_tie`), kept as regressions of the fix that closed them, beside
`test_network_certifier_io_tie{,_adversarial}.py`.

v3 tied the cells to a table the prover supplied, and every author tamper was single-sided: the cells changed against
the honest table, or the table against the honest cells. A prover who forged the cells and a matching table passed, so
a wrong id and a response from a request that never served were both accepted. The tie now has no table: the verifier
computes the window's map from the advice, and each id is an entry of the Commit's rows, opened under the root the
Commit registered. These cases forge everything the prover controls together and are refused:

- A wrong id, with a map and a port of the prover's own that match it, or with the Commit's rows carrying it: the
  prover's map isn't the verifier's, and its rows open only under their own tree's root.
- A fabricated response, with a matching map and port, or with port rows of its ids: the verifier's map has no output
  there, whatever the port says.
- The in-bucket order channel (log2 n_t!): the map fixes each slot's stream and kind and each data
  cell's entry, so the only orders left free are byte-identical cells (a connection's pads, or a response's repeated
  token id in one bucket), which commit to one value: zero bits."""

from __future__ import annotations

import itertools
from dataclasses import replace

from verity.core.protocols.certifier import io_tie as IT

from test_network_certifier_io_tie import BIG, _cell_tampers, busiest, busy, commit, serve, wardens, with_cells


def _witness_matching(inst: IT.Instance, cells) -> IT.Instance:
    """v3's matching table carried over: a map and a port the prover builds from its own cells, each slot mapped to its
    cell's stream and kind and each data cell to the next entry of its bucket's port, whose rows (steps t·S … t·S +
    S − 1 of a tree of the prover's own) carry the bucket's ids. `reference` is 1 on it for any cells the prover can
    parse; the verifier's map and the Commit's root refuse it."""
    sh = inst.shape
    cmap, rows = [], [bytes(4 * sh.P)] * (1 << sh.DS)
    for t in range(sh.T):
        ids = []
        for c in cells[inst.bucket(t)]:
            p = IT.parse_cell(c)
            cmap.append(IT.MapEntry(p.stream, len(ids) if p.data else IT.END_ENTRY if p.kind == IT.END else IT.EMPTY))
            ids += [p.id] if p.data else []
        for s in range(sh.S):
            rows[t * sh.S + s] = IT.port_row(ids[s * sh.P:(s + 1) * sh.P], sh.P)
    salts = [IT.fake_salt(b"prover", s) for s in range(len(rows))]
    levels = IT.rows_levels(inst.port_domain, rows, salts)
    steps = tuple(tuple(range(t * sh.S, (t + 1) * sh.S)) for t in range(sh.T))
    opened = [IT.open_port(levels, rows, salts, st) for st in steps]
    return replace(with_cells(inst, cells), map=tuple(cmap), steps=steps, port=tuple(o[0] for o in opened),
                   port_salts=tuple(o[1] for o in opened), paths=tuple(o[2] for o in opened), port_root=levels[-1][0])


def _verifiers(forged: IT.Instance, inst: IT.Instance) -> IT.Instance:
    """The prover's cells and openings under the window's public values: the verifier's map and steps and the Commit's
    root."""
    return replace(forged, map=inst.map, steps=inst.steps, port_root=inst.port_root)


def test_a_wrong_id_with_a_witness_forged_to_match_is_refused() -> None:
    """v3 accepted a wrong id once the table was forged to match. Its analogue now, the prover's own map and port, passes
    only against itself; and the prover's best port under the verifier's map, the Commit's rows with the wrong id in
    its entry, opens only under its own tree's root, not the Commit's."""
    link = busy(11)
    w = busiest(link)
    inst = commit(BIG, link, w)
    assert IT.reference(inst)

    wrong = _cell_tampers(inst)["a wrong id"]
    assert not IT.reference(with_cells(inst, wrong))
    forged = _witness_matching(inst, wrong)
    assert IT.reference(forged)
    assert not IT.reference(_verifiers(forged, inst))

    (i,) = [k for k, (a, b) in enumerate(zip(wrong, inst.cells)) if a != b]
    t, e = BIG.bucket(i), inst.map[i].e
    step, r = inst.steps[t][e // BIG.P], e % BIG.P
    rows = list(link.port)
    rows[step] = rows[step][:4 * r] + IT.parse_cell(wrong[i]).id.to_bytes(4, "big") + rows[step][4 * r + 4:]
    spliced = with_cells(commit(BIG, link._replace(port=rows), w), wrong)
    assert IT.reference_parts(spliced) == {"cells_root": True, "port": True, "content": True}
    assert spliced.port_root != inst.port_root
    assert IT.reference_parts(replace(spliced, port_root=inst.port_root)) == {
        "cells_root": True, "port": False, "content": True}


def test_a_fabricated_response_with_a_witness_forged_to_match_is_refused() -> None:
    """v3 accepted cells from a request that never served with their matching table. The verifier's map has pads where
    nothing was released, so they are refused under it with the prover's own port, with port rows of their ids, or with
    the Commit's."""
    link = busy(11)
    w = busiest(link)
    inst = commit(BIG, link, w)
    fabricated = [IT.pad_cell(BIG.B)] * BIG.N
    fabricated[0] = IT.data_cell(123_456, 424_242, BIG.B)
    fabricated[1] = IT.data_cell(222, 424_242, BIG.B)
    forged = _witness_matching(inst, fabricated)
    assert IT.reference(forged)
    assert not IT.reference(_verifiers(forged, inst))
    assert IT.reference_parts(with_cells(inst, fabricated))["content"] is False

    pads = [k for k, m in enumerate(inst.map) if m.e == IT.EMPTY]
    beside = list(inst.cells)
    for k, stream in zip(pads[:2], (424_242, next(m.q for m in inst.map if m.q))):
        beside[k] = IT.data_cell(123_456, stream, BIG.B)
    rows = [IT.port_row([123_456] * BIG.P, BIG.P)] * len(link.port)
    for name, s in {"beside the honest cells": link, "and a port of their ids": link._replace(port=rows)}.items():
        assert IT.reference_parts(with_cells(commit(BIG, s, w), beside))["content"] is False, name


def test_the_only_free_in_bucket_order_is_byte_identical_cells() -> None:
    """The log2 n_t! channel: the map fixes each slot's stream and kind and each data cell's entry, so a cell's
    place names its output. With r = 2, connection 1 emits the same id twice in bucket 1 and connection 2 two distinct
    ids: the two cells of connection 1 swap freely, but they are byte-identical, so they commit to one value; so do
    each connection's two pads in bucket 0. Every other order is refused."""
    one = IT.Shape(BIG.B, 2, 4, P=4, S=4, DS=4)
    link = serve(one, wardens(one, 2, 1), [0, 0], [[10, 10], [12, 13]], 2 * one.T)
    inst = commit(one, link, 0)
    assert IT.reference(inst)
    assert [(p.stream, p.id) for p in map(IT.parse_cell, inst.cells[one.R:2 * one.R])] == [(1, 10), (1, 10), (2, 12),
                                                                                         (2, 13)]
    for t, want in ((0, 4), (1, 2)):
        honest = inst.cells[inst.bucket(t)]
        accepted = [order for order in itertools.permutations(range(one.R)) if IT.cells_match(
            one, [honest[k] for k in order], inst.map[inst.bucket(t)], inst.port[t])]
        assert len(accepted) == want and {tuple(honest[k] for k in order) for order in accepted} == {honest}, t
