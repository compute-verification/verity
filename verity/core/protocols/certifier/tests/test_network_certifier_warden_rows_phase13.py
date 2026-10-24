"""REDTEAM phase 13 against WARDEN's wiring binding (#3766, `warden_rows.Units`): I re-run my phase 11 pad-for-data
case against the binding and hunt the two gaps the lead named — a statement that names the Program's digest but opens a
different cell, and a `Units` built from anything but public structure. Both are closed, so this file is a regression,
not a finding.

The binding: the Program's units are a tuple of (connection, cell) int pairs (`Units`), their digest goes into the
port's value (`<link>/<direction>/<window>/units/<digest>`), and `refusal` — the host's read of Lean's registered
reads — ties a statement's header to that value and to the rows the verifier's own units place. Phase 11 left a
residual: which of a connection's own cells each unit read was the prover's, so a statement could open a pad where the
Program reads a data cell. Here, on the live multi-connection traffic (not a hand-built record), I show:

- a statement that keeps the Program's digest but moves a read to a pure-pad *row* is refused by position (the
  row-level form of the residual; the phase 11 regression covers the slot-level form, where the pad shares the data
  cell's row);
- the Program's digest cannot be made to open a different cell: swapping two of the Program's cells gives a different
  digest (refused by value), and forcing the honest value back yields the honest statement byte for byte, so the
  verifier reads the cell its own units place, whatever the prover meant;
- a `Units` carries only public positions: its encoding is the count and the (q, k) ints, with no cell bytes and no
  cell kind, so a population built to tell a data cell from a pad has no digest the verifier expects.
"""

from __future__ import annotations

import pytest

from verity.core.protocols.certifier import warden_rows as W
from verity.core.protocols.certifier.certificate import Direction
from verity.core.protocols.certifier.tests.signers import SESSION, signer
from verity.core.protocols.certifier.certificate import Issuer
from verity.core.protocols.certifier.tests.test_network_certifier_warden_rows import _traffic
from verity.core.protocols.certifier.tests.test_network_certifier_warden import PARAMS, records


def _cells(tmp_path):
    warden, _ = _traffic()
    issuer = Issuer(signer(tmp_path, b"certifier"), SESSION)
    cells_, _ = W.attest(issuer, records(warden)[1][0], PARAMS, Direction.EGRESS)
    return cells_


def test_phase13_the_right_digest_moved_to_a_pad_row_is_refused_by_position(tmp_path):
    """The row-level pad-for-data: the Program's units read connection q's data cells, and the port names their digest.
    A statement that keeps that digest but opens a pure-pad row (no connection's cell) in place of a data row is refused
    by position, naming the unit, the row it opened and the row the Program reads."""
    cells_ = _cells(tmp_path)
    head, wiring = cells_.head, cells_.wiring
    q, _, n = max(wiring.connections, key=lambda c: c[2])
    units = W.Units.span(q, 0, n)
    port = W.port(head, wiring, PARAMS, units)
    positions = port["reads"]
    pad_rows = [t for t, r in enumerate(cells_.rows) if r == bytes(len(r))]
    assert pad_rows and pad_rows[0] not in positions, "a pure-pad row the Program never reads"

    honest = cells_.statement(positions, units)[2]
    assert W.refusal(port, cells_.domain, honest, [cells_.bcs[t] for t in positions]) is None

    moved = positions[:]
    moved[0] = pad_rows[0]
    at_pad = dict(honest) | {"positions": moved, "paths": [cells_.path(p).hex() for p in moved]}
    why = W.refusal(port, cells_.domain, at_pad, [cells_.bcs[p] for p in moved])
    assert why == f"registered reads: unit 0 reads position {pad_rows[0]}, and the verifier's program reads " \
        f"{positions[0]}", why


def test_phase13_the_programs_digest_cannot_open_a_different_cell(tmp_path):
    """The named gap: a statement that names the right digest but opens a different cell. There is none. Swapping two of
    the Program's cells changes the digest, so the statement is refused by value; and a statement that copies the
    Program's value is the Program's statement byte for byte — the cell read is the slot the verifier's own units place,
    not the prover's."""
    cells_ = _cells(tmp_path)
    head, wiring = cells_.head, cells_.wiring
    q, _, n = max(wiring.connections, key=lambda c: c[2])
    assert n >= 2
    units = W.Units.span(q, 0, n)
    port = W.port(head, wiring, PARAMS, units)
    positions = port["reads"]
    bcs = [cells_.bcs[t] for t in positions]
    ours = cells_.statement(positions, units)[2]

    swapped = W.Units(((q, 1), (q, 0)) + units.cells[2:])
    assert W.port(head, wiring, PARAMS, swapped)["reads"] == positions, "a within-row swap keeps every row"
    made_for = cells_.statement(positions, swapped)[2]
    assert made_for["positions"] == positions
    assert "not another value" in W.refusal(port, cells_.domain, made_for, bcs)
    assert made_for | {"value": port["value"]} == ours, "forcing the honest value is the honest statement, byte for byte"
    assert W.refusal(port, cells_.domain, ours, bcs) is None


def test_phase13_units_carry_only_public_positions_no_bytes_or_kind(tmp_path):
    """A `Units` built from anything but public structure: there is no such channel. The encoding is the count and the
    (q, k) ints, with no cell bytes and no cell kind, so the digest the verifier expects is a function of the public
    Program and wiring alone; a population built to open other cells has a digest the verifier never names."""
    cells_ = _cells(tmp_path)
    head, wiring = cells_.head, cells_.wiring
    q, _, n = max(wiring.connections, key=lambda c: c[2])
    units = W.Units.span(q, 0, n)
    encoded = units.encode()
    assert encoded == n.to_bytes(4, "big") + b"".join(q.to_bytes(4, "big") + k.to_bytes(4, "big") for k in range(n))
    assert len(encoded) == 4 + 8 * n, "count and (q, k) ints only: no cell bytes, no data/pad kind"
    assert units.digest() == W.tagged_sha512(W.UNITS_TAG, encoded)
    assert W.Units.span(q, 0, n).cells == tuple((q, k) for k in range(n)), "span is a pure function of public (q, n)"

    port = W.port(head, wiring, PARAMS, units)
    other = W.Units(units.cells[::-1])
    assert other.digest() != units.digest() and W.port(head, wiring, PARAMS, other)["value"] != port["value"]
    bcs = [cells_.bcs[t] for t in port["reads"]]
    made_for = cells_.statement(W.port(head, wiring, PARAMS, other)["reads"], other)[2]
    assert "not another value" in W.refusal(port, cells_.domain, made_for, bcs)
