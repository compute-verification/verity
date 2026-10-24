"""REDTEAM phase 11 against the warden's bucket rows with reads tied to the layout (#3698, `warden_rows`): my phase 8
reads channel re-run, and the lead's question on the PAD cells.

Phase 8 found the reads (the units' wiring) unconstrained: a unit could open any of the N = T·S rows of the correct
link-window — a foreign connection's cell, a cell twice, or an empty/padding slot — an ≈ log2(N) bits a read channel.
#3698 closes the open vectors: the warden signs the wiring in its head, and `Wiring.place`/`port` confine every read to
a connection's own cells, refusing a foreign connection, a cell past the connection's, a read twice, an empty slot and
a wiring the head didn't sign. Here those are re-run against the v2 layout and shown refused.

The new shape the lead asks about: a PAD cell now carries its connection's ordinal (`cell(PAD, q, b"")`) and sits in
the connection's own slots, so it counts toward the connection's cells and `place` will open it. These cases show it is
not a new *content* channel — a pad's bytes are fixed by (PAD, q): header and zero fill, no prover-chosen bits — and that
carrying the ordinal *narrows* the read space to a connection's own cells. The residual freedom is the stated named gap,
not new: the units' wiring (which of a connection's own cells each unit reads) is opened to the verifier, not yet bound
to the Program, so a prover may read a pad where a data cell is intended or reorder reads within the connection, bounded
by log2(cells) bits a read (`warden_rows`'s docstring: "The wiring is opened to the verifier tonight, as the units'
wiring is (the named gap)"). Owner: WARDEN.
"""

from __future__ import annotations

import pytest

from verity.core.protocols.certifier import warden_rows as W
from verity.core.protocols.certifier.active import DATA, PAD
from verity.core.protocols.certifier.certificate import Direction
from verity.core.protocols.certifier.grid import COMPLETE, Record
from verity.core.protocols.certifier.warden import cell
from verity.core.protocols.certifier.tests.test_network_certifier_warden import PARAMS

B, R, LANES = PARAMS.cell_bytes, PARAMS.slots, PARAMS.lanes
S, T = PARAMS.grid.allocation, PARAMS.window
P = PARAMS.payload
LINK, SESSION = "responses", bytes(64)


def _record():
    """One clean link-window: connection 0 holds a DATA cell and then a PAD in its own two slots of bucket 0;
    connection 1 holds two DATA cells. Later buckets are idle (empty slots, B zero bytes)."""
    bucket0 = (cell(DATA, 0, bytes(P), B), cell(PAD, 0, b"", B), cell(DATA, 1, bytes(P), B), cell(DATA, 1, bytes(P), B))
    grid = (bucket0,) + ((None,) * S,) * (T - 1)
    return Record(LINK, 0, grid, COMPLETE)


def _cells():
    return W.Cells.of(LINK, Direction.EGRESS, _record(), PARAMS, SESSION)


def test_a_pad_carries_its_connections_ordinal_and_counts_toward_its_cells():
    """The wiring reads a pad's ordinal from its header, so connection 0's PAD is its own cell: connection 0 has two
    cells (one DATA, one PAD) in its two slots of bucket 0, and `place` opens the pad at its slot."""
    cells_ = _cells()
    assert cells_.wiring.connections == ((0, 0, 2), (1, 0, 2))  # the pad counts toward connection 0's two cells
    placed = cells_.wiring.place(PARAMS, [(0, 0), (0, 1)])
    assert placed == [(0, 0), (0, 1)]  # connection 0's data cell at slot 0, its pad at slot 1, same bucket row
    pad = cells_.rows[0][1 * B:2 * B]
    assert pad == cell(PAD, 0, b"", B) and pad[0] == PAD, "a pad's bytes are fixed by (PAD, ordinal): no prover bits"


def test_my_phase8_reads_are_refused_foreign_duplicate_cell_past_and_empty():
    """The open phase 8 vectors, re-run on the v2 layout: a foreign connection, a cell past the connection's (which
    is the only way an empty slot could be addressed), and a read twice are refused by `port`, and a wiring the head
    didn't sign is refused; so every read opens a connection's own cell and no empty, foreign or repeated row."""
    cells_ = _cells()
    head, wiring = cells_.head, cells_.wiring
    assert W.port(head, wiring, PARAMS, [(0, 0), (1, 1)])["reads"] == [0, 0]  # honest reads open the signed root
    absent = max(c[0] for c in wiring.connections) + 1
    for bad, why in [([(absent, 0)], "holds no slot"),          # a foreign connection
                     ([(0, 2)], "not cell"),                     # a cell past connection 0's two (no empty slot is addressable)
                     ([(0, -1)], "not cell"),
                     ([(0, 0), (0, 0)], "read twice"),           # the same cell twice
                     ([(0, 0.0)], "is \\(connection, cell\\)")]:
        with pytest.raises(ValueError, match=why):
            W.port(head, wiring, PARAMS, bad)
    for other in [W.Wiring(wiring.connections, bytes(64)), W.Wiring(wiring.connections[:-1], wiring.salt)]:
        with pytest.raises(ValueError, match="isn't the one the warden signed"):
            W.port(head, other, PARAMS, [(0, 0)])


def test_the_pad_ordinal_is_not_a_new_channel_but_the_units_wiring_is_the_named_gap():
    """The PAD-carries-ordinal shape is not a new content channel: a pad's bytes are fixed, and carrying the ordinal
    confines a read to the connection's own cells. The residual is the stated named gap — the units' wiring is unbound,
    so a prover may open connection 0's pad (cell 1) where its data cell (cell 0) is intended, each a port the verifier
    accepts; nothing here binds a unit to the data cell rather than the pad."""
    cells_ = _cells()
    head, wiring = cells_.head, cells_.wiring
    on_data = W.port(head, wiring, PARAMS, [(0, 0)])
    on_pad = W.port(head, wiring, PARAMS, [(0, 1)])
    assert on_data["reads"] == on_pad["reads"] == [0], "both the data cell and the pad are in connection 0's bucket row"
    # the two reads open different slots of the same row; the port names only the row, so which slot (data or pad) a
    # unit reads is the prover's within the connection — the named gap, bounded by log2(cells) bits a read
    data_cell = cells_.rows[0][0:B]
    pad_cell = cells_.rows[0][B:2 * B]
    assert data_cell[0] == DATA and pad_cell[0] == PAD and data_cell != pad_cell
