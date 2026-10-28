"""The warden's bucket rows with reads tied to the layout (`warden_rows`), held as regressions: a unit reads only its
connection's own cells, and a PAD cell is no content channel.

With the reads (the units' wiring) unconstrained, a unit could open any of the N = T·S rows of the correct link-window
— a foreign connection's cell, a cell twice, or an empty/padding slot — an ≈ log2(N) bits a read channel. The warden
signs the wiring in its head, and `Wiring.place`/`port` confine every read to a connection's own cells, refusing a
foreign connection, a cell past the connection's, a read twice, an empty slot and a wiring the head didn't sign. Here
each is run against the v2 layout, with ordinals from 1, and shown refused.

A PAD cell carries its connection's ordinal (`cell(PAD, q, b"")`) and sits in the connection's own slots, so it counts
toward the connection's cells and `place` will open it. These cases show it is not a *content* channel — a pad's bytes
are fixed by (PAD, q): header and zero fill, no prover-chosen bits — and that carrying the ordinal *narrows* the read
space to a connection's own cells. Which of a connection's own cells each unit reads is the Program's
(`warden_rows.Units`), and the port's value names its digest, so a statement made for a pad where a data cell is
intended is refused by the verifier's registered reads.
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
    """One clean link-window: connection 1 holds a DATA cell and then a PAD in its own two slots of bucket 0;
    connection 2 holds two DATA cells. Later buckets are idle (empty slots, B zero bytes)."""
    bucket0 = (cell(DATA, 1, bytes(P), B), cell(PAD, 1, b"", B), cell(DATA, 2, bytes(P), B), cell(DATA, 2, bytes(P), B))
    grid = (bucket0,) + ((None,) * S,) * (T - 1)
    return Record(LINK, 0, grid, COMPLETE)


def _cells():
    return W.Cells.of(LINK, Direction.EGRESS, _record(), PARAMS, SESSION)


def test_a_pad_carries_its_connections_ordinal_and_counts_toward_its_cells():
    """The wiring reads a pad's ordinal from its header, so connection 1's PAD is its own cell: connection 1 has two
    cells (one DATA, one PAD) in its two slots of bucket 0, and `place` opens the pad at its slot."""
    cells_ = _cells()
    assert cells_.wiring.connections == ((1, 0, 2), (2, 0, 2))  # the pad counts toward connection 1's two cells
    placed = cells_.wiring.place(PARAMS, [(1, 0), (1, 1)])
    assert placed == [(0, 0), (0, 1)]  # connection 1's data cell at slot 0, its pad at slot 1, same bucket row
    pad = cells_.rows[0][1 * B:2 * B]
    assert pad == cell(PAD, 1, b"", B) and pad[0] == PAD, "a pad's bytes are fixed by (PAD, ordinal): no prover bits"


def test_my_phase8_reads_are_refused_foreign_duplicate_cell_past_and_empty():
    """The open read vectors on the v2 layout: a foreign connection, a cell past the connection's (which
    is the only way an empty slot could be addressed), and a read twice are refused by `port`, and a wiring the head
    didn't sign is refused; so every read opens a connection's own cell and no empty, foreign or repeated row."""
    cells_ = _cells()
    head, wiring = cells_.head, cells_.wiring
    assert W.port(head, wiring, PARAMS, [(1, 0), (2, 1)])["reads"] == [0, 0]  # honest reads open the signed root
    absent = max(c[0] for c in wiring.connections) + 1
    for bad, why in [([(absent, 0)], "holds no slot"),          # a foreign connection
                     ([(1, 2)], "not cell"),                     # past connection 1's two: no empty slot is addressable
                     ([(1, -1)], "not cell"),
                     ([(1, 0), (1, 0)], "read twice"),           # the same cell twice
                     ([(1, 0.0)], "is \\(connection, cell\\)")]:
        with pytest.raises(ValueError, match=why):
            W.port(head, wiring, PARAMS, bad)
    for other in [W.Wiring(wiring.connections, bytes(64)), W.Wiring(wiring.connections[:-1], wiring.salt)]:
        with pytest.raises(ValueError, match="isn't the one the warden signed"):
            W.port(head, other, PARAMS, [(1, 0)])


def test_a_pad_read_where_the_program_reads_a_data_cell_is_refused():
    """The residual, flipped: the Program pins its units' wiring, here unit 0 on connection 1's data cell (cell 0). A
    statement made for connection 1's pad (cell 1) opens the same bucket row at the same position, so the rows alone
    can't tell them apart; but the verifier's port names the Program's units' wiring in its value, the statement's
    header names the pad's, and the registered reads refuse it ("not another value"). The Program's own statement is
    accepted; one that names no units' wiring is refused too. A pad statement that copies the Program's value (it is
    public) is the Program's statement, byte for byte: the unit's cell is the slot the verifier places the Program's
    read at, the data cell's, whatever the prover meant."""
    cells_ = _cells()
    head, wiring = cells_.head, cells_.wiring
    program, pad = W.Units(((1, 0),)), W.Units(((1, 1),))
    port = W.port(head, wiring, PARAMS, program)
    assert port["reads"] == W.port(head, wiring, PARAMS, pad)["reads"] == [0], "both cells are in connection 1's row"
    data_cell, pad_cell = cells_.rows[0][0:B], cells_.rows[0][B:2 * B]
    assert data_cell[0] == DATA and pad_cell[0] == PAD and data_cell != pad_cell
    bcs = [cells_.bcs[0]]
    ours = cells_.statement([0], program)[2]
    assert W.refusal(port, cells_.domain, ours, bcs) is None
    for made_for in [cells_.statement([0], pad)[2], cells_.statement([0])[2]]:
        assert made_for["positions"] == port["reads"]
        assert "not another value" in W.refusal(port, cells_.domain, made_for, bcs)
        assert made_for | {"value": port["value"]} == ours
    assert wiring.place(PARAMS, program.cells) == [(0, 0)] and cells_.rows[0][0:B][0] == DATA
