"""warden_zk's host side on a warden run driven tick by tick (`test_warden_serve._run`): the driver reads the egress
registration under the deployment's key and session and holds the warden's opening to its head; the Program's units
read the egress cells of the connection admitted in the link-window with the most cells, from cell D·r on, chosen
from the wiring and the schedule alone, each at its bucket's row and the slot the warden's wiring places it; each
member's statement is accepted or refused by the host's read of Lean's registered reads exactly as the member says
(sender and receiver accepted; unsigned and altered by their rows; pad and swapped, made for other units' wiring, by
their value, swapped at the Program's own rows; pad-named, under the Program's value, by its row's position); the
scan's public pieces hold every pad the warden wrote, pad's read among them, and none of the Program's cells; the toy
unit is the identity on a row's bits. Nothing here lowers or proves (`warden_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import DATA, HEADER, PAD
from verity.core.protocols.certifier.warden import END

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tests"))
import test_warden_serve as tws  # noqa: E402


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_warden_zk", HERE / "warden_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wz = _load()
UNITS = 8


@pytest.fixture(scope="module")
def at(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("warden-zk")
    out, key = tmp / "run", tmp / "key"
    out.mkdir()
    requests = [tws.ws.http_request({"prompt": "x" * n}, "h:1") for n in (10, 30)]
    tws._run(out, key, requests, [tws._response([" word"] * 40), tws._response([" a"])])
    return wz.attestation(out, UNITS)


def test_the_programs_units_read_the_egress_cells_after_the_delay(at):
    p = tws.PARAMS
    assert at.head.value == "warden/egress/0" and at.head.rows == p.window
    assert at.opening.head == at.head and at.registration["root"] == at.head.root.hex()
    assert at.registration["wiring"] == at.opening.wiring.digest().hex()
    pads = p.delay * p.slots
    assert at.pads == pads and at.units == WR.Units.span(at.connection, pads, UNITS) and at.connection == 1
    assert at.reads == list(at.units.cells)
    q, first, k = next(c for c in at.opening.wiring.connections if c[0] == at.connection)
    assert k == max(c[2] for c in at.opening.wiring.connections) and k >= pads + UNITS
    assert at.window == 0 or first > 0, "admitted in the link-window, so its first D·r cells are its delay's pads"
    cells = [wz.cell_of(at, at.opening.rows[t], i) for i, t in enumerate(at.positions)]
    assert all(HEADER.unpack_from(c)[1] == at.connection for c in cells)
    assert at.data_rows == sum(c[0] in (DATA, END) for c in cells) == UNITS
    delay = [at.opening.rows[t][j * p.cell_bytes:(j + 1) * p.cell_bytes]
             for t, j in at.opening.wiring.place(p, [(q, i) for i in range(pads)])]
    assert {c[0] for c in delay} == {PAD}
    assert len(set(at.positions)) < UNITS, "a bucket's r cells of the connection share its row"
    port = WR.port(at.head, at.opening.wiring, p, at.units)
    assert port["reads"] == at.positions and port["words"] == WR.row_bytes(p) // 2
    assert port["value"] == f"warden/egress/0/units/{at.units.digest().hex()}"


def test_each_members_statement_is_read_as_the_member_says(at):
    rows = {name: wz.member_rows(name, at) for name in wz.MEMBERS}
    got = {name: wz.refused(at, mr, reads_from=5) for name, mr in rows.items()}
    for name, (_port, accepted) in wz.MEMBERS.items():
        assert (got[name] is None) == accepted, (name, got[name])
        assert accepted or wz.WHY[name] in got[name], (name, got[name])
    assert got["pad-named"].startswith(f"registered reads: unit 5 reads position {rows['pad'].positions[0]}")
    opens = {name: wz.opened(at, mr) for name, mr in rows.items()}
    assert opens["sender"] == opens["receiver"] == opens["pad"] == opens["pad-named"] == opens["swapped"] == \
        [True] * UNITS
    assert opens["unsigned"] == [False] * UNITS
    assert opens["altered"] == [False] + [True] * (UNITS - 1)
    sender = rows["sender"]
    assert sender.units == at.units and sender.positions == at.positions and sender.entry["value"] == \
        at.units.value(at.head)
    pad, swapped = rows["pad"], rows["swapped"]
    assert pad.units.cells[0] == (at.connection, 0) and pad.positions[0] != at.positions[0]
    (t, j), = at.opening.wiring.place(tws.PARAMS, [pad.units.cells[0]])
    assert t == pad.positions[0] and pad.rows[0][j * tws.PARAMS.cell_bytes] == PAD
    assert pad.entry["value"] != sender.entry["value"]
    assert rows["pad-named"].entry == pad.entry | {"value": sender.entry["value"]}
    assert swapped.units.cells[:2] == at.units.cells[1::-1] and swapped.positions == at.positions
    assert swapped.rows == sender.rows and swapped.entry == sender.entry | {"value": swapped.units.value(at.head)}
    altered = rows["altered"]
    assert altered.salts == sender.salts and altered.entry == sender.entry
    diff = [i for i, (a, b) in enumerate(zip(altered.rows, sender.rows)) if a != b]
    assert diff == [0] and sum(bin(x ^ y).count("1") for x, y in zip(altered.rows[0], sender.rows[0])) == 1
    assert wz.cell_of(at, altered.rows[0], 0) != wz.cell_of(at, sender.rows[0], 0)
    unsigned = rows["unsigned"]
    assert unsigned.own.root != at.head.root and not set(unsigned.salts) & set(at.opening.salts)
    assert unsigned.entry["value"] == sender.entry["value"]
    assert wz.cell_of(at, unsigned.rows[0], 0) != wz.cell_of(at, sender.rows[0], 0)
    assert [WR.opens(unsigned.own.domain, unsigned.own.root, q, WR.bc(r, s), bytes.fromhex(path))
            for q, r, s, path in zip(at.positions, unsigned.rows, unsigned.salts, unsigned.entry["paths"])] == \
        [True] * UNITS


def test_the_scans_public_pieces_hold_every_pad_and_none_of_the_programs_cells(at):
    B = tws.PARAMS.cell_bytes
    known = wz.pad_cells(at)
    pads = {known[k:k + B] for k in range(0, len(known), B)}
    assert len(pads) == len(at.opening.wiring.connections) and {c[0] for c in pads} == {PAD}
    written = {r[k:k + B] for r in at.opening.rows for k in range(0, len(r), B)}
    assert {c for c in written if c[0] == PAD and any(c)} == pads & written and pads & written
    pad = wz.member_rows("pad", at)
    (_, j), = at.opening.wiring.place(tws.PARAMS, [pad.units.cells[0]])
    assert pad.rows[0][j * B:(j + 1) * B] in pads
    assert not any(wz.cell_of(at, at.opening.rows[t], i) in pads for i, t in enumerate(at.positions))


def test_the_toy_unit_is_the_identity_on_a_rows_bits(at):
    defn = bind(wz.WardenRowIdentity, B=WR.row_bytes(tws.PARAMS))
    row = at.opening.rows[at.positions[0]]
    bits = np.asarray([[(row[k // 8] >> (k % 8)) & 1 for k in range(8 * len(row))]], dtype=np.uint8)
    assert np.asarray(evaluate_bits(defn, bits)).reshape(-1).tolist() == bits.reshape(-1).tolist()
    assert defn.gates > 0
