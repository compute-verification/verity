"""warden_zk's host side on a warden run driven tick by tick (`test_warden_serve._run`): the driver reads the egress
registration under the deployment's key and session and holds the warden's opening to its head, its units read the
busiest connection's cells, data cells first, each at its bucket's row and the slot the warden's wiring places it, and
each member's registered rows open the warden's signed root exactly as the member says (sender and receiver every row,
unsigned none, altered all but unit 0's); the toy unit is the identity on a row's bits. Nothing here lowers or proves
(`warden_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import DATA, HEADER
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


def test_the_driver_reads_the_registration_and_the_busiest_connections_cells(at):
    p = tws.PARAMS
    assert at.head.value == "warden/egress/0" and at.head.rows == p.window
    assert at.opening.head == at.head and at.registration["root"] == at.head.root.hex()
    assert at.registration["wiring"] == at.opening.wiring.digest().hex()
    cells = [wz.cell_of(at, at.opening.rows[t], i) for i, t in enumerate(at.positions)]
    assert len(at.reads) == UNITS and {q for q, _ in at.reads} == {at.connection} == {0}
    assert all(HEADER.unpack_from(c)[1] == at.connection for c in cells)
    assert all(c[0] in (DATA, END) for c in cells[:at.data_rows]) and at.data_rows >= UNITS
    assert len(set(at.positions)) < UNITS, "a bucket's r cells of the connection share its row"
    port = WR.port(at.head, at.opening.wiring, p, at.reads)
    assert port["reads"] == at.positions and port["words"] == WR.row_bytes(p) // 2


def test_each_members_rows_open_the_wardens_root_as_the_member_says(at):
    got = {name: wz.opened(at, wz.member_rows(name, at)) for name in wz.MEMBERS}
    assert got["sender"] == got["receiver"] == [True] * UNITS
    assert got["unsigned"] == [False] * UNITS
    assert got["altered"] == [False] + [True] * (UNITS - 1)
    altered, sender = wz.member_rows("altered", at), wz.member_rows("sender", at)
    assert altered.salts == sender.salts and altered.entry == sender.entry
    diff = [i for i, (a, b) in enumerate(zip(altered.rows, sender.rows)) if a != b]
    assert diff == [0] and sum(bin(x ^ y).count("1") for x, y in zip(altered.rows[0], sender.rows[0])) == 1
    assert wz.cell_of(at, altered.rows[0], 0) != wz.cell_of(at, sender.rows[0], 0)
    unsigned = wz.member_rows("unsigned", at)
    assert unsigned.own.root != at.head.root and not set(unsigned.salts) & set(at.opening.salts)
    assert wz.cell_of(at, unsigned.rows[0], 0) != wz.cell_of(at, sender.rows[0], 0)
    assert [WR.opens(unsigned.own.domain, unsigned.own.root, q, WR.bc(r, s), bytes.fromhex(path))
            for q, r, s, path in zip(at.positions, unsigned.rows, unsigned.salts, unsigned.entry["paths"])] == \
        [True] * UNITS


def test_the_toy_unit_is_the_identity_on_a_rows_bits(at):
    defn = bind(wz.WardenRowIdentity, B=WR.row_bytes(tws.PARAMS))
    row = at.opening.rows[at.positions[0]]
    bits = np.asarray([[(row[k // 8] >> (k % 8)) & 1 for k in range(8 * len(row))]], dtype=np.uint8)
    assert np.asarray(evaluate_bits(defn, bits)).reshape(-1).tolist() == bits.reshape(-1).tolist()
    assert defn.gates > 0
