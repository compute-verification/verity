"""transfer_zk's host side: the TINY server block's circuit record and blocks pass the verifier's blocks check, whose
pins, from the network certifier's registrations alone, leave no free input (and none leave every input free); a
response without its certificate, the prover's own root under the certifier's certificate and the certifier's
certificate of another window are refused; each member's statement is read as the member says (honest accepted, forged
and unsigned refused at o0 alone, each by its `WHY`), forged's response being one byte off unit 0's output; the
leak scan's blobs are whole pieces. Nothing here lowers or proves (`transfer_zk.prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.certifier import transfer_rows as TR

HERE = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_transfer_zk", HERE / "transfer_zk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tz = _load()
UNITS = 4


@pytest.fixture(scope="module")
def at(tmp_path_factory):
    return tz.attestation(UNITS, tmp_path_factory.mktemp("transfer-zk"))


def test_the_certified_pins_leave_no_free_input(at):
    v = tz.verifier(at, tz.presented("honest", at))
    assert set(v.pins) == {"request", "request_scale", "w", "sw"}
    assert all(v.pins[x] == at.transfers[x].root.hex() for x in v.pins)
    assert v.free == 0 and v.free_unpinned == UNITS * 16 * (64 + 2 + 64 + 2)
    assert {p: v.ports[p]["root"] for p in tz.PORTS} == {p: at.transfers[tz.VALUES[p][0]].root.hex() for p in tz.PORTS}
    forged = tz.verifier(at, tz.presented("forged", at))
    assert forged.pins == v.pins and forged.heads["response"].root == at.forged.root != at.transfers["response"].root


def test_the_verifier_refuses_a_response_root_that_isnt_certified_for_its_link_window(at):
    got = tz.refusals(at)
    assert got["no-certificate"].endswith("no certificate; a transfer's root is pinned from its network certifier's "
                                          "certificate alone")
    assert got["unsigned"].endswith("doesn't sign its head") and got["another-window"].endswith("doesn't sign its head")
    assert all(r.startswith("refused: transfer 'response' (response egress on s0/uplink, window 1)") for r in got.values())


def test_each_members_statement_is_read_as_the_member_says(at):
    got = {}
    for name in tz.MEMBERS:
        v = tz.verifier(at, tz.presented(name, at))
        got[name] = tz.refused(at, v, tz.member_ports(name, at), base=3)
    assert got["honest"] == {}
    for name in ("forged", "unsigned"):
        assert set(got[name]) == {"o0"} and tz.WHY[name] in got[name]["o0"], (name, got[name])
        assert got[name]["o0"].startswith("registered reads: table row 0 does not open")
    rows = np.asarray(at.forged.committed.rows, dtype=np.uint16)
    assert (rows != at.words["o0"]).sum() == 1 and rows[0, 0] ^ at.words["o0"][0, 0] == 1
    assert at.own.root != at.transfers["response"].root and not set(at.own.salts) & set(at.transfers["response"].salts)
    honest = tz.member_ports("honest", at)
    for p in tz.PORTS:
        salts, entry = honest[p]
        t = at.transfers[tz.VALUES[p][0]]
        assert salts == list(t.salts) and entry["positions"] == list(range(UNITS)) and entry["value"] == t.value
        assert [TR.bc(list(map(int, r)), s) for r, s in zip(at.words[p], salts)] == list(t.committed.bcs)


def test_the_blocks_are_the_circuits_server_and_its_transfers(at):
    b = tz.BK.check(at.doc, at.blocks, tz.layouts(at.doc))
    assert [(t.value, t.kind, t.direction.value, t.window, t.words, t.rows) for t in b.transfers] == [
        ("request", "request", "ingress", 1, 64, UNITS), ("request_scale", "request", "ingress", 1, 2, UNITS),
        ("w", "load", "ingress", 0, 64, UNITS), ("sw", "load", "ingress", 0, 2, UNITS),
        ("response", "response", "egress", 1, 1, UNITS)]
    assert sorted(at.doc["external"]) == sorted(f"{tz.TID}/{p}" for p in ("x", "sx", "o0"))


def test_the_scans_blobs_are_whole_pieces(at):
    m = {"salts": {p: list(at.transfers[tz.VALUES[p][0]].salts) for p in tz.PORTS}}
    blobs, held = tz.blobs_of(at, m)
    assert held == set(blobs) and len(blobs) == 2 * UNITS + len(tz.PORTS) * UNITS
    assert {len(b) for k, b in blobs.items() if not k.startswith("salt")} == {128}
    assert {len(b) for k, b in blobs.items() if k.startswith("salt")} == {TR.SALT_BYTES}
