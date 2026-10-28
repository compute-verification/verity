"""transfer_zk's host side, on both blocks: the circuit record and blocks pass the verifier's blocks check, whose pins,
from the network certifier's registrations alone, leave no free input (and none leave every input free); a response
without its certificate, the prover's own root under the certifier's certificate and the certifier's certificate of
another window are refused; each member's statement is read as the member says (honest accepted, forged and unsigned
refused at o0 alone, request at x alone and weight at wrow alone, each by its `WHY`), forged's response being one byte
off unit 0's output and request's and weight's unit 0 one code off the signed row with its output unchanged; the leak
scan's blobs are whole pieces. The h200 block's load is the seeded weights' ``l0.wk`` and ``l0.wk.scale``: its rows
under the fixture's seed hash to the fixture's digests (``demos/h200/tests/fixtures/weights-v1.json``: SHA-256
of each tensor's first and last generation row), and its registered
rows are interface 1's (4,096 of 128 words, 64 of 2). Nothing here lowers or proves (`transfer_zk.prove` does, on a
node)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.certifier import transfer_rows as TR

HERE = Path(__file__).resolve().parents[1]
#: interface 1's fixture at H200 under its test seed: ``[name, SHA-256 of generation row 0, of the last]``
FIXTURE_ROWS = {
    "l0.wk": ("277aaa74a2fe0f9c469c2ffa3f9fcffa6faeeb82e1b5879c696b92373e604dea",
              "2d74b2630ba0a45a01026d1a9b7678524c3cc7ef4a9236dbd1fc2e39c040a00c"),
    "l0.wk.scale": ("c46545c4dcdbd4084ea2d9e54b07252dc03d176b2f320b67e3e4c9d017189c95",) * 2}


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_transfer_zk", HERE / "transfer_zk.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tz = _load()
UNITS = 4


@pytest.fixture(scope="module", params=["tiny", "h200"])
def at(request, tmp_path_factory):
    return tz.attestation(UNITS, tmp_path_factory.mktemp(f"transfer-zk-{request.param}"), tz.BLOCKS[request.param])


def test_the_certified_pins_leave_no_free_input(at):
    v = tz.verifier(at, tz.presented("honest", at))
    loads = [at.block.values[p][0] for p in ("wrow", "sw")]
    assert set(v.pins) == {"request", "request_scale", *loads}
    assert all(v.pins[x] == at.transfers[x].root.hex() for x in v.pins)
    assert v.free == 0 and v.free_unpinned == UNITS * 16 * (2 * at.block.k // 2 + 2 + 2)
    assert {p: v.ports[p]["root"] for p in tz.PORTS} == {p: at.transfers[at.block.values[p][0]].root.hex() for p in tz.PORTS}
    forged = tz.verifier(at, tz.presented("forged", at))
    assert forged.pins == v.pins and tz.head(forged.heads, at.block, "o0").root == at.forged.root != at.transfers["response"].root


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
        got[name] = tz.refused(at, v, tz.member_ports(name, at), base=3, name=name)
    assert got["honest"] == {}
    for name, p in tz.REFUSED_AT.items():
        assert set(got[name]) == {p} and tz.WHY[name] in got[name][p], (name, got[name])
        assert got[name][p].startswith("registered reads: table row 0 does not open")
    for name, p in tz.CHANGED.items():
        rows, k = tz.member_rows(name, at), at.changed_code[name]
        diff = rows[p].view(np.uint8) ^ at.words[p].view(np.uint8)
        assert diff.sum() == 1 and diff[0, k] == 1, name
        assert all(np.array_equal(rows[q], at.words[q]) for q in tz.PORTS if q != p)
        assert np.array_equal(tz.outputs(at.block, rows), at.words["o0"])
    rows = np.asarray(at.forged.committed.rows, dtype=np.uint16)
    assert (rows != at.words["o0"]).sum() == 1 and rows[0, 0] ^ at.words["o0"][0, 0] == 1
    assert at.own.root != at.transfers["response"].root and not set(at.own.salts) & set(at.transfers["response"].salts)
    honest = tz.member_ports("honest", at)
    for p in tz.PORTS:
        salts, entry = honest[p]
        t = at.transfers[at.block.values[p][0]]
        assert entry["positions"] == at.reads[p] and entry["value"] == t.value
        assert salts == [t.salts[r] for r in at.reads[p]]
        assert [TR.bc(list(map(int, r)), s) for r, s in zip(at.words[p], salts)] == [t.committed.bcs[r] for r in at.reads[p]]


def test_the_blocks_are_the_circuits_server_and_its_transfers(at):
    b = tz.BK.check(at.doc, at.blocks, tz.layouts(at.doc))
    w, sw = (at.block.values[p][0] for p in ("wrow", "sw"))
    load_rows = (4096, 64) if at.block is tz.H200 else (UNITS, UNITS)
    assert [(t.value, t.kind, t.direction.value, t.window, t.words, t.rows) for t in b.transfers] == [
        ("request", "request", "ingress", 1, at.block.k // 2, UNITS), ("request_scale", "request", "ingress", 1, 2, UNITS),
        (w, "load", "ingress", 0, at.block.k // 2, load_rows[0]), (sw, "load", "ingress", 0, 2, load_rows[1]),
        ("response", "response", "egress", 1, 1, UNITS)]
    tid = at.block.tid
    assert sorted(at.doc["external"]) == sorted(f"{tid}/{p}" for p in ("x", "sx", "o0"))


def test_the_scans_blobs_are_whole_pieces(at):
    m = {"salts": {p: [at.transfers[at.block.values[p][0]].salts[r] for r in at.reads[p]] for p in tz.PORTS}}
    blobs, held = tz.blobs_of(at, m)
    assert held == set(blobs) and len(blobs) == 2 * UNITS + len(tz.PORTS) * UNITS
    assert {len(b) for k, b in blobs.items() if not k.startswith("salt")} == {at.block.k}
    assert {len(b) for k, b in blobs.items() if k.startswith("salt")} == {TR.SALT_BYTES}


def test_the_h200_load_is_interface_1s_tensors():
    """`seeded` is interface 1's generator (its fixture's row digests under its test seed), and `h200_load` its
    registered rows of l0.wk and l0.wk.scale: channel c's part j at row 64·c + j, and the f32 scales in [2^-14, 2^-13)."""
    s, width = tz.TEST_SEED, tz.H200_WIDTH
    wk = [tz.seeded(s, "l0.wk", c, width) for c in (0, tz.H200_HEAD - 1)]
    assert tuple(hashlib.sha256(r).hexdigest() for r in wk) == FIXTURE_ROWS["l0.wk"]
    load = tz.h200_load(UNITS, s)
    (w, w_stride), (sw, sw_stride) = load["wrow"], load["sw"]
    assert w.shape == (4096, 128) and sw.shape == (64, 2) and (w_stride, sw_stride) == (64, 1)
    assert w[64 * 63:].tobytes() == wk[1] and w[:64].tobytes() == wk[0]
    assert hashlib.sha256(sw.tobytes()).hexdigest() == FIXTURE_ROWS["l0.wk.scale"][0]
    f = sw.view("<f4").ravel()
    assert ((2.0 ** -14 <= f) & (f < 2.0 ** -13)).all()
