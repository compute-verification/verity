"""The warden's cells as registered rows (`warden_rows`): a link-window is a row a bucket whatever the load, each row is
its bucket's cells in slot order, the wiring places each connection's cells where its client got them, each row opens
the signed root at its position and only there, the registration reads back under the warden's key and session and the
schedule, and refuses any field, key, session or certificate but its own, and the port reads only what the warden's
wiring places."""

from __future__ import annotations

import dataclasses
import json

import pytest

import signers
from verity.ml.commitments import hm96
from verity.ml.crypto import lms
from verity.core.protocols.certifier import warden_rows as W
from verity.core.protocols.certifier.active import DATA
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsKey
from verity.core.protocols.certifier.grid import Reason, BlockGrid, Status
from verity.core.protocols.certifier.tests.test_network_certifier_warden import (B, EOF, P, PARAMS, cells, records,
                                                                                 request, run)
from verity.core.protocols.certifier.warden import cell
from verity.core.protocols.sampled_proofs.one_stage import registered as G

T, R, S = PARAMS.window, PARAMS.slots, PARAMS.lanes * PARAMS.slots


def _traffic():
    reqs = [request(10, 1), request(2 * P + 3, 2), request(P, 3)]
    resps = [bytes(range(256)), b"", b"z" * (P + 1)]
    warden, clients, _ = run([[(0, r), (0, EOF)] for r in resps],
                             {1: [cells(reqs[0], B)], 2: [cells(reqs[1], B)], 4: [cells(reqs[2], B)]}, windows=2)
    return warden, clients


@pytest.fixture(scope="module")
def traffic():
    return _traffic()


def _attest(tmp_path, record, direction=Direction.EGRESS, session=signers.SESSION, seed=b"certifier"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    issuer = Issuer(signers.signer(tmp_path, seed), session)
    cells_, cert = W.attest(issuer, record, PARAMS, direction)
    return issuer, cells_, cert


def test_a_link_window_is_a_row_a_bucket_whatever_the_load(traffic):
    """T rows in every window, loaded or idle; row b is bucket b's S slots in slot order, an empty slot B zero bytes."""
    warden, _ = traffic
    idle, _, _ = run([], {}, windows=1)
    for record in [*records(warden)[1], *records(idle)[1], *records(warden)[0]]:
        rows = W.window_rows(record, PARAMS)
        assert len(rows) == W.rows_of(PARAMS) == T and {len(r) for r in rows} == {W.row_bytes(PARAMS)} == {S * B}
        assert rows == [b"".join(bytes(B) if f is None else f for f in row) for row in record.grid]
    assert all(r == bytes(S * B) for r in W.window_rows(records(idle)[1][0], PARAMS))
    assert W.wiring(records(idle)[1][0], PARAMS).connections == ()


def test_the_wiring_places_each_connections_cells_where_its_client_got_them(traffic):
    """Connection q's cell k of the link-window is at its row's slot `place` gives: the cells its client got that
    window, in order. Ordinals count connections from 1 in the order they reached the warden (the wire's ingress
    ordinal: the active certifier's `Requests` and the I/O tie's map number them so). A bucket's cells are sorted by
    stream."""
    warden, clients = traffic
    seen, ordinals, shared = 0, set(), 0
    for record in records(warden)[1]:
        rows, first = W.window_rows(record, PARAMS), record.window * T
        wiring = W.wiring(record, PARAMS)
        for q, _, n in wiring.connections:
            wire = b"".join(d for t, d in clients[q - 1].writes if first <= t < first + T)
            placed = wiring.place(PARAMS, [(q, k) for k in range(n)])
            assert [rows[t][j * B:(j + 1) * B] for t, j in placed] == [wire[i:i + B] for i in range(0, len(wire), B)]
            seen += n
            ordinals.add(q)
        for row in record.grid:
            live = [f[1:5] for f in row if f is not None]
            assert live == sorted(live) and all(f is None for f in row[len(live):])
            shared += len({x for x in live}) > 1
    assert seen == sum(len(c.cells()) for c in clients) > 0 and shared > 0
    assert sorted(ordinals) == list(range(1, len(clients) + 1))


def test_the_wiring_is_the_schedules_and_only_a_record_laid_out_by_it_has_one(traffic):
    """A wiring whose ordinals don't increase, whose admissions aren't in ordinal order, whose connection isn't r cells
    a row within the window, which puts more connections in a row than there are lanes, or whose salt isn't 64 bytes, is
    refused; so is a record whose cells aren't where the schedule puts them."""
    salt = bytes(64)
    good = W.Wiring(((0, 0, 2 * R), (1, 1, R)), salt)
    assert good.live(PARAMS)[:3] == [[0], [0, 1], []]
    for bad in [((1, 0, R), (0, 0, R)), ((0, 2, R), (1, 1, R)), ((0, 0, R + 1),), ((0, 0, 0),), ((0, T - 1, 2 * R),),
                ((0, 0, R), (1, 0, R), (2, 0, R)), ((0, -1, R),), ((0, 0, True),), ((1 << 32, 0, R),)]:
        with pytest.raises(ValueError):
            W.Wiring(bad, salt).live(PARAMS)
    with pytest.raises(ValueError, match="salt"):
        W.Wiring(good.connections, bytes(63)).live(PARAMS)
    warden, _ = traffic
    record = records(warden)[1][0]
    t = next(t for t, row in enumerate(record.grid) if len({f[1:5] for f in row if f is not None}) > 1)
    swapped = record.grid[t][R:2 * R] + record.grid[t][:R] + record.grid[t][2 * R:]
    with pytest.raises(ValueError, match="laid out"):
        W.wiring(dataclasses.replace(record, grid=record.grid[:t] + (swapped,) + record.grid[t + 1:]), PARAMS)


def test_every_row_opens_the_signed_root_at_its_position_and_nowhere_else(traffic, tmp_path):
    """A row's b‖c is the statement's (the `hm96-sha512/row/v2` x of its u16 words under its salt), climbs its path to the
    root under the verifier's own domain, and fails at another position, with a changed bit, under another salt or
    another root."""
    warden, _ = traffic
    _, cells_, cert = _attest(tmp_path, records(warden)[1][0])
    head = cells_.head
    assert cert.digest == head.digest()
    dom = W.domain(PARAMS, signers.SESSION, "responses", Direction.EGRESS, 0)
    assert dom.domain_id == head.domain
    for k in range(len(cells_.rows)):
        bc = W.bc(cells_.rows[k], cells_.salts[k])
        assert bc == hm96.SHA512.commit_string(
            hm96.SHA512.default_key, G.row_digest(W.words(cells_.rows[k]), 16, G.SCHEMA_V2), cells_.salts[k])
        assert W.opens(dom, head.root, k, bc, cells_.path(k))
    k = next(i for i, r in enumerate(cells_.rows) if r != bytes(S * B))
    flipped = bytes([cells_.rows[k][0] ^ 1]) + cells_.rows[k][1:]
    assert not W.opens(dom, head.root, k, W.bc(flipped, cells_.salts[k]), cells_.path(k))
    assert not W.opens(dom, head.root, k, W.bc(cells_.rows[k], hm96.SHA512.fresh_salt()), cells_.path(k))
    assert not W.opens(dom, head.root, k ^ 1, W.bc(cells_.rows[k], cells_.salts[k]), cells_.path(k))
    other = W.Cells.commit("responses", Direction.EGRESS, 0, cells_.rows, cells_.wiring, PARAMS, signers.SESSION)
    assert other.root != head.root
    assert not W.opens(dom, other.root, k, W.bc(cells_.rows[k], cells_.salts[k]), cells_.path(k))
    rows, salts, entry = cells_.statement([k, 0])
    assert rows == [W.words(cells_.rows[k]), W.words(cells_.rows[0])] and salts == [cells_.salts[k], cells_.salts[0]]
    assert entry == {"value": "responses/egress/0", "positions": [k, 0],
                     "paths": [cells_.path(k).hex(), cells_.path(0).hex()]}


def test_salts_are_fresh_and_never_shared_and_the_witness_reads_back():
    rows, wiring = [bytes(S * B)] * T, W.Wiring((), bytes(64))
    a = W.Cells.commit("warden", Direction.EGRESS, 0, rows, wiring, PARAMS, signers.SESSION)
    b = W.Cells.commit("warden", Direction.EGRESS, 0, rows, wiring, PARAMS, signers.SESSION)
    assert len(set(a.salts + b.salts)) == 2 * T and len(set(a.bcs)) == T and a.root != b.root
    assert W.read_witness(a.witness) == (list(a.salts), wiring.salt)
    with pytest.raises(ValueError, match="never reused"):
        W.Cells.commit("warden", Direction.EGRESS, 0, rows, wiring, PARAMS, signers.SESSION, [a.salts[0]] * T)
    for bad in ([bytes(S * B)] * (T - 1), [bytes(B)] * T):
        with pytest.raises(ValueError, match="one a bucket"):
            W.Cells.commit("warden", Direction.EGRESS, 0, bad, wiring, PARAMS, signers.SESSION)
    with pytest.raises(ValueError, match="witness"):
        W.read_witness(a.witness[:-1])


def test_the_registration_reads_back_and_refuses_anything_but_its_own(traffic, tmp_path):
    """The verifier's read takes the link-window it expects and the schedule's parameters; any other field, an extra
    key, a root or wiring the certificate doesn't sign, another key, session or link-window, or the wire record's
    certificate of the same link-window, is refused."""
    warden, _ = traffic
    record = records(warden)[1][1]
    issuer, cells_, cert = _attest(tmp_path / "a", record)
    doc = json.loads(json.dumps(W.registration(cells_.head, cert, PARAMS)))
    key = issuer.signer.public_key()
    args = (PARAMS, "responses", Direction.EGRESS, 1)
    head, dom = W.check_registration(doc, key, signers.SESSION, *args)
    assert head == cells_.head and dom.domain_id == cells_.domain.domain_id
    assert doc["layout"] == {"buckets": T, "slots": S, "connection_slots": R, "first_bucket": T}
    assert (doc["rows"], doc["words"], doc["word_bits"], doc["row_schema"]) == (T, S * B // 2, 16, G.SCHEMA_V2)
    assert doc["wiring"] == cells_.wiring.digest().hex()
    for k, v in [("rows", T - 1), ("cell_bytes", B * 2), ("words", B // 2), ("word_bits", 32),
                 ("row_schema", G.SCHEMA), ("window", 0), ("direction", "ingress"), ("link", "requests"),
                 ("value", "responses/egress/0"), ("parameters", "00" * 64), ("domain", "00" * 64),
                 ("root", "00" * 64), ("wiring", "00" * 64), ("wiring", "00"), ("format", "x"),
                 ("layout", {**doc["layout"], "first_bucket": 0}), ("layout", {**doc["layout"], "connection_slots": 1})]:
        with pytest.raises(ValueError):
            W.check_registration({**doc, k: v}, key, signers.SESSION, *args)
    with pytest.raises(ValueError, match="no other"):
        W.check_registration({**doc, "extra": 1}, key, signers.SESSION, *args)
    with pytest.raises(ValueError, match="domain"):
        W.check_registration(doc, key, signers.OTHER_SESSION, *args)
    other = LmsKey(signers.SMALL, lms.public_key(signers.key(b"another")))
    with pytest.raises(ValueError, match="doesn't verify"):
        W.check_registration(doc, other, signers.SESSION, *args)
    with pytest.raises(ValueError):
        W.check_registration(doc, key, signers.SESSION, PARAMS, "responses", Direction.EGRESS, 2)
    wire = issuer.certify(record, Direction.EGRESS, PARAMS.grid, PARAMS.digest)
    assert wire.certificate.verify(key, signers.SESSION) and wire.certificate.digest != cert.digest
    swapped = {**doc, "certificate": {"key": wire.certificate.key_id.hex(), "digest": wire.certificate.digest.hex(),
                                      "signature": wire.certificate.signature.hex()}}
    with pytest.raises(ValueError, match="doesn't sign its head"):
        W.check_registration(swapped, key, signers.SESSION, *args)
    forged = {**doc, "certificate": {**doc["certificate"], "digest": wire.certificate.digest.hex()}}
    with pytest.raises(ValueError, match="doesn't sign its head"):
        W.check_registration(forged, key, signers.SESSION, *args)


def test_a_refused_signer_gives_no_certificate(traffic, tmp_path):
    warden, _ = traffic
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    issuer.signer.start(signers.OTHER_SESSION)
    cells_, cert = W.attest(issuer, records(warden)[1][0], PARAMS, Direction.EGRESS)
    assert cert is None and issuer.refused == [("responses", Direction.EGRESS, 0)]
    assert len(cells_.rows) == T


def test_the_port_reads_the_rows_the_wardens_wiring_places_and_nothing_else(traffic, tmp_path):
    """`port` holds only a `verity/registered-reads/v1` port's keys (`Flock.Registered.PORT_KEYS`), reads each (q, k)'s
    row under the wiring the head names, and names the head's root, domain and row format. A wiring the head doesn't
    name, a connection the wiring doesn't hold, a cell past its connection's or read twice, and a read that isn't
    (connection, cell) are refused."""
    warden, _ = traffic
    _, cells_, _ = _attest(tmp_path, records(warden)[1][0])
    head, wiring = cells_.head, cells_.wiring
    q, first, n = max(wiring.connections, key=lambda c: c[2])
    reads = [(q, k) for k in range(n)]
    p = W.port(head, wiring, PARAMS, reads, reads_from=3)
    assert set(p) <= {"value", "root", "leaves", "words", "word_bits", "domain", "row_schema", "reads", "reads_from",
                      "known"}
    assert p == {"value": "responses/egress/0", "root": cells_.root.hex(), "leaves": T, "words": S * B // 2,
                 "word_bits": 16, "domain": cells_.domain.domain_id.hex(), "row_schema": G.SCHEMA_V2,
                 "reads": [first + k // R for k in range(n)], "reads_from": 3}
    assert "reads_from" not in W.port(head, wiring, PARAMS, reads)
    assert W.reads_file({"o0": p}) == {"format": G.READS_V1, "ports": {"o0": p}}
    others = [W.Wiring(wiring.connections, bytes(64)), W.Wiring(wiring.connections[:-1], wiring.salt)]
    for other in others:
        with pytest.raises(ValueError, match="isn't the one the warden signed"):
            W.port(head, other, PARAMS, reads)
    absent = max(c[0] for c in wiring.connections) + 1
    for bad, why in [([(absent, 0)], "holds no slot"), ([(q, n)], "not cell"), ([(q, -1)], "not cell"),
                     ([(q, 0), (q, 0)], "read twice"), ([q], "is \\(connection, cell\\)"),
                     ([(q, 0.0)], "is \\(connection, cell\\)")]:
        with pytest.raises(ValueError, match=why):
            W.port(head, wiring, PARAMS, bad)


def test_the_bucket_port_reads_every_row_whole(traffic, tmp_path):
    """`bucket_port` (one unit a bucket) is `port`'s entry with unit `reads_from + t` reading row t, an idle
    window's rows too, whose slots `port` can't name, under the head's value. The statement for those rows is accepted
    by the host's read of Lean's registered reads (`refusal`), and one naming a units' wiring or with two rows swapped
    is refused. A head of another schedule is refused."""
    warden, _ = traffic
    idle, _, _ = run([], {}, windows=1)
    for record in (records(warden)[1][0], records(idle)[1][0]):
        _, cells_, _ = _attest(tmp_path / str(id(record)), record)
        p = W.bucket_port(cells_.head, PARAMS, reads_from=5)
        q = cells_.wiring.connections[:1]
        if q:
            assert {k: v for k, v in p.items() if k not in ("reads", "reads_from")} == \
                {k: v for k, v in W.port(cells_.head, cells_.wiring, PARAMS, [(q[0][0], 0)]).items() if k != "reads"}
            units = W.Units(((q[0][0], 0),))
            named = cells_.statement(W.port(cells_.head, cells_.wiring, PARAMS, units)["reads"], units)[2]
            assert "not another value" in W.refusal(p, cells_.domain, named, [cells_.bcs[0]])
        assert p["reads"] == list(range(T)) and p["reads_from"] == 5 and p["root"] == cells_.root.hex()
        assert p["value"] == cells_.head.value
        assert "reads_from" not in W.bucket_port(cells_.head, PARAMS)
        _, _, entry = cells_.statement(list(range(T)))
        assert W.refusal(p, cells_.domain, entry, list(cells_.bcs)) is None
        swapped = [1, 0, *range(2, T)]
        assert "unit 5 reads position 1" in \
            W.refusal(p, cells_.domain, cells_.statement(swapped)[2], [cells_.bcs[t] for t in swapped])
    assert W.wiring(records(idle)[1][0], PARAMS).connections == ()
    with pytest.raises(ValueError, match="schedule's rows"):
        W.bucket_port(cells_.head, dataclasses.replace(PARAMS, delay=PARAMS.delay + 1))


def test_the_programs_units_name_their_wiring_and_a_statement_for_any_other_is_refused(traffic, tmp_path):
    """The Program's `Units` read the rows a bare sequence of the same reads does, and the port's value names their
    digest (`<link>/<direction>/<window>/units/<digest>`), as the statement made for them does, which the host's read
    of Lean's registered reads (`refusal`) accepts. A statement made for any other units' wiring is refused by its
    value, even where its rows are the same (two cells of a row swapped), and so is one that names no units' wiring;
    rows not the units' are refused by position. `Units` refuses no read, ordinal 0, and anything but (int, int)."""
    warden, _ = traffic
    _, cells_, _ = _attest(tmp_path, records(warden)[1][0])
    head, wiring = cells_.head, cells_.wiring
    q, first, n = max(wiring.connections, key=lambda c: c[2])
    units = W.Units.span(q, 0, n)
    assert units.cells == tuple((q, k) for k in range(n))
    assert units.encode() == n.to_bytes(4, "big") + b"".join(q.to_bytes(4, "big") + k.to_bytes(4, "big")
                                                              for k in range(n))
    assert units.digest() == W.tagged_sha512(W.UNITS_TAG, units.encode())
    p = W.port(head, wiring, PARAMS, units, reads_from=3)
    bare = W.port(head, wiring, PARAMS, list(units.cells), reads_from=3)
    assert p == bare | {"value": f"responses/egress/0/units/{units.digest().hex()}"} and bare["value"] == head.value
    positions = p["reads"]
    _, _, entry = cells_.statement(positions, units)
    bcs = [cells_.bcs[t] for t in positions]
    assert entry["value"] == p["value"] and entry["positions"] == positions
    assert W.refusal(p, cells_.domain, entry, bcs) is None
    swapped = W.Units(((q, 1), (q, 0)) + units.cells[2:])
    assert W.port(head, wiring, PARAMS, swapped)["reads"] == positions
    for other in [cells_.statement(positions, swapped)[2], cells_.statement(positions)[2]]:
        assert other["positions"] == positions
        assert "not another value" in W.refusal(p, cells_.domain, other, bcs)
    def named(other: W.Units) -> tuple[dict, list[bytes]]:
        e = cells_.statement(W.port(head, wiring, PARAMS, other)["reads"], other)[2] | {"value": p["value"]}
        return e, [cells_.bcs[t] for t in e["positions"]]

    backwards, later = W.Units(units.cells[::-1]), W.Units.span(q, R, n - R)
    moved, moved_bcs = named(backwards)
    assert moved["positions"][0] != positions[0] and W.refusal(p, cells_.domain, moved, moved_bcs) == \
        f"registered reads: unit 3 reads position {moved['positions'][0]}, and the verifier's program reads " \
        f"{positions[0]}"
    assert f"{n - R} rows for the {n} units" in W.refusal(p, cells_.domain, *named(later))
    assert "does not open" in W.refusal(p, cells_.domain, entry, [bytes(len(b)) for b in bcs])
    with pytest.raises(ValueError, match="aren't the rows the units' wiring reads"):
        cells_.statement(positions[::-1] if len(set(positions)) > 1 else [positions[0] + 1] * n, units)
    for bad in [(), [(q, 0)], ((0, 0),), ((q, -1),), ((q, 0.0),), ((q,),), (q,)]:
        with pytest.raises(ValueError, match="a population reads|a unit reads"):
            W.Units(bad)


def test_the_issuers_sign_is_certifys_signature(tmp_path):
    """`certify` signs through `sign`: one leaf a certificate, both verifying under the key in the session."""
    idle, _, _ = run([], {}, windows=1)
    record = records(idle)[1][0]
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    wire = issuer.certify(record, Direction.EGRESS, PARAMS.grid, PARAMS.digest)
    cells_, cert = W.attest(issuer, record, PARAMS, Direction.EGRESS)
    key = issuer.signer.public_key()
    assert wire.verify(key, signers.SESSION) and cert.verify(key, signers.SESSION)
    assert signers.used(issuer.signer) == [0, 1]
    assert dataclasses.replace(cert, digest=wire.certificate.digest).verify(key, signers.SESSION) is False


def test_a_faulted_window_still_has_its_rows_and_its_wiring():
    """A grid that faulted mid-window holds padding from the fault on: its rows are still T, and its wiring is the
    cells before the fault, still the schedule's."""
    live = tuple(cell(DATA, q, bytes(P), B) for q in (0, 1) for _ in range(R))
    grid = (live,) * 3 + ((None,) * S,) * (T - 3)
    record = BlockGrid("responses", 0, grid, Status(Reason.MONITOR_UNAVAILABLE))
    assert W.wiring(record, PARAMS).connections == ((0, 0, 3 * R), (1, 0, 3 * R))
    assert len(W.window_rows(record, PARAMS)) == T
