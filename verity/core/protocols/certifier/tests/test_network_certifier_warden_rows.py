"""The warden's cells as registered rows (`warden_rows`): every slot of a link-window is a row whatever the load, each
row is its cell's bytes, opens the signed root at its position and only there, the registration reads back under the
warden's key and session and the schedule's grid, and refuses any field, key, session or certificate but its own."""

from __future__ import annotations

import dataclasses
import json

import pytest

import signers
from verity.ml.commitments import hm96
from verity.ml.crypto import lms
from verity.core.protocols.certifier import warden_rows as W
from verity.core.protocols.certifier.certificate import Direction, Issuer, LmsKey
from verity.core.protocols.certifier.tests.test_network_certifier_warden import (B, EOF, P, PARAMS, cells, records,
                                                                                 request, run)
from verity.core.protocols.sampled_proofs.one_stage import registered as G

GRID = PARAMS.grid
DIGEST = PARAMS.digest


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
    cells_, cert = W.attest(issuer, record, GRID, DIGEST, direction)
    return issuer, cells_, cert


def test_a_link_window_is_every_slot_of_its_grid_whatever_the_load(traffic):
    """N = T·lanes·r rows in every window, loaded or idle; each row is its slot's cell, an empty slot B zero bytes."""
    warden, _ = traffic
    idle, _, _ = run([], {}, windows=1)
    for record in [*records(warden)[1], *records(idle)[1], *records(warden)[0]]:
        rows = W.window_rows(record, GRID)
        assert len(rows) == W.rows_of(GRID) == PARAMS.window * PARAMS.lanes * PARAMS.slots
        assert rows == [bytes(B) if f is None else f for row in record.grid for f in row]
    assert all(r == bytes(B) for r in W.window_rows(records(idle)[1][0], GRID))


def test_a_connections_rows_are_the_cells_its_client_got_in_order(traffic):
    warden, clients = traffic
    seen = 0
    for record in records(warden)[1]:
        rows, first = W.window_rows(record, GRID), record.window * PARAMS.window
        for ordinal, positions in W.connection_rows(record, GRID).items():
            wire = b"".join(d for t, d in clients[ordinal].writes if first <= t < first + PARAMS.window)
            assert [rows[p] for p in positions] == [wire[i:i + B] for i in range(0, len(wire), B)]
            seen += len(positions)
    assert seen == sum(len(c.cells()) for c in clients) > 0


def test_every_row_opens_the_signed_root_at_its_position_and_nowhere_else(traffic, tmp_path):
    """A row's b‖c is the statement's (the `hm96-sha512/row/v2` x of its u16 words under its salt), climbs its path to the
    root under the verifier's own domain, and fails at another position, with a changed bit, under another salt or
    another root."""
    warden, _ = traffic
    _, cells_, cert = _attest(tmp_path, records(warden)[1][0])
    head = cells_.head
    assert cert.digest == head.digest()
    dom = W.domain(DIGEST, signers.SESSION, "responses", Direction.EGRESS, 0, B, W.rows_of(GRID))
    assert dom.domain_id == head.domain
    for k in range(len(cells_.rows)):
        bc = W.bc(cells_.rows[k], cells_.salts[k])
        assert bc == hm96.SHA512.commit_string(
            hm96.SHA512.default_key, G.row_digest(W.words(cells_.rows[k]), 16, G.SCHEMA_V2), cells_.salts[k])
        assert W.opens(dom, head.root, k, bc, cells_.path(k))
    k = next(i for i, r in enumerate(cells_.rows) if r != bytes(B))
    flipped = bytes([cells_.rows[k][0] ^ 1]) + cells_.rows[k][1:]
    assert not W.opens(dom, head.root, k, W.bc(flipped, cells_.salts[k]), cells_.path(k))
    assert not W.opens(dom, head.root, k, W.bc(cells_.rows[k], hm96.SHA512.fresh_salt()), cells_.path(k))
    assert not W.opens(dom, head.root, k ^ 1, W.bc(cells_.rows[k], cells_.salts[k]), cells_.path(k))
    other = W.Cells.commit("responses", Direction.EGRESS, 0, cells_.rows, DIGEST, signers.SESSION)
    assert other.root != head.root
    assert not W.opens(dom, other.root, k, W.bc(cells_.rows[k], cells_.salts[k]), cells_.path(k))
    rows, salts, entry = cells_.statement([k, 0])
    assert rows == [W.words(cells_.rows[k]), W.words(cells_.rows[0])] and salts == [cells_.salts[k], cells_.salts[0]]
    assert entry == {"value": "responses/egress/0", "positions": [k, 0],
                     "paths": [cells_.path(k).hex(), cells_.path(0).hex()]}


def test_salts_are_fresh_and_never_shared():
    rows = [bytes(B)] * 4
    a = W.Cells.commit("warden", Direction.EGRESS, 0, rows, DIGEST, signers.SESSION)
    b = W.Cells.commit("warden", Direction.EGRESS, 0, rows, DIGEST, signers.SESSION)
    assert len(set(a.salts + b.salts)) == 8 and len(set(a.bcs)) == 4 and a.root != b.root
    with pytest.raises(ValueError, match="never reused"):
        W.Cells.commit("warden", Direction.EGRESS, 0, rows, DIGEST, signers.SESSION, [a.salts[0]] * 4)
    with pytest.raises(ValueError, match="u16 words"):
        W.Cells.commit("warden", Direction.EGRESS, 0, [bytes(3)], DIGEST, signers.SESSION)


def test_the_registration_reads_back_and_refuses_anything_but_its_own(traffic, tmp_path):
    """The verifier's read takes the link-window it expects and the schedule's grid and parameters; any other field, an
    extra key, a root the certificate doesn't sign, another key, session or link-window, or the wire record's
    certificate of the same link-window, is refused."""
    warden, _ = traffic
    record = records(warden)[1][1]
    issuer, cells_, cert = _attest(tmp_path / "a", record)
    doc = json.loads(json.dumps(W.registration(cells_.head, cert, GRID)))
    key = issuer.signer.public_key()
    args = (GRID, DIGEST, "responses", Direction.EGRESS, 1)
    head, dom = W.check_registration(doc, key, signers.SESSION, *args)
    assert head == cells_.head and dom.domain_id == cells_.domain.domain_id
    assert doc["layout"] == {"buckets": PARAMS.window, "slots": PARAMS.lanes * PARAMS.slots,
                             "first_bucket": PARAMS.window}
    assert (doc["rows"], doc["words"], doc["word_bits"], doc["row_schema"]) == (W.rows_of(GRID), B // 2, 16,
                                                                              G.SCHEMA_V2)
    for k, v in [("rows", doc["rows"] - 1), ("cell_bytes", B * 2), ("words", B), ("word_bits", 32),
                 ("row_schema", G.SCHEMA), ("window", 0), ("direction", "ingress"), ("link", "requests"),
                 ("value", "responses/egress/0"), ("parameters", "00" * 64), ("domain", "00" * 64),
                 ("root", "00" * 64), ("format", "x"), ("layout", {**doc["layout"], "first_bucket": 0})]:
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
        W.check_registration(doc, key, signers.SESSION, GRID, DIGEST, "responses", Direction.EGRESS, 2)
    wire = issuer.certify(record, Direction.EGRESS, GRID, DIGEST)
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
    cells_, cert = W.attest(issuer, records(warden)[1][0], GRID, DIGEST, Direction.EGRESS)
    assert cert is None and issuer.refused == [("responses", Direction.EGRESS, 0)]
    assert len(cells_.rows) == W.rows_of(GRID)


def test_the_port_is_the_verifiers_registered_port(traffic, tmp_path):
    """`port` holds only a `verity/registered-reads/v1` port's keys (`Flock.Registered.PORT_KEYS`), reads the rows the
    wiring gives, and names the head's root, domain and row format."""
    warden, _ = traffic
    _, cells_, _ = _attest(tmp_path, records(warden)[1][0])
    reads = W.connection_rows(records(warden)[1][0], GRID)[0]
    p = W.port(cells_.head, reads, reads_from=3)
    assert set(p) <= {"value", "root", "leaves", "words", "word_bits", "domain", "row_schema", "reads", "reads_from",
                      "known"}
    assert p == {"value": "responses/egress/0", "root": cells_.root.hex(), "leaves": W.rows_of(GRID), "words": B // 2,
                 "word_bits": 16, "domain": cells_.domain.domain_id.hex(), "row_schema": G.SCHEMA_V2, "reads": reads,
                 "reads_from": 3}
    assert "reads_from" not in W.port(cells_.head, reads)
    assert W.reads_file({"o0": p}) == {"format": G.READS_V1, "ports": {"o0": p}}


def test_the_issuers_sign_is_certifys_signature(tmp_path):
    """`certify` signs through `sign`: one leaf a certificate, both verifying under the key in the session."""
    idle, _, _ = run([], {}, windows=1)
    record = records(idle)[1][0]
    issuer = Issuer(signers.signer(tmp_path), signers.SESSION)
    wire = issuer.certify(record, Direction.EGRESS, GRID, DIGEST)
    cells_, cert = W.attest(issuer, record, GRID, DIGEST, Direction.EGRESS)
    key = issuer.signer.public_key()
    assert wire.verify(key, signers.SESSION) and cert.verify(key, signers.SESSION)
    assert signers.used(issuer.signer) == [0, 1]
    assert dataclasses.replace(cert, digest=wire.certificate.digest).verify(key, signers.SESSION) is False
