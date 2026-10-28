"""Regressions on the warden's bucket rows (`warden_rows`): replays of real warden-signed roots and certificates across
windows and directions, and a certificate the warden signed over another digest, stay refused. Two gaps are closed:
- the layout is bound: the signed head carries S, T and the wiring's digest, and the domain S and T, so a head of
  another layout is another digest, which the certificate doesn't sign;
- the reads aren't the prover's: a read is (connection, cell), and the verifier places it with the warden's signed
  wiring, so a crossed, duplicate or padding read, and a wiring the warden didn't sign, are refused, and no read carries
  log2 N bits.

The link-window is synthetic: connection A (ordinal 0) holds rows 0 and 1, connection B (ordinal 1) row 0, so row 1's
last r slots are padding (empty: cells no connection sent)."""
from __future__ import annotations

import dataclasses

import pytest

import signers
from verity.core.protocols.certifier import warden_rows as W
from verity.core.protocols.certifier.active import DATA
from verity.core.protocols.certifier.certificate import Direction, Issuer, certificate_digest
from verity.core.protocols.certifier.grid import BlockGrid, Status
from verity.core.protocols.certifier.tests.test_network_certifier_warden import PARAMS
from verity.core.protocols.certifier.warden import cell

SESSION = signers.SESSION
T, R, S, B = PARAMS.window, PARAMS.slots, PARAMS.lanes * PARAMS.slots, PARAMS.cell_bytes
LINK = "warden"
QA, QB = 0, 1


def _cell(q: int, k: int) -> bytes:
    return cell(DATA, q, bytes([0xAA if q == QA else 0xBB, k]) * ((B - 9) // 2) + bytes((B - 9) % 2), B)


def _record(window: int) -> BlockGrid:
    rows = [tuple(_cell(QA, k) for k in range(R)) + tuple(_cell(QB, k) for k in range(R)),
            tuple(_cell(QA, k) for k in range(R, 2 * R)) + (None,) * (S - R)]
    return BlockGrid(LINK, window, tuple(rows) + ((None,) * S,) * (T - 2), Status())


def _issuer(tmp_path, seed=b"certifier") -> Issuer:
    return Issuer(signers.signer(tmp_path, seed), SESSION)


def _commit(window: int, direction: Direction, session=SESSION) -> W.Cells:
    return W.Cells.of(LINK, direction, _record(window), PARAMS, session)


def _attest(issuer: Issuer, cells_: W.Cells):
    return issuer.sign(cells_.link, cells_.direction, cells_.window, cells_.head.digest())


def test_the_honest_registration_reads_back(tmp_path):
    issuer = _issuer(tmp_path)
    c0 = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, c0)
    key = issuer.signer.public_key()
    doc = W.registration(c0.head, cert, PARAMS)
    head, dom = W.check_registration(doc, key, SESSION, PARAMS, LINK, Direction.EGRESS, 0)
    assert head == c0.head and dom.domain_id == c0.domain.domain_id


def test_a_real_cross_window_root_and_certificate_replay_is_refused(tmp_path):
    """A REAL warden-signed window-0 root and certificate can't be presented for window 1: the window-0 document is
    refused by the field check, and relabelling it to window 1 with the window-0 root and certificate is refused because
    the certificate signs the window-0 head."""
    issuer = _issuer(tmp_path)
    c0, c1 = _commit(0, Direction.EGRESS), _commit(1, Direction.EGRESS)
    cert0 = _attest(issuer, c0)
    key = issuer.signer.public_key()
    doc0 = W.registration(c0.head, cert0, PARAMS)
    assert W.check_registration(doc0, key, SESSION, PARAMS, LINK, Direction.EGRESS, 0)
    with pytest.raises(ValueError):
        W.check_registration(doc0, key, SESSION, PARAMS, LINK, Direction.EGRESS, 1)
    forged = {**W.registration(c1.head, cert0, PARAMS), "root": c0.head.root.hex(), "wiring": c0.head.wiring.hex()}
    with pytest.raises(ValueError, match="doesn't sign its head|doesn't verify"):
        W.check_registration(forged, key, SESSION, PARAMS, LINK, Direction.EGRESS, 1)


def test_a_real_cross_direction_replay_is_refused(tmp_path):
    issuer = _issuer(tmp_path)
    eg = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, eg)
    key = issuer.signer.public_key()
    doc = W.registration(eg.head, cert, PARAMS)
    assert W.check_registration(doc, key, SESSION, PARAMS, LINK, Direction.EGRESS, 0)
    with pytest.raises(ValueError):
        W.check_registration(doc, key, SESSION, PARAMS, LINK, Direction.INGRESS, 0)


def test_a_warden_certificate_over_another_digest_is_not_a_cells_certificate(tmp_path):
    """The wire record's certificate signs `certificate_digest(salt, record_digest)`. A certificate the warden really
    signed over such a digest verifies on its own, but is refused here: `cert.digest` must be `head.digest()`."""
    issuer = _issuer(tmp_path)
    c0 = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, c0)
    key = issuer.signer.public_key()
    doc = W.registration(c0.head, cert, PARAMS)
    record_style = certificate_digest(bytes(64), c0.head.digest())
    cert_other = issuer.sign(LINK, Direction.EGRESS, 0, record_style)
    assert cert_other.verify(key, SESSION) and cert_other.digest != c0.head.digest()
    swapped = {**doc, "certificate": {"key": cert_other.key_id.hex(), "digest": cert_other.digest.hex(),
                                      "signature": cert_other.signature.hex()}}
    with pytest.raises(ValueError, match="doesn't sign its head"):
        W.check_registration(swapped, key, SESSION, PARAMS, LINK, Direction.EGRESS, 0)


def test_the_layout_is_bound_by_the_signed_head(tmp_path):
    """Was `test_the_layout_is_schedule_checked_but_not_bound_by_the_root`. A disagreeing layout is refused by the field
    check, and the head's signed bytes now carry the layout: B, S, T and the wiring's digest. A head of another slot
    count, row count or wiring is another digest, which the warden's certificate doesn't sign."""
    issuer = _issuer(tmp_path)
    c0 = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, c0)
    key = issuer.signer.public_key()
    doc = W.registration(c0.head, cert, PARAMS)
    for bad in ({**doc["layout"], "slots": doc["layout"]["slots"] + 1},
                {**doc["layout"], "first_bucket": doc["layout"]["first_bucket"] + 1},
                {**doc["layout"], "buckets": doc["layout"]["buckets"] + 1},
                {**doc["layout"], "connection_slots": doc["layout"]["connection_slots"] + 1}):
        with pytest.raises(ValueError, match="layout"):
            W.check_registration({**doc, "layout": bad}, key, SESSION, PARAMS, LINK, Direction.EGRESS, 0)
    head = c0.head
    assert len(head.encode()) == 2 + len(LINK) + 1 + 8 + 64 + 4 + 4 + 4 + 64 + 64 + 64
    assert head.encode()[2 + len(LINK) + 1 + 8 + 64:][:12] == B.to_bytes(4, "big") + S.to_bytes(4, "big") + \
        T.to_bytes(4, "big")
    assert cert.digest == head.digest()
    for change in ({"slots": S + 1}, {"rows": T * S}, {"wiring": bytes(64)}):
        assert dataclasses.replace(head, **change).digest() != cert.digest


def test_the_wiring_reads_are_tied_to_the_wardens_layout(tmp_path):
    """Was `test_the_wiring_reads_are_unconstrained_prover_witness`, the main finding. A read is (connection, cell),
    and the verifier places it with the wiring the warden signed: the honest reads open A's cells at their row and
    slot; a duplicate, a read of a cell no connection sent (past B's cells, or of a connection that isn't there) and a
    crossed or padded wiring are refused. So no read opens a foreign connection's cell, a cell twice or an empty slot."""
    issuer = _issuer(tmp_path)
    c0 = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, c0)
    key = issuer.signer.public_key()
    head, dom = W.check_registration(W.registration(c0.head, cert, PARAMS), key, SESSION, PARAMS, LINK,
                                     Direction.EGRESS, 0)
    wiring = c0.wiring
    assert wiring.connections == ((QA, 0, 2 * R), (QB, 0, R))
    honest = [(QA, 0), (QA, 1), (QA, R), (QB, 0)]
    p = W.port(head, wiring, PARAMS, honest)
    assert p["reads"] == [0, 0, 1, 0] and p["root"] == head.root.hex() and p["value"] == "warden/egress/0"
    for (q, k), (t, j) in zip(honest, wiring.place(PARAMS, honest)):
        assert c0.rows[t][j * B:(j + 1) * B] == _cell(q, k)
        assert W.opens(dom, head.root, t, W.bc(c0.rows[t], c0.salts[t]), c0.path(t))
    for reads, why in [([(QA, 0), (QA, 0)], "read twice"), ([(QB, R)], "not cell"), ([(2, 0)], "holds no slot"),
                       ([0, 1], "is \\(connection, cell\\)")]:
        with pytest.raises(ValueError, match=why):
            W.port(head, wiring, PARAMS, reads)
    crossed = W.Wiring(((QA, 0, R), (QB, 0, 2 * R)), wiring.salt)
    padded = W.Wiring(wiring.connections + ((2, 1, R),), wiring.salt)
    assert crossed.place(PARAMS, [(QB, R)]) == wiring.place(PARAMS, [(QA, R)])
    assert padded.place(PARAMS, [(2, 0)]) == [(1, R)] and c0.rows[1][R * B:(R + 1) * B] == bytes(B)
    for forged in (crossed, padded):
        with pytest.raises(ValueError, match="isn't the one the warden signed"):
            W.port(head, forged, PARAMS, [(QA, 0)])
