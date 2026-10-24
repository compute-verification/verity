"""REDTEAM phase 8, target 2: the warden's cells-as-rows registration (#3594, `warden_rows`, cursor/warden-rows-061a).

Each link-window's cells are committed one `hm96-sha512/row/v2` row a cell under a warden-signed frame-v3-sha512 root,
read through `--registered` (`verity-network-warden/registration/v1`, layout row = (t − w·T)·S + j).
`check_registration` is the verifier's read. The lead's four vectors: a row from another window or link, a replayed
root, a registration whose layout disagrees with the warden's, and the warden's signing of a head that names the root.

Executable red-team evidence, NOT a merge-gate suite. These complement the author's own `test_...warden_rows.py`
negatives (which use a ZEROED root and the wire record's cert); here the replays use REAL warden-signed roots and
certificates, and the main finding is the wiring (`reads`) channel the author names as the open gap.

Findings:
- Cross-window, cross-direction and cross-link replays are REFUSED even with a real warden-signed root and certificate:
  the domain binds {parameters, session, link, direction, window, cell_bytes, rows, row_schema}, the signed `CellsHead`
  binds those plus the root, and the certificate signs the head digest under (key, session, link, direction, window).
- A certificate the warden really signed over a different digest (the wire record's style) is refused: `cert.digest`
  must equal `head.digest()`, so the two certificate kinds are domain-separated by the signed digest.
- The `layout` is field-checked against the schedule, but it is NOT in the signed head or the domain. It cannot lie
  through `check_registration` (equality), and nothing downstream reads it for soundness, so it is informational — no
  exploit, but not cryptographically bound.
- MAIN FINDING (the named gap): the root binds the committed rows and their POSITIONS, never which unit/connection each
  belongs to. `reads` (the wiring) is prover-chosen; `check_registration` never reads it and `port` accepts any in-range
  positions, each of which opens the root. So a prover can wire any unit to any row of the correct link-window — a
  foreign connection's cell, the same cell twice, or an empty/padding slot. Channel ≈ log2(N) bits a read (N = T·S),
  bounded by N (≈13 bits a read at the served schedule).
"""
from __future__ import annotations

import math

import pytest

import signers
from verity.core.protocols.certifier import warden_rows as W
from verity.core.protocols.certifier.certificate import Direction, Issuer, certificate_digest
from verity.core.protocols.certifier.tests.test_network_certifier_warden import PARAMS

GRID = PARAMS.grid
DIGEST = PARAMS.digest
SESSION = signers.SESSION
N = W.rows_of(GRID)
B = GRID.frame_bytes
LINK = "warden"
A_POS, B_POS, PAD_POS = (0, 1), (2, 3), N - 1


def _row(k: int) -> bytes:
    if k in A_POS:
        return bytes([0xAA]) * B
    if k in B_POS:
        return bytes([0xBB]) * B
    if k == PAD_POS:
        return bytes(B)  # an empty slot: a cell no connection sent
    return k.to_bytes(2, "little") + bytes(B - 2)


ROWS = [_row(k) for k in range(N)]


def _issuer(tmp_path, seed=b"certifier") -> Issuer:
    return Issuer(signers.signer(tmp_path, seed), SESSION)


def _commit(window: int, direction: Direction, session=SESSION) -> W.Cells:
    return W.Cells.commit(LINK, direction, window, ROWS, DIGEST, session)


def _attest(issuer: Issuer, cells_: W.Cells):
    return issuer.sign(cells_.link, cells_.direction, cells_.window, cells_.head.digest())


def test_the_honest_registration_reads_back(tmp_path):
    issuer = _issuer(tmp_path)
    c0 = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, c0)
    key = issuer.signer.public_key()
    doc = W.registration(c0.head, cert, GRID)
    head, dom = W.check_registration(doc, key, SESSION, GRID, DIGEST, LINK, Direction.EGRESS, 0)
    assert head == c0.head and dom.domain_id == c0.domain.domain_id


def test_a_real_cross_window_root_and_certificate_replay_is_refused(tmp_path):
    """A REAL warden-signed window-0 root + certificate cannot be presented for window 1: the window-0 doc is refused by
    the field check, and relabelling the fields to window 1 while keeping the window-0 root and certificate is refused
    because the certificate signs the window-0 head, not the window-1 one."""
    issuer = _issuer(tmp_path)
    c0, c1 = _commit(0, Direction.EGRESS), _commit(1, Direction.EGRESS)
    cert0 = _attest(issuer, c0)
    key = issuer.signer.public_key()
    doc0 = W.registration(c0.head, cert0, GRID)
    assert W.check_registration(doc0, key, SESSION, GRID, DIGEST, LINK, Direction.EGRESS, 0)
    with pytest.raises(ValueError):  # window-0 doc read as window 1: field mismatch (window/value/domain/layout)
        W.check_registration(doc0, key, SESSION, GRID, DIGEST, LINK, Direction.EGRESS, 1)
    forged = {**W.registration(c1.head, cert0, GRID), "root": c0.head.root.hex()}  # window-1 shape, window-0 root+cert
    with pytest.raises(ValueError, match="doesn't sign its head|doesn't verify"):
        W.check_registration(forged, key, SESSION, GRID, DIGEST, LINK, Direction.EGRESS, 1)


def test_a_real_cross_direction_replay_is_refused(tmp_path):
    issuer = _issuer(tmp_path)
    eg = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, eg)
    key = issuer.signer.public_key()
    doc = W.registration(eg.head, cert, GRID)
    assert W.check_registration(doc, key, SESSION, GRID, DIGEST, LINK, Direction.EGRESS, 0)
    with pytest.raises(ValueError):  # the egress registration read as the ingress link-window
        W.check_registration(doc, key, SESSION, GRID, DIGEST, LINK, Direction.INGRESS, 0)


def test_a_warden_certificate_over_another_digest_is_not_a_cells_certificate(tmp_path):
    """The wire record's certificate signs `certificate_digest(salt, record_digest)`, a different 64-byte digest under a
    different tag. A certificate the warden really signed over such a digest verifies on its own, but is refused here
    because `cert.digest` must equal `head.digest()` — the two certificate kinds are domain-separated by the digest."""
    issuer = _issuer(tmp_path)
    c0 = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, c0)
    key = issuer.signer.public_key()
    doc = W.registration(c0.head, cert, GRID)
    record_style = certificate_digest(bytes(64), c0.head.digest())  # the wire record's style of signed digest
    cert_other = issuer.sign(LINK, Direction.EGRESS, 0, record_style)
    assert cert_other.verify(key, SESSION) and cert_other.digest != c0.head.digest()
    swapped = {**doc, "certificate": {"key": cert_other.key_id.hex(), "digest": cert_other.digest.hex(),
                                      "signature": cert_other.signature.hex()}}
    with pytest.raises(ValueError, match="doesn't sign its head"):
        W.check_registration(swapped, key, SESSION, GRID, DIGEST, LINK, Direction.EGRESS, 0)


def test_the_layout_is_schedule_checked_but_not_bound_by_the_root(tmp_path):
    """A disagreeing layout is refused by the field check; but the layout is in no signed field. The head's encoding has
    no layout, so the SAME head digest and certificate stand whatever the layout — its integrity rests entirely on
    `check_registration`'s equality with the schedule (informational, no exploit)."""
    issuer = _issuer(tmp_path)
    c0 = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, c0)
    key = issuer.signer.public_key()
    doc = W.registration(c0.head, cert, GRID)
    for bad in ({**doc["layout"], "slots": doc["layout"]["slots"] + 1},
                {**doc["layout"], "first_bucket": doc["layout"]["first_bucket"] + 1},
                {**doc["layout"], "buckets": doc["layout"]["buckets"] + 1}):
        with pytest.raises(ValueError, match="layout"):
            W.check_registration({**doc, "layout": bad}, key, SESSION, GRID, DIGEST, LINK, Direction.EGRESS, 0)
    # the head's signed bytes carry no layout: changing the schedule's layout does not change the head digest or cert.
    head_bytes = c0.head.encode()
    assert len(head_bytes) == 2 + len(LINK) + 1 + 8 + 64 + 4 + 4 + 64 + 64  # link, dir, window, params, B, N, domain, root


def test_the_wiring_reads_are_unconstrained_prover_witness(tmp_path):
    """MAIN FINDING: the registration binds the committed rows and their positions, not which unit/connection each
    belongs to. `check_registration` never reads `reads`, and `port` accepts any in-range positions, each of which opens
    the root. So a prover wires any unit to any row of the correct link-window: a foreign connection's cell, the same
    cell twice, or an empty/padding slot (a cell no connection sent)."""
    issuer = _issuer(tmp_path)
    c0 = _commit(0, Direction.EGRESS)
    cert = _attest(issuer, c0)
    key = issuer.signer.public_key()
    doc = W.registration(c0.head, cert, GRID)
    head, dom = W.check_registration(doc, key, SESSION, GRID, DIGEST, LINK, Direction.EGRESS, 0)
    honest = [A_POS[0], A_POS[1]]       # the two units read connection A's cells (the truth)
    crossed = [B_POS[0], A_POS[0]]      # unit 0 reads connection B's cell instead
    dup = [A_POS[0], A_POS[0]]          # two units read the same cell
    to_pad = [PAD_POS, B_POS[0]]        # a unit reads an empty slot: a cell no connection sent
    for reads in (honest, crossed, dup, to_pad):
        p = W.port(head, reads)
        assert p["reads"] == reads and p["root"] == head.root.hex() and p["value"] == "warden/egress/0"
        for pos in reads:  # the prover furnishes a valid opening for whatever position it wired a unit to
            assert W.opens(dom, head.root, pos, W.bc(c0.rows[pos], c0.salts[pos]), c0.path(pos))
    # the channel: one of N positions a read (or None), bounded only by N.
    served_N = 600 * (8 * 2)  # T=600, lanes=8, r=2 -> S=16, N=9600
    assert N == PARAMS.window * PARAMS.lanes * PARAMS.slots
    assert round(math.log2(N), 1) == 6.6
    assert round(math.log2(served_N), 1) == 13.2
