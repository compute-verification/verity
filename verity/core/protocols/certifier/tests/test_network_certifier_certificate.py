"""Network certificates (record v2 on SHA-512): the layout restated from PROTOCOL.md, the frame tree, what the digest
and the signed message bind, the signer's refusal failing closed, and the certified audit's verdicts against the raw
audit's on every scenario of the audit and commitment tests."""

import hashlib
import itertools
import shutil
import struct

import pytest

from verity.ml.commitments.identity import tagged_sha512
from verity.ml.crypto import lms
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.certificate import (
    STATUS_CODES, Certificate, CertifiedRecord, Direction, Issuer, LmsSigner, RecordHead, certificate_digest,
    certificate_message, depth, empty, frames_root, open_frame, parameters_digest, real_frames, record_head,
    verify_frame)
from verity.core.protocols.certifier.commitment import commit
from verity.core.protocols.certifier.grid import COMPLETE, GridParams, Reason, Record, Status, Certifier, fill
from verity.core.protocols.certifier.schedule import ClockSync, run_shaper

from test_network_certifier_audit import FRAMES, P, SYNC_SET, SYNCS, honest_ready, ingress_run
import signers

EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
PARAMS = bytes(range(64))
SESSION, OTHER_SESSION = signers.SESSION, signers.OTHER_SESSION


def block(data: bytes) -> bytes:
    return data + bytes(128 - len(data))


def test_the_record_is_the_documented_bytes():
    """PROTOCOL.md's layout, rebuilt field by field with struct, and the tree from its three messages."""
    params = GridParams(buckets=3, allocation=2, frame_bytes=2, bucket_ms=50)
    grid = fill([b"ab", b"cd", b"ef"], [2, 0, 1], 2)
    head = record_head("lnk", INGRESS, 7, grid, Status(Reason.DROPPED), params, PARAMS, deadline=False)

    def leaf(f):
        return hashlib.sha512(block(b"verity-network-certifier/frame/v2\0" + struct.pack(">I", 2)) + f).digest()

    def node(a, b):
        return hashlib.sha512(block(b"verity-network-certifier/node/v2\0") + a + b).digest()

    e0 = hashlib.sha512(block(b"verity-network-certifier/empty/v2\0")).digest()
    e1 = node(e0, e0)
    assert depth(params) == 3 and empty(1) == e1
    root = node(node(node(leaf(b"ab"), leaf(b"cd")), node(leaf(b"ef"), e0)), node(e1, e1))
    want = (struct.pack(">H", 3) + b"lnk" + bytes([1]) + struct.pack(">Q", 7) + struct.pack(">IIII", 3, 2, 2, 50)
            + bytes([4, 0]) + PARAMS + struct.pack(">III", 2, 0, 1) + root)
    assert head.root == root and head.encode() == want
    assert head.digest() == tagged_sha512("verity-network-certifier/record/v2", want)
    assert head.digest().hex() == ("545323d2d0e56bd1b5da2f725e82dba77c531e7bc9b1949dd9a3524ed38dc55d"
                                   "c94240506a7f25e288353dc17722b0b2d696904a80560be830cce0f309171f40")


def test_status_codes_cover_every_reason_once():
    assert set(STATUS_CODES) == {None, *Reason} and sorted(STATUS_CODES.values()) == list(range(len(Reason) + 1))


def test_the_digest_is_injective_on_left_packed_grids():
    """Every left-packed grid on T = 2, r = 2 over a 2-frame alphabet has its own record digest, padding elided."""
    params = GridParams(buckets=2, allocation=2, frame_bytes=1)
    seen = {}
    for counts in itertools.product(range(3), repeat=2):
        for frames in itertools.product([b"a", b"b"], repeat=sum(counts)):
            grid = fill(list(frames), counts, 2)
            d = record_head("l", EGRESS, 0, grid, COMPLETE, params, PARAMS).digest()
            assert seen.setdefault(d, grid) == grid
    assert len(seen) == sum(2 ** (a + b) for a, b in itertools.product(range(3), repeat=2))


def test_the_head_binds_every_field():
    params = GridParams(buckets=2, allocation=2, frame_bytes=1)
    grid = fill([b"x", b"y"], [1, 1], 2)
    base = record_head("l", EGRESS, 0, grid, COMPLETE, params, PARAMS)
    variants = [
        record_head("m", EGRESS, 0, grid, COMPLETE, params, PARAMS),
        record_head("l", INGRESS, 0, grid, COMPLETE, params, PARAMS),
        record_head("l", EGRESS, 1, grid, COMPLETE, params, PARAMS),
        record_head("l", EGRESS, 0, grid + (fill([], [0], 2)[0],), COMPLETE,
                    GridParams(buckets=3, allocation=2, frame_bytes=1), PARAMS),
        record_head("l", EGRESS, 0, tuple(row + (None,) for row in grid), COMPLETE,
                    GridParams(buckets=2, allocation=3, frame_bytes=1), PARAMS),
        record_head("l", EGRESS, 0, grid, COMPLETE, GridParams(buckets=2, allocation=2, frame_bytes=1, bucket_ms=50),
                    PARAMS),
        record_head("l", EGRESS, 0, grid, COMPLETE, params, PARAMS, deadline=False),
        record_head("l", EGRESS, 0, grid, COMPLETE, params, bytes(64)),
        record_head("l", EGRESS, 0, fill([b"x", b"y"], [2, 0], 2), COMPLETE, params, PARAMS),
        record_head("l", EGRESS, 0, fill([b"y", b"x"], [1, 1], 2), COMPLETE, params, PARAMS),
        *(record_head("l", EGRESS, 0, grid, Status(r), params, PARAMS) for r in Reason),
    ]
    two_byte = GridParams(buckets=2, allocation=2, frame_bytes=2)
    variants.append(record_head("l", EGRESS, 0, fill([b"x\0", b"y\0"], [1, 1], 2), COMPLETE, two_byte, PARAMS))
    digests = {base.digest(), *(v.digest() for v in variants)}
    assert len(digests) == 1 + len(variants)


def test_padding_and_packing():
    params = GridParams(buckets=1, allocation=4, frame_bytes=1)
    with pytest.raises(ValueError, match="left-packed"):
        real_frames(((None, b"a", None, None),), params)
    with pytest.raises(ValueError, match="frames are 1 bytes"):
        frames_root([b"aa"], params)
    assert frames_root([], params) == empty(depth(params))


def test_every_frame_opens_at_one_depth_and_nothing_else_does():
    """Frame k opens with D = ⌈log2(T·r)⌉ siblings at every occupancy; another frame, index or path fails."""
    params = GridParams(buckets=3, allocation=3, frame_bytes=1)
    assert depth(params) == 4
    for n in range(10):
        frames = [bytes([k]) for k in range(n)]
        root = frames_root(frames, params)
        for k in range(n):
            path = open_frame(frames, params, k)
            assert len(path) == 4 and verify_frame(root, params, k, frames[k], path)
            assert not verify_frame(root, params, k, b"\xff", path)
            assert not verify_frame(root, params, k ^ 1, frames[k], path)
            assert not verify_frame(root, params, k, frames[k], path[:-1])


def test_parameters_digest_binds_queue_capacity_and_sync_set():
    base = parameters_digest(P, SYNC_SET)
    assert len(base) == 64 and base != parameters_digest(P)
    assert base != parameters_digest(GridParams(buckets=4, allocation=2, frame_bytes=1, queue_capacity=8), SYNC_SET)
    assert base != parameters_digest(P, SYNC_SET.__class__((0, 1), (1, 2)))


def certify_all(issuer, records, direction=EGRESS, params=P, sync_set=SYNC_SET):
    parameters = parameters_digest(params, sync_set if direction is EGRESS else None)
    return [c for r in records if (c := issuer.certify(r, direction, params, parameters)) is not None]


def egress_verdicts(records, key=None, syncs=SYNCS, link="responses", session=SESSION):
    return [v.violation for v in Audit().check_link(link, records, FRAMES, syncs, SYNC_SET, P, key=key,
                                                    session=session)]


def ingress_verdicts(records, windows, key=None, session=SESSION):
    return [v.violation for v in Audit().check_ingress_link("requests", records, windows, key=key, params=P,
                                                            session=session)]


def certified(raw):
    """The raw audit's verdicts as the certified audit gives them: the issuer certifies no incomplete record, so a
    window the raw audit rejects on its status has no certificate and is a `no-verdict`."""
    return ["no-verdict" if v is not None and v.startswith("status:") else v for v in raw]


def scenarios():
    """The audit and commitment tests' egress runs: honest, late, tampered, faulted, duplicated, missing a window,
    and out-of-set advice with and without a tampered later window."""
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    late_ready = honest_ready()
    late_ready[3] = 7
    late = run_shaper(FRAMES, late_ready, SYNCS, Certifier("responses", P))
    rows = [list(row) for row in honest[0].grid]
    rows[0][0] = b"\xff"
    tampered = [Record("responses", 0, tuple(tuple(r) for r in rows), COMPLETE)] + honest[1:]
    w = Certifier("responses", P)
    faulted = run_shaper(FRAMES, honest_ready(), SYNCS[:1], w)
    w.fault(Reason.MONITOR_UNAVAILABLE)
    for _ in range(P.buckets):
        w.tick()
    faulted.append(w.close_window())
    duplicated = tampered[:1] + honest
    declared = [SYNCS[0], ClockSync(3, 1), SYNCS[2]]
    advised = run_shaper(FRAMES, honest_ready(), declared, Certifier("responses", P))
    rows = [list(row) for row in advised[2].grid]
    rows[2][0] = b"\xff"
    advised_tampered = advised[:2] + [Record("responses", 2, tuple(tuple(r) for r in rows), advised[2].status)]
    return {"honest": (honest, SYNCS), "late": (late, SYNCS), "tampered": (tampered, SYNCS),
            "faulted": (faulted, SYNCS), "duplicated": (duplicated, SYNCS), "missing": (honest[:2], SYNCS),
            "advice": (advised, declared), "advice-tampered": (advised_tampered, declared)}


def test_certified_audit_matches_raw_audit_on_every_egress_scenario(tmp_path):
    """The difftest on the test vectors: with an honest issuer, every egress scenario gets the raw audit's verdicts,
    rejections included (a status one as a `no-verdict`), and the committed (v1) audit agrees."""
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    seen = set()
    for name, (records, syncs) in scenarios().items():
        raw = egress_verdicts(records, syncs=syncs)
        assert egress_verdicts(certify_all(Issuer(signer, SESSION), records), key, syncs) == certified(raw), name
        assert egress_verdicts([commit(r, P) for r in records], syncs=syncs) == raw, name
        seen.update(raw)
    assert seen == {None, "mismatch", "status:monitor-unavailable", "no-verdict", "duplicate", "advice"}


def test_certified_audit_matches_raw_audit_with_two_links_and_one_clock(tmp_path):
    """One clock per spatial unit: a second link with another in-set sync in window 1 is advice there, certified too."""
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    other = [SYNCS[0], ClockSync(2, 1), SYNCS[2]]
    responses = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    tokens = run_shaper(FRAMES, honest_ready(), other, Certifier("tokens", P))
    issuer = Issuer(signer, SESSION)
    for recs in ((responses, tokens), (certify_all(issuer, responses), certify_all(issuer, tokens))):
        audit = Audit()
        audit.check_link("responses", recs[0], FRAMES, SYNCS, SYNC_SET, P, key=key, session=SESSION)
        audit.check_link("tokens", recs[1], FRAMES, other, SYNC_SET, P, key=key, session=SESSION)
        assert [(v.link, v.window, v.violation) for v in audit.violations] == [("tokens", 1, "advice")]


def test_certified_audit_matches_raw_audit_on_ingress(tmp_path):
    """Ingress: different timings are both accepted, and a duplicate and a missing window are the raw audit's
    violations, certified; an incomplete status, certified nothing, is a `no-verdict`."""
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    early = ingress_run([[2, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    late = ingress_run([[0, 0, 0, 2], [0, 0, 1, 1], [0, 0, 0, 3]])
    faulted = ingress_run([[1, 0, 0, 0], [0, 2, 0, 0]], faults={1: Reason.DELIVERY_REFUSED})
    for records, windows in ((early, 3), (late, 3), ([faulted[0], *faulted], 3), (faulted, 2)):
        raw = ingress_verdicts(records, windows)
        assert ingress_verdicts(certify_all(Issuer(signer, SESSION), records, INGRESS), windows, key) == certified(raw)
    assert ingress_verdicts([faulted[0], *faulted], 3) == ["duplicate", "status:delivery-refused", "no-verdict"]


def test_an_honest_window_is_complete_so_the_issuer_refuses_nothing_honest(tmp_path):
    """Honest runs close every window complete, egress and ingress at either timing, so the issuer, which certifies
    only a complete record, certifies every window and refuses none."""
    signer = signers.signer(tmp_path)
    egress = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    early = ingress_run([[2, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    late = ingress_run([[0, 0, 0, 2], [0, 0, 1, 1], [0, 0, 0, 3]])
    issuer = Issuer(signer, SESSION)
    for records, direction in ((egress, EGRESS), (early, INGRESS), (late, INGRESS)):
        assert all(r.status.complete for r in records)
        assert len(certify_all(issuer, records, direction)) == len(records)
    assert issuer.refused == []


def test_a_certificate_cannot_move_to_another_link_window_or_key(tmp_path):
    """Moving a certificate to another window, link or direction, editing its head, signing with another key or
    forging the signature is a `signature` violation, and only in that window."""
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    certified = certify_all(Issuer(signer, SESSION), honest)
    c0, c1 = certified[0], certified[1]

    def with_window_0(rec: CertifiedRecord, *, head=True, cert=True) -> CertifiedRecord:
        h = RecordHead(**{f: getattr(rec.head, f) for f in RecordHead.__slots__} | {"window": 0}) if head else rec.head
        c = Certificate(**{f: getattr(rec.certificate, f) for f in Certificate.__slots__} | {"window": 0}) if cert \
            else rec.certificate
        return CertifiedRecord(h, rec.salt, c)

    assert egress_verdicts([with_window_0(c1), *certified[1:]], key) == ["signature", None, None]
    assert egress_verdicts([with_window_0(c1, cert=False), *certified[1:]], key) == ["signature", None, None]
    # the head still says window 1, so the moved certificate is a second record there
    assert egress_verdicts([with_window_0(c1, head=False), *certified[1:]], key) == ["no-verdict", "duplicate", None]
    relabelled = [CertifiedRecord(
        RecordHead(**{f: getattr(c.head, f) for f in RecordHead.__slots__} | {"link": "tokens"}), c.salt,
        Certificate(**{f: getattr(c.certificate, f) for f in Certificate.__slots__} | {"link": "tokens"}))
        for c in certified]
    assert egress_verdicts(relabelled, key, link="tokens") == ["signature"] * 3
    zeros = {"signature": bytes(len(c0.certificate.signature))}
    forged = CertifiedRecord(c0.head, c0.salt,
                             Certificate(**{f: getattr(c0.certificate, f) for f in Certificate.__slots__} | zeros))
    assert egress_verdicts([forged, *certified[1:]], key) == ["signature", None, None]
    (tmp_path / "stranger").mkdir()
    stranger = signers.signer(tmp_path / "stranger", seed=b"other")
    assert egress_verdicts(certify_all(Issuer(stranger, SESSION), honest), key) == ["signature"] * 3
    assert egress_verdicts(certified, key) == [None, None, None]


def test_an_incomplete_record_gets_no_certificate_and_a_relayed_status_is_bound(tmp_path):
    """The issuer certifies only a complete record: a faulted one is listed in `refused`, with no session started and
    no leaf spent, and its window is a `no-verdict`. The status is the signed head's: a faulted head signed anyway (by
    `sign`, as a forger would have to) that a relay edits to complete breaks the certificate."""
    signer = signers.signer(tmp_path)
    w = Certifier("requests", P)
    w.fault(Reason.DROPPED)
    for _ in range(P.buckets):
        w.tick()
    record = w.close_window()
    issuer = Issuer(signer, SESSION)
    key = signer.public_key()
    assert certify_all(issuer, [record], INGRESS) == [] and not issuer.started
    assert issuer.refused == [("requests", INGRESS, 0)]
    assert ingress_verdicts([], 1, key) == ["no-verdict"]
    head = record_head("requests", INGRESS, 0, record.grid, record.status, P, parameters_digest(P, None))
    salt = bytes(64)
    faulted = CertifiedRecord(head, salt, issuer.sign("requests", INGRESS, 0, certificate_digest(salt, head.digest())))
    lying = CertifiedRecord(RecordHead(**{f: getattr(faulted.head, f) for f in RecordHead.__slots__}
                                       | {"status": COMPLETE}), faulted.salt, faulted.certificate)
    assert ingress_verdicts([faulted], 1, key) == ["status:dropped"]
    assert ingress_verdicts([lying], 1, key) == ["signature"]


def test_a_record_past_the_deadline_is_no_verdict_and_other_parameters_a_mismatch(tmp_path):
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    issuer = Issuer(signer, SESSION)
    parameters = parameters_digest(P, SYNC_SET)
    late = [issuer.certify(honest[0], EGRESS, P, parameters, deadline=False)] + certify_all(issuer, honest[1:])
    assert egress_verdicts(late, key) == ["no-verdict", None, None]
    other_set = certify_all(issuer, honest, sync_set=SYNC_SET.__class__((0, 1), (1, 2)))
    assert egress_verdicts(other_set, key) == ["mismatch"] * 3
    ingress = ingress_run([[1, 0, 0, 0]])
    assert ingress_verdicts(certify_all(issuer, ingress, INGRESS), 1, key) == [None]
    assert ingress_verdicts(certify_all(issuer, ingress, INGRESS, sync_set=None), 1, key) == [None]
    q8 = GridParams(buckets=4, allocation=2, frame_bytes=1, queue_capacity=8)
    wrong = [issuer.certify(ingress[0], INGRESS, P, parameters_digest(q8))]
    assert ingress_verdicts(wrong, 1, key) == ["mismatch"]
    egress_as_ingress = certify_all(issuer, ingress, EGRESS)
    assert ingress_verdicts(egress_as_ingress, 1, key) == ["no-verdict"]


def test_certified_records_need_the_key_and_ingress_its_parameters(tmp_path):
    signer = signers.signer(tmp_path)
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    certified = certify_all(Issuer(signer, SESSION), honest)
    with pytest.raises(ValueError, match="public key"):
        egress_verdicts(certified)
    with pytest.raises(ValueError, match="session"):
        egress_verdicts(certified, signer.public_key(), session=None)
    ingress = certify_all(Issuer(signer, SESSION), ingress_run([[1, 0, 0, 0]]), INGRESS)
    with pytest.raises(ValueError, match="params"):
        Audit().check_ingress_link("requests", ingress, 1, key=signer.public_key(), session=SESSION)


def test_a_signer_with_missing_or_behind_state_refuses_and_the_window_fails_closed(tmp_path):
    """No state: nothing is signed and every window is `no-verdict`. State restored from an old copy: the signer
    refuses rather than reuse a leaf, and the windows from then on are `no-verdict` until the state is current."""
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    old, current = tmp_path / "old-state.json", tmp_path / "current-state.json"
    shutil.copy(signer.state, old)
    issuer = Issuer(signer, SESSION)
    first = issuer.certify(honest[0], EGRESS, P, parameters_digest(P, SYNC_SET))
    shutil.copy(signer.state, current)
    shutil.copy(old, signer.state)
    rest = certify_all(issuer, honest[1:])
    assert rest == [] and issuer.refused == [("responses", EGRESS, 1), ("responses", EGRESS, 2)]
    assert egress_verdicts([first], key) == [None, "no-verdict", "no-verdict"]
    assert signers.used(signer) == [0]
    shutil.copy(current, signer.state)
    assert egress_verdicts([first, *certify_all(issuer, honest[1:])], key) == [None, None, None]
    assert signers.used(signer) == [0, 1, 2]
    signer.state.unlink()
    missing = Issuer(signer, SESSION)
    assert certify_all(missing, honest) == [] and len(missing.refused) == 3
    assert egress_verdicts([], key) == ["no-verdict"] * 3


def test_a_second_signer_on_one_state_or_below_its_floor_refuses(tmp_path):
    """The leaf state is held by the signer that opened it: a second signer on it refuses until the first closes, and a
    signer whose floor is past the state (a restore from before a published leaf) refuses outright."""
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    parameters = parameters_digest(P, SYNC_SET)
    first = signers.signer(tmp_path)
    assert Issuer(first, SESSION).certify(honest[0], EGRESS, P, parameters) is not None
    second = Issuer(LmsSigner(first.key, first.state), OTHER_SESSION)
    assert second.certify(honest[1], EGRESS, P, parameters) is None and second.refused == [("responses", EGRESS, 1)]
    first.close()
    floored = Issuer(LmsSigner(first.key, first.state, floor=5), OTHER_SESSION)
    assert floored.certify(honest[1], EGRESS, P, parameters) is None
    again = second.certify(honest[1], EGRESS, P, parameters)
    assert again is not None and again.verify(first.public_key(), OTHER_SESSION)
    assert lms.Signature.decode(signers.SMALL, again.certificate.signature).q == 1


def test_a_session_already_started_is_refused_and_its_windows_fail_closed(tmp_path):
    """The device runs once per session id: a second run of the key under a session its state has started, after a
    restart, signs nothing and every window is `no-verdict`; under a fresh session it signs from the next leaf."""
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    first = signers.signer(tmp_path)
    key = first.public_key()
    assert egress_verdicts(certify_all(Issuer(first, SESSION), honest), key) == [None, None, None]
    first.close()
    replay = Issuer(LmsSigner(first.key, first.state), SESSION)
    assert certify_all(replay, honest) == [] and len(replay.refused) == 3 and not replay.started
    assert signers.used(first) == [0, 1, 2]
    fresh = Issuer(replay.signer, OTHER_SESSION)
    assert egress_verdicts(certify_all(fresh, honest), key, session=OTHER_SESSION) == [None, None, None]
    assert signers.used(first) == [0, 1, 2, 3, 4, 5]


def test_a_certificate_from_another_session_is_a_signature_violation(tmp_path):
    """The cross-session replay: certificates the key signed in another session of the certifier, honest ones of the
    same link-windows, are a `signature` violation in this session, every window; in their own they are accepted."""
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    other = certify_all(Issuer(signer, OTHER_SESSION), honest)
    assert egress_verdicts(other, key, session=OTHER_SESSION) == [None, None, None]
    assert egress_verdicts(other, key, session=SESSION) == ["signature"] * 3
    ingress = certify_all(Issuer(signer, OTHER_SESSION), ingress_run([[1, 0, 0, 0]]), INGRESS)
    assert ingress_verdicts(ingress, 1, key, session=SESSION) == ["signature"]
    assert ingress_verdicts(ingress, 1, key, session=OTHER_SESSION) == [None]
    assert not other[0].verify(key, bytes(31)) and not other[0].verify(key, bytes(33))


def test_the_certificate_signs_the_record_digest_salted_by_the_certifier(tmp_path):
    """The signed digest is SHA-512("verity/net-cert/v1" ‖ salt_c ‖ R) under 64 bytes of the certifier's own coins,
    fresh per record: one record certified twice is signed under two digests, neither of them R. The host's check
    reads the certificate alone; a record handed with another salt, or a salt of another length, is refused."""
    signer = signers.signer(tmp_path)
    honest = run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))
    fixed = certify_all(Issuer(signer, SESSION, coins=lambda n: b"\x07" * n), honest[:1])[0]
    R = fixed.head.digest()
    assert fixed.salt == b"\x07" * 64 and certificate_digest(fixed.salt, R) == fixed.certificate.digest
    assert fixed.certificate.digest == hashlib.sha512(b"verity/net-cert/v1" + b"\x07" * 64 + R).digest() != R
    assert fixed.verify(signer.public_key(), SESSION)
    (tmp_path / "default").mkdir()
    other = signers.signer(tmp_path / "default", seed=b"other")
    key = other.public_key()
    twice = certify_all(Issuer(other, SESSION), honest[:1] * 2)
    assert twice[0].head == twice[1].head and twice[0].salt != twice[1].salt
    assert twice[0].certificate.digest != twice[1].certificate.digest
    assert all(c.verify(key, SESSION) and c.certificate.verify(key, SESSION) for c in twice)
    swapped = CertifiedRecord(twice[0].head, twice[1].salt, twice[0].certificate)
    assert swapped.certificate.verify(key, SESSION) and not swapped.verify(key, SESSION)
    with pytest.raises(ValueError, match="64 bytes"):
        CertifiedRecord(fixed.head, bytes(63), fixed.certificate)
    with pytest.raises(ValueError, match="63 bytes"):
        Issuer(signer, SESSION, coins=lambda n: bytes(n - 1)).certify(honest[0], EGRESS, P, PARAMS)


def test_certificate_message_binds_key_session_link_direction_window_and_digest():
    d, s = bytes(64), SESSION
    base = certificate_message(b"I" * 16, s, "l", EGRESS, 0, d)
    others = [certificate_message(b"J" * 16, s, "l", EGRESS, 0, d), certificate_message(b"I" * 16, s, "m", EGRESS, 0, d),
              certificate_message(b"I" * 16, s, "l", INGRESS, 0, d), certificate_message(b"I" * 16, s, "l", EGRESS, 1, d),
              certificate_message(b"I" * 16, s, "l", EGRESS, 0, b"\1" + bytes(63)),
              certificate_message(b"I" * 16, OTHER_SESSION, "l", EGRESS, 0, d)]
    assert len(base) == 64 and len({base, *others}) == 7
    assert base == lms.certificate_message(b"I" * 16, s, "l", "egress", 0, d)
    tag = lms.ROLES["network-certificate"].encode()
    field = lambda b: struct.pack(">Q", len(b)) + b  # noqa: E731
    context = field(s) + field(b"l") + field(b"egress") + bytes(8)
    assert tag == b"verity/network-certificate-message/v2"
    assert base == hashlib.sha512(field(tag) + b"I" * 16 + field(context) + field(d)).digest()
    for n in (31, 33):
        with pytest.raises(ValueError, match="session id is 32 bytes"):
            certificate_message(b"I" * 16, bytes(n), "l", EGRESS, 0, d)
