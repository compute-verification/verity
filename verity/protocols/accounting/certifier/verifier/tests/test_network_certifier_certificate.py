"""Network certificates (record v2 on SHA-512): the layout restated from PROTOCOL.md, the frame tree, what the digest
and the signed message bind, the signer's refusal failing closed, and the certified audit's verdicts against the raw
audit's on every scenario of the audit and commitment tests. The mark certificate: what it signs, each refusal of
its opening by name, the issuer signing none when its signer refuses or its coins repeat, and no second one of a mark
it has certified. A certificate's bytes: the layout, its inverse, the same bytes inside both record encoders, and each
refusal by name."""

import hashlib
import itertools
import shutil
import struct
from dataclasses import replace

import pytest

from verity.commitments import domains
from verity.commitments.identity import tagged_sha512
from verity.commitments.crypto import lms
from verity.protocols.accounting.certifier.verifier import lean_audit, service
from verity.protocols.accounting.certifier.verifier.audit import Audit
from verity.protocols.accounting.certifier.verifier.certificate import (
    MARK_TAG, SALT_BYTES, STATUS_CODES, Certificate, CertifiedMark, CertifiedRecord, Direction, Issuer, LmsSigner,
    MarkRefused, RecordHead, certificate_digest, certificate_message, depth, empty, frames_root, mark_digest, marks_link, open_frame,
    parameters_digest, real_frames, record_head, verify_frame)
from verity.protocols.accounting.certifier.verifier.commitment import commit
from verity.protocols.accounting.certifier.verifier.grid import COMPLETE, GridParams, Reason, BlockGrid, Status, Certifier, fill
from verity.protocols.accounting.certifier.verifier.schedule import ClockSync, run_shaper

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
    tampered = [BlockGrid("responses", 0, tuple(tuple(r) for r in rows), COMPLETE)] + honest[1:]
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
    advised_tampered = advised[:2] + [BlockGrid("responses", 2, tuple(tuple(r) for r in rows), advised[2].status)]
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
    same execution phases, are a `signature` violation in this session, every window; in their own they are accepted."""
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


MARK, OTHER_MARK = hashlib.sha512(b"a mark").digest(), hashlib.sha512(b"another mark").digest()


def fresh_coins():
    """Coins that never repeat: the i-th salt is 64 bytes of i, from 1."""
    salts = itertools.count(1)
    return lambda n: bytes([next(salts)]) * n


def test_a_mark_is_signed_salted_under_its_own_tag_on_the_marks_link_name(tmp_path):
    """A mark certificate signs SHA-512("verity-network-certifier/mark/v1" ‖ salt_c ‖ mark) under 64 bytes of the
    certifier's own coins, as an egress certificate on `<link>-marks` at the execution phase its caller gave, over the
    message every certificate signs, with one leaf. Its opening holds, and its execution phase is the certificate's.
    The issuer keeps the mark, and asked for it again it raises `MarkRefused` and signs no second certificate."""
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    issuer = Issuer(signer, SESSION, coins=fresh_coins())
    m = issuer.mark("responses", 5, MARK)
    c = m.certificate
    assert (m.mark, m.salt, m.window) == (MARK, b"\x01" * 64, 5)
    assert (c.key_id, c.link, c.direction, c.window) == (key.key_id, "responses-marks", EGRESS, 5)
    assert marks_link("responses") == "responses-marks"
    assert MARK_TAG == domains.CERTIFIER_MARK.encode() == b"verity-network-certifier/mark/v1"
    assert c.digest == mark_digest(m.salt, MARK) == hashlib.sha512(MARK_TAG + m.salt + MARK).digest()
    assert c.digest not in (MARK, certificate_digest(m.salt, MARK))
    assert c.message(SESSION) == lms.certificate_message(key.key_id, SESSION, "responses-marks", "egress", 5, c.digest)
    assert c.verify(key, SESSION) and m.refusal(key, SESSION, "responses") is None
    assert m.verify(key, SESSION, "responses")
    assert signers.used(signer) == [0] and issuer.refused == [] and issuer.mark_salts == {m.salt}
    assert issuer.marked == {MARK}
    with pytest.raises(MarkRefused, match="it certifies a mark once"):
        issuer.mark("responses", 9, MARK)
    assert signers.used(signer) == [0] and issuer.marked == {MARK}


def test_the_issuer_signs_no_second_mark_of_a_message(tmp_path):
    """The issuer marks a message at most once: a mark it has certified is refused when asked for again, at the same
    execution phase, at a later one and for another link, by `MarkRefused`, and nothing is handed back. A refused call
    takes no coins, uses no leaf and lists nothing in `refused`, so the next mark, another message, is certified under
    the next salt at the next leaf, and is then refused too. A restarted signer on the key's state signs no mark in
    the session the state has started, the certified one or another: the session's marks are the first signer's."""
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    taken = []

    def coins(n):
        taken.append(n)
        return bytes([len(taken)]) * n

    issuer = Issuer(signer, SESSION, coins=coins)
    m = issuer.mark("responses", 5, MARK)
    assert m.verify(key, SESSION, "responses") and (taken, signers.used(signer)) == ([64], [0])
    for link, w in (("responses", 5), ("responses", 9), ("requests", 5)):
        with pytest.raises(MarkRefused, match="no second certificate is signed over the same 64 bytes"):
            issuer.mark(link, w, MARK)
    assert (taken, signers.used(signer), issuer.refused) == ([64], [0], [])
    assert (issuer.marked, issuer.mark_salts) == ({MARK}, {m.salt})
    other = issuer.mark("responses", 9, OTHER_MARK)
    assert (other.mark, other.salt, other.window) == (OTHER_MARK, b"\x02" * 64, 9)
    assert other.verify(key, SESSION, "responses") and (taken, signers.used(signer)) == ([64, 64], [0, 1])
    assert issuer.marked == {MARK, OTHER_MARK}
    with pytest.raises(MarkRefused):
        issuer.mark("responses", 9, OTHER_MARK)
    assert signers.used(signer) == [0, 1]
    signer.close()
    restarted = Issuer(LmsSigner(signer.key, signer.state), SESSION, coins=fresh_coins())
    third = hashlib.sha512(b"a third mark").digest()
    assert restarted.mark("responses", 9, MARK) is None and restarted.mark("responses", 9, third) is None
    assert restarted.refused == [("responses-marks", EGRESS, 9)] * 2 and restarted.marked == set()
    assert signers.used(signer) == [0, 1]


def test_a_mark_that_is_not_bytes_is_signed_nowhere(tmp_path):
    """A mark of 64 bytes that are not `bytes` (a `bytearray`, a `memoryview`) is no mark `marked` can hold, so it passes
    by the once-only refusal; it takes a salt and is refused by `mark_digest` with `ValueError` before the signer is
    asked anything, before and after the same bytes are certified: no leaf is used, nothing is listed in `refused`,
    and neither the salt nor the mark is kept."""
    signer = signers.signer(tmp_path)
    taken = []

    def coins(n):
        taken.append(n)
        return bytes([len(taken)]) * n

    issuer = Issuer(signer, SESSION, coins=coins)
    for mark in (bytearray(MARK), memoryview(MARK)):
        with pytest.raises(ValueError, match="a mark is 64 bytes"):
            issuer.mark("responses", 5, mark)
    assert (len(taken), signers.used(signer), issuer.refused, issuer.marked, issuer.mark_salts) == (2, [], [], set(), set())
    m = issuer.mark("responses", 5, MARK)
    assert (m.salt, signers.used(signer), issuer.marked) == (b"\x03" * 64, [0], {MARK})
    with pytest.raises(ValueError, match="a mark is 64 bytes"):
        issuer.mark("responses", 9, bytearray(MARK))
    assert (len(taken), signers.used(signer), issuer.refused, issuer.mark_salts) == (4, [0], [], {m.salt})


def test_a_mark_certificate_is_refused_by_name(tmp_path):
    """A mark certificate's opening is refused with why: another mark, another salt or an altered digest (`opening`);
    read as another link's mark, or a certificate on the link's own name (`link`); an ingress certificate (`egress`);
    another key's (`key`); and a certificate relabelled to another link name or another execution phase, another
    session's and a forged signature (`signature`). A record's salted digest signed on the marks' link name opens as
    no mark."""
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    issuer = Issuer(signer, SESSION, coins=fresh_coins())
    m = issuer.mark("responses", 5, MARK)
    c = m.certificate

    def refusal(opening, link="responses", session=SESSION, key=key):
        return opening.refusal(key, session, link)

    assert refusal(m) is None
    assert refusal(replace(m, mark=OTHER_MARK)) == "opening"
    assert refusal(replace(m, salt=b"\x02" * 64)) == "opening"
    assert refusal(replace(m, certificate=replace(c, digest=mark_digest(m.salt, OTHER_MARK)))) == "opening"
    assert refusal(m, link="requests") == "link"
    assert refusal(replace(m, certificate=replace(c, link="requests-marks")), link="requests") == "signature"
    assert refusal(replace(m, certificate=replace(c, window=6))) == "signature"
    assert refusal(m, session=OTHER_SESSION) == "signature" and refusal(m, session=bytes(31)) == "signature"
    assert refusal(replace(m, certificate=replace(c, signature=bytes(len(c.signature))))) == "signature"
    assert not replace(m, mark=OTHER_MARK).verify(key, SESSION, "responses")
    (tmp_path / "stranger").mkdir()
    stranger = signers.signer(tmp_path / "stranger", seed=b"other")
    foreign = Issuer(stranger, SESSION, coins=fresh_coins()).mark("responses", 5, MARK)
    assert refusal(m, key=stranger.public_key()) == "key" and refusal(foreign) == "key"
    assert refusal(foreign, key=stranger.public_key()) is None
    salt = b"\x09" * 64
    digest = mark_digest(salt, MARK)
    unmarked = CertifiedMark(MARK, salt, issuer.sign("responses", EGRESS, 5, digest))
    ingress = CertifiedMark(MARK, salt, issuer.sign("responses-marks", INGRESS, 5, digest))
    record = CertifiedMark(MARK, salt, issuer.sign("responses-marks", EGRESS, 5, certificate_digest(salt, MARK)))
    assert all(x.certificate.verify(key, SESSION) for x in (unmarked, ingress, record))
    assert [refusal(x) for x in (unmarked, ingress, record)] == ["link", "egress", "opening"]
    for other in ({"mark": MARK[:-1]}, {"salt": m.salt + b"\0"}):
        with pytest.raises(ValueError, match="64 bytes"):
            replace(m, **other)


def test_the_issuer_signs_no_mark_when_its_signer_refuses_or_its_coins_repeat(tmp_path):
    """A signer whose leaf state is behind refuses: the issuer signs nothing, lists the marks' link name at the
    execution phase in `refused` and takes no salt, so the same coins certify the mark once the state is current.
    Coins that then give that salt again sign no second mark, and a mark or coins of another length sign none: no
    refusal spends a leaf."""
    signer = signers.signer(tmp_path)
    key = signer.public_key()
    old, current = tmp_path / "old-state.json", tmp_path / "current-state.json"
    shutil.copy(signer.state, old)
    issuer = Issuer(signer, SESSION, coins=lambda n: b"\x5a" * n)
    assert issuer.sign("responses", EGRESS, 0, bytes(64)) is not None
    shutil.copy(signer.state, current)
    shutil.copy(old, signer.state)
    assert issuer.mark("responses", 5, MARK) is None
    assert issuer.refused == [("responses-marks", EGRESS, 5)] and issuer.mark_salts == set() == issuer.marked
    assert signers.used(signer) == [0]
    shutil.copy(current, signer.state)
    m = issuer.mark("responses", 5, MARK)
    assert m.salt == b"\x5a" * 64 and m.verify(key, SESSION, "responses") and signers.used(signer) == [0, 1]
    with pytest.raises(ValueError, match="gave an earlier mark certificate's salt again"):
        issuer.mark("responses", 6, OTHER_MARK)
    with pytest.raises(ValueError, match="a mark is 64 bytes"):
        Issuer(signer, SESSION, coins=fresh_coins()).mark("responses", 6, MARK[:-1])
    with pytest.raises(ValueError, match="gave 63 bytes for a 64-byte salt"):
        Issuer(signer, SESSION, coins=lambda n: bytes(n - 1)).mark("responses", 6, MARK)
    assert signers.used(signer) == [0, 1] and issuer.refused == [("responses-marks", EGRESS, 5)]
    assert issuer.mark_salts == {m.salt} and issuer.marked == {MARK}


def layout(c: Certificate) -> bytes:
    """A certificate's bytes restated field by field: u16be len(I) ‖ I ‖ u16be len(link) ‖ link ‖ u8 direction ‖
    u64be execution phase ‖ digest (64) ‖ u32be len(signature) ‖ signature."""
    link = c.link.encode()
    return (struct.pack(">H", len(c.key_id)) + c.key_id + struct.pack(">H", len(link)) + link
            + struct.pack(">BQ", c.direction.code, c.window) + c.digest + struct.pack(">I", len(c.signature))
            + c.signature)


def one_of_each(tmp_path):
    """The signer's key, a certified record, and a certificate of each kind the issuer signs: a record's at each end,
    a clock's and a mark's."""
    signer = signers.signer(tmp_path)
    issuer = Issuer(signer, SESSION, coins=fresh_coins())
    record = certify_all(issuer, run_shaper(FRAMES, honest_ready(), SYNCS, Certifier("responses", P))[:1])[0]
    ingress = certify_all(issuer, ingress_run([[2, 0, 0, 0]]), INGRESS)[0]
    clock = issuer.salted_clock("responses", 0, SYNCS[0])
    mark = issuer.mark("responses", 3, MARK)
    return signer.public_key(), record, {"record": record.certificate, "ingress": ingress.certificate,
                                         "clock": clock.certificate, "mark": mark.certificate}


def test_a_certificates_bytes_are_the_layout_and_decode_is_its_inverse(tmp_path):
    """`Certificate.encode` is the layout field by field for a record's certificate at each end, a clock's and a
    mark's, and `Certificate.decode` gives the certificate back, which still verifies and encodes to the same bytes.
    Two certificates that differ in one field have different bytes."""
    key, _, kinds = one_of_each(tmp_path)
    assert {c.direction for c in kinds.values()} == {EGRESS, INGRESS}
    assert kinds["mark"].link == "responses-marks" and len({c.encode() for c in kinds.values()}) == 4
    for c in kinds.values():
        data = c.encode()
        assert data == layout(c) and len(data) == 2 + 16 + 2 + len(c.link) + 1 + 8 + 64 + 4 + len(c.signature)
        back = Certificate.decode(data)
        assert back == c and back.verify(key, SESSION) and back.encode() == data
        assert Certificate.decode(bytearray(data)) == c
    c = kinds["mark"]
    others = [replace(c, key_id=b"J" * 16), replace(c, link="responses"), replace(c, link="réponses-marks"),
              replace(c, direction=INGRESS), replace(c, window=c.window + 1), replace(c, window=2 ** 64 - 1),
              replace(c, digest=bytes(64)), replace(c, signature=c.signature[:-1]), replace(c, signature=b""),
              replace(c, key_id=b"")]
    assert len({x.encode() for x in [c, *others]}) == len(others) + 1
    assert all(Certificate.decode(x.encode()) == x and x.encode() == layout(x) for x in others)


def test_a_certificates_bytes_are_those_inside_both_record_encoders(tmp_path):
    """The commitment's leaf of a certified record (`service.encode`) is its head, its 64-byte salt and then the
    certificate, which is where the Lean reads it; the Lean audit's encoding of the record (`lean_audit.encode`) is
    a kind byte, the leaf's length and that leaf. The certificate's part of both is `Certificate.encode`, for a
    record's certificate at each end, a clock's and a mark's, and `Certificate.decode` reads it back from there."""
    _, record, kinds = one_of_each(tmp_path)
    head = record.head.encode()
    for c in kinds.values():
        held = replace(record, certificate=c)
        leaf = service.encode(held)
        assert leaf[:len(head)] == head and leaf[len(head):len(head) + SALT_BYTES] == record.salt
        assert leaf[len(head) + SALT_BYTES:] == c.encode()
        body = lean_audit.encode(held)
        assert body[0] == lean_audit.CERTIFIED and int.from_bytes(body[1:9], "big") == len(leaf)
        assert body[9:] == leaf and body[9 + len(head) + SALT_BYTES:] == c.encode()
        assert Certificate.decode(leaf[len(head) + SALT_BYTES:]) == c


def test_a_certificates_bytes_are_refused_by_name():
    """`Certificate.decode` refuses bytes left after the signature, bytes that end inside a fixed-width part, and a
    length that names more bytes than are left, each saying which; every proper prefix of a certificate's bytes is
    one of the last two. It refuses a direction code other than 0 and 1, a link that is not UTF-8 and anything that
    is not bytes: a number is not that many zero bytes. `encode` refuses a certificate the layout cannot hold."""
    c = Certificate(b"I" * 16, "responses-marks", EGRESS, 5, bytes(range(64)), b"signature")
    data = c.encode()
    assert Certificate.decode(data) == c and Certificate.decode(memoryview(data)) == c
    assert Certificate.decode(bytes(81)) == Certificate(b"", "", EGRESS, 0, bytes(64), b"")
    for other in (81, data.hex(), None, [data]):
        with pytest.raises(ValueError, match=f"a certificate's bytes are bytes, not a {type(other).__name__}"):
            Certificate.decode(other)
    for more in (b"\0", data):
        with pytest.raises(ValueError, match=f"{len(more)} bytes are left after a certificate's signature"):
            Certificate.decode(data + more)
    key_end, link_end = 2 + 16, 2 + 16 + 2 + len("responses-marks")
    ends = {0: "key id's length", 1: "key id's length", key_end: "link's length", key_end + 1: "link's length",
            link_end: "direction", link_end + 1: "execution phase", link_end + 8: "execution phase",
            link_end + 9: "digest", link_end + 9 + 63: "digest", link_end + 9 + 64: "signature's length",
            link_end + 9 + 64 + 3: "signature's length"}
    for n, part in ends.items():
        with pytest.raises(ValueError, match=f"a certificate's bytes end inside its {part}: "):
            Certificate.decode(data[:n])
    fits = {2: "key id", key_end - 1: "key id", key_end + 2: "link", link_end - 1: "link",
            link_end + 9 + 64 + 4: "signature", len(data) - 1: "signature"}
    for n, part in fits.items():
        with pytest.raises(ValueError, match=f"a certificate's {part} does not fit: "):
            Certificate.decode(data[:n])
    for n in range(len(data)):
        with pytest.raises(ValueError, match="end inside its|does not fit"):
            Certificate.decode(data[:n])
    longer = data[:link_end + 9 + 64] + struct.pack(">I", len(c.signature) + 1) + c.signature
    with pytest.raises(ValueError, match="a certificate's signature does not fit: it is 10 bytes by its length, and 9"):
        Certificate.decode(longer)
    with pytest.raises(ValueError, match="a certificate's key id does not fit: it is 65535 bytes"):
        Certificate.decode(b"\xff\xff" + data[2:])
    shorter = data[:link_end + 9 + 64] + struct.pack(">I", len(c.signature) - 1) + c.signature
    with pytest.raises(ValueError, match="1 bytes are left after a certificate's signature"):
        Certificate.decode(shorter)
    for code in (2, 255):
        with pytest.raises(ValueError, match=f"a certificate's direction code is 0 or 1, not {code}"):
            Certificate.decode(data[:link_end] + bytes([code]) + data[link_end + 1:])
    with pytest.raises(ValueError, match="a certificate's link is not UTF-8"):
        Certificate.decode(data[:key_end + 2] + b"\xff" + data[key_end + 3:])
    for bad, why in ((replace(c, digest=bytes(63)), "a certificate's digest is 64 bytes, not 63"),
                     (replace(c, digest=bytes(65)), "a certificate's digest is 64 bytes, not 65"),
                     (replace(c, window=-1), "a certificate's execution phase is an unsigned 64-bit integer"),
                     (replace(c, window=2 ** 64), "a certificate's execution phase is an unsigned 64-bit integer"),
                     (replace(c, window=True), "a certificate's execution phase is an unsigned 64-bit integer"),
                     (replace(c, key_id=bytes(2 ** 16)), "an unsigned 16-bit integer"),
                     (replace(c, link="l" * 2 ** 16), "an unsigned 16-bit integer"),
                     (replace(c, direction="egress"), "its direction a Direction"),
                     (replace(c, signature=bytearray(b"signature")), "signature are bytes")):
        with pytest.raises(ValueError, match=why):
            bad.encode()
