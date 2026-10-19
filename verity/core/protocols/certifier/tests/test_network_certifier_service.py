"""One link-window's network certificates through the proof service (N2): the round trip to an accepted, exhaustive
outcome, the public-items rule on the registration, the head check against the declared grid and the occupancy
profile, and each way a link-window fails closed, on injected coins and an injected clock. Every round's verdict is
held to the stand-in (`LinkWindows.stand_in`), the in-the-clear verdict from before the head check."""

import json
from dataclasses import replace

import pytest

from verity.ml.circuits import codec
from verity.ml.circuits.program import Program
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier.certificate import (CertifiedRecord, Direction, Issuer, LmsSigner,
                                                                          RecordHead, parameters_digest)
from verity.core.protocols.certifier.grid import Certifier, GridParams, Record, occupancy
from verity.core.protocols.certifier.schedule import SyncSet, run_shaper
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.frame_v3 import root_ref
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest
from verity.core.protocols.certifier.service import (HEADS, OUTPUTS, PUBLIC, SCHEMA, VALUE, LinkWindows, encode,
                                                                      rows)
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R

from test_network_certifier_audit import FRAMES, P as PARAMS, SYNC_SET, SYNCS, honest_ready, ingress_run
import signers

LINK = "responses"
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
DELTA = 2.0 ** -40
SALTS = [tuple(bytes([4 * w + k + 1]) * 192 for k in range(4)) for w in range(3)]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def egress_records():
    return run_shaper(FRAMES, honest_ready(), SYNCS, Certifier(LINK, PARAMS))


def ingress_records(arrivals=([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])):
    return [replace(r, link=LINK) for r in ingress_run([list(a) for a in arrivals])]


def honest_occupancy():
    """Each end's counts in each window of the honest run: the profile the placement declares."""
    return tuple(tuple(occupancy(r.grid) for r in records) for records in (egress_records(), ingress_records()))


class Run:
    """A link with its network certifier (an LMS signer on a small tree), the protocol's parameters, and the challenger's stream."""

    def __init__(self, tmp_path, *, proof_deadline=60.0, coins=b"\x07", occupancy=(None, None)) -> None:
        self.signer = signers.signer(tmp_path)
        self.issuer = Issuer(self.signer, signers.SESSION)
        self.lw = LinkWindows(LINK, PARAMS, SYNC_SET, self.signer.public_key(), "r-network-test", signers.SESSION,
                              proof_deadline=proof_deadline, occupancy=occupancy)
        self.clock = Clock()
        self.stream = self.lw.stream(budget=8, delta=DELTA, challenger="challenger-test", coins=lambda n: coins * n,
                                     clock=self.clock)
        self.previous = None

    def egress(self, record: Record, **kw) -> CertifiedRecord:
        return self.issuer.certify(record, EGRESS, PARAMS, parameters_digest(PARAMS, SYNC_SET), **kw)

    def ingress(self, record: Record, **kw) -> CertifiedRecord:
        return self.issuer.certify(record, INGRESS, PARAMS, parameters_digest(PARAMS), **kw)

    def honest(self, w: int):
        return self.lw.commit(w, self.egress(egress_records()[w]), self.ingress(ingress_records()[w]), SALTS[w])

    def register(self, opening) -> S.Registration:
        reg = self.lw.register(self.stream, self.lw.registration(opening, self.previous))
        self.previous = reg.digest
        return reg

    def round(self, opening, delivered=None) -> S.Outcome:
        """One registration through its outcome, held to the stand-in."""
        draw = self.stream.select(self.register(opening))
        assert self.stream.deliver(draw, [self.lw.session(delivered or opening, draw)]) is None
        out = self.lw.outcome(self.stream, draw, delivered or opening, FRAMES, SYNCS)
        assert self.lw.stand_in(delivered or opening, draw, FRAMES, SYNCS) == out.accepted
        return out


def reject(out: S.Outcome) -> bool:
    return not out.accepted and out.verdict.codes == ("reject",) and out.public == {} and out.exhaustive is None


def test_one_link_window_round_trip_is_the_exhaustive_fact(tmp_path):
    run = Run(tmp_path)
    opening = run.honest(0)
    reg = run.register(opening)
    assert reg.order == 0 and reg.receipt["received"] == "0" and reg.record["population"] == 1
    assert reg.record["roots"] == [{"port": VALUE, "root": run.lw.root(opening), **run.lw.ports(0)[VALUE]}]
    assert S.check_public(run.lw.spec, reg.record).ok
    draw = run.stream.select(reg)
    assert draw.units == [0] and draw.draw["law"] == "subset" and draw.draw["k"] == 1
    assert run.stream.deliver(draw, [run.lw.session(opening, draw)]) is None
    out = run.lw.outcome(run.stream, draw, opening, FRAMES, SYNCS)
    assert out.accepted and out.verdict.ok and out.exhaustive == S.Exhaustive(1) and out.profile is None
    assert out.terms == S.Terms(None) and out.public == {"verdict": 1, "deadline": 1}
    body = run.stream.records[0]
    assert S._sha512(body) == out.record and body["registration"] == reg.digest and body["symbol"] == "accept"


def test_every_window_of_the_link_in_order(tmp_path):
    run = Run(tmp_path)
    outs = [run.round(run.honest(w)) for w in range(3)]
    assert all(o.accepted for o in outs) and run.stream.symbols == ("accept",) * 3
    assert [r["order"] for r in run.stream.records] == [0, 1, 2]


def test_the_spec_declares_each_public_item_with_its_reason(tmp_path):
    lw = Run(tmp_path).lw
    assert {p.item: p.kind for p in lw.spec.public} == {
        "population": "structure", "window": "structure", f"leaves:{VALUE}": "structure",
        f"domain:{VALUE}": "structure", VALUE: "commitment", "parameters": "structure", "charge": "structure",
        "occupancy": "structure", "verdict": "output", "deadline": "output"}
    assert lw.spec.public == PUBLIC and lw.spec.check.public_outputs == OUTPUTS and isinstance(lw.spec.selection, S.All)
    assert lw.spec.check.program == P.program_sha512(Program(lw.unit))
    assert replace(lw, sync_set=SyncSet(tuple(range(16)), tuple(range(1, 9)))).charge == 7.0
    assert lw.spec.digest() != replace(lw, sync_set=SyncSet((0, 1), (1, 2))).spec.digest()
    assert lw.spec.digest() != replace(lw, run="another").spec.digest()
    assert lw.spec.digest() != replace(lw, certifier_session=signers.OTHER_SESSION).spec.digest()
    declared = replace(lw, occupancy=honest_occupancy())
    assert lw.spec.digest() != declared.spec.digest() != replace(lw, occupancy=(None, honest_occupancy()[1])).spec.digest()


def test_each_heads_counts_are_the_declared_occupancy(tmp_path):
    """Under the honest run's profile every window is accepted. An ingress window holding a frame fewer than the profile
    (a port that stopped short) or a frame more is rejected, though its certificate verifies and the audit compares no
    ingress grid; with no profile the same records are accepted."""
    (tmp_path / "declared").mkdir()
    (tmp_path / "bare").mkdir()
    declared, bare = Run(tmp_path / "declared", occupancy=honest_occupancy()), Run(tmp_path / "bare")
    short = ingress_records(([1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]))[1]
    extra = ingress_records(([1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0]))[2]
    for run in (declared, bare):
        assert run.round(run.honest(0)).accepted
        for w, ingress in ((1, short), (2, extra)):
            assert occupancy(ingress.grid) != honest_occupancy()[1][w]
            out = run.round(run.lw.commit(w, run.egress(egress_records()[w]), run.ingress(ingress), SALTS[w]))
            assert reject(out) if run is declared else out.accepted
    assert declared.stream.symbols == ("accept", "reject", "reject") and bare.stream.symbols == ("accept",) * 3


def test_the_profile_binds_egress_too_and_ends_where_it_ends(tmp_path):
    """An egress profile other than the Program's grid rejects the honest records the audit accepts, and a window past
    the profile is rejected."""
    eg, ig = honest_occupancy()
    row = eg[1]
    moved = ((row[0] + 1) % (PARAMS.allocation + 1),) + row[1:]
    run = Run(tmp_path, occupancy=((eg[0], moved), ig))
    assert [run.round(run.honest(w)).accepted for w in range(3)] == [True, False, False]
    assert run.lw.declares(0, 0, eg[0]) and not run.lw.declares(0, 1, row) and not run.lw.declares(0, 2, eg[2])
    assert run.lw.declares(1, 2, ig[2]) and not run.lw.declares(1, 3, ig[2])


def test_under_an_egress_profile_the_egress_grid_is_still_compared(tmp_path):
    """With an egress profile the head check holds the head's status, grid and counts, and the audit still compares the
    egress grid in the clear, until N3 runs on every window (NET-38): a window whose frames differ from the Program's
    grid in content alone, at the profile's counts, is rejected, as the stand-in rejects it. An egress head under
    another parameters digest is rejected though its certificate verifies."""
    run = Run(tmp_path, occupancy=honest_occupancy())
    honest = egress_records()[0]
    slots = [list(row) for row in honest.grid]
    b, k = next((b, k) for b, row in enumerate(slots) for k, s in enumerate(row) if s is not None)
    slots[b][k] = bytes(x ^ 0xFF for x in slots[b][k])
    content = replace(honest, grid=tuple(tuple(row) for row in slots))
    assert occupancy(content.grid) == occupancy(honest.grid) and content.grid != honest.grid
    opening = run.lw.commit(0, run.egress(content), run.ingress(ingress_records()[0]), SALTS[0])
    assert all(i is not None and HC.reference(i) for i in run.lw.head_checks(opening, bytes.fromhex(run.lw.root(opening))))
    assert reject(run.round(opening))
    unswept = run.issuer.certify(egress_records()[1], EGRESS, PARAMS, parameters_digest(PARAMS))
    assert reject(run.round(run.lw.commit(1, unswept, run.ingress(ingress_records()[1]), SALTS[1])))
    assert run.round(run.honest(2)).accepted and run.stream.symbols == ("reject", "reject", "accept")


def test_a_head_of_another_size_class_is_a_reject(tmp_path):
    """An ingress head of 8 buckets on a link whose ingress grid has 4 is longer than the unit's head, so no head check
    takes it: rejected, though its certificate verifies and its status is complete."""
    run = Run(tmp_path)
    wide = GridParams(8, PARAMS.allocation, PARAMS.frame_bytes)
    ingress = Certifier(LINK, wide)
    for _ in range(wide.buckets):
        ingress.tick()
    signed = run.issuer.certify(ingress.close_window(), INGRESS, wide, parameters_digest(wide))
    opening = run.lw.commit(0, run.egress(egress_records()[0]), signed, SALTS[0])
    assert run.lw.head_checks(opening, bytes.fromhex(run.lw.root(opening)))[1] is None
    assert reject(run.round(opening))


def test_an_occupancy_row_is_t_counts_within_the_allocation(tmp_path):
    lw = Run(tmp_path).lw
    four = (0,) * PARAMS.buckets
    for bad in ((None,), (((0,) * (PARAMS.buckets - 1),), None), (None, (four, (PARAMS.allocation + 1,) + four[1:])),
                ((four, (True,) + four[1:]), None)):
        with pytest.raises(ValueError, match="occupancy"):
            replace(lw, occupancy=bad)
    assert replace(lw, occupancy=((four,), (four, four))).declares(1, 1, four)


def test_the_unit_is_each_ends_head_check_beside_the_certificates(tmp_path):
    """N2's unit holds the certificates' check and each end's head check, at the link's size class: L the name's bytes,
    T the end's buckets and COUNTS whether the end has a profile, each head check one gate that names its Definition. A
    profile changes the unit, and so the spec's Program."""
    lw = Run(tmp_path).lw
    declared = replace(lw, occupancy=(honest_occupancy()[0], None))
    assert lw.unit.id == "NetworkLinkWindow_v2{EGRESS=(9,4,0),INGRESS=(9,4,0)}"
    assert declared.unit.id == "NetworkLinkWindow_v2{EGRESS=(9,4,1),INGRESS=(9,4,0)}"
    assert lw.spec.check.program != declared.spec.check.program and lw.spec.check.query != declared.spec.check.query
    defs = codec.encode_program(P.population_program(declared.unit, 1))["definitions"]
    assert {"NetworkCertificatesCheck_v1", "Gate[NetworkHeadCheck_v1{L=9,T=4,COUNTS=1}]_v1",
            "Gate[NetworkHeadCheck_v1{L=9,T=4,COUNTS=0}]_v1"} <= set(defs)
    assert not any(f.startswith(("NetworkHidingRead", "Hm96Sha512Leaf", "Sha512Compress")) for f in defs)
    params, _ = declared.unit.signature()
    gate = HC.gate(9, 4, True)
    assert gate.params == HC.unit(9, 4, True).signature()[0]
    assert [n for n, _ in params] == ["egress", "ingress", *(f"egress_{n}" for n, _ in gate.params),
                                      *(f"ingress_{n}" for n, _ in HC.gate(9, 4, False).params)]
    wide = replace(lw, ingress=GridParams(8, 2, 1))
    assert wide.head_key(1) == (9, 8, 0) and wide.head_key(0) == (9, 4, 0)


def test_the_domain_is_one_stages_served_domain(tmp_path):
    lw = Run(tmp_path).lw
    part = P.partition(P.population_program(lw.unit, 1), lw.spec.check.query)
    for w in range(3):
        want = R.served_domain(part["program"], P.digest(part), VALUE, "hm96-sha512/row/v2", 4, f"r-network-test/{LINK}/{w}")
        assert lw.ports(w)[VALUE]["domain"] == want
    assert lw.ports(0) != lw.ports(1)


def test_the_challenger_never_holds_a_certificate_or_a_violation_kind(tmp_path):
    """Everything the challenger keeps (the registrations, receipts, draws and outcome records) names the salted root,
    the order and digests: no certificate, no head, no frame root, and a rejected window doesn't say why."""
    run = Run(tmp_path)
    honest = run.honest(0)
    eg, ig = run.honest(1).records
    forged = CertifiedRecord(eg.head, replace(eg.certificate, signature=bytes(len(eg.certificate.signature))))
    assert run.round(honest).accepted
    assert reject(run.round(run.lw.commit(1, forged, ig, SALTS[1])))
    kept = json.dumps([run.stream.records, [e.registration.record for e in run.stream._entries],
                       [e.registration.receipt for e in run.stream._entries]])
    for rec in honest.records:
        for secret in (rec.certificate.signature[4:68].hex(), rec.certificate.digest.hex(), rec.head.root.hex(),
                       rec.head.parameters.hex(), encode(rec)[:96].hex()):
            assert secret not in kept
    for kind in ("signature", "no-verdict", "mismatch", "status:", "duplicate", "advice"):
        assert kind not in kept


def test_an_extra_public_field_is_refused(tmp_path):
    run = Run(tmp_path)
    rec = run.lw.registration(run.honest(0))
    for extra in ({"deadline": 1}, {"verdict": 1}, {"charge": 7}):
        with pytest.raises(S.Refused) as e:
            run.lw.register(run.stream, {**rec, **extra})
        assert e.value.verdict.codes == ("keys",)
    with pytest.raises(S.Refused) as e:
        run.lw.register(run.stream, {**rec, "window": {**rec["window"], "deadline": 1}})
    assert e.value.verdict.codes == ("window",)
    assert run.lw.register(run.stream, rec).order == 0


def test_a_certificate_in_the_clear_is_refused(tmp_path):
    run = Run(tmp_path)
    opening = run.honest(0)
    rec = run.lw.registration(opening)
    cert = opening.records[0].certificate
    in_root = [dict(rec["roots"][0], signature=cert.signature.hex())]
    per_direction = [dict(rec["roots"][0], port=f"{VALUE}/egress"), dict(rec["roots"][0], port=f"{VALUE}/ingress")]
    for bad, code in (({**rec, "certificates": [cert.signature.hex()]}, "keys"), ({**rec, "roots": in_root}, "keys"),
                      ({**rec, "roots": per_direction}, "roots"),
                      ({**rec, "window": {**rec["window"], "certificate": cert.signature.hex()}}, "window")):
        with pytest.raises(S.Refused) as e:
            run.lw.register(run.stream, bad)
        assert code in e.value.verdict.codes
    with pytest.raises(S.Refused) as e:
        run.lw.register(run.stream, {**rec, "scheme": "frame-v3-sha512/sha512"})
    assert "unsalted" in e.value.verdict.codes
    assert run.stream.records == () and run.lw.register(run.stream, rec).order == 0


def test_the_window_is_the_order(tmp_path):
    run = Run(tmp_path)
    with pytest.raises(S.Refused) as e:
        run.lw.register(run.stream, run.lw.registration(run.honest(1)))
    assert e.value.verdict.codes == ("window",)
    assert run.round(run.honest(0)).accepted
    with pytest.raises(S.Refused):
        run.lw.register(run.stream, run.lw.registration(run.honest(0), run.previous))


def test_a_bad_signature_is_a_reject(tmp_path):
    run = Run(tmp_path)
    good = run.honest(0)
    eg, ig = good.records
    zeros = CertifiedRecord(eg.head, replace(eg.certificate, signature=bytes(len(eg.certificate.signature))))
    assert reject(run.round(run.lw.commit(0, zeros, ig, SALTS[0])))
    (tmp_path / "stranger").mkdir()
    stranger = Issuer(signers.signer(tmp_path / "stranger", seed=b"other"), signers.SESSION)
    signed = stranger.certify(ingress_records()[1], INGRESS, PARAMS, parameters_digest(PARAMS))
    assert reject(run.round(run.lw.commit(1, run.egress(egress_records()[1]), signed, SALTS[1])))
    assert run.round(run.honest(2)).accepted and run.stream.symbols == ("reject", "reject", "accept")


def test_a_certificate_from_another_session_is_a_reject(tmp_path):
    """The cross-session replay: the certifier's own key signed the link-window in an earlier session, and its
    certificate signs another message than this session's, so it is rejected."""
    run = Run(tmp_path)
    earlier = LmsSigner(run.signer.key, run.signer.state)
    replayed = Issuer(earlier, signers.OTHER_SESSION).certify(egress_records()[0], EGRESS, PARAMS,
                                                              parameters_digest(PARAMS, SYNC_SET))
    earlier.close()
    assert replayed.verify(run.signer.public_key(), signers.OTHER_SESSION)
    assert reject(run.round(run.lw.commit(0, replayed, run.ingress(ingress_records()[0]), SALTS[0])))
    assert run.round(run.honest(1)).accepted and run.stream.symbols == ("reject", "accept")


def test_a_digest_mismatch_is_a_reject(tmp_path):
    """A head edited under its certificate (the certified status or counts), and a grid other than the Program's that
    the certifier honestly signed."""
    run = Run(tmp_path)
    eg, ig = run.honest(0).records
    counts = RecordHead(**{f: getattr(eg.head, f) for f in RecordHead.__slots__} | {"counts": (0,) * PARAMS.buckets})
    assert reject(run.round(run.lw.commit(0, CertifiedRecord(counts, eg.certificate), ig, SALTS[0])))
    honest = egress_records()[1]
    rows = [list(row) for row in honest.grid]
    rows[0][0] = b"\xff"
    tampered = replace(honest, grid=tuple(tuple(r) for r in rows))
    assert reject(run.round(run.lw.commit(1, run.egress(tampered), run.ingress(ingress_records()[1]), SALTS[1])))


def test_another_window_or_link_is_a_reject(tmp_path):
    run = Run(tmp_path)
    w1 = run.lw.commit(0, run.egress(egress_records()[1]), run.ingress(ingress_records()[1]), SALTS[0])
    assert reject(run.round(w1))
    tokens = [replace(r, link="tokens") for r in (egress_records()[1], ingress_records()[1])]
    assert reject(run.round(run.lw.commit(1, run.egress(tokens[0]), run.ingress(tokens[1]), SALTS[1])))
    swapped = run.lw.commit(2, run.ingress(ingress_records()[2]), run.egress(egress_records()[2]), SALTS[2])
    assert reject(run.round(swapped))
    assert run.stream.symbols == ("reject",) * 3


def test_an_unset_deadline_bit_is_a_reject(tmp_path):
    run = Run(tmp_path)
    for w, (e_late, i_late) in enumerate(((True, False), (False, True))):
        opening = run.lw.commit(w, run.egress(egress_records()[w], deadline=not e_late),
                                run.ingress(ingress_records()[w], deadline=not i_late), SALTS[w])
        reg = run.register(opening)
        draw = run.stream.select(reg)
        run.stream.deliver(draw, [run.lw.session(opening, draw)])
        verdict = run.lw.verify(opening, draw, FRAMES, SYNCS)
        assert verdict["public"] == {"verdict": 0, "deadline": 0} and verdict["accepted"] is False
        assert reject(run.lw.outcome(run.stream, draw, opening, FRAMES, SYNCS))


def test_a_late_proof_is_a_reject(tmp_path):
    run = Run(tmp_path)
    opening = run.honest(0)
    draw = run.stream.select(run.register(opening))
    run.clock.now = 60.5
    out = run.stream.deliver(draw, [run.lw.session(opening, draw)])
    assert out is not None and not out.accepted and out.verdict.codes == ("late",) and out.public == {}
    with pytest.raises(S.Refused):
        run.lw.outcome(run.stream, draw, opening, FRAMES, SYNCS)
    opening = run.honest(1)
    draw = run.stream.select(run.register(opening))
    run.clock.now = 125.0
    assert run.lw.outcome(run.stream, draw, opening, FRAMES, SYNCS).verdict.codes == ("late",)
    assert run.stream.symbols == ("late", "late")


def test_the_outcome_binds_the_registered_root(tmp_path):
    """The same certified records under other salts open another root: the session's root is not the registered one,
    and the outcome is a reject. Accepted outcomes under two salts are two records."""
    run = Run(tmp_path)
    registered = run.honest(0)
    other = replace(registered, salts=SALTS[2])
    assert run.lw.root(other) != run.lw.root(registered)
    out = run.round(registered, delivered=other)
    assert reject(out)
    audit = run.stream.records[0]["audit"]
    assert audit["registration"]["roots"][0]["root"] == run.lw.root(registered)
    assert audit["checks"]["roots"]["ok"] is False and audit["verdicts"]["network-audit"]["accepted"] is False

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = Run(tmp_path / "a"), Run(tmp_path / "b")
    opening_a = a.honest(0)
    opening_b = replace(b.honest(0), salts=SALTS[1])
    assert opening_a.records == opening_b.records
    out_a, out_b = a.round(opening_a), b.round(opening_b)
    assert out_a.accepted and out_b.accepted and out_a.record != out_b.record
    assert [r.stream.records[0]["audit"]["registration"]["roots"][0]["root"] for r in (a, b)] == [
        a.lw.root(opening_a), b.lw.root(opening_b)]


def test_a_signer_that_refuses_leaves_the_window_unregistered(tmp_path):
    run = Run(tmp_path)
    run.signer.state.unlink()
    assert run.egress(egress_records()[0]) is None and run.issuer.refused == [(LINK, EGRESS, 0)]
    assert run.stream.records == ()


def test_the_opening_must_be_the_registered_link_windows(tmp_path):
    run = Run(tmp_path)
    opening = run.honest(0)
    draw = run.stream.select(run.register(opening))
    with pytest.raises(ValueError, match="link-window"):
        run.lw.verify(run.honest(1), draw, FRAMES, SYNCS)
    for salts in ((SALTS[0][0],) * 4, SALTS[0][:2]):
        with pytest.raises(ValueError, match="salts"):
            run.lw.commit(0, *opening.records, salts=salts)


def test_a_head_leaf_opens_from_the_head_alone(tmp_path):
    """Leaves 2 and 3 are the heads' hiding leaves: a proof that holds a head, its salt and two siblings reaches the
    root without the certificate, hashing the head in at most three compressions after the row prefix's block."""
    run = Run(tmp_path)
    opening = run.honest(0)
    dom, key = run.lw.domain(0), hm96.SHA512.default_key
    levels = tree_levels(dom, run.lw.leaves(opening))
    assert rows(opening.records)[:2] == [encode(r) for r in opening.records]
    for i, rec in enumerate(opening.records):
        head, salt = rec.head.encode(), opening.salts[HEADS + i]
        assert -(-(len(head) + 17) // 128) <= 3
        bc = hm96.SHA512.commit_string(key, sha512_row_v2_digest(head, 8 * len(head)), salt)
        leaf = hm96.SHA512.tree_leaf(key, bc)
        path = [level[((HEADS + i) >> k) ^ 1] for k, level in enumerate(levels[:-1])]
        assert root_ref(dom.domain_id, SCHEMA, leaf, HEADS + i, path).hex() == run.lw.root(opening)


def test_a_head_leaf_that_isnt_its_records_head_is_a_reject(tmp_path):
    """A root whose egress head leaf holds another window's head: no opening of the records reproduces it, so the
    link-window is rejected though both certificates verify."""
    run = Run(tmp_path)
    opening = run.honest(0)
    other = run.honest(1).records[0].head.encode()
    leaves = run.lw.leaves(opening)
    dom, key = run.lw.domain(0), hm96.SHA512.default_key
    bc = hm96.SHA512.commit_string(key, sha512_row_v2_digest(other, 8 * len(other)), opening.salts[HEADS])
    leaves[HEADS] = row_leaf(dom, HEADS, hm96.SHA512.tree_leaf(key, bc), SCHEMA)
    rec = run.lw.registration(opening)
    rec = {**rec, "roots": [{**rec["roots"][0], "root": tree_levels(dom, leaves)[-1][0].hex()}]}
    reg = run.lw.register(run.stream, rec)
    draw = run.stream.select(reg)
    assert run.stream.deliver(draw, [run.lw.session(opening, draw)]) is None
    assert reject(run.lw.outcome(run.stream, draw, opening, FRAMES, SYNCS))
