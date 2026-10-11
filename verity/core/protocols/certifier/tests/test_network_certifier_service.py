"""One link-window's network certificates through the proof service (N2): the round trip to an accepted, exhaustive
outcome, the public-items rule on the registration, and each way a link-window fails closed, on injected coins and an
injected clock."""

import json
from dataclasses import replace

import pytest

from verity.ml.circuits.program import Program
from verity.core.protocols.certifier.certificate import (CertifiedRecord, Direction, Issuer, LmsSigner,
                                                                          RecordHead, parameters_digest)
from verity.core.protocols.certifier.grid import Certifier, Record
from verity.core.protocols.certifier.schedule import SyncSet, run_shaper
from verity.core.protocols.certifier.service import (OUTPUTS, PUBLIC, UNIT, VALUE, LinkWindows,
                                                                      encode)
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R

from test_network_certifier_audit import FRAMES, P as PARAMS, SYNC_SET, SYNCS, honest_ready, ingress_run
import signers

LINK = "responses"
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
DELTA = 2.0 ** -40
SALTS = [(bytes([2 * w + 1]) * 192, bytes([2 * w + 2]) * 192) for w in range(3)]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def egress_records():
    return run_shaper(FRAMES, honest_ready(), SYNCS, Certifier(LINK, PARAMS))


def ingress_records():
    return [replace(r, link=LINK) for r in ingress_run([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])]


class Run:
    """A link with its network certifier (an LMS signer on a small tree), the protocol's parameters, and the challenger's stream."""

    def __init__(self, tmp_path, *, proof_deadline=60.0, coins=b"\x07") -> None:
        self.signer = signers.signer(tmp_path)
        self.issuer = Issuer(self.signer, signers.SESSION)
        self.lw = LinkWindows(LINK, PARAMS, SYNC_SET, self.signer.public_key(), "r-network-test", signers.SESSION,
                              proof_deadline=proof_deadline)
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
        draw = self.stream.select(self.register(opening))
        assert self.stream.deliver(draw, [self.lw.session(delivered or opening, draw)]) is None
        return self.lw.outcome(self.stream, draw, delivered or opening, FRAMES, SYNCS)


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
        "verdict": "output", "deadline": "output"}
    assert lw.spec.public == PUBLIC and lw.spec.check.public_outputs == OUTPUTS and isinstance(lw.spec.selection, S.All)
    assert lw.spec.check.program == P.program_sha512(Program(UNIT))
    assert replace(lw, sync_set=SyncSet(tuple(range(16)), tuple(range(1, 9)))).charge == 7.0
    assert lw.spec.digest() != replace(lw, sync_set=SyncSet((0, 1), (1, 2))).spec.digest()
    assert lw.spec.digest() != replace(lw, run="another").spec.digest()
    assert lw.spec.digest() != replace(lw, certifier_session=signers.OTHER_SESSION).spec.digest()


def test_the_domain_is_one_stages_served_domain(tmp_path):
    lw = Run(tmp_path).lw
    part = P.partition(P.population_program(UNIT, 1), lw.spec.check.query)
    for w in range(3):
        want = R.served_domain(part["program"], P.digest(part), VALUE, "hm96-sha512/row/v2", 2, f"r-network-test/{LINK}/{w}")
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
    other = replace(registered, salts=(SALTS[2][0], SALTS[2][1]))
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
    with pytest.raises(ValueError, match="salts"):
        run.lw.commit(0, *opening.records, salts=(SALTS[0][0], SALTS[0][0]))
