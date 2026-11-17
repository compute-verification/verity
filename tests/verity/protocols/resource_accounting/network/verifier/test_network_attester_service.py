"""One execution phase's network certificates through the proof system (N2), on a network attester's link
(`NetworkAttesterProfile`): the round trip to an accepted, exhaustive outcome, the public-items rule on the registration, the
head check against the network attester's grid and digest, and each way an execution phase fails closed, on injected
coins and an injected clock. Each round's verdict of record is decided on a stand-in proof of the proof block
(`standin`: the prover's rows and output word by the gates' references, the Lean run reading its session file)."""

import itertools
import json
from dataclasses import replace

import pytest

from verity.circuits import boolean as BL, codec
from verity.circuits.defs import bind
from verity.circuits.program import Program
from verity.protocols.resource_accounting.network.verifier import head_check as HC, service as SVC
from verity.protocols.resource_accounting.network.verifier.certificate import (CertifiedRecord, Direction, Issuer, LmsSigner,
                                                                          RecordHead, parameters_digest)
from verity.protocols.resource_accounting.network.verifier.grid import Certifier, GridParams, BlockGrid, occupancy
from verity.protocols.resource_accounting.network.verifier.schedule import run_shaper
from verity.commitments import hm96
from verity.commitments.gates.frame_v3 import root_ref
from verity.commitments.merkle import tree_levels
from verity.commitments.rowleaf import row_leaf, sha512_row_v2_digest
from verity.protocols.resource_accounting.network.verifier.service import (HEADS, OUTPUTS, PUBLIC, SCHEMA, VALUE, TransferBlocks, encode,
                                                                      rows)
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import partition as P, docket as R
from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
from verity.protocols.proofs.spot.verifier.one_stage.proofs import Finding

from test_network_attester_audit import FRAMES, P as PARAMS, SYNC_SET, SYNCS, honest_ready, ingress_run
import signers
import standin

LINK = "responses"
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
DELTA = 2.0 ** -40
SALTS = [tuple(bytes([4 * w + k + 1]) * 192 for k in range(4)) for w in range(3)]
#: the network attester's parameters digest (`network_attester.NetworkAttesterParams.digest`), which both ends' heads bind
NETWORK_ATTESTER_DIGEST = bytes(range(64))


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def egress_records():
    return run_shaper(FRAMES, honest_ready(), SYNCS, Certifier(LINK, PARAMS))


def ingress_records(arrivals=([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])):
    return [replace(r, link=LINK) for r in ingress_run([list(a) for a in arrivals])]


class Run:
    """A network attester's link (an LMS signer on a small tree, its records under `NETWORK_ATTESTER_DIGEST`), the protocol's
    parameters, and the challenger's stream. The attester's coins give each record a fresh salt, the same in every
    run."""

    def __init__(self, tmp_path, *, proof_deadline=60.0, coins=b"\x07") -> None:
        self.tmp = tmp_path
        self.signer = signers.signer(tmp_path)
        sampled = itertools.count(0xc0)
        self.issuer = Issuer(self.signer, signers.SESSION, coins=lambda n: bytes([next(sampled) % 256]) * n)
        self.lw = TransferBlocks(LINK, PARAMS, SYNC_SET, self.signer.public_key(), "r-network-test", signers.SESSION,
                              proof_deadline=proof_deadline, network_attester=SVC.NetworkAttesterProfile(3, NETWORK_ATTESTER_DIGEST))
        self.clock = Clock()
        self.stream = self.lw.stream(budget=8, delta=DELTA, challenger="challenger-test", coins=lambda n: coins * n,
                                     clock=self.clock)
        self.previous = None

    def egress(self, record: BlockGrid, **kw) -> CertifiedRecord:
        return self.issuer.certify(record, EGRESS, PARAMS, NETWORK_ATTESTER_DIGEST, **kw)

    def ingress(self, record: BlockGrid, **kw) -> CertifiedRecord:
        return self.issuer.certify(record, INGRESS, PARAMS, NETWORK_ATTESTER_DIGEST, **kw)

    def honest(self, w: int):
        return self.lw.commit(w, self.egress(egress_records()[w]), self.ingress(ingress_records()[w]), SALTS[w])

    def register(self, opening) -> S.WindowCommitmentRecord:
        reg = self.lw.register(self.stream, self.lw.registration(opening, self.previous))
        self.previous = reg.digest
        return reg

    def prove(self, opening, sample, **kw) -> SVC.Delivered:
        return standin.prove(self.lw, opening, sample, SYNCS, self.tmp / f"proof-{sample.docket.order}", **kw)

    def decide(self, sample, opening, **kw) -> S.Outcome:
        return standin.decide(self.lw, self.stream, sample, opening, SYNCS, self.tmp / f"proof-{sample.docket.order}",
                              **kw)

    def round(self, opening, delivered=None) -> S.Outcome:
        """One registration through its outcome on a proof over `delivered` (default: the registered opening)."""
        return self.decide(self.stream.select(self.register(opening)), delivered or opening)


def reject(out: S.Outcome) -> bool:
    return not out.accepted and out.verdict.codes == ("reject",) and out.public == {} and out.exhaustive is None


def test_one_phase_round_trip_is_the_exhaustive_fact(tmp_path):
    run = Run(tmp_path)
    opening = run.honest(0)
    reg = run.register(opening)
    assert reg.order == 0 and reg.receipt["received"] == "0" and reg.record["population"] == 1
    assert reg.record["roots"] == [{"port": VALUE, "root": run.lw.root(opening), **run.lw.ports(0)[VALUE]}]
    assert S.check_public(run.lw.spec, reg.record).ok
    sample = run.stream.select(reg)
    assert sample.units == [0] and sample.draw["law"] == "subset" and sample.draw["k"] == 1
    out = run.decide(sample, opening)
    assert out.accepted and out.verdict.ok and out.exhaustive == S.Exhaustive(1) and out.profile is None
    assert out.terms == S.Terms(None) and out.public == {"verdict": 1, "deadline": 1}
    body = run.stream.records[0]
    assert S._sha512(body) == out.record and body["registration"] == reg.digest and body["symbol"] == "accept"


def test_every_window_of_the_link_in_order(tmp_path):
    run = Run(tmp_path)
    outs = [run.round(run.honest(w)) for w in range(3)]
    assert all(o.accepted for o in outs) and run.stream.symbols == ("accept",) * 3
    assert [r["order"] for r in run.stream.records] == [0, 1, 2]


def test_the_network_verdict_accepts_a_link_whose_every_window_accepted(tmp_path):
    """`network_verdict` over the profile's three windows: each outcome of record is read in window order, and a run
    that stops short of the profile's last window is a reject until that window's outcome is in."""
    run = Run(tmp_path)
    for w in range(2):
        run.round(run.honest(w))
    short = run.lw.network_verdict(run.stream)
    assert not short["accepted"] and short["refused"] == [2] and short["records"][2] is None
    run.round(run.honest(2))
    verdict = run.lw.network_verdict(run.stream)
    assert verdict["accepted"] and verdict["windows"] == 3 and verdict["refused"] == []
    assert verdict["records"] == [S._sha512(r) for r in run.stream.records] and verdict["digest"] != short["digest"]
    assert set(verdict) == {"accepted", "windows", "refused", "records", "digest"}


def test_one_rejected_window_rejects_the_link(tmp_path):
    run = Run(tmp_path)
    eg, ig = run.honest(1).records
    forged = CertifiedRecord(eg.head, eg.salt, replace(eg.certificate, signature=bytes(len(eg.certificate.signature))))
    outs = [run.round(run.honest(0)), run.round(run.lw.commit(1, forged, ig, SALTS[1])), run.round(run.honest(2))]
    assert [o.accepted for o in outs] == [True, False, True]
    verdict = run.lw.network_verdict(run.stream)
    assert not verdict["accepted"] and verdict["refused"] == [1] and None not in verdict["records"]


def test_a_registered_window_with_no_outcome_rejects_the_link(tmp_path):
    run = Run(tmp_path)
    for w in range(2):
        run.round(run.honest(w))
    opening = run.honest(2)
    sample = run.stream.select(run.register(opening))
    assert run.stream.deliver(sample, run.prove(opening, sample).sessions()) is None
    verdict = run.lw.network_verdict(run.stream)
    assert len(run.stream.log) == 3 and len(run.stream.records) == 2
    assert not verdict["accepted"] and verdict["refused"] == [2] and verdict["records"][2] is None


def test_a_gap_in_the_registration_order_rejects_the_link(tmp_path):
    """Window 2 registered second, past `TransferBlocks.register`'s order check, and accepted on its own: window 1 is never
    registered, so the link is a reject, as it is with a registration past the windows asked for."""
    run = Run(tmp_path)
    run.round(run.honest(0))
    skipped = run.honest(2)
    reg = run.stream.register(run.lw.registration(skipped, run.previous), run.lw.ports(2))
    sample = run.stream.select(reg)
    assert run.decide(sample, skipped).accepted
    verdict = run.lw.network_verdict(run.stream)
    assert run.stream.symbols == ("accept", "accept")
    assert not verdict["accepted"] and verdict["refused"] == [1, 2] and verdict["records"][1:] == [None, None]
    assert run.lw.network_verdict(run.stream, windows=1)["refused"] == [1]


def test_the_network_verdict_is_only_of_its_own_links_stream(tmp_path):
    run = Run(tmp_path)
    run.round(run.honest(0))
    other = replace(run.lw, link="tokens")
    with pytest.raises(ValueError, match="not this link's"):
        other.network_verdict(run.stream)
    with pytest.raises(ValueError, match="one or more execution phases"):
        run.lw.network_verdict(run.stream, windows=0)


def test_the_spec_declares_each_public_item_with_its_reason(tmp_path):
    lw = Run(tmp_path).lw
    assert {p.item: p.kind for p in lw.spec.public} == {
        "population": "structure", "window": "structure", f"leaves:{VALUE}": "structure",
        f"domain:{VALUE}": "structure", VALUE: "commitment", "parameters": "structure", "verdict": "output",
        "deadline": "output", "warden": "structure"}
    assert lw.spec.public == PUBLIC and lw.spec.check.public_outputs == OUTPUTS and isinstance(lw.spec.selection, S.All)
    assert lw.spec.check.program == P.program_sha512(Program(lw.unit))
    assert lw.spec.digest() != replace(lw, run="another").spec.digest()
    assert lw.spec.digest() != replace(lw, certifier_session=signers.OTHER_SESSION).spec.digest()
    assert lw.spec.digest() != replace(lw, network_attester=SVC.NetworkAttesterProfile(3, bytes(64))).spec.digest()
    assert lw.spec.digest() != replace(lw, network_attester=SVC.NetworkAttesterProfile(4, NETWORK_ATTESTER_DIGEST)).spec.digest()


def test_n2_compares_no_grid_and_holds_no_frames(tmp_path):
    """The head check holds the head's status, grid and counts, and N2 compares no grid: a window whose frames differ
    from the Program's grid in content alone, or in order within a bucket, at the same counts, is accepted, since the
    frames are the scan's. An egress head under another parameters digest is rejected though its certificate
    verifies."""
    run = Run(tmp_path)

    def edited(w, edit):
        honest = egress_records()[w]
        slots = [list(row) for row in honest.grid]
        b = next(b for b, row in enumerate(slots) if sum(s is not None for s in row) == 2)
        slots[b] = edit(slots[b])
        other = replace(honest, grid=tuple(tuple(row) for row in slots))
        assert occupancy(other.grid) == occupancy(honest.grid) and other.grid != honest.grid
        opening = run.lw.commit(w, run.egress(other), run.ingress(ingress_records()[w]), SALTS[w])
        assert all(HC.reference(i) for i in run.lw.head_checks(opening, bytes.fromhex(run.lw.root(opening))))
        return opening

    assert run.round(edited(0, lambda row: [bytes(x ^ 0xFF for x in row[0]), row[1]])).accepted
    unswept = run.issuer.certify(egress_records()[1], EGRESS, PARAMS, parameters_digest(PARAMS, SYNC_SET))
    assert reject(run.round(run.lw.commit(1, unswept, run.ingress(ingress_records()[1]), SALTS[1])))
    assert run.round(edited(2, lambda row: row[::-1])).accepted
    assert run.stream.symbols == ("accept", "reject", "accept")


def test_a_head_of_another_size_class_is_a_reject(tmp_path):
    """An ingress head of 8 buckets on a link whose ingress grid has 4 is longer than the proof block's head, so no head
    check takes it: rejected, though its certificate verifies and its status is complete."""
    run = Run(tmp_path)
    wide = GridParams(8, PARAMS.allocation, PARAMS.frame_bytes)
    ingress = Certifier(LINK, wide)
    for _ in range(wide.buckets):
        ingress.tick()
    signed = run.issuer.certify(ingress.close_window(), INGRESS, wide, parameters_digest(wide))
    opening = run.lw.commit(0, run.egress(egress_records()[0]), signed, SALTS[0])
    assert run.lw.head_checks(opening, bytes.fromhex(run.lw.root(opening)))[1] is None
    assert reject(run.round(opening))


def test_the_unit_is_each_ends_head_check(tmp_path):
    """N2's proof block is each end's head check, at the link's size class: L the name's bytes, T the end's buckets and
    COUNTS 0, no end having a profile, each head check one gate that names its Definition, reads the proof block's
    signed digest for its end and its own direction, a constant. Its two outputs are the verdict and the deadline. Its
    public row is whole u16 words, as its witness row is."""
    lw = Run(tmp_path).lw
    assert lw.unit.id == "NetworkTransferBlocks_v5{EGRESS=(9,4,0),INGRESS=(9,4,0)}"
    defs = codec.encode_program(P.population_program(lw.unit, 1))["definitions"]
    pins = json.loads(HC.PINS.read_text())["digests"]
    fid = "NetworkHeadCheck_v2{L=9,T=4,COUNTS=0}"
    assert f"Gate[{fid}@{pins[fid]['sha512']}]_v2" in set(defs)
    assert not any(f.startswith(("NetworkHidingRead", "Hm96Sha512Leaf", "Sha512Compress", "NetworkCertificates"))
                   for f in defs)
    params, ret = lw.unit.signature()
    gate = HC.gate(9, 4, False)
    assert gate.params == HC.unit(9, 4, False).signature()[0] and ret.leaves == len(OUTPUTS) == 2
    fixed = ("signed", "direction")
    assert [n for n, _ in params] == ["egress", "ingress", *(f"{end}_{n}" for end in ("egress", "ingress")
                                                              for n, _ in gate.params if n not in fixed)]
    assert set(lw.layout["pub"]) | set(lw.layout["wit"]) == {n for n, _ in params}
    assert all(n.split("_", 1)[1] in ("head", "salt", "path", "salt_c") for n in lw.layout["wit"])
    assert lw.bits("pub") % 16 == lw.bits("wit") % 16 == 0 and lw.bits("o0") == 16
    wide = replace(lw, params=GridParams(8, 2, 1))
    assert wide.head_key(0) == wide.head_key(1) == (9, 8, 0)


def test_the_proof_block_is_the_registered_template(tmp_path):
    """The proof block a proof's circuit lowers has the registered proof block's id, signature and descriptor id, so the
    registered query names it as its template; it is bound without tracing, since each gate's expansion traces the head
    check (`head_check.expand`, minutes), which the proving run (`verity/protocols/resource_accounting/network/prover/n2_zk.py`) lowers and checks
    against the IR reference."""
    lw = Run(tmp_path).lw
    assert lw.proof_block.id == lw.unit.id and lw.proof_block.signature() == lw.unit.signature()
    assert codec.descriptor_id(lw.proof_block) == codec.descriptor_id(lw.unit) == P.template_id(lw.unit)
    assert lw.proof_block.defn is not lw.unit.defn and lw.partition is lw._partition


def test_n2s_program_reads_no_registered_value(tmp_path):
    """N2's verifier runs Lean without registered reads (`proofs.verify` with `registered` None), which
    `proofs.verify` refuses for a program that reads registered values; N2's proof block reads none, so its proofs reach
    Lean."""
    assert not PF.reads_registered(Run(tmp_path).lw.program)


def test_the_verdict_of_record_is_the_proofs(tmp_path):
    """The verifier decides on the delivered proof and public inputs alone, and reads the opening only for
    the debug comparison it is given. Each of these rejects an honest execution phase's sample: the Lean run rejecting
    the session, a circuit file other than the committed one, a public file whose instance is another proof block, the
    public or the output row's salt not the one committed, a proof on a head edited under its certificate (its
    verdict bit 0), with or without a forged output word, a forged signature, a certificate whose signed digest isn't
    the one the proof's public row holds, and the certificates swapped."""
    run = Run(tmp_path)
    lw = run.lw
    opening = run.honest(0)
    sample = run.stream.select(run.register(opening))
    certs = [r.certificate for r in opening.records]

    def verdict(delivered, certificates=certs, syncs=SYNCS, circuit=None, **kw):
        return lw.verify(sample, certificates, syncs, delivered,
                         circuit_sha512=circuit or standin.circuit_sha512(delivered), **kw)

    def prove(o, k, **kw):
        return standin.prove(lw, o, sample, SYNCS, tmp_path / f"proof-{k}", **kw)

    honest = prove(opening, "honest")
    v = verdict(honest)
    assert v["accepted"] and v["public"] == {"verdict": 1, "deadline": 1} and v["decided_by"] == "proof"
    assert v["by"] == SVC.OF_RECORD == "lean" and v["reference"] == {"accepted": True}
    assert "clear" not in v and verdict(honest, opening=opening)["clear"] == {"accepted": True}
    assert set(honest.salts) == {"pub", "o0"} and honest.output == 0b11
    eg, ig = opening.records
    zero = RecordHead(**{f: getattr(eg.head, f) for f in RecordHead.__slots__} | {"counts": (0,) * PARAMS.buckets})
    off = lw.commit(0, CertifiedRecord(zero, eg.salt, eg.certificate), ig, SALTS[0])
    forged = replace(eg.certificate, signature=bytes(len(eg.certificate.signature)))
    recertified = run.honest(0).records[0].certificate
    assert recertified.digest != eg.certificate.digest
    cases = [(prove(opening, 0, accepted=False), {}), (honest, {"circuit": "0" * 128}),
             (prove(opening, 1, units=[1]), {}), (prove(opening, 2, salts={"pub": bytes(192)}), {}),
             (prove(opening, 3, salts={"o0": bytes(192)}), {}), (prove(off, 4), {}), (prove(off, 5, word=0b11), {}),
             (honest, {"certificates": [forged, certs[1]]}), (honest, {"certificates": [recertified, certs[1]]}),
             (honest, {"certificates": certs[::-1]})]
    assert prove(off, 6).output == 0b10
    for k, (delivered, kw) in enumerate(cases):
        v = verdict(delivered, **kw)
        assert v["accepted"] is False and v["public"]["verdict"] == 0 and v["reference"] == {"accepted": False}, k
        assert set(v) == {"of_record", "decided_by", "by", "accepted", "digest", "sessions", "public", "reference"}, k


def _accepts_every_head(B, S_, *args):
    return B.call(BL.constant(1))


def _finding(program, accepted=True):
    return Finding("session-0", accepted, "" if accepted else "rejected", program, {}, (0,) if accepted else ())


def test_a_proof_of_another_circuit_with_the_same_interface_is_refused(tmp_path):
    """Each head-check gate names its Definition's pinned circuit digest, so the proof block's Program, which the spec
    and each registration bind, binds the head check's circuit. A head check with the same id and signature that accepts
    every head has another circuit digest: the pinned gate doesn't expand to it, its own gate is another gate, and a
    proof block holding that gate is another Program, whose proof the verifier refuses while it accepts one of
    the registered Program. A size class no sidecar pins names no digest, and no proof of its proof block is
    accepted."""
    run = Run(tmp_path)
    lw = run.lw
    key = lw.head_key(0)
    honest, gate = HC.unit(*key), HC.gate(*key)
    pinned = json.loads(HC.PINS.read_text())["digests"][honest.id]["sha512"]
    assert key == (9, 4, 0) and gate.id == f"Gate[{honest.id}@{pinned}]_v2" and HC.pinned(gate)
    forged = bind(replace(HC.NetworkHeadCheck, body=_accepts_every_head, register=False), L=9, T=4, COUNTS=0)
    assert forged.id == honest.id and forged.signature() == honest.signature()
    digest = HC.circuit_digest(forged)
    assert digest != pinned
    with pytest.raises(ValueError, match="another circuit"):
        HC.expand(gate, forged)
    swapped = HC.gate_of(forged, digest, "a head check that accepts every head")
    assert swapped.id != gate.id and swapped.signature() == gate.signature() and HC.expand(swapped, forged) is forged

    registered, query = lw._partition["program"], lw.query
    other = bind(SVC._unit(lambda *k: swapped if k == key else HC.gate(*k)), EGRESS=key, INGRESS=lw.head_key(1))
    program = P.partition(P.population_program(other, 1), query)["program"]
    assert other.id == lw.unit.id and program != registered and lw._partition["program"] == registered
    assert lw.proved([_finding(registered)])
    assert not lw.proved([_finding(program)]) and not lw.proved([])
    assert not lw.proved([_finding(registered), _finding(registered, accepted=False)])

    outs = []
    for w, proved in enumerate((program, registered)):
        opening = run.honest(w)
        sample = run.stream.select(run.register(opening))
        outs.append(run.decide(sample, opening, program=proved))
    assert reject(outs[0]) and outs[1].accepted

    wide = replace(lw, params=GridParams(16, 2, 1))
    assert wide.head_key(1) == (9, 16, 0)
    assert not HC.pinned(HC.gate(*wide.head_key(1))) and HC.gate(*wide.head_key(1)).id.endswith("@unpinned]_v2")
    assert not wide.proved([_finding(wide._partition["program"])])
    with pytest.raises(ValueError, match="no pinned circuit digest"):
        HC.expand(HC.gate(*wide.head_key(1)), HC.unit(*wide.head_key(1)))


def test_the_domain_is_one_stages_window_domain(tmp_path):
    lw = Run(tmp_path).lw
    part = P.partition(P.population_program(lw.unit, 1), lw.spec.check.query)
    for w in range(3):
        want = R.window_domain(part["program"], P.digest(part), VALUE, "hm96-sha512/row/v2", 4, f"r-network-test/{LINK}/{w}")
        assert lw.ports(w)[VALUE]["domain"] == want
    assert lw.ports(0) != lw.ports(1)


def test_the_challenger_never_holds_a_certificate_or_a_violation_kind(tmp_path):
    """Everything the challenger keeps (the registrations, receipts, samples and outcome records) names the salted root,
    the order and digests: no certificate, no head, no frame root, and a rejected window doesn't say why."""
    run = Run(tmp_path)
    honest = run.honest(0)
    eg, ig = run.honest(1).records
    forged = CertifiedRecord(eg.head, eg.salt, replace(eg.certificate, signature=bytes(len(eg.certificate.signature))))
    assert run.round(honest).accepted
    assert reject(run.round(run.lw.commit(1, forged, ig, SALTS[1])))
    kept = json.dumps([run.stream.records, [e.docket.record for e in run.stream._entries],
                       [e.docket.receipt for e in run.stream._entries]])
    for rec in honest.records:
        for secret in (rec.certificate.signature[4:68].hex(), rec.certificate.digest.hex(), rec.head.root.hex(),
                       rec.head.parameters.hex(), rec.salt.hex(), encode(rec)[:96].hex()):
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
    zeros = CertifiedRecord(eg.head, eg.salt, replace(eg.certificate, signature=bytes(len(eg.certificate.signature))))
    assert reject(run.round(run.lw.commit(0, zeros, ig, SALTS[0])))
    (tmp_path / "stranger").mkdir()
    stranger = Issuer(signers.signer(tmp_path / "stranger", seed=b"other"), signers.SESSION)
    signed = stranger.certify(ingress_records()[1], INGRESS, PARAMS, NETWORK_ATTESTER_DIGEST)
    assert reject(run.round(run.lw.commit(1, run.egress(egress_records()[1]), signed, SALTS[1])))
    assert run.round(run.honest(2)).accepted and run.stream.symbols == ("reject", "reject", "accept")


def test_a_certificate_from_another_session_is_a_reject(tmp_path):
    """The cross-session replay: the attester's own key signed the execution phase in an earlier session, and its
    certificate signs another message than this session's, so it is rejected."""
    run = Run(tmp_path)
    earlier = LmsSigner(run.signer.key, run.signer.state)
    replayed = Issuer(earlier, signers.OTHER_SESSION).certify(egress_records()[0], EGRESS, PARAMS, NETWORK_ATTESTER_DIGEST)
    earlier.close()
    assert replayed.verify(run.signer.public_key(), signers.OTHER_SESSION)
    assert reject(run.round(run.lw.commit(0, replayed, run.ingress(ingress_records()[0]), SALTS[0])))
    assert run.round(run.honest(1)).accepted and run.stream.symbols == ("reject", "accept")


def test_a_digest_mismatch_is_a_reject(tmp_path):
    """A head edited under its certificate (the certified counts) is rejected."""
    run = Run(tmp_path)
    eg, ig = run.honest(0).records
    counts = RecordHead(**{f: getattr(eg.head, f) for f in RecordHead.__slots__} | {"counts": (0,) * PARAMS.buckets})
    assert reject(run.round(run.lw.commit(0, CertifiedRecord(counts, eg.salt, eg.certificate), ig, SALTS[0])))


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
        sample = run.stream.select(reg)
        delivered = run.prove(opening, sample)
        assert delivered.output == 0b01
        verdict = run.lw.verify(sample, [r.certificate for r in opening.records], SYNCS, delivered,
                                circuit_sha512=standin.circuit_sha512(delivered))
        assert verdict["public"] == {"verdict": 0, "deadline": 0} and verdict["accepted"] is False
        assert "clear" not in verdict
        assert reject(run.decide(sample, opening))


def test_a_late_proof_is_a_reject(tmp_path):
    run = Run(tmp_path)
    certificates = [r.certificate for r in run.honest(0).records]
    opening = run.honest(0)
    sample = run.stream.select(run.register(opening))
    delivered = run.prove(opening, sample)
    run.clock.now = 60.5
    out = run.stream.deliver(sample, delivered.sessions())
    assert out is not None and not out.accepted and out.verdict.codes == ("late",) and out.public == {}
    with pytest.raises(S.Refused):
        run.lw.outcome(run.stream, sample, certificates, SYNCS, delivered,
                       circuit_sha512=standin.circuit_sha512(delivered))
    opening = run.honest(1)
    sample = run.stream.select(run.register(opening))
    delivered = run.prove(opening, sample)
    run.clock.now = 125.0
    assert run.lw.outcome(run.stream, sample, [r.certificate for r in opening.records], SYNCS, delivered,
                          circuit_sha512=standin.circuit_sha512(delivered)).verdict.codes == ("late",)
    assert run.stream.symbols == ("late", "late")


def test_the_outcome_binds_the_registered_root(tmp_path):
    """The same certified records under other salts open another root. A proof on them under the registered root
    opens no head to it, so its verdict is 0; one under their own root has a public row that isn't the verifier's own
    copy, which names the registered root, so it is refused unopened. Both are rejects, and the session's root is the
    registered one, the root the public row names. Accepted outcomes under two salts are two records."""
    run = Run(tmp_path)
    registered = run.honest(0)
    other = replace(registered, salts=SALTS[2])
    assert run.lw.root(other) != run.lw.root(registered)
    out = run.round(registered, delivered=other)
    assert reject(out)
    audit = run.stream.records[0]["audit"]
    assert audit["registration"]["roots"][0]["root"] == run.lw.root(registered)
    assert all(c["ok"] for c in audit["checks"].values()) and audit["verdicts"]["network-audit"]["accepted"] is False
    assert audit["verdicts"]["network-audit"]["public"] == {"verdict": 0, "deadline": 1}
    honest = run.honest(1)
    sample = run.stream.select(run.register(honest))
    mine = replace(honest, salts=SALTS[2])
    own = run.decide(sample, mine, root=bytes.fromhex(run.lw.root(mine)))
    assert reject(own) and run.stream.records[1]["audit"]["verdicts"]["network-audit"]["public"] == {
        "verdict": 0, "deadline": 0}

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


def test_the_opening_must_be_the_registered_phases(tmp_path):
    run = Run(tmp_path)
    opening = run.honest(0)
    sample = run.stream.select(run.register(opening))
    with pytest.raises(ValueError, match="execution phase"):
        run.lw.clear(run.honest(1), sample, SYNCS)
    delivered = run.prove(opening, sample)
    with pytest.raises(ValueError, match="execution phase"):
        run.lw.verify(sample, [r.certificate for r in opening.records], SYNCS, delivered,
                      circuit_sha512=standin.circuit_sha512(delivered), opening=run.honest(1))
    for salts in ((SALTS[0][0],) * 4, SALTS[0][:2]):
        with pytest.raises(ValueError, match="salts"):
            run.lw.commit(0, *opening.records, salts=salts)


def test_a_link_is_a_network_attesters_with_both_heads_under_its_digest_and_no_sync(tmp_path):
    """A link (`NetworkAttesterProfile`) is the network attester's records: its public items are `PUBLIC`, a window whose two
    records the network attester certified under its parameters digest is accepted with no sync declared, by the
    Lean's verdict of record, the Python beside it and the in-the-clear check alike, and a window whose egress record was
    certified under a shaper's digest instead is a reject. A link without a `NetworkAttesterProfile` has no verdict, and its two
    ends are one grid."""
    run = Run(tmp_path)
    lw = run.lw
    assert lw.spec.public == SVC.PUBLIC and lw.windows == 3 and SVC.OF_RECORD == "lean"
    honest = run.honest(0)
    sample = run.stream.select(run.register(honest))
    out = standin.decide(lw, run.stream, sample, honest, (), tmp_path / "p0")
    assert out.accepted and out.public == {"verdict": 1, "deadline": 1}
    assert lw.clear(honest, sample)["accepted"]
    verdict = run.stream.records[-1]["audit"]["verdicts"][SVC.VERIFIER]
    assert verdict["by"] == "lean" and verdict["reference"] == {"accepted": True}
    shaped = run.issuer.certify(egress_records()[1], EGRESS, PARAMS, parameters_digest(PARAMS, SYNC_SET))
    other = lw.commit(1, shaped, run.ingress(ingress_records()[1]), SALTS[1])
    assert reject(standin.decide(lw, run.stream, run.stream.select(run.register(other)), other, (), tmp_path / "p1"))
    verdict = run.stream.records[-1]["audit"]["verdicts"][SVC.VERIFIER]
    assert verdict["by"] == "lean" and verdict["reference"] == {"accepted": False}
    with pytest.raises(ValueError, match="the network attester's"):
        replace(lw, network_attester=None)
    with pytest.raises(ValueError, match="one grid"):
        replace(lw, ingress=replace(PARAMS, buckets=PARAMS.buckets + 1))
    with pytest.raises(ValueError, match="one or more windows"):
        SVC.NetworkAttesterProfile(0, NETWORK_ATTESTER_DIGEST)
    with pytest.raises(ValueError, match="64 bytes"):
        SVC.NetworkAttesterProfile(3, bytes(32))


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
    execution phase is rejected though both certificates verify."""
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
    sample = run.stream.select(reg)
    assert reject(run.decide(sample, opening))


def test_the_commitments_of_a_shared_row_file_are_its_instances_refs_rows(tmp_path):
    """A shared-row public file (`verity_flock.circuit.write`'s ``shared_rows``): each port's table port-major, then
    every instance's refs into them. An instance's commitments are its refs' rows, an instance two of whose ports share
    a row reads that row on each, and a ref past its table, a short file or a header naming other ports reads none."""
    bc = {(p, k): bytes([j * 16 + k]) * SVC.BC_BYTES for j, p in enumerate(SVC.PORTS) for k in range(3)}
    counts = {"pub": 1, "wit": 3, "o0": 2}
    refs = {7: {"pub": 0, "wit": 2, "o0": 1}, 9: {"pub": 0, "wit": 0, "o0": 0}}

    def write(name, header=None, refs_=refs, cut=0):
        head = {"units": {"indices": list(refs_)}, "shared_rows": counts, **(header or {})}
        body = b"".join(bc[p, k] for p in SVC.PORTS for k in range(counts[p]))
        body += b"".join(r[p].to_bytes(4, "little") for r in refs_.values() for p in SVC.PORTS)
        path = tmp_path / name
        path.write_bytes(json.dumps(head).encode() + b"\n" + body[:len(body) - cut])
        return path

    public = write("pub.bin")
    for u, r in refs.items():
        assert SVC._commitments(public, u) == {p: bc[p, r[p]] for p in SVC.PORTS}
    assert SVC._commitments(public, 8) is None
    assert SVC._commitments(write("past.bin", refs_={7: {"pub": 1, "wit": 0, "o0": 0}}), 7) is None
    assert SVC._commitments(write("short.bin", cut=1), 7) is None
    assert SVC._commitments(write("ports.bin", {"shared_rows": {"pub": 1, "wit": 3}}), 7) is None
