"""N2's link-window verdict of record in Lean (`NetworkCheck.LinkWindow`, `LinkWindows.lean_verdict`) held to the
Python (`LinkWindows.python_verdict`), the reference that keeps running beside it: the same accept bit, public outputs
and digest on plain, served, advised and warden link-windows, honest and tampered (the session refused, rejected, naming
another Program or other units; the public file's bytes flipped, cut, missing, naming another unit or its unit twice, or
in a shared-row layout; the salts and the output word forged; each certificate's fields edited, the certificates
swapped, missing or not certificates; the declared syncs off Σ_sync or missing; an advised window without an advice;
a warden link under another digest).
And what `verify` does with the two: the switch (`OF_RECORD`) names the verdict of record, a disagreement or a case the
Lean can't state (a key other than an `LmsKey`) rejects. And certifier/v0's period check (`NetworkCheck.Period`) opens
a plain link-window's stand-in proof as the verifier of record does (`network-period opening`)."""

import itertools
import json
import random
import subprocess
from dataclasses import replace
from fractions import Fraction

import pytest

import verity_difftest as D
from verity.ml.circuits.codec import canonical_json
from verity.core.protocols.certifier import lean_audit as LA, service as SVC
from verity.core.protocols.certifier.certificate import CertifiedRecord, Direction, RecordHead, parameters_digest
from verity.core.protocols.certifier.schedule import ClockSync
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.core.protocols.sampled_proofs.one_stage.proofs import Finding

import standin
import test_network_certifier_advised as advised
import test_network_certifier_served as served
import test_network_certifier_service as plain
from advised_traffic import serve


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


class Case:
    """One link-window's draw, certificates, syncs and delivered proof, with the findings of the stand-in Lean run on
    the honest proof."""

    def __init__(self, lw, opening, draw, syncs, tmp, advice=None):
        self.lw, self.draw, self.syncs, self.tmp, self.advice = lw, draw, syncs, tmp, advice
        self.certificates = [r.certificate for r in opening.records]
        self.delivered = standin.prove(lw, opening, draw, syncs, tmp / "honest", advice=advice)
        self.findings, refused = standin.lean(self.delivered.proof, circuit_sha512=standin.circuit_sha512(self.delivered),
                                              program=lw.program, partition=lw.partition)
        assert refused is None
        self._files = itertools.count()

    def both(self, *, certificates=None, syncs=None, delivered=None, findings=(), advice=()):
        """The Lean's and the Python's verdicts on the case with the given inputs replaced."""
        args = (self.draw, self.certificates if certificates is None else certificates,
                self.syncs if syncs is None else syncs, delivered or self.delivered,
                self.findings if findings == () else findings)
        sessions = args[3].sessions()
        advice = self.advice if advice == () else advice
        return (self.lw.lean_verdict(*args, sessions, advice), self.lw.python_verdict(*args, sessions, advice))

    def public(self, data):
        """The honest delivery with the public file's bytes `data` (None: no file)."""
        path = self.tmp / f"public-{next(self._files)}.bin"
        if data is not None:
            path.write_bytes(data)
        return replace(self.delivered, proof=replace(self.delivered.proof, public=path))


def _plain(tmp):
    run = plain.Run(tmp)
    opening = run.honest(0)
    return Case(run.lw, opening, run.stream.select(run.register(opening)), plain.SYNCS, tmp)


def _served(tmp):
    run = served.Run(tmp)
    egress = served.egress_records(served.frames_of(served.ARRIVALS))
    ingress = served.ingress_records(served.ARRIVALS)
    opening = run.lw.commit(0, run.issuer.certify(egress[0], Direction.EGRESS, served.P,
                                                  parameters_digest(served.P, served.SYNC_SET)),
                            run.issuer.certify(ingress[0], Direction.INGRESS, served.P, parameters_digest(served.P)),
                            served.SALTS[0])
    reg = run.lw.register(run.stream, run.lw.registration(opening, None))
    return Case(run.lw, opening, run.stream.select(reg), served.SYNCS, tmp)


def _advised(tmp):
    run, session = advised.Run(tmp), serve(advised.SEEDS[0])
    opening, draw = run.draw(0, session.egress[0], session.ingress[0])
    return Case(run.lw, opening, draw, advised.SYNCS, tmp, advice=session.link)


def _warden(tmp):
    """The plain link as a warden's (`WardenProfile`): both records certified under the warden's parameters digest, no
    occupancy profile."""
    run = plain.Run(tmp)
    run.lw = lw = replace(run.lw, occupancy=(None, None), warden=SVC.WardenProfile(3, plain.WARDEN_DIGEST))
    run.stream = lw.stream(budget=8, delta=plain.DELTA, challenger="challenger-test", coins=lambda n: b"\x07" * n,
                           clock=run.clock)
    warden = lambda d, r: run.issuer.certify(r, d, plain.PARAMS, plain.WARDEN_DIGEST)
    opening = lw.commit(0, warden(plain.EGRESS, plain.egress_records()[0]),
                        warden(plain.INGRESS, plain.ingress_records()[0]), plain.SALTS[0])
    return Case(lw, opening, run.stream.select(run.register(opening)), plain.SYNCS, tmp)


KINDS = {"plain": _plain, "served": _served, "advised": _advised, "warden": _warden}


def _header(data):
    head, body = data.split(b"\n", 1)
    return json.loads(head), body


def _file(head, body):
    return json.dumps(head).encode() + b"\n" + body


def _shared(data, extra=0, first=False):
    """The public file as a shared-row file: each port's table, `extra` instances of unit 1 beside the drawn unit (the
    drawn one first or last), then the refs."""
    head, body = _header(data)
    unit, k = head["units"]["indices"][0], len(SVC.PORTS)
    bcs = [body[j * SVC.BC_BYTES:(j + 1) * SVC.BC_BYTES] for j in range(k)]
    n = extra + 1
    mine = 0 if first else extra
    tables = b"".join(b"".join(bcs[j] if i == mine else bytes([i + 1]) * SVC.BC_BYTES for i in range(n))
                      for j in range(k))
    refs = b"".join(i.to_bytes(4, "little") for i in range(n) for _ in range(k))
    indices = [unit if i == mine else unit + 1 for i in range(n)]
    return _file({"units": {"indices": indices}, "instances": n, "shared_rows": {p: n for p in SVC.PORTS}},
                 tables + refs)


def _public_files(c, rng):
    data = c.delivered.proof.public.read_bytes()
    head, body = _header(data)
    unit = head["units"]["indices"][0]
    out = [None, b"", data[:-1], data[:-SVC.BC_BYTES], data.split(b"\n")[0], data + b"\0" * 7, b"not json\n" + body,
           _file({"units": {"indices": [unit + 1]}}, body), _file({"units": {"indices": [unit, unit]}}, body * 2),
           _file({"units": {"indices": [unit + 1, unit]}}, bytes(len(body)) + body),
           _file({"units": {"indices": [unit, unit + 1]}}, body + bytes(len(body))),
           _file({"units": {"indices": [str(unit)]}}, body), _file({"units": {}}, body), _file([], body),
           _shared(data), _shared(data, 2), _shared(data, 2, first=True),
           _file({**_header(_shared(data))[0], "shared_rows": None}, body)]
    shared_head, shared_body = _header(_shared(data, 1))
    out += [_file(shared_head, shared_body[:-1]),
            _file({**shared_head, "shared_rows": {"pub": 2, "wit": 2}}, shared_body),
            _file({**shared_head, "shared_rows": {"pub": 2, "wit": 2, "o0": 0}}, shared_body),
            _file({**shared_head, "shared_rows": {**shared_head["shared_rows"], "x": 1}}, shared_body),
            _file(shared_head, shared_body[:-4] + (9).to_bytes(4, "little"))]
    for _ in range(24):
        flipped = bytearray(data)
        i = rng.randrange(len(head) if rng.random() < 0.2 else len(data))
        flipped[i] ^= 1 << rng.randrange(8)
        out.append(bytes(flipped))
    return out


def _findings(c):
    f = c.findings[0]
    return [None, [], [replace(f, accepted=False)], [replace(f, program=None)], [replace(f, program="0" * 128)],
            [replace(f, units=(1,))], [replace(f, units=())], [f, f], [f, replace(f, accepted=False)]]


def _certificates(c):
    e, i = c.certificates
    flip = lambda b, k=0: b[:k] + bytes([b[k] ^ 1]) + b[k + 1:]
    edits = [{"digest": flip(e.digest)}, {"digest": e.digest[:-1]}, {"signature": flip(e.signature, 40)},
             {"signature": e.signature[:-1]}, {"window": e.window + 1}, {"window": -1},
             {"direction": Direction.INGRESS}, {"link": e.link + "x"}, {"key_id": flip(e.key_id)}, {"key_id": "x"}]
    return ([[replace(e, **kw), i] for kw in edits] + [[e, replace(i, digest=flip(i.digest))], [i, e], [e], [e, i, i],
            [None, i], [e, None], [object(), i], []])


def _syncs(c):
    w = c.draw.registration.record["window"]["window"]
    s = c.syncs[w]
    moved = lambda new: c.syncs[:w] + [new] + c.syncs[w + 1:]
    return [[], c.syncs[:w], moved(ClockSync(s.delta + 7, s.rho)), moved(ClockSync(s.delta, s.rho * 3)),
            moved(ClockSync(1 << 16, s.rho)), moved(ClockSync(s.delta, Fraction(1 << 16, 1))),
            moved(ClockSync(s.delta, s.rho * Fraction(1, 1 << 16)))]


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_the_lean_verdict_is_the_pythons(tmp_path, kind):
    """On every case the Lean gives the Python's accept bit, public outputs and digest; the honest proof is accepted,
    with both bits proved."""
    c = KINDS[kind](tmp_path)
    rng = random.Random(f"link-window/{kind}")
    lean, python = c.both()
    assert lean == python and python["accepted"] and python["public"]["verdict"] == 1 == python["public"]["deadline"]
    cases = [{"delivered": c.public(d)} for d in _public_files(c, rng)]
    cases += [{"findings": f} for f in _findings(c)]
    cases += [{"delivered": replace(c.delivered, salts={**c.delivered.salts, **s})} for s in (
        {"pub": bytes(192)}, {"o0": bytes(192)}, {"pub": c.delivered.salts["pub"][:-1]}, {"pub": "x"}, {"pub": None})]
    cases += [{"delivered": replace(c.delivered, salts={"o0": c.delivered.salts["o0"]})}]
    cases += [{"delivered": replace(c.delivered, output=w)} for w in (0, 1, 2, 4, 1 << 16, -1, "3", None, True)]
    cases += [{"certificates": cs} for cs in _certificates(c)]
    if kind != "warden":
        cases += [{"syncs": s} for s in _syncs(c)]
    if kind == "advised":
        cases += [{"advice": None}, {"advice": "not an advice"}]
    rejected = 0
    for k, kw in enumerate(cases):
        lean, python = c.both(**kw)
        assert lean == python, (k, kw)
        rejected += not python["accepted"]
    assert rejected >= len(cases) - 30


def test_a_warden_link_window_reads_the_wardens_digest_and_no_sync(tmp_path):
    """On a warden link the Lean holds both ends' grids to the warden's parameters digest and reads no sync, as the
    Python does: syncs missing or off Σ_sync change neither verdict; the link under another warden's digest, or as the
    plain link whose grids carry the certifier's digests, is a reject by both; and a warden case the Python never makes
    (a digest of another length or none, an end with a profile, two grids) is refused, which `verify` rejects."""
    c = _warden(tmp_path)
    honest = c.both()
    assert honest[0] == honest[1] and honest[1]["accepted"] and honest[1]["public"] == {"verdict": 1, "deadline": 1}
    for s in _syncs(c):
        assert c.both(syncs=s) == honest
    lw = c.lw
    for other in (replace(lw, warden=SVC.WardenProfile(3, bytes(64))), replace(lw, warden=None)):
        assert other.program == lw.program
        c.lw = other
        lean, python = c.both()
        assert lean == python and not python["accepted"]
    c.lw = lw
    case = lw.lean_case(c.draw, c.certificates, c.syncs, c.delivered, c.findings, c.delivered.sessions())
    assert case["kind"] == "warden" and case["warden"] == plain.WARDEN_DIGEST.hex()
    assert SVC.LA.SERVER.call("link-window", case)["accepted"]
    grid = case["ingress"]["grid"]
    for edit in ({"warden": case["warden"][:-2]}, {"warden": case["warden"] + "00"}, {"warden": None},
                 {"warden": 7}, {"egress": {**case["egress"], "profile": [[0] * grid["buckets"]]}},
                 {"ingress": {**case["ingress"], "grid": {**grid, "allocation": grid["allocation"] + 1}}},
                 {"ingress": {**case["ingress"], "grid": {**grid, "queue_capacity": grid["queue_capacity"] + 1}}}):
        with pytest.raises(ValueError):
            SVC.LA.SERVER.call("link-window", {**case, **edit})
    del case["warden"]
    with pytest.raises(ValueError):
        SVC.LA.SERVER.call("link-window", case)


def test_a_disagreement_or_a_case_the_lean_cant_state_rejects(tmp_path, monkeypatch):
    """The switch names whose verdict is the record's (`by`), the other's accept bit is `reference`'s; where the two
    differ, or the Lean has no case (a key other than an `LmsKey`), the link-window is rejected with deadline 0."""
    c = _plain(tmp_path)
    run = lambda lw=c.lw: lw.verify(c.draw, c.certificates, c.syncs, c.delivered,
                                    circuit_sha512=standin.circuit_sha512(c.delivered))
    lean = run()
    assert lean["by"] == "lean" and lean["accepted"] and lean["reference"] == {"accepted": True}
    monkeypatch.setitem(SVC.OF_RECORD, "plain", "python")
    python = run()
    assert python["by"] == "python" and {k: v for k, v in python.items() if k != "by"} == \
        {k: v for k, v in lean.items() if k != "by"}
    monkeypatch.setitem(SVC.OF_RECORD, "plain", "lean")
    real = SVC.LinkWindows.lean_verdict
    monkeypatch.setattr(SVC.LinkWindows, "lean_verdict",
                        lambda self, *a, **kw: {**real(self, *a, **kw), "public": {"verdict": 1, "deadline": 0}})
    v = run()
    assert not v["accepted"] and v["public"] == {"verdict": 0, "deadline": 0} and v["reference"] == {"accepted": True}
    monkeypatch.setattr(SVC.LinkWindows, "lean_verdict", real)

    class Wrapped:
        """The link's LMS key behind another type, which `lean_case` doesn't state."""

        def __init__(self, key):
            self.key = key

        @property
        def key_id(self):
            return self.key.key_id

        def verify(self, message, signature):
            return self.key.verify(message, signature)

    other = replace(c.lw, key=Wrapped(c.lw.key))
    assert other.lean_case(c.draw, c.certificates, c.syncs, c.delivered, c.findings, c.delivered.sessions()) is None
    v = run(other)
    assert not v["accepted"] and v["public"] == {"verdict": 0, "deadline": 0} and v["reference"] == {"accepted": True}


def _cert(c) -> dict:
    return {"digest": c.digest.hex(), "direction": c.direction.value, "key_id": c.key_id.hex(), "link": c.link,
            "signature": c.signature.hex(), "window": c.window}


def _end(grid, profile) -> dict:
    return {"grid": {k: getattr(grid, k) for k in ("allocation", "bucket_ms", "buckets", "frame_bytes", "queue_capacity")},
            "profile": None if profile is None else [list(row) for row in profile]}


def period_case(lw, draw, body: dict, delivered, certificates) -> tuple[dict, bytes, bytes, list]:
    """certifier/v0's clock-form case of link-window 0 alone, as the all-protocols leg's `case_of` makes it from N2's
    files (`n2_zk.one_stage_params`, `one_stage_evidence`): the link's ends, run and one-stage v2 parameters, the
    registration, the stream's audit record of the window's outcome (``body``) as v2's evidence with the delivered
    proof's files (`proofs.evidence_v2`), the delivered salts and the certificates; and the files as blobs."""
    audit, reg, key = body["audit"], draw.registration, lw.key
    proofs, blobs = PF.evidence_v2(delivered.proof, body["sessions"])
    one_stage = {"draw": audit["draw"], "instances": len(audit["proved_units"]), "proofs": proofs,
                 "proved_units": audit["proved_units"], "receipt": audit["receipt"], "served": None,
                 "session_roots": {SVC.VALUE: reg.record["roots"][0]["root"]}, "sessions": list(body["sessions"]),
                 "statement": audit["statement"]}
    params = {"audits": [], "cells": [], "egress": _end(lw.params, lw.occupancy[0]), "first": 0,
              "ingress": _end(lw.ingress_params, lw.occupancy[1]), "jitter": 1, "key": key.public.hex(),
              "link": lw.link, "lms": {"H": key.params.H, "N": key.params.N, "W": key.params.W, "toy": key.params.toy},
              "one_stage": {"anchors": [], "delta": str(Fraction(plain.DELTA)), "law": S.selection_law(lw.spec.selection, 1),
                            "leaf_layer": None, "log": [], "partition": lw.partition, "program": json.loads(lw.program),
                            "proofs": {"circuit": standin.circuit_sha512(delivered), "registered": None},
                            "query": lw.query, "registered": None, "scheme": S.SCHEME, "work_table": None},
              "run": lw.run, "session": lw.certifier_session.hex(),
              "sync_set": {"deltas": list(lw.sync_set.deltas),
                           "rhos": [[r.numerator, r.denominator] for r in lw.sync_set.rhos]},
              "windows": 1}
    evidence = {"audits": [], "windows": [{
        "certificates": [_cert(c) for c in certificates], "clock": _cert(certificates[0]), "one_stage": one_stage,
        "salts": {k: delivered.salts[k].hex() for k in ("o0", "pub")}}]}
    return (params, canonical_json({"audits": [], "windows": [reg.record]}), canonical_json(evidence), blobs)


def opened(tmp_path) -> tuple:
    """A plain link's honest link-window 0 through its outcome on N2's stand-in proof: (run, opening, draw, delivered,
    the stream's record of the outcome)."""
    run = plain.Run(tmp_path)
    opening = run.honest(0)
    draw = run.stream.select(run.register(opening))
    delivered = standin.prove(run.lw, opening, draw, plain.SYNCS, tmp_path / "honest")
    assert run.stream.deliver(draw, delivered.sessions()) is None
    out = run.lw.outcome(run.stream, draw, [r.certificate for r in opening.records], plain.SYNCS, delivered,
                         circuit_sha512=standin.circuit_sha512(delivered))
    assert out.accepted
    return run, opening, draw, delivered, run.stream.records[-1]


def test_the_period_check_opens_a_link_windows_proof_as_the_verifier_of_record_does(tmp_path):
    """`network-period opening` (`NetworkCheck.Period.openings`: the period check's reading of a link-window once
    one-stage v2 accepted its proof, which no stand-in proof gets from the real verifier) on certifier/v0's case of a
    plain link's link-window, made from the stream's record as the all-protocols leg makes it from N2's files: the
    honest proof's public row opens to the verifier's own and its output row to the word 3; a proof on a head off the
    profile (verdict bit 0), a forged pub or o0 salt, a recertified certificate and the certificates swapped are each
    refused at the opening by name."""
    exe = D.build(LA.PACKAGE, "network-period")
    run, opening, draw, delivered, body = opened(tmp_path)
    certs = [r.certificate for r in opening.records]
    eg, ig = opening.records
    zero = RecordHead(**{f: getattr(eg.head, f) for f in RecordHead.__slots__} | {"counts": (0,) * plain.PARAMS.buckets})
    off = run.lw.commit(0, CertifiedRecord(zero, eg.salt, eg.certificate), ig, plain.SALTS[0])

    def opens(d=delivered, cs=certs):
        params, commitment, evidence, blobs = period_case(run.lw, draw, body, d, cs)
        files = [tmp_path / f for f in ("params.json", "commitment.bin", "evidence.json")]
        for f, data in zip(files, (canonical_json(params), commitment, evidence)):
            f.write_bytes(data)
        p = subprocess.run([str(exe), "opening", *map(str, files), *map(str, blobs)], capture_output=True, text=True)
        out = json.loads(p.stdout.strip().splitlines()[-1])
        assert p.returncode == 0 and "refused" not in out, out
        return out["windows"]

    assert opens() == [None]
    refused = [opens(standin.prove(run.lw, off, draw, plain.SYNCS, tmp_path / "off")),
               opens(replace(delivered, salts={**delivered.salts, "pub": bytes(192)})),
               opens(replace(delivered, salts={**delivered.salts, "o0": bytes(192)})),
               opens(cs=[run.honest(0).records[0].certificate, certs[1]]), opens(cs=certs[::-1])]
    assert all(len(w) == 1 and w[0].startswith("window 0: opening: ") for w in refused), refused
