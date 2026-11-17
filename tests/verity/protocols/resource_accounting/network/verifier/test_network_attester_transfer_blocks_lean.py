"""N2's execution phase verdict of record in Lean (`NetworkCheck.TransferBlocks`, `TransferBlocks.lean_verdict`) held to the
Python (`TransferBlocks.python_verdict`), the reference that keeps running beside it: the same accept bit, public outputs
and digest on a network attester's execution phases, honest and tampered (the session refused, rejected, naming another
Program or other units; the public file's bytes flipped, cut, missing, naming another unit or its unit twice, or in a
shared-row layout; the salts and the output word forged; each attestation's fields edited, the attestations swapped,
missing or not attestations; the link under another digest). The Lean's other kinds of link (`Kind.plain`,
`Kind.served`) are cases the Python never makes.
And what `verify` does with the two: the switch (`OF_RECORD`) names the verdict of record, a disagreement or a case the
Lean can't state (a key other than an `LmsKey`) rejects."""

import itertools
import json
import random
from dataclasses import replace
from fractions import Fraction

import pytest

from verity.protocols.resource_accounting.network.verifier import service as SVC
from verity.protocols.resource_accounting.network.verifier.certificate import Direction
from verity.protocols.resource_accounting.network.verifier.schedule import ClockSync

import standin
import test_network_attester_service as plain


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


class Case:
    """One execution phase's sample, attestations, syncs and delivered proof, with the findings of the stand-in Lean run
    on the honest proof."""

    def __init__(self, lw, opening, sample, syncs, tmp):
        self.lw, self.sample, self.syncs, self.tmp = lw, sample, syncs, tmp
        self.certificates = [r.certificate for r in opening.records]
        self.delivered = standin.prove(lw, opening, sample, syncs, tmp / "honest")
        self.findings, refused = standin.lean(self.delivered.proof, circuit_sha512=standin.circuit_sha512(self.delivered),
                                              program=lw.program, partition=lw.partition)
        assert refused is None
        self._files = itertools.count()

    def both(self, *, certificates=None, syncs=None, delivered=None, findings=()):
        """The Lean's and the Python's verdicts on the case with the given inputs replaced."""
        args = (self.sample, self.certificates if certificates is None else certificates,
                self.syncs if syncs is None else syncs, delivered or self.delivered,
                self.findings if findings == () else findings)
        sessions = args[3].sessions()
        return self.lw.lean_verdict(*args, sessions), self.lw.python_verdict(*args, sessions)

    def public(self, data):
        """The honest delivery with the public file's bytes `data` (None: no file)."""
        path = self.tmp / f"public-{next(self._files)}.bin"
        if data is not None:
            path.write_bytes(data)
        return replace(self.delivered, proof=replace(self.delivered.proof, public=path))


def _network_attester(tmp):
    """The network attester's link: both records attested under its parameters digest, no occupancy profile."""
    run = plain.Run(tmp)
    opening = run.honest(0)
    return Case(run.lw, opening, run.stream.select(run.register(opening)), plain.SYNCS, tmp)


def _header(data):
    head, body = data.split(b"\n", 1)
    return json.loads(head), body


def _file(head, body):
    return json.dumps(head).encode() + b"\n" + body


def _shared(data, extra=0, first=False):
    """The public file as a shared-row file: each port's table, `extra` instances of proof block 1 beside the sampled
    proof block (the sampled one first or last), then the refs."""
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


def _attestations(c):
    e, i = c.certificates
    flip = lambda b, k=0: b[:k] + bytes([b[k] ^ 1]) + b[k + 1:]
    edits = [{"digest": flip(e.digest)}, {"digest": e.digest[:-1]}, {"signature": flip(e.signature, 40)},
             {"signature": e.signature[:-1]}, {"window": e.window + 1}, {"window": -1},
             {"direction": Direction.INGRESS}, {"link": e.link + "x"}, {"key_id": flip(e.key_id)}, {"key_id": "x"}]
    return ([[replace(e, **kw), i] for kw in edits] + [[e, replace(i, digest=flip(i.digest))], [i, e], [e], [e, i, i],
            [None, i], [e, None], [object(), i], []])


def _syncs(c):
    w = c.sample.docket.record["window"]["window"]
    s = c.syncs[w]
    moved = lambda new: c.syncs[:w] + [new] + c.syncs[w + 1:]
    return [[], c.syncs[:w], moved(ClockSync(s.delta + 7, s.rho)), moved(ClockSync(s.delta, s.rho * 3)),
            moved(ClockSync(1 << 16, s.rho)), moved(ClockSync(s.delta, Fraction(1 << 16, 1))),
            moved(ClockSync(s.delta, s.rho * Fraction(1, 1 << 16)))]


def test_the_lean_verdict_is_the_pythons(tmp_path):
    """On every case the Lean gives the Python's accept bit, public outputs and digest; the honest proof is accepted,
    with both bits proved."""
    c = _network_attester(tmp_path)
    rng = random.Random("link-window/warden")
    lean, python = c.both()
    assert lean == python and python["accepted"] and python["public"]["verdict"] == 1 == python["public"]["deadline"]
    cases = [{"delivered": c.public(d)} for d in _public_files(c, rng)]
    cases += [{"findings": f} for f in _findings(c)]
    cases += [{"delivered": replace(c.delivered, salts={**c.delivered.salts, **s})} for s in (
        {"pub": bytes(192)}, {"o0": bytes(192)}, {"pub": c.delivered.salts["pub"][:-1]}, {"pub": "x"}, {"pub": None})]
    cases += [{"delivered": replace(c.delivered, salts={"o0": c.delivered.salts["o0"]})}]
    cases += [{"delivered": replace(c.delivered, output=w)} for w in (0, 1, 2, 4, 1 << 16, -1, "3", None, True)]
    cases += [{"certificates": cs} for cs in _attestations(c)]
    rejected = 0
    for k, kw in enumerate(cases):
        lean, python = c.both(**kw)
        assert lean == python, (k, kw)
        rejected += not python["accepted"]
    assert rejected >= len(cases) - 30


def test_a_network_attester_phase_reads_the_network_attesters_digest_and_no_sync(tmp_path):
    """On a network attester link the Lean holds both ends' grids to the network attester's parameters digest and
    reads no sync, as the Python does: syncs missing or off Σ_sync change neither verdict; the link under another
    network attester's digest is a reject by both; and a network attester case the Python never makes (a digest of another length or none, an end with a profile, two
    grids) is refused, which `verify` rejects."""
    c = _network_attester(tmp_path)
    honest = c.both()
    assert honest[0] == honest[1] and honest[1]["accepted"] and honest[1]["public"] == {"verdict": 1, "deadline": 1}
    for s in _syncs(c):
        assert c.both(syncs=s) == honest
    lw = c.lw
    other = replace(lw, network_attester=SVC.NetworkAttesterProfile(3, bytes(64)))
    assert other.program == lw.program
    c.lw = other
    lean, python = c.both()
    assert lean == python and not python["accepted"]
    c.lw = lw
    case = lw.lean_case(c.sample, c.certificates, c.syncs, c.delivered, c.findings, c.delivered.sessions())
    assert case["kind"] == "warden" and case["warden"] == plain.NETWORK_ATTESTER_DIGEST.hex()
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


def test_an_advised_phase_is_refused_by_name(tmp_path):
    """No advised link has a verdict of record: the Lean refuses a case of kind ``advised`` by name, its view and advice
    included or not, as it does a kind it doesn't know."""
    c = _network_attester(tmp_path)
    case = c.lw.lean_case(c.sample, c.certificates, c.syncs, c.delivered, c.findings, c.delivered.sessions())
    assert case["kind"] == "warden" and SVC.LA.SERVER.call("link-window", case)["accepted"]
    view = {"view": {"released": [0], "floor": [0], "received": 0}, "advice": "00" * 64}
    for advised in ({**case, "kind": "advised"}, {**case, "kind": "advised", **view}):
        with pytest.raises(ValueError, match="an advised link is refused"):
            SVC.LA.SERVER.call("link-window", advised)
    with pytest.raises(ValueError, match="plain, served or a network certifier's, not shaped"):
        SVC.LA.SERVER.call("link-window", {**case, "kind": "shaped"})


def test_a_disagreement_or_a_case_the_lean_cant_state_rejects(tmp_path, monkeypatch):
    """The switch names whose verdict is the record's (`by`), the other's accept bit is `reference`'s; where the two
    differ, or the Lean has no case (a key other than an `LmsKey`), the execution phase is rejected with deadline 0."""
    c = _network_attester(tmp_path)
    run = lambda lw=c.lw: lw.verify(c.sample, c.certificates, c.syncs, c.delivered,
                                    circuit_sha512=standin.circuit_sha512(c.delivered))
    lean = run()
    assert lean["by"] == "lean" and lean["accepted"] and lean["reference"] == {"accepted": True}
    monkeypatch.setattr(SVC, "OF_RECORD", "python")
    python = run()
    assert python["by"] == "python" and {k: v for k, v in python.items() if k != "by"} == \
        {k: v for k, v in lean.items() if k != "by"}
    monkeypatch.setattr(SVC, "OF_RECORD", "lean")
    real = SVC.TransferBlocks.lean_verdict
    monkeypatch.setattr(SVC.TransferBlocks, "lean_verdict",
                        lambda self, *a, **kw: {**real(self, *a, **kw), "public": {"verdict": 1, "deadline": 0}})
    v = run()
    assert not v["accepted"] and v["public"] == {"verdict": 0, "deadline": 0} and v["reference"] == {"accepted": True}
    monkeypatch.setattr(SVC.TransferBlocks, "lean_verdict", real)

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
    assert other.lean_case(c.sample, c.certificates, c.syncs, c.delivered, c.findings, c.delivered.sessions()) is None
    v = run(other)
    assert not v["accepted"] and v["public"] == {"verdict": 0, "deadline": 0} and v["reference"] == {"accepted": True}
