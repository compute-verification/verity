"""`core-firewall serve` (`Firewall.Protocol`, `verity/firewall-release/v0`) against a reference built on Python's
ZK server: the case framing as `Core.Protocol` decodes it (`verity_difftest.protocol.answer`), the record as
`verity_flock.firewall` reads it, its slot and admission, the send check and the release time. The send check is
`Flock.Firewall.Contract.run` on both sides: the port reuses it, so the reference asks `flock-firewall contract` for its
verdict, and what is compared is everything around it. Transcripts are `firewall_agree.py`'s sessions (honest ones, the
red team's, tamper controls) and review r6's leaks, under the honest public input, a leak's own, or one that doesn't decode;
records and outer sessions are `_release.scene`'s."""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pytest
from hypothesis import strategies as st

import _release as R
import verity_difftest as D
from verity_difftest import protocol as P
from verity_flock import firewall as FW

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ID = "verity/firewall-release/v0"
FORMAT = "verity/firewall-transcript/v0"
SEED = 0


class Contract:
    """`flock-firewall contract`'s verdicts on canonical texts, each (public input, transcript) asked once: a public input
    it can't read gives `{"public": why}`, a transcript it can't read `{"error": why}`, else `{"accepted", "released",
    "why"}`. The public input `None` is the empty one, which says whether a transcript reads."""

    def __init__(self, exe: Path, tmp: Path):
        self.exe, self.tmp, self.memo = exe, tmp, {}

    def ask(self, pub: str | None, transcripts: list[str]) -> None:
        todo = [t for t in dict.fromkeys(transcripts) if (pub, t) not in self.memo]
        if not todo:
            return
        d = Path(tempfile.mkdtemp(dir=self.tmp))
        files = [d / f"{k}.json" for k in range(len(todo))]
        for f, t in zip(files, todo):
            f.write_text(t)
        argv = [str(self.exe), "contract"]
        if pub is not None:
            (d / "public.json").write_text(pub)
            argv += ["--public", str(d / "public.json")]
        lines = [json.loads(x) for x in subprocess.run(argv + [str(f) for f in files], capture_output=True, text=True).stdout.splitlines()]
        if len(lines) == 1 and "public" in lines[0]:
            self.memo |= {(pub, t): {"public": lines[0]["error"]} for t in todo}
            return
        got = {v.pop("file"): v for v in lines}
        self.memo |= {(pub, t): got[str(f)] for f, t in zip(files, todo)}

    def verdict(self, pub: str | None, transcript: str) -> dict:
        self.ask(pub, [transcript])
        return self.memo[(pub, transcript)]


@functools.cache
def pool() -> tuple[list, list, list, list]:
    """(transcripts the contract accepts under the honest public input, the others, public inputs, ones that don't
    decode): the honest public input first."""
    FA = R.firewall_agree()
    # every case on the honest case's coins and challenger's salts, so that the committed cases share one public input
    sessions = [(FA.run_case(n, r, p, f, rand=FA._rand(b"honest"), coin_salt=FA._salts(b"challenger:honest"), challenger=c), e)
                for n, r, p, f, e, c in FA.cases()]
    leaks = FA.leaks()
    pub = sessions[0][0].public()
    accepted = [e["lean"] and s.public() == pub for s, e in sessions]
    honest = [s.transcript() for (s, _), a in zip(sessions, accepted) if a]
    others = [s.transcript() for (s, _), a in zip(sessions, accepted) if not a] + [t for _, t, _, _ in leaks]
    h = honest[0]
    others += [{"kind": "nope"}, {k: v for k, v in h.items() if k != "items"}, h | {"items": [7]},
               h | {"tape": {"pads": [], "salts": ["zz"]}}, 7, []]
    pubs = list({D.canonical(p): p for p in [pub] + [p for _, _, p, _ in leaks]}.values())
    bad = [{}, pub | {"kind": "verity/firewall-public/v1"}, pub | {"statement": 7}, pub | {"relations": [1]}, [pub], "x", None]
    return honest, others, pubs, bad


@contextmanager
def _firewall(record: bytes):
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "record.jsonl").write_bytes(record)
        yield FW.Firewall(Path(d))


def reference(contract: Contract, tally: Counter | None = None):
    """The ZK server's release in Python, as `Firewall.Protocol` decides it: parameters are a public input the contract
    reads; the commitment a record Python's ZK server reads (its digest the SHA-512 of its bytes); the evidence canonical
    JSON `{"now_ms", "part", "session", "transcript"}` with a transcript the contract reads; the check takes the slot,
    admits the session (a record its own ops left unreadable rejects), holds the transcript to the contract, then the
    release time."""

    def params(j):
        pub = D.canonical(j)
        if "public" in (v := contract.verdict(pub, "{}")):
            raise D.Reject(v["public"])
        return pub

    def commitment(b: bytes) -> bytes:
        try:
            b.decode("utf-8")
        except UnicodeDecodeError:
            raise D.Reject("the record is not UTF-8") from None
        with _firewall(b) as fw:
            try:
                fw.record()
            except json.JSONDecodeError as e:
                raise D.Reject(f"the record: {e}") from None
        return b

    def evidence(b: bytes):
        j = P.canonical_json(b)
        if not (isinstance(j, dict) and all(isinstance(j.get(k), str) for k in ("session", "part")) and "transcript" in j):
            raise D.Reject("the evidence: a session and a part, strings, and a transcript")
        now, t = P.nat(j, "now_ms"), D.canonical(j["transcript"])
        if "error" in (v := contract.verdict(None, t)):
            raise D.Reject(v["error"])
        return j["session"], j["part"], now, j["transcript"], t

    def check(pub: str, record: bytes, ev) -> tuple:
        s, p, now, t, text = ev
        with _firewall(record) as fw:
            try:
                with fw.slot(s, p):
                    try:
                        at = fw.admit(s, p, timed=True)
                    except FW.Refused as e:
                        return "reject", str(e)
                    stream = next(e["stream"] for e in fw.record() if e["event"] == "open" and e["session"] == s)
            except json.JSONDecodeError as e:
                return "reject", f"the record, after the session's slot: {e}"
        if not (v := contract.verdict(pub, text))["accepted"]:
            return "reject", f"the send check refused: {v['why']}"
        if at is not None and now > at:
            return "reject", f"missed its release time ({now} ms, after {at} ms)"
        return ("accept", {"part": p, "release_ms": at, "session": s, "stream": stream},
                {"contract": FORMAT, "items": len(t["items"]), "phase": t["phase"]})

    def answer(case):
        out = P.answer(ID, case, params=params, commitment=commitment, evidence=evidence, check=check,
                       digest=lambda b: hashlib.sha512(b).digest())
        if tally is not None:
            why = out["reason"] or ""
            tally[out["verdict"] + next((f":{k}" for k in ("case", "params", "commitment", "evidence", "FIREWALL-REFUSED",
                                                           "the send check", "missed", "the record") if why.startswith(k)), "")] += 1
        return out

    return answer


@st.composite
def valid(draw) -> dict:
    honest, others, pubs, bad = pool()
    record, s, p, _ = draw(R.scene(("served", "served", "served", "budget", "none")))
    t = draw(st.one_of(st.sampled_from(honest), st.sampled_from(honest), st.sampled_from(others)))
    params = draw(st.one_of(st.just(pubs[0]), st.just(pubs[0]), st.just(pubs[0]), st.sampled_from(pubs), st.sampled_from(bad)))
    ev = {"now_ms": draw(st.integers(0, 150)), "part": p, "session": s, "transcript": t}
    return {"commitment": record.encode().hex(), "evidence": D.canonical(ev).encode().hex(), "params": params}


def _evidence(case: dict, f) -> dict:
    return case | {"evidence": D.canonical(f(json.loads(bytes.fromhex(case["evidence"])))).encode().hex()}


#: One change to a case: hex `Flock.unhex` refuses, a field missing or not a string, a record that is not UTF-8, evidence
#: not canonical, without its transcript, its clock or session not of their type.
MUTATIONS = [
    lambda c, x: c | {"commitment": c["commitment"].upper()},
    lambda c, x: c | {"evidence": c["evidence"].upper()},
    lambda c, x: c | {"evidence": c["evidence"][:-1]},
    lambda c, x: {k: v for k, v in c.items() if k != ["commitment", "evidence", "params"][x % 3]},
    lambda c, x: c | {["commitment", "evidence"][x % 2]: x},
    lambda c, x: c | {"commitment": "ff" + c["commitment"]},
    lambda c, x: c | {"evidence": json.dumps(json.loads(bytes.fromhex(c["evidence"]))).encode().hex()},
    lambda c, x: _evidence(c, lambda e: {k: v for k, v in e.items() if k != "transcript"}),
    lambda c, x: _evidence(c, lambda e: e | {"now_ms": -1 - x}),
    lambda c, x: _evidence(c, lambda e: e | {"session": x}),
]
CASES = st.one_of(valid(), valid(), valid(),
                  st.builds(lambda c, m, x: MUTATIONS[m](c, x), valid(), st.integers(0, len(MUTATIONS) - 1), st.integers(0, 63)))


def port(ref) -> D.ProtocolPort:
    return D.ProtocolPort("core-firewall", ref, CASES, package=R.PKG)


@pytest.fixture(scope="module")
def server():
    with D.Server(port(None).argv()) as s:
        yield s


@pytest.fixture(scope="module")
def contract(tmp_path_factory) -> Contract:
    c = Contract(D.build(R.LEAN_FLOCK, "flock-firewall"), tmp_path_factory.mktemp("contract"))
    honest, others, pubs, bad = pool()
    texts = [D.canonical(t) for t in honest + others] + ["{}"]
    for pub in [None, *(D.canonical(p) for p in pubs + bad)]:
        c.ask(pub, texts)
    return c


def _case(ops: list, session: str, part: str, now: int, transcript: dict | None = None, params=None) -> dict:
    honest, _, pubs, _ = pool()
    ev = {"now_ms": now, "part": part, "session": session, "transcript": transcript or honest[0]}
    return {"commitment": R.record_of(ops).encode().hex(), "evidence": D.canonical(ev).encode().hex(),
            "params": pubs[0] if params is None else params}


SCHEDULED = [["open", "s", ["L0", "L1"], None, {"start_ms": 10, "windows_ms": [5, 5]}], ["finish", "r0000"]]


def test_an_admitted_session_on_time_is_released(server, contract):
    """The honest transcript, its session admitted at L0 and ready before L0's release time (10 + 5 ms): accepted, with
    what leaves and when public, and the record's digest; at 16 ms it missed its time; asked for L1, it is refused."""
    ref = port(reference(contract))
    kind, line = D.replay(ref, _case(SCHEDULED, "r0000", "L0", 15), server)
    out = json.loads(line)
    assert kind == "accept" and out["public"] == {"part": "L0", "release_ms": 15, "session": "r0000", "stream": "s"}, out
    assert out["profile"]["contract"] == FORMAT and out["commitment"] == hashlib.sha512(R.record_of(SCHEDULED).encode()).hexdigest()
    assert D.replay(ref, _case(SCHEDULED, "r0000", "L0", 16), server)[0] == "reject"
    assert D.replay(ref, _case(SCHEDULED, "r0000", "L1", 0), server)[0] == "reject"


def test_a_transcript_the_contract_cuts_is_not_released(server, contract):
    """Every session `firewall_agree.py` expects the contract to refuse, admitted on time: rejected, by both."""
    _, others, _, _ = pool()
    ref = port(reference(contract))
    for t in others[:5]:
        assert D.replay(ref, _case(SCHEDULED, "r0000", "L0", 0, t), server)[0] == "reject"


def test_a_record_its_slot_leaves_unreadable_releases_nothing(server, contract):
    """A record whose last line has no newline reads, but the slot's line joins it, so the admission reads a line that
    isn't JSON: Rust's `outer_session` fails there, and the Lean and Python reject."""
    case = _case(SCHEDULED, "r0000", "L0", 0)
    case |= {"commitment": bytes.fromhex(case["commitment"])[:-1].hex()}
    kind, line = D.replay(port(reference(contract)), case, server)
    assert kind == "reject" and json.loads(line)["commitment"] is not None


@pytest.mark.slow
def test_a_reference_without_the_slots_rule_is_caught(server, contract, monkeypatch):
    """Rule 2's orphan: a reference whose slot doesn't stop the session of a holder that died admits it again, on the
    orphan itself and among drawn cases (at this seed the first orphan the sample reaches is past its 600th case)."""
    orphan = _case([["open", "s", ["L0", "L1"], None, None], ["finish", "r0000"], ["slot", "r0000", "L0"]], "r0000", "L0", 0)

    @contextmanager
    def slot(self, session, part, waiting=None):
        yield

    monkeypatch.setattr(FW.Firewall, "slot", slot)
    for run in (lambda: D.replay(port(reference(contract)), orphan, server),
                lambda: D.run(port(reference(contract)), examples=1500, seed=SEED, server=server)):
        with pytest.raises(D.Mismatch) as e:
            run()
        assert (e.value.lean[0], e.value.python[0]) == ("reject", "accept")


@pytest.mark.slow
def test_a_reference_without_the_release_time_is_caught(server, contract, monkeypatch):
    """Rule 3: a reference that releases a scheduled statement as if unscheduled differs in what it makes public."""
    admit = FW.Firewall.admit
    monkeypatch.setattr(FW.Firewall, "admit", lambda self, s, p, timed=False: admit(self, s, p, timed=timed) and None)
    with pytest.raises(D.Mismatch) as e:
        D.run(port(reference(contract)), examples=600, seed=SEED, server=server)
    assert e.value.lean[0] == e.value.python[0] == "accept" and '"release_ms":null' in e.value.python[1]


@pytest.mark.slow
def test_the_release_agrees_on_drawn_cases(server, contract):
    tally: Counter = Counter()
    stats = D.run(port(reference(contract, tally)), examples=600, seed=SEED, server=server)
    print(f"core-firewall serve, seed {SEED}: {stats}, {dict(sorted(tally.items()))}")
    assert stats.counts["accept"] >= 30, stats
    for k in ("case", "params", "commitment", "evidence", "FIREWALL-REFUSED", "the send check", "missed", "the record"):
        assert tally[f"reject:{k}"] >= 3, (k, tally)
