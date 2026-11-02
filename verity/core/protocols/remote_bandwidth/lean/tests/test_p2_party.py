"""The P2 session's verifier as a party of its own (`p2_party.py`), the Lean of record (`remote-audit-verifier`,
`RemoteAudit.P2Session`) with the Python's books beside it, over a channel to a prover serving the 9-block P2 store of
`tests/test_p2.py`'s fixture: an honest prover accepted with its credited reads, under every unit or the registration's
plan, one whose receipts skip a unit's reads, one whose openings are out of form and one that reads every unit when the
plan is fewer rejected, and a request for another node, or off the auditor's session parameters, refused before any
coin, the party `main` starts holding its sessions to `SESSION` unless its flags name others; and the books' veto: a Lean that
accepts a window the books reject, books that accept one the Lean rejects, a Lean whose answer isn't the books', one
that fails mid-session, and a Lean that holds another store, each rejected or refused."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from verity.core.protocols.pous import challenger_party as CP
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import run as R
from verity.core.protocols.pous import setup
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.remote_bandwidth import coins, p2, p2_lean, p2_party, receipts

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
SOURCE = b"the verifier's secret"
NODE = b"p2-node-under-test"
STEP_NS = 400
TOY = {"windows": 2, "passes": 2, "samples": 4, "window_ns": 2 * STEP_NS, "deadline_ns": STEP_NS}
"""The auditor's session parameters these sessions are held to (`p2_party.session`'s `auditor`)."""


class Clock:
    """The verifier's clock, which the prover's steps and the verifier's waits advance."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now

    def sleep_until(self, _clock, t: int) -> None:
        self.now = max(self.now, t)


class Entropy:
    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]


@pytest.fixture(scope="module")
def store() -> p2.Store:
    doc = json.loads((FIXTURES / "p2-store.json").read_text())
    data = (FIXTURES / "p2-c-star.bin").read_bytes()
    blocks = [data[p2.BLOCK_BYTES * j:p2.BLOCK_BYTES * (j + 1)] for j in range(len(data) // p2.BLOCK_BYTES)]
    m = doc["params"]
    params = P2Params(m["w"], m["c0"], m["blocks_per_segment"], m["segments"], m["expander"])
    commitment, salt, n = bytes.fromhex(doc["commitment"]), bytes.fromhex(doc["salt"]), params.blocks_per_segment
    vk = setup.verifier_key_from_encoding(params, None, commitment, salt,
                                          [blocks[n * s:n * (s + 1)] for s in range(params.segments)])
    return p2.Store(p2.register(SOURCE, NODE, commitment, vk), blocks)


@pytest.fixture(scope="module")
def exe() -> Path:
    return p2_lean.build()


class Prover:
    """The honest prover over `store`, each pass the units of the plan the verifier holds, each step `STEP_NS` on the
    verifier's clock; `skip` names a unit whose reads it skips, its receipt from zeros, `garble` sends openings whose
    leaf salts are a byte short, and `whole` reads every unit whatever the plan."""

    def __init__(self, store: p2.Store, clock: Clock, *, skip: int | None = None, garble: bool = False,
                 whole: bool = False) -> None:
        self.store, self.clock, self.skip, self.garble, self.whole = store, clock, skip, garble, whole
        self.passes: dict[tuple[int, int], object] = {}
        self.held = self.plan = None
        self.ended: tuple | None = None

    def hold(self, registration) -> None:
        if p2_party.registration_of(registration) != self.store.registration:
            raise ValueError("not this store's registration")
        self.plan = p2_party.plan_of({} if self.whole else registration, self.store.registration)
        self.held = p2.Prover(self.store, entropy=Entropy(b"p2-party-test/leaf-salts"))

    def run(self, challenge, t, i, previous) -> bytes:
        self.held.received.setdefault(t, challenge)
        seed = receipts.seed(coins.coin(challenge), i, previous)
        sk = p2.Sketcher(seed, self.store.registration.rows)
        ys = [sk(g, bytes(p2.UNIT_BYTES) if g == self.skip else self.store.unit(g)) for g in self.plan]
        ps = self.held.commit(t, i, seed, self.plan, ys)
        self.passes[t, i] = ps
        self.clock.now += STEP_NS
        return ps.root

    def open(self, t, draws):
        out = [self.held.open(self.passes[t, i], p) for i, p in draws]
        return [dataclasses.replace(o, leaf_salt=o.leaf_salt[:-1]) for o in out] if self.garble else out

    def end(self, commitment, openings) -> None:
        self.ended = (commitment, tuple(openings))
        self.held.end(commitment, openings)


class Lying:
    """The Lean verifier, but for its answers to `op`, which `lie(request, answer)` replaces."""

    def __init__(self, lean: p2_lean.Lean, op: str, lie) -> None:
        self.lean, self.op, self.lie, self.last = lean, op, lie, None

    def send(self, request: dict, body: bytes = b"") -> None:
        self.last = request
        self.lean.send(request, body)

    def read(self):
        got = self.lean.read()
        return self.lie(self.last, got) if self.last["op"] == self.op else got

    def call(self, request: dict, body: bytes = b""):
        self.send(request, body)
        return self.read()


def run(exe: Path, store: p2.Store, prover_of, request: dict, plan: receipts.Plan | None = None,
        lean_of=lambda lean: lean) -> tuple[dict, dict]:
    """One session over a socket pair: the verifier's side in a thread, its Lean (`lean_of` of it) holding the
    registration file with `plan`, the prover's here; both answers."""
    clock = Clock()
    a, b = socket.socketpair()
    data = json.dumps(p2_party.registration_json(store.registration, plan)).encode()
    got: dict = {}

    def verifier() -> None:
        with p2_lean.Lean(exe) as lean, b, b.makefile("rb") as r, b.makefile("wb") as w:
            reg, held = p2_party.hold(lean, data)
            got["answer"] = p2_party.session(r, w, reg, coins=b"\x07" * 64, lean=lean_of(lean), plan=held,
                                             clock=clock, sleep_until=clock.sleep_until, host="auditor-under-test",
                                             auditor=TOY)

    th = threading.Thread(target=verifier)
    th.start()
    prover = prover_of(clock)
    with a, a.makefile("rb") as r, a.makefile("wb") as w:
        out = p2_party.prove(r, w, prover, request)
    th.join()
    return out, got["answer"]


def request(store: p2.Store, **kw) -> dict:
    return p2_party.request_of(store.registration.node, **{**TOY, **kw})


def test_a_plan_of_fewer_units_is_what_each_pass_reads_and_is_credited(exe, store) -> None:
    units = store.registration.blocks
    plan = receipts.Plan(((1, units - 1),), units)
    proved, answer = run(exe, store, lambda c: Prover(store, c), request(store), plan)
    assert proved == answer and answer["verdict"] == "accept", answer
    assert answer["units"] == units and answer["plan_units"] == units - 1
    assert answer["receipts"] == 2 * 2 * (units - 1)
    _, answer = run(exe, store, lambda c: Prover(store, c, whole=True), request(store), plan)
    assert answer["verdict"] == "reject" and "q_bytes" not in answer, answer


def test_an_honest_prover_is_accepted_with_its_credited_reads(exe, store) -> None:
    proved, answer = run(exe, store, lambda c: Prover(store, c), request(store))
    assert proved == answer
    assert answer["verdict"] == "accept" and "disagreement" not in answer, answer
    assert answer["host"] == "auditor-under-test"
    units = store.registration.blocks
    assert answer["receipts"] == 2 * 2 * units and answer["plan_units"] == units
    assert [w["passes"] for w in answer["windows"]] == [2, 2]
    assert all(w["verdict"]["accepted"] for w in answer["windows"])
    assert answer["q_bytes"] == (answer["receipts"] - answer["wrong_at_most"]) * p2.UNIT_BYTES
    assert answer["span_ns"] > 0 and answer["credited_bytes_per_s"] is not None


def test_a_prover_that_skips_a_units_reads_is_rejected(exe, store) -> None:
    _, answer = run(exe, store, lambda c: Prover(store, c, skip=1), request(store))
    assert answer["verdict"] == "reject" and answer["failure"] == "receipt", answer
    assert answer["window"] == 0 and answer["at"][1] % store.registration.blocks == 1
    assert "q_bytes" not in answer


def test_openings_out_of_form_are_rejected(exe, store) -> None:
    _, answer = run(exe, store, lambda c: Prover(store, c, garble=True), request(store))
    assert answer["verdict"] == "reject" and answer["failure"] == "malformed", answer


def test_a_request_for_another_node_is_refused_before_any_coin(exe, store) -> None:
    bad = request(store) | {"node": b"another-node".hex()}
    _, answer = run(exe, store, lambda c: Prover(store, c), bad)
    assert answer["verdict"] == "refused" and "node" in answer["reason"]


@pytest.mark.parametrize("off", [{"deadline_ns": 2 * STEP_NS}, {"samples": 3}])
def test_a_request_off_the_auditors_session_is_refused_before_any_coin(exe, store, off) -> None:
    """A request for a longer allowance or fewer openings than the auditor's, which the Lean takes (it reads the
    request alone), is refused by the books' veto before the registration or any coin is sent, naming the value."""
    provers: list[Prover] = []

    def prover_of(clock):
        provers.append(Prover(store, clock))
        return provers[-1]

    (key, value), = off.items()
    proved, answer = run(exe, store, prover_of, request(store, **off))
    assert proved == answer and answer["verdict"] == "refused", answer
    assert f"the request's {key} is not the auditor's ({value}, not {TOY[key]})" in answer["reason"]
    assert provers[0].held is None and provers[0].ended is None


def test_the_party_holds_its_sessions_to_the_served_parameters_unless_its_flags_name_others(exe, store,
                                                                                            tmp_path) -> None:
    """`main` as the auditor's gate starts it, with no session flags: the toy's request is refused naming each value
    off `SESSION`, and a request of `SESSION`'s is held (the registration sent first); with flags naming the toy's
    values, the toy's request is held."""
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(p2_party.registration_json(store.registration)))

    def first(req: dict, *flags: str) -> dict:
        p = subprocess.run([sys.executable, "-m", p2_party.MODULE, "--registration", str(path), "--stdio",
                            "--lean-exe", str(exe), *flags], input=json.dumps({"request": req}).encode() + b"\n",
                           capture_output=True, check=True)
        return json.loads(p.stdout.splitlines()[0])

    refused = first(request(store))["answer"]
    assert refused["verdict"] == "refused", refused
    for k in ("deadline_ns", "passes", "samples", "window_ns", "windows"):
        assert f"the request's {k} is not the auditor's ({TOY[k]}, not {p2_party.SESSION[k]})" in refused["reason"]
    assert "hold" in first(p2_party.request_of(store.registration.node, **p2_party.SESSION))
    assert "hold" in first(request(store), *[x for k, v in TOY.items() for x in ("--" + k.replace("_", "-"), str(v))])


def test_a_lean_that_accepts_what_the_books_reject_is_vetoed(exe, store) -> None:
    def accepting(lean):
        return Lying(lean, "p2_openings", lambda req, _got: {"accepted": req["t"]})

    _, answer = run(exe, store, lambda c: Prover(store, c, skip=1), request(store), lean_of=accepting)
    assert answer["verdict"] == "reject" and answer["failure"] == p2_party.DISAGREEMENT, answer
    d = answer["disagreement"]
    assert d["event"] == "openings" and d["lean"] == {"accepted": 0} and d["python"]["failure"] == "receipt"
    assert "q_bytes" not in answer


def test_books_that_accept_what_the_lean_rejects_are_vetoed(exe, store, monkeypatch) -> None:
    monkeypatch.setattr(p2, "check", lambda *_a, **_k: None)
    _, answer = run(exe, store, lambda c: Prover(store, c, skip=1), request(store))
    assert answer["verdict"] == "reject" and answer["failure"] == p2_party.DISAGREEMENT, answer
    d = answer["disagreement"]
    assert d["event"] == "openings" and d["lean"]["failure"] == "receipt" and d["python"] == {"accepted": 0}


def test_a_lean_answer_that_isnt_the_books_is_vetoed(exe, store) -> None:
    def inflated(lean):
        return Lying(lean, "p2_answer", lambda _req, got: got | {"q_bytes": 2 * got["q_bytes"]})

    _, answer = run(exe, store, lambda c: Prover(store, c), request(store), lean_of=inflated)
    assert answer["verdict"] == "reject" and answer["failure"] == p2_party.DISAGREEMENT, answer
    assert answer["disagreement"]["event"] == "answer" and "q_bytes" not in answer


def test_a_lean_that_fails_mid_session_rejects_it(exe, store) -> None:
    def failing(_req, _got):
        raise p2_lean.LeanError("the process ended (-9)")

    _, answer = run(exe, store, lambda c: Prover(store, c), request(store),
                    lean_of=lambda lean: Lying(lean, "p2_root", failing))
    assert answer["verdict"] == "reject" and answer["failure"] == p2_party.DISAGREEMENT, answer
    assert answer["disagreement"]["event"] == "root" and answer["window"] == 0


def test_a_lean_that_holds_another_store_is_refused(exe, store) -> None:
    data = json.dumps(p2_party.registration_json(store.registration)).encode()
    with p2_lean.Lean(exe) as lean:
        other = Lying(lean, "p2_hold", lambda _req, got: got | {"units": got["units"] + 1})
        with pytest.raises(ValueError, match=p2_party.DISAGREEMENT):
            p2_party.hold(other, data)
    with p2_lean.Lean(exe) as lean:
        assert p2_party.hold(lean, data) == (store.registration, receipts.Plan.whole(store.registration.blocks))
        with pytest.raises(ValueError, match="refuses"):
            p2_party.hold(lean, data)


def _challenger(device: Path) -> tuple[mc.SaltLog, mc.SessionLog]:
    """The memory challenger's salt log and session log on its host, as `challenger_party` makes them in its device."""
    device.mkdir(parents=True, exist_ok=True)
    mc.SaltLog.create(device / CP.SALTS)
    mc.SessionLog.create(device / "sessions")
    return mc.SaltLog(device / CP.SALTS), mc.SessionLog(device / "sessions")


def _ran(device: Path, salts: mc.SaltLog, sessions: mc.SessionLog, params: P2Params, setup: R.Setup,
         audit: bytes) -> tuple[bytes, dict]:
    """A PoUS session of `audit` at P2's `params` that the challenger on `device` ran over `setup`, as
    `challenger_party.main` runs one: designated against the salt it logged for the audit (`designate`), its id claimed
    and its record kept (`keep`); its id and request."""
    request = CP.request_of(NODE.decode(), R.Point(params, 1, 1), setup.root, audit=audit, published=setup)
    sid = CP.designate(request, salts, sessions)
    sessions.claim(sid)
    CP.keep(device, sid, request)
    return sid, request


def test_the_registration_is_the_challengers_record_of_the_windows_pous_session(store, tmp_path) -> None:
    """`--pous-record DIR --pous-audit AUDIT`: the registration of the setup the memory challenger kept for the PoUS
    session of the window's binding AUDIT, its id derived from the salt the challenger logged for AUDIT
    (`challenger_party.session_id`), its whole store the plan; refused for an audit out of form (a session id among
    them), a host whose challenger drew no salt for it or kept no session of it, and a record under the derived id that
    names another session, has no setup, carries a setup on a request naming no audit (so its commitment and salt may
    be the prover's own), or is of another audit."""
    reg = store.registration
    audit = hashlib.sha512(b"a window's binding").digest()
    with pytest.raises(ValueError, match="salt log on this host is unreadable"):
        p2_party.pous_record(tmp_path, audit.hex())
    salts, sessions = _challenger(tmp_path)
    with pytest.raises(ValueError, match="drew no salt for audit"):
        p2_party.pous_record(tmp_path, audit.hex())
    salts.record(audit, reg.commitment, reg.salt)
    with pytest.raises(ValueError, match="kept no session for audit"):
        p2_party.pous_record(tmp_path, audit.hex())
    sid, request = _ran(tmp_path, salts, sessions, reg.params,
                        R.Setup(reg.params.blocks, reg.commitment, reg.root, reg.salt), audit)
    doc = json.loads(p2_party.pous_record(tmp_path, audit.hex()))
    assert p2_party.registration_of(doc) == reg and p2_party.plan_of(doc, reg) == receipts.Plan.whole(reg.blocks)
    assert "plan" not in doc
    for bad in (sid.hex(), audit.hex().upper(), "../" + audit.hex()[3:]):
        with pytest.raises(ValueError, match="128 lowercase hex"):
            p2_party.pous_record(tmp_path, bad)
    path = tmp_path / CP.SETUPS / f"{sid.hex()}.json"
    bare = CP.request_of(NODE.decode(), R.Point(reg.params, 1, 1), reg.root)
    for kept, why in [({"request": request, "session_id": bytes(32).hex()}, "names another session"),
                      ({"request": bare, "session_id": sid.hex()}, "has no setup"),
                      ({"request": {k: v for k, v in request.items() if k != "audit"}, "session_id": sid.hex()},
                       "ride only on a request naming its audit"),
                      ({"request": request | {"audit": hashlib.sha512(b"another window").hexdigest()},
                        "session_id": sid.hex()}, "not of its audit's session: its id is not session_id")]:
        path.write_text(json.dumps(kept))
        with pytest.raises(ValueError, match=why):
            p2_party.pous_record(tmp_path, audit.hex())


def test_an_earlier_windows_session_on_the_host_is_never_this_windows_store(store, tmp_path) -> None:
    """The challenger on this host ran an earlier window's PoUS session over another store: its salt drawn after that
    store's commitment (`give_salt`), the session `designate`d and kept. This window's bandwidth session never holds
    that store: the earlier session's id is no audit, and this window's audit is refused until the challenger drew a
    salt for it and kept its session, and refused when the record under its id is the earlier window's; once this
    window's session is kept, each window's audit gives its own session's store."""
    reg = store.registration
    earlier, window = (hashlib.sha512(b"window %d's binding" % k).digest() for k in (1, 2))
    salts, sessions = _challenger(tmp_path)
    commitment, root = (hashlib.sha512(b"the earlier store's " + x).digest() for x in (b"commitment", b"root"))
    CP.give_salt(salts, {"audit": earlier.hex(), "commitment": commitment.hex()}, draw=Entropy(b"the earlier source"))
    _, salt = salts.get(earlier)
    earlier_sid, earlier_request = _ran(tmp_path, salts, sessions, reg.params,
                                        R.Setup(reg.params.blocks, commitment, root, salt), earlier)
    with pytest.raises(ValueError, match="128 lowercase hex"):
        p2_party.pous_record(tmp_path, earlier_sid.hex())
    with pytest.raises(ValueError, match=f"drew no salt for audit {window.hex()}"):
        p2_party.pous_record(tmp_path, window.hex())
    salts.record(window, reg.commitment, reg.salt)
    with pytest.raises(ValueError, match=f"kept no session for audit {window.hex()}"):
        p2_party.pous_record(tmp_path, window.hex())
    sid = CP.session_id(window, reg.commitment, reg.salt)
    planted = CP.keep(tmp_path, sid, earlier_request)
    with pytest.raises(ValueError, match="not of its audit's session"):
        p2_party.pous_record(tmp_path, window.hex())
    planted.unlink()
    assert _ran(tmp_path, salts, sessions, reg.params,
                R.Setup(reg.params.blocks, reg.commitment, reg.root, reg.salt), window)[0] == sid
    assert p2_party.registration_of(json.loads(p2_party.pous_record(tmp_path, window.hex()))) == reg
    held = p2_party.registration_of(json.loads(p2_party.pous_record(tmp_path, earlier.hex())))
    assert (held.commitment, held.salt, held.root) == (commitment, salt, root) != (reg.commitment, reg.salt, reg.root)
