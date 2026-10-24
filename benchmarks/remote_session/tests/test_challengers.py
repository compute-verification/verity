"""`challengers.Gate` against the auditor host's own gate (`benchmarks/live_challenger/auditor.py`), in this process: its
requests go to `auditor.gate` as the lane key's forced command would pass them, and each opened session is served by
`auditor.run` through `serve.py` (the Lean's coins stubbed, `stub_serve.py`) on a thread, with the gate's root steps
(chown, systemctl) and the clock injected. An honest session whose coins the gate's session released is accepted and
held to the ledger; a challenger key other than the pinned one, a reserve outside the audit's period, a second session
for the same window and a challenger whose commitment isn't the open's are each refused before a coin leaves. The
verifier's own paced sessions, through the gate or serve.py, reserve only the coins they use, so an honest one ends
complete and a dropper doesn't; and `auditor_line` credits a gate session to the line only when its period committed
its verifier, at the commit its host run recorded, on the auditor's first-parent list of main, it and its period closed
complete, the Lean accepted it, the host's pins log its open sealed records its pin, a commit on that list too
(RT4-INT-40, CMP-74), and the verifier bound the store to the model, reporting filler held beside weights held. The
peer a gate session's ledger entries must carry is the one the run's record names."""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import shlex
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LIVE = ROOT / "benchmarks" / "live_challenger"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(LIVE))
sys.path.insert(0, str(ROOT / "verity" / "core" / "protocols" / "remote_bandwidth" / "tests"))

import auditor as A  # noqa: E402
import auditor_line as al  # noqa: E402
import challengers  # noqa: E402
import verifier  # noqa: E402
import wire  # noqa: E402
from test_remote_bandwidth import SCHEDULE, drive, finish, params, store  # noqa: E402,F401
from test_session import run as run_sessions  # noqa: E402
from verity.core.protocols.remote_bandwidth import coins  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import NO_MAIN, NO_PINS, CoinError, pin_line, pins_seal  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402

P = lms.Params(64, 4, 4)
T0 = datetime(2026, 10, 21, 1, 0, 0, tzinfo=timezone.utc)
PIN = "c" * 40
LANE = "memory-accounting"
SECRET = bytes(range(32))
CIRCUIT = hashlib.sha512(b"verity/remote-bandwidth/v0").hexdigest()
#: the host's pins log: an earlier pin, then the tree it runs (`PIN`), each a commit of main's
PINS = [pin_line("d" * 40, "e" * 40, "2026-10-20T08:00:00Z"), pin_line(PIN, "f" * 40, "2026-10-21T00:30:00Z")]
#: the auditor's first-parent list of main, newest first: both pins on it
MAIN = ["f" * 40, PIN, "e" * 40, "d" * 40]
#: the model the store is bound to (`remote_bandwidth.store`'s pin)
MODEL = {"name": "synthetic-bf16", "checkpoint_sha512": "ab" * 64}
#: the commit the verifier ran at (`--commit`): main's head
VERIFIER = MAIN[0]


class Host:
    """The auditor host in `root`: its gate under the lane's key, and the sessions its opens started, each `auditor.run`
    on a thread until its client closes it."""

    def __init__(self, root: Path) -> None:
        (root / "var" / "sessions").mkdir(parents=True)
        (root / "peers").write_text("vy-challenger-1 research@10.0.0.1\nvy-nebius-1 research@10.0.0.2\n"
                                    "vy-auditor-1 challenger@10.0.0.3\n")
        (root / "pins.jsonl").write_text("".join(line + "\n" for line in PINS))
        self.host = A.Host(root=root / "var", peers=root / "peers", pins=root / "pins.jsonl")
        self.signer = A.key_signer(self.host, P, workers=1)
        public = (self.host.key / "public").read_bytes()
        self.pin = hashlib.sha512(public).hexdigest()
        (self.host.root / "pubkey.json").write_text(json.dumps(
            {"W": P.W, "H": P.H, "public": public.hex(), "public_sha512": self.pin}))
        self.tokens = (f"{i:04x}" for i in itertools.count())
        self.runs: dict[str, threading.Thread] = {}

    def call(self, words: list[str]) -> tuple[int, bytes]:
        host, signer = self.host, self.signer
        return A.gate(["--ssh", shlex.join(words), "--key", LANE], host, now=T0, token=next(self.tokens),
                      own=lambda p: None, start=self.start, active=lambda s: False, secret=SECRET,
                      reserve=lambda s: A.reserve(host, s, signer=signer, pinned=PIN, now=T0),
                      admit=lambda s: A.admit(host, s, signer=signer, pinned=PIN, now=T0, active=lambda s: False),
                      period=lambda args: A.period(host, args, signer=signer, pinned=PIN, now=T0,
                                                   active=lambda s: False))

    def start(self, session: str) -> None:
        """The session's unit: serve.py on a port of its own (no tunnel), returning once it listens."""
        body = A.sessions_of(A.read_ledger(self.host.ledger))[session]["open"]["body"]
        argv = A.serve_argv(self.host, session, body)
        i = argv.index("--peer")
        del argv[i:i + 2]
        argv[argv.index("--port") + 1] = "0"
        argv = [sys.executable, str(LIVE / "tests" / "stub_serve.py"), *argv[2:], "--params", "4,4",
                "--keygen-workers", "1"]
        listening = threading.Event()

        def wait(proc, seconds):
            listening.set()
            return proc.wait()

        self.runs[session] = threading.Thread(target=A.run, args=(self.host, session), kwargs=dict(
            signer=self.signer, pinned=PIN, now=lambda: T0, argv=argv, wait=wait))
        self.runs[session].start()
        listening.wait()

    def address(self, answer: dict) -> str:
        return (self.host.sessions / answer["session"] / "out" / "stdout.log").read_text().split()[1]

    def pause(self) -> None:
        for t in self.runs.values():
            t.join()

    def gate(self, out: Path, *, pin: str | None = None, period: str = "p-1", main: list[str] | None = MAIN,
             call=None, peer: str = "vy-nebius-1") -> challengers.Gate:
        return challengers.Gate(out, call or self.call, peer, pin or self.pin, period, main, address=self.address,
                                pause=self.pause, polls=3)

    def period(self, name: str = "p-1", circuit: str = CIRCUIT) -> None:
        rc, out = self.call(["period-open", "--name", name, "--cap", "4", "--circuit", circuit])
        assert rc == 0, out

    def ledger(self) -> list[dict]:
        return A.entries(A.read_ledger(self.host.ledger))

    @staticmethod
    def seal(monkeypatch, pins: list[str] | None) -> None:
        """Each open from here on seals the pins log `pins` (`challenger.pins_seal`) in place of the host's own log,
        which `auditor.admit` seals; `pins` None: the open seals no log, as on every gate before RT4-INT-40's."""
        append = A.append

        def resealed(body: dict) -> dict:
            body = {k: v for k, v in body.items() if k != "pins"}
            return body if pins is None else {**body, "pins": pins_seal(pins)}

        def sealed(host, kind, session, body_of, I, sign):
            if kind == "open":
                return append(host, kind, session, lambda lines: resealed(body_of(lines)), I, sign)
            return append(host, kind, session, body_of, I, sign)

        monkeypatch.setattr(A, "append", sealed)


@pytest.fixture
def host(tmp_path, monkeypatch) -> Host:
    monkeypatch.setenv("COINS_LOG", str(tmp_path / "coins.log"))
    return Host(tmp_path / "host")


def session(store, gate: challengers.Gate):
    """An honest session over `store` whose coins come from `gate`'s session for window 0: the session and verdicts."""
    p = params()
    client = gate.open(store.registration.digest, 0, p.windows, 8)
    c = coins.Coins(client, store.registration.digest, 0, p.windows, clock=itertools.count().__next__)
    s, prover, passes, clock = drive(store, p, SCHEDULE, coins_=c)
    return s, finish(s, prover, passes, clock)


def test_a_session_whose_coins_the_gates_session_released_is_held_to_the_ledger_and_accepted(store, host, tmp_path):
    host.period()
    gate = host.gate(tmp_path / "out")
    s, verdicts = session(store, gate)
    assert all(v.accepted for v in verdicts) and s.outcome == "accepted"
    doc = gate.finish(s)
    assert doc["reason"] is None, doc["reason"]
    sid = doc["session"]
    assert doc["closed"] is None and doc["end"]["closed"] and doc["end"]["rc"] == 0 and "secret" not in doc["open"]
    assert doc["open"]["root"] == coins.window(store.registration.digest, 0)["root"]
    kinds = [(e["kind"], e["session"]) for e in host.ledger()]
    assert kinds == [("period-open", None), ("reserve", sid), ("open", sid), ("end", sid)]
    reserve = host.ledger()[1]["body"]
    assert (reserve["by"], reserve["peer"], reserve["scheme"], reserve["epoch"]) == (LANE, "vy-nebius-1", coins.SCHEME, 0)
    assert reserve["families"] == [list(f) for f in coins.families(params().windows, 8)]
    assert (tmp_path / "out" / f"{sid}.ledger.jsonl").read_bytes() == host.host.ledger.read_bytes()
    assert (tmp_path / "out" / f"{sid}.pins.jsonl").read_bytes() == host.host.pins.read_bytes()
    assert not (tmp_path / "out" / sid / "secret").exists()
    key = gate.key()
    heads = [tmp_path / "out" / f"{sid}.ledger.jsonl"]
    assert s.trusted_reason(tmp_path / "out" / sid / "out", heads, key, PINS, MAIN, challengers.lms_verify, "p-2") == (
        "the challenger's session for this session's window is not trusted: the auditor's ledger holds the session's "
        "window in period p-1, not the audit's period p-2")


@pytest.mark.parametrize("case", ["no-main", "stale-main", "no-pins"])
def test_a_gate_session_is_refused_as_coins_unless_the_hosts_pins_log_and_the_verifiers_main_hold_its_pin(
        store, host, tmp_path, case):
    """`Gate.finish` reads the window's session through `coins.trusted`, so through `challenger.pinned_session`, with
    the host's pins log as the gate serves it (`pins`, fetched beside the ledger) and the verifier's own first-parent
    list of main. An honest session, its coins the gate's session's and its open sealing the host's log, is refused when the
    verifier holds no copy of main, when its copy predates the host's pin, and when the gate serves no pins log."""
    host.period()
    main = {"no-main": None, "stale-main": MAIN[2:], "no-pins": MAIN}[case]

    def call(words):
        return (3, b'{"refused": "usage"}\n') if words[0] == "pins" and case == "no-pins" else host.call(words)

    gate = host.gate(tmp_path / "out", main=main, call=call)
    s, verdicts = session(store, gate)
    assert all(v.accepted for v in verdicts)
    doc = gate.finish(s)
    sid = doc["session"]
    assert doc["closed"] is None and ("end" in doc) is (case != "no-pins")
    assert doc["reason"] == {
        "no-main": f"the challenger's session for this session's window is not trusted: {NO_MAIN}",
        "stale-main": (f"the challenger's session for this session's window is not trusted: session {sid} ran commit "
                       f"{PIN}, which the host's pins log records but which is not on main's first-parent line in the "
                       "verifier's copy (`git rev-list --first-parent origin/main`): a branch commit merged into main "
                       "is in main's history but no tree main's check ran on; repin the challenger host to a commit "
                       "on main's first-parent line (vy-challenger-pin) and rerun the window there"),
        "no-pins": ("the auditor's session is not readable: the challenger gate answered pins with exit 3: "
                    '{"refused": "usage"}')}[case]


def test_a_gate_verifier_needs_the_auditors_first_parent_list_of_main(tmp_path):
    """`--challenger gate` without `--challenger-main FILE`, or with a FILE it can't read, is refused before the run
    starts, so no gate run takes coins it would refuse at every session's end."""
    flags = dict(ssh="-i /nonexistent/key research@192.0.2.1", peer="vy-nebius-1", pin="ab" * 64, period="p-1")
    with pytest.raises(ValueError, match=re.escape("--challenger-main FILE (the auditor's own `git rev-list "
                                                   "--first-parent origin/main`)")):
        challengers.of("gate", tmp_path / "out", None, None, **flags)
    with pytest.raises(CoinError, match=re.escape(f"{NO_MAIN} (none at {tmp_path / 'absent'})")):
        challengers.of("gate", tmp_path / "out", None, None, **flags, main=tmp_path / "absent")
    (tmp_path / "main.txt").write_text("".join(c + "\n" for c in MAIN))
    assert challengers.of("gate", tmp_path / "out", None, None, **flags, main=tmp_path / "main.txt").main == MAIN


def test_a_challenger_key_other_than_the_pinned_one_is_refused_before_a_reserve(store, host, tmp_path):
    host.period()
    with pytest.raises(CoinError, match="not the pinned"):
        host.gate(tmp_path / "out", pin="0" * 128).open(store.registration.digest, 0, params().windows, 8)
    assert [e["kind"] for e in host.ledger()] == ["period-open"]


def test_a_reserve_outside_the_audits_period_is_refused_before_its_open(store, host, tmp_path):
    host.period("p-0")
    with pytest.raises(CoinError, match="in period p-0, not the audit's period p-1: it expires unopened"):
        host.gate(tmp_path / "out").open(store.registration.digest, 0, params().windows, 8)
    assert [e["kind"] for e in host.ledger()] == ["period-open", "reserve"]


def test_a_second_session_for_the_window_is_refused_at_its_open(store, host, tmp_path):
    host.period()
    gate = host.gate(tmp_path / "out")
    s, verdicts = session(store, gate)
    assert gate.finish(s)["reason"] is None
    with pytest.raises(CoinError, match="one session a root, ever"):
        host.gate(tmp_path / "again").open(store.registration.digest, 0, params().windows, 8)
    assert [e["kind"] for e in host.ledger()] == ["period-open", "reserve", "open", "end", "reserve"]


def test_a_challenger_whose_commitment_isnt_the_opens_is_refused_before_its_first_coin(store, host, tmp_path):
    def call(words):
        rc, out = host.call(words)
        if words[0] == "open" and rc == 0:
            out = json.dumps({**json.loads(out), "commitment": "ab" * 64}).encode()
        return rc, out

    host.period()
    with pytest.raises(CoinError, match="refused before its first coin"):
        host.gate(tmp_path / "out", call=call).open(store.registration.digest, 0, params().windows, 8)
    host.pause()
    end = host.ledger()[-1]["body"]
    assert host.ledger()[-1]["kind"] == "end" and end["released"] == {label: 0 for label in coins.LABELS}


def test_the_gates_ports_are_the_hosts():
    assert challengers.GATE_PORTS == A.PORTS


@pytest.mark.parametrize("kind", ["gate", "serve"])
def test_an_honest_paced_session_closes_complete_and_a_dropper_does_not(host, tmp_path, kind):
    """The verifier's paced sessions over one registration, an honest one and a 1/16 dropper, their coins from the
    gate's sessions (`challengers.Gate`) or from serve.py on this machine (`challengers.Serve`), its Lean coins stubbed.
    The plan appends nothing (its hello says so), so each reserves W challenges and W openings coins and no append
    family. The honest session's
    challenger released every coin it reserved: its end on the served log is complete. The prover sends each window's
    openings as its draws arrive (`defer_openings=False`), so the dropper, refused at its first window, left openings
    coins unreleased and is not, and the period's close lists the two so."""
    if kind == "gate":
        host.period()
        source = host.gate(tmp_path / "out")
    else:
        python = tmp_path / "python"
        python.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} "
                          f"{shlex.quote(str(LIVE / 'tests' / 'stub_serve.py'))} \"$@\" --keygen-workers 1\n")
        python.chmod(0o755)
        (tmp_path / "served").mkdir()
        source = challengers.Serve(tmp_path / "served", tmp_path / "key", "4,4", python=str(python))
    pace = verifier.pace_of("1:4:160:2", 1.0)
    (honest, dropper), _ = run_sessions("honest,skip:0.0625", paces=[pace, pace], challenger=source, verdict="python",
                                        defer_openings=False)
    w = honest["windows"]
    end, dropped = ((r["challenger"]["end"] if kind == "gate" else r["challenger"]) for r in (honest, dropper))
    assert honest["accepted"] and honest["challenger"]["reason"] is None
    assert end["released"] == {coins.CHALLENGE: w, coins.OPENINGS: w} and end["complete"] is True
    assert not dropper["accepted"] and dropper["failure"]["failure"] == "receipt"
    assert dropper["challenger"]["reason"] is None
    assert dropped["released"][coins.OPENINGS] < w and dropped["complete"] is False
    if kind == "gate":
        reserves = [e["body"]["families"] for e in host.ledger() if e["kind"] == "reserve"]
        assert reserves == [[[coins.CHALLENGE, w], [coins.OPENINGS, w]]] * 2
        rc, out = host.call(["period-close", "--name", "p-1"])
        assert rc == 0, out
        close = json.loads(out)["body"]
        assert [x["complete"] for x in close["windows"]] == [True, False] and close["complete"] is False


def gate_run(host: Host, out: Path, plan: str, defer_openings: bool,
             peer: str = "vy-nebius-1") -> tuple[list[dict], dict, bytes]:
    """A gate run of `plan`'s sessions in period p-1, which committed the descriptor of the verifier that ran them, its
    verifier at the challenger host as `peer`: the sessions, their registration and that descriptor."""
    desc = al.descriptor("ab" * 20)
    host.period(circuit=hashlib.sha512(desc).hexdigest())
    got = {}
    pace = verifier.pace_of("1:4:160:2", 1.0)
    sessions, _ = run_sessions(plan, paces=[pace] * len(plan.split(",")),
                               challenger=host.gate(out / "challenger", peer=peer), verdict="python",
                               defer_openings=defer_openings,
                               then=lambda v: got.update(reg=wire.registration_json(v.registration)))
    return sessions, got["reg"], desc


def analysed(out: Path, host: Host, sessions: list[dict], reg: dict, desc: bytes, *, verdict: str = "lean",
             pins: list[str] | None = PINS, main: list[str] | None = MAIN, bound: dict | None = MODEL,
             model: dict | None = MODEL, peer: str | None = None, lean: dict | None = None,
             commit: str = VERIFIER, line_by: al.lv.Lean | None = None) -> dict:
    """`auditor_line`'s analysis of the run, its verifier.json recording a binding of the store to `bound` (none if
    None), the Lean's `hold` answering `lean` (by default bound at the registration's root), the line held to `model`
    (`--model-pin`'s), `peer` its --peer (None: the record's), `commit` the commit its verifier ran at, and each line
    computed by the Lean verifier `line_by` (None: line.py alone)."""
    held = {"regions": [], "expected_root": reg["payload_root"]} if lean is None else lean
    binding = {} if bound is None else {"binding": {**bound, "payload_root": reg["payload_root"],
                                                    "expected_root": reg["payload_root"], "bound": True, "lean": held}}
    (out / "verifier.json").write_text(json.dumps(
        {"registration": reg, "sessions": [{**s, "verdict": verdict} for s in sessions]} | binding))
    return al.analyse(out, host_bytes=1 << 30, pin=host.pin, period="p-1", heads=[host.host.ledger], desc=desc,
                      commit=commit, pins=pins, main=main, model=model, peer=peer, lean=line_by)


@pytest.mark.parametrize("defer_openings", [True, False])
def test_the_line_credits_a_session_only_when_credited_complete_accepted_by_the_lean_pinned_and_bound(
        host, tmp_path, monkeypatch, defer_openings):
    """`auditor_line`'s line of record over a gate run of an honest session and a 1/16 dropper, in a period that
    committed the descriptor of the verifier that ran them, on a host whose opens seal its pins log, which records the
    host's tree as a commit of main's, its store bound to the model. When the prover defers each window's openings, the
    dropper's are all asked, so both windows and the period close complete: the honest session, once its verdict is
    the Lean's, is the line, its weights and filler held apart, and the dropper, which the Lean refused, earns none;
    the Lean verifier's line (`line_by` lean) is line.py's, the run's whole analysis the same. When it doesn't, the
    dropper's window and so the period close incomplete, and the line credits zero. A python verdict, a period not yet
    closed, a verifier the period didn't commit, no pins log (under both *credited*, whose
    `credited_session` reads the pin first, and *pinned*), a store bound to no model or to another, a binding the Lean
    didn't decide bound at the registration's root, and no model to bind it to each credit nothing, with the reason."""
    host.seal(monkeypatch, PINS)
    out = tmp_path / "run"
    sessions, reg, desc = gate_run(host, out, "honest,skip:0.0625", defer_openings)
    assert [s["accepted"] for s in sessions] == [True, False]

    def line(verdict: str = "python", d: bytes = desc, pins: list[str] | None = PINS) -> dict:
        return analysed(out, host, sessions, reg, d, verdict=verdict, pins=pins)

    def zero(doc: dict) -> dict:
        return {k: [why.split(":")[0] for why in v] for k, v in doc["credit"]["zero"].items()}

    doc = line("lean")
    assert zero(doc)[0] == ["not complete"]
    assert "period p-1 hasn't closed" in doc["sessions"][0]["credit"]["complete"]["reason"]
    rc, closed = host.call(["period-close", "--name", "p-1"])
    assert rc == 0, closed
    assert json.loads(closed)["body"]["complete"] is defer_openings
    doc = line("lean")
    honest, dropper = doc["sessions"]
    assert honest["of_record"] and dropper["of_record"] and honest["credit"]["credited"]["ok"]
    assert dropper["credit"]["accepted"]["reason"] == "the Lean refused it (receipt)"
    for s in (honest, dropper):
        assert s["credit"]["pinned"] == {"commit": PIN, "seal": pins_seal(PINS), "ok": True, "reason": None}
    assert doc["pins"] == pins_seal(PINS)
    t = honest["T"]
    assert doc["bound"]["ok"] and doc["bound"]["source"] == "--model-pin"
    assert all(s["credit"]["bound"]["ok"] for s in (honest, dropper))
    if defer_openings:
        assert honest["credit"]["ok"] and zero(doc) == {1: ["not accepted"]}
        mine = honest["shares_auditor_clock"][al.line.TIME_HELD]
        assert doc["line"][t]["credited_sessions"] == [0] and doc["line"][t][al.line.TIME_HELD] == {
            "k": 0, **{x: mine[x] for x in (*al.SHARES, "exact")}}
        assert all(float(al.line.number(mine["exact"][x])) == mine[x] for x in al.SHARES)
        assert doc["credit"]["summary"].startswith("the line credits session(s) 0; session(s) 1 earn none")
        shares, filler = (" / ".join(f"{honest['shares_auditor_clock'][b][x]:.4%}" for b in al.line.FORMS)
                          for x in ("time", "filler_time"))
        assert (f"| 0 | {t} | yes | yes | yes | yes | yes | credited: weights {shares}, filler {filler} |"
                in al.table(doc).splitlines())
        by = al.lv.Lean(al.lv.build(), lambda _: None)
        try:
            assert doc["line_by"] == "python" and analysed(out, host, sessions, reg, desc, line_by=by) == {
                **doc, "line_by": "lean"}
        finally:
            by.close()
    else:
        assert zero(doc) == {0: ["not complete"], 1: ["not complete", "not accepted"]}
        assert honest["credit"]["complete"]["reason"] == "period p-1 closed incomplete (1 of its 2 windows incomplete)"
        assert re.fullmatch(r"its window closed incomplete: the challenger released (\d+ of its \d+ \w+ coins, )?"
                            r"\d+ of its \d+ openings coins; period p-1 closed incomplete \(1 of its 2 windows "
                            r"incomplete\)", dropper["credit"]["complete"]["reason"])
        assert doc["line"][t][al.line.TIME_HELD] == {**dict.fromkeys(al.SHARES, 0.0),
                                                     "exact": dict.fromkeys(al.SHARES, "0")}
        assert doc["line"][t]["credited_sessions"] == []
        assert doc["credit"]["summary"].startswith("the line credits zero: no session is credited, complete, accepted, "
                                                   "pinned and bound; session 0: not complete: period p-1 closed "
                                                   "incomplete")
    assert doc["measured"][t]["auditor"][al.line.TIME_HELD]["k"] == 0
    unlogged = line("lean", pins=None)
    assert unlogged["sessions"][0]["credit"]["pinned"]["reason"] == NO_PINS and unlogged["pins"] is None
    assert unlogged["sessions"][0]["credit"]["credited"]["reason"] == NO_PINS
    assert zero(unlogged)[0][0] == "not credited" and zero(unlogged)[0][-1] == "not pinned"
    assert not unlogged["credit"]["credited_sessions"]
    assert zero(line("python"))[0][-1] == "not accepted"
    other = line("lean", al.descriptor("cd" * 20))
    assert not other["credit"]["credited_sessions"] and zero(other)[0][0] == "not credited"
    assert "the one period p-1 committed" in other["sessions"][0]["credit"]["credited"]["reason"]
    for bound, model, why in (
            (None, MODEL, "the verifier bound the store to no model (verifier.py --model ROLE --checkpoint DIR --pin "
                          "FILE), so its payload root is the prover's choice"),
            ({**MODEL, "name": "llama31-8b"}, MODEL, f"the verifier bound the store to llama31-8b ({'ab' * 8}...), not "
                                                     f"synthetic-bf16 ({'ab' * 8}...)"),
            (MODEL, None, "no model to bind the store to: the audit's period signs none and no --model-pin names one")):
        unbound = analysed(out, host, sessions, reg, desc, bound=bound, model=model)
        assert not unbound["credit"]["credited_sessions"] and unbound["bound"]["reason"] == why
        assert zero(unbound)[0][-1] == "not bound" and unbound["sessions"][0]["credit"]["bound"]["reason"] == why
    alone = "the Lean verifier didn't hold the hello's payload root to the store (`hold`): the binding is store.py's alone"
    for lean in ({}, {"refused": "no", "expected_root": reg["payload_root"]}, {"regions": [], "expected_root": "00" * 64}):
        unheld = analysed(out, host, sessions, reg, desc, lean=lean)
        assert not unheld["credit"]["credited_sessions"] and unheld["bound"]["reason"] == alone
    with pytest.raises(SystemExit, match="--commit"):
        al.analyse(out, host_bytes=1 << 30, pin=host.pin, period="p-1", heads=[host.host.ledger])


@pytest.mark.parametrize("sealed, main", [(PINS[:1], MAIN), (None, MAIN), (PINS, MAIN[2:]), (PINS, None)],
                         ids=["unpinned", "unsealed", "branch", "no-main"])
def test_a_session_whose_pin_isnt_a_logged_commit_on_mains_first_parent_line_earns_no_credit(host, tmp_path,
                                                                                              monkeypatch, sealed,
                                                                                              main):
    """An honest session, complete in a period that committed its verifier, on a host whose tree (`PIN`) isn't a
    logged commit on main's first-parent line when the session opens: the log the open sealed holds only an earlier
    pin, and the pin appended after the open doesn't count; or, as on every gate before RT4-INT-40, the open seals no
    log at all; or the log records the pin but the auditor's first-parent list of main doesn't hold it, as it doesn't a
    branch commit merged into main; or the auditor gives no copy of main. The line credits zero, and the session's
    `credited` check (`credited_session` reads the pin first) and its `pinned` check each name the fix. When the open
    sealed no log or one without the pin, the verifier's own read of the session (`coins.trusted`, under the host's
    log and its own copy of main) has already refused it as `coins`, so it is neither accepted nor of record; when only
    the auditor's copy of main fails, the verifier, holding main's own, accepted it."""
    host.seal(monkeypatch, sealed)
    out = tmp_path / "run"
    sessions, reg, desc = gate_run(host, out, "honest", defer_openings=True)
    rc, closed = host.call(["period-close", "--name", "p-1"])
    assert rc == 0 and json.loads(closed)["body"]["complete"] is True, closed
    later = PINS[:1] + [pin_line(PIN, "f" * 40, "2026-10-21T02:00:00Z")]
    doc = analysed(out, host, sessions, reg, desc, pins=PINS if sealed == PINS else later, main=main)
    (s,) = doc["sessions"]
    sid = s["challenger_session"]
    if sealed is None:
        why = (f"the auditor's ledger's open of session {sid} seals no pins log (pins: {{lines, sha512}}): open the "
               "window on a host whose gate seals its pins log in each open")
    elif sealed != PINS:
        why = (f"session {sid} ran commit {PIN}, which the host's pins log doesn't record as a commit of main's "
               "history: repin the challenger host to a commit origin/main contains (vy-challenger-pin) and rerun the "
               "window there")
    elif main is None:
        why = NO_MAIN
    else:
        why = (f"session {sid} ran commit {PIN}, which the host's pins log records but which is not on main's "
               "first-parent line in the verifier's copy (`git rev-list --first-parent origin/main`): a branch commit "
               "merged into main is in main's history but no tree main's check ran on; repin the challenger host to a "
               "commit on main's first-parent line (vy-challenger-pin) and rerun the window there")
    assert s["credit"]["complete"]["ok"]
    assert s["credit"]["credited"]["ok"] is False and s["credit"]["credited"]["reason"] == why
    assert s["credit"]["pinned"] == {"commit": PIN, "seal": None if sealed is None else pins_seal(sealed), "ok": False,
                                     "reason": why}
    if sealed != PINS:
        assert not s["accepted"] and s["failure"]["failure"] == "coins" and not s["of_record"]
        assert s["verifier_coin_check"] == f"the challenger's session for this session's window is not trusted: {why}"
        refused = "the Lean refused it (coins)"
        zero = [f"not credited: {why}", f"not accepted: {refused}", f"not pinned: {why}",
                "not of record: its ledger entries, its release log or its coins fail the checks beside it (ledger, "
                "log, verifier_coin_check)"]
        cells = f"no: {why} | yes | no: {refused} | no: {why}"
    else:
        assert s["credit"]["accepted"]["ok"] and s["of_record"] and s["verifier_coin_check"] is None
        zero = [f"not credited: {why}", f"not pinned: {why}"]
        cells = f"no: {why} | yes | yes | no: {why}"
    assert s["credit"]["zero"] == zero and not s["credit"]["ok"]
    t = s["T"]
    assert doc["line"][t]["credited_sessions"] == [] and doc["line"][t][al.line.LINK_CHARGED] == {
        **dict.fromkeys(al.SHARES, 0.0), "exact": dict.fromkeys(al.SHARES, "0")}
    assert doc["credit"]["summary"] == ("the line credits zero: no session is credited, complete, accepted, pinned "
                                        f"and bound; session 0: {'; '.join(zero)}")
    assert f"| 0 | {t} | {cells} | yes | zero |" in al.table(doc).splitlines()


@pytest.mark.parametrize("peer", ["vy-nebius-1", "vy-auditor-1"], ids=["n1", "host-run"])
def test_the_gate_sessions_peer_is_the_one_the_runs_record_names_and_a_record_naming_none_is_refused(host, tmp_path,
                                                                                                    peer):
    """The peer each gate session's reserve and open must carry on the ledger is the one the run's record names, with
    no --peer: verifier.json's gate sessions (`challengers.Gate.finish`) for a verifier on vy-nebius-1, or a host
    run's request.json beside out/ for one on vy-auditor-1 whose verifier.json predates that field (gate run 7's),
    the request's own `peer` being the audited node. Either way its session is of record. --peer overrides the record,
    and a session reserved for another peer is then not of record; a record naming two peers, or none, is refused with
    the flag."""
    out, other = tmp_path / "run", {"vy-nebius-1": "vy-auditor-1", "vy-auditor-1": "vy-nebius-1"}[peer]
    sessions, reg, desc = gate_run(host, out, "honest", defer_openings=True, peer=peer)
    assert [s["challenger"]["peer"] for s in sessions] == [peer]
    bare = [{**s, "challenger": {k: x for k, x in s["challenger"].items() if k != "peer"}} for s in sessions]
    request = tmp_path / "request.json"

    def requested(name: str) -> None:
        request.write_text(json.dumps({"peer": "vy-nebius-2", "argv": [
            "/usr/bin/python3", "-I", "verifier.py", "--challenger", "gate", "--challenger-peer", name,
            "--challenger-pin", host.pin]}))

    if peer == "vy-auditor-1":
        requested(peer)
    doc = analysed(out, host, sessions if peer == "vy-nebius-1" else bare, reg, desc)
    (s,) = doc["sessions"]
    assert doc["peer"] == {"name": peer, "from": "record"}
    assert s["ledger"]["ok"] and s["ledger"]["peer"] == s["ledger"]["open_peer"] == peer and s["of_record"]
    over = analysed(out, host, sessions, reg, desc, peer=other)
    (s,) = over["sessions"]
    assert over["peer"] == {"name": other, "from": "--peer"}
    assert not s["ledger"]["ok"] and not s["of_record"] and s["credit"]["zero"][-1].startswith("not of record")
    requested(other)
    with pytest.raises(SystemExit) as two:
        analysed(out, host, sessions, reg, desc)
    request.unlink()
    with pytest.raises(SystemExit) as none:
        analysed(out, host, bare, reg, desc)
    for e, named in ((two, "challenger peers vy-auditor-1, vy-nebius-1"), (none, "no challenger peer")):
        assert str(e.value) == (f"auditor_line: the run's record names {named} (verifier.json's gate sessions, "
                                f"{request}'s --challenger-peer): pass --peer NAME, the verifier machine's name in the "
                                "challenger host's /etc/vy/peers")


def test_a_session_earns_credit_only_under_the_commit_its_verifier_ran_at_on_mains_first_parent_line(
        host, tmp_path, monkeypatch):
    """An honest session, complete in a period that committed its verifier's descriptor, on a challenger host pinned
    on main's first-parent line. The line credits it only when the commit its verifier ran at (`--commit`) is on the
    auditor's first-parent list of main too: a verifier host pinned at a branch commit (pin.sh pins any pushed commit,
    and a period commits whichever verifier its opener names) earns none, with the fix. A host run's request.json
    records the commit its verifier ran at (`pinned`), and a --commit other than it is refused."""
    host.seal(monkeypatch, PINS)
    out = tmp_path / "run"
    sessions, reg, desc = gate_run(host, out, "honest", defer_openings=True)
    rc, closed = host.call(["period-close", "--name", "p-1"])
    assert rc == 0 and json.loads(closed)["body"]["complete"] is True, closed
    request, branch = tmp_path / "request.json", "9" * 40
    for ran in (VERIFIER, branch):
        request.write_text(json.dumps({"peer": "vy-nebius-2", "pinned": ran}))
        doc = analysed(out, host, sessions, reg, desc, commit=ran)
        (s,) = doc["sessions"]
        assert all(s["credit"][c]["ok"] for c in ("complete", "accepted", "pinned", "bound"))
        if ran == VERIFIER:
            assert s["credit"]["ok"] and doc["credit"]["credited_sessions"] == [0]
            continue
        why = (f"its verifier ran commit {branch}, which is not on main's first-parent line in the auditor's copy "
               "(`git rev-list --first-parent origin/main`), so no tree main's check ran on: repin the verifier host "
               "to a commit on main's first-parent line (vy-verifier-pin) and rerun the window there")
        assert s["credit"]["credited"] == {"descriptor_sha512": hashlib.sha512(desc).hexdigest(), "period": "p-1",
                                           "ok": False, "reason": why}
        assert s["credit"]["zero"] == [f"not credited: {why}"] and doc["credit"]["credited_sessions"] == []
    with pytest.raises(SystemExit, match=f"ran its verifier at {branch} .*pass --commit {branch}"):
        analysed(out, host, sessions, reg, desc, commit=VERIFIER)


def test_the_clock_rate_beside_the_line_doesnt_read_the_first_coins_round_trip():
    """Coins 110 ms apart on a challenger clock 30 ppm fast, σ_0 and σ_1 stamped a 100 ms round trip before the
    challenger released them and every later coin 50 ms after: the cross-check's rate reads 30 ppm, where the ratio end
    to end reads the round trip over the session's span."""
    ms = 1_000_000
    Rv = [g * 110 * ms for g in range(96)]
    Ra = [round(t * (1 + 30e-6)) + 7_000 * ms + (100 * ms if g < 2 else -50 * ms) for g, t in enumerate(Rv)]
    assert al.clock_rate_ppm(Rv, Ra) == pytest.approx(30, abs=0.01)
    assert ((Ra[-1] - Ra[0]) / (Rv[-1] - Rv[0]) - 1) * 1e6 < -14_000
    assert al.clock_rate_ppm(Rv[:3], Ra[:3]) is None
