"""`challengers.ChallengerGate` against the auditor host's own gate (`benchmarks/live_challenger/auditor.py`), in this process: its
requests go to `auditor.gate` as the lane key's forced command would pass them, and each opened session is served by
`auditor.run` through `serve.py` (the Lean's coins stubbed, `stub_serve.py`) on a thread, with the gate's root steps
(chown, systemctl) and the clock injected. An honest session whose coins the gate's session released is accepted and
held to the ledger; a challenger key other than the pinned one, a reserve outside the audit's period, a second session
for the same window and a challenger whose commitment isn't the open's are each refused before a coin leaves. The
verifier's own paced sessions, through the gate or serve.py, reserve only the coins they use, so an honest one ends
complete and a dropper doesn't; and `auditor_line` credits a gate session to the line only when its period committed
its verifier, at the commit its host run recorded, on the auditor's first-parent list of main, it and its period closed
complete, the Lean accepted it, the host's pins log its open sealed records its pin, a commit on that list too, and
the verifier bound the store to the model, reporting filler held beside weights held. The
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
from fractions import Fraction
from functools import lru_cache
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
from verity.core.protocols.certifier import certificate as cert  # noqa: E402
from verity.core.protocols.certifier.grid import COMPLETE, GridParams, BlockGrid, fill  # noqa: E402
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
        (root / "peers").write_text("challenger-host research@192.0.2.11\ncpu-node research@192.0.2.12\n"
                                    "auditor-host challenger@192.0.2.13\n")
        (root / "coin-ports").write_text("cpu-node 7811-7854\nauditor-host 7855-7899\n")
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
        """The session's unit: serve.py on a port of its own, returning once it listens."""
        body = A.sessions_of(A.read_ledger(self.host.ledger))[session]["open"]["body"]
        argv = A.serve_argv(self.host, session, body)
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
             call=None, peer: str = "cpu-node") -> challengers.ChallengerGate:
        return challengers.ChallengerGate(out, call or self.call, peer, pin or self.pin, period, main,
                                          address=self.address, pause=self.pause, polls=3)

    def period(self, name: str = "p-1", circuit: str = CIRCUIT) -> None:
        rc, out = self.call(["period-open", "--name", name, "--cap", "4", "--circuit", circuit])
        assert rc == 0, out

    def ledger(self) -> list[dict]:
        return A.entries(A.read_ledger(self.host.ledger))

    @staticmethod
    def seal(monkeypatch, pins: list[str] | None) -> None:
        """Each open from here on seals the pins log `pins` (`challenger.pins_seal`) in place of the host's own log,
        which `auditor.admit` seals; `pins` None: the open seals no log, as on the earlier gates."""
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


def session(store, gate: challengers.ChallengerGate):
    """An honest session over `store` whose coins come from `gate`'s session for window 0: the session and verdicts."""
    p = params()
    client = gate.open(store.registration.digest, 0, p.windows, 8)
    c = coins.SessionCoins(client, store.registration.digest, 0, p.windows, clock=itertools.count().__next__)
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
    assert (reserve["by"], reserve["peer"], reserve["scheme"], reserve["epoch"]) == (LANE, "cpu-node", coins.SCHEME, 0)
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
    """`ChallengerGate.finish` reads the window's session through `coins.trusted`, so through
    `challenger.pinned_session`, with the host's pins log as the gate serves it (`pins`, fetched beside the ledger) and the verifier's own first-parent
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
    flags = dict(ssh="-i /nonexistent/key research@192.0.2.1", peer="cpu-node", pin="ab" * 64, period="p-1")
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
    gate's sessions (`challengers.ChallengerGate`) or from serve.py on this machine (`challengers.Serve`), its Lean coins stubbed.
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
    (honest, dropper), _ = run_sessions("honest,skip:0.0625", paces=[pace, pace], challenger=source,
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
             peer: str = "cpu-node", groups_per_coin: int = 1) -> tuple[list[dict], dict, bytes]:
    """A gate run of `plan`'s sessions in period p-1, which committed the descriptor of the verifier that ran them, its
    verifier at the challenger host as `peer`, each session's windows drawn `groups_per_coin` a coin: the sessions,
    their registration and that descriptor."""
    desc = al.descriptor("ab" * 20)
    host.period(circuit=hashlib.sha512(desc).hexdigest())
    got = {}
    pace = verifier.pace_of("1:4:160:2", 1.0)
    sessions, _ = run_sessions(plan, paces=[pace] * len(plan.split(",")),
                               challenger=host.gate(out / "challenger", peer=peer), defer_openings=defer_openings,
                               groups_per_coin=groups_per_coin,
                               then=lambda v: got.update(reg=wire.registration_json(v.registration)))
    return sessions, got["reg"], desc


def analysed(out: Path, host: Host, sessions: list[dict], reg: dict, desc: bytes, *, verdict: str = "lean",
             pins: list[str] | None = PINS, main: list[str] | None = MAIN, bound: dict | None = MODEL,
             model: dict | None = MODEL, peer: str | None = None, lean: dict | None = None,
             commit: str = VERIFIER, line_by: al.lv.Lean | None = None, boundary: dict | None = None,
             lockdown: dict | None = None) -> dict:
    """`auditor_line`'s analysis of the run, its verifier.json recording a binding of the store to `bound` (none if
    None), the Lean's `hold` answering `lean` (by default bound at the registration's root), the line held to `model`
    (`--model-pin`'s), `peer` its --peer (None: the record's), `commit` the commit its verifier ran at, each line
    computed by the Lean verifier `line_by` (None: line.py alone), its credit held to the certifiers' records
    `boundary` (`read_boundary`'s; None: not checked), and its sessions' NoOutsideCompute grounded by `lockdown`
    (`read_lockdown`'s; None: by nothing)."""
    held = {"regions": [], "expected_root": reg["payload_root"]} if lean is None else lean
    binding = {} if bound is None else {"binding": {**bound, "payload_root": reg["payload_root"],
                                                    "expected_root": reg["payload_root"], "bound": True, "lean": held}}
    (out / "verifier.json").write_text(json.dumps(
        {"registration": reg, "sessions": [{**s, "verdict": verdict} for s in sessions]} | binding))
    return al.analyse(out, host_bytes=1 << 30, pin=host.pin, period="p-1", heads=[host.host.ledger], desc=desc,
                      commit=commit, pins=pins, main=main, model=model, peer=peer, lean=line_by, boundary=boundary,
                      lockdown=lockdown)


def test_the_line_reads_the_sketch_at_its_outside_factor() -> None:
    """`sketch_at` reads a session's SketchBound δ at F times the compressions its verifier charged the node
    (gate run 8's session 0: 2^-51.59 at F = 1, the red team's 2^-41.59 at 2^10), at the record's F unless asked
    another (1 for a record written before F), and the line's error there is the horizon's plus it; a session with no
    sketch term has none."""
    c, p, n, horizon = 12_708_215_084_432, 2118, 97_517_568, 9.64e-10
    sf = {"compressions": c, "log2": -51.59}
    s = {"ledger": {"sketch_failure": sf, "inputs": {"passes": p, "rows": 16}, "spec": {"error": {"horizon": horizon}}}}
    for f, want in ((1, -51.59), (2 ** 10, -41.59), (2 ** 20, -31.59)):
        got = verifier.sketch_at(s, {"blocks": n}, f)
        assert (got["outside"], got["compressions"], got["queries"], round(got["log2"], 2)) == (f, c, f * c, want)
        assert got["error"] == pytest.approx(horizon + 2.0 ** got["log2"])
    assert verifier.sketch_at(s, {"blocks": n}) == verifier.sketch_at(s, {"blocks": n}, 1)
    sf["outside"] = 2 ** 10
    assert verifier.sketch_at(s, {"blocks": n}) == verifier.sketch_at(s, {"blocks": n}, 2 ** 10)
    assert verifier.sketch_at({"ledger": {}}, {"blocks": n}) is None


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
    assert doc["sketch_outside"] == [2 ** 20] and honest["sketch"]["outside"] == 2 ** 20
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
        assert honest["error"] == al.session_error(sessions[0]) and "error" not in dropper
        lg = sessions[0]["ledger"]
        assert honest["error"]["total"] == lg["declared"]["memory"][al.ledger.SNAPSHOT]["delta"]
        assert [honest["error"][f"delta_{n}"] for n in al.ledger.DELTA_PARTS] == [
            lg["delta"][n] for n in al.ledger.DELTA_PARTS]
        assert honest["error"]["delta_rejection"] > 0 and honest["error"]["named_terms"] == al.NAMED_TERMS
        assert doc["line"][t]["error"] == doc["credit"]["error"] == al.union_error([honest])
        assert next(x for x in al.table(doc).splitlines() if x.startswith(f"line {t}:")).endswith(
            "): the ledger's δ (profile δ + P·2^-slack + Σ_t hide_t + bind + δs + Σ_t k_t/2^64) summed over the "
            "credited "
            "sessions, "
            "beside Σ_t Adv(prf t) (claim prf/sha-512), named and in no total")
        assert doc["conditional_on"] == al.CONDITIONAL_ON
        assert list(doc["conditional_on"]) == ["NoOutsideCompute", "binding-split"]
        assert doc["boundary"] == {"checked": False, "reason": "off by default (--boundary)"}
        lines = set(al.table(doc).splitlines())
        assert {f"conditional on {n}: {what}" for n, what in al.CONDITIONAL_ON.items()} <= lines
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
        assert doc["line"][t]["credited_sessions"] == [] and doc["line"][t]["error"]["sessions"] == []
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


def test_a_session_drawn_four_windows_a_coin_is_credited_by_the_line_of_record_and_a_skipper_is_not(
        host, tmp_path, monkeypatch):
    """`--groups-per-coin 4` at two windows in flight: `auditor_line`'s line of record over a gate run of an honest
    session and a prover that skips half its reads, each paced in twelve windows drawn on three openings coins, which
    the gate reserves and releases (⌈12/4⌉). The period closes complete; the honest session, which the Lean accepts
    and whose record names g, is credited, its twelve windows' last roots read on the challenger's clock off its
    three openings coins and the challenges after them (`auditor_seen`), its pooled bound over its three draws
    (`line.draws_of`), and the Lean verifier's line is line.py's; the skipper, rejected at a receipt, earns none."""
    host.seal(monkeypatch, PINS)
    out = tmp_path / "run"
    sessions, reg, desc = gate_run(host, out, "honest,skip:0.5", True, groups_per_coin=4)
    honest, skipper = sessions
    assert honest["accepted"] and honest["windows"] == 12 and honest["record"]["public"]["groups_per_coin"] == 4
    assert not skipper["accepted"] and skipper["failure"]["failure"] == "receipt"
    reserves = [e["body"]["families"] for e in host.ledger() if e["kind"] == "reserve"]
    assert reserves == [[[coins.CHALLENGE, 12], [coins.OPENINGS, 3]]] * 2
    rc, closed = host.call(["period-close", "--name", "p-1"])
    assert rc == 0 and json.loads(closed)["body"]["complete"] is True, closed
    doc = analysed(out, host, sessions, reg, desc)
    h, s = doc["sessions"]
    assert h["credit"]["ok"] and h["of_record"] and s["credit"]["accepted"]["reason"] == "the Lean refused it (receipt)"
    assert {k: [why.split(":")[0] for why in v] for k, v in doc["credit"]["zero"].items()} == {1: ["not accepted"]}
    t, mine = h["T"], h["shares_auditor_clock"][al.line.TIME_HELD]
    assert doc["line"][t]["credited_sessions"] == [0] and doc["line"][t][al.line.TIME_HELD] == {
        "k": 0, **{x: mine[x] for x in (*al.SHARES, "exact")}}
    assert h["groups"] == 12 and sum(h["named"]["last_root_first_named_by"].values()) == 12
    assert len(verifier.recorded_windows(honest)[0]) == 3
    by = al.lv.Lean(al.lv.build(), lambda _: None)
    try:
        assert doc["line_by"] == "python" and analysed(out, host, sessions, reg, desc, line_by=by) == {
            **doc, "line_by": "lean"}
    finally:
        by.close()


@pytest.mark.parametrize("sealed, main", [(PINS[:1], MAIN), (None, MAIN), (PINS, MAIN[2:]), (PINS, None)],
                         ids=["unpinned", "unsealed", "branch", "no-main"])
def test_a_session_whose_pin_isnt_a_logged_commit_on_mains_first_parent_line_earns_no_credit(host, tmp_path,
                                                                                              monkeypatch, sealed,
                                                                                              main):
    """An honest session, complete in a period that committed its verifier, on a host whose tree (`PIN`) isn't a
    logged commit on main's first-parent line when the session opens: the log the open sealed holds only an earlier
    pin, and the pin appended after the open doesn't count; or, as on the earlier gates, the open seals no
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


@pytest.mark.parametrize("peer", ["cpu-node", "auditor-host"], ids=["n1", "host-run"])
def test_the_gate_sessions_peer_is_the_one_the_runs_record_names_and_a_record_naming_none_is_refused(host, tmp_path,
                                                                                                    peer):
    """The peer each gate session's reserve and open must carry on the ledger is the one the run's record names, with
    no --peer: verifier.json's gate sessions (`challengers.ChallengerGate.finish`) for a verifier on a CPU node, or a
    host run's request.json beside out/ for one on the verifier host whose verifier.json predates that field (gate run 7's),
    the request's own `peer` being the audited node. Either way its session is of record. --peer overrides the record,
    and a session reserved for another peer is then not of record; a record naming two peers, or none, is refused with
    the flag."""
    out, other = tmp_path / "run", {"cpu-node": "auditor-host", "auditor-host": "cpu-node"}[peer]
    sessions, reg, desc = gate_run(host, out, "honest", defer_openings=True, peer=peer)
    assert [s["challenger"]["peer"] for s in sessions] == [peer]
    bare = [{**s, "challenger": {k: x for k, x in s["challenger"].items() if k != "peer"}} for s in sessions]
    request = tmp_path / "request.json"

    def requested(name: str) -> None:
        request.write_text(json.dumps({"peer": "gpu-node", "argv": [
            "/usr/bin/python3", "-I", "verifier.py", "--challenger", "gate", "--challenger-peer", name,
            "--challenger-pin", host.pin]}))

    if peer == "auditor-host":
        requested(peer)
    doc = analysed(out, host, sessions if peer == "cpu-node" else bare, reg, desc)
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
    for e, named in ((two, "challenger peers auditor-host, cpu-node"), (none, "no challenger peer")):
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
        request.write_text(json.dumps({"peer": "gpu-node", "pinned": ran}))
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


BOUNDARY_GRID = GridParams(buckets=4, allocation=2, frame_bytes=64)
#: the boundary certifiers' session
CERTIFIER_SESSION = bytes(range(32))
#: the verifier's messages over the session as the boundary links frame them: two frames on the uplink, none on the BMC's
VERIFIER_FRAMES = {"uplink": [b"\x01" * 64, b"\x02" * 64], "bmc": []}


@lru_cache(maxsize=None)
def certifier_key() -> lms.PrivateKey:
    return lms.keygen(lms.Params(64, 1, 6), hashlib.sha512(b"boundary certifier").digest())


def boundary(d: Path, let_in: dict[str, list[bytes]], *, links: tuple[str, ...] = ("uplink", "bmc"),
             windows: tuple[tuple[int, int, int], ...] = ((0, 0, 1 << 62),)) -> tuple[Path, str]:
    """A boundary file in `d` (`auditor_line.read_boundary`'s form) for `links`, whose certifier signed each link's
    ingress window 0 letting in `let_in`'s frames (a link it doesn't name has no record), its windows `windows`
    ((window, start_ns, end_ns) on the challenger's clock) and the verifier's frames `VERIFIER_FRAMES`; and its key's
    pin."""
    d.mkdir(parents=True)
    key = certifier_key()
    issuer = cert.Issuer(cert.LmsSigner.create(key, d / "signer.json"), CERTIFIER_SESSION, coins=lambda n: b"\x07" * n)
    T, r = BOUNDARY_GRID.buckets, BOUNDARY_GRID.allocation
    records = [issuer.certify(BlockGrid(link, 0,
                                        fill(frames, [max(0, min(r, len(frames) - r * t)) for t in range(T)], r),
                                        COMPLETE), cert.Direction.INGRESS, BOUNDARY_GRID,
                              cert.parameters_digest(BOUNDARY_GRID)) for link, frames in let_in.items()]
    public = lms.public_key(key)
    (d / "boundary.json").write_text(json.dumps({
        "links": list(links), "key": {"N": 64, "W": 1, "H": 6, "public": public.hex()},
        "session": CERTIFIER_SESSION.hex(), "grid": {"buckets": T, "allocation": r, "frame_bytes": 64},
        "windows": [{"window": w, "start_ns": s, "end_ns": e} for w, s, e in windows],
        "verifier_frames": {link: {"0": [f.hex() for f in fs]} for link, fs in VERIFIER_FRAMES.items()},
        "records": [al.certified_record_json(x) for x in records]}))
    return d / "boundary.json", hashlib.sha512(public).hexdigest()


def test_a_session_earns_no_credit_when_its_boundarys_certifiers_show_ingress_beside_the_verifiers_messages(
        host, tmp_path, monkeypatch):
    """`--boundary` (off by default): an honest session, credited without it, keeps its credit when the network
    certifiers' signed ingress records of the gate node's boundary links over its credited span let in exactly the
    verifier's messages, and earns none, with the reason, when they show more: a frame on a link the verifier sent
    nothing on (a relay's helper talking to the node), frames as many as the verifier's but not its own (their root), a
    boundary link with no record, or windows that leave the span uncovered. Either way the line stays conditional on
    `NoOutsideCompute`. A file whose certifier key isn't the pinned one is refused."""
    host.seal(monkeypatch, PINS)
    out = tmp_path / "run"
    sessions, reg, desc = gate_run(host, out, "honest", defer_openings=True)
    rc, closed = host.call(["period-close", "--name", "p-1"])
    assert rc == 0 and json.loads(closed)["body"]["complete"] is True, closed

    def held(name: str, let_in: dict[str, list[bytes]], **kw) -> dict:
        return analysed(out, host, sessions, reg, desc, boundary=al.read_boundary(*boundary(tmp_path / name, let_in,
                                                                                           **kw)))

    doc = held("exact", VERIFIER_FRAMES)
    (s,) = doc["sessions"]
    assert s["credit"]["ok"] and doc["credit"]["credited_sessions"] == [0]
    assert s["credit"]["boundary"]["ok"] and s["credit"]["boundary"]["windows"] == [0]
    assert doc["boundary"]["checked"] and doc["boundary"]["links"] == ["uplink", "bmc"]
    assert doc["conditional_on"] == al.CONDITIONAL_ON
    assert any(x.startswith(f"| 0 | {s['T']} | yes | yes | yes | yes | yes | yes | credited: ")
               for x in al.table(doc).splitlines())
    relay = b"\x09" * 64
    for name, let_in, kw, why in (
            ("helper", {**VERIFIER_FRAMES, "bmc": [relay]}, {},
             "bmc's ingress window 0 let in 1 frames, the verifier's messages 0"),
            ("swapped", {**VERIFIER_FRAMES, "uplink": [b"\x01" * 64, relay]}, {},
             "uplink's ingress window 0 let in frames other than the verifier's messages (its root is not theirs)"),
            ("unrecorded", VERIFIER_FRAMES, {"links": ("uplink", "bmc", "nvlink")},
             "nvlink's ingress window 0 has no record to read (no-verdict)"),
            ("uncovered", VERIFIER_FRAMES, {"windows": ((0, 0, 1),)}, None)):
        doc = held(name, let_in, **kw)
        (s,) = doc["sessions"]
        got = s["credit"]["boundary"]["reason"]
        if why is None:
            assert re.fullmatch(r"the certifiers' windows cover its span \[(\d+), (\d+)\] ns only to \1", got), got
        else:
            assert got == why
        assert s["credit"]["zero"] == [f"not boundary: {got}"] and doc["credit"]["credited_sessions"] == []
        assert doc["conditional_on"] == al.CONDITIONAL_ON
    path, _ = boundary(tmp_path / "pinned", VERIFIER_FRAMES)
    with pytest.raises(SystemExit, match="not the one --boundary-pin names"):
        al.read_boundary(path, "00" * 64)


def test_a_session_inside_the_providers_lockdown_rests_on_it_and_any_other_on_no_outside_compute(tmp_path, host,
                                                                                                monkeypatch):
    """With infra's lockdown record (`--lockdown`), a gate session whose span as the challenger signed it (its
    open's started_utc to its end's ended_utc) lies strictly inside an applied interval has its NoOutsideCompute
    grounded by `ProviderLockdown`, and the line is conditional on that in its place. A span that starts or ends on
    an interval's edge at whole seconds, a node that mounts a shared filesystem, or no record leaves it conditional
    on NoOutsideCompute. Either way the line stays conditional on the gaps in its charge (`binding-split`). It
    changes no credit, and the line's label names what the file names (`line_label`). A record not of
    `read_lockdown`'s form is refused."""
    host.seal(monkeypatch, PINS)
    out = tmp_path / "run"
    sessions, reg, desc = gate_run(host, out, "honest", defer_openings=True)
    rc, closed = host.call(["period-close", "--name", "p-1"])
    assert rc == 0 and json.loads(closed)["body"]["complete"] is True, closed

    def record(name: str, intervals: list[tuple[str, str]], shared: bool | None = None) -> Path:
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"node": "gpu-node", "allow": [{"host": "auditor-host", "ip": "192.0.2.7"}],
                                 "intervals": [{"applied_utc": a, "lifted_utc": b, "records": ["art:" + "0" * 64]}
                                               for a, b in intervals], "shared_filesystem": shared}))
        return p

    def locked(name: str, intervals: list[tuple[str, str]], shared: bool | None = None) -> dict:
        return analysed(out, host, sessions, reg, desc, line_by=None,
                        lockdown=al.read_lockdown(record(name, intervals, shared)))

    doc = analysed(out, host, sessions, reg, desc, line_by=None)
    (s,) = doc["sessions"]
    start, end = s["no_relay"]["span_utc"]
    assert s["credit"]["ok"] and s["no_relay"]["grounded_by"] is None and doc["conditional_on"] == al.CONDITIONAL_ON
    assert doc["lockdown"] == {"read": False, "reason": "no lockdown record (--lockdown)"}
    assert al.line_label(doc) == "python conditional on NoOutsideCompute, binding-split"
    before, after = A.later(start, -1), A.later(end, 1)
    doc = locked("inside", [(A.later(start, -30), A.later(start, -20)), (before, after)])
    (s,) = doc["sessions"]
    assert s["credit"]["ok"] and s["no_relay"]["grounded_by"] == "ProviderLockdown"
    assert s["no_relay"]["interval"] == [before, after] and s["no_relay"]["shared_filesystem"] == "pending (infra)"
    assert doc["conditional_on"] == {**al.GROUNDED, "binding-split": al.CONDITIONAL_ON["binding-split"]}
    assert al.line_label(doc) == "python conditional on ProviderLockdown, binding-split"
    assert doc["lockdown"]["read"] and doc["lockdown"]["sha512"] == hashlib.sha512(
        (tmp_path / "inside.json").read_bytes()).hexdigest()
    assert f"no relay, session 0: ProviderLockdown, its span inside the lockdown applied {before} to {after}" in \
        al.table(doc).splitlines()
    for name, intervals, shared, why in (
            ("at-start", [(start, after)], None, "lies inside no interval"),
            ("at-end", [(before, end)], None, "lies inside no interval"),
            ("shared", [(before, after)], True, "gpu-node mounts a shared filesystem")):
        doc = locked(name, intervals, shared)
        (s,) = doc["sessions"]
        assert s["credit"]["ok"] and s["no_relay"]["grounded_by"] is None and why in s["no_relay"]["reason"]
        assert doc["conditional_on"] == al.CONDITIONAL_ON
    assert locked("answered", [(before, after)], False)["sessions"][0]["no_relay"]["shared_filesystem"] == \
        "none mounted"
    for bad in ({"node": "n", "allow": [], "intervals": [], "shared_filesystem": None},
                {"node": "n", "allow": [{"host": "a", "ip": "1"}], "intervals": [
                    {"applied_utc": after, "lifted_utc": before, "records": []}], "shared_filesystem": None},
                {"node": "n", "allow": [{"host": "a", "ip": "1"}], "intervals": []}):
        (tmp_path / "bad.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match="is not a lockdown record"):
            al.read_lockdown(tmp_path / "bad.json")


def test_the_boundary_check_needs_the_certifiers_windows_that_meet_the_span_to_cover_it():
    """A credited span from σ_0's release at 100 ns to the last root's arrival at 250 ns is covered by windows
    [0, 120) and [120, 251), whatever lies beyond it, and not with a gap at 120 or a last window ending at 250."""
    seen = [{"released_ns": 100, "arrivals_ns": [150]}, {"released_ns": 160, "arrivals_ns": [250]}]

    def check(*windows: tuple[int, int, int]) -> dict:
        return al.boundary_check({"links": [], "windows": list(windows)}, seen)

    got = check((0, 0, 120), (1, 120, 251), (7, 400, 500))
    assert got["ok"] and got["windows"] == [0, 1] and got["span_ns"] == [100, 250]
    assert check((0, 0, 120), (1, 121, 300))["reason"] == "the certifiers' windows cover its span [100, 250] ns only to 120"
    assert check((0, 0, 250))["reason"] == "the certifiers' windows cover its span [100, 250] ns only to 250"


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


def test_the_line_records_the_ledgers_delta_in_its_parts_and_their_sum_over_its_credited_sessions():
    """Each accepted paced session's error is the memory line's δ as the ledger computed it (its record's `declared`
    forms', exactly; a refused form records none), beside its five parts (verifier.json's `delta`,
    `ledger.delta_parts`), which must sum to it as the ledger adds them: forms that record two δ, or parts that don't
    sum to it, give no line. The hiding part, n·2^-191, is below the float δ's precision, so the exact parts exceed
    the recorded total by it. A ledger that didn't charge the draw's rejection has Σ_t k_t/2^64 added at the session's
    `samples`, so it reads as one that did, a window that opened all its receipts included. Every error names the
    windows' PRF advantages (claim prf/sha-512) with no value, in no total. The line at each T, and over every T,
    fails with at most the sum over its credited sessions, and a T that credits none has error 0."""
    def session(profile: float, passes: int, sketch: float, *, charged: bool = True) -> dict:
        d = {"profile": profile, "slack": passes * 2.0 ** -41, "hiding": al.ledger.hiding_over(passes),
             "sketch": sketch}
        if charged:
            d["rejection"] = 48 * passes * 2.0 ** -64
        total = 0.0
        for x in d.values():
            total += x
        memory = {f: {"delta": total} for f in al.ledger.FORMS} | {al.ledger.TIME_HELD: {"refused": "unpaced"}}
        window = {"lengths": [4096], "profile": {"delta": profile, "drawn": {"receipt": 48}}}
        return {"k": 0, "samples": 48, "record": {"public": {}, "profile": {"windows": [window] * passes}},
                "ledger": {"delta": d, "inputs": {"passes": passes}, "declared": {"memory": memory}}}

    a, b = session(2.0 ** -40, 700, 2.0 ** -60), session(2.0 ** -30, 96, 0.0)
    e = al.session_error(a)
    total = Fraction(a["ledger"]["declared"]["memory"][al.ledger.SNAPSHOT]["delta"])
    parts = {"delta_profile": Fraction(1, 2 ** 40), "delta_slack": Fraction(700, 2 ** 41),
             "delta_hiding": Fraction(al.ledger.hiding_over(700)), "delta_binding": Fraction(0),
             "delta_sketch": Fraction(1, 2 ** 60), "delta_rejection": Fraction(48 * 700, 2 ** 64)}
    assert e["exact"] == {**{k: str(x) for k, x in parts.items()}, "total": str(total)}
    assert e["total"] == float(total) and e["log2_total"] == al.timing.log2(total)
    assert sum(parts.values()) - total == parts["delta_hiding"] and e["delta_slack"] > 300 * e["delta_profile"]
    assert e["named_terms"] == al.NAMED_TERMS == {"prf_advantage": {
        "claim": "prf/sha-512", "term": "Σ_t Adv(prf t)", "value": None, "in_total": False}}
    uncharged = session(2.0 ** -40, 700, 2.0 ** -60, charged=False)
    assert Fraction(uncharged["ledger"]["declared"]["memory"][al.ledger.SNAPSHOT]["delta"]) == total - Fraction(
        48 * 700, 2 ** 64) and al.session_error(uncharged) == e
    opened = json.loads(json.dumps(uncharged))
    opened["record"]["profile"]["windows"] = [{"lengths": [32], "profile": {"delta": 2.0 ** -40,
                                                                            "drawn": {"receipt": 32}}}] * 700
    assert al.session_error(opened) == e, "a window that opens all 32 of its receipts is charged its 48 requested"
    bad = json.loads(json.dumps(a))
    bad["ledger"]["delta"]["sketch"] = 2.0 ** -31
    with pytest.raises(al.Disagree, match="don't sum to its memory line's δ"):
        al.session_error(bad)
    two = json.loads(json.dumps(a))
    two["ledger"]["declared"]["memory"][al.ledger.SNAPSHOT]["delta"] *= 2
    with pytest.raises(al.Disagree, match="the ledger computes one"):
        al.session_error(two)
    held = {f: {**dict.fromkeys(al.SHARES, 0.5), "exact": dict.fromkeys(al.SHARES, "1/2")} for f in al.line.FORMS}

    def row(k: int, t: str, ok: bool, error: dict) -> dict:
        return {"k": k, "T": t, "shares_auditor_clock": held, "error": error,
                "credit": {"ok": ok, "zero": [] if ok else ["not accepted: the Lean refused it (receipt)"]}}

    out, credit = al.credited_line([row(0, "T=1s", True, e), row(1, "T=1s", True, al.session_error(b)),
                                    row(2, "T=60s", False, e)])
    both = total + Fraction(b["ledger"]["declared"]["memory"][al.ledger.SNAPSHOT]["delta"])
    assert out["T=1s"]["error"] == credit["error"] == {"sessions": [0, 1], "total": float(both), "exact": str(both),
                                                       "log2": al.timing.log2(both), "named_terms": al.NAMED_TERMS}
    assert out["T=60s"]["error"] == {"sessions": [], "total": 0.0, "exact": "0", "log2": None, "named_terms": {}}


# Gate runs 8 and 9's accepted sessions (their verifier.json): (passes, the memory line's δ every form records, the
# sketch's log2 its `sketch_failure` records, the session's samples, which each window drew, one window a pass, every
# one of 9.75·10^7 receipts or more).
GATE_RUNS = {(8, 0): (2118, 9.640646787142758e-10, -51.59094185809943, 48),
             (8, 2): (96, 4.4565254207300316e-11, -56.00597544028051, 232),
             (9, 0): (2118, 9.6406468008027e-10, -51.58427317163358, 48),
             (9, 2): (96, 4.4565254698432485e-11, -55.95559677531685, 232)}


@pytest.mark.parametrize("run,k", GATE_RUNS)
def test_the_lines_error_is_the_ledgers_delta_on_gate_runs_8_and_9(run, k):
    """On gate runs 8 and 9's records, written before verifier.json carried δ's parts or the slack (`slack_log2`), a
    session's error is the memory line's δ its ledger recorded, bit for bit, plus the coin commitment's hiding and the
    draw's rejection it didn't charge: 2^-40 of profile δ, AuditRO's 2^-41 a pass (`UNRECORDED_SLACK_LOG2`, the slack they were charged at),
    which dominates (2^-29.95 over a 60 s session's 2118 passes, 2^-34.39 over a 1 s session's 96), δs, the rest,
    which is the sketch the record charged, n·2^-191 over its n = P windows, and Σ_t k_t/2^64 at the session's samples,
    which every window drew (2118·48 = 101,664 draws, 2^-47.37; 96·232 = 22,272, 2^-49.56). A record that states its
    slack is split at it: at 67 bits the total is 2^-39.991 over 60 s and 2^-39.998 over 1 s, where it was 2^-39.9995
    and 2^-40.0000 before the rejection."""
    passes, delta, sketch_log2, drawn = GATE_RUNS[run, k]
    window = {"lengths": [2 ** 27], "profile": {"delta": 2.0 ** -40, "drawn": {"receipt": drawn}}}
    memory = {f: {"delta": delta} for f in al.ledger.FORMS}
    s = {"k": k, "samples": drawn, "record": {"public": {}, "profile": {"windows": [window] * passes}},
         "ledger": {"inputs": {"passes": passes}, "declared": {"memory": memory}}}
    e = al.session_error(s)
    parts = {n: al.line.number(e["exact"][f"delta_{n}"]) for n in al.ledger.DELTA_PARTS}
    rejection = Fraction(passes * drawn, 2 ** al.ledger.REJECTION_LOG2)
    assert parts["rejection"] == rejection == (Fraction(3177, 2 ** 59) if passes == 2118 else Fraction(87, 2 ** 56))
    hiding = Fraction(al.ledger.hiding_over(passes))
    assert parts["hiding"] == hiding and float(hiding) == pytest.approx(passes * 2.0 ** -191, rel=1e-12)
    assert al.line.number(e["exact"]["total"]) == Fraction(delta) + hiding + rejection == sum(parts.values())
    assert e["total"] == float(Fraction(delta) + rejection) and e["named_terms"] == al.NAMED_TERMS
    assert al.V.UNRECORDED_SLACK_LOG2 == 41 < al.audit.SLACK_LOG2
    assert parts["profile"] == Fraction(1, 2 ** 40) and parts["slack"] == Fraction(passes, 2 ** 41)
    assert al.timing.log2(parts["sketch"]) == pytest.approx(sketch_log2, abs=1e-6)
    assert e["log2_total"] == pytest.approx(-29.950 if passes == 2118 else -34.385, abs=1e-3)
    stated = json.loads(json.dumps(s))
    stated["ledger"]["inputs"]["slack_log2"] = al.audit.SLACK_LOG2
    slack = al.line.number(al.session_error(stated)["exact"]["delta_slack"])
    assert slack == Fraction(passes, 2 ** al.audit.SLACK_LOG2) and al.timing.log2(slack) < -55.9
    at67 = parts["profile"] + slack + parts["sketch"]
    assert al.timing.log2(at67) == pytest.approx(-39.9995 if passes == 2118 else -40.0000, abs=1e-4)
    assert al.timing.log2(at67 + rejection) == pytest.approx(-39.9908 if passes == 2118 else -39.9981, abs=1e-4)
