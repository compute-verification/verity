"""H1's coins from the auditor host's challenger: each attack below is refused, the release log's rewrite by the
challenger's signed close, sealed under its ledger.

* a window is held to the one challenge phase the challenger gate opened for its Commit's root, in the gate's
  signed v2 ledger (`pouw_h1_challenger.session_reason`, `Ledger`, core's check), ended closed and complete under the
  coin commitment the gate's answer gave it, which the serving side checked before its first coin; its tree is read
  through the gate's ``fetch`` and sealed by its end entry. Two challenge phases for one root, a reused challenge phase, a key off the ledger's pin, leaves
  out of order, a record leaf outside its challenge phase, a two-step open off its reserve and two published heads that don't
  extend one another are refused, and so is a tampered byte of the gate's own v2 head. A period (`period_reason`) is its
  lane's windows between its signed period-open and period-close: one with no close, an incomplete one, an accepted
  window outside it and a window of it abandoned after its coins are refused.
* the window's draw coin opens once, after a root over the manifest, the last receipts and every round's root
  and coin; an early draw, a draw from another family and a challenge phase closed with no draw are refused.
* round s's root commits the A and z tables of its calls and round s - 1's receipts.
* a window whose coins were held in the serving process is no challenge phase's and credits nothing; pouw-serve
  takes the auditor host's challenger (the Commit's coins, its C2 replay challenge among them: `test_commit_coins`).
* a window admits in round 0 only unless its ``admission`` is ``per-round``.
* the release log is the only evidence that a round's coin followed its root, so an unsigned log would pass roots chosen
  after every coin once it was rewritten to name them. The challenger's signed close signs it and the auditor's ledger
  seals the close, and the gate's v2 end entry seals the log itself: a rewritten log is refused either way.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import salt as PS  # noqa: E402
from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.sampled_proofs import challenger as LC  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import LiveCommitment, record_message  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402
from verity_vllm.pipeline import pouw_serve  # noqa: E402
from verity_vllm.protocols import pouw  # noqa: E402
from verity_vllm.protocols.pouw import h1 as PH  # noqa: E402
from verity_vllm.protocols.pouw.h1 import challenger as PHC  # noqa: E402
from verity_vllm.protocols.pouw.h1 import requests as RQ  # noqa: E402

from tests.protocols import challenger_host as CH  # noqa: E402
from tests.protocols import test_pouw_h1 as T  # noqa: E402

PRIVATE, KEY = CH.challenger_key(1, H=5)
COMMIT_A, COMMIT_B = "ca" * 64, "cb" * 64
#: two H1 windows' challenge phases, reserved on the gate before it opens them
SA, SB = CH.session_id(0xa), CH.session_id(0xb)
#: the gate's v2 head (31 entries: v1 opens, one-step v2 opens, then periods infra-smoke-20261020a to d,
#: c's and d's windows two-step) and seven of its challenge phases' trees as the gate's ``fetch`` gave them; registered
#: (fixtures/artifacts.json), fetched by `research data fetch-fixtures`
SMOKE = Path(__file__).parent / "fixtures" / "challenger_periods"
#: the SHA-512 of the challenger key the gate signed its ledger with; a verifier reads its
#: pin from config, never from code
SMOKE_PIN = ("27271eaf917d1a817198cbcfe7684debd9bdb11b9e386460948bb981642fab3a"
             "e065c0a877330a0da68ae76969550eb786dc172319876aaca21fd116e5a193bc")


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (T.Linear,))


class _Session:
    """One challenge phase `name` of the auditor's challenger under its one key `KEY`, its record signed at LMS leaf `leaf` for
    the gate's host ``vy-challenger-1.<name>`` and its close at `close()` (by default `leaf` + 1), with its own coins
    (`seed`), reached at `T.AUDITOR` until the next challenge phase is made, each hello's secret kept in `secrets`; its tree (the
    gate's ``fetch``) is `tree`, `directory` its ``out/`` and `commitment` its coin commitment's root, which `T.SESSION`
    names while it is reached; `open` is the gate's open of it on a window's open root, and `window` the window
    ``{root}`` its ledger then opens it for (`CH.WINDOW` until then). A `reserved` challenge phase (its name a gate challenge phase id,
    `CH.session_id`) is reserved in `audited`'s ledger before its open, as the gate's two-step open does."""

    def __init__(self, tmp_path, monkeypatch, name: str, leaf: int, seed: int, challenger=None, close=None,
                 reserved: bool = False) -> None:
        self.name, self.tree, self.directory = name, tmp_path / name, tmp_path / name / "out"
        self.leaf, self.reserved = leaf, reserved
        self.challenger = challenger or Wd.live_challenger(0, 3, CH.entropy(seed))
        self.directory.mkdir(parents=True)
        (self.directory / R.HELLOS).write_text("")
        leaves = [leaf]

        def sign(m: bytes) -> bytes:
            q = leaves.pop() if leaves else (close() if close else leaf + 1)
            return lms.sign_at(PRIVATE, q, m).encode()
        session = R.open_session(self.challenger, KEY, f"{CH.HOST}.{name}", sign, self.directory, name)
        ours, theirs = socket.socketpair()
        self.reason: str | None = None
        self._ledger: Path | None = None
        self.window: dict = CH.WINDOW
        self._thread = threading.Thread(target=self._serve, args=(ours, session))
        self._thread.start()
        self.client = R.RemoteChallenger(theirs)
        self.commitment, self.secrets = self.challenger.commitment.root.hex(), []
        monkeypatch.setattr(R.RemoteChallenger, "connect", classmethod(lambda cls, at, secret=None: self._reach(at, secret)))
        monkeypatch.setitem(T.SESSION, "commitment", self.commitment)

    def open(self, session: str, root: str) -> dict:
        assert session == self.name
        self.window = {"root": root}
        return {"session": session, "seq": 0, "root": root, "commitment": self.commitment}

    def remote(self, **kw) -> dict:
        """`test_pouw_h1._remote`'s settings, the window reserving this challenge phase and opening it through `open`."""
        return T._remote(session={"session": self.name}, gate=self.open, **kw)

    def _reach(self, at, secret):
        self.secrets.append(secret)
        return self.client if at == T.AUDITOR else None

    def _serve(self, sock: socket.socket, session: R.Session) -> None:
        with sock:
            self.reason = R.serve(sock, session, iter(range(1 << 20)).__next__)

    def ended(self) -> str | None:
        self._thread.join()
        return self.reason

    def audited(self) -> Path:
        """The auditor's signed ledger (`CH.Ledger`, one head) under `KEY`, reserving this challenge phase at the leaf two before
        its record's when it is `reserved`, opening it for `window` at the leaf before its record's and ending it over its
        ``out/`` at the key's last leaf once it has ended (closed when its client closed it)."""
        if self._ledger is None:
            leaves = (self.leaf - 2,) * self.reserved + (self.leaf - 1, (1 << KEY.H) - 1)
            ledger = CH.Ledger(self.tree / "windows.jsonl", 1, iter(leaves), H=KEY.H)
            if self.reserved:
                ledger.reserve(self.name)
            ledger.open(self.name, self.window)
            ledger.end(self.name, self.directory, self.ended() is None)
            self._ledger = ledger.path
        return self._ledger

    def trusted(self, commits) -> str | None:
        """`trusted_rounds_reason` of `commits` against this challenge phase, as `audited` opens and ends it."""
        return PHC.trusted_rounds_reason(commits, self.directory, [self.audited()], self.window, KEY,
                                         pins=[self.audited().with_name("windows.pins.jsonl")], main=CH.MAIN)


def _serve(settings, seed: int = 9, prompt: int = 5, run: bool = False):
    """`test_pouw_h1._serve_window`'s two served steps of a 3-step window, its activations drawn from `seed` and request
    a's one prompt token `prompt`: the commitment (and the run, with `run`)."""
    model, runner = T.Model(T.Linear(32, 1), T.Linear(40, 2)), settings["pouw"]["runner"]
    comp = T.PO.install(model, T.PO.Context(stage="serve", enabled=(T._entry(),), settings=settings))
    try:
        g = torch.Generator().manual_seed(seed)
        for so, rows in ((T._so(("a", [prompt], 4)), (1, 5)),
                         (T._so(("b", [7], 2), ("c", list(range(8, 39)), 3), running={"a": 1}), (33, 2))):
            runner.execute_model(so)
            for m, layer in zip(rows, model.layers):
                layer.quant_method.apply(layer, (torch.randn(m, T.K, generator=g) * 3).bfloat16())
        h1 = pouw.h1_run(T._entry())
    finally:
        comp.close()
    return (comp.installed[0].commitment(), h1) if run else comp.installed[0].commitment()


def _coins(commitment):
    return [r["coin"]["coin"] for r in commitment["rounds"]]


def _window(gate, tmp_path, monkeypatch, name: str, seed: int):
    """A window served in challenge phase `name` (a gate challenge phase id), which `gate` reserved and then opened on the window's open
    root once round 0 was admitted (its answer's commitment, ``status SESSION``'s, once the challenge phase's record is signed)
    and ended over its ``out/``: its commitment and its challenge phase."""
    made = {}
    gate.reserve(name, Wd.live_families(3))

    def opened(session: str, root: str) -> dict:
        assert session == name
        made["reply"] = gate.open(session, root)
        made["session"] = _Session(tmp_path, monkeypatch, name, gate.take(), seed, close=gate.take, reserved=True)
        made["session"].window = {"root": root}
        return {**made["reply"], "commitment": made["session"].commitment}
    served = _serve(T._remote(session={"session": name}, gate=opened))
    session = made["session"]
    assert session.ended() is None and session.secrets == [bytes.fromhex(made["reply"]["secret"])]
    gate.end(name, session.directory)
    return served, session


def _gate():
    return CH.Gate(PRIVATE, KEY)


def test_a_window_is_held_to_the_one_session_the_auditors_ledger_opened_for_its_commits_root(tmp_path, monkeypatch):
    """Two challenge phases under the auditor's one key both pass `trusted_rounds_reason`, but the window carries the
    gate's answer and the ledger opened one challenge phase for its Commit's root: another challenge phase's window claiming it is
    refused, and so are an answer's seq off the ledger's, a root the ledger has no challenge phase for, a window with no challenge phase
    or another Commit, weights other than the period's registered ones, a challenge phase tree core's check refuses (another
    challenge phase's, a rewritten release log, a missing tree, a record signed under another key than the pinned one), a
    challenge phase that never ended, ended unclosed or short of its coins, an end entry under another commitment than the
    window's, a record whose commitment or leaf aren't the ones its end entry records, and a challenge phase the verifier holds
    no first-parent history of main for, or whose pin isn't on that history (`challenger.pinned_session`)."""
    gate = _gate()
    first, a = _window(gate, tmp_path, monkeypatch, SA, 11)
    second, b = _window(gate, tmp_path, monkeypatch, SB, 12)
    ra, rb = first["session"]["root"], second["session"]["root"]
    assert ra == PS.window_open_root(bytes.fromhex(first["weights_root"]), bytes.fromhex(first["requests_root"]),
                                     SA).hex() != rb
    assert a.trusted(first) is None and b.trusted(second) is None
    assert _coins(first) != _coins(second) and first["session"]["session"] == SA and first["session"]["seq"] == 1
    assert "secret" not in first["session"] and first["session"]["commitment"] == a.commitment
    ledger, fetch = gate.ledger(), {SA: a.tree, SB: b.tree}.get
    assert PHC.session_reason(first, ledger, fetch) is None and PHC.session_reason(second, ledger, fetch) is None
    assert PHC.session_reason(first, gate.ledger(main=None), fetch) == f"session {SA} is not trusted: {LC.NO_MAIN}"
    assert PHC.session_reason(first, gate.ledger(main=CH.MAIN[:1]), fetch) == (
        f"session {SA} is not trusted: {LC.first_parent_reason(SA, CH.PINNED, list(CH.MAIN[:1]))}")
    claims_a = {**second, "session": {**second["session"], "root": ra}}
    assert PHC.session_reason(claims_a, ledger, fetch) == (
        f"the window's session {SB} is not {SA}, the one the auditor's ledger opened for Commit {ra}")
    assert PHC.session_reason({**first, "session": {**first["session"], "seq": 3}}, ledger, fetch) == (
        f"the window's session {SA} is ledger entry 3 in its reply, but entry 1 opened it")
    assert PHC.session_reason({**first, "session": {**first["session"], "root": "cc" * 64}}, ledger, fetch) == (
        f"the auditor's ledger has no session for Commit {'cc' * 64}, so the window's coins are no session's of record")
    assert PHC.session_reason({k: v for k, v in first.items() if k != "session"}, ledger, fetch).startswith(
        "the window carries no challenger session")
    assert PHC.session_reason({**first, "session": {"id": "s-a"}}, ledger, fetch).startswith(
        "the window's challenger session is malformed")
    assert PHC.session_reason(first, ledger, fetch, commit=rb) == (
        f"the window's session is for Commit {ra}, not the window's Commit {rb}")
    assert PHC.session_reason(first, ledger, fetch, registered=bytes(64)) == (
        "the window's weights' root is not the one the period registered")
    assert PHC.session_reason(first, ledger, fetch, registered=bytes.fromhex(first["weights_root"])) is None
    assert PHC.session_reason({**first, "session": {**first["session"], "commitment": b.commitment}}, ledger, fetch) == (
        f"session {SA}'s end entry records coin commitment {a.commitment[:16]}..., not {b.commitment[:16]}..., the "
        "window's (the gate's answer): a refused window")
    assert PHC.session_reason(first, ledger, {SA: b.tree}.get) == (
        f"session {SA} is not trusted: the challenger's session is {SB}, not {SA}, the one the auditor's ledger opens "
        "for this window")
    assert PHC.session_reason(first, ledger, {}.get) == (
        f"session {SA} is not trusted: session {SA} has no tree on the auditor's side (the gate's fetch)")
    rewritten = tmp_path / "rewritten"
    shutil.copytree(a.tree, rewritten)
    (rewritten / "out" / R.RELEASES).write_text((a.directory / R.RELEASES).read_text().replace(
        first["rounds"][0]["root"], "ee" * 64))
    assert PHC.session_reason(first, ledger, {SA: rewritten}.get) == (
        f"session {SA} is not trusted: the challenger's release log is not the one its close record signs")

    def ended_over(out, closed=True, end=True, **override):
        g = _gate()
        g.reserve(SA, Wd.live_families(3))
        g.open(SA, ra)
        if end:
            g.end(SA, out, closed, **override)
        return g.ledger()
    assert PHC.session_reason({**first, "session": {**first["session"], "commitment": b.commitment}},
                              ended_over(b.directory), {SA: b.tree}.get) == (
        f"session {SA} is not trusted: the challenger's session is {SB}, not {SA}, the one the auditor's ledger opens "
        "for this window")
    other_private, other = CH.challenger_key(2)
    keyed = tmp_path / "keyed"
    shutil.copytree(a.tree, keyed)
    doc = json.loads((a.directory / R.RECORD).read_text())
    c = LiveCommitment.from_json(doc["commitment"])
    doc["signature"] = lms.sign_at(other_private, 2, record_message(other.public[:16], doc["host"], c)).encode().hex()
    (keyed / "out" / R.RECORD).write_text(json.dumps(doc))
    assert PHC.session_reason(first, ended_over(keyed / "out"), {SA: keyed}.get) == (
        f"session {SA} is not trusted: the challenger record's signature does not verify under the verifier's "
        "challenger key")
    assert PHC.session_reason(first, ended_over(a.directory, end=False), fetch) == (
        f"session {SA} never ended (the auditor's ledger has its open entry and no end): a refused window")
    assert PHC.session_reason(first, ended_over(a.directory, closed=False), fetch) == (
        f"session {SA} ended without its client closing it (serve.py exited 1): a refused window")
    short = {Wd.DRAW: 1, Wd.ROUND: 2, Wd.SALT: 1}
    assert PHC.session_reason(first, ended_over(a.directory, complete=False, released=short), fetch) == (
        f"session {SA} ended without releasing every coin of its families (released {short}): a refused window")
    assert PHC.session_reason({**first, "session": {**first["session"], "commitment": "dd" * 64}},
                              ended_over(a.directory, commitment="dd" * 64), fetch) == (
        f"session {SA} is not trusted: the challenger's record is signed at leaf 2 over commitment "
        f"{a.commitment[:16]}..., not at leaf 2 over {'dd' * 8}..., the ones the auditor's ledger's end names")
    assert PHC.session_reason(first, ended_over(a.directory, record_leaf=3), fetch) == (
        f"session {SA} is not trusted: the challenger's record is signed at leaf 2 over commitment "
        f"{a.commitment[:16]}..., not at leaf 3 over {a.commitment[:16]}..., the ones the auditor's ledger's end names")


def test_a_session_whose_tree_the_hosts_pins_log_doesnt_record_is_refused(tmp_path, monkeypatch):
    """Every challenge phase is read through core's `pinned_session`, so a window passes only from a tree the host's
    pins log records as main's, the log as the gate published it beginning with the lines the open sealed. Refused,
    through `session_reason` and `trusted_rounds_reason`: an open run from a commit the log doesn't record; no pins
    log (none given, or none readable at the published path); an open that seals none; a log cut before the lines the
    open sealed, or with one of them rewritten; and two published heads of it that don't extend one another."""
    gate = _gate()
    first, a = _window(gate, tmp_path, monkeypatch, SA, 11)
    ra, fetch = first["session"]["root"], {SA: a.tree}.get
    assert PHC.session_reason(first, gate.ledger(), fetch) is None

    def opened(log=CH.PINS, **override):
        g = _gate()
        g.pins = list(log)
        g.reserve(SA, Wd.live_families(3))
        g.open(SA, ra, **override)
        g.end(SA, a.directory)
        return g
    other = "ab" * 20
    unpinned = f"session {SA} is not trusted: session {SA} ran commit {other}, which the host's pins log doesn't record"
    assert PHC.session_reason(first, opened(pinned=other).ledger(), fetch).startswith(unpinned)
    assert PHC.session_reason(first, PHC.Ledger.of([gate.lines], KEY), fetch) == (
        f"session {SA} is not trusted: {LC.NO_PINS}")
    assert PHC.session_reason(first, opened(pins=None).ledger(), fetch).startswith(
        f"session {SA} is not trusted: the auditor's ledger's open of session {SA} seals no pins log")
    second = LC.pin_line("cd" * 20, "cd" * 20, "2026-10-20T10:30:00Z")
    two = opened([*CH.PINS, second])
    assert PHC.session_reason(first, two.ledger(), fetch) is None
    assert PHC.session_reason(first, two.ledger(CH.PINS), fetch) == (
        f"session {SA} is not trusted: the host's pins log has 1 lines, fewer than the 2 the open of session {SA} "
        "sealed: take the auditor host's operator's newest published pins log")
    assert PHC.session_reason(first, two.ledger([second, second]), fetch) == (
        f"session {SA} is not trusted: the host's pins log is not the one the open of session {SA} sealed: its first 2 "
        "lines were edited or rewritten")
    with pytest.raises(LC.CoinError, match="pins logs 0 and 1 don't extend one another"):
        PHC.Ledger.of([two.lines], KEY, pins=[[*CH.PINS, second], [second]])
    assert PHC.pins_of([CH.PINS, [*CH.PINS, second]]) == [*CH.PINS, second] and PHC.pins_of([]) is None

    heads = [a.audited()]
    published = heads[0].with_name("windows.pins.jsonl")
    assert a.trusted(first) is None
    assert PHC.trusted_rounds_reason(first, a.directory, heads, a.window, KEY, pins=None, main=CH.MAIN) == (
        f"the challenger's session is not trusted: {LC.NO_PINS}")
    assert PHC.trusted_rounds_reason(first, a.directory, heads, a.window, KEY, pins=[tmp_path / "none.jsonl"],
                                     main=CH.MAIN).startswith(
        f"the challenger's session is not trusted: {LC.NO_PINS} (none readable at {tmp_path / 'none.jsonl'}")
    CH.write_pins(published, [LC.pin_line(other, other, "2026-10-20T10:00:00Z")])
    assert PHC.trusted_rounds_reason(first, a.directory, heads, a.window, KEY, pins=[published], main=CH.MAIN) == (
        f"the challenger's session is not trusted: the host's pins log is not the one the open of session {a.name} "
        "sealed: its first 1 lines were edited or rewritten")


def test_the_ledger_refuses_two_sessions_for_one_root_a_reused_session_another_key_and_a_broken_chain():
    """`Ledger.of` (core's ``ledger_head``, ``verify_ledger`` and ``verify_periods``) refuses a whole ledger
    with a second challenge phase for one root, one challenge phase for two roots, an open of no root or under another key's pin, an
    end with no open or a second end, an entry signed under another key, edited after signing or at a leaf not after
    the last one's, a v1 entry after a v2, a record leaf outside its challenge phase's open and end, used twice, off form or
    also an entry's, a line dropped, and an entry of another kind; v1 entries before v2 ones are one chain. It reads the
    longest of every published head and refuses two that don't extend one another."""
    families = Wd.live_families(3)
    ended = {"closed": True, "files": {}}

    def refused(build, match):
        g = _gate()
        build(g)
        with pytest.raises(LC.CoinError, match=match):
            g.ledger()
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-b", COMMIT_A, families)),
            f"entry 1: a second open for root {COMMIT_A[:16]}… or session s-b")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-a", COMMIT_B, families)),
            f"entry 1: a second open for root {COMMIT_B[:16]}… or session s-a")
    refused(lambda g: g.open("s-a", COMMIT_A.upper(), families), "the auditor's ledger entry 0 is malformed")
    refused(lambda g: g.open("s-a", COMMIT_A, families, key_sha512="00" * 64),
            r"entry 0 is signed under challenger key 0+\.\.\., not the window's pinned key")
    refused(lambda g: g.append("end", "s-a", ended), "entry 0: an end for session s-a, which has no open or has ended")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.append("end", "s-a", ended), g.append("end", "s-a", ended)),
            "entry 2: an end for session s-a, which has no open or has ended")
    refused(lambda g: g.append("close", "s-a", {}), "the auditor's ledger entry 0 is malformed")
    refused(lambda g: (g.open("s-a", COMMIT_A, families, leaf=5), g.open("s-b", COMMIT_B, families, leaf=3)),
            "entry 1: signed at leaf 3, not after the last entry's 5")
    refused(lambda g: (g.open("s-a", COMMIT_A, families, leaf=5), g.open("s-b", COMMIT_B, families, leaf=5)),
            "entry 1: signed at leaf 5, not after the last entry's 5")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-b", COMMIT_B, families, v1=True)),
            "entry 1: not entry 1 of a verity/challenger-auditor/v2 ledger chained to its predecessor")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.append("end", "s-a", {**ended, "record_leaf": 0})),
            "entry 1: record leaf 0 isn't between its open's and its end's, once")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.append("end", "s-a", {**ended, "record_leaf": "2"}, leaf=3)),
            "the auditor's ledger entry 1 is malformed")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-b", COMMIT_B, families),
                       g.append("end", "s-a", {**ended, "record_leaf": 2}, leaf=3),
                       g.append("end", "s-b", {**ended, "record_leaf": 2})),
            "entry 3: record leaf 2 isn't between its open's and its end's, once")
    refused(lambda g: (g.open("s-a", COMMIT_A, families), g.open("s-b", COMMIT_B, families),
                       g.append("end", "s-a", {**ended, "record_leaf": 1})),
            "entry 2: record leaf 1 is also an entry's")
    g = _gate()
    g.open("s-0", "c0" * 64, families, v1=True)
    g.open("s-a", COMMIT_A, families)
    g.open("s-b", COMMIT_B, families)
    assert set(g.ledger().windows) == {"c0" * 64, COMMIT_A, COMMIT_B}
    assert g.ledger().window(COMMIT_A)["open"]["session"] == "s-a"
    with pytest.raises(LC.CoinError, match="entry 0 is signed under challenger key .*, not the window's pinned key"):
        PHC.Ledger.of([g.lines], CH.challenger_key(2)[1])
    edited = g.lines[1].replace('"port":7811', '"port":7812')
    with pytest.raises(LC.CoinError, match="entry 1: its signature doesn't verify under the challenger key"):
        PHC.Ledger.of([[g.lines[0], edited, g.lines[2]]], KEY)
    with pytest.raises(LC.CoinError, match="entry 0: not entry 0 of a verity/challenger-auditor/v2 ledger"):
        PHC.Ledger.of([g.lines[1:]], KEY)
    with pytest.raises(LC.CoinError, match="the auditor's ledger entry 0 is malformed"):
        PHC.Ledger.of([["not json"]], KEY)
    assert set(PHC.Ledger.of([g.lines[:1], g.lines, g.lines[:2]], KEY).windows) == {"c0" * 64, COMMIT_A, COMMIT_B}
    fork = _gate()
    fork.lines = g.lines[:2]
    fork._leaf = 2
    fork.open("s-c", "cc" * 64, families)
    assert set(PHC.Ledger.of([fork.lines], KEY).windows) == {"c0" * 64, COMMIT_A, "cc" * 64}
    with pytest.raises(LC.CoinError, match="the auditor's published heads 0 and 1 don't extend one another"):
        PHC.Ledger.of([g.lines, fork.lines], KEY)
    with pytest.raises(LC.CoinError, match="published head 1 is refused: .*entry 1: its signature doesn't verify"):
        PHC.Ledger.of([g.lines, [g.lines[0], edited]], KEY)
    with pytest.raises(LC.CoinError, match="no auditor's ledger"):
        PHC.Ledger.of([], KEY)


def test_a_period_is_its_lanes_windows_between_its_signed_open_and_close_and_credits_only_if_complete(tmp_path,
                                                                                                    monkeypatch):
    """A period is its lane's windows between its period-open and period-close entries (a window of another
    lane's isn't one); it credits only when the ledger closes it complete and every window of it was accepted. Refused: a
    period the ledger doesn't have or doesn't close, one closed with no window, an incomplete window (one that ended
    short of its coins, or never ended), an accepted window that isn't the period's, and a window of it abandoned after
    its coins."""
    families = Wd.live_families(3)
    sk, si, sn, so = (CH.session_id(n) for n in (0xe1, 0xe2, 0xe3, 0xe4))

    def opened(session, root, by="vllm"):
        gate.reserve(session, families, by=by)
        gate.open(session, root, by=by)
    gate = _gate()
    gate.period_open("p0")
    gate.period_close("p0")
    gate.period_open("p1", cap=4, circuit="c1" * 64)
    kept, a = _window(gate, tmp_path, monkeypatch, SA, 11)
    dropped, b = _window(gate, tmp_path, monkeypatch, SB, 12)
    opened(sk, "ce" * 64, by="kernels")
    gate.period_close("p1")
    gate.period_open("p2")
    opened(si, "cc" * 64)
    idle = _Session(tmp_path, monkeypatch, si, gate.take(), seed=13, close=gate.take)
    idle.client.close()
    assert idle.ended() is None
    gate.end(si, idle.directory)
    gate.period_close("p2")
    gate.period_open("p3")
    opened(sn, "cd" * 64)
    gate.period_close("p3")
    gate.period_open("p4")
    opened(so, "cf" * 64)
    ledger = gate.ledger()
    assert {n: p["windows"] for n, p in ledger.periods.items()} == {
        "p0": [], "p1": [SA, SB], "p2": [si], "p3": [sn], "p4": [so]}
    assert ledger.periods["p1"]["close"]["body"]["complete"] is True and ledger.periods["p4"]["close"] is None
    assert PHC.session_reason(kept, ledger, {SA: a.tree}.get) is None
    assert PHC.period_reason(ledger, "p1", [SA, SB]) == []
    assert PHC.period_reason(ledger, "p1", [SA]) == [
        f"period p1's window {SB} (Commit {dropped['session']['root']}) ended complete and has no accepted window: a "
        "window abandoned after its coins"]
    assert PHC.period_reason(ledger, "p1", [SA, SB, sk]) == [
        f"session {sk}, an accepted window, is not one of period p1's windows ({[SA, SB]}, lane vllm's between its "
        "period-open and its period-close)"]
    released = ledger.window("cc" * 64)["end"]["body"]["released"]
    assert set(released.values()) == {0} and PHC.period_reason(ledger, "p2", [si]) == [
        f"period p2's window {si} is incomplete (Commit {'cc' * 64}, ended 'closed by the client', released {released}): "
        "a refused window refuses its period"]
    assert PHC.period_reason(ledger, "p3", []) == [
        f"period p3's window {sn} is incomplete (Commit {'cd' * 64}, never ended): a refused window refuses its period"]
    assert PHC.period_reason(ledger, "p0", []) == ["period p0 closed with no window"]
    assert PHC.period_reason(ledger, "p4", [so]) == [
        "period p4 has no period-close in the auditor's ledger, so its windows aren't all known: an open period credits "
        "nothing"]
    assert PHC.period_reason(ledger, "p5", []) == ["the auditor's ledger has no period p5"]


def test_the_ledger_refuses_periods_and_reserves_off_the_gates_rules():
    """Core's ``verify_periods`` and two-step windows: `Ledger.of` refuses a period entry with a challenge phase, a
    period-open under another key's pin, a period off its name and lane, a period name used twice, a lane's second open
    period, a cap or circuit off form, a window past its cap, a close of a period that isn't its lane's open one or that
    doesn't list exactly its windows and their outcomes; a reserve that isn't first in its challenge phase or whose expiry isn't
    its start plus its minutes; and an open that binds no reserve of its challenge phase, doesn't carry its request, isn't by its
    prover, falls outside its minutes or after its period's close, or is one-step after the first reserve. A two-step
    window joins its prover's (``for``'s) period at its reserve, and a reserve never opened is listed with no root."""
    families, circuit = Wd.live_families(3), "c1" * 64
    s, t = "s20261020T110800Z-000a", "s20261020T110900Z-000b"

    def refused(build, match):
        g = _gate()
        build(g)
        with pytest.raises(LC.CoinError, match=match):
            g.ledger()
    refused(lambda g: g.append("period-open", "s-a", {"period": "p", "lane": "vllm", "key_sha512": g.pin}),
            "the auditor's ledger entry 0 is malformed")
    refused(lambda g: g.period_open("p", key_sha512="00" * 64),
            r"entry 0 is signed under challenger key 0+\.\.\., not the window's pinned key")
    refused(lambda g: g.period_open("p/1"), "the auditor's ledger entry 0 is malformed")
    refused(lambda g: (g.period_open("p"), g.period_close("p"), g.period_open("p")),
            "entry 2: period p is named twice, or lane vllm has a period open")
    refused(lambda g: (g.period_open("p"), g.period_open("q")), "entry 1: period q is named twice, or lane vllm has a")
    refused(lambda g: g.append("period-close", None, {"period": "p", "lane": "vllm", "windows": [], "complete": False}),
            "entry 0: a period-close for p, which isn't lane vllm's open period")
    refused(lambda g: (g.period_open("p"), g.period_close("p", lane="kernels")),
            "entry 1: a period-close for p, which isn't lane kernels's open period")
    refused(lambda g: (g.period_open("p"), g.open("s-a", COMMIT_A, families), g.period_close("p", windows=[])),
            "entry 2: period p's close doesn't list exactly its windows and their outcomes")
    refused(lambda g: (g.period_open("p"), g.period_close("p", complete=True)), "entry 1: period p's close doesn't list")
    for cap, c in ((0, circuit), (2, circuit.upper()), (2, None), (True, circuit)):
        refused(lambda g: g.period_open("p", cap=cap, circuit=c),
                "entry 0: period p's cap is a positive integer and its circuit a SHA-512 in lowercase")
    refused(lambda g: (g.period_open("p", cap=1, circuit=circuit), g.open("s-a", COMMIT_A, families),
                       g.open("s-b", COMMIT_B, families)), "entry 2: period p's window 1 is past its cap 1")
    refused(lambda g: g.reserve("s-a", families), "entry 0: a reserve is v2 and the first entry of its session s-a")
    refused(lambda g: (g.reserve(s, families), g.reserve(s, families)), f"entry 1: a reserve is v2 and the first entry of "
                                                                         f"its session {s}")
    refused(lambda g: (g.open(s, COMMIT_A, families), g.reserve(s, families)), f"entry 1: a reserve is v2 and the first "
                                                                                 f"entry of its session {s}")
    refused(lambda g: g.reserve(s, families, expires_utc="2026-10-20T11:09:00Z"),
            r"entry 0: a reserve names its prover, its minutes, its start and its expiry \(start \+ minutes\)")
    refused(lambda g: g.reserve(s, families, minutes=0), "entry 0: a reserve names its prover")
    refused(lambda g: (g.reserve(s, families), g.open(s, COMMIT_A, reserve=5)),
            f"entry 1: an open for session {s} binds no reserve of it")
    refused(lambda g: (g.reserve(s, families), g.open(s, COMMIT_A, peer="vy-nebius-2")),
            f"entry 1: session {s}'s open doesn't carry its reserve's request")
    refused(lambda g: (g.reserve(s, families), g.open(s, COMMIT_A, by="kernels")),
            f"entry 1: session {s} was opened by 'kernels', not its reserve's prover")
    refused(lambda g: (g.reserve(s, families), g.open(s, COMMIT_A, started_utc="2026-10-20T11:18:01Z")),
            f"entry 1: session {s} was opened outside its reserve's minutes")
    refused(lambda g: (g.period_open("p"), g.reserve(s, families), g.period_close("p"), g.open(s, COMMIT_A)),
            f"entry 3: session {s} was opened after its period p closed")
    refused(lambda g: (g.reserve(s, families), g.open(t, COMMIT_B, families)),
            "entry 1: a v2 open without a reserve after the ledger's first reserve")
    g = _gate()
    g.open("s-0", COMMIT_A, families)
    g.period_open("p", cap=2, circuit=circuit)
    g.reserve(s, families)
    g.reserve(t, families, by="vy-admin", for_="vllm")
    reply = g.open(t, COMMIT_B, started_utc="2026-10-20T11:17:59Z")
    g.append("end", t, {"closed": False, "files": {}, "complete": False})
    g.period_close("p")
    ledger = g.ledger()
    assert reply["reserve"] == 3 and reply["for"] == "vllm"
    assert PHC.session_of({**reply, "commitment": COMMIT_A})["opened_by"] == "vllm"
    assert ledger.periods["p"]["windows"] == [s, t] and (ledger.periods["p"]["cap"], ledger.periods["p"]["circuit"]) == (
        2, circuit)
    assert ledger.periods["p"]["close"]["body"]["windows"] == [{"root": None, "session": s, "complete": False},
                                                               {"root": COMMIT_B, "session": t, "complete": False}]
    assert ledger.window(COMMIT_B)["reserve"]["seq"] == 3 and ledger.window(COMMIT_A)["reserve"] is None
    assert PHC.period_reason(ledger, "p", []) == [
        f"period p's window {s} is incomplete (reserved and never opened): a refused window refuses its period",
        f"period p's window {t} is incomplete (Commit {COMMIT_B}, ended None, released None): a refused window refuses "
        "its period"]


def test_the_session_setting_is_the_gates_open_answer_for_the_window():
    """The window carries the gate's open answer (challenge phase, seq, root and coin commitment),
    without the hello's secret; any other form is refused."""
    reply = {"session": "s20261020T110845Z-1355", "seq": 8, "root": COMMIT_A, "commitment": COMMIT_B}
    full = {**reply, "host": "vy-challenger-1.s", "port": 7811, "peer": "vy-nebius-1", "by": "vy-admin", "for": "vllm",
            "reserve": 7, "opened_by": "vllm", "secret": "5e" * 32}
    assert PHC.session_of(full)["seq"] == 8 and PHC.carried(full) == {k: v for k, v in full.items() if k != "secret"}
    for bad in ({**reply, "session": ""}, {**reply, "seq": True}, {**reply, "seq": -1}, {**reply, "root": "ab"},
                {**reply, "root": COMMIT_A.upper()}, {**reply, "commitment": None}, {**reply, "secret": "ab"},
                {**reply, "secret": "5E" * 32}, {**reply, "period": "7"}, {k: v for k, v in reply.items() if k != "seq"},
                {"session": "s", "seq": 0, "period": "7", "commit": COMMIT_A}):
        with pytest.raises(ValueError, match="pouw setting 'session'"):
            PHC.session_of(bad)


def test_a_challenger_off_the_answers_commitment_is_refused_before_its_first_coin(tmp_path, monkeypatch):
    """The serving side's hello carries the answer's secret, and a challenger whose
    coin commitment isn't the one the gate's answer gives the challenge phase (another process's on that port) is closed and
    refused before its salt coin; the challenge phase it reached released nothing."""
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=1, seed=19)
    def other(session: str, root: str) -> dict:
        return {"session": session, "seq": 0, "root": root, "commitment": "dd" * 64, "secret": "5e" * 32}
    with pytest.raises(ValueError, match=f"commits to coins {host.commitment[:16]}..., not {'dd' * 8}..., the commitment "
                                         "the gate's answer gives session"):
        _serve(T._remote(session={"session": "auditor"}, gate=other))
    assert host.ended() is None and host.secrets == [bytes.fromhex("5e" * 32)]
    assert R.released_roots(host.directory / R.RELEASES, host.challenger.commitment) == {
        label: () for label, _n in host.challenger.commitment.families}


def test_infras_v2_head_and_sessions_bind_under_the_gates_key():
    """The gate's own bytes (the auditor host's v2 head and seven challenge phases as its fetch gave
    them), through core's check: the head verifies under the key its challenge phases' key.json holds and the pin, v1
    entries then v2, and binds a window by its root; challenge phase s20261020T110845Z-1355 ended closed and complete under the
    commitment its end entry records (`held`), s20261020T105232Z-1281 (a hello without the secret),
    s20261020T105206Z-9738 (before ends recorded their commitment) and s20261020T114752Z-9e52 (a salt after a root not
    its window's) are refused windows. Its periods: infra-smoke-20261020a and b closed complete, each with one window,
    whose hellos.jsonl its end seals; c's two-step windows bind their reserves, one complete and one killed (refused);
    d's one window was reserved and never opened (refused). Every challenge phase of the head predates the challenger's signed
    close, its challenger.json the gate's status and its release log's times floats, so core reads none of them
    as a challenge phase of record: 1355's end seals no close or refused hellos, a's period commits no cap or circuit, and b's
    and c's windows carry no signed close. Every open of the head also predates the host's pins log, so b's
    and c's windows, which core's ledger check passes, are refused for it first: with no pins log, and with one, since
    their opens seal none. One edited byte of the head, or a published head with an entry cut, is refused."""
    sid = "s20261020T110845Z-1355"
    root = ("34aaf35c44afbbeaf9919a1228b0405c67fa2218a25b6c578087719ba8e89b5587c7d3f3d705a490bac294499691f1af5b0450fc3c7b6c"
            "14c3d8c90584456a9a")
    commitment = ("d573a082258c1cbad6c03486976124d1626b82154c9c97d3add9e3fdbe6f15cd709add3e2968cb9059a1ee371e7ee0ff6a4677"
                  "e344321622df1c2f17200be868")
    assert (SMOKE / sid / "out" / R.KEY).is_file(), (
        f"{SMOKE} is not fetched: `uv run research data fetch-fixtures "
        "integrations/vllm/tests/protocols/fixtures/challenger_periods`")
    key = PHC.pinned_key(SMOKE / sid / "out" / R.KEY, SMOKE_PIN)
    assert (key.W, key.H) == (1, 20)
    head = R.ledger_lines(SMOKE / "ledger" / "windows.jsonl")
    ledger = PHC.Ledger.of([head[:4], head, head[:9], head[:23]], key)
    assert len(head) == 31 and len(ledger.windows) == 10 and list(ledger.lines) == head
    assert [json.loads(line)["schema"] for line in head] == [LC.LEDGER_SCHEMA_V1] * 4 + [LC.LEDGER_SCHEMA] * 27
    window = ledger.window(root)
    assert window["open"]["session"] == sid and window["open"]["seq"] == 8
    assert window["open"]["body"]["host"] == f"vy-challenger-1.{sid}" and window["end"]["body"]["complete"] is True
    fetch = lambda s: SMOKE / s if (SMOKE / s).is_dir() else None  # noqa: E731
    given = {"session": sid, "seq": 8, "root": root, "commitment": commitment}
    assert PHC.held(given, ledger, root) == (None, window)
    smoke = {"coin_source": "challenger", "session": given}
    assert PHC.session_reason(smoke, ledger, fetch) == (
        f"session {sid} is not trusted: the auditor's ledger's end of session {sid} is malformed: {LC.END_FORM}")
    unclosed = json.loads(head[6])
    assert PHC.held({**given, "session": unclosed["session"], "seq": 6, "root": LC.root_of(unclosed)}, ledger)[0] == (
        f"session {unclosed['session']} ended without its client closing it (a client's first frame is hello with the "
        "session's secret): a refused window")
    early = json.loads(head[4])
    assert PHC.held({**given, "session": early["session"], "seq": 4, "root": LC.root_of(early)}, ledger)[0].startswith(
        f"session {early['session']}'s end entry records coin commitment None..., not")

    def answer(seq):
        opened, end = json.loads(head[seq]), json.loads(head[seq + 1])
        return {"session": opened["session"], "seq": seq, "root": LC.root_of(opened),
                "commitment": end["body"]["commitment"]}
    salted = answer(10)
    assert PHC.held(salted, ledger)[0].startswith(
        f"session {salted['session']} ended without its client closing it (a pouw/salt coin opens only after the "
        f"window's Commit root {salted['root'][:16]}…")
    periods = {n: (p["windows"], p["close"]["body"]["complete"]) for n, p in ledger.periods.items()}
    a, b, c, d = (f"infra-smoke-20261020{x}" for x in "abcd")
    w4080, w306d, w6e92, wb89c = answer(13), answer(17), answer(25), answer(23)
    assert periods == {a: ([w4080["session"]], True), b: ([w306d["session"]], True),
                       c: ([w6e92["session"], wb89c["session"]], False), d: (["s20261020T123859Z-90aa"], False)}
    assert ledger.periods[b]["cap"] == 1 and ledger.periods[a]["cap"] is None
    pinned = PHC.Ledger.of([head], key, pins=[CH.PINS])
    for given, why in ((w4080, f"the auditor's ledger's period {a}, which holds session {w4080['session']}, commits no "
                                "cap or circuit"), (w306d, LC.NO_PINS), (wb89c, LC.NO_PINS)):
        held, w = PHC.held(given, ledger)
        assert held is None and "hellos.jsonl" in w["end"]["body"]["files"]
        with pytest.raises(LC.CoinError) as refused:
            PHC.evidence(ledger, given["root"], fetch)
        assert str(refused.value) == why
        if why == LC.NO_PINS:
            with pytest.raises(LC.CoinError, match=f"open of session {given['session']} seals no pins log"):
                PHC.evidence(pinned, given["root"], fetch)
    assert w4080["commitment"].startswith("43a09815f5424466")
    assert ledger.window(wb89c["root"])["reserve"]["seq"] == 22 and ledger.window(w4080["root"])["reserve"] is None
    assert PHC.period_reason(ledger, a, [w4080["session"]]) == [] and PHC.period_reason(ledger, b, [w306d["session"]]) == []
    assert PHC.period_reason(ledger, a, []) == [
        f"period {a}'s window {w4080['session']} (Commit {w4080['root']}) ended complete and has no accepted window: a "
        "window abandoned after its coins"]
    assert PHC.period_reason(ledger, a, [w306d["session"]])[0].startswith(
        f"session {w306d['session']}, an accepted window, is not one of period {a}'s windows")
    assert PHC.period_reason(ledger, c, [wb89c["session"]]) == [
        f"period {c}'s window {w6e92['session']} is incomplete (Commit {w6e92['root']}, ended 'serve.py exited -15', "
        "released {'pouw/round': 0, 'pouw/salt': 0}): a refused window refuses its period"]
    assert PHC.period_reason(ledger, d, []) == [
        f"period {d}'s window s20261020T123859Z-90aa is incomplete (reserved and never opened): a refused window refuses "
        "its period"]
    with pytest.raises(LC.CoinError, match="entry 9: its signature doesn't verify"):
        PHC.Ledger.of([head[:9] + [head[9].replace('"replay":2', '"replay":3')]], key)
    with pytest.raises(LC.CoinError, match="entry 30: its signature doesn't verify"):
        PHC.Ledger.of([head[:30] + [head[30].replace('"complete":false', '"complete":true')]], key)
    with pytest.raises(LC.CoinError, match="published head 1 is refused: .*entry 8: not entry 8 of a"):
        PHC.Ledger.of([head, head[:8] + [head[9]]], key)
    with pytest.raises(LC.CoinError, match="not the auditor's pinned"):
        PHC.pinned_key(SMOKE / sid / "out" / R.KEY, "00" * 64)


def test_an_early_draw_is_refused_and_the_windows_own_draw_fails_visibly(tmp_path, monkeypatch):
    """A draw coin opened after a root of the prover's, before the window's first round, is the challenger's one
    draw coin; the window's own draw at `finish` is then refused by the challenger (it has no second draw coin, and a
    retry never redraws), the window records the refusal, and the trusted check refuses the early draw."""
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=1, seed=13)
    probe = hashlib.sha512(b"a root of the prover's, before any round").digest()
    draw = host.client.open(Wd.DRAW, 0, probe)
    commitment = _serve(host.remote())
    nothing = f"the next {Wd.DRAW} coin is 1, not 0"
    assert nothing in host.ended()
    assert nothing in commitment["draw"]["refused"] and commitment["draw"]["coin"] is None
    received = R.released_roots(host.directory / R.RELEASES, host.challenger.commitment)
    assert received[Wd.DRAW] == (probe,) and draw.after == probe
    assert host.trusted(commitment) == (
        "the challenger's session is not trusted: the auditor's ledger ends session auditor unclosed: its client didn't "
        "close it")
    assert PHC.draw_reason(commitment, host.challenger.commitment, received[Wd.DRAW]) == (
        "the challenger's log does not show the window's draw root as the one root its draw coin followed: a draw opened "
        "before the window's rounds, or after another root")


def test_a_draw_from_another_family_off_its_root_or_never_opened_is_refused(tmp_path, monkeypatch):
    """The honest window's draw passes; its draw coin swapped for a round coin, a draw root not over the
    window's parts, and a challenge phase closed at `finish` with no draw opened are refused."""
    host = _Session(tmp_path, monkeypatch, SA, leaf=2, seed=14, reserved=True)
    honest = _serve(host.remote())
    assert host.ended() is None
    assert host.trusted(honest) is None
    assert honest["draw"]["root"] == RQ.draw_root(bytes.fromhex(honest["calls_root"]), bytes.fromhex(
        honest["draw"]["receipts"]), honest["rounds"]).hex()
    swapped = {**honest, "draw": {**honest["draw"], "coin": honest["rounds"][0]["coin"]}}
    assert host.trusted(swapped) == (
        f"the window's draw coin: coin is the epoch's {Wd.ROUND} coin 0, not its {Wd.DRAW} coin 0")
    bent = {**honest, "draw": {**honest["draw"], "receipts": "ee" * 64}}
    assert host.trusted(bent) == (
        "the window's draw root is not over its manifest, its last round's receipts and every round's root and coin")
    quiet = _Session(tmp_path, monkeypatch, SB, leaf=2, seed=15, reserved=True)
    opened = PS.window_open_root(bytes.fromhex(honest["weights_root"]), bytes.fromhex(honest["requests_root"]), SB)
    quiet.open(SB, opened.hex())
    salt = quiet.client.open(Wd.SALT, 0, opened)
    rounds = [quiet.client.open(Wd.ROUND, s, bytes.fromhex(r["root"])) for s, r in enumerate(honest["rounds"])]
    quiet.client.close()
    assert quiet.ended() is None
    c = quiet.challenger.commitment.to_json()
    closed = {**honest, "coins": c, "round_coins": c, "salt_opening": salt.to_json(),
              "session": {**honest["session"], "session": SB, "root": opened.hex()},
              "rounds": [{**r, "coin": o.to_json()} for r, o in zip(honest["rounds"], rounds)]}
    assert quiet.trusted(closed) == (
        "the challenger's session closed with no draw coin opened, so the window has no draw of the challenger's")


def test_round_roots_commit_the_windows_a_and_z_and_the_receipts_before_them(monkeypatch):
    """Round s's root, the one round s's coin follows, changes with the A the window serves (its A and z
    tables are in the step's manifest rows), is the same for the same A, and changes with round s - 1's receipts."""
    monkeypatch.setattr(PH.os, "urandom", lambda n: bytes(n))
    served = [_serve(T._settings(), seed) for seed in (9, 10, 9)]
    roots = [[r["root"] for r in c["rounds"]] for c in served]
    assert roots[0] == roots[2] and len(roots[0]) == 3
    assert roots[0][0] != roots[1][0] and roots[0][1] != roots[1][1] and roots[0][2] == roots[1][2]
    after = [[r["coin"]["after"] for r in c["rounds"]] for c in served]
    assert after == roots and after[0][:2] != after[1][:2]
    other_request = [r["root"] for r in _serve(T._settings(), 9, prompt=6)["rounds"]]
    assert other_request[0] != roots[0][0]

    def window(receipt):
        rng = np.random.default_rng(12)
        torch.manual_seed(12)
        run = PH.H1Run("t", T.SALT, {0: "q", 1: "o"}, [(32, T.K), (40, T.K)], 32, 2, rounds=T._rounds(2), key=T.KEY)
        run.record(0, *T._tap(rng, 3, 32))
        with pytest.raises(RuntimeError, match="call 0's receipt before round 0's coin"):
            run.receipt(0, bytes(64))
        run.record(1, *T._tap(rng, 3, 40))
        if receipt:
            run.receipt(0, b"\1" * 64)
            with pytest.raises(RuntimeError, match="or a second one"):
                run.receipt(0, b"\1" * 64)
            with pytest.raises(ValueError, match="64 bytes"):
                run.receipt(1, b"\1" * 32)
        run.record(0, *T._tap(rng, 3, 32))
        run.record(1, *T._tap(rng, 3, 40))
        with pytest.raises(RuntimeError, match="after the root that commits round 0's receipts"):
            run.receipt(1, b"\2" * 64)
        run.receipt(3, b"\3" * 64)
        run.finish()
        return run
    plain, signed = window(False), window(True)
    assert plain.round_log[0]["root"] == signed.round_log[0]["root"]
    assert plain.round_log[1]["root"] != signed.round_log[1]["root"]
    assert signed.round_log[1]["receipts"] == signed.receipts_of(0).hex() != RQ.NO_RECEIPTS.hex()
    assert signed.commits["draw"]["receipts"] == signed.receipts_of(1).hex()
    with pytest.raises(RuntimeError, match="after the root that commits round 1's receipts"):
        signed.receipt(2, b"\4" * 64)
    row = signed.manifest()[1]
    assert set(row["tables"]) == {"a", "z"} and row["tables"]["z"] is None
    assert signed.manifest()[-1]["tables"]["a"] != row["tables"]["a"]


def test_coins_held_in_the_serving_process_are_no_session_and_credit_nothing(tmp_path, monkeypatch):
    """The round rule alone passes in-process coins: a window whose coins
    were held in the serving process passes `rounds_reason`, but no challenge phase binds it and it credits nothing."""
    local, run = _serve(T._settings(), run=True)
    assert local["coin_source"] == "in-process"
    assert RQ.rounds_reason(local["rounds"], LiveCommitment.from_json(local["round_coins"])) is None
    assert PHC.session_reason(local, _gate().ledger(), {}.get).startswith(
        "the window's coins were held in the serving process")
    assert run.credit()["coins"] == "in-process" and run.credit()["gate_passed"] == {"rows": 0, "macs": 0}
    auditor = _Session(tmp_path, monkeypatch, "auditor", leaf=1, seed=14)
    auditor.client.close()
    assert auditor.ended() is None
    assert "coins from a challenger the verifier doesn't hold" in auditor.trusted(local)


@pytest.mark.slow
def test_pouw_serve_and_ncp_v2_take_the_auditor_hosts_challenger():
    """``pouw-serve``'s ``--challenger`` and ``--session`` are the pouw settings ncp-v2's salt comes from, one
    never without the other; ncp-v2 accepts the setting and ncp-v1 doesn't."""
    opts = pouw_serve.Options(workload="w.json", challenger=T.AUDITOR, session=json.dumps(T.SESSION))
    assert pouw_serve.protocols_config(opts) == {"pouw": {"challenger": T.AUDITOR, "session": T.SESSION, "draws": 16}}
    assert "coins" in pouw_serve.protocols_config(pouw_serve.Options(workload="w.json"))["pouw"]
    with pytest.raises(ValueError, match="--challenger needs --session"):
        pouw_serve.protocols_config(pouw_serve.Options(workload="w.json", challenger=T.AUDITOR))
    with pytest.raises(ValueError, match="--session is the session"):
        pouw_serve.protocols_config(pouw_serve.Options(workload="w.json", session=json.dumps(T.SESSION)))

    def refused(entry, settings):
        return T.PO.refusals(T.PO.Context(stage="serve", enabled=(entry,), settings=settings))
    remote = {"pouw": {"challenger": T.AUDITOR, "session": T.SESSION}}
    assert not any("challenger" in r or "session" in r for r in refused("pouw:ncp-v2", remote))
    assert any("needs 'session'" in r for r in refused("pouw:ncp-v2", {"pouw": {"challenger": T.AUDITOR}}))
    assert any("'challenger' is for eps1-sm120-h1fx1, ncp-v2," in r for r in refused("pouw:ncp-v1", remote))


class _NcpLinear:
    def __init__(self, weight):
        self.weight, self.quant_method = weight, SimpleNamespace(apply=lambda layer, x, bias=None: None)


def test_ncp_v2s_salt_is_the_auditor_hosts_salt_coin_after_the_weights_root(tmp_path, monkeypatch):
    """ncp-v2 under the setting ``challenger`` takes its epoch's salt from the auditor host's salt coin, opened
    after the weights' root, carries its challenge phase, and closes the challenge phase at release; pouw-serve parses the flags."""
    from verity.core.protocols.pouw.audit import salt_of

    from verity_vllm.pipeline import cli
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=1, seed=18)
    monkeypatch.setattr(pouw, "linear_types", lambda: (_NcpLinear,))
    w = (torch.randn(32, 64, generator=torch.Generator().manual_seed(3))).bfloat16().float()
    model = SimpleNamespace(modules=lambda: iter([layer]), named_modules=lambda: iter([("up", layer)]))
    layer = _NcpLinear(w)
    ctx = T.PO.Context(stage="serve", enabled=("pouw:ncp-v2",), settings={"pouw": {"challenger": T.AUDITOR,
                                                                                   "session": T.SESSION}})
    with T.PO.install(model, ctx) as comp:
        c = comp.get("pouw").commitment()
    assert host.ended() is None and c["coin_source"] == PH.CHALLENGER and c["session"] == T.SESSION
    received = R.released_roots(host.directory / R.RELEASES, host.challenger.commitment)
    weights = bytes.fromhex(c["weights_root"])
    assert received[Wd.SALT] == (weights,)
    opening = LC.LiveOpening.from_json(c["salt_opening"])
    assert c["salt"] == salt_of(host.challenger.commitment, opening.coin, weights).hex()
    parsed = cli.parse("pouw-serve", ["--workload", "w.json", "--challenger", T.AUDITOR, "--session", json.dumps(T.SESSION)])
    assert pouw_serve.protocols_config(parsed)["pouw"]["session"] == T.SESSION


def test_a_window_admits_in_round_0_only_unless_its_admission_is_per_round():
    """By default (the live tiles are fixed before the first coin) a request admitted after round 0 gets
    no entry, its rows are a gap and the window credits nothing; the verifier refuses an entry in a later round under
    ``round-0``, and accepts it under ``per-round``; an unknown admission is refused."""
    once, run = _serve(T._settings(admission=RQ.ROUND0), run=True)
    assert once["admission"] == RQ.ROUND0 and [e.round for e in run.requests] == [0]
    assert run.refused_requests == [{"round": 1, "reason": RQ.late_admission(1)}] * 2
    assert run.request_gaps() == ["step 1: 1 row of a request with no entry", "step 1: 31 rows of a request with no entry"]
    assert run.credit()["gate_passed"] == {"rows": 0, "macs": 0}
    _c, every = _serve(T._settings(), run=True)
    coins = every.rounds.challenger.commitment
    e = every.requests[1]
    args = (every.round_log, coins, every.salt, 64, e, every.open_request(1), every.open_first(1))
    assert e.round == 1 and RQ.request_reason(*args, admission=RQ.PER_ROUND) is None
    assert RQ.request_reason(*args) == f"request 1 was {RQ.late_admission(1)}"
    owner = every.open_slot(1, 1)
    assert RQ.slot_reason(every.round_log, coins, every.salt, 64, [m for m in every.manifest() if m["step"] == 1], owner,
                          e, every.open_request(1), every.open_first(1)) == f"request 1 was {RQ.late_admission(1)}"
    with pytest.raises(ValueError, match="pouw setting 'admission'"):
        pouw.Settings.of({"admission": "late"})
    with pytest.raises(ValueError, match="an admission is one of"):
        PH.H1Run("t", T.SALT, {0: "q"}, [(32, T.K)], 32, 1, admission="late")


def test_the_order_of_roots_and_coins_is_signed_by_the_challengers_close_and_a_rewrite_of_it_is_refused(tmp_path, monkeypatch):
    """Roots chosen after every coin, with a draw opened after a root over them, are refused against the
    honest log, and so is a log rewritten to name them, which the challenger's signed close no longer signs."""
    host = _Session(tmp_path, monkeypatch, SA, leaf=2, seed=15, reserved=True)
    weights, requests = hashlib.sha512(b"the weights' root").digest(), hashlib.sha512(b"round 0's requests").digest()
    opened = PS.window_open_root(weights, requests, SA).hex()
    session = host.open(SA, opened)
    salt = host.client.open(Wd.SALT, 0, bytes.fromhex(opened))
    early = [hashlib.sha512(b"before any round, %d" % s).digest() for s in range(3)]
    coins = [host.client.open(Wd.ROUND, s, early[s]) for s in range(3)]
    late = [hashlib.sha512(b"a root chosen after every coin, %d" % s + c.coin).digest() for s, c in enumerate(coins)]
    log = [{"round": s, "root": late[s].hex(), "coin": {**coins[s].to_json(), "after": late[s].hex()}} for s in range(3)]
    log[0]["requests"] = requests.hex()
    calls_root = hashlib.sha512(b"the manifest").digest()
    draw_root = RQ.draw_root(calls_root, RQ.NO_RECEIPTS, log)
    draw = host.client.open(Wd.DRAW, 0, draw_root)
    host.client.close()
    assert host.ended() is None
    c = host.challenger.commitment.to_json()
    commits = {"coins": c, "round_coins": c, "weights_root": weights.hex(), "requests_root": requests.hex(),
               "session": session, "salt_opening": salt.to_json(), "steps": 3,
               "rounds": log, "calls_root": calls_root.hex(),
               "draw": {"root": draw_root.hex(), "receipts": RQ.NO_RECEIPTS.hex(), "coin": draw.to_json()}}
    assert host.trusted(commits) == (
        "the window's rounds: the round roots are not the roots the challenger's log shows it received, in order")
    path = host.directory / R.RELEASES
    text = path.read_text()
    for e, x in zip(early, late):
        text = text.replace(e.hex(), x.hex())
    path.write_text(text)
    assert host.trusted(commits) == (
        "the challenger's session is not trusted: the challenger's release log is not the one its close record signs")


def test_the_challengers_record_is_tied_to_its_session_by_the_gates_host_string_and_end_entry(tmp_path, monkeypatch):
    """The challenger's signed record names its host and coin commitment, not the challenge phase or Commit; the
    gate's host string ``vy-challenger-1.<session>`` ties it to its challenge phase's open entry, and the v2 end entry records
    its commitment's root, the leaf it was signed at (between the open's and the end's) and the coins each family
    released (none here, so not complete)."""
    gate = _gate()
    reply = gate.open("s-a", COMMIT_A, Wd.live_families(3))
    a = _Session(tmp_path, monkeypatch, "s-a", leaf=gate.take(), seed=11, close=gate.take)
    a.client.close()
    assert a.ended() is None
    record = json.loads((a.directory / R.RECORD).read_text())
    assert sorted(record) == ["commitment", "host", "signature"] and record["host"] == reply["host"]
    c = LiveCommitment.from_json(record["commitment"])
    assert PHC.lms_verify(KEY, record_message(KEY.public[:16], record["host"], c), bytes.fromhex(record["signature"]))
    end = gate.end("s-a", a.directory)["body"]
    assert reply["commitment"] is None and end["commitment"] == c.root.hex() == a.commitment
    assert end["record_leaf"] == LC.leaf_of(record["signature"]) == 1 and end["complete"] is False
    assert end["released"] == {label: 0 for label, _n in c.families}
    assert gate.ledger().window(COMMIT_A)["end"]["body"] == end
