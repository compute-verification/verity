"""The Commit's coins from the auditor host's challenger: one challenge phase a Commit, which draws its openings
and its served C2 replay challenge after each run's root of record.

* **Serving** (the Commit's ``--challenger-gate CMD`` and ``--challenger-peer NAME``; `of_options`): before its engine,
  the Commit reserves a challenge phase on the auditor's challenger gate (`Gate`: ``reserve --peer NAME --scheme SCHEME --families
  families(runs)``, `benchmarks/live_challenger/auditor.py` on the auditor host, under the caller's own gate key). Once its
  first instrumented run's root of record is fixed (rows-sha512's 64-byte root, `commit.step_rows`; a TP Commit's run
  root v2, `tp.run_root`), it opens that challenge phase on it (``open --session S --root R``): the gate signs the open entry
  binding the challenge phase to the root before the challenger serves a coin, and refuses a second challenge phase for the root, ever.
  The Commit then reaches the challenger on its own 127.0.0.1 (this machine's certifier coin link), refuses one whose coin
  commitment isn't the gate's answer before its first coin (`pouw_h1_window.connect`), and for run i opens coin i of
  each family in `FAMILIES` after run i's root (`Session.draw`); it closes the challenge phase after its last run. The draws
  derive from the coins (`verity.ml.randomness.derive`): opening positions (`positions`) and population identities
  (`identities`) from `POSITIONS` and `IDENTITIES`, and the served C2 replay's challenge (`replay_challenge`) is the
  `REPLAY` coin.
* **Without a gate** each run's coins are this process's own bytes (``source`` ``in-process``): the run is a measurement,
  and its coins credit nothing, as H1's in-process coins (`pouw_h1_challenger.session_reason`). ``--replay-challenge``
  still names the C2 replay's challenge there (``source`` ``flag``).
* **Checking** (`reason`): the verifier reads the auditor's signed ledger as published (every head,
  `challenger.ledger_head`), the one challenge phase it opens for the Commit's root, run from a commit the host's pins log
  records on the verifier's own first-parent history of main (`challenger.pinned_session`, the log's heads read by
  `pouw_h1_challenger.pins_head`, main's history by `remote.main_lines`), and that challenge phase's files from the gate's ``fetch`` (never the prover's bundle) under the key the deployment pins
  (`remote.trusted_session`): the coin commitment the challenger signed and the roots its signed release log shows it
  received. The Commit's challenge phase must be that challenge phase, its coins that commitment's, and the log must show each run's
  coin of each family released after that run's root of record: coins held in the Commit's process, a challenge phase opened on
  another root, a second challenge phase for the Commit's root, a coin drawn before its root and a ledger cut before the
  challenge phase's end are refused.
"""
from __future__ import annotations

import json
import secrets
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import CoinError, LiveOpening, live_reason

from verity_vllm.protocols.pouw.h1 import challenger as PHC
from verity_vllm.protocols.pouw.h1 import window as PHW

SCHEME = "verity/vllm/commit-coins/v1"
POSITIONS, IDENTITIES, REPLAY = "opening-positions", "opening-identities", "replay"
FAMILIES = (POSITIONS, IDENTITIES, REPLAY)
CHALLENGER, IN_PROCESS = "challenger", "in-process"
ROOT_BYTES = remote.ROOT_BYTES


def families(runs: int) -> list[list[Any]]:
    """The challenge phase's layout for a Commit of `runs` instrumented runs: a coin of each family a run."""
    return [[f, int(runs)] for f in FAMILIES]


#: ``run(argv)``: the gate's (exit code, stdout) for one request
Run = Callable[[list[str]], "tuple[int, str]"]


class Gate:
    """Infra's challenger gate over `command` (its argv prefix as one string, e.g. ``ssh -i KEY -o IdentitiesOnly=yes
    research@HOST``): each request goes as one shell-quoted word, which the gate's forced command splits. `run` replaces
    the subprocess (tests)."""

    def __init__(self, command: str, run: Run | None = None) -> None:
        self.command, self._run = shlex.split(command), run

    def call(self, *words: str) -> dict[str, Any]:
        """The gate's JSON answer to `words`; `CoinError` naming a refusal (exit 3, ``{"refused"}``) or a broken answer."""
        rc, text = self._answer(words)
        try:
            doc = json.loads(text)
        except ValueError:
            doc = None
        if rc != 0 or not isinstance(doc, dict) or "refused" in doc:
            why = doc.get("refused") if isinstance(doc, dict) else (text or "").strip()[:300]
            raise CoinError(f"the challenger gate refused {words[0]} (exit {rc}): {why}")
        return doc

    def ledger(self) -> list[str]:
        """The auditor's ledger as the gate's ``ledger`` answers it, one entry a line; `CoinError` when the gate refuses."""
        rc, text = self._answer(("ledger",))
        if rc != 0:
            raise CoinError(f"the challenger gate refused ledger (exit {rc}): {(text or '').strip()[:300]}")
        return text.splitlines()

    def _answer(self, words: Sequence[str]) -> tuple[int, str]:
        argv = [*self.command, shlex.join(words)]
        if self._run is not None:
            return self._run(argv)
        p = subprocess.run(argv, capture_output=True, text=True, check=False)
        return p.returncode, p.stdout


class Session:
    """The Commit's challenge phase on the auditor host's challenger: reserved for `runs` runs (`reserve`), opened on the first
    run's root of record (`draw` of run 0), a coin of each family a run, closed after the last (`close`)."""

    def __init__(self, gate: Gate, reserved: Mapping[str, Any], runs: int, address: Callable[[int], str] | None = None) -> None:
        self.gate, self.reserved, self.runs = gate, dict(reserved), int(runs)
        self._address = address or (lambda port: f"127.0.0.1:{port}")
        self.given: dict[str, Any] | None = None
        self.client: Any = None
        self.index = 0

    @classmethod
    def reserve(cls, gate: Gate, peer: str, runs: int, **kw: Any) -> Session:
        if runs < 1:
            raise SystemExit("--challenger-gate draws the coins of the Commit's instrumented runs: not without one")
        return cls(gate, gate.call("reserve", "--peer", peer, "--scheme", SCHEME, "--families", json.dumps(families(runs))),
                   runs, **kw)

    def _open(self, root: bytes) -> None:
        answer = self.gate.call("open", "--session", self.reserved["session"], "--root", root.hex())
        if answer.get("commitment") is None:
            answer["commitment"] = self.gate.call("status", self.reserved["session"]).get("commitment")
        try:
            self.given = PHC.session_of({k: v for k, v in answer.items() if k in (*PHC.SESSION_FIELDS, *PHC.REPLY_FIELDS)})
        except ValueError as e:
            raise CoinError(f"the challenger gate's open answer: {e}") from e
        self.client = PHW.connect(self._address(int(answer["port"])), self.given)
        c = self.client.commitment
        if c.scheme != SCHEME or [list(f) for f in c.families] != families(self.runs):
            self.close()
            raise CoinError(f"the challenger's coins are {c.scheme} {list(c.families)}, not the Commit's {SCHEME} "
                            f"{families(self.runs)}")

    def draw(self, root: bytes) -> dict[str, Any]:
        """The next run's coins, coin i of each family opened after its root of record `root` (the challenge phase opened on run
        0's), and its record (`record_of`)."""
        root = bytes(root)
        if len(root) != ROOT_BYTES:
            raise CoinError(f"the challenger's coins follow a {ROOT_BYTES}-byte root of record (rows-sha512, the TP run root "
                            f"v2), not this run's {len(root)}-byte root (vllm-v1)")
        if self.index >= self.runs:
            raise CoinError(f"the Commit's session holds coins for {self.runs} runs, and each is drawn")
        if self.client is None:
            self._open(root)
        i, self.index = self.index, self.index + 1
        openings = {f: self.client.open(f, i, root) for f in FAMILIES}
        return {"source": CHALLENGER, "index": i, "root": root.hex(), "session": PHC.carried(self.given),
                "commitment": self.client.commitment.to_json(), "keys": {f: o.coin.hex() for f, o in openings.items()},
                "openings": {f: o.to_json() for f, o in openings.items()}}

    def close(self) -> str | None:
        """Close the challenge phase (the gate's end entry then seals it); why it didn't close cleanly, or None."""
        if self.client is None:
            return None
        client, self.client = self.client, None
        try:
            client.close()
        except CoinError as e:
            return str(e)
        return None


def of_options(args: Any, runs: int, gate_run: Run | None = None) -> Session | None:
    """The Commit's challenge phase from its ``--challenger-gate`` and ``--challenger-peer`` (reserved now, before any root), or
    None without a gate; `SystemExit` when they don't go together."""
    gate, peer = getattr(args, "challenger_gate", None), getattr(args, "challenger_peer", None)
    if bool(gate) != bool(peer):
        raise SystemExit("--challenger-gate and --challenger-peer go together: the gate's command and this machine's name in "
                         "the auditor host's peers, where its challenger reaches the Commit")
    if not gate:
        return None
    if getattr(args, "replay_challenge", None):
        raise SystemExit("--challenger-gate and --replay-challenge are two sources of the C2 replay's challenge: give one")
    try:
        return Session.reserve(Gate(gate, gate_run), peer, runs)
    except CoinError as e:
        raise SystemExit(f"--challenger-gate: {e}") from e


def draw(session: Session | None, rc: Any) -> dict[str, Any]:
    """Run `rc`'s coins after its root (``rc.extra["coins"]``): `session`'s, or this process's own bytes without one."""
    if session is not None:
        rec = session.draw(bytes(rc.run_root))
    else:
        rec = {"source": IN_PROCESS, "root": bytes(rc.run_root).hex(), "keys": {f: secrets.token_hex(64) for f in FAMILIES}}
    rc.extra["coins"] = rec
    return rec


def _key(rc: Any, family: str) -> bytes:
    rec = (getattr(rc, "extra", None) or {}).get("coins") or {}
    if rec.get("root") != bytes(rc.run_root).hex():
        raise ValueError(f"run root {bytes(rc.run_root).hex()[:16]}: no coins drawn after it (commit_coins.draw)")
    return bytes.fromhex(rec["keys"][family])


_DRAWN = object()


def positions(rc: Any, session: Any = _DRAWN) -> dict[str, Any]:
    """`challenge.challenge_positions`' keywords for run `rc`: off legacy, from its `POSITIONS` coin; given `session`
    (None without a gate), the run's coins are drawn first (`draw`)."""
    if session is not _DRAWN:
        draw(session, rc)
    return {"legacy": False, "source": _key(rc, POSITIONS)}


def identities(rc: Any) -> dict[str, Any]:
    """`binding.challenge_identities`' keywords for run `rc`: off legacy, from its `IDENTITIES` coin."""
    return {"legacy": False, "source": _key(rc, IDENTITIES)}


def replay_challenge(rc: Any, given: str | None) -> tuple[str, dict[str, Any]]:
    """The served C2 replay's challenge and its record: run `rc`'s `REPLAY` coin, or ``--replay-challenge`` `given` (a
    measurement) when the run's coins are this process's own."""
    rec = rc.extra["coins"]
    if rec["source"] == IN_PROCESS and given:
        return given, {"source": "flag"}
    return _key(rc, REPLAY).hex(), {"source": rec["source"], "family": REPLAY, "index": rec.get("index")}


def tp_draw(session: Session | None, root: bytes) -> dict[str, Any]:
    """A TP Commit's coins for one pair after its TP run root `root` (v2: SHA-512 over the ranks' rows-sha512 roots)."""
    from types import SimpleNamespace

    rc = SimpleNamespace(run_root=bytes(root), extra={})
    return draw(session, rc)


def tp_keys(rec: Mapping[str, Any]) -> tuple[bytes, bytes]:
    """The (positions, identities) sources of a TP pair's coins record (`tp_draw`)."""
    return bytes.fromhex(rec["keys"][POSITIONS]), bytes.fromhex(rec["keys"][IDENTITIES])


def reason(runs: Sequence[Mapping[str, Any]], roots: Sequence[bytes], heads: Sequence[Any], directory: Any,
           key: Ch.ChallengerKey, verify: Ch.Verify = PHC.lms_verify, *, pins: Sequence[Any] | None,
           main: Sequence[str] | None) -> str | None:
    """Why the Commit's coins (`runs`, each instrumented run's ``coins`` record, in order) aren't the auditor host's
    challenger's, drawn after `roots` (each run's root of record as the verifier holds it); None when they are. The
    challenge phase is read from `directory` (its ``out/`` from the gate's ``fetch``) and the ledger from `heads` (every head
    published, each a ``windows.jsonl``), under `key`, the challenger key the deployment pins, the host's pins log
    from `pins` (every head published, each a ``pins.jsonl``) and `main`, the verifier's own first-parent history
    of main (``git rev-list --first-parent origin/main``), which must hold the challenge phase's pin. Fails closed."""
    if not runs or len(runs) != len(roots):
        return f"the Commit's coins name {len(runs)} runs' draws, not one for each of its {len(roots)} runs of record"
    if any(r.get("source") != CHALLENGER for r in runs):
        return ("the Commit's coins were drawn in its own process, not by the auditor host's challenger: a measurement, "
                "which credits nothing")
    roots = [bytes(r) for r in roots]
    if any(len(r) != ROOT_BYTES for r in roots):
        return f"a run's root of record is not {ROOT_BYTES} bytes (rows-sha512 or the TP run root v2)"
    window = {"root": roots[0].hex()}
    try:
        lines = Ch.ledger_head([remote.ledger_lines(p) for p in heads], key, verify)
        audited = Ch.pinned_session(lines, window, key, PHC.pins_head(pins), None if main is None else list(main),
                                    verify)
        trusted, received = remote.trusted_session(directory, heads, window, key, verify)
    except CoinError as e:
        return f"the Commit's challenger session is not trusted: {e}"
    for i, run in enumerate(runs):
        given = run.get("session") or {}
        if given.get("session") != audited.session or given.get("root") != window["root"]:
            return (f"run {i}'s coins are session {given.get('session')}'s, opened for root {str(given.get('root'))[:16]}, "
                    f"not {audited.session}, the one the auditor's ledger opened for the Commit's root {window['root'][:16]}")
        if given.get("commitment") != trusted.root.hex() or run.get("index") != i:
            return f"run {i}'s coins are not coin {i} of the commitment the auditor's challenger signed"
    if trusted.scheme != SCHEME or [list(f) for f in trusted.families] != families(len(roots)):
        return f"the challenger's coins are {trusted.scheme} {list(trusted.families)}, not the Commit's {families(len(roots))}"
    for f in FAMILIES:
        if tuple(received.get(f, ())) != tuple(roots):
            return (f"the challenger's log shows its {f} coins released after roots other than the runs' roots of record, "
                    "in order: a coin drawn before its run's root, or after another")
    for i, (run, root) in enumerate(zip(runs, roots)):
        for f in FAMILIES:
            try:
                opening = LiveOpening.from_json((run.get("openings") or {}).get(f))
                why = live_reason(trusted, opening, f, i, root)
            except CoinError as e:
                why = str(e)
            if why is not None:
                return f"run {i}'s {f} coin: {why}"
            if (run.get("keys") or {}).get(f) != opening.coin.hex():
                return f"run {i}'s {f} key is not the coin the challenger opened"
    return None


__all__ = ["CHALLENGER", "FAMILIES", "Gate", "IDENTITIES", "IN_PROCESS", "POSITIONS", "REPLAY", "ROOT_BYTES", "SCHEME",
           "Session", "draw", "families", "identities", "of_options", "positions", "reason", "replay_challenge", "tp_draw",
           "tp_keys"]
