"""Where a session's coins come from (`verifier.py --challenger`), each an epoch of the proof system's one challenger
(`coins.SessionCoins`, `sampled_proofs.challenger`):

- `Local`: a `LiveChallenger` in the verifier's process, drawing from `secrets.token_bytes`.
- `Serve`: the challenger's own process, `benchmarks/live_challenger/serve.py`, started for the session on the
  verifier's machine with its LMS key (`--challenger-key`), reached over its 127.0.0.1 by `sampled_proofs.remote`'s
  client. It draws its coins in the Lean (`one_stage.verifier.coins`), signs its record before it listens and its
  release log when the client closes the session. Not the auditor host: the verifier's machine is the audit's (never
  the prover's), but no ledger opens the session for its window, so the read is `challenger.session_of`'s, under the
  key the run wrote.
- `ChallengerGate`: the auditor's challenger host, through its gate (`benchmarks/live_challenger/auditor.py`)
  under the lane's own gate key, over ssh (`ssh_gate`). Each session is reserved for this machine (`--peer`), inside
  the audit's period, and opened at its window root R (`coins.window`; the ledger opens one session a root, ever); the
  host serves it from its pinned tree and reaches this machine's 127.0.0.1 at the open's port through this machine's
  certifier coin link (the auditor's coins' path); the
  hello carries the open's secret, and a challenger whose coin commitment isn't the open's is refused before its first
  coin. The read is the ledger's (`audit.AuditRun.trusted_reason`): the one session it opens for R inside the period,
  ended and closed, its files the ones the end sealed, under the key whose public key's SHA-512 the run pins, and run
  from main's code (`challenger.pinned_session`): the host's pins log, fetched from the gate (`pins`) at the session's
  end beside the ledger, is the one its open sealed and records its pin, and that commit is on the auditor's own
  first-parent list of main (`--challenger-main FILE`). With no list, no gate verifier runs.

`finish` reads back what the challenger signed and holds the session's coins to it (`audit.AuditRun.released_reason`):
the same commitment, each coin opened after the root the session recorded and in order, σ_0 after the session's window
root R, each window's openings coin after its last root. Its `reason` is None when they hold. A served or gate
session's `complete` (the gate's end's, and `Serve`'s read of the same signed log) is whether the challenger released
every coin it committed to: a session reserves only the families it uses (`coins.families`), so an honest session
that ran to its end is complete and one cut short is not. A gate session's record names the `peer` it reserved for
(`--challenger-peer`), which `auditor_line.py` holds the ledger's reserve and open to."""
from __future__ import annotations

import hashlib
import io
import json
import secrets
import shlex
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable, Sequence

from verity.core.protocols.remote_bandwidth import audit, coins
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote
from verity.ml.crypto import lms

ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / "benchmarks" / "live_challenger" / "serve.py"
KINDS = ("local", "serve", "gate")
GATE_PORTS = range(7811, 7900)
"""The ports the gate gives its sessions (`auditor.PORTS`), each bound on this machine's 127.0.0.1 by its certifier
coin link to the host: the verifier's own port is none of them."""


def lms_verify(key: Ch.ChallengerKey, message: bytes, signature: bytes) -> bool:
    """`challenger.Verify`: LMS over SHA-512 at the key's (64, W, H)."""
    p = lms.Params(64, key.W, key.H)
    try:
        return lms.verify(p, key.public, message, lms.Signature.decode(p, signature))
    except ValueError:
        return False


def session_id(registration: bytes, k: int, run: str) -> str:
    """The challenger session of session k over the registration whose digest is `registration`, in the run tagged
    `run` (the key's leaf state refuses a session id it has started before)."""
    return f"remote-bandwidth-{registration.hex()[:16]}-{k}-{run}"


class Local:
    """Session coins from a challenger in this process."""

    kind = "local"

    def open(self, registration: bytes, k: int, windows: int, appends: int, per_coin: int = 1) -> Ch.LiveChallenger:
        return Ch.LiveChallenger(coins.SCHEME, k, coins.families(windows, appends, per_coin), secrets.token_bytes)

    def finish(self, s: audit.AuditRun) -> dict:
        c = s.coins
        return {"challenger": self.kind, "commitment": c.commitment.root.hex(),
                "reason": s.released_reason(c.commitment, {label: c.afters(label) for label, _ in c.commitment.families})}


class Serve:
    """Session coins from `serve.py`, one process a session, its `out/` under `out`/SESSION, its key in `key`
    (made there on first use at `params`, `W,H`, or `lms.DEPLOYED`'s)."""

    kind = "serve"

    def __init__(self, out: Path, key: Path, params: str | None = None, python: str = sys.executable) -> None:
        self.out, self.key, self.params, self.python = out, key, params, python
        self.run = secrets.token_hex(4)
        self._live: dict[int, tuple[subprocess.Popen, Path, str, remote.RemoteChallenger]] = {}

    def open(self, registration: bytes, k: int, windows: int, appends: int,
             per_coin: int = 1) -> remote.RemoteChallenger:
        sid = session_id(registration, k, self.run)
        out = self.out / sid
        families = json.dumps([list(f) for f in coins.families(windows, appends, per_coin)])
        with open(self.out / f"{sid}.log", "w") as log:
            proc = subprocess.Popen(
                [self.python, str(SERVE), "--out", str(out), "--port", "0", "--scheme", coins.SCHEME,
                 "--families", families, "--session", sid,
                 "--epoch", str(k), "--key", str(self.key), "--host", socket.gethostname(),
                 *(["--params", self.params] if self.params else [])],
                stdout=subprocess.PIPE, stderr=log, text=True, cwd=ROOT)
        line = proc.stdout.readline().split()
        if line[:1] != ["listening"] or len(line) != 2:
            proc.wait()
            raise RuntimeError(f"serve.py for {sid} did not listen (exit {proc.returncode}; {self.out / sid}.log)")
        try:
            client = remote.RemoteChallenger.connect(line[1])
        except (Ch.CoinError, OSError):
            proc.kill()
            proc.wait()
            raise
        self._live[k] = (proc, out, sid, client)
        return client

    def finish(self, s: audit.AuditRun) -> dict:
        proc, out, sid, client = self._live.pop(s.coins.session)
        try:
            client.close()
            closed = None
        except Ch.CoinError as e:
            closed = str(e)
        rc = proc.wait()
        doc = {"challenger": self.kind, "session": sid, "dir": str(out), "serve_rc": rc, "closed": closed,
               "commitment": s.coins.commitment.root.hex()}
        try:
            key = Ch.ChallengerKey.from_json(json.loads((out / remote.KEY).read_text()))
            files = remote.session_files(out)
            signed, received, _ = Ch.session_of(key, files["record"], files["log"], files["close"], lms_verify)
        except (Ch.CoinError, OSError, ValueError) as e:
            return {**doc, "reason": f"the challenger's session is not readable or not signed: {e}"}
        released = {label: len(received.get(label, ())) for label, _ in signed.families}
        return {**doc, "key_sha512": hashlib.sha512(key.public).hexdigest(), "released": released,
                "complete": all(released[label] == n for label, n in signed.families),
                "reason": s.released_reason(signed, received)}


Call = Callable[[list[str]], tuple[int, bytes]]
"""One gate request: its words, to the gate's (exit code, stdout); a refusal is exit 3 with {"refused": why}."""


def ssh_gate(ssh: Sequence[str]) -> Call:
    """Gate requests over ssh: `ssh` is ssh's options and destination (the lane's gate key with `-i`, and the host's
    line of /etc/vy/peers); the host's authorized_keys forces the key's command, the challenger's gate with LANE, which
    reads the request's words from ssh's original command."""
    def call(words: list[str]) -> tuple[int, bytes]:
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", *ssh, shlex.join(words)],
                           capture_output=True, stdin=subprocess.DEVNULL)
        return p.returncode, p.stdout if p.returncode != 255 else p.stderr
    return call


class ChallengerGate:
    """Session coins from the auditor host's challenger, one gate session a session, its fetched directory, the ledger
    and the host's pins log under `out`. `pin` is the SHA-512 of the challenger's public key, `period` the audit's
    period on the ledger (the lane opens it, `period-open`, before the first session), `main` the auditor's own
    first-parent list of main (`remote.main_lines`; None: every session is refused), `minutes` each reserve's (the
    most a session serves). `address` is where the open's session listens on this machine, `pause` runs between two
    looks for what the host hasn't done yet (a tunnel, an end entry), at most `polls` times."""

    kind = "gate"

    def __init__(self, out: Path, call: Call, peer: str, pin: str, period: str, main: list[str] | None, *,
                 minutes: int = 120, address: Callable[[dict], str] = lambda answer: f"127.0.0.1:{answer['port']}",
                 pause: Callable[[], None] = lambda: time.sleep(1), polls: int = 600) -> None:
        self.out, self.call, self.peer, self.pin, self.period, self.minutes = out, call, peer, pin, period, minutes
        self.main, self.address, self.pause, self.polls = main, address, pause, polls
        self._key: Ch.ChallengerKey | None = None
        self._live: dict[int, tuple[dict, remote.RemoteChallenger]] = {}

    def _ask(self, words: list[str]) -> bytes:
        rc, got = self.call(words)
        if rc != 0:
            raise Ch.CoinError(f"the challenger gate answered {words[0]} with exit {rc}: "
                               f"{got.decode(errors='replace').strip()}")
        return got

    def key(self) -> Ch.ChallengerKey:
        """The host's challenger key (`pubkey`), refused unless its public key's SHA-512 is the pin."""
        if self._key is None:
            doc = json.loads(self._ask(["pubkey"]))
            key = Ch.ChallengerKey(doc["W"], doc["H"], bytes.fromhex(doc["public"]))
            got = hashlib.sha512(key.public).hexdigest()
            if got != self.pin:
                raise Ch.CoinError(f"the gate's challenger key is {got[:16]}…, not the pinned {self.pin[:16]}…")
            self._key = key
        return self._key

    def open(self, registration: bytes, k: int, windows: int, appends: int,
             per_coin: int = 1) -> remote.RemoteChallenger:
        self.key()
        families = json.dumps([list(f) for f in coins.families(windows, appends, per_coin)])
        r = json.loads(self._ask(["reserve", "--peer", self.peer, "--scheme", coins.SCHEME, "--families", families,
                                  "--epoch", str(k), "--minutes", str(self.minutes)]))
        if r.get("period") != self.period:
            raise Ch.CoinError(f"the gate reserved session {r['session']} in period {r.get('period')}, not the audit's "
                               f"period {self.period}: it expires unopened")
        o = json.loads(self._ask(["open", "--session", r["session"], "--root", coins.window(registration, k)["root"]]))
        commitment = o["commitment"] or json.loads(self._ask(["status", r["session"]]))["commitment"]
        client = None
        for _ in range(self.polls):
            try:
                client = remote.RemoteChallenger.connect(self.address(o), bytes.fromhex(o["secret"]))
                break
            except ConnectionRefusedError:
                self.pause()
        if client is None:
            raise Ch.CoinError(f"the challenger of session {r['session']} never listened at {self.address(o)}")
        if client.commitment.root.hex() != commitment:
            client.close()
            raise Ch.CoinError(f"the challenger at {self.address(o)} commits to coins "
                               f"{client.commitment.root.hex()[:16]}…, not {str(commitment)[:16]}…, session "
                               f"{r['session']}'s: refused before its first coin")
        self._live[k] = ({key: v for key, v in o.items() if key != "secret"}, client)
        return client

    def finish(self, s: audit.AuditRun) -> dict:
        answer, client = self._live.pop(s.coins.session)
        sid = answer["session"]
        try:
            client.close()
            closed = None
        except Ch.CoinError as e:
            closed = str(e)
        doc = {"challenger": self.kind, "session": sid, "peer": self.peer, "open": answer, "period": self.period,
               "closed": closed, "commitment": s.coins.commitment.root.hex(), "key_sha512": self.pin}
        try:
            end = None
            for _ in range(self.polls):
                if (end := json.loads(self._ask(["status", sid]))["end"]) is not None:
                    break
                self.pause()
            if end is None:
                return {**doc, "reason": f"the auditor's ledger has no end for session {sid}"}
            with tarfile.open(fileobj=io.BytesIO(self._ask(["fetch", sid])), mode="r:gz") as tar:
                if any(n != sid and not n.startswith(f"{sid}/") for n in tar.getnames()):
                    raise ValueError(f"the gate's fetch of {sid} holds files outside it")
                tar.extractall(self.out, filter="data")
            ledger = self.out / f"{sid}.ledger.jsonl"
            ledger.write_bytes(self._ask(["ledger"]))
            pins = self.out / f"{sid}.pins.jsonl"
            pins.write_bytes(self._ask(["pins"]))
            lines = remote.ledger_lines(pins)
        except (Ch.CoinError, OSError, ValueError, KeyError, tarfile.TarError) as e:
            return {**doc, "reason": f"the auditor's session is not readable: {e}"}
        return {**doc, "end": {f: end["body"].get(f) for f in ("rc", "closed", "released", "complete", "ended")},
                "reason": s.trusted_reason(self.out / sid / "out", [ledger], self.key(), lines, self.main, lms_verify,
                                           self.period)}


def of(kind: str, out: Path, key: Path | None, params: str | None, *, ssh: str | None = None, peer: str | None = None,
       pin: str | None = None, period: str | None = None, main: Path | None = None,
       minutes: int = 120) -> Local | Serve | ChallengerGate:
    """`--challenger KIND`'s source; `serve` needs its key directory, `gate` the host's ssh, this machine's peer name
    there, the challenger key's pin, the audit's period and the auditor's first-parent list of main, read here so a
    missing or unreadable one refuses the run before its first session."""
    if kind == "local":
        return Local()
    if kind == "serve":
        if key is None:
            raise ValueError("--challenger serve needs --challenger-key DIR, the challenger's LMS key directory, "
                             "outside the run's out/")
        out.mkdir(parents=True, exist_ok=True)
        return Serve(out, key, params)
    if kind == "gate":
        if ssh is None or peer is None or pin is None or period is None or main is None:
            raise ValueError("--challenger gate needs --challenger-ssh WORDS (ssh's -i KEY and the host), "
                             "--challenger-peer NAME (this machine in the host's /etc/vy/peers), --challenger-pin "
                             "SHA512 (the challenger's public key), --challenger-period P (the lane's open period) and "
                             "--challenger-main FILE (the auditor's own `git rev-list --first-parent origin/main`)")
        first_parent = remote.main_lines(main)
        out.mkdir(parents=True, exist_ok=True)
        return ChallengerGate(out, ssh_gate(shlex.split(ssh)), peer, pin, period, first_parent, minutes=minutes)
    raise ValueError(f"--challenger is one of {KINDS}")
