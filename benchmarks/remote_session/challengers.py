"""Where a session's coins come from (`verifier.py --challenger`), each an epoch of the proof service's one challenger
(`coins.Coins`, `sampled_proofs.challenger`):

- `Local`: a `LiveChallenger` in the verifier's process, drawing from `secrets.token_bytes`.
- `Serve`: the challenger's own process, `benchmarks/live_challenger/serve.py`, started for the session on the
  verifier's machine with its LMS key (`--challenger-key`), reached over its 127.0.0.1 by `sampled_proofs.remote`'s
  client. It draws its coins in the Lean (`one_stage.verifier.coins`), signs its record before it listens and its
  release log when the client closes the session. Not the auditor host: the verifier's machine is the audit's (never
  the prover's), but no ledger opens the session for its window, so the read is `challenger.session_of`'s, under the
  key the run wrote.

`finish` reads back what the challenger signed and holds the session's coins to it (`audit.Session.released_reason`):
the same commitment, each coin opened after the root the session recorded and in order, σ_0 after the session's window
root R, each window's openings coin after its last root. Its `reason` is None when they hold."""
from __future__ import annotations

import hashlib
import json
import secrets
import socket
import subprocess
import sys
from pathlib import Path

from verity.core.protocols.remote_bandwidth import audit, coins
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote
from verity.ml.crypto import lms

ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / "benchmarks" / "live_challenger" / "serve.py"
KINDS = ("local", "serve")


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

    def open(self, registration: bytes, k: int, windows: int, appends: int) -> Ch.LiveChallenger:
        return Ch.LiveChallenger(coins.SCHEME, k, coins.families(windows, appends), secrets.token_bytes)

    def finish(self, s: audit.Session) -> dict:
        c = s.coins
        return {"challenger": self.kind, "commitment": c.commitment.root.hex(),
                "reason": s.released_reason(c.commitment, {label: c.afters(label) for label in coins.LABELS})}


class Serve:
    """Session coins from `serve.py`, one process a session, its `out/` under `out`/SESSION, its key in `key`
    (made there on first use at `params`, `W,H`, or `lms.DEPLOYED`'s)."""

    kind = "serve"

    def __init__(self, out: Path, key: Path, params: str | None = None, python: str = sys.executable) -> None:
        self.out, self.key, self.params, self.python = out, key, params, python
        self.run = secrets.token_hex(4)
        self._live: dict[int, tuple[subprocess.Popen, Path, str, remote.RemoteChallenger]] = {}

    def open(self, registration: bytes, k: int, windows: int, appends: int) -> remote.RemoteChallenger:
        sid = session_id(registration, k, self.run)
        out = self.out / sid
        with open(self.out / f"{sid}.log", "w") as log:
            proc = subprocess.Popen(
                [self.python, str(SERVE), "--out", str(out), "--port", "0", "--scheme", coins.SCHEME,
                 "--families", json.dumps([list(f) for f in coins.families(windows, appends)]), "--session", sid,
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

    def finish(self, s: audit.Session) -> dict:
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
        return {**doc, "key_sha512": hashlib.sha512(key.public).hexdigest(),
                "released": {label: len(received.get(label, ())) for label in coins.LABELS},
                "reason": s.released_reason(signed, received)}


def of(kind: str, out: Path, key: Path | None, params: str | None) -> Local | Serve:
    """`--challenger KIND`'s source; `serve` needs its key directory."""
    if kind == "local":
        return Local()
    if kind == "serve":
        if key is None:
            raise ValueError("--challenger serve needs --challenger-key DIR, the challenger's LMS key directory, "
                             "outside the run's out/")
        out.mkdir(parents=True, exist_ok=True)
        return Serve(out, key, params)
    raise ValueError(f"--challenger is one of {KINDS}")
