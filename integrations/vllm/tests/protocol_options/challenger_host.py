"""The proof service's challenger on its auditor host, in this process: a `LiveChallenger` its test hands in, its record signed
with an LMS key (`remote.open_session`), and `remote.serve` on one end of a socket pair, the serving side's `RemoteChallenger`
on the other. The verifier reads `directory` under `key`. #2643's `tests/check/challenger_host.py`, over any challenger."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import socket
import threading
from pathlib import Path

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as LC
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms
from verity_vllm.protocol_options import pouw_h1_challenger as PHC

HOST = "vy-challenger-1"


def entropy(seed: int):
    """A seeded entropy source: a challenger's coins and salts, a distinct seed per challenger."""
    return random.Random(seed).randbytes


def challenger_key(seed: int, H: int = 2) -> tuple[lms.PrivateKey, LC.ChallengerKey]:
    k = lms.keygen(lms.Params(64, 4, H), bytes([seed]) * 64)
    return k, LC.ChallengerKey(k.params.W, k.params.H, lms.public_key(k))


class Gate:
    """Infra's challenger gate (`benchmarks/live_challenger/auditor.py`) in this process, its ledger's entries written
    as the gate's `append` writes them and signed under `private` at the one-time keys `leaves`: `open` appends a
    window's open entry and returns the gate's reply with the window, `end` a session's end entry sealing the files of
    its ``out/``, and `lines` is the ledger as infra publishes it. Unlike the gate it appends whatever it is asked, so a
    test writes the ledgers the verifier must refuse."""

    def __init__(self, private: lms.PrivateKey, key: LC.ChallengerKey, leaves) -> None:
        self.private, self.key, self._leaves, self.lines = private, key, iter(leaves), []

    def append(self, kind: str, session: str, body: dict) -> dict:
        prev = hashlib.sha512(self.lines[-1]).hexdigest() if self.lines else "0" * 128
        entry = {"schema": PHC.LEDGER_SCHEMA, "seq": len(self.lines), "prev": prev, "kind": kind, "session": session,
                 "body": body}
        message = PHC.entry_message(self.key.public[:16], entry)
        entry["sig"] = lms.sign_at(self.private, next(self._leaves), message).encode().hex()
        self.lines.append(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
        return entry

    def open(self, session: str, period: str, commit: str, families, *, scheme: str = Wd.LIVE_SCHEME, epoch: int = 0,
             key_sha512: str | None = None) -> dict:
        body = {"window": {"commit": commit, "period": period}, "pinned": "0" * 40, "host": f"{HOST}.{session}",
                "key_sha512": key_sha512 or hashlib.sha512(self.key.public).hexdigest(), "peer": "vy-nebius-1",
                "port": 7811, "scheme": scheme, "epoch": epoch, "families": [list(f) for f in families], "minutes": 10,
                "lane": "vllm", "by": "research", "started_utc": "2026-10-20T09:45:48Z"}
        e = self.append("open", session, body)
        return {"session": session, "seq": e["seq"], "period": period, "commit": commit, "host": body["host"],
                "port": body["port"], "peer": body["peer"]}

    def end(self, session: str, out: Path, closed: bool = True) -> dict:
        files = {n: hashlib.sha512((out / n).read_bytes()).hexdigest() for n in sorted(os.listdir(out))}
        return self.append("end", session, {"rc": 0 if closed else 1, "closed": closed, "files": files,
                                            "ended": "closed by the client" if closed else "serve.py exited 1",
                                            "ended_utc": "2026-10-20T09:46:13Z"})

    def ledger(self) -> PHC.Ledger:
        return PHC.Ledger.verify(self.lines, self.key)


class Host:
    """`challenger` with its record signed under `key` (seed `seed`) and written to `directory`, serving one session."""

    def __init__(self, directory, challenger: LC.LiveChallenger, seed: int) -> None:
        self.directory, self.challenger = directory, challenger
        directory.mkdir()
        private, self.key = challenger_key(seed)
        R.open_session(challenger, self.key, HOST, lambda m: lms.sign_at(private, 0, m).encode(), directory)
        ours, theirs = socket.socketpair()
        self.reason: str | None = None
        self._thread = threading.Thread(target=self._serve, args=(ours,))
        self._thread.start()
        self.client = R.RemoteChallenger(theirs)

    def _serve(self, sock: socket.socket) -> None:
        with sock:
            self.reason = R.serve(sock, self.challenger, self.directory, itertools.count().__next__)

    def ended(self) -> str | None:
        """Join the session once its client has ended it: None when the client closed it, else why it ended."""
        self._thread.join()
        return self.reason
