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
    """Infra's challenger gate (`benchmarks/live_challenger/auditor.py`, v2) in this process, its ledger's entries
    written as the gate's `append` writes them and signed under `private` at rising one-time keys, from `leaf` on:
    `open` appends a window's open entry and returns the gate's open answer (``commitment`` null, as before serve.py
    has signed its record), `take` the next leaf for a session's record, `end` a session's end entry over its ``out/``
    as the gate's ``run`` computes it, and `lines` is the ledger as infra publishes it. Unlike the gate it appends
    whatever it is asked, so a test writes the ledgers the verifier must refuse."""

    def __init__(self, private: lms.PrivateKey, key: LC.ChallengerKey, leaf: int = 0) -> None:
        self.private, self.key, self._leaf, self.lines, self._opens = private, key, leaf, [], {}

    def take(self) -> int:
        self._leaf += 1
        return self._leaf - 1

    def append(self, kind: str, session: str, body: dict, *, schema: str = PHC.LEDGER_SCHEMA,
               leaf: int | None = None) -> dict:
        prev = hashlib.sha512(self.lines[-1]).hexdigest() if self.lines else "0" * 128
        entry = {"schema": schema, "seq": len(self.lines), "prev": prev, "kind": kind, "session": session, "body": body}
        message = PHC.entry_message(self.key.public[:16], entry)
        if leaf is not None:
            self._leaf = max(self._leaf, leaf)
        entry["sig"] = lms.sign_at(self.private, self.take() if leaf is None else leaf, message).encode().hex()
        if leaf is not None:
            self._leaf = max(self._leaf, leaf + 1)
        self.lines.append(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
        return entry

    def open(self, session: str, root: str, families, *, scheme: str = Wd.LIVE_SCHEME, epoch: int = 0,
             key_sha512: str | None = None, by: str = "vllm", v1: bool = False, leaf: int | None = None) -> dict:
        secret = hashlib.sha512(session.encode()).hexdigest()[:64]
        body = {"window": {"commit": root, "period": "p"} if v1 else {"root": root}, "pinned": "0" * 40,
                "host": f"{HOST}.{session}", "key_sha512": key_sha512 or hashlib.sha512(self.key.public).hexdigest(),
                "peer": "vy-nebius-1", "port": 7811, "scheme": scheme, "epoch": epoch,
                "families": [list(f) for f in families], "minutes": 10, "started_utc": "2026-10-20T11:08:46Z",
                **({"lane": by, "by": "research"} if v1 else
                   {"by": by, "secret_sha512": hashlib.sha512(bytes.fromhex(secret)).hexdigest()})}
        e = self.append("open", session, body, schema=PHC.LEDGER_V1 if v1 else PHC.LEDGER_SCHEMA, leaf=leaf)
        self._opens[session] = body
        return {"session": session, "port": body["port"], "peer": body["peer"], "host": body["host"], "seq": e["seq"],
                "root": root, "by": by, "secret": secret, "commitment": None}

    def end(self, session: str, out: Path, closed: bool = True, **override) -> dict:
        """The end entry the gate's ``run`` appends over `out`, its fields replaced by `override`, signed after the leaf
        its record was signed at."""
        files = {n: hashlib.sha512((out / n).read_bytes()).hexdigest() for n in sorted(os.listdir(out))}
        doc = json.loads((out / R.RECORD).read_text())
        record = LC.ChallengerRecord.from_json(doc)
        try:
            released = {k: len(r) for k, r in R.released_roots(out / R.RELEASES, record.commitment).items()}
        except LC.CoinError:
            released = None
        families = self._opens[session]["families"]
        body = {"rc": 0 if closed else 1, "closed": closed, "files": files,
                "ended": "closed by the client" if closed else "serve.py exited 1", "released": released,
                "complete": released is not None and all(released.get(k) == n for k, n in families),
                "record_leaf": PHC.leaf_of(doc["signature"]), "commitment": record.commitment.root.hex(),
                "ended_utc": "2026-10-20T11:08:53Z", **override}
        if type(body["record_leaf"]) is int:
            self._leaf = max(self._leaf, body["record_leaf"] + 1)
        return self.append("end", session, body)

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
