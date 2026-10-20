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
from datetime import datetime, timedelta
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
    `reserve` appends a window's reserve entry, `open` its open entry (two-step after its reserve, else one-step) and
    returns the gate's open answer (``commitment`` null, as before serve.py has signed its record), `take` the next leaf
    for a session's record, `end` a session's end entry over its ``out/`` as the gate's ``run`` computes it,
    `period_open` and `period_close` a lane's period entries (the close listing its windows' outcomes as the gate
    counts them), and `lines` is the ledger as infra publishes it. Unlike the gate it appends whatever it is asked, so a
    test writes the ledgers the verifier must refuse."""

    def __init__(self, private: lms.PrivateKey, key: LC.ChallengerKey, leaf: int = 0) -> None:
        self.private, self.key, self._leaf, self.lines, self._opens = private, key, leaf, [], {}
        self._reserves: dict[str, dict] = {}
        self._periods: dict[str, dict] = {}
        self._current: dict[str, str] = {}
        self._complete: dict[str, bool] = {}

    @property
    def pin(self) -> str:
        return hashlib.sha512(self.key.public).hexdigest()

    def _join(self, lane: str, session: str) -> None:
        if (name := self._current.get(lane)) is not None:
            self._periods[name]["windows"].append(session)

    def reserve(self, session: str, families, *, scheme: str = Wd.LIVE_SCHEME, epoch: int = 0, by: str = "vllm",
                for_: str | None = None, minutes: int = 10, started: str = "2026-10-20T11:08:00Z", **override) -> dict:
        """A window's reserve entry, its expiry `started` plus `minutes` (the gate's ``reserve``), its fields replaced by
        `override`; its session's `open` is then two-step."""
        expires = (datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ") + timedelta(minutes=minutes)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        body = {"by": by, **({"for": for_} if for_ else {}), "peer": "vy-nebius-1", "scheme": scheme, "epoch": epoch,
                "families": [list(f) for f in families], "minutes": minutes, "pinned": "0" * 40, "key_sha512": self.pin,
                "host": f"{HOST}.{session}", "started_utc": started, "expires_utc": expires, **override}
        e = self.append("reserve", session, body)
        self._reserves[session] = e
        self._join(for_ or by, session)
        return e

    def period_open(self, name: str, lane: str = "vllm", *, by: str = "vllm", cap: int | None = None,
                    circuit: str | None = None, **override) -> dict:
        body = {"period": name, "lane": lane, "by": by, **({"cap": cap} if cap is not None else {}),
                **({"circuit": circuit} if circuit is not None else {}), "pinned": "0" * 40, "key_sha512": self.pin,
                "opened_utc": "2026-10-20T11:00:00Z", **override}
        self._periods[name], self._current[lane] = {"lane": lane, "windows": []}, name
        return self.append("period-open", None, body)

    def period_close(self, name: str, **override) -> dict:
        """The period's close as the gate's ``period`` writes it: each of its windows, in order, with its root (None
        when it was never opened) and whether its end is complete, and complete when it has one and all are."""
        p = self._periods[name]
        windows = [{"root": self._opens[s]["window"]["root"] if s in self._opens else None, "session": s,
                    "complete": self._complete.get(s, False)} for s in p["windows"]]
        body = {"period": name, "lane": p["lane"], "by": "vllm", "windows": windows,
                "complete": bool(windows) and all(w["complete"] for w in windows), "closed_utc": "2026-10-20T12:00:00Z",
                **override}
        if self._current.get(p["lane"]) == name:
            del self._current[p["lane"]]
        return self.append("period-close", None, body)

    def take(self) -> int:
        self._leaf += 1
        return self._leaf - 1

    def append(self, kind: str, session: str | None, body: dict, *, schema: str = PHC.LEDGER_SCHEMA,
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

    def open(self, session: str, root: str, families=(), *, scheme: str = Wd.LIVE_SCHEME, epoch: int = 0,
             key_sha512: str | None = None, by: str = "vllm", v1: bool = False, leaf: int | None = None,
             **override) -> dict:
        """A window's open entry and the gate's answer: two-step (binding its reserve, carrying its request, opened by
        `by`) when `reserve` reserved the session, else one-step (``families`` and the rest its own); its body's fields
        replaced by `override`."""
        secret = hashlib.sha512(session.encode()).hexdigest()[:64]
        reserved = self._reserves.get(session)
        if reserved is not None:
            rb = reserved["body"]
            body = {"window": {"root": root}, "reserve": reserved["seq"], **{k: rb[k] for k in PHC.RESERVED if k in rb},
                    "opened_by": by, "pinned": "0" * 40, "key_sha512": key_sha512 or self.pin, "host": rb["host"],
                    "port": 7811, "secret_sha512": hashlib.sha512(bytes.fromhex(secret)).hexdigest(),
                    "started_utc": rb["started_utc"]}
        else:
            body = {"window": {"commit": root, "period": "p"} if v1 else {"root": root}, "pinned": "0" * 40,
                    "host": f"{HOST}.{session}", "key_sha512": key_sha512 or self.pin,
                    "peer": "vy-nebius-1", "port": 7811, "scheme": scheme, "epoch": epoch,
                    "families": [list(f) for f in families], "minutes": 10, "started_utc": "2026-10-20T11:08:46Z",
                    **({"lane": by, "by": "research"} if v1 else
                       {"by": by, "secret_sha512": hashlib.sha512(bytes.fromhex(secret)).hexdigest()})}
        body.update(override)
        e = self.append("open", session, body, schema=PHC.LEDGER_V1 if v1 else PHC.LEDGER_SCHEMA, leaf=leaf)
        self._opens[session] = body
        if reserved is None and not v1:
            self._join(body.get("for") or body["by"], session)
        return {"session": session, "port": body["port"], "peer": body["peer"], "host": body["host"], "seq": e["seq"],
                "root": root, "by": body["by"], **({"for": body["for"]} if "for" in body else {}),
                **({"reserve": body["reserve"], "opened_by": by} if reserved is not None else {}), "secret": secret,
                "commitment": None}

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
        self._complete[session] = body["complete"] is True
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
