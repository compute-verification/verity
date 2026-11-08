"""The auditor host's gate on loopback, for M3's coin (`m3coin.coin`): the verbs it asks (``pins``, ``pubkey``, ``reserve``,
``open``, ``status``, ``fetch``, ``ledger``) answered in this process as `benchmarks/live_challenger/auditor.py` answers
them, each session's challenger serving on 127.0.0.1 (`remote.open_session`, `remote.serve`) and every ledger entry,
record and close signed with this gate's own LMS key at its next leaf.

It reserves a session withheld when the reserve asks (``--withhold``) and the gate `withholds`, as the auditor host's
gate does: the ledger's reserve and open entries carry ``withheld: true``, its challenger answers with its layout and
raw coins alone, the open's answer says ``withheld`` and ``signed``, and the open's and status's ``commitment`` stay
null until the session's end. With `withholds` False it ignores ``--withhold`` and serves every session plain, as a gate
that didn't withhold would: the challenger's root goes out with the open's answer and at the hello. The tests and
`m3withheld.py`'s recorded run use it; it is no part of what a verifier trusts.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import random
import socket
import tarfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

HOST = "vy-challenger-1"


class LocalGate:
    """A gate over `directory` (each session's tree ``<sid>/out``, the ledger ``windows.jsonl``), its LMS key from `seed`
    (64 bytes) at W = 4 and height `H`, the host pinned to `pin` (a commit on main, recorded at `utc`), its challengers'
    coins from `entropy` (bytes), withholding a session whose reserve asks for it only when `withholds`."""

    def __init__(self, directory: Path, seed: bytes, pin: str, utc: str, *, H: int = 5, withholds: bool = True,
                 entropy: bytes = b"m3gate") -> None:
        self.dir, self.utc, self.withholds = Path(directory), utc, withholds
        self.dir.mkdir(parents=True, exist_ok=True)
        self.private = lms.keygen(lms.Params(64, 4, H), seed)
        self.public = lms.public_key(self.private)
        self.key = Ch.ChallengerKey(4, H, self.public)
        self.pin, self.pins = pin, [Ch.pin_line(pin, pin, utc)]
        self.leaves, self.lines, self.entropy = itertools.count(), [], entropy
        self.sessions: dict[str, dict[str, Any]] = {}

    def key_record(self) -> dict:
        """The challenger key record a verifier pins (`challenger.key_of_record`)."""
        return {"schema": Ch.KEY_RECORD_SCHEMA, "W": self.key.W, "H": self.key.H, "pubkey": self.public.hex(),
                "pubkey_sha512": hashlib.sha512(self.public).hexdigest()}

    def sign(self, message: bytes) -> bytes:
        return lms.sign_at(self.private, next(self.leaves), message).encode()

    def append(self, kind: str, session: str | None, body: dict) -> int:
        self.lines.append(Ch.ledger_line(self.lines, kind, session, body, self.public[:16], self.sign))
        (self.dir / "windows.jsonl").write_text("".join(line + "\n" for line in self.lines))
        return len(self.lines) - 1

    def __call__(self, words: list[str]) -> bytes:
        verb, withhold = words[0], "--withhold" in words[1:]
        args = [w for w in words[1:] if w != "--withhold"]
        opts = {k[2:]: v for k, v in zip(args[::2], args[1::2]) if k.startswith("--")}
        if verb == "pins":
            return "".join(p + "\n" for p in self.pins).encode()
        if verb == "pubkey":
            return json.dumps(self.key.to_json()).encode()
        if verb == "ledger":
            return "".join(line + "\n" for line in self.lines).encode()
        if verb == "reserve":
            return json.dumps(self.reserve(opts, withhold)).encode()
        if verb == "open":
            return json.dumps(self.open(opts["session"], opts["root"])).encode()
        if verb == "status":
            return json.dumps(self.status(args[0])).encode()
        if verb == "fetch":
            return self.fetch(args[0])
        raise Ch.CoinError(f"the gate has no verb {verb}")

    def reserve(self, opts: dict, withhold: bool = False) -> dict:
        sid, withheld = f"s20261027T000000Z-{len(self.sessions):04x}", withhold and self.withholds
        req = {"by": "compute-accounting", "peer": opts["peer"], "scheme": opts["scheme"], "epoch": int(opts["epoch"]),
               "families": json.loads(opts["families"]), "minutes": int(opts["minutes"])}
        start = datetime.strptime(self.utc, "%Y-%m-%dT%H:%M:%SZ")
        expires = (start + timedelta(minutes=req["minutes"])).strftime("%Y-%m-%dT%H:%M:%SZ")
        seq = self.append("reserve", sid, {**req, "pinned": self.pin, "key_sha512": hashlib.sha512(self.public).hexdigest(),
                                           "host": f"{HOST}.{sid}", "started_utc": self.utc, "expires_utc": expires,
                                           **({"withheld": True} if withheld else {})})
        self.sessions[sid] = {"request": req, "reserve": seq, "withheld": withheld}
        return {"session": sid, "seq": seq, "peer": opts["peer"], "by": req["by"], "expires_utc": expires,
                **({"withheld": True} if withheld else {})}

    def open(self, sid: str, root: str) -> dict:
        s = self.sessions[sid]
        req, withheld = s["request"], s["withheld"]
        secret = hashlib.sha512(self.entropy + b"secret" + sid.encode()).digest()[:32]
        listener = socket.create_server(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        self.append("open", sid, {"window": {"root": root}, "reserve": s["reserve"], **req, "opened_by": req["by"],
                                  "pinned": self.pin, "pins": Ch.pins_seal(self.pins),
                                  "key_sha512": hashlib.sha512(self.public).hexdigest(), "host": f"{HOST}.{sid}",
                                  "port": port, "secret_sha512": hashlib.sha512(secret).hexdigest(),
                                  "started_utc": self.utc, **({"withheld": True} if withheld else {})})
        out = self.dir / sid / "out"
        out.mkdir(parents=True)
        challenger = Ch.LiveChallenger(req["scheme"], req["epoch"], tuple((f, n) for f, n in req["families"]),
                                       random.Random(self.entropy + sid.encode()).randbytes)
        session = R.open_session(challenger, self.key, f"{HOST}.{sid}", self.sign, out, sid, withheld)
        (out / R.HELLOS).write_text("")
        s.update(out=out, withheld=withheld, root=challenger.commitment.root.hex(), reason=None, ended=None)

        def run() -> None:
            conn, _ = listener.accept()
            listener.close()
            with conn:
                s["reason"] = R.serve(conn, session, itertools.count().__next__, secret)
        s["thread"] = threading.Thread(target=run)
        s["thread"].start()
        return {"session": sid, "seq": len(self.lines) - 1, "root": root, "port": port, "secret": secret.hex(),
                "commitment": None if withheld else s["root"], **({"withheld": True, "signed": True} if withheld else {})}

    def status(self, sid: str) -> dict:
        """The session's status once its challenger has stopped: its callers ask only after their client's close."""
        s = self.sessions[sid]
        s["thread"].join()
        if s["ended"] is None:
            self.end(sid)
        return {"session": sid, "commitment": None if s["withheld"] and s["ended"] is None else s["root"],
                "end": s["ended"]}

    def end(self, sid: str) -> None:
        """The session's end entry, as the gate's ``run`` writes it once its challenger has stopped."""
        s = self.sessions[sid]
        d = s["out"]
        record = Ch.ChallengerRecord.from_json(json.loads((d / R.RECORD).read_text()))
        released = {k: len(r) for k, r in R.released_roots(d / R.RELEASES, record.commitment).items()}
        closed = s["reason"] is None
        body = {"rc": 0 if closed else 1, "closed": closed, "ended": "closed by the client" if closed else s["reason"],
                "files": {p.name: hashlib.sha512(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())},
                "released": released, "complete": all(released[k] == n for k, n in record.commitment.families),
                "record_leaf": int.from_bytes(record.signature[:4], "big"), "commitment": record.commitment.root.hex(),
                "refused_hellos": len((d / R.HELLOS).read_text().splitlines()), "ended_utc": self.utc}
        self.append("end", sid, body)
        s["ended"] = {"body": body}

    def fetch(self, sid: str) -> bytes:
        s = self.sessions[sid]
        if s["withheld"] and s["ended"] is None:
            raise Ch.CoinError(f"session {sid} has not ended: a withheld session's record is fetched only after its end")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(self.dir / sid, arcname=sid)
        return buf.getvalue()

    def join(self) -> None:
        for s in self.sessions.values():
            if "thread" in s:
                s["thread"].join()
