"""The proof service's challenger on its auditor host, in this process: a `LiveChallenger` its test hands in, its session
opened under an id with its record signed with an LMS key (`remote.open_session`), and `remote.serve` on one end of a socket
pair, the serving side's `RemoteChallenger` on the other; the session's close is signed at the key's next leaf. The verifier
reads `directory` under `key`, as the auditor's record (`audit`) names it. #2643's `tests/check/challenger_host.py`, over any
challenger."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import socket
import threading
from collections.abc import Iterator

from verity.core.protocols.sampled_proofs import challenger as LC
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

HOST = "vy-challenger-1"


def entropy(seed: int):
    """A seeded entropy source: a challenger's coins and salts, a distinct seed per challenger."""
    return random.Random(seed).randbytes


def challenger_key(seed: int) -> tuple[lms.PrivateKey, LC.ChallengerKey]:
    k = lms.keygen(lms.Params(64, 4, 2), bytes([seed]) * 64)
    return k, LC.ChallengerKey(k.params.W, k.params.H, lms.public_key(k))


def window(seed: int) -> dict:
    """A window's identity, ``{commit, period, pouw_salt, stream}``, as the auditor's record names it."""
    rng = random.Random(f"window/{seed}")
    return {"commit": rng.randbytes(64).hex(), "period": seed, "pouw_salt": rng.randbytes(64).hex(),
            "stream": rng.randbytes(64).hex()}


def audit(path, window: dict, session: str, key: LC.ChallengerKey) -> None:
    """Append the auditor's entry naming `session` for `window` under `key` (its public key's SHA-512) to the record at
    `path`, as infra's launcher does before the session's first coin."""
    entry = {**window, "code": "0" * 40, "key": hashlib.sha512(key.public).hexdigest(), "session": session}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


class Host:
    """`challenger` in session `session`, its record and close signed under key `seed` at the next of `leaves` (hosts
    sharing a key share them) and written to `directory`, serving one client."""

    def __init__(self, directory, challenger: LC.LiveChallenger, seed: int, session: str = "s-1",
                 leaves: Iterator[int] | None = None) -> None:
        self.directory, self.challenger, self.session_id = directory, challenger, session
        directory.mkdir()
        private, self.key = challenger_key(seed)
        leaves = itertools.count() if leaves is None else leaves
        session_ = R.open_session(challenger, self.key, HOST, lambda m: lms.sign_at(private, next(leaves), m).encode(),
                                  directory, session)
        ours, theirs = socket.socketpair()
        self.reason: str | None = None
        self._thread = threading.Thread(target=self._serve, args=(ours, session_))
        self._thread.start()
        self.client = R.RemoteChallenger(theirs)

    def _serve(self, sock: socket.socket, session: R.Session) -> None:
        with sock:
            self.reason = R.serve(sock, session, itertools.count().__next__)

    def audit(self, path, window: dict) -> None:
        """The auditor's record at `path` names this host's session for `window`, under its key."""
        audit(path, window, self.session_id, self.key)

    def ended(self) -> str | None:
        """Join the session once its client has ended it: None when the client closed it, else why it ended."""
        self._thread.join()
        return self.reason
