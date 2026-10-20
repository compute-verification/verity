"""The proof service's challenger on its auditor host, in this process: a `LiveChallenger` its test hands in, its session
opened under an id with its record signed with an LMS key (`remote.open_session`), and `remote.serve` on one end of a socket
pair, the serving side's `RemoteChallenger` on the other; the session's close is signed at the key's next leaf. The
auditor's signed ledger (`Ledger`, infra's ``windows.jsonl``) opens the session for its window before the first coin and
ends it once it has ended, sealing its files. The verifier reads `directory` under `key`, as the ledger opens and ends it.
#2643's `tests/check/challenger_host.py`, over any challenger."""
from __future__ import annotations

import hashlib
import itertools
import random
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

from verity.core.protocols.sampled_proofs import challenger as LC
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

HOST = "vy-challenger-1"


def entropy(seed: int):
    """A seeded entropy source: a challenger's coins and salts, a distinct seed per challenger."""
    return random.Random(seed).randbytes


def challenger_key(seed: int) -> tuple[lms.PrivateKey, LC.ChallengerKey]:
    """Key `seed`: sixteen leaves, four a session (the ledger's open, the record, the close and the ledger's end)."""
    k = lms.keygen(lms.Params(64, 4, 4), bytes([seed]) * 64)
    return k, LC.ChallengerKey(k.params.W, k.params.H, lms.public_key(k))


def window(seed: int) -> dict:
    """A window's identity, ``{commit, period}``, as the auditor's ledger opens it."""
    return {"commit": random.Random(f"window/{seed}").randbytes(64).hex(), "period": f"p-{seed}"}


WINDOW = window(0)


class Ledger:
    """The auditor's ledger at `path`, its entries signed under key `seed` at the next of `leaves` (shared with the
    challenger's own signatures under that key), naming key `named` in its opens (by default its own)."""

    def __init__(self, path: Path, seed: int, leaves: Iterator[int] | None = None,
                 named: LC.ChallengerKey | None = None) -> None:
        self.path, self.lines = path, []
        self.private, self.key = challenger_key(seed)
        self.leaves = itertools.count() if leaves is None else leaves
        self.key_sha512 = hashlib.sha512((named or self.key).public).hexdigest()

    def append(self, kind: str, session: str, body: dict) -> None:
        sign = lambda m: lms.sign_at(self.private, next(self.leaves), m).encode()  # noqa: E731
        self.lines.append(LC.ledger_line(self.lines, kind, session, body, self.key.public[:16], sign))
        self.path.write_text("".join(line + "\n" for line in self.lines))

    def open(self, session: str, window: dict) -> None:
        self.append("open", session, {"window": window, "host": f"{HOST}.{session}", "pinned": "0" * 40,
                                      "key_sha512": self.key_sha512, "port": 7811})

    def end(self, session: str, directory: Path, closed: bool) -> None:
        files = {p.name: hashlib.sha512(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
        self.append("end", session, {"closed": closed, "files": files, "rc": 0 if closed else 1})


class Host:
    """`challenger` in session `session` of `window` (`WINDOW` by default), its record and close signed under key
    `seed` at the next of `leaves` (hosts sharing a key share them) and written to `directory`, serving one client; the
    ledger `ledger` (a new one beside `directory` unless given) opens the session before its record and ends it once
    `ended`."""

    def __init__(self, directory: Path, challenger: LC.LiveChallenger, seed: int, session: str = "s-1",
                 leaves: Iterator[int] | None = None, ledger: Ledger | None = None, window: dict | None = None) -> None:
        self.directory, self.challenger, self.session_id = directory, challenger, session
        directory.mkdir()
        private, self.key = challenger_key(seed)
        leaves = itertools.count() if leaves is None else leaves
        self.ledger = ledger or Ledger(directory.parent / f"{directory.name}.jsonl", seed, leaves)
        self.ledger.open(session, WINDOW if window is None else window)
        session_ = R.open_session(challenger, self.key, f"{HOST}.{session}",
                                  lambda m: lms.sign_at(private, next(leaves), m).encode(), directory, session)
        ours, theirs = socket.socketpair()
        self.reason: str | None = None
        self._ended = False
        self._thread = threading.Thread(target=self._serve, args=(ours, session_))
        self._thread.start()
        self.client = R.RemoteChallenger(theirs)

    def _serve(self, sock: socket.socket, session: R.Session) -> None:
        with sock:
            self.reason = R.serve(sock, session, itertools.count().__next__)

    def ended(self) -> str | None:
        """Join the session once its client has ended it, and the ledger ends it: None when the client closed it, else
        why it ended."""
        self._thread.join()
        if not self._ended:
            self._ended = True
            self.ledger.end(self.session_id, self.directory, self.reason is None)
        return self.reason
