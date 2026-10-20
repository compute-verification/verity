"""The proof service's challenger on its auditor host, in this process: a `LiveChallenger` its test hands in, its record signed
with an LMS key (`remote.open_session`), and `remote.serve` on one end of a socket pair, the serving side's `RemoteChallenger`
on the other. The verifier reads `directory` under `key`. #2643's `tests/check/challenger_host.py`, over any challenger."""
from __future__ import annotations

import itertools
import random
import socket
import threading

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
