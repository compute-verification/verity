"""The Commit's challenger on its auditor host, in this process: a `LiveChallenger` under the Commit's scheme with a seeded entropy
source, its record signed with an LMS key (`remote.open_session`) and `remote.serve` on one end of a socket pair, the Commit's
`RemoteChallenger` on the other.  The verifier reads `directory` (`challenge.trust_session`) under `key`."""
from __future__ import annotations

import itertools
import random
import socket
import threading

from verity.core.protocols.sampled_proofs import challenger as LC
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms

from verity_vllm.commit import challenge as CH

HOST = "vy-nebius-1"


def entropy(seed: int):
    """A seeded entropy source: a challenger's coins and salts, a distinct seed per challenger."""
    return random.Random(seed).randbytes


def challenger_key(seed: int) -> tuple[lms.PrivateKey, LC.ChallengerKey]:
    k = lms.keygen(lms.Params(64, 4, 2), bytes([seed]) * 64)
    return k, LC.ChallengerKey(k.params.W, k.params.H, lms.public_key(k))


class Host:
    """A challenger over `challenge.layout(runs)` whose record is signed under `key` and written to `directory`, serving one session."""

    def __init__(self, tmp_path, runs: int, seed: int) -> None:
        self.directory = tmp_path / f"challenger-{seed}"
        self.directory.mkdir()
        self.challenger = LC.LiveChallenger(CH.SCHEME, 0, CH.layout(runs), entropy(seed))
        private, self.key = challenger_key(seed)
        R.open_session(self.challenger, self.key, HOST, lambda m: lms.sign_at(private, 0, m).encode(), self.directory)
        ours, theirs = socket.socketpair()
        self.reason: str | None = None
        self._thread = threading.Thread(target=self._serve, args=(ours,))
        self._thread.start()
        self.client = R.RemoteChallenger(theirs)

    def _serve(self, sock: socket.socket) -> None:
        with sock:
            self.reason = R.serve(sock, self.challenger, self.directory, itertools.count().__next__)

    def ended(self) -> str | None:
        """Join the session once its client has closed it: None when the client closed it, else why it ended."""
        self._thread.join()
        return self.reason
