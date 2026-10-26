"""A stand-in for the wipe challenger's side on its own host (`wipe_party.main`), for `test_pous_wipe_period`'s leg: the
same stdin, stdout and socket, signed by a key at S1's deployed parameters whose tree holds leaf 0 alone (as
`test_pous_memory_challenger._SparseTree`), so no deployed key is made, and timed on a counting clock, so no reading is
the machine's. `python wipe_side.py DIR` keeps its session log in `DIR`."""

from __future__ import annotations

import hashlib
import itertools
import json
import socket
import sys
from pathlib import Path

from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import wipe_party as WP
from verity.ml.crypto import lms


class _SparseTree:
    def __init__(self, I: bytes, seed: bytes) -> None:
        self.root = b""
        probe = lms.sign_at(lms.PrivateKey(lms.DEPLOYED, I, seed, self), 0, b"probe")  # type: ignore[arg-type]
        self.root = lms.root_of(lms.DEPLOYED, I, b"probe", probe)

    def __getitem__(self, r: int) -> bytes:
        return self.root if r == 1 else hashlib.sha512(b"sibling%d" % r).digest()


class LeafZero:
    """An `LmsSigner` at S1's deployed parameters that signs at leaf 0, once."""

    scheme = mc.LMS

    def __init__(self) -> None:
        I, seed = hashlib.sha512(b"wipe-side/I").digest()[:16], hashlib.sha512(b"wipe-side/seed").digest()
        self.key = lms.PrivateKey(lms.DEPLOYED, I, seed, _SparseTree(I, seed))  # type: ignore[arg-type]
        self.public_key = lms.public_key(self.key)

    def sign(self, session: bytes, record: bytes) -> bytes:
        return lms.sign_at(self.key, 0, lms.memory_challenger_message(self.key.I, session, record)).encode()


def main() -> int:
    device = Path(sys.argv[1])
    request = json.loads(sys.stdin.readline())
    signer = LeafZero()
    mc.SessionLog.create(device / "sessions")
    with socket.create_server(("127.0.0.1", 0)) as srv:
        print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
        conn, _ = srv.accept()
        with conn, conn.makefile("rwb") as f:
            c = WP.serve(f, request, signer=signer, log=mc.SessionLog(device / "sessions"), egress_bits=WP.UNCOUNTED,
                         clock=itertools.count(10 ** 9, 1_000).__next__)
    print(json.dumps(WP.report(c, signer.public_key), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
