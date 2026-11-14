"""A TEST signer for the memory challenger's sessions: S1's LMS (`verity.commitments.crypto.lms`) at a small tree height,
so a key is made in milliseconds.  Its scheme id starts with `memory_challenger.TEST_PREFIX`, which
`memory_challenger.register` refuses, so a record it signs is accepted only by a test's own mapping (`SCHEMES`), never
by `memory_challenger.SCHEMES`.  Its leaf state, and a test's session log (`session_log`), are each in a fresh directory
inside `directory`, else inside `DIR`: the directory a suite's fixture gives the running tests, under pytest's basetemp,
which the check removes when the suite ends.  With neither they refuse, since nothing removes a directory made in
`$TMPDIR`."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from verity.commitments.crypto import lms
from verity.protocols.accounting.pous.device import memory_challenger as mc

HEIGHTS = range(1, 11)
DIR: Path | None = None
"""Where the helpers make their directories when the caller names none: set by the suite's fixture (`tmp_path_factory`)."""


def scheme(height: int) -> str:
    return mc.TEST_PREFIX + f"lms-sha512/n64-w1-h{height}/v0"


def _key(secret: bytes, height: int) -> lms.PrivateKey:
    if height not in HEIGHTS:
        raise ValueError(f"a test signer's height is in {HEIGHTS}, not {height}")
    params = lms.Params(64, 1, height)
    return lms.keygen(params, hashlib.sha512(b"verity-pous/test-signer/v0" + secret).digest()[:params.N])


def _fresh(prefix: str, directory: str | Path | None) -> Path:
    parent = directory or DIR
    if parent is None:
        raise RuntimeError("the test signer makes its files in a directory the caller owns: pass `directory` (a "
                           "TemporaryDirectory outside pytest), or set `pous_test_signer.DIR` from `tmp_path_factory` in "
                           "the suite's conftest")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def public_key(secret: bytes, height: int = 3) -> bytes:
    """`TestSigner(secret, height)`'s public key, with no leaf state made."""
    return lms.public_key(_key(secret, height))


def TestSigner(secret: bytes, height: int = 3, directory: str | Path | None = None) -> mc.LmsSigner:
    """2^`height` one-time keys from `secret`; each `sign` uses the next, and one past the last is refused."""
    key = _key(secret, height)
    path = _fresh("pous-test-signer-", directory) / "leaf-state.json"
    lms.LeafState.create(path, lms.public_key(key), height)
    return mc.LmsSigner(key, path, scheme=scheme(height))


class _SparseTree:
    """A deployed key's tree for leaf 0 alone: its path's siblings arbitrary, its root the one they give."""

    def __init__(self, I: bytes, seed: bytes) -> None:
        self.root = b""
        probe = lms.sign_at(lms.PrivateKey(lms.DEPLOYED, I, seed, self), 0, b"probe")  # type: ignore[arg-type]
        self.root = lms.root_of(lms.DEPLOYED, I, b"probe", probe)

    def __getitem__(self, r: int) -> bytes:
        return self.root if r == 1 else hashlib.sha512(b"sibling%d" % r).digest()


class AtLeaf:
    """`memory_challenger.LmsSigner` at S1's deployed parameters (`mc.LMS`), so a check that verifies only at those
    accepts what it signs, without a deployed key's 2^20-leaf keygen: it signs at leaf 0 every time and keeps no leaf
    state, which a deployed signer refuses."""

    scheme = mc.LMS

    def __init__(self, label: bytes) -> None:
        I, seed = hashlib.sha512(label + b"/I").digest()[:16], hashlib.sha512(label + b"/seed").digest()
        self.key = lms.PrivateKey(lms.DEPLOYED, I, seed, _SparseTree(I, seed))  # type: ignore[arg-type]
        self.public_key = lms.public_key(self.key)

    def sign(self, session: bytes, record: bytes) -> bytes:
        return lms.sign_at(self.key, 0, lms.memory_challenger_message(self.key.I, session, record)).encode()


def session_log(directory: str | Path | None = None) -> mc.SessionLog:
    """A new memory challenger session log in a fresh directory, so a test's sessions may reuse its ids."""
    path = _fresh("pous-session-log-", directory) / "sessions"
    mc.SessionLog.create(path)
    return mc.SessionLog(path)


SCHEMES: dict[str, mc.Verify] = {scheme(h): mc.lms_verify(lms.Params(64, 1, h)) for h in HEIGHTS}
"""The mapping a test passes to `memory_challenger.verdict`: the test signers accepted, in that test alone."""
