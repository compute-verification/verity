"""A TEST signer for the memory challenger's sessions: S1's LMS (`verity.ml.crypto.lms`) at a small tree height,
so a key is made in milliseconds.  Its scheme id starts with `memory_challenger.TEST_PREFIX`, which
`memory_challenger.register` refuses, so a record it signs is accepted only by a test's own mapping (`SCHEMES`), never
by `memory_challenger.SCHEMES`.  Its leaf state lives in a fresh temporary directory unless `state_dir` names one, as
does a test's session log (`session_log`)."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from verity.ml.crypto import lms
from verity.core.protocols.pous import memory_challenger as mc

HEIGHTS = range(1, 11)


def scheme(height: int) -> str:
    return mc.TEST_PREFIX + f"lms-sha512/n64-w1-h{height}/v0"


def TestSigner(secret: bytes, height: int = 3, state_dir: str | Path | None = None) -> mc.LmsSigner:  # noqa: N802
    """2^`height` one-time keys from `secret`; each `sign` uses the next, and one past the last is refused."""
    if height not in HEIGHTS:
        raise ValueError(f"a test signer's height is in {HEIGHTS}, not {height}")
    params = lms.Params(64, 1, height)
    key = lms.keygen(params, hashlib.sha512(b"verity-pous/test-signer/v0" + secret).digest()[:params.N])
    path = Path(state_dir or tempfile.mkdtemp(prefix="pous-test-signer-")) / "leaf-state.json"
    lms.LeafState.create(path, lms.public_key(key), height)
    return mc.LmsSigner(key, lms.LeafState(path, lms.public_key(key), height), scheme=scheme(height))


def session_log() -> mc.SessionLog:
    """A new memory challenger session log in a fresh temporary directory, so a test's sessions may reuse its ids."""
    path = Path(tempfile.mkdtemp(prefix="pous-session-log-")) / "sessions"
    mc.SessionLog.create(path)
    return mc.SessionLog(path)


SCHEMES: dict[str, mc.Verify] = {scheme(h): mc.lms_verify(lms.Params(64, 1, h)) for h in HEIGHTS}
"""The mapping a test passes to `memory_challenger.verdict`: the test signers accepted, in that test alone."""
