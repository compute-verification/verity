"""A TEST signer for the memory challenger's sessions: S1's LMS (`verity.ml.crypto.lms`) at a small tree height,
so a key is made in milliseconds.  Its scheme id starts with `memory_challenger.TEST_PREFIX`, which
`memory_challenger.register` refuses, so a record it signs is accepted only by a test's own mapping (`SCHEMES`), never
by `memory_challenger.SCHEMES`.  Its leaf state, and a test's session log (`session_log`), are each in a fresh directory
inside `directory`, else inside `DIR`: the directory a suite's fixture gives the running tests, under pytest's basetemp,
which the check removes when the suite ends.  With neither they refuse, since nothing removes a directory made in
`$TMPDIR`."""
from __future__ import annotations

import hashlib
import itertools
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from verity.ml.crypto import lms
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous.audit import DEADLINE_NS

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


def TestSigner(secret: bytes, height: int = 3, directory: str | Path | None = None) -> mc.LmsSigner:  # noqa: N802
    """2^`height` one-time keys from `secret`; each `sign` uses the next, and one past the last is refused."""
    key = _key(secret, height)
    path = _fresh("pous-test-signer-", directory) / "leaf-state.json"
    lms.LeafState.create(path, lms.public_key(key), height)
    return mc.LmsSigner(key, path, scheme=scheme(height))


def session_log(directory: str | Path | None = None) -> mc.SessionLog:
    """A new memory challenger session log in a fresh directory, so a test's sessions may reuse its ids."""
    path = _fresh("pous-session-log-", directory) / "sessions"
    mc.SessionLog.create(path)
    return mc.SessionLog(path)


SCHEMES: dict[str, mc.Verify] = {scheme(h): mc.lms_verify(lms.Params(64, 1, h)) for h in HEIGHTS}
"""The mapping a test passes to `memory_challenger.verdict`: the test signers accepted, in that test alone."""

COIN_KEY = mc.HM96.key_from_label(b"pous test coin key")


def one_coin_key(monkeypatch) -> None:
    """Draw every coin tree under `COIN_KEY` unless the caller names a key, so hm96 builds its tables once rather than
    once a session (about 15 ms each).  The key only hides the coins, which a test of the rounds' timing and verdicts
    never reads."""
    draw = mc.CoinTree.draw.__func__
    monkeypatch.setattr(mc.CoinTree, "draw",
                        classmethod(lambda cls, *a, **kw: draw(cls, *a, **({"key": COIN_KEY} | kw))))


def audit(blocks: Sequence[bytes], binding: bytes, k: int, respond: Callable[[int], bytes] | None = None, *,
          limit_ns: int = DEADLINE_NS, step_ns: int = 1_000, source: bytes = b"audit",
          session: bytes = bytes(32)) -> mc.Verdict:
    """One memory challenger session of `k` rounds over `blocks` (the `C*` whose setup root is `binding`), answered by
    `respond(index)` (the stored block by default) on a clock that ticks `step_ns` a reading, signed by a test signer
    and judged by `memory_challenger.verdict` against the blocks' own digests."""
    signer = TestSigner(source, height=1)
    c = mc.MemoryChallenger(mc.CoinTree.draw(source, session, len(blocks), k), signer, log=session_log(),
                            limit_ns=limit_ns, block_bytes=len(blocks[0]), binding=binding,
                            clock=itertools.count(0, step_ns).__next__)
    signed = c.run(respond or blocks.__getitem__)
    return mc.verdict(signed, c.commitment, c.openings(), lambda i: mc.block_digest(blocks[i]),
                      trusted={signer.public_key}, schemes=SCHEMES)
