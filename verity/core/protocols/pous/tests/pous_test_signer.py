"""A TEST signer for the memory challenger's sessions: S1's LMS (`verity.ml.crypto.lms`) at a small tree height,
so a key is made in milliseconds.  Its scheme id starts with `memory_challenger.TEST_PREFIX`, which
`memory_challenger.register` refuses, so a record it signs is accepted only by a test's own mapping (`SCHEMES`), never
by `memory_challenger.SCHEMES`.  Its leaf state, and a test's session log (`session_log`), are each in a fresh directory
inside `directory`, else inside `DIR`: the directory a suite's fixture gives the running tests, under pytest's basetemp,
which the check removes when the suite ends.  With neither they refuse, since nothing removes a directory made in
`$TMPDIR`.  A test's registered site (`site_of`) registers its wipe's server as a host with the tiers it declares."""
from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from verity.core.site import Loc, Registration, Site
from verity.ml.crypto import lms
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw

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


def site_log(host: str, tiers: Sequence[tuple[str, int]]) -> Registration:
    """A remote composite's registration log registering `host` with one memory leaf a tier, each `(name, bytes)` under
    its wipe name (`streamed_wipe.TIER_NAMES`), directly under the host."""
    spec = {name: t for t, name in sw.TIER_NAMES.items()}
    leaves = [Loc(f"{host}/{name}", "memory", host, tier=spec[name], bytes=b) for name, b in tiers]
    log = Registration("remote")
    return log.append(log.entry(locs=[Loc("site", "site"), Loc(host, "host", "site"), *leaves]))


def site_of(host: str, tiers: Sequence[tuple[str, int]]) -> Site:
    """The site `site_log(host, tiers)` registers: the wipe's tiers for `host` are `tiers`."""
    return Site.of_log(site_log(host, tiers))


def write_site(directory: Path, host: str, tiers: Sequence[tuple[str, int]]) -> str:
    """`site_log(host, tiers)` written to `directory/site.json`; its name, as `period.json`'s `site` names it."""
    (directory / "site.json").write_text(json.dumps(site_log(host, tiers).to_json()))
    return "site.json"
