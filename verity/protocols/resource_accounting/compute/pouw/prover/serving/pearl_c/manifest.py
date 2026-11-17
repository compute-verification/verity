"""The Pearl-C device path's manifest as a verifier reads it (`pouw_pearl_c_device`, which writes it): its schemas, the
epoch a salt coin opens (`epoch_of`), a transcript file's entry (`_write`), the live session a pass names (`session_head`), and a retained pass's execution phase re-started from it (`manifest_epoch`), from the
execution phase's coins or, for a pass recorded before them, from its beacon.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import cast

from verity.protocols.resource_accounting.compute.pouw.verifier import core_check as CC
from verity.protocols.resource_accounting.compute.pouw.verifier import salt as PS
from verity.protocols.resource_accounting.compute.pouw.verifier import window as Wd
from verity.protocols.resource_accounting.compute.pouw.verifier.audit import Epoch, Workload, salt_of, weights_root
from verity.protocols.resource_accounting.compute.pouw.verifier.coins import CoinCommitment, CoinOpening
from verity.protocols.resource_accounting.compute.pouw.verifier.protocol import (
    DomainError,
    Matmul,
    Shape,
)
from verity.protocols.resource_accounting.compute.pouw.verifier.window import LIVE_SCHEME
from verity.protocols.proofs.spot.verifier.challenger import challenger as Ch
from verity.protocols.proofs.spot.verifier.challenger.challenger import LiveCommitment, LiveOpening

__all__ = ["SaltCoins", "SaltOpening", "epoch_of", "manifest_epoch", "session_head"]

#: the salt coin's commitment and opening a device run takes: the challenger's held in this process, or a live session's
type SaltCoins = CoinCommitment | LiveCommitment
type SaltOpening = CoinOpening | LiveOpening


def epoch_of(workload: Workload, roots: dict[int, bytes], coins: SaltCoins, opening: SaltOpening,
             pass_index: int | None = None) -> Epoch:
    """The epoch whose salt is the challenger's salt coin `opening`, opened against `coins` after the weights' root: a
    `CoinOpening` of a `CoinCommitment` through `Epoch.start`, or a live session's `LiveOpening` of its `LiveCommitment`
    (the auditor device's, `pouw_h1_window.open_session`) under the same rules: every weight committed and only those, the
    commitment a Pearl-C salt session's, its salt coin alone (`window.salt_epoch_reason`), and the opening that coin,
    opening against it after the salt epoch's root (`challenger.live_reason`), `salt.epoch_open_root` of the weights' root
    W and `pass_index`, the epoch's number k over W, an integer in [0, 2^64).  Either is refused with `DomainError`; a
    live salt is `audit.salt_of`'s over W, as H1's and ncp-v2's are."""
    live = isinstance(coins, LiveCommitment), isinstance(opening, LiveOpening)
    if not any(live):
        return Epoch.start(workload, roots, coins, opening)
    if not all(live):
        raise DomainError("a live session's salt coin opens against that session's live coin commitment, and nothing else")
    if set(roots) != set(workload.weights()):
        raise DomainError("every weight of the workload is committed, and only those")
    root = weights_root(roots)
    why = Wd.salt_epoch_reason(coins) or Ch.live_reason(coins, opening, Wd.SALT, 0, PS.epoch_open_root(root, pass_index))
    if why is not None:
        raise DomainError(f"the salt's {why}")
    return Epoch(workload, dict(roots), salt_of(coins, opening.coin, root))


def _entry(path: Path, n: int) -> dict:
    nested = path.parent.name in ("calls", "weights", "y")
    return {"path": f"{path.parent.name}/{path.name}" if nested else path.name, "bytes": n}


def _write(path: Path, data: bytes, digest: bool = True) -> dict:
    """A transcript file.  Row files carry no digest: the committed roots bind them, and the verifier checks every opened row
    against them."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(data)
    out = _entry(path, len(data))
    if digest:
        out["sha256"] = hashlib.sha256(data).hexdigest()
    return out


#: what a live pass's manifest names of its session: the gate's open answer's session, ledger seq, root (its salt execution phase's
#: execution phase root, `salt.epoch_open_root`) and coin commitment's root (`pouw_h1_challenger.SESSION_FIELDS`), never its hello's
#: secret
SESSION_KEYS = ("session", "seq", "root", "commitment")


def session_head(commits: dict) -> dict:
    """The manifest head's ``session`` for a run whose salt a live session gave (``commits["session"]``, the gate's
    answer as the run carries it, `pouw_h1_challenger.carried`): its `SESSION_KEYS` alone, and the salt execution phase's number
    ``pass_index``, from which and the weights the verifier recomputes the session's root. None given, none named."""
    s = commits.get("session")
    return {} if s is None else {"session": {k: s[k] for k in SESSION_KEYS}, "pass_index": commits.get("pass_index")}


def manifest_schema(man: dict, recorded: bool = False) -> str:
    """`man`'s schema, if a verifier reads it: `core_check.MANIFEST_SCHEMA`, or `core_check.MANIFEST_RECORDED` only as a
    replay of a recorded pass (`recorded`).  Raises `DomainError` otherwise."""
    schema = man.get("schema")
    if schema == CC.MANIFEST_RECORDED and not recorded:
        raise DomainError(f"the pass's manifest is {schema}, recorded before epoch coins: only a replay of a recorded pass "
                          "reads it")
    if schema not in (CC.MANIFEST_SCHEMA, CC.MANIFEST_RECORDED):
        raise DomainError(f"schema {schema!r}, not {CC.MANIFEST_SCHEMA}" + (f" or {CC.MANIFEST_RECORDED}" if recorded else ""))
    return cast(str, schema)


def manifest_epoch(man: dict, coins: CoinCommitment | LiveCommitment | None, recorded: bool = False) -> tuple[Epoch, str]:
    """A retained pass's execution phase, as its verifier re-starts it, and how its salt was had.  A `core_check.MANIFEST_SCHEMA` pass
    carries the execution phase's coin commitment and its salt coin's opening: the commitment must be `coins`, the one the challenger
    published (the verifier's copy, not the pass's; a live session's when the pass's coins are `LIVE_SCHEME`'s), and the
    opening the salt coin's against it, opened after every weight's root (`epoch_of`; a live session's after the execution phase root
    of the weights' root and the manifest's ``pass_index``).  A pass recorded before the execution phase's coins
    (`core_check.MANIFEST_RECORDED`) carries a beacon instead, which only a replay of a recorded pass reads (`recorded`:
    `Epoch.unverified`, a salt that claims nothing).  Either way the pass's salt must be the execution phase's.  Raises `DomainError`
    naming what doesn't hold."""
    schema = manifest_schema(man, recorded)
    matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
    roots = {int(w): bytes.fromhex(v["root"]) for w, v in man["weights"].items()}
    if schema == CC.MANIFEST_SCHEMA:
        if "salt_opening" not in man:
            raise DomainError("the pass has no salt coin")
        if coins is None:
            raise DomainError("the pass's salt coin opens against the epoch's published coin commitment, which the verifier holds")
        for key in ("salt_opening", "coins"):
            if key in man and not isinstance(man[key], dict):
                raise DomainError(f"the pass's coins are malformed: {key} is not an object")
        live = isinstance(man.get("coins"), dict) and man["coins"].get("scheme") == LIVE_SCHEME
        try:
            opening = (LiveOpening if live else CoinOpening).from_json(man["salt_opening"])
            claimed = (LiveCommitment if live else CoinCommitment).from_json(man["coins"])
        except (KeyError, TypeError, ValueError) as e:
            raise DomainError(f"the pass's coins are malformed: {e}") from None
        if claimed != coins:
            raise DomainError("the pass's coin commitment is not the epoch's")
        epoch, how = epoch_of(Workload(matmuls), roots, coins, opening, man.get("pass_index") if live else None), "coin"
    elif "beacon" in man:
        epoch, how = Epoch.unverified(Workload(matmuls), roots, bytes.fromhex(man["beacon"])), "recorded beacon"
    else:
        raise DomainError("the recorded pass has no beacon")
    if epoch.salt.hex() != man.get("salt"):
        raise DomainError(f"the pass's salt is not its {how}'s and the weights'")
    return epoch, how
