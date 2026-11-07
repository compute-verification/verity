"""The Pearl-C device path's manifest as a verifier reads it (`pouw_pearl_c_device`, which writes it): its schemas, the
live session a pass names (`session_head`), and a retained pass's epoch re-started from it (`manifest_epoch`), from the
epoch's coins or, for a pass recorded before them, from its beacon.
"""
from __future__ import annotations

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw.audit import Epoch, Workload
from verity.core.protocols.pouw.coins import CoinCommitment, CoinOpening
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.core.protocols.pouw.window import LIVE_SCHEME
from verity.core.protocols.sampled_proofs.challenger import LiveCommitment, LiveOpening

from verity_vllm.protocols.pouw.pearl_c.host import epoch_of

#: what a live pass's manifest names of its session: the gate's open answer's session, ledger seq, root (its salt epoch's
#: epoch root, `salt.epoch_open_root`) and coin commitment's root (`pouw_h1_challenger.SESSION_FIELDS`), never its hello's
#: secret
SESSION_KEYS = ("session", "seq", "root", "commitment")


def session_head(commits: dict) -> dict:
    """The manifest head's ``session`` for a run whose salt a live session gave (``commits["session"]``, the gate's
    answer as the run carries it, `pouw_h1_challenger.carried`): its `SESSION_KEYS` alone, and the salt epoch's number
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
    return schema


def manifest_epoch(man: dict, coins: CoinCommitment | LiveCommitment | None, recorded: bool = False) -> tuple[Epoch, str]:
    """A retained pass's epoch, as its verifier re-starts it, and how its salt was had.  A `core_check.MANIFEST_SCHEMA` pass
    carries the epoch's coin commitment and its salt coin's opening: the commitment must be `coins`, the one the challenger
    published (the verifier's copy, not the pass's; a live session's when the pass's coins are `LIVE_SCHEME`'s), and the
    opening the salt coin's against it, opened after every weight's root (`epoch_of`; a live session's after the epoch root
    of the weights' root and the manifest's ``pass_index``).  A pass recorded before epoch coins
    (`core_check.MANIFEST_RECORDED`) carries a beacon instead, which only a replay of a recorded pass reads (`recorded`:
    `Epoch.unverified`, a salt that claims nothing).  Either way the pass's salt must be the epoch's.  Raises `DomainError`
    naming what doesn't hold."""
    schema = manifest_schema(man, recorded)
    matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
    roots = {int(w): bytes.fromhex(v["root"]) for w, v in man["weights"].items()}
    if schema == CC.MANIFEST_SCHEMA:
        if "salt_opening" not in man:
            raise DomainError("the pass has no salt coin")
        if coins is None:
            raise DomainError("the pass's salt coin opens against the epoch's published coin commitment, which the verifier holds")
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
