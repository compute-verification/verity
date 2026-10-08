"""The Pearl-C device path's manifest as a verifier reads it (`pouw_pearl_c_device`, which writes it): its schemas, and a
retained pass's epoch re-started from it (`manifest_epoch`), from the epoch's coins or, for a pass recorded before them,
from its beacon.
"""
from __future__ import annotations

from verity.core.protocols.pouw.audit import Epoch, Workload
from verity.core.protocols.pouw.coins import CoinCommitment, CoinOpening
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape

MANIFEST_SCHEMA = "verity-vllm/pouw-pearl-c-run/v2"
#: a pass recorded before epoch coins, salted from a beacon: only a replay of a recorded pass reads it (`manifest_epoch`)
MANIFEST_SCHEMA_RECORDED = "verity-vllm/pouw-pearl-c-run/v1"
MANIFEST_SCHEMAS = (MANIFEST_SCHEMA, MANIFEST_SCHEMA_RECORDED)


def manifest_schema(man: dict, recorded: bool = False) -> str:
    """`man`'s schema, if a verifier reads it: `MANIFEST_SCHEMA`, or `MANIFEST_SCHEMA_RECORDED` only as a replay of a
    recorded pass (`recorded`).  Raises `DomainError` otherwise."""
    schema = man.get("schema")
    if schema == MANIFEST_SCHEMA_RECORDED and not recorded:
        raise DomainError(f"the pass's manifest is {schema}, recorded before epoch coins: only a replay of a recorded pass "
                          "reads it")
    if schema not in MANIFEST_SCHEMAS:
        raise DomainError(f"schema {schema!r}, not {MANIFEST_SCHEMA}" + (f" or {MANIFEST_SCHEMA_RECORDED}" if recorded else ""))
    return schema


def manifest_epoch(man: dict, coins: CoinCommitment | None, recorded: bool = False) -> tuple[Epoch, str]:
    """A retained pass's epoch, as its verifier re-starts it, and how its salt was had.  A `MANIFEST_SCHEMA` pass carries
    the epoch's coin commitment and its salt coin's opening: the commitment must be `coins`, the one the challenger published
    (the verifier's copy, not the pass's), and the opening the salt coin's against it, opened after every weight's root
    (`Epoch.start`).  A pass recorded before epoch coins (`MANIFEST_SCHEMA_RECORDED`) carries a beacon instead, which only a
    replay of a recorded pass reads (`recorded`: `Epoch.unverified`, a salt that claims nothing).  Either way the pass's salt
    must be the epoch's.  Raises `DomainError` naming what doesn't hold."""
    schema = manifest_schema(man, recorded)
    matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
    roots = {int(w): bytes.fromhex(v["root"]) for w, v in man["weights"].items()}
    if schema == MANIFEST_SCHEMA:
        if "salt_opening" not in man:
            raise DomainError("the pass has no salt coin")
        if coins is None:
            raise DomainError("the pass's salt coin opens against the epoch's published coin commitment, which the verifier holds")
        try:
            opening = CoinOpening.from_json(man["salt_opening"])
            claimed = CoinCommitment.from_json(man["coins"])
        except (KeyError, TypeError, ValueError) as e:
            raise DomainError(f"the pass's coins are malformed: {e}") from None
        if claimed != coins:
            raise DomainError("the pass's coin commitment is not the epoch's")
        epoch, how = Epoch.start(Workload(matmuls), roots, coins, opening), "coin"
    elif "beacon" in man:
        epoch, how = Epoch.unverified(Workload(matmuls), roots, bytes.fromhex(man["beacon"])), "recorded beacon"
    else:
        raise DomainError("the recorded pass has no beacon")
    if epoch.salt.hex() != man.get("salt"):
        raise DomainError(f"the pass's salt is not its {how}'s and the weights'")
    return epoch, how
