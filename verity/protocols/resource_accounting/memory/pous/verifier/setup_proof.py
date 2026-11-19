"""PoUS's setup as a sampled proof about the store, the auditor's side: what its check takes in place of the store.

Today's setup check (`setup.verifier_key_from_encoding`, `PousCheck.SetupProtocol`) reads all of C* and decodes it.  In
proofs it reads none of it. The setup is one circuit C (`verity/protocols/resource_accounting/memory/pous/prover/window.py`,
`Setup`): ``PousP2SetupCheck_v2`` over every block of the store, one proof block a store block, the store (the table
``store/blocks``, `window.store_table`) and W's rows registered values. The prover commits the store and W, then the
setup's values; the proof layer draws blocks on its live coins after that (its `Draw`, `service.Draw`), and `check`
reads the door's verdict on each drawn block (SPOT's `proofs.verify` on C, the partition and the draw, the block's units
the partition's, the caller's door), with no sample or design of its own. The claim (`Claim`) carries the parameters
and commitment as today, the store's registered root, the draw's law and SPOT's outcome to the Lean check
`PousCheck.SetupInProofs.check`.

Two places stop it. The one value the verifier supplies to C is the setup's salt (`pous.setup.salt/v1`, the memory
attester's coin, drawn after W is committed); each unit builds its base from it and its place inside, its place and slot
are constants of the unit in C, and its ok word is an output held to 1. No supplied input binds the salt to C yet, nor a
constant of a unit or a held output (circuits' and proofs' pair), so `bound` refuses (`NotBound`). And the door's
statement of a block is not derived (`one_circuit.lower`, proofs').
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .schemes.p2 import P2Params

GOAL = "memory-accounting/pous-setup-as-proof"

#: where C's one supplied value waits on the supplied input
NOT_BOUND = "pous setup: no supplied input yet: "
SALT_INPUT = ("salt = pous.setup.salt/v1, the memory attester's coin drawn after W is committed, the one value the "
              "verifier supplies; unit u builds its base expand_prefix(params, salt) || u64be(u / S) || u64be(u mod S) "
              "inside")
CONSTANTS = ("its block's u64be(u / S) || u64be(u mod S) and its slot u mod n constants of the unit in C, and its ok a "
             "public output held to 1")


class NotBound(ValueError):
    """C's salt is bound to nothing, nor its units' constants or its held ok: the supplied input (circuits' and proofs'
    pair) isn't in C's format yet."""


def bound(params: P2Params, salt: bytes, row_bytes: int) -> None:
    """C's salt input bound to the verifier's `salt`, each unit's place and slot to its constants and its ok word held
    to 1 (`SALT_INPUT`, `CONSTANTS`): refused, `NotBound`, until C's format has the supplied input."""
    raise NotBound(f"{NOT_BOUND}{SALT_INPUT}; {CONSTANTS}")


class Undrawn(ValueError):
    """Proofs of other blocks than the proof layer's draw: a block it did not draw, a drawn block with no proof, or a
    finding naming another block than the drawn one it was offered for."""


def check(block: Callable[..., Any], verify: Callable[..., tuple[Any, Any]], proofs: Mapping[int, Any], c: bytes,
          partition: Mapping[str, Any], registered: Mapping[str, Any], draw: Any, params: P2Params,
          salt: bytes, row_bytes: int) -> list[Any]:
    """The door's findings on each block `k` of the proof layer's `draw` (`service.Draw`, drawn on its live coins after
    the store is committed: its ``units`` the blocks, its ``draw`` the sample object the door matches a proof to) of the
    registered C `c` and its `partition`: `proofs` exactly one a drawn block, else `Undrawn`; its units the partition's
    (`block`, SPOT's `one_circuit.block`), C's salt supplied (`bound`), and its proof `proofs[k]` verified against C,
    never the prover's circuit file (`verify`, SPOT's `proofs.verify`, `registered` the store's and W's registration),
    each finding naming block k. The door is the caller's, so the verifier loads none of SPOT. A partition or block the
    door refuses raises its refusal, a refusal of its verdict ValueError with its reason, and a session the door rejects
    ValueError naming it: a unit's ok word is held to 1 inside its proof, so an accepted session is every unit's 1."""
    drawn = list(draw.units)
    if sorted(proofs) != sorted(drawn):
        raise Undrawn(f"the proofs are of blocks {sorted(proofs)}, the proof layer drew {sorted(drawn)}")
    for k in drawn:
        block(c, partition, k)
    bound(params, salt, row_bytes)
    out: list[Any] = []
    for k in drawn:
        findings, why = verify(proofs[k], program=c, partition=partition, draw=draw.draw, registered=registered)
        if why is not None:
            raise ValueError(f"setup block {k}: the door refused ({why[0]}): {why[1]}")
        for f in findings or []:
            if f.block != k:
                raise Undrawn(f"setup block {k}: session {f.session} is of block {f.block}")
            if not f.accepted:
                raise ValueError(f"setup block {k}: session {f.session} rejected: {f.why}")
        out += findings or []
    return out


@dataclass(frozen=True)
class Claim:
    """What the setup's check reads in place of C*: `params`, the codec's manifest, and `commitment`, the server's
    `w_commitment ‖ salt`, as today's check reads them; `store`, the store's registered root (`window.commit_store`),
    which the memory attester signs as the session's binding; `law`, the proof layer's draw's (`Claim.of`); `outcome`,
    SPOT's outcome record of the sampled audit."""

    params: dict[str, Any]
    commitment: bytes
    store: bytes
    law: Mapping[str, Any]
    outcome: bytes

    @classmethod
    def of(cls, params: Mapping[str, Any], commitment: bytes, store: Mapping[str, Any], draw: Any,
           outcome: bytes) -> "Claim":
        """The claim of a check over `draw`: its law the draw's, its store the root of `store`, the store's registered
        entry as the verifier pinned it."""
        return cls(dict(params), commitment, bytes.fromhex(store["root"]), dict(draw.law), outcome)

    def to_json(self) -> dict[str, Any]:
        return {"params": dict(self.params), "commitment": self.commitment.hex(), "store": self.store.hex(),
                "law": dict(self.law), "outcome": self.outcome.hex()}

    @classmethod
    def from_json(cls, j: dict[str, Any]) -> "Claim":
        return cls(dict(j["params"]), bytes.fromhex(j["commitment"]), bytes.fromhex(j["store"]), dict(j["law"]),
                   bytes.fromhex(j["outcome"]))

    def encode(self) -> bytes:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()


__all__ = ["CONSTANTS", "Claim", "GOAL", "NOT_BOUND", "NotBound", "SALT_INPUT", "Undrawn", "bound", "check"]
