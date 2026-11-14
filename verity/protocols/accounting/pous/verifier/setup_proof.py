"""PoUS's setup as a sampled proof about the store, the auditor's side: what its check takes in place of the store.

Today's setup check (`setup.verifier_key_from_encoding`, `PousCheck.SetupProtocol`) reads all of C* and decodes it.  In
proofs it reads none of it: the store's blocks are committed and their root registered, the challenger samples
`SAMPLE` blocks on its live coins after that (`law`), each sampled block is proved to decode to the registered rows of W
at its place (`verity/protocols/accounting/pous/prover/setup_proof.py`, one instance of `PousP2SetupCheck_v1` a block),
and the check takes the `Claim`: the parameters and commitment as today, the store's root, the unit's Program digest
and SPOT's outcome.  The Lean check `PousCheck.SetupInProofs.check` decides; Python only carries the claim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

GOAL = "memory-accounting/pous-setup-as-proof"
#: the blocks the challenger samples: a uniform 8,828-subset misses ⌊B/100⌋ + 1 wrong blocks with probability at most
#: 0.99^8828 ≤ 2^-128, at any store size
SAMPLE = 8828


def law(blocks: int) -> str | None:
    """The sampling design over a store of `blocks` blocks: `rate:SAMPLE/B`, which samples `SAMPLE` of them; None, every
    block, when there are no more than `SAMPLE`."""
    if not (isinstance(blocks, int) and not isinstance(blocks, bool) and blocks >= 1):
        raise ValueError(f"a store has a positive number of blocks, not {blocks!r}")
    return None if blocks <= SAMPLE else f"rate:{SAMPLE}/{blocks}"


@dataclass(frozen=True)
class Claim:
    """What the setup's check reads in place of C*: `params`, the codec's manifest, and `commitment`, the server's
    `w_commitment ‖ salt`, as today's check reads them; `store`, the store's registered root; `unit`, the setup unit's
    Program digest, which the auditor pins; `law`, the sampling design (`law`); `outcome`, SPOT's outcome record of the
    sampled audit."""

    params: dict[str, Any]
    commitment: bytes
    store: bytes
    unit: str
    law: str | None
    outcome: bytes

    def to_json(self) -> dict[str, Any]:
        return {"params": dict(self.params), "commitment": self.commitment.hex(), "store": self.store.hex(),
                "unit": self.unit, "law": self.law, "outcome": self.outcome.hex()}

    @classmethod
    def from_json(cls, j: dict[str, Any]) -> "Claim":
        return cls(dict(j["params"]), bytes.fromhex(j["commitment"]), bytes.fromhex(j["store"]), j["unit"], j["law"],
                   bytes.fromhex(j["outcome"]))

    def encode(self) -> bytes:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()


__all__ = ["Claim", "GOAL", "SAMPLE", "law"]
