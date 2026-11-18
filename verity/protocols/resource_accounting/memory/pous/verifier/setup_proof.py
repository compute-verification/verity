"""PoUS's setup as a sampled proof about the store, the auditor's side: what its check takes in place of the store.

Today's setup check (`setup.verifier_key_from_encoding`, `PousCheck.SetupProtocol`) reads all of C* and decodes it.  In
proofs it reads none of it. The setup is one circuit C (`verity/protocols/resource_accounting/memory/pous/prover/window.py`,
`Setup`): ``PousP2SetupCheck_v1`` over every block of the store, one proof block a store block, its blocks and W's rows
registered values. The prover commits the store and W, then the setup's values; the proof system samples `SAMPLE`
blocks on its live coins after that (`sampling_design`), and `check` reads the door's verdict on each sampled block
(SPOT's `proofs.verify` on C and the partition, the block's units the partition's, the caller's door), with no list of positions of its own. The
claim (`Claim`) carries the parameters and commitment as today, the store's root and SPOT's outcome to the Lean check
`PousCheck.SetupInProofs.check`.

Two places stop it. C's base and slot are input nodes, and the base is a function of the verifier's salt and the slot
of the unit's position: no input kind binds an input node to such a function yet (circuits' format, Core's side lean's),
so `bound` refuses (`NotBound`). And the door's statement of a block is not derived (`one_circuit.lower`, proofs').
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schemes.p2 import P2Params

GOAL = "memory-accounting/pous-setup-as-proof"
#: the blocks the challenger samples: a uniform 8,828-subset misses ⌊B/100⌋ + 1 wrong blocks with probability at most
#: 0.99^8828 ≤ 2^-128, at any store size
SAMPLE = 8828


def sampling_design(blocks: int) -> str | None:
    """The sampling design over a store of `blocks` blocks: `rate:SAMPLE/B`, which samples `SAMPLE` of them; None, every
    block, when there are no more than `SAMPLE`."""
    if not (isinstance(blocks, int) and not isinstance(blocks, bool) and blocks >= 1):
        raise ValueError(f"a store has a positive number of blocks, not {blocks!r}")
    return None if blocks <= SAMPLE else f"rate:{SAMPLE}/{blocks}"


def base(prefix: bytes, blocks_per_segment: int, u: int) -> bytes:
    """Block `u`'s expansion base, `prefix ‖ u64be(s) ‖ u64be(v)`: block v of segment s."""
    s, v = divmod(u, blocks_per_segment)
    return prefix + s.to_bytes(8, "big") + v.to_bytes(8, "big")


#: where C's input nodes wait on circuits' coin-bound input kind
NOT_BOUND = "pous setup: no coin-bound input kind yet: "
BASE_INPUT = ("base(u) = expand_prefix(params, salt) || u64be(u / S) || u64be(u mod S), the salt the verifier's coin after "
              "W is committed")
SLOT_INPUT = "slot(u) = u mod n, of unit u's position in C"


class NotBound(ValueError):
    """C's base and slot inputs are bound to nothing: an input node bound to a named function of the verifier's coins
    or of a unit's position is circuits' input kind, which isn't in C's format yet."""


def bound(params: P2Params, salt: bytes, row_bytes: int) -> None:
    """C's base and slot input nodes bound to `params`' expansion of `salt` and to each unit's position (`BASE_INPUT`,
    `SLOT_INPUT`): refused, `NotBound`, until C's format has an input kind that binds them."""
    raise NotBound(f"{NOT_BOUND}{BASE_INPUT}; {SLOT_INPUT}")


def check(block: Callable[..., Any], verify: Callable[..., tuple[Any, Any]], proofs: Mapping[int, Any], c: bytes,
          partition: Mapping[str, Any], registered: Mapping[str, Any], sample: Sequence[int], params: P2Params,
          salt: bytes, row_bytes: int) -> list[Any]:
    """The door's findings on each sampled block `k` of the registered C `c` and its `partition` (`sample`, the proof
    system's, sampled after the store is committed): its units the partition's (`block`, SPOT's `one_circuit.block`), C's
    base and slot bound (`bound`), and its proof `proofs[k]` verified against C, never the prover's circuit file
    (`verify`, SPOT's `proofs.verify`, `registered` the store's and W's registration). The door is the caller's, so the
    verifier loads none of SPOT. A partition or block the door refuses raises its refusal, and a refusal of its verdict
    ValueError with its reason."""
    for k in sample:
        block(c, partition, k)
    bound(params, salt, row_bytes)
    out: list[Any] = []
    for k in sample:
        findings, why = verify(proofs[k], program=c, partition=partition, block=k, registered=registered)
        if why is not None:
            raise ValueError(f"setup block {k}: the door refused ({why[0]}): {why[1]}")
        out += findings or []
    return out


@dataclass(frozen=True)
class Claim:
    """What the setup's check reads in place of C*: `params`, the codec's manifest, and `commitment`, the server's
    `w_commitment ‖ salt`, as today's check reads them; `store`, the store's registered root; `law`, the sampling design
    (`sampling_design`); `outcome`, SPOT's outcome record of the sampled audit."""

    params: dict[str, Any]
    commitment: bytes
    store: bytes
    law: str | None
    outcome: bytes

    def to_json(self) -> dict[str, Any]:
        return {"params": dict(self.params), "commitment": self.commitment.hex(), "store": self.store.hex(),
                "law": self.law, "outcome": self.outcome.hex()}

    @classmethod
    def from_json(cls, j: dict[str, Any]) -> "Claim":
        return cls(dict(j["params"]), bytes.fromhex(j["commitment"]), bytes.fromhex(j["store"]), j["law"],
                   bytes.fromhex(j["outcome"]))

    def encode(self) -> bytes:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()


__all__ = ["BASE_INPUT", "Claim", "GOAL", "NOT_BOUND", "NotBound", "SAMPLE", "SLOT_INPUT", "base", "bound", "check",
           "sampling_design"]
