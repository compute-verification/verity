"""PoUS's setup as a sampled proof about the store, the auditor's side: what its check takes in place of the store.

Today's setup check (`setup.verifier_key_from_encoding`, `PousCheck.SetupProtocol`) reads all of C* and decodes it.  In
proofs it reads none of it: the store's blocks are committed and their root registered, the challenger samples
`SAMPLE` blocks on its live coins after that (`law`), each sampled block is proved to decode to the registered rows of W
at its place (`verity/protocols/accounting/pous/prover/window.py`, one instance of `PousP2SetupCheck_v1` a block),
and the check takes the `Claim`: the parameters and commitment as today, the store's root, the unit's Program digest
and SPOT's outcome.  The Lean check `PousCheck.SetupInProofs.check` decides; Python only carries the claim.

Each sampled unit's base and slot are rows the verifier makes from its own `expand_prefix` and the sample (`supplied`),
which SPOT's one-stage verifier takes as its ``supplied`` values (`BASE`, `SLOT`), so a sampled block is tied to its
index.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .schemes.p2 import P2Params, expand_prefix

GOAL = "memory-accounting/pous-setup-as-proof"
#: the supplied values a sampled unit reads: its expansion base and its slot in its row of W
BASE, SLOT = "setup/base", "setup/slot"
#: the blocks the challenger samples: a uniform 8,828-subset misses ⌊B/100⌋ + 1 wrong blocks with probability at most
#: 0.99^8828 ≤ 2^-128, at any store size
SAMPLE = 8828


def law(blocks: int) -> str | None:
    """The sampling design over a store of `blocks` blocks: `rate:SAMPLE/B`, which samples `SAMPLE` of them; None, every
    block, when there are no more than `SAMPLE`."""
    if not (isinstance(blocks, int) and not isinstance(blocks, bool) and blocks >= 1):
        raise ValueError(f"a store has a positive number of blocks, not {blocks!r}")
    return None if blocks <= SAMPLE else f"rate:{SAMPLE}/{blocks}"


def base(prefix: bytes, blocks_per_segment: int, u: int) -> bytes:
    """Block `u`'s expansion base, `prefix ‖ u64be(s) ‖ u64be(v)`: block v of segment s."""
    s, v = divmod(u, blocks_per_segment)
    return prefix + s.to_bytes(8, "big") + v.to_bytes(8, "big")


def leaf_bytes(data: bytes) -> bytes:
    """The bytes whose 16-bit leaves, as SPOT reads a supplied row (`one_stage.registered.bytes_leaves`: each big-endian
    32-bit half, its two leaves least significant first), are `data`'s little-endian 16-bit words in order, the words
    the unit's port holds: each 4-byte group reversed."""
    if len(data) % 4:
        raise ValueError(f"{len(data)} bytes are no whole 32-bit halves")
    return b"".join(data[i:i + 4][::-1] for i in range(0, len(data), 4))


def supplied(params: P2Params, salt: bytes, row_bytes: int, sample: Sequence[int]) -> dict[str, list[list[dict[str, str]]]]:
    """The rows the verifier makes for the sampled units, in sample order, as SPOT's one-stage verifier takes its
    ``supplied`` values: each unit's base (`base`, from the verifier's own `expand_prefix`) as one ``{"bytes": hex}``
    source of its port's words (`leaf_bytes`), and its slot u mod n, n the ⌈N/P⌉ slots of a row of `row_bytes` bytes, as
    one big-endian 16-bit leaf."""
    prefix, n = expand_prefix(params, salt), -(-row_bytes // params.payload_bytes)
    return {BASE: [[{"bytes": leaf_bytes(base(prefix, params.blocks_per_segment, u)).hex()}] for u in sample],
            SLOT: [[{"bytes": f"{u % n:04x}"}] for u in sample]}


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


__all__ = ["BASE", "Claim", "GOAL", "SAMPLE", "SLOT", "base", "law", "leaf_bytes", "supplied"]
