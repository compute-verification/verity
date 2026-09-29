"""`ncp-v2`: NCP-INT's noised GEMM as a `verity.ir` Program, verified by sampled proofs (design note
`docs/pouw/sampled-proofs-circuit.md`; `PROTOCOL.md` §`ncp-v2`).

* `words`: PoUW's word primitives (every output at most 32 bits).
* `hashes`: SHA-512 and SHAKE256 in gates (`Sha512Compress`, `KeccakF1600`).
* `ncp2`: the unit templates and `linear(shape, kind)`, the Program of one linear call.
* `reference`: the native reference `linear` must evaluate to.
* `partition`: the `Q_template_instances` partition, the whole-call cut (no value computed in two units), the Z separator and
  the width rule with the separator port pinned by the protocol.
* `plan`: closure draws behind a draw-law seam (`CountLaw` today, `WorkLaw` when `Flock/Draw.lean` lands).

`traced_as(scheme, shape)` is what a composition (the vLLM integration's protocol options) calls: for a scheme whose circuit is
here, the Definition sampled proofs traces for one PoUW linear, its partition query, its served outputs and its draw layout;
None for any other scheme, which stays refused beside sampled proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: the schemes that have a circuit here
SCHEMES = ("ncp-v2",)


@dataclass(frozen=True)
class Traced:
    scheme: str
    shape: Any                  # `ncp2.Shape`
    definition: Any             # `NcpLinear_v1{M,K,N,PERM}`: its inputs are `reference.Call.args()`'s, in that order
    query: dict                 # `verity/partition/v1`'s query: `Q_template_instances` over the unit templates
    served: int                 # the first `served` root outputs are delivered (y); the credited words are committed only

    def layout(self):
        """The closure-draw layout (`plan.layout`)."""
        from . import plan

        return plan.layout(self.shape, self.definition.S().PERM)


def traced_as(scheme: str, m: int, k: int, n: int, kind: str = "shift24") -> Traced | None:
    """The circuit sampled proofs traces for one PoUW linear of shape m × k → n under ``scheme``, or None when the scheme has
    no circuit here. Raises `protocol.DomainError` for a shape outside the circuit (16 | m, 16 | n, 64 | k)."""
    if scheme not in SCHEMES:
        return None
    from . import ncp2, partition

    shape = ncp2.Shape(m, k, n)
    return Traced(scheme, shape, ncp2.linear(shape, kind), partition.query(shape, kind), m * n)


__all__ = ["SCHEMES", "Traced", "traced_as"]
