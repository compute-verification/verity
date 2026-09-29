"""`ncp-v2`: NCP-INT's noised GEMM as a `verity.ir` Program, verified by sampled proofs (design note
`docs/pouw/sampled-proofs-circuit.md`; `PROTOCOL.md` §`ncp-v2`).

* `words`: PoUW's word primitives (every output at most 32 bits).
* `hashes`: SHA-512 and SHAKE256 in gates (`Sha512Compress`, `KeccakF1600`).
* `ncp2`: the unit templates; `weight(shape, kind)`, a weight's Y units once per run; `linear(shape, kind)`, one call's
  Program, reading them; `window(shape, calls, kind)`, calls on one weight in one root.
* `reference`: the native reference those must evaluate to.
* `partition`: the `Q_template_instances` partition, the whole-call cut (no value computed in two units), the Z separator and
  the width rule with the separator port pinned by the protocol.
* `anchors`: the anchored inputs' anchors, which the verifier builds, and the window's refusal of a repeated call index.
* `plan`: closure draws behind a draw-law seam (`CountLaw`, and `WorkLaw`, root's #362 law).

`traced_as(scheme, m, k, n)` is what a composition (the vLLM integration's protocol options) calls: for a scheme whose circuit
is here, the Programs sampled proofs traces for one PoUW linear and its weight, their partition queries, the served outputs
and the draw layout; None for any other scheme, which stays refused beside sampled proofs.  The verifier builds these
Programs itself, from (scheme, m, k, n) and its own anchors, and never takes a prover's descriptor: `Traced.check_descriptor`
compares an offered one's digest with its own (X-SPC-73).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: the schemes that have a circuit here, each with the permutation P it pins (as `ncp-v1` and `ncp-v1-shift24` do)
SCHEMES = {"ncp-v2": "word", "ncp-v2-shift24": "shift24"}


@dataclass(frozen=True)
class Traced:
    scheme: str
    shape: Any                  # `ncp2.Shape`
    definition: Any             # `NcpLinear_v1{M,K,N,PERM}`
    query: dict                 # `verity/partition/v1`'s query: `Q_template_instances` over the unit templates
    served: int                 # the first `served` root outputs are delivered (y); the credited words are committed only
    weight: Any                 # `NcpWeight_v1{K,N,PERM}`: the weight's Y strips and c₀, once per run and weight
    weight_query: dict

    @property
    def kind(self) -> str:
        return SCHEMES[self.scheme]

    def digest(self) -> str:
        """The call Program's digest, as the verifier builds it."""
        from verity.ir.codec import program_digest
        from verity.ir.program import Program

        return program_digest(Program(self.definition))

    def weight_digest(self) -> str:
        from verity.ir.codec import program_digest
        from verity.ir.program import Program

        return program_digest(Program(self.weight))

    def check_descriptor(self, descriptor: dict) -> None:
        """Refuse a descriptor that is not the verifier's own call Program."""
        from verity.ir.codec import decode_program, program_digest
        from verity.ir.defs import REGISTRY

        from ..protocol import DomainError

        if program_digest(decode_program(descriptor, REGISTRY)) != self.digest():
            raise DomainError(f"{self.scheme}: the descriptor is not the verifier's Program for {self.shape}")

    def call_args(self, anchors, a: Sequence[int], s: Sequence[int], ys: Sequence[int], c0s: Sequence[int],
                  t: Sequence[int], bias: Sequence[int]) -> list:
        """The call's flat inputs, its anchored ones built from the verifier's ``anchors`` (`anchors.Anchors`)."""
        anchors.check(self.shape)
        an = anchors.call_inputs()
        return [list(a), list(s), list(ys), list(c0s), list(t), list(bias), an["salt"], an["forward"], an["position"],
                an["weight"], an["rowids"]]

    def weight_args(self, anchors, b: Sequence[int]) -> list:
        anchors.check(self.shape)
        an = anchors.weight_inputs()
        return [list(b), an["salt"], an["weight"], an["colids"]]

    def check_layout(self, offered) -> None:
        """Refuse a layout that isn't the verifier's own, derived from the shape (`layout`): the tiles, each tile's own unit
        and its closure are the partition's, never a prover's, since the Layout theorems hold only for the window's own."""
        from ..protocol import DomainError

        if offered != self.layout():
            raise DomainError(f"{self.scheme}: the layout is not the verifier's for {self.shape}")

    def layout(self):
        """The closure-draw layout (`plan.layout`), from the shape alone."""
        from . import plan

        return plan.layout(self.shape, self.kind)


def traced_as(scheme: str, m: int, k: int, n: int) -> Traced | None:
    """The circuit sampled proofs traces for one PoUW linear of shape m × k → n under ``scheme``, or None when the scheme has
    no circuit here. Raises `protocol.DomainError` for a shape outside the circuit (16 | m, 16 | n, 64 | k). P is the
    scheme's, never the caller's."""
    if scheme not in SCHEMES:
        return None
    from . import ncp2, partition

    kind, shape = SCHEMES[scheme], ncp2.Shape(m, k, n)
    return Traced(scheme, shape, ncp2.linear(shape, kind), partition.query(shape, kind), m * n, ncp2.weight(shape, kind),
                  partition.weight_query(shape, kind))


__all__ = ["SCHEMES", "Traced", "traced_as"]
