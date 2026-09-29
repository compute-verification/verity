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
* `fp8`, `fp8_words`, `hoist`, `fp8_reference`: NCP-FP8 (#295's D-3s, `fp8-is-h100-d3s-v0[-f16]`) under the same layout (a),
  its per-row hashes split so that their call-wide gates are computed once, and its reference, #295's scheme itself.

`traced_as(scheme, m, k, n)` is what a composition (the vLLM integration's protocol options) calls: for a scheme whose circuit
is here, the Programs sampled proofs traces for one PoUW linear and its weight, their partition queries, the served outputs
and the draw layout; None for any other scheme, which stays refused beside sampled proofs.  The verifier builds these
Programs itself, from (scheme, m, k, n) and its own anchors, and never takes a prover's descriptor: `Traced.check_descriptor`
compares an offered one's digest with its own (X-SPC-73).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: the schemes that have a circuit here, each with the parameter it pins: `ncp-v2`'s permutation P (as `ncp-v1` and
#: `ncp-v1-shift24` do), and NCP-FP8's D-3s sums (f16, #295's target cell, and f32)
NCP_KINDS = {"ncp-v2": "word", "ncp-v2-shift24": "shift24"}
FP8_SUMS = {"fp8-is-h100-d3s-v0-f16": "f16", "fp8-is-h100-d3s-v0": "f32"}
SCHEMES = {**NCP_KINDS, **FP8_SUMS}


@dataclass(frozen=True)
class Traced:
    scheme: str
    shape: Any                  # `ncp2.Shape` or `fp8.Shape`
    definition: Any             # `NcpLinear_v1{M,K,N,PERM}` or `NcpFp8Linear_v1{M,K,N,SUMS,IL,OL,WL}`
    query: dict                 # `verity/partition/v1`'s query: `Q_template_instances` over the unit templates
    served: int                 # the first `served` root outputs are delivered (y); the credited words are committed only
    weight: Any = None          # `NcpWeight_v1{K,N,PERM}`: the weight's Y strips and c₀, once per run and weight
    weight_query: dict | None = None
    lengths: Any = None         # NCP-FP8: `fp8.Lengths`, the call's integer encodings' byte lengths

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

    def call_args(self, anchors, *data: Sequence[int]) -> list:
        """The call's flat inputs: its committed ``data`` in parameter order (`ncp-v2`: a, s, ys, c0s, t, bias; NCP-FP8: a,
        s, b, t, bias), then its anchored ones, built from the verifier's ``anchors`` (`anchors.Anchors`,
        `anchors.Fp8Anchors`)."""
        from ..protocol import DomainError

        anchors.check(self.shape)
        if self.lengths is not None and anchors.lengths != self.lengths:
            raise DomainError(f"{self.scheme}: the anchors' index and weight id take {anchors.lengths}, not {self.lengths}")
        an = anchors.call_inputs()
        names = [p for p, _t in self.definition.signature()[0]]
        if sorted(names[len(data):]) != sorted(an):
            raise DomainError(f"{self.definition.id} takes {len(names) - len(an)} committed inputs, not {len(data)}")
        return [list(d) for d in data] + [an[p] for p in names[len(data):]]

    def weight_args(self, anchors, b: Sequence[int]) -> list:
        anchors.check(self.shape)
        an = anchors.weight_inputs()
        return [list(b), an["salt"], an["weight"], an["colids"]]

    def layout(self):
        """The closure-draw layout (`plan.layout`, `plan.fp8_layout`)."""
        from . import plan

        if self.scheme in FP8_SUMS:
            return plan.fp8_layout(self.shape, self.kind, self.lengths)
        return plan.layout(self.shape, self.kind)


def traced_as(scheme: str, m: int, k: int, n: int, *, index: int = 0, weight: int = 0) -> Traced | None:
    """The circuit sampled proofs traces for one PoUW linear of shape m × k → n under ``scheme``, or None when the scheme has
    no circuit here. Raises `protocol.DomainError` for a shape outside the circuit (`ncp-v2`: 16 | m, 16 | n, 64 | k;
    NCP-FP8: 16 | m, 16 | n, 32 | k, k admissible, m, n <= 2^24). The scheme pins its parameter, never the caller. NCP-FP8's
    Definition depends on the byte lengths of the matmul's ``index`` and ``weight`` id (`fp8.Lengths`), the verifier's
    anchors."""
    if scheme not in SCHEMES:
        return None
    from . import partition

    if scheme in FP8_SUMS:
        from . import fp8

        shape, lengths = fp8.Shape(m, k, n), fp8.Lengths.of(index, weight)
        c = partition.fp8_circuit(shape, FP8_SUMS[scheme], lengths)
        return Traced(scheme, shape, c.fn, partition.query_of(c), m * n, lengths=lengths)
    from . import ncp2

    kind, shape = NCP_KINDS[scheme], ncp2.Shape(m, k, n)
    return Traced(scheme, shape, ncp2.linear(shape, kind), partition.query(shape, kind), m * n, ncp2.weight(shape, kind),
                  partition.weight_query(shape, kind))


__all__ = ["FP8_SUMS", "NCP_KINDS", "SCHEMES", "Traced", "traced_as"]
