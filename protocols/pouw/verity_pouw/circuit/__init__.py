"""`ncp-v2`: NCP-INT's noised GEMM as a `verity.ir` Program, verified by sampled proofs (design note
`docs/pouw/sampled-proofs-circuit.md`; `PROTOCOL.md` §`ncp-v2`).

* `words`: PoUW's word primitives (every output at most 32 bits).
* `hashes`: SHA-512 and SHAKE256 in gates (`Sha512Compress`, `KeccakF1600`).
* `ncp2`: the unit templates and `linear(shape, kind)`, the Program of one linear call.
* `reference`: the native reference `linear` must evaluate to.
* `partition`: the `Q_template_instances` partition, the whole-call cut (no value computed in two units), the Z separator and
  the width rule with the separator port pinned by the protocol.
* `plan`: closure draws behind a draw-law seam (`CountLaw` today, `WorkLaw`, root's #362 law, when it merges).
* `fp8`, `fp8_words`, `hoist`, `fp8_reference`: NCP-FP8 (#295's D-3s, `fp8-is-h100-d3s-v0[-f16]`) under the same layout (a),
  its per-row hashes split so that their call-wide gates are computed once, and its reference, #295's scheme itself.

`traced_as(scheme, shape)` is what a composition (the vLLM integration's protocol options) calls: for a scheme whose circuit is
here, the Definition sampled proofs traces for one PoUW linear, its partition query, its served outputs and its draw layout;
None for any other scheme, which stays refused beside sampled proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: the schemes that have a circuit here: NCP-INT's, and NCP-FP8's two D-3s cells (f16 sums, #295's target, and f32 sums)
SCHEMES = ("ncp-v2", "fp8-is-h100-d3s-v0-f16", "fp8-is-h100-d3s-v0")
FP8_SUMS = {"fp8-is-h100-d3s-v0-f16": "f16", "fp8-is-h100-d3s-v0": "f32"}


@dataclass(frozen=True)
class Traced:
    scheme: str
    shape: Any                  # `ncp2.Shape`
    definition: Any             # `NcpLinear_v1{M,K,N,PERM}`: its inputs are `reference.Call.args()`'s, in that order
    query: dict                 # `verity/partition/v1`'s query: `Q_template_instances` over the unit templates
    served: int                 # the first `served` root outputs are delivered (y); the credited words are committed only
    lengths: Any = None         # NCP-FP8: `fp8.Lengths`, the call's integer encodings' byte lengths

    def layout(self):
        """The closure-draw layout (`plan.layout`, `plan.fp8_layout`)."""
        from . import plan

        if self.scheme in FP8_SUMS:
            return plan.fp8_layout(self.shape, FP8_SUMS[self.scheme], self.lengths)
        return plan.layout(self.shape, self.definition.S().PERM)


def traced_as(scheme: str, m: int, k: int, n: int, kind: str = "shift24", *, index: int = 0, weight: int = 0) -> Traced | None:
    """The circuit sampled proofs traces for one PoUW linear of shape m × k → n under ``scheme``, or None when the scheme has
    no circuit here. Raises `protocol.DomainError` for a shape outside the circuit (`ncp-v2`: 16 | m, 16 | n, 64 | k;
    NCP-FP8: 16 | m, 16 | n, 32 | k, k admissible, m, n <= 256). NCP-FP8's Definition depends on the byte lengths of the
    matmul's ``index`` and ``weight`` id (`fp8.Lengths`), which are its inputs' values."""
    if scheme not in SCHEMES:
        return None
    from . import partition

    if scheme in FP8_SUMS:
        from . import fp8

        shape, lengths = fp8.Shape(m, k, n), fp8.Lengths.of(index, weight)
        c = partition.fp8_circuit(shape, FP8_SUMS[scheme], lengths)
        return Traced(scheme, shape, c.fn, partition.query_of(c), m * n, lengths)
    from . import ncp2

    shape = ncp2.Shape(m, k, n)
    return Traced(scheme, shape, ncp2.linear(shape, kind), partition.query(shape, kind), m * n)


__all__ = ["FP8_SUMS", "SCHEMES", "Traced", "traced_as"]
