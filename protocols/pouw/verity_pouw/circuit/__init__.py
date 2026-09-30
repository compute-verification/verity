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


#: a call's inputs that are its weight Program's outputs, in that Program's output order (`ncp-v2`: its Y strips and c₀;
#: NCP-FP8: its h and amplitudes)
LINKED = ("ys", "c0s", "hs", "amps")


@dataclass(frozen=True)
class Traced:
    scheme: str
    shape: Any                  # `ncp2.Shape` or `fp8.Shape`
    definition: Any             # `NcpLinear_v1{M,K,N,PERM}` or `NcpFp8Linear_v1{M,K,N,SUMS,IL,OL,WL}`
    query: dict                 # `verity/partition/v1`'s query: `Q_template_instances` over the unit templates
    served: int                 # the first `served` root outputs are delivered (y); the credited words are committed only
    weight: Any = None          # `NcpWeight_v1{K,N,PERM}` (the weight's Y strips and c₀) or `NcpFp8Weight_v1{K,N,SUMS}`
                                # (its h and amplitudes): once per run and weight
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
        s, hs, amps, t, bias), then its anchored ones, built from the verifier's ``anchors`` (`anchors.Anchors`,
        `anchors.Fp8Anchors`)."""
        return self._args(self.definition, anchors, anchors.call_inputs, data)

    def weight_args(self, anchors, *data: Sequence[int]) -> list:
        """The weight Program's flat inputs: B, then its anchored ones (`ncp-v2`: salt, weight id, column ids)."""
        return self._args(self.weight, anchors, anchors.weight_inputs, data)

    def _check_anchors(self, anchors) -> None:
        from ..protocol import DomainError

        anchors.check(self.shape)
        if self.lengths is not None and (anchors.lengths, anchors.sums) != (self.lengths, self.kind):
            raise DomainError(f"{self.scheme}: the anchors take {anchors.lengths}, {anchors.sums} sums, not {self.lengths}")

    def _args(self, fn, anchors, inputs, data) -> list:
        from ..protocol import DomainError

        self._check_anchors(anchors)
        an = inputs()
        names = [p for p, _t in fn.signature()[0]]
        if sorted(names[len(data):]) != sorted(an):
            raise DomainError(f"{fn.id} takes {len(names) - len(an)} committed inputs, not {len(data)}")
        return [list(d) for d in data] + [an[p] for p in names[len(data):]]

    def draw(self, law, anchors, ledger):
        """The verifier's plan for this call: `plan.draw` over its own layout (`layout`), never a given one (X-SPC-97), each
        stratum's subset keyed by the call's index (``anchors.call_index``) with its template (X-SPC-96), under the key of
        ``ledger``, the window's `anchors.Ledger` opened from its complete `Receipt` (`Ledger.open`): no key is taken from
        the caller. The draw is refused for a ledger not opened from a receipt or whose draw was sent, and for a call the
        receipt doesn't list or the window already holds, before anything is drawn. So every call's commitment precedes
        the key (C1), and each call of a window draws under a context of its own: the product law root's window pin takes
        as M1."""
        from . import plan
        from .anchors import Ledger

        if not isinstance(ledger, Ledger):
            raise TypeError("a call is drawn only under its window's Ledger")
        ledger.check_draw()
        self._check_anchors(anchors)
        ledger.admit(anchors, self.shape)
        index = anchors.call_index
        return plan.draw(ledger.key, self.layout(), law, index if isinstance(index, tuple) else (index,))

    def check_openings(self, plan, anchors, ledger, domain, openings, limits=None) -> None:
        """Refuse the call's A-row openings unless they are the plan's A rows, once each, and each authenticates under
        the call's commitment as the window's receipt lists it (`Ledger.commitment`, by call index), never one supplied
        with the openings. ``domain`` is the verifier's row domain for the call (`audit.rows_domain`)."""
        from verity.commitments import Commitment, VerificationLimits, verify_opening

        from ..audit import ROW_SCHEMA
        from ..protocol import DomainError

        root = ledger.commitment(anchors)
        if sorted(o.position for o in openings) != sorted(r for kind, r in plan.openings if kind == "a"):
            raise DomainError("the openings are not the plan's A rows, once each")
        commitment, limits = Commitment(root, domain.count), VerificationLimits() if limits is None else limits
        if not all(verify_opening(domain, commitment, o, ROW_SCHEMA, limits) for o in openings):
            raise DomainError("an A row does not open under the call's commitment on the window's receipt")

    def widen(self, unit_draw, law):
        """The verifier's plan for a work draw its verifier of record issued (`plan.widen`), over its own layout, never a
        given one, and under its own law (K_c and floors), never the draw's (X-SPC-114, 115)."""
        from . import plan

        return plan.widen(self.layout(), unit_draw, law)

    def check_link(self, plan, call_args: Sequence[Sequence[int]], weight_outputs: Sequence[int]) -> None:
        """Refuse a call whose weight-side inputs differ from the weight Program's outputs at a column strip the plan opens
        (its ("y", strip) leaves, the drawn tiles' column strips): the call's committed inputs opened against the weight
        Program's output commitment (X-SPC-95, X-SPC-101). The inputs are the call's parameters the weight Program returns,
        in its output order (`ncp-v2`: ys, c0s; NCP-FP8: hs, amps)."""
        from ..protocol import DomainError

        params = self.definition.signature()[0]
        names = [p for p, _t in params]
        linked = [(names.index(p), t) for p, t in params if p in LINKED]
        strips = [cs for role, cs in plan.openings if role == "y"]
        at = 0
        for i, t in linked:
            per = t.elem.leaves
            for cs in strips:
                if list(call_args[i][cs * per:(cs + 1) * per]) != list(weight_outputs[at + cs * per:at + (cs + 1) * per]):
                    raise DomainError(f"{self.scheme}: the call's {names[i]} at column strip {cs} is not the weight Program's")
            at += t.n * per

    def check_layout(self, offered) -> None:
        """Refuse a layout that isn't the verifier's own, derived from the shape (`layout`): the tiles, each tile's own unit
        and its closure are the partition's, never a prover's, since the Layout theorems hold only for the window's own."""
        from ..protocol import DomainError

        if offered != self.layout():
            raise DomainError(f"{self.scheme}: the layout is not the verifier's for {self.shape}")

    def layout(self):
        """The closure-draw layout (`plan.layout`, `plan.fp8_layout`), from the shape (and NCP-FP8's lengths) alone."""
        from . import plan

        if self.scheme in FP8_SUMS:
            return plan.fp8_layout(self.shape, self.kind, self.lengths, self.definition.S().BIAS)
        return plan.layout(self.shape, self.kind)


def traced_as(scheme: str, m: int, k: int, n: int, *, index: int = 0, weight: int = 0, bias: bool = True) -> Traced | None:
    """The circuit sampled proofs traces for one PoUW linear of shape m × k → n under ``scheme``, or None when the scheme has
    no circuit here. Raises `protocol.DomainError` for a shape outside the circuit (`ncp-v2`: 16 | m, 16 | n, 64 | k;
    NCP-FP8: 16 | m, 16 | n, 32 | k, k admissible, m, n <= 2^24). The scheme pins its parameter, never the caller. NCP-FP8's
    Definition depends on the byte lengths of the matmul's ``index`` and ``weight`` id (`fp8.Lengths`), the verifier's
    anchors, and on whether the linear has a ``bias``, which picks vLLM's epilogue (`fp8.NcpFp8Dequant`)."""
    if scheme not in SCHEMES:
        return None
    from . import partition

    if scheme in FP8_SUMS:
        from . import fp8

        shape, lengths = fp8.Shape(m, k, n), fp8.Lengths.of(index, weight)
        sums = FP8_SUMS[scheme]
        c = partition.fp8_circuit(shape, sums, lengths, bias=bias)
        return Traced(scheme, shape, c.fn, partition.query_of(c), m * n, fp8.weight(shape, sums),
                      partition.fp8_weight_query(shape, sums), lengths)
    from . import ncp2

    kind, shape = NCP_KINDS[scheme], ncp2.Shape(m, k, n)
    return Traced(scheme, shape, ncp2.linear(shape, kind), partition.query(shape, kind), m * n, ncp2.weight(shape, kind),
                  partition.weight_query(shape, kind))


__all__ = ["FP8_SUMS", "LINKED", "NCP_KINDS", "SCHEMES", "Traced", "traced_as"]
