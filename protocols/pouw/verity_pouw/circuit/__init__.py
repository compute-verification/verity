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
#: H-1T's circuit (`h1t`, layout A) has no registered scheme yet, so `traced_h1t` names it
H1T = "h1t"


#: a call's inputs that are its weight Program's outputs, in that Program's output order
LINKED = ("ys", "c0s")


@dataclass(frozen=True)
class Traced:
    scheme: str
    shape: Any                  # `ncp2.Shape` or `h1t.Shape`
    definition: Any             # `NcpLinear_v1{M,K,N,PERM}` or `H1tLinear_v1{M,K,N}`
    query: dict                 # `verity/partition/v1`'s query: `Q_template_instances` over the unit templates
    served: int                 # the first `served` root outputs are delivered (y); the credited words are committed only
    weight: Any = None          # `NcpWeight_v1{K,N,PERM}`: the weight's Y strips and c₀, once per run and weight (H-1T: none)
    weight_query: dict | None = None

    @property
    def kind(self) -> str | None:
        return SCHEMES.get(self.scheme)

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
        """The call's flat inputs: its committed ``data`` in parameter order (`ncp-v2`: a, s, ys, c0s, t, bias; H-1T: a, s,
        b, t, bias), then its anchored ones, built from the verifier's ``anchors`` (`anchors.Anchors`,
        `anchors.H1tAnchors`)."""
        return self._args(self.definition, anchors, anchors.call_inputs, data)

    def weight_args(self, anchors, *data: Sequence[int]) -> list:
        """The weight Program's flat inputs: B, then its anchored ones (salt, weight id, column ids)."""
        return self._args(self.weight, anchors, anchors.weight_inputs, data)

    def _args(self, fn, anchors, inputs, data) -> list:
        from ..protocol import DomainError

        anchors.check(self.shape)
        an = inputs()
        names = [p for p, _t in fn.signature()[0]]
        if sorted(names[len(data):]) != sorted(an):
            raise DomainError(f"{fn.id} takes {len(names) - len(an)} committed inputs, not {len(data)}")
        return [list(d) for d in data] + [an[p] for p in names[len(data):]]

    def draw(self, key, law, anchors, ledger):
        """The verifier's plan for this call: `plan.draw` over its own layout (`layout`), never a given one (X-SPC-97), each
        stratum's subset keyed by the call's index (``anchors.call_index``) with its template (X-SPC-96). The call is first
        admitted to ``ledger``, the window's `anchors.Ledger` (one per window key), so a call index the window already
        holds is refused before anything is drawn. Each call of a window then draws under a context of its own: the
        product law root's window pin takes as M1."""
        from . import plan
        from .anchors import Ledger

        if not isinstance(ledger, Ledger):
            raise TypeError("a call is drawn only once its window's Ledger admits it")
        ledger.admit(anchors, self.shape)
        index = anchors.call_index
        return plan.draw(key, self.layout(), law, index if isinstance(index, tuple) else (index,))

    def check_link(self, plan, call_args: Sequence[Sequence[int]], weight_outputs: Sequence[int]) -> None:
        """Refuse a call whose weight-side inputs differ from the weight Program's outputs at a column strip the plan opens
        (its ("y", strip) leaves, the drawn tiles' column strips): the call's committed inputs opened against the weight
        Program's output commitment (X-SPC-95). The inputs are the call's parameters the weight Program returns, in its
        output order (`ncp-v2`: ys, c0s)."""
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
        """The closure-draw layout (`plan.layout`, `plan.h1t_layout`), from the shape alone."""
        from . import plan

        if self.scheme == H1T:
            return plan.h1t_layout(self.shape, self.definition.S().BIAS)
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


def traced_h1t(m: int, k: int, n: int, bias: bool = True) -> Traced:
    """H-1T's circuit for one linear of shape m × k → n, as the verifier builds it: the Definition, its partition query,
    the served outputs and the draw layout, from the shape alone. Raises `protocol.DomainError` outside the granted scope
    (layout A at 16 × 16 tiles: 16 | m, 16 | n, 8 | k, 29 < k <= 32,768). ``bias`` says whether the linear has one, which
    picks vLLM's epilogue (`h1t.H1tDequant`). `traced_as` returns it once H-1T is a registered scheme."""
    from . import h1t, partition

    shape = h1t.Shape(m, k, n)
    return Traced(H1T, shape, h1t.linear(shape, bias), partition.query_of(partition.h1t_circuit(shape, bias=bias)), m * n)


__all__ = ["H1T", "LINKED", "SCHEMES", "Traced", "traced_as", "traced_h1t"]
