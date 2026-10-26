"""`ncp-v2`: NCP-INT's noised GEMM as a `verity.ml.circuits` Program, verified by sampled proofs
(`PROTOCOL.md` §`ncp-v2`).

* `words`: PoUW's word primitives (every output at most 32 bits but `pc8`'s exact squares, which stay inside a row unit).
* `boolean`: the six of them `pc4`'s and `pc8`'s row units call, and the four Keccak words their hiding tiles' digest calls,
  as Boolean Definitions (`_v2`, the same functions on bits).
* `hashes`: SHA-512, SHAKE256 and TurboSHAKE128 in gates (`Sha512Compress`, `KeccakF1600`, `KeccakP1600R12`).
* `leaves`: the tile checks' committed row schemas (C-Flock's ports), Pearl-C's tile digest in gates, and the rows' native
  reference.
* `ncp2`: the unit templates; `weight(shape, kind)`, a weight's Y units once per run; `linear(shape, kind)`, one call's
  Program, reading them; `window(shape, calls, kind)`, calls on one weight in one root.
* `reference`: the native reference those must evaluate to.
* `partition`: the `Q_template_instances` partition, the whole-call cut (no value computed in two units), the Z separator and
  the width rule with the separator port pinned by the protocol.
* `anchors`: the anchored inputs' anchors, which the verifier builds (the tile checks' derived noise too, `TileAnchors`, with
  each unit's public input regions), and the window's refusal of a repeated call index.
* `plan`: closure draws behind a draw-law seam (`CountLaw`, and `WorkLaw`, root's #362 law), and the tile checks' layout.
* `pc4`, `pc8`: Pearl-C4's and Pearl-C's (FP8, forming v1) tile checks as Programs, cut into row and tile units, and
  their hiding Programs (``Shape.hidden``), whose tiles return only their tile digest; the rows are C-Flock-committed
  ports and the anchored inputs public input regions (`PORTS`).  No scheme of theirs is in `SCHEMES` yet.
* `rowk`: their row units cut along k into a head, one form per segment and a tail (``Shape.row_k``, opt-in), so each
  piece fits C-Flock's block.
* `compile`: PoUW as one pass over a served Program (decision 102, `PROTOCOL.md` §`compile`): `compile(program, scheme,
  groups)` replaces every registered matmul's dots (`MATMULS`) with the scheme's r × c tile units, each deriving its seed
  and noise in its own proof from the one coin, the salt (decisions 105, 106), and checks the result; `rebuild` is the
  verifier's side. Its schemes are its own `SCHEMES`, not this module's.
* `seeds`: the seed and the noise blocks a unit derives (`PouwSeed`, `PouwNoiseBlock`, on words and on bits), and its two
  seeds where its receipt folds take keys (`PouwUnitSeeds`: seed_j and σ_R, decision 121).
* `h1tile`: `eps1-sm120-h1fx1` as a compile-pass scheme, H1 + fx1 on a warp's tile under CP-W (decision 104).
* `h1device`: H1's device records, one an architecture (sm_120, and sm_90 for the H200): the s8 step and its binding, the
  contractions its kernels serve, and its data-sheet facts.
* `h1zy`: the H1 tile with its served z/y record checked (`PouwH1TileZy`, RT4-INT-33): the record an input, committed
  before its round's coin, its z the tile's A Bᵀ and its y the served dequant (`PouwZyDequant`, the activation scale
  first, no bias).
* `h1window`: an H1 window as one public Program of those tiles in fixed slots (`PouwH1Window`), the tile R × C a
  parameter and one tile template per distinct K (`PouwH1WindowTile`, its A and B rows ``a_k<K>`` and ``b_k<K>``),
  each linear's B rows registered once for all its calls; its template-instance partition derives its units, and
  `unit_reads` the rows each unit reads: the Program a served path registers its trees under, a mixed-K window in one
  read.
* `admit`: a window's admission from its committed request queue (`PouwQueueAdmit`), a call's 16 slots a unit, generic over
  the window.
* `live`: a round's served rows under rule A (`PouwLiveSlots`), one unit a round: ok iff exactly its slots [0, m) name
  requests of the gate-signed request table, none more often than it can take in round s (its prompt at its prefill, one
  row a decode), for the public m that dense credit counts.

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


#: a call's inputs that are its weight Program's outputs, in that Program's output order
LINKED = ("ys", "c0s")


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
        from verity.ml.circuits.codec import program_digest
        from verity.ml.circuits.program import Program

        return program_digest(Program(self.definition))

    def weight_digest(self) -> str:
        from verity.ml.circuits.codec import program_digest
        from verity.ml.circuits.program import Program

        return program_digest(Program(self.weight))

    def check_descriptor(self, descriptor: dict) -> None:
        """Refuse a descriptor that is not the verifier's own call Program."""
        from verity.ml.circuits.codec import decode_program, program_digest
        from verity.ml.circuits.defs import REGISTRY

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
        ledger.admit(anchors, self.shape)
        index = anchors.call_index
        return plan.draw(ledger.key, self.layout(), law, index if isinstance(index, tuple) else (index,))

    def check_openings(self, plan, anchors, ledger, domain, openings, limits=None) -> None:
        """Refuse the call's A-row openings unless they are the plan's A rows, once each, and each authenticates under
        the call's commitment as the window's receipt lists it (`Ledger.commitment`, by call index), never one supplied
        with the openings. ``domain`` is the verifier's row domain for the call (`audit.rows_domain`)."""
        from verity.ml.commitments import Commitment, VerificationLimits, verify_opening

        from ..audit import ROW_SCHEMA
        from ..protocol import DomainError

        root = ledger.commitment(anchors)
        if sorted(o.position for o in openings) != sorted(r for kind, r in plan.openings if kind == "a"):
            raise DomainError("the openings are not the plan's A rows, once each")
        commitment, limits = Commitment(root, domain.count), VerificationLimits() if limits is None else limits
        if not all(verify_opening(domain, commitment, o, ROW_SCHEMA, limits) for o in openings):
            raise DomainError("an A row does not open under the call's commitment on the window's receipt")

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


__all__ = ["LINKED", "SCHEMES", "Traced", "traced_as"]
