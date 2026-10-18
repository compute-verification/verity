import Definitions.Accounting.Attribution
import Models.Accounting.Assumptions.Attribution
import Models.Accounting.Assumptions.PouwFloor
import Models.Accounting.Assumptions.PouwWork

/-!
# Compute attribution: PoUW's credit at one block (trusted, not proved)

The physical layer's compute statement (decisions 111 and 113, `note:20261018T0730Z-draft-attrib`): PoUW's logical
certificate says the work was done; this says the block did it. A block is a spatial unit of the hardware
(`Hardware.SpatialUnit`: every link across its boundary has a certifier on it), its work is the actual circuit's gates
placed at its leaves in the window (`Accounting.PouwWork.credits`, `Placed.work` over `Placed.region`), and the
outsourcing term is the outside's advice about the window's salt (`Attribution.advice`), not the bytes that cross:

* the salt reaches the window only through the block (`Assumptions.SaltCustody`);
* the certifiers' records in the window take at most `2^krec` values across salts (`Assumptions.EgressAdvice`: N3,
  NET-5 and the coin-cone rule);
* B4 (`Accounting.Guarantees.B4Covered`, proved) then bounds what the outside does in the window to
  `2^{advice H krec ts td}` values at each draw;
* A3 at that advice (`Assumptions.PouwFloorAdvice`, conjectured) puts the work at the block.

It replaces `PouwLocalOff`'s isolation (help 0) with custody: a serving block takes in prompts, activations and
anything else that doesn't depend on the salt, and the statement charges none of it.

Placement is the actual circuit's: the statement quantifies over every strategy, since a prover that passes PoUW need
not run the declared circuit as placed. A declared placement (`Accounting.Requirement.Placement`, #2399) enters
through its run (`Placement.actual`, a strategy constant in the coins) and through the block: each spatial unit has its
own salt and its own verdict, whose blocks are the PoUW units the placement puts at the unit's leaves.

Not proved. The intended proof: B4Covered on each draw's strategy `fun s => S (s, d)` (the four premises restrict to
it), then `EgressAdvice` for `OutsideAdvice` at `advice`, then `PouwFloorAdvice` and the union bound of
`Accounting.SecurityProofs.PouwLocalOff`. Listed under `shape.frozen` as `unstated` until it is.
-/

namespace Accounting.Guarantees

open Accounting.PouwWork

/-- **PoUW's work lower bound at one block.** For any hardware `H`, any strategy `S` on coins split into the window's
salt and the later draw, any verdict `P`, any spatial unit `X` of `H` (the block), credit pricing `π` and marking `m`:
if the strategy is causal, the links are as declared and the certifiers are trusted (B4's premises), the salt reaches
the window only through the block, the certifiers' records in `[ts, td)` take at most `2^krec` values at each draw,
and PoUW's A2, A2b and A3 at `k = advice H krec ts td` bits of advice hold, then the verifier accepts while the block's
own gates outside `m` made fewer than `Wb · Nmin` credits in the window with probability at most
`δ_det + δ_luck + 2^k · δ_skip`. It says which spatial unit computed; within the unit, not which GPU. -/
def PouwBlockCredit : Prop :=
  ∀ {L E Ωs Ωd : Type} [LinearOrder L] [Fintype L] [Fintype E] [Fintype Ωs] [Fintype Ωd]
    (H : Hardware L E) (S : Ωs × Ωd → Actual.Run L) (P : Verdict (Ωs × Ωd) L) (X : Finset L)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (spec : ∀ r, (S r).A.Spec)
    (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N)) (Wb Nmin krec : ℕ) (δdet δluck δskip : ℝ),
    H.SpatialUnit X → CausalStrategy S H.ℓ → DeclaredLinks S H.ends H.β link →
    TrustedUnits S H.trusted H.cert spec H.record H.bucket H.jitter →
    Assumptions.SaltCustody S H.cert X P.ts → Assumptions.EgressAdvice S H.cert H.record P.ts P.td krec →
    Assumptions.PouwDetection S P Nmin δdet → Assumptions.PouwLuck S P Nmin δluck →
    Assumptions.PouwFloorAdvice S P X H.cert π m Wb Nmin (Attribution.advice H krec P.ts P.td) δskip →
    Pouw.pr (fun r => Accepts S P r ∧ credits S P X π m r < Wb * Nmin) ≤
      δdet + δluck + 2 ^ Attribution.advice H krec P.ts P.td * δskip

end Accounting.Guarantees
