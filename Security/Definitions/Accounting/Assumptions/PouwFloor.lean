import Definitions.Accounting.PouwWork

/-!
# PoUW's work floor on one node, an open conjecture (trusted)

A3 of the converged model (v3.1), over `Accounting.PouwWork`'s vocabulary on the actual circuit: a `Prop` about one
strategy `S` and the verifier `P`, taken as a hypothesis by `Accounting.Guarantees.PouwLocalOff`, never an `axiom`. It
is an open conjecture, so it is not under `shape.premises`: it becomes a premise under `shape.conjectures` once it has a
falsification test that runs. The other three premises are in `Definitions.Accounting.Assumptions.PouwWork`.
-/

namespace Accounting.Assumptions

open Accounting.PouwWork

variable {Ω : Type} [Fintype Ω] {L : Type*} (S : Ω → Actual.Run L) (P : Verdict Ω L)

/-- **A3 the work floor, off a skip event** (v3.1's `FloorOff`; on paper at `L = 32` given a merge price, conjectured at
served η): for the marking `m`, some event `Z` of probability at most `δ` is such that at every coin outside `Z`, `Nmin`
blocks right at every coin force the node's credits outside `m`, plus `ρ` a bit of help, to at least `Wb · Nmin`.

`Z` covers a guarded circuit that is right at every coin yet skips credited gates at some. The floor is meant at a
removable marking: one whose erasure leaves the verdict and the right blocks unchanged at every coin (the converged
model's `Removable`), so that covert work earns no credit. The actual-circuit model has no erasure of gates, so the
marking is taken as given, and the premise is claimed only at markings that are removable. A factor-2 projection form
(`#unmixed + 2·#mixed ≥ floor`) is being proved as a bound that implies this one. -/
def PouwFloorOff (X : Finset L) (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N))
    (Wb Nmin ρ : ℕ) (δ : ℝ) : Prop :=
  ∃ Z : Ω → Prop, Pouw.pr Z ≤ δ ∧
    ∀ r, ¬ Z r → Nmin ≤ goodAll S P → Wb * Nmin ≤ credits S P X π m r + ρ * help S P X r

end Accounting.Assumptions
