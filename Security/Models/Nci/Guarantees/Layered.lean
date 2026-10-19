import Models.Nci.Guarantees.Threads
import Definitions.Nci.Layered

/-!
# NCI over threads: keyed circuits are threaded

The premise of the thread draft's Theorems 1 and 2 (`Threading.Threaded`) holds for every keyed reference circuit
(`Layered`), at every `Γ` and at every `S` below the width of `E + 1` lookup columns, with `(d, e)` read off the
circuit's structure.
-/

namespace Nci.Guarantees

open Verity

/-- **Keyed circuits are threaded**: a keyed reference circuit whose roots have `d` legs, whose values each reach at
most `e` counted multiplications and whose lookups each reach at most `eLk`, with lookup columns `b` bits wide, is
`(d − E, e + E · eLk)`-threaded at every `Γ` and every `S < (E + 1) · b`: a small subcircuit reads too little to hold
more than `E` of a thread's lookups, and one that reads fewer than `d − E` values of a thread holds none of its roots. -/
def ThreadedOfLayered : Prop :=
  ∀ (c : Declaration) (T : Threading c) (d e eLk b : ℕ), Layered T d e eLk b →
    ∀ (Γ : ℕ) (S : ℝ) (E : ℕ), S < ((E + 1) * b : ℕ) → T.Threaded Γ S (d - E) (e + E * eLk)

/-- **Training is slowed, for keyed circuits**: `ThreadedTraining` with its threading premise discharged by
`ThreadedOfLayered` at `S = αΓ`. -/
def LayeredTraining : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (c dcl : Declaration) (T : Threading c) (d e eLk b E : ℕ),
    Layered T d e eLk b → 0 ≤ α → 0 < b_w → 0 < b_a → E < d → c.Contains dcl → Complies Γ α dcl →
    α * Γ < ((E + 1) * b : ℕ) → Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a →
    κ * c.D.size c.D.computed ≤ 2 * T.M.card →
      κ * b_a * c.D.size c.D.computed ≤
        2 * α * T.bound (α * Γ) b_w (d - E) (e + E * eLk) * dcl.D.size dcl.D.computed

end Nci.Guarantees
