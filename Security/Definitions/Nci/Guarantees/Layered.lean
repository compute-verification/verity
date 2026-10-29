import Definitions.Nci.Guarantees.Threads
import Definitions.Nci.Assumptions.Layered

/-!
# The inference-only policy over threads: keyed circuits are threaded

The premise of the thread draft's Theorems 1 and 2 (`Threading.Threaded`) holds for every keyed reference circuit
(`Layered`), at every `c` and at every `S` below the width of `E + 1` lookup columns, with `(d, e)` read off the
circuit's structure. At Kimi K2's shapes, written in `Dot16` steps, under the named conjecture that the reference
circuit is keyed there (`Assumptions.ReferenceLayered`), that is `(10, 2328)` for `S ≤ 2·10⁷`, counting steps
(`ReferenceThreaded`, `ReferenceTraining`).
-/

namespace Nci.Guarantees

open Verity

/-- **Keyed circuits are threaded**: a keyed reference circuit whose roots have `d` legs, whose values each reach at
most `e` counted multiplications and whose lookups each reach at most `eLk`, with lookup columns `b` bits wide, is
`(d − E, e + E · eLk)`-threaded at every `c` and every `S < (E + 1) · b`: a small subcircuit reads too little to hold
more than `E` of a thread's lookups, and one that reads fewer than `d − E` values of a thread holds none of its roots. -/
def ThreadedOfLayered : Prop :=
  ∀ (r : Declaration) (T : Threading r) (d e eLk b : ℕ), Layered T d e eLk b →
    ∀ (c : ℕ) (S : ℝ) (E : ℕ), S < ((E + 1) * b : ℕ) → T.Threaded c S (d - E) (e + E * eLk)

/-- **Training is slowed, for keyed circuits**: `ThreadedTraining` with its threading premise discharged by
`ThreadedOfLayered` at `S = k`. -/
def LayeredTraining : Prop :=
  ∀ (c k : ℕ) (b_w b_a κ : ℝ) (r dcl : Declaration) (T : Threading r) (d e eLk b E : ℕ) (M : Containment r.D dcl.D),
    Layered T d e eLk b → 0 < c → 0 < b_w → 0 < b_a → E < d → Complies c k dcl →
    Assumptions.ThetaRegistered r dcl (PartialContainment.ofContainment M) →
    (k : ℝ) < ((E + 1) * b : ℕ) → Assumptions.WeightsIncompressible r T c k b_w b_a →
    κ * r.D.size r.D.computed ≤ 2 * T.M.card →
      κ * b_a * r.D.size r.D.computed ≤
        2 * (k / c : ℝ) * T.bound k b_w (d - E) (e + E * eLk) * dcl.D.size dcl.D.computed

/-- **The reference circuit is threaded, at Kimi K2's shapes**: `ThreadedOfLayered` at `ReferenceLayered`'s parameters.
Every `S ≤ 2·10⁷` is below `7 + 1` bf16 embedding columns (`8 · 163840 · 16 = 20971520` bits), so a reference circuit
keyed at `(d₀, e, e_lk) = (17, 2328, 0)` is `(17 − 7, 2328 + 7 · 0) = (10, 2328)`-threaded at every `c`. -/
def ReferenceThreaded : Prop :=
  ∀ (r : Declaration) (T : Threading r), Assumptions.ReferenceLayered r T →
    ∀ (c : ℕ) (S : ℝ), S ≤ 2 * 10 ^ 7 → T.Threaded c S 10 2328

/-- **Training is slowed, at Kimi K2's shapes**: `LayeredTraining` at `ReferenceLayered`'s parameters, for a policy
whose inference blocks read at most `k ≤ 2·10⁷` bits, with `ReferenceThreaded`'s `(d, e) = (10, 2328)`. -/
def ReferenceTraining : Prop :=
  ∀ (c k : ℕ) (b_w b_a κ : ℝ) (r dcl : Declaration) (T : Threading r) (M : Containment r.D dcl.D),
    Assumptions.ReferenceLayered r T → 0 < c → 0 < b_w → 0 < b_a → Complies c k dcl →
    Assumptions.ThetaRegistered r dcl (PartialContainment.ofContainment M) →
    (k : ℝ) ≤ 2 * 10 ^ 7 → Assumptions.WeightsIncompressible r T c k b_w b_a →
    κ * r.D.size r.D.computed ≤ 2 * T.M.card →
      κ * b_a * r.D.size r.D.computed ≤ 2 * (k / c : ℝ) * T.bound k b_w 10 2328 * dcl.D.size dcl.D.computed

end Nci.Guarantees
