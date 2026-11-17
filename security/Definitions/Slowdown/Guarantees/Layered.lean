module

public import Definitions.Slowdown.Guarantees.Threads
public import Definitions.Slowdown.Assumptions.Layered

/-! Every keyed reference training algorithm meets the threading premise of the thread draft's Theorems 1 and 2, with `(d, e)` read off its structure. -/

@[expose] public section

open InferenceOnly

namespace Slowdown.Guarantees

open Verity

/-- **Keyed circuits are threaded**: a keyed reference training algorithm whose roots have fan-in `d`, whose values each reach at
most `e` counted multiplications and whose lookups each reach at most `eLk`, with lookup columns `b` bits wide, is
`(d − E, e + E · eLk)`-threaded at every `c` and every `S < (E + 1) · b`: a small subcircuit reads too little to hold
more than `E` of a thread's lookups, and one that reads fewer than `d − E` values of a thread holds none of its roots. -/
def ThreadedOfLayered : Prop :=
  ∀ (r : RegisteredCircuit) (T : Threading r) (d e eLk b : ℕ), Layered T d e eLk b →
    ∀ (c : ℕ) (S : ℝ) (E : ℕ), S < ((E + 1) * b : ℕ) → T.Threaded c S (d - E) (e + E * eLk)

/-- **Training is slowed, for keyed circuits**: `ThreadedTraining` with its threading premise discharged by
`ThreadedOfLayered` at `S = α Γ`. -/
def LayeredTraining : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (r dcl : RegisteredCircuit) (T : Threading r) (d e eLk b E : ℕ)
    (M : Containment r.D dcl.D),
    Layered T d e eLk b → 0 ≤ α → 0 < b_w → 0 < b_a → E < d → CompliesRate Γ α dcl →
    Assumptions.ThetaRegistered r dcl (PartialContainment.ofContainment M) →
    Assumptions.CapBelowColumns (α * Γ) E b → Assumptions.WeightsIncompressible r T Γ (α * Γ) b_w b_a →
    Assumptions.Coverage κ r (2 * T.M.card) →
      κ * b_a * r.D.size r.D.computed ≤
        2 * α * T.bound (α * Γ) b_w (d - E) (e + E * eLk) * dcl.D.size dcl.D.computed

/-- **The reference training algorithm is threaded, at Kimi K2's shapes**: `ThreadedOfLayered` at `ReferenceLayered`'s parameters.
Every `S ≤ 2·10⁷` is below `7 + 1` bf16 embedding columns (`8 · 163840 · 16 = 20971520` bits), so a reference training algorithm
keyed at `(d₀, e, e_lk) = (17, 2328, 0)` is `(17 − 7, 2328 + 7 · 0) = (10, 2328)`-threaded at every `c`. -/
def ReferenceThreaded : Prop :=
  ∀ (c : RegisteredCircuit) (T : Threading c), Assumptions.ReferenceLayered c T →
    ∀ (Γ : ℕ) (S : ℝ), S ≤ 2 * 10 ^ 7 → T.Threaded Γ S 10 2328

end Slowdown.Guarantees
