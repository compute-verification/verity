import Models.Nci.Assumptions.Threads

/-!
# NCI over threads: the guarantees

The thread draft's Theorems 1 and 2. Under the NCI policy an IU reads at most `α` bits per gate and has at most `Γ`
gates, so it reads at most `αΓ` bits: the draft's `S*` is `αΓ` here.
-/

namespace Nci.Guarantees

open Verity

/-- **Theorem 1 (training's NCI floor)** (the thread draft, Appendix A): if the reference circuit `c` is
`(d, e)`-threaded and its weights are incompressible, every small subcircuit `R` of `c` computes at most
`Λ = max(e, c · S / (4 d b))` counted multiplications per `b` bits of novel information it reads. Counting 2 flops per
counted multiplication, its NCI is at least `b / (2Λ) = (b/2) · min(1/e, 4db/(cS))`, the draft's statement, written
multiplied out so that a subcircuit that reads nothing needs no division. -/
def ThreadedReads : Prop :=
  ∀ (c : Declaration) (T : Threading c) (Γ : ℕ) (S b : ℝ) (d e : ℕ),
    0 < b → 0 < d → T.Threaded Γ S d e → Assumptions.WeightsIncompressible c T Γ S b →
    ∀ R, Small c Γ S R → b * (T.mults R).card ≤ novelReads c R * T.bound S b d e

/-- **Theorem 2 (high NCI means padding)** (the thread draft, Appendix B), for the reference circuits of Theorem 1: a
declaration `d` that contains `c` and complies at `(Γ, α)` has at least `b / (αΛ)` gates per counted multiplication of
`c`, where `Λ` is Theorem 1's bound at `S = αΓ`. -/
def ThreadedThrottled : Prop :=
  ∀ (Γ : ℕ) (α b : ℝ) (c d : Declaration) (T : Threading c) (dd e : ℕ),
    0 ≤ α → 0 < b → 0 < dd → c.Contains d → Complies Γ α d →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressible c T Γ (α * Γ) b →
      b * T.M.card ≤ α * T.bound (α * Γ) b dd e * d.D.size d.D.computed

/-- **Training is slowed** (the thread draft, "Putting the two together"): when counted multiplications are a fraction
`κ` of the reference circuit's flops (2 flops each), every declaration that contains it and complies at `(Γ, α)` is
larger by a factor of at least `κ · b / (2αΛ)`. -/
def ThreadedTraining : Prop :=
  ∀ (Γ : ℕ) (α b κ : ℝ) (c d : Declaration) (T : Threading c) (dd e : ℕ),
    0 ≤ α → 0 < b → 0 < dd → c.Contains d → Complies Γ α d →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressible c T Γ (α * Γ) b →
    κ * c.D.size c.D.computed ≤ 2 * T.M.card →
      κ * b * c.D.size c.D.computed ≤ 2 * α * T.bound (α * Γ) b dd e * d.D.size d.D.computed

end Nci.Guarantees
