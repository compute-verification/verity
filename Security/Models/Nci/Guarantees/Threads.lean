import Models.Nci.Assumptions.Threads

/-!
# NCI over threads: the guarantees

The thread draft's Theorems 1 and 2. Under the NCI policy an IU reads at most `α` bits per gate and has at most `Γ`
gates, so it reads at most `αΓ` bits: the draft's `S*` is `αΓ` here.
-/

namespace Nci.Guarantees

open Verity

/-- **Theorem 1 (training's NCI floor)** (the thread draft, Appendix A): if the reference circuit `c` is
`(d, e)`-threaded and its weights are incompressible at `b_w` bits per weight and `b_a` per activation, every small
subcircuit `R` of `c` computes at most `Λ = max(e, c · S / (4 d b_w))` counted multiplications per `b_a` bits of novel
information it reads. Counting 2 flops per counted multiplication, its NCI is at least
`b_a / (2Λ) = (b_a/2) · min(1/e, 4 d b_w/(cS))`, the draft's statement at `b_w = b_a = b`, written multiplied out so that
a subcircuit that reads nothing needs no division. -/
def ThreadedReads : Prop :=
  ∀ (c : Declaration) (T : Threading c) (Γ : ℕ) (S b_w b_a : ℝ) (d e : ℕ),
    0 < b_w → 0 < b_a → 0 < d → T.Threaded Γ S d e → Assumptions.WeightsIncompressible c T Γ S b_w b_a →
    ∀ R, Small c Γ S R → b_a * (T.mults R).card ≤ novelReads c R * T.bound S b_w d e

/-- **Theorem 2 (high NCI means padding)** (the thread draft, Appendix B), for the reference circuits of Theorem 1: a
declaration `d` that contains `c` and complies at `(Γ, α)` has at least `b_a / (αΛ)` gates per counted multiplication
of `c`, where `Λ` is Theorem 1's bound at `S = αΓ`. -/
def ThreadedThrottled : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (T : Threading c) (dd e : ℕ),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → c.Contains d → Complies Γ α d →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a →
      b_a * T.M.card ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **Training is slowed** (the thread draft, "Putting the two together"): when counted multiplications are a fraction
`κ` of the reference circuit's flops (2 flops each), every declaration that contains it and complies at `(Γ, α)` is
larger by a factor of at least `κ · b_a / (2αΛ)`. -/
def ThreadedTraining : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (c d : Declaration) (T : Threading c) (dd e : ℕ),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → c.Contains d → Complies Γ α d →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a →
    κ * c.D.size c.D.computed ≤ 2 * T.M.card →
      κ * b_a * c.D.size c.D.computed ≤ 2 * α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

end Nci.Guarantees
