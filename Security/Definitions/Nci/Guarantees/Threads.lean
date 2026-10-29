import Definitions.Nci.Assumptions.Threads
import Definitions.Nci.Assumptions.Baseline
import Definitions.Nci.Dark

/-!
# NCI over threads: the guarantees

The thread draft's Theorems 1 and 2. Under the NCI policy an IU reads at most `α` bits per gate and has at most `Γ`
gates, so it reads at most `αΓ` bits: the draft's `S*` is `αΓ` here.

Theorem 2 also holds when the declaration contains the reference circuit only partly (`PartialContainment`): the
counted multiplications it computes are throttled as before, and every other one is a dark gate, computed off the books
(`ThreadedThrottledOn`, `ThreadedThrottledDark`, `ThreadedTrainingOn`). Every form through a containment, total or
partial, takes `ThetaRegistered` for it: no dark value, and no novel input of the reference circuit, is in the
declaration's θ.

The `Slots` forms take `WeightsIncompressibleSlots`, weights charged per parameter slot, in place of
`WeightsIncompressible`, with the same bound: a subcircuit that spans several optimizer steps on a parameter is charged
for it once.
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
declaration `d` that contains `c` (`M`) and complies at `(Γ, α)`, with its θ registered (`ThetaRegistered`),
has at least `b_a / (αΛ)` gates per counted multiplication of `c`, where `Λ` is Theorem 1's bound at `S = αΓ`. -/
def ThreadedThrottled : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (T : Threading c) (dd e : ℕ) (M : Containment c.D d.D),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → Complies Γ α d →
    Assumptions.ThetaRegistered c d (PartialContainment.ofContainment M) →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a →
      b_a * T.M.card ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **Training is slowed** (the thread draft, "Putting the two together"): when counted multiplications are a fraction
`κ` of the reference circuit's flops (2 flops each), every declaration that contains it and complies at `(Γ, α)`, with
its θ registered (`ThetaRegistered`), is larger by a factor of at least `κ · b_a / (2αΛ)`. -/
def ThreadedTraining : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (c d : Declaration) (T : Threading c) (dd e : ℕ) (M : Containment c.D d.D),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → Complies Γ α d →
    Assumptions.ThetaRegistered c d (PartialContainment.ofContainment M) →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a →
    κ * c.D.size c.D.computed ≤ 2 * T.M.card →
      κ * b_a * c.D.size c.D.computed ≤ 2 * α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **Theorem 2 with dark gates**: a declaration `d` that complies at `(Γ, α)` and contains the reference circuit `c`
partly, computing the gates `S` of `c` and leaving the rest dark (`PartialContainment`), has at least `b_a / (αΛ)` gates
per counted multiplication of `c` in `S`, where `Λ` is Theorem 1's bound at read cap `αΓ`. Each IU of `d` computes a piece
of `S` that reads no more than the IU, since a dark value enters `d` only as an input, and outside its θ by
`ThetaRegistered`. At `S = computed` it is `ThreadedThrottled`. -/
def ThreadedThrottledOn : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (T : Threading c) (dd e : ℕ) (M : PartialContainment c d),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → Complies Γ α d → Assumptions.ThetaRegistered c d M →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a →
      b_a * (T.M ∩ M.S).card ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **Counted multiplications off the books are dark gates**: under `ThreadedThrottledOn`'s hypotheses, `c`'s counted
multiplications number at most its dark gates plus `α Λ size(d) / b_a`: `M ≤ U + C / ρ` with `U` the gates computed
off the books, `C = size(d)` and `ρ = b_a / (αΛ)` the gates per counted multiplication. -/
def ThreadedThrottledDark : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (T : Threading c) (dd e : ℕ) (M : PartialContainment c d),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → Complies Γ α d → Assumptions.ThetaRegistered c d M →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a →
      (T.M.card : ℝ) ≤ M.dark.card + α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed / b_a

/-- **Training is slowed, with dark gates**: when counted multiplications are a fraction `κ` of the reference circuit's
flops, a declaration that complies at `(Γ, α)` and contains it partly, with its θ registered
(`ThetaRegistered`), is larger by a factor of at least `κ b_a / (2αΛ)`, less what its dark gates absorb:
`κ b_a size(c) ≤ 2 b_a #dark + 2 α Λ size(d)`. -/
def ThreadedTrainingOn : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (c d : Declaration) (T : Threading c) (dd e : ℕ) (M : PartialContainment c d),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → Complies Γ α d → Assumptions.ThetaRegistered c d M →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a →
    κ * c.D.size c.D.computed ≤ 2 * T.M.card →
      κ * b_a * c.D.size c.D.computed ≤
        2 * b_a * M.dark.card + 2 * α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **Theorem 1 per parameter slot**: `ThreadedReads` with the weights charged per parameter slot of a slot map `σ`
(`WeightsIncompressibleSlots`) rather than per weight gate. Its bound is the same, `Λ = max(e, c · S / (4 d b_w))`,
since a thread's counted multiplications on the versions of one slot number at most `c` (`σ.reuse_le`). -/
def ThreadedReadsSlots : Prop :=
  ∀ (c : Declaration) (T : Threading c) (σ : T.Slots) (Γ : ℕ) (S b_w b_a : ℝ) (d e : ℕ),
    0 < b_w → 0 < b_a → 0 < d → T.Threaded Γ S d e → Assumptions.WeightsIncompressibleSlots c T σ Γ S b_w b_a →
    ∀ R, Small c Γ S R → b_a * (T.mults R).card ≤ novelReads c R * T.bound S b_w d e

/-- **Theorem 2 per parameter slot**: `ThreadedThrottled` under `WeightsIncompressibleSlots`. -/
def ThreadedThrottledSlots : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (T : Threading c) (σ : T.Slots) (dd e : ℕ)
    (M : Containment c.D d.D),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → Complies Γ α d →
    Assumptions.ThetaRegistered c d (PartialContainment.ofContainment M) →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressibleSlots c T σ Γ (α * Γ) b_w b_a →
      b_a * T.M.card ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **Theorem 2 with dark gates, per parameter slot**: `ThreadedThrottledOn` under `WeightsIncompressibleSlots`. -/
def ThreadedThrottledOnSlots : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (T : Threading c) (σ : T.Slots) (dd e : ℕ)
    (M : PartialContainment c d),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → Complies Γ α d → Assumptions.ThetaRegistered c d M →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressibleSlots c T σ Γ (α * Γ) b_w b_a →
      b_a * (T.M ∩ M.S).card ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **Counted multiplications off the books are dark gates, per parameter slot**: `ThreadedThrottledDark` under
`WeightsIncompressibleSlots`. -/
def ThreadedThrottledDarkSlots : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (T : Threading c) (σ : T.Slots) (dd e : ℕ)
    (M : PartialContainment c d),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → Complies Γ α d → Assumptions.ThetaRegistered c d M →
    T.Threaded Γ (α * Γ) dd e → Assumptions.WeightsIncompressibleSlots c T σ Γ (α * Γ) b_w b_a →
      (T.M.card : ℝ) ≤ M.dark.card + α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed / b_a

end Nci.Guarantees
