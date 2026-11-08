module

public import Definitions.Nci.Assumptions
public import Definitions.Nci.Assumptions.Training
public import Definitions.Nci.Dark

/-!
# The inference-only policy's guarantees on a reference training circuit, under one assumption

`ThreadedThrottled` and `ThreadedThrottledOn` restated with `TrainingIncompressible` as their only named assumption, in
place of `WeightsIncompressible`, and the partial-sum rule on a reference circuit under it, whole and through a partial
containment. Each statement names both the chain and the threading the assumption charges, and uses one of them.

The assumption is in reads, so a dark gate of a partial containment absorbs at most its fresh input reads of the partial
sums, where in support form it absorbed at most its width: the dark-gate term of `IncompressibleContainedMatMulOn` is
the dark gates' reads.
-/

@[expose] public section

namespace Nci.Guarantees

open Verity

/-- **A high fresh-input rate means padding, under training's incompressibility** (`ThreadedThrottled` with
`TrainingIncompressible`): a declaration `d` that contains a `(dd, e)`-threaded reference circuit `r` and complies at
`(Γ, α)` (`CompliesRate`), with its registered information free of the reference circuit's fresh input
(`ThetaRegistered`), has at least `b_a / (α Λ)` gates per counted multiplication of `r`, where `Λ` is the thread bound
at `S = α Γ`. -/
def IncompressibleThreadedThrottled : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (r d : RegisteredCircuit) (ch : Chain r.D) (T : Threading r) (dd e : ℕ)
    (M : Containment r.D d.D),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → CompliesRate Γ α d →
    Assumptions.ThetaRegistered r d (PartialContainment.ofContainment M) →
    T.Threaded Γ (α * Γ) dd e → Assumptions.TrainingIncompressible r ch T Γ (α * Γ) b_w b_a →
      b_a * T.M.card ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **A high fresh-input rate means padding, with dark gates, under training's incompressibility**
(`ThreadedThrottledOn` with `TrainingIncompressible`): a declaration `d` that complies at `(Γ, α)` and contains a
`(dd, e)`-threaded reference circuit `r` partly, computing the gates `S` of `r` and leaving the rest dark, with its
registered information free of the reference circuit's fresh input (`ThetaRegistered`), has at least `b_a / (α Λ)`
gates per counted multiplication of `r` in `S`, forward and activation-gradient alike, with no term for the dark gates.
-/
def IncompressibleThreadedThrottledOn : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a : ℝ) (r d : RegisteredCircuit) (ch : Chain r.D) (T : Threading r) (dd e : ℕ)
    (M : PartialContainment r d),
    0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd → CompliesRate Γ α d → Assumptions.ThetaRegistered r d M →
    T.Threaded Γ (α * Γ) dd e → Assumptions.TrainingIncompressible r ch T Γ (α * Γ) b_w b_a →
      b_a * (T.M ∩ M.S).card ≤ α * T.bound (α * Γ) b_w dd e * d.D.size d.D.computed

/-- **Training is slowed by the partial-sum rule, under training's incompressibility** (`TrainingIncompressible`): when
the products of a chain `ch` that `r` runs, at 2 flops each, are a fraction `κ` of `r`'s size, every declaration that
contains `r` and complies at `(c, k)`, with its registered information free of the reference circuit's fresh input
(`ThetaRegistered`), is larger by a factor of at least `κ F`. `dd` is the chain's smallest inner dimension. -/
def IncompressibleContainedTraining : Prop :=
  ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (r d : RegisteredCircuit) (ch : Chain r.D) (T : Threading r) (dd : ℕ)
    (M : Containment r.D d.D),
    0 < α → 0 < b_w → 0 < b_a → (∀ ℓ, dd ≤ ch.d ℓ) → CompliesRate Γ α d →
    Assumptions.ThetaRegistered r d (PartialContainment.ofContainment M) →
    Assumptions.TrainingIncompressible r ch T Γ (α * Γ) b_w b_a →
    Assumptions.Coverage κ r (2 * ch.G) →
      κ * F α (α * Γ) b_w b_a dd * r.D.size r.D.computed ≤ d.D.size d.D.computed

/-- **The partial-sum rule with dark gates, under training's incompressibility** (`TrainingIncompressible`): a
declaration `d` that complies at `(c, k)` and contains `r` partly (`PartialContainment`,
computing the gates `S` of `r` and leaving the rest dark) computes the chain's products in `S`, `G_S` of them, with
`max (min c₁ c₂) (μ b_a / Λ) · G_S ≤ α size(d) + (1 − μ) ∑_{v dark} novelReads(v)` for each `μ ∈ [0, 1]`,
where `min c₁ c₂` is `F`'s minimum at read cap `K` and smallest inner dimension `dd`, and `Λ = max e_max (K / (4 b_w dd))`
with `e_max` the largest output dimension. The assumption is on `r` at read cap `K ≥ α Γ`, beside the
registered-information premise `ThetaRegistered`, and each dark gate reads at most `K` itself, so that it is an
inference block of its own: the dark gates absorb at most their reads of the partial sums. -/
def IncompressibleContainedMatMulOn : Prop :=
  ∀ (Γ : ℕ) (α K b_w b_a μ : ℝ) (r d : RegisteredCircuit) (ch : Chain r.D) (T : Threading r) (dd emax : ℕ)
    (M : PartialContainment r d),
    0 ≤ α → 0 < K → 0 < b_w → 0 < b_a → 0 < dd → 1 ≤ Γ → α * Γ ≤ K → 0 ≤ μ → μ ≤ 1 →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → CompliesRate Γ α d → Assumptions.ThetaRegistered r d M →
    Assumptions.DarkReads M K →
    Assumptions.TrainingIncompressible r ch T Γ K b_w b_a →
      max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
          (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
        (Finset.univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
          ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card ≤
      α * d.D.size d.D.computed + (1 - μ) * ∑ v ∈ M.dark, (novelReads r {v} : ℝ)

/-- **Training is slowed, by the partial-sum rule, with dark gates, under training's incompressibility**
(`TrainingIncompressible`): when the chain's products, at 2 flops each, are a fraction `κ` of
the reference circuit's size, each product off the books is a dark gate, so
`max (min c₁ c₂) (μ b_a / Λ) · (κ size(r) − 2 #dark) ≤ 2 (α size(d) + (1 − μ) ∑_{v dark} novelReads(v))`. -/
def IncompressibleContainedTrainingOn : Prop :=
  ∀ (Γ : ℕ) (α K b_w b_a μ κ : ℝ) (r d : RegisteredCircuit) (ch : Chain r.D) (T : Threading r) (dd emax : ℕ)
    (M : PartialContainment r d),
    0 ≤ α → 0 < K → 0 < b_w → 0 < b_a → 0 < dd → 1 ≤ Γ → α * Γ ≤ K → 0 ≤ μ → μ ≤ 1 →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → CompliesRate Γ α d → Assumptions.ThetaRegistered r d M →
    Assumptions.DarkReads M K →
    Assumptions.TrainingIncompressible r ch T Γ K b_w b_a →
    Assumptions.Coverage κ r (2 * ch.G) →
      max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
          (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
        (κ * r.D.size r.D.computed - 2 * M.dark.card) ≤
      2 * (α * d.D.size d.D.computed + (1 - μ) * ∑ v ∈ M.dark, (novelReads r {v} : ℝ))

end Nci.Guarantees
