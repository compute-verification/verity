import Definitions.InferenceOnly.Assumptions
import Definitions.InferenceOnly.Assumptions.Training
import Definitions.InferenceOnly.Dark

/-!
# NCI's guarantees on a reference training circuit, under one assumption

`ThreadedThrottled`, `ThreadedThrottledOn`, `ContainedTraining`, `ContainedMatMulOn` and `ContainedTrainingOn` restated
with `TrainingIncompressible` as their only named assumption, in place of `WeightsIncompressible` for the thread forms and
`ActivationsSupport` and `PartialSumsSupport` for the chain forms. Each statement names both the chain and the threading
the assumption charges, and uses one of them.

The assumption is in reads, so a dark gate of a partial containment absorbs at most its novel reads of the partial sums,
where in support form it absorbed at most its width: the dark-gate term of `IncompressibleContainedMatMulOn` is the dark
gates' reads. At `μ = 1` that term goes, and with it the partial sums: `ContainedFanOutOn` bounds the products a
declaration computes under the activations' half of the assumption alone, `ActivationsIncompressible` on the reference
circuit.
-/

namespace InferenceOnly.Guarantees

open Verity

/-- **High NCI means padding, under training's incompressibility** (`ThreadedThrottled` with `TrainingIncompressible`): a
declaration `d` that contains a `(dd, e)`-threaded reference circuit `c` and complies at `(Γ, α)`, with its baseline
registered (`BaselineRegistered`), has at least `b_a / (αΛ)` gates per counted multiplication of `c`, where `Λ` is the thread bound at `S = αΓ`. -/
def IncompressibleThreadedThrottled : Prop :=
  ∀ (c k : ℕ) (b_w b_a : ℝ) (r d : Declaration) (ch : Chain r.D) (T : Threading r) (dd e : ℕ) (M : Containment r.D d.D),
    0 < c → 0 < b_w → 0 < b_a → 0 < dd → Complies c k d →
    Assumptions.BaselineRegistered r d (PartialContainment.ofContainment M) →
    T.Threaded c k dd e → Assumptions.TrainingIncompressible r ch T c k b_w b_a →
      b_a * T.M.card ≤ (k / c : ℝ) * T.bound k b_w dd e * d.D.size d.D.computed

/-- **High NCI means padding, with dark gates, under training's incompressibility** (`ThreadedThrottledOn` with
`TrainingIncompressible`): a declaration `d` that complies at `(Γ, α)` and contains a `(dd, e)`-threaded reference circuit
`c` partly, computing the gates `S` of `c` and leaving the rest dark, with its baseline registered (`BaselineRegistered`),
has at least `b_a / (αΛ)` gates per counted multiplication of `c` in `S`, forward and activation-gradient alike, with no
term for the dark gates. -/
def IncompressibleThreadedThrottledOn : Prop :=
  ∀ (c k : ℕ) (b_w b_a : ℝ) (r d : Declaration) (ch : Chain r.D) (T : Threading r) (dd e : ℕ)
    (M : PartialContainment r d),
    0 < c → 0 < b_w → 0 < b_a → 0 < dd → Complies c k d → Assumptions.BaselineRegistered r d M →
    T.Threaded c k dd e → Assumptions.TrainingIncompressible r ch T c k b_w b_a →
      b_a * (T.M ∩ M.S).card ≤ (k / c : ℝ) * T.bound k b_w dd e * d.D.size d.D.computed

/-- **Training is slowed by the partial-sum law, under training's incompressibility** (`ContainedTraining` with
`TrainingIncompressible`): when the products of a chain `ch` that `c` runs, at 2 flops each, are a fraction `κ` of `c`'s
size, every declaration that contains `c` and complies at `(Γ, α)`, with its baseline registered (`BaselineRegistered`),
is larger by a factor of at least `κ F`. `dd` is the
chain's smallest inner dimension. -/
def IncompressibleContainedTraining : Prop :=
  ∀ (c k : ℕ) (b_w b_a κ : ℝ) (r d : Declaration) (ch : Chain r.D) (T : Threading r) (dd : ℕ) (M : Containment r.D d.D),
    0 < c → 0 < k → 0 < b_w → 0 < b_a → (∀ ℓ, dd ≤ ch.d ℓ) → Complies c k d →
    Assumptions.BaselineRegistered r d (PartialContainment.ofContainment M) →
    Assumptions.TrainingIncompressible r ch T c k b_w b_a →
    κ * r.D.size r.D.computed ≤ 2 * ch.G →
      κ * F (k / c) k b_w b_a dd * r.D.size r.D.computed ≤ d.D.size d.D.computed

/-- **The partial-sum law with dark gates, under training's incompressibility** (`ContainedMatMulOn` with
`TrainingIncompressible`): a declaration `d` that complies at `(Γ, α)` and contains `c` partly (`PartialContainment`,
computing the gates `S` of `c` and leaving the rest dark) computes the chain's products in `S`, `G_S` of them, with
`max (min c₁ c₂) (μ b_a / Λ) · G_S ≤ α size(d) + (1 − μ) ∑_{v dark} novelReads(v)` for each `μ ∈ [0, 1]`,
where `min c₁ c₂` is `F`'s minimum at read cap `K` and smallest inner dimension `dd`, and `Λ = max e_max (K / (4 b_w dd))`
with `e_max` the largest output dimension. The assumption is on `c` at read cap `K ≥ αΓ`, beside the baseline premise
`BaselineRegistered`, and each dark gate reads at most `K` itself, so that it is an IU of its own: the dark gates absorb
at most their reads of the partial sums. -/
def IncompressibleContainedMatMulOn : Prop :=
  ∀ (c k : ℕ) (K b_w b_a μ : ℝ) (r d : Declaration) (ch : Chain r.D) (T : Threading r) (dd emax : ℕ)
    (M : PartialContainment r d),
    0 < K → 0 < b_w → 0 < b_a → 0 < dd → 0 < c → (k : ℝ) ≤ K → 0 ≤ μ → μ ≤ 1 →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → Complies c k d → Assumptions.BaselineRegistered r d M →
    (∀ v ∈ M.dark, (novelReads r {v} : ℝ) ≤ K) →
    Assumptions.TrainingIncompressible r ch T c K b_w b_a →
      max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
          (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
        (Finset.univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
          ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card ≤
      (k / c : ℝ) * d.D.size d.D.computed + (1 - μ) * ∑ v ∈ M.dark, (novelReads r {v} : ℝ)

/-- **Training is slowed, by the partial-sum law, with dark gates, under training's incompressibility**
(`ContainedTrainingOn` with `TrainingIncompressible`): when the chain's products, at 2 flops each, are a fraction `κ` of
the reference circuit's size, each product off the books is a dark gate, so
`max (min c₁ c₂) (μ b_a / Λ) · (κ size(c) − 2 #dark) ≤ 2 (α size(d) + (1 − μ) ∑_{v dark} novelReads(v))`. -/
def IncompressibleContainedTrainingOn : Prop :=
  ∀ (c k : ℕ) (K b_w b_a μ κ : ℝ) (r d : Declaration) (ch : Chain r.D) (T : Threading r) (dd emax : ℕ)
    (M : PartialContainment r d),
    0 < K → 0 < b_w → 0 < b_a → 0 < dd → 0 < c → (k : ℝ) ≤ K → 0 ≤ μ → μ ≤ 1 →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → Complies c k d → Assumptions.BaselineRegistered r d M →
    (∀ v ∈ M.dark, (novelReads r {v} : ℝ) ≤ K) →
    Assumptions.TrainingIncompressible r ch T c K b_w b_a →
    κ * r.D.size r.D.computed ≤ 2 * ch.G →
      max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
          (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
        (κ * r.D.size r.D.computed - 2 * M.dark.card) ≤
      2 * ((k / c : ℝ) * d.D.size d.D.computed + (1 - μ) * ∑ v ∈ M.dark, (novelReads r {v} : ℝ))

/-- **Products done in the declaration are bounded by their fan-out** (`IncompressibleContainedMatMulOn` at `μ = 1`): a
declaration `d` that complies at `(Γ, α)` and contains `c` partly computes the chain's products in `S`, `G_S` of them,
with `b_a / Λ · G_S ≤ α size(d)`, `Λ = max e_max (K / (4 b_w dd))`, and no term for the dark gates, so it holds even when
every partial sum is added off the books. Each IU makes at most `Λ` products per `b_a` bits it reads: `e_max` per input
entry it uses at a position it completes no entry of, one for each row of the multiplication, and, at positions it
completes entries of, `K / (4 b_w dd)` per `b_a` bits by AM–GM on its weight and input reads. Of training's
incompressibility it needs only the activations' half, `ActivationsIncompressible` on `c` at read cap `K ≥ αΓ`, besides
the baseline premise `BaselineRegistered`. -/
def ContainedFanOutOn : Prop :=
  ∀ (c k : ℕ) (K b_w b_a : ℝ) (r d : Declaration) (ch : Chain r.D) (dd emax : ℕ) (M : PartialContainment r d),
    0 < c → 0 < K → 0 < b_w → 0 < b_a → 0 < dd → (k : ℝ) ≤ K →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → Complies c k d → Assumptions.BaselineRegistered r d M →
    Assumptions.ActivationsIncompressible r ch c K b_w b_a →
      b_a / max (emax : ℝ) (K / (4 * b_w * dd)) *
        (Finset.univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
          ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card ≤
      (k / c : ℝ) * d.D.size d.D.computed

end InferenceOnly.Guarantees
