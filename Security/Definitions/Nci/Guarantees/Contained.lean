import Definitions.Nci.Assumptions
import Definitions.Nci.Dark

/-!
# The inference-only policy's partial-sum law on a reference circuit

`UpdatedMatMul` states its incompressibility assumptions about the declared circuit, whose wiring the prover controls,
and they are false there. These guarantees state the same law with the chain and both assumptions on a reference
training circuit `r`, which the declared circuit reaches through containment, as the thread guarantees do
(`Nci.Guarantees.ThreadedThrottled`), and in support form (`ActivationsSupport`, `PartialSumsSupport`), so that they say
what `r`'s values determine rather than how it is wired. The assumptions are then claims about honest training: a small
piece of `r` holds at least `b_w` bits per updated weight entry it uses and `b_a` bits per activation entry or partial
sum it does not compute itself. `b_w` and `b_a` are rates, below the entries' widths by whatever they compress to, so
these guarantees are parameterized by how compressible training is.

`ContainedMatMulOn` and `ContainedTrainingOn` state the law when the declaration contains `r` only partly
(`PartialContainment`): the products it computes are throttled, less what the dark gates, computed off the books, absorb
of the partial sums, which is at most their width.

Both support forms are falsified (`Assumptions.ActivationsSupport`, `Assumptions.PartialSumsSupport`), so every
guarantee here rests on a falsified premise. The same laws in reads form, under `TrainingIncompressible` on `r`, are
`IncompressibleContainedTraining`, `IncompressibleContainedMatMulOn` and `IncompressibleContainedTrainingOn`.
-/

namespace Nci.Guarantees

open Verity

/-- **A reference circuit's updated matrix multiplications are throttled**: a declaration `d` that contains `r` and
complies at `(c, k)`, with its registered information free of the reference circuit's fresh input
(`ThetaRegistered`), has at least `2 F` gates per product of a chain `ch` that `r` runs, when `r`'s weights,
activations and partial sums are incompressible at `b_w` and `b_a` bits per entry, with `r`'s registered information at
its committed values `θv`. `dd` is the chain's smallest inner dimension. -/
def ContainedMatMul : Prop :=
  ∀ (c k : ℕ) (b_w b_a : ℝ) (r d : Declaration) (θv : r.D.Transcript) (ch : Chain r.D) (dd : ℕ)
    (M : Containment r.D d.D),
    0 < c → 0 < k → 0 < b_w → 0 < b_a → (∀ ℓ, dd ≤ ch.d ℓ) → Complies c k d →
    Assumptions.ThetaRegistered r d (PartialContainment.ofContainment M) →
    Assumptions.ActivationsSupport r θv ch (k + b_w + b_a) b_w b_a →
    Assumptions.PartialSumsSupport r θv ch c k b_w b_a →
      F (k / c) k b_w b_a dd * (2 * ch.G) ≤ d.D.size d.D.computed

/-- **Training is slowed, by the partial-sum law**: when the chain's products, at 2 flops each, are a fraction `κ` of
the reference circuit's size, every declaration that contains it and complies at `(c, k)`, with its registered
information free of the reference circuit's fresh input (`ThetaRegistered`), is larger by a factor of at least `κ F`.
-/
def ContainedTraining : Prop :=
  ∀ (c k : ℕ) (b_w b_a κ : ℝ) (r d : Declaration) (θv : r.D.Transcript) (ch : Chain r.D) (dd : ℕ)
    (M : Containment r.D d.D),
    0 < c → 0 < k → 0 < b_w → 0 < b_a → (∀ ℓ, dd ≤ ch.d ℓ) → Complies c k d →
    Assumptions.ThetaRegistered r d (PartialContainment.ofContainment M) →
    Assumptions.ActivationsSupport r θv ch (k + b_w + b_a) b_w b_a →
    Assumptions.PartialSumsSupport r θv ch c k b_w b_a →
    Assumptions.Coverage κ r (2 * ch.G) →
      κ * F (k / c) k b_w b_a dd * r.D.size r.D.computed ≤ d.D.size d.D.computed

/-- **The partial-sum law with dark gates**: a declaration `d` that complies at `(c, k)` and contains `r` partly
(`PartialContainment`, computing the gates `S` of `r` and leaving the rest dark) computes the chain's products in `S`,
`G_S` of them, with
`max (min c₁ c₂) (μ b_a / Λ) · G_S ≤ (k / c) size(d) + (1 − μ) width(dark)` for each `μ ∈ [0, 1]`,
where `min c₁ c₂` is `F`'s minimum at read cap `K` and smallest inner dimension `dd`, and `Λ = max e_max (K / (4 b_w dd))`
with `e_max` the largest output dimension. The assumptions are on `r` at read cap `K ≥ k`, beside the
registered-information premise `ThetaRegistered`, and each dark gate reads at most `K` itself, so that it is an
inference block of its own: the dark gates absorb at most their width of the partial sums. -/
def ContainedMatMulOn : Prop :=
  ∀ (c k : ℕ) (K b_w b_a μ : ℝ) (r d : Declaration) (θv : r.D.Transcript) (ch : Chain r.D) (dd emax : ℕ)
    (M : PartialContainment r d),
    0 < K → 0 < b_w → 0 < b_a → 0 < dd → 0 < c → (k : ℝ) ≤ K → 0 ≤ μ → μ ≤ 1 →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → Complies c k d → Assumptions.ThetaRegistered r d M →
    Assumptions.DarkReads M K →
    Assumptions.ActivationsSupport r θv ch (K + b_w + b_a) b_w b_a →
    Assumptions.PartialSumsSupport r θv ch c K b_w b_a →
      max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
          (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
        (Finset.univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
          ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card ≤
      (k / c : ℝ) * d.D.size d.D.computed + (1 - μ) * r.D.width M.dark

/-- **Training is slowed, by the partial-sum law, with dark gates**: when the chain's products, at 2 flops each, are a
fraction `κ` of the reference circuit's size, each product off the books is a dark gate, so
`max (min c₁ c₂) (μ b_a / Λ) · (κ size(r) − 2 #dark) ≤ 2 ((k / c) size(d) + (1 − μ) width(dark))`. -/
def ContainedTrainingOn : Prop :=
  ∀ (c k : ℕ) (K b_w b_a μ κ : ℝ) (r d : Declaration) (θv : r.D.Transcript) (ch : Chain r.D) (dd emax : ℕ)
    (M : PartialContainment r d),
    0 < K → 0 < b_w → 0 < b_a → 0 < dd → 0 < c → (k : ℝ) ≤ K → 0 ≤ μ → μ ≤ 1 →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → Complies c k d → Assumptions.ThetaRegistered r d M →
    Assumptions.DarkReads M K →
    Assumptions.ActivationsSupport r θv ch (K + b_w + b_a) b_w b_a →
    Assumptions.PartialSumsSupport r θv ch c K b_w b_a →
    Assumptions.Coverage κ r (2 * ch.G) →
      max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
          (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
        (κ * r.D.size r.D.computed - 2 * M.dark.card) ≤
      2 * ((k / c : ℝ) * d.D.size d.D.computed + (1 - μ) * r.D.width M.dark)

end Nci.Guarantees
