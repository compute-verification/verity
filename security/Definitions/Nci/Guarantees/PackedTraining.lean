module

public import Definitions.Nci.Assumptions.PackedTraining
public import Definitions.Nci.Dark

/-!
# The inference-only policy's partial-sum rule with dark gates, on a chain with several products to a gate

`IncompressibleContainedMatMulOn` and `IncompressibleContainedTrainingOn` for a `PackedChain`. The product bound counts
products and does not read `k`. The training form converts products off the books into dark gates, at most `k` to a
gate, so its dark-gate term is `2 k #dark` where the per-product form's is `2 #dark`.
-/

@[expose] public section

namespace Nci.Guarantees

open Verity

/-- **The partial-sum rule with dark gates, at most `k` products to a gate** (`IncompressibleContainedMatMulOn` on a
`PackedChain`): `max (min c₁ c₂) (μ b_a / Λ) · G_S ≤ α size(d) + (1 − μ) ∑_{v dark} novelReads(v)` for each
`μ ∈ [0, 1]`, with `G_S` the chain's products whose gates are in `S`, whatever `k`. -/
def PackedIncompressibleContainedMatMulOn : Prop :=
  ∀ (k Γ : ℕ) (α K b_w b_a μ : ℝ) (r d : RegisteredCircuit) (ch : PackedChain r.D k) (T : Threading r) (dd emax : ℕ)
    (M : PartialContainment r d),
    0 ≤ α → 0 < K → 0 < b_w → 0 < b_a → 0 < dd → 1 ≤ Γ → α * Γ ≤ K → 0 ≤ μ → μ ≤ 1 →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → CompliesRate Γ α d → Assumptions.ThetaRegistered r d M →
    Assumptions.DarkReads M K →
    Assumptions.PackedTrainingIncompressible r ch T Γ K b_w b_a →
      max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
          (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
        (Finset.univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
          ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card ≤
      α * d.D.size d.D.computed + (1 - μ) * ∑ v ∈ M.dark, (novelReads r {v} : ℝ)

/-- **Training is slowed, by the partial-sum rule, with dark gates, at most `k` products to a gate**
(`IncompressibleContainedTrainingOn` on a `PackedChain`): when the chain's products, at 2 flops each, are a fraction `κ`
of the reference circuit's size, the products off the books lie in dark gates, at most `k` to a gate, so
`max (min c₁ c₂) (μ b_a / Λ) · (κ size(r) − 2 k #dark) ≤ 2 (α size(d) + (1 − μ) ∑_{v dark} novelReads(v))`. -/
def PackedIncompressibleContainedTrainingOn : Prop :=
  ∀ (k Γ : ℕ) (α K b_w b_a μ κ : ℝ) (r d : RegisteredCircuit) (ch : PackedChain r.D k) (T : Threading r) (dd emax : ℕ)
    (M : PartialContainment r d),
    0 ≤ α → 0 < K → 0 < b_w → 0 < b_a → 0 < dd → 1 ≤ Γ → α * Γ ≤ K → 0 ≤ μ → μ ≤ 1 →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → CompliesRate Γ α d → Assumptions.ThetaRegistered r d M →
    Assumptions.DarkReads M K →
    Assumptions.PackedTrainingIncompressible r ch T Γ K b_w b_a →
    Assumptions.Coverage κ r (2 * ch.G) →
      max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
          (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
        (κ * r.D.size r.D.computed - 2 * (k * M.dark.card)) ≤
      2 * (α * d.D.size d.D.computed + (1 - μ) * ∑ v ∈ M.dark, (novelReads r {v} : ℝ))

end Nci.Guarantees
