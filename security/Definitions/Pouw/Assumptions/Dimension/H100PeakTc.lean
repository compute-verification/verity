module

public import Definitions.Pouw.Assumptions.Dimension.H100Floor

/-! A distinct-writes citation with no measurement at `tc`: `h100PeakTc`, legacy tensor-core forms at the `wgmma` peak rate (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.H100

/-- **The H100 table with legacy tensor-core forms at the peak rate** (draft semantics): `tc = 8` is m8n8k4's price
per register at the `wgmma` per-MAC rate. -/
def h100PeakTc : Prices100 := { h100Floor with tc := 8 }

/-! ## Guarantees (draft semantics, pending review) -/

/-- **The distinct-writes bound with no measurement at `tc`** (draft semantics, pending review). -/
def DistinctWritesPeakTc : Prop :=
  ∀ (G : Type) (F K : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G))
      (T₁ T₂ : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100PeakTc F) → (∀ o ∈ P, o.CheapIn 32 K) →
      DistinctOn S F T₁ → DistinctOn S F T₂ → (∀ t ∈ T₁, ∀ t' ∈ T₂, ¬ ∀ g ∈ S, t g = t' g) →
      (∀ t ∈ T₁, ∀ f ∈ K, ¬ ∀ g ∈ S, t g = f g) →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      32 * T₁.card + 8 * T₂.card ≤ cost100 h100PeakTc P

end Pouw.Dimension.H100
