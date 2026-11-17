module

public import Definitions.Pouw.Assumptions.Dimension.Assumptions
public import Definitions.Pouw.Assumptions.NCP.Chain

/-! A1's witness and Theorem D for the protocol's checked values, taken mod `2^32`; the probabilistic lifting `Lift_MinRank` is not proved here. -/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

/-- **A1's witness**: Theorem D's own count; the assumption's content is `c = 2`. -/
def A1Witness : Prop :=
  ∀ (k : ℕ) (R : Mat) (c γ₁ : ℝ), CheckpointIndependent k 16 R → c * (1 - γ₁) ≤ 1 → A1_multiplicative c γ₁ k R

/-- **Theorem D for the protocol's checked values**, mod `2^32` (`Params.Y`'s by `paramsYLink`), where the wrap is the
identity (`int7Int32`). Its hypotheses on `R` hold jointly at the lead (`leadIndWitness`). -/
def AlgBoundWrapped : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → SignedPerm k R → k ≤ 2 ^ 16 → U ⊆ range N →
    (∀ F, ∀ u ∈ U, ∀ i, ∀ l < k, InInt7 (A F u i l)) → (∀ u ∈ U, ∀ l < k, ∀ j, InInt7 (B u l j)) →
    (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n, (fun ω => (Chain.wrapped k m n N R (A ω.2) B ω u p : ℚ)) ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **A1 implies MinRank ≥ 2**, a necessary condition on `R`, so A1 at W1's `c = 2` is stated under MinRank ≥ 2
(`A1_W1`). -/
def A1MinRank : Prop :=
  ∀ (k : ℕ) (R : Mat) (c γ₁ : ℝ), 1 < c * (1 - γ₁) → A1_multiplicative c γ₁ k R → MinRank2 k R

end Pouw.Dimension
