module

public import Definitions.Pouw.Assumptions.Dimension.PipeModel
public import Definitions.Pouw.Dimension.Lift

/-! The named assumptions of the closed-pipe chain and the lifting that `Pouw.Dimension.Assumptions` does not hold, each with a satisfiability witness. -/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

variable {N m n k : ℕ}

/-- **Assumption `A1_cost(c, γ)`** (named; W1's cost model, additive form, `c` a parameter). -/
def A1_cost (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R 5 → A1CostOn IsPipe c γ k R

/-- `a1CostQuadHolds` proves it at `c = 2`, `γ = 0` (Theorem 1). -/
def A1_cost_quad (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R 5 → A1CostOn IsQuadPipe c γ k R

/-- **Assumption `A1_nonquad(c, γ)`** (named): `A1_cost` for the closed pipes that multiply a non-free value (a merge). -/
def A1_nonquad (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R 5 → A1CostOn (fun Rs => IsPipe Rs ∧ ¬ IsQuadPipe Rs) c γ k R

/-- **`Lift_MinRank`** (named), with parameters `ρ` and `ε`. -/
def LiftMinRank (ρ : ℕ) (ε : ℤ → ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R ρ →
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (σ : ℕ → ℕ → ℕ),
    U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    (Nat.card {ω : Noise N m n k // ∀ u ∈ U, ∀ p < nonFinal k m n,
        (slots P L).getD (σ u p) 0 ω = word k m n N R A B u p ω} : ℝ) ≤
      ε ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) - (nPriced P : ℤ)) * Nat.card (Noise N m n k)

/-- **`Lift_MinRank` in L5e's form** (named; proved on paper, not in Lean). -/
def LiftMinRankL5e (ρ G Λ k : ℕ) (R : Mat) : Prop :=
  1 ≤ G → G ≤ ρ → 1 ≤ Λ → Λ + G ≤ ρ + 1 → CheckpointIndependent k 16 R → MinRankGe k R ρ →
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (σ : ℕ → ℕ → ℕ),
    U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    (Nat.card {ω : Noise N m n k // ∀ u ∈ U, ∀ p < nonFinal k m n,
        (slots P L).getD (σ u p) 0 ω = word k m n N R A B u p ω} : ℝ) ≤
      deltaL5e ρ G Λ k (N * m) n (3 * k / 16 - 1)
        ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) - (nPriced P : ℤ)) * Nat.card (Noise N m n k)

end Pouw.Dimension
