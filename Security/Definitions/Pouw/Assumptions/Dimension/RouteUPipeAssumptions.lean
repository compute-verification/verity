import Definitions.Pouw.Assumptions.Dimension.PipeAssumptions
import Definitions.Pouw.Assumptions.Dimension.RouteU

/-!
# The closed pipe's named assumptions on route U

`A1_cost` and `A1_cost_quad` (`Pouw.Dimension.PipeAssumptions`) restated for route U's words, `A1_cost_U` and
`A1_cost_quad_U`, with their conclusion `A1CostOnU`. `A1_cost` is not a statement modulo `V₀`: it counts the FP32 steps
of a pipe whose outputs *contain* the checked words. A pipe that outputs route U's words outputs the unbiased ones only
after one more step per word (`1·wordU + (−β·s_t)`), and the converse costs the same, so neither form gives the other at
the same `c`. Theorem 1 proves `A1_cost_quad_U β 2 0` for every `β`, `k` and `R` (`a1CostQuadUHolds`, from
`Forest.a1CostQuadNear`); `A1_cost_U` stays open, as `A1_cost` does. Each has a satisfiability witness
(`A1CostWitnessU`).
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

/-- `A1_cost`'s conclusion on route U for one class of pipes `C`: every pipe in `C` whose outputs include the
non-final route-U words of `S ⊆ U × [0, nonFinal)` has at least `c·|S| − c·γ·D` steps. -/
def A1CostOnU (β : ℤ) (C : ∀ {N m n k : ℕ}, List (Noise N m n k → ℚ) → Prop) (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    ∀ Rs : List (Noise N m n k → ℚ), C Rs →
    ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, wordU β k m n N R A B x.1 x.2 ∈ Rs) →
      c * S.card - c * γ * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ Rs.length

/-- **Assumption `A1_cost_U(c, γ)`** (named, restated for route U): under checkpoint independence and MinRank ≥ 5,
every closed FP32 pipe that outputs `|S|` non-final route-U words has at least `c·|S| − c·γ·D` steps. -/
def A1_cost_U (β : ℤ) (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R 5 → A1CostOnU β IsPipe c γ k R

/-- **`A1_cost_U` for quadratic pipes** (`A1_cost_quad` restated for route U). Theorem 1 proves it at `c = 2`,
`γ = 0` for every `β`, `k` and `R` (`A1CostQuadUHolds`). -/
def A1_cost_quad_U (β : ℤ) (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R 5 → A1CostOnU β IsQuadPipe c γ k R

end Pouw.Dimension
