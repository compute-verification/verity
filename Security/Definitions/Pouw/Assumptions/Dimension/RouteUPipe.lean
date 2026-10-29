import Definitions.Pouw.Assumptions.Dimension.Pipe
import Definitions.Pouw.Assumptions.Dimension.RouteU
import Definitions.Pouw.Assumptions.Dimension.RouteUPipeAssumptions

/-!
# `A1_cost` in the closed scalar pipe on route U (trusted statements)

The closed-pipe chain (`Pouw.Dimension.Pipe`, `Pouw.Dimension.Forest`) for route U's words, with `A1_cost_U` and
`A1_cost_quad_U` (`Pouw.Dimension.RouteUPipeAssumptions`) in place of `A1_cost` and `A1_cost_quad`. Everything else in
the chain reads the words only through their independence modulo `V₀`, which transfers
(`Pouw.Dimension.RouteUProofs.indep_subU`), so `AlgBoundPipeU` and `AlgBoundQuadU` hold with the same bounds.
Theorem 1 holds for every family that differs from the checked words by elements of `V₀` (`Forest.A1CostQuadNear`), so
`A1_cost_quad_U β 2 0` holds (`A1CostQuadUHolds`), and with it Theorem D for the quadratic closed pipe at `c_s = 8`
(`AlgBoundQuad8U`).
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

/-! ## Pinned statements -/

/-- **Theorem D with a closed scalar pipe on route U** (`AlgBoundPipe` for route-U words, with `A1_cost_U`). -/
def AlgBoundPipeU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ : ℝ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    ClosedPipe cs P → OutputsU β k m n N R A B U P L → A1_cost_U β c γ k R → 16 ≤ c * cs →
    16 * (1 - γ) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

/-- **Theorem D with a quadratic closed pipe on route U** (`AlgBoundQuad` for route-U words, with
`A1_cost_quad_U`). -/
def AlgBoundQuadU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ : ℝ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    QuadPipe cs P → OutputsU β k m n N R A B U P L → A1_cost_quad_U β c γ k R → 16 ≤ c * cs →
    16 * (1 - γ) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

/-- **Theorem 1 on route U** (`A1CostQuadHolds` for route-U words): `A1_cost_quad_U β 2 0` holds for every `β`, `k`
and `R`. Under checkpoint independence and MinRank ≥ 5, a quadratic closed FP32 pipe whose outputs include `|S|`
non-final route-U words has at least `2·|S|` steps. -/
def A1CostQuadUHolds : Prop := ∀ (β : ℤ) (k : ℕ) (R : Mat), A1_cost_quad_U β 2 0 k R

/-- **Theorem D for the quadratic closed pipe at `c_s = 8` on route U** (`AlgBoundQuad8` for route-U words), with no
`A1` assumption: under checkpoint independence and MinRank ≥ 5, every `QuadPipe 8` program on inputs in `V₀` that
outputs the non-final route-U words of the units in `U` costs at least `16·|U|·m·n·(3k/16 − 1)`. -/
def AlgBoundQuad8U : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    QuadPipe 8 P → OutputsU β k m n N R A B U P L →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **`A1_cost_U`'s witness** (`A1CostWitness` on route U): `A1_cost_U` and `A1_cost_quad_U` hold at `c = 2`,
`γ = 1/2` for every `k` and `R`, by the count of one step per output word. Their content is `γ` near 0. -/
def A1CostWitnessU : Prop := ∀ (β : ℤ) (k : ℕ) (R : Mat), A1_cost_U β 2 (1 / 2) k R ∧ A1_cost_quad_U β 2 (1 / 2) k R

/-- **The closed pipe's witness on route U** (`PipeWitness` for route-U words): for every `k ≥ 1024` with `16 ∣ k`,
some admissible, checkpoint-independent `R` with MinRank ≥ 16 and some quadratic closed-pipe program at `c_s = 8` on
the noise coordinates output the non-final route-U words, with one FP32 step (`1·1 + 0`, at 8), at cost
`16·|U|·m·n·(3k/16 − 1) + 8`. -/
def PipeWitnessU : Prop :=
  ∀ (β : ℤ) (k : ℕ), 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ), U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      MinRankGe k R 16 ∧ QuadPipe 8 P ∧ OutputsU β k m n N R A B U P (coords N m n k) ∧ (prodOuts P).length = 1 ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 8

end Pouw.Dimension
