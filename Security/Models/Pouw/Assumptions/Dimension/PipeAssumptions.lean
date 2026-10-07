import Models.Pouw.Assumptions.Dimension.PipeModel

/-!
# The closed pipe's and the lifting's named assumptions

Every open named `Prop` of the closed-pipe chain (`Pouw.Dimension.Pipe`, `Pouw.Dimension.Forest`) and of the lifting
(`Pouw.Dimension.Lift`) that is not in `Pouw.Dimension.Assumptions`: `A1_cost`, `A1_cost_quad` (Theorem 1 proves it
at `c = 2`, `γ = 0`: `a1CostQuadHolds`), `A1_nonquad`, `LiftMinRank` and `LiftMinRankL5e`. Each is a family of
`Prop`s with parameters, and each has a satisfiability witness (`A1CostWitness`, `LiftMinRankWitness`,
`LiftMinRankL5eWitness`).
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

variable {N m n k : ℕ}

/-- **Assumption `A1_cost(c, γ)`** (named; W1's cost model, additive form, `c` a parameter): under checkpoint
independence and MinRank ≥ 5, every closed FP32 pipe that outputs `|S|` non-final checked words has at least
`c·|S| − c·γ·D` steps. -/
def A1_cost (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R 5 → A1CostOn IsPipe c γ k R

/-- **`A1_cost` for quadratic pipes.** `a1-cost`'s Theorem 1 (the forest argument; proved on paper, red team Phase 8:
granted with P8-1 and P8-2, which this model meets) is `A1_cost_quad 2 0 k R` for every `k` and `R`. -/
def A1_cost_quad (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R 5 → A1CostOn IsQuadPipe c γ k R

/-- **Assumption `A1_nonquad(c, γ)`** (named, open): `A1_cost` for the closed pipes that are not quadratic, the ones
that multiply a non-free value (a merge, `a1-cost.md` §3.4). -/
def A1_nonquad (c γ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R 5 → A1CostOn (fun Rs => IsPipe Rs ∧ ¬ IsQuadPipe Rs) c γ k R

/-- **`Lift_MinRank`** (P7-3; named, open), `ρ` and `ε` parameters: under checkpoint independence and MinRank ≥ ρ, for
every program on inputs in `V₀` and every assignment `σ` of its slots to the non-final checked words of the units in
`U` (`A = g(F₁)`), the number of noise values at which every word equals its slot is at most `ε(D − p)` times the
grid's size, with `D = |U|·m·n·(3k/16 − 1)` and `p` the number of priced instructions. -/
def LiftMinRank (ρ : ℕ) (ε : ℤ → ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRankGe k R ρ →
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (σ : ℕ → ℕ → ℕ),
    U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    (Nat.card {ω : Noise N m n k // ∀ u ∈ U, ∀ p < nonFinal k m n,
        (slots P L).getD (σ u p) 0 ω = word k m n N R A B u p ω} : ℝ) ≤
      ε ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) - (nPriced P : ℤ)) * Nat.card (Noise N m n k)

/-- **`Lift_MinRank` in L5e's form** (named; proved on paper by the lifting worker, not in Lean): for `1 ≤ G ≤ ρ` and
`1 ≤ Λ ≤ ρ − G + 1`, under checkpoint independence and MinRank ≥ ρ, `LiftMinRank`'s count is at most
`epsL5e ρ G Λ k (N·m) n (3k/16 − 1) (D − p)` times the grid's size. -/
def LiftMinRankL5e (ρ G Λ k : ℕ) (R : Mat) : Prop :=
  1 ≤ G → G ≤ ρ → 1 ≤ Λ → Λ + G ≤ ρ + 1 → CheckpointIndependent k 16 R → MinRankGe k R ρ →
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (σ : ℕ → ℕ → ℕ),
    U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    (Nat.card {ω : Noise N m n k // ∀ u ∈ U, ∀ p < nonFinal k m n,
        (slots P L).getD (σ u p) 0 ω = word k m n N R A B u p ω} : ℝ) ≤
      epsL5e ρ G Λ k (N * m) n (3 * k / 16 - 1)
        ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) - (nPriced P : ℤ)) * Nat.card (Noise N m n k)

end Pouw.Dimension
