module

public import Definitions.Pouw.Assumptions.Dimension.PipeAssumptions

/-! Theorem D with a closed scalar pipe, the model of `A1_cost`, and the witnesses of its hypotheses. -/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

variable {N m n k : ℕ}

def shift24 (k : ℕ) : Mat := fun i j => if j < k ∧ i = (if j + 24 < k then j + 24 else j + 24 - k) then 1 else 0

def A1CostOfSplit : Prop :=
  ∀ (c γ : ℝ) (k : ℕ) (R : Mat), A1_cost_quad c γ k R → A1_nonquad c γ k R → A1_cost c γ k R

def AlgBoundPipe : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ : ℝ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    ClosedPipe cs P → Outputs k m n N R A B U P L → A1_cost c γ k R → 16 ≤ c * cs →
    16 * (1 - γ) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

/-- With Theorem 1 (`A1_cost_quad 2 0`) and `c_s = 8` the bound is `16·D`, Theorem D's own. -/
def AlgBoundQuad : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (cs : ℕ) (c γ : ℝ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    QuadPipe cs P → Outputs k m n N R A B U P L → A1_cost_quad c γ k R → 16 ≤ c * cs →
    16 * (1 - γ) * ((U.card * (m * n) * (3 * k / 16 - 1) : ℕ) : ℝ) ≤ cost P

def PipeWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      MinRankGe k R 16 ∧ QuadPipe 8 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧ (prodOuts P).length = 1 ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 8

/-- As `PipeWitness`, with `u₀`'s non-final checked word at position 0 output by a 16-FFMA chain in the pipe. -/
def PipeWordWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (u₀ : ℕ), U ⊆ range N → u₀ ∈ U → 0 < m → 0 < n →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      MinRankGe k R 16 ∧ QuadPipe 8 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧ (prodOuts P).length = 16 ∧
      word k m n N R A B u₀ 0 ∈ prodOuts P ∧ word k m n N R A B u₀ 0 ∉ V0 N m n k ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1)) + 16 * 8

/-- **The `A1_cost` family's witness**, by the count of one step per output word; their content is `γ` near 0. -/
def A1CostWitness : Prop :=
  ∀ (k : ℕ) (R : Mat), A1_cost 2 (1 / 2) k R ∧ A1_cost_quad 2 (1 / 2) k R ∧ A1_nonquad 2 (1 / 2) k R

/-- 16 is the maximum: one step increment has rank 16. -/
def MinRankGeShift16 : Prop := ∀ k, 1024 ≤ k → 16 ∣ k → MinRankGe k (shift24 k) 16

def MinRankGeFacts : Prop :=
  (∀ (k : ℕ) (R : Mat) (ρ ρ' : ℕ), ρ' ≤ ρ → MinRankGe k R ρ → MinRankGe k R ρ') ∧
  (∀ (k : ℕ) (R : Mat), MinRankGe k R 2 → MinRank2 k R)

/-- **`Lift_MinRank`'s witness** (trivial): it holds at `ε = 1` for every `ρ`, `k` and `R`. The content is `ε` small. -/
def LiftMinRankWitness : Prop := ∀ (ρ k : ℕ) (R : Mat), LiftMinRank ρ (fun _ => 1) k R

end Pouw.Dimension
