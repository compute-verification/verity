module

public import Definitions.Pouw.Dimension.Defs

/-! `Lift_MinRank`, the chain's probabilistic node, as a named placeholder at parameters `ε` and `ρ`, and MinRank ≥ ρ; in Lean only its lemmas L2, L3 and L4's counting core are proved. -/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation

def MinRankGe (k : ℕ) (R : Mat) (ρ : ℕ) : Prop :=
  ∀ q : ℕ → ℚ, ∀ a < k, ∀ b < k,
    ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k R (16 * (t + 1)) a b : ℚ) ≠ 0 →
    ∃ r s : Fin ρ → ℕ, Function.Injective r ∧ Function.Injective s ∧ (∀ i, r i < k ∧ s i < k) ∧
      (∀ i, ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k R (16 * (t + 1)) (r i) (s i) : ℚ) ≠ 0) ∧
      (∀ i l, i < l → ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k R (16 * (t + 1)) (r i) (s l) : ℚ) = 0)

/-- `T` is the number of non-final checkpoints per entry and `M` the number of rows (`N·m` for `N` units sharing `F₁`). -/
noncomputable def deltaL5e (ρ G Λ k M n T : ℕ) (d : ℤ) : ℝ :=
  ((1 + (64 : ℝ) ^ ((Λ : ℤ) - ρ + G - 1)) / 64) ^
    ⌈(d : ℝ) / (((n * T + k : ℕ) : ℝ) / G + T + ((M * T : ℕ) : ℝ) / Λ)⌉

/-- The witness is at `ρ = G = Λ = 1` for every `k ≥ 1024` with `16 ∣ k`; the bound is at least 1 at every saving
`d ≤ 0` and at most 1 at every `d ≥ 0`. -/
def LiftMinRankL5eWitness : Prop :=
  (∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧ MinRankGe k R 1) ∧
  (∀ (ρ G Λ k M n T : ℕ) (d : ℤ), 1 ≤ G → 1 ≤ Λ → Λ + G ≤ ρ + 1 → d ≤ 0 → 1 ≤ deltaL5e ρ G Λ k M n T d) ∧
  (∀ (ρ G Λ k M n T : ℕ) (d : ℤ), 1 ≤ G → 1 ≤ Λ → Λ + G ≤ ρ + 1 → 0 ≤ d → deltaL5e ρ G Λ k M n T d ≤ 1)

/-- The witness is at `ρ = 1`; `MinRankGe k R 16` for the shift by 24 is not pinned. -/
def MinRankGeWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧ MinRank2 k R ∧ MinRankGe k R 1

end Pouw.Dimension
