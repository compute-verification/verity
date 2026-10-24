import Definitions.Pouw.Dimension.Lift

/-! The proof-only part of `Definitions.Pouw.Dimension.Lift`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation

/-- **`LiftMinRankL5e`'s witness**, in three parts. Its hypotheses are jointly satisfiable, at `ρ = G = Λ = 1` for every
`k ≥ 1024` with `16 ∣ k`. Its bound is at least 1 at every saving `d ≤ 0`, so every program that saves nothing (the
honest one included) meets it. And its bound is at most 1 at every `d ≥ 0`, so it is no weaker than `ε = 1`. -/
def LiftMinRankL5eWitness : Prop :=
  (∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧ MinRankGe k R 1) ∧
  (∀ (ρ G Λ k M n T : ℕ) (d : ℤ), 1 ≤ G → 1 ≤ Λ → Λ + G ≤ ρ + 1 → d ≤ 0 → 1 ≤ epsL5e ρ G Λ k M n T d) ∧
  (∀ (ρ G Λ k M n T : ℕ) (d : ℤ), 1 ≤ G → 1 ≤ Λ → Λ + G ≤ ρ + 1 → 0 ≤ d → epsL5e ρ G Λ k M n T d ≤ 1)

/-- **MinRank ≥ ρ's witness, jointly**: for every `k ≥ 1024` with `16 ∣ k`, some `R` passes `Admissible`, is
checkpoint independent, has MinRank ≥ 2 (`MinRank2`) and satisfies `MinRankGe k R 1`. (`minrank.md` certifies
MinRank = 16 for the shift by 24 and the production P, on paper and by script; `MinRankGe k R 16` is not pinned.) -/
def MinRankGeWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧ MinRank2 k R ∧ MinRankGe k R 1

end Pouw.Dimension
