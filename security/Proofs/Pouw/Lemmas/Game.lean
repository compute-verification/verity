module

public import Definitions.Pouw.Guarantees.Game

/-! Steps of the game's guarantees' proofs, stated as Props that no record pins. -/

@[expose] public section

namespace Pouw.Guarantees

open Pouw.GameStatements Finset

def Theorem1 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ),
    g < 1 → GGap CM P D g δ →
    ∀ U : Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, A.Bounded U T q →
    ∀ (C : Type) [Fintype C] (accept : (Q → R) → S → C → Prop) (εs δs : ℝ),
      pr (fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧
          εs * totalWref P U.layout <
            ∑ u ∈ range U.layout.N \ correctSet P U x.1 x.2.1 (A.trace U x.1 x.2.1), (P.Wref U.layout u : ℝ))
        ≤ δs →
      pr (fun x : (Q → R) × S × C => accept x.1 x.2.1 x.2.2 ∧ T < (1 - g) * (1 - εs) * totalWref P U.layout ∧
          ¬ A.Breaks U x.1 x.2.1)
        ≤ δs + δ q U.layout.N

def WorkWeightedSampling : Prop :=
  ∀ (α : Type) [Fintype α] (w : ℕ → ℕ) (N t : ℕ) (K : α → Finset ℕ) (εs : ℝ), εs ≤ 1 →
    pr (fun x : α × (Fin t → Ticket w N) => AuditAccepts (K x.1) x.2 ∧
        εs * ∑ u ∈ range N, (w u : ℝ) < ∑ u ∈ range N \ K x.1, (w u : ℝ)) ≤ (1 - εs) ^ t

end Pouw.Guarantees
