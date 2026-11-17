module

public import Definitions.Pouw.Barrier.Pinned

/-! The degenerate-input barrier's generic lemma: a step of proofs, not a guarantee, which no record of the lock
pins. -/

@[expose] public section

namespace Pouw.Guarantees.Barrier

open Pouw.Barrier Pouw.Barrier.Pinned Finset

def Generic : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] [Fintype S] [Nonempty S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (U : Workload) (A : Adversary CM) (c₀ : ℝ)
    (q : ℕ),
    CheapPoint CM P D U A c₀ q → 0 < totalWref P U.layout →
    ∀ g : ℝ, g < 1 - c₀ / totalWref P U.layout → ∀ δ : ℕ → ℕ → ℝ, GGap CM P D g δ → 1 ≤ δ q U.layout.N

end Pouw.Guarantees.Barrier
