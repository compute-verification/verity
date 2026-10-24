import Definitions.Pouw.Dimension.Causal

/-! The proof-only part of `Definitions.Pouw.Dimension.Causal`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation

/-- **Theorem D with causal inputs** (P7-1): under checkpoint independence and H_16, for every causal `A`, every
program on inputs in `V₀` that outputs the non-final checked words of the units in `U` costs at least
`16·|U|·m·n·(3k/16 − 1)`. -/
def AlgBoundCausal : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → Causal A → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n, wordC k m n N R A B u p ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **`Causal`'s witness, one that reads earlier noise**: `A_u = E₁` of unit `u − 1` (and `A_0 = 0`) is causal. -/
def CausalWitness : Prop :=
  ∀ N m n k : ℕ, Causal (N := N) (m := m) (n := n) (k := k) fun ω u => if u = 0 then 0 else e1Of ω (u - 1)

/-- **The causal bound's hypotheses hold jointly, and it is attained**: for every `k ≥ 1024` with `16 ∣ k`, every
shape, every `A` (causal or not), `B` and unit set, some `R` passes `Admissible` and is checkpoint independent, and
an H_16 program on the noise coordinates outputs the non-final checked words at cost exactly
`16·|U|·m·n·(3k/16 − 1)`. -/
def AlgWitnessCausal : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧ H16 P ∧
      (∀ u ∈ U, ∀ p < nonFinal k m n, wordC k m n N R A B u p ∈ slots P (coords N m n k)) ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1))

end Pouw.Dimension
