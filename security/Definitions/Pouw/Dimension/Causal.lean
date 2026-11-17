module

public import Definitions.Pouw.Dimension.Defs

/-! Theorem D with causal inputs: unit `u`'s input may read `F₁` and the `E₁` of the units before it. -/

@[expose] public section

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation

def Causal {N m n k : ℕ} (A : Noise N m n k → ℕ → Mat) : Prop :=
  ∀ (ω ω' : Noise N m n k) (u : ℕ), ω.2 = ω'.2 → (∀ u' : Fin N, (u' : ℕ) < u → ω.1 u' = ω'.1 u') →
    A ω u = A ω' u

def wordC (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ) :
    Noise N m n k → ℚ :=
  fun ω => (checked k m n N R (A ω) B ω u p : ℚ)

def AlgBoundCausal : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → Causal A → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n, wordC k m n N R A B u p ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- The witness reads earlier noise: `A_u = E₁` of unit `u − 1`, and `A_0 = 0`. -/
def CausalWitness : Prop :=
  ∀ N m n k : ℕ, Causal (N := N) (m := m) (n := n) (k := k) fun ω u => if u = 0 then 0 else e1Of ω (u - 1)

/-- The witness holds for every `k ≥ 1024` with `16 ∣ k` and attains the bound exactly. -/
def AlgWitnessCausal : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧ H16 P ∧
      (∀ u ∈ U, ∀ p < nonFinal k m n, wordC k m n N R A B u p ∈ slots P (coords N m n k)) ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1))

end Pouw.Dimension
