module

public import Definitions.Pouw.Dimension.Defs

/-! The named assumption A1, on the scalar-pipe products Theorem D's H_16 leaves out, as `Prop`s the statements take as hypotheses. -/

@[expose] public section

namespace Pouw.Dimension.Assumptions

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension

/-- The output of one scalar-pipe product. -/
def ProdFree {N m n k : ℕ} (φ : Noise N m n k → ℚ) : Prop :=
  ∃ ℓ ∈ V0 N m n k, ∃ ℓ' ∈ V0 N m n k, φ = ℓ * ℓ'

def targetSpan (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) :
    Submodule ℚ (Noise N m n k → ℚ) :=
  Submodule.span ℚ (Set.range fun x : ↥(U ×ˢ range (nonFinal k m n)) => word k m n N R A B x.1.1 x.1.2)

/-- **Assumption A1(c, γ₁), multiplicative complexity**: `L(V) ≥ c·(1 − γ₁)·dim V` over `V₀`. With `c·(1 − γ₁) > 1` it
implies MinRank ≥ 2 (`Pouw.Dimension.A1MinRank`), so use it with MinRank as a hypothesis. -/
def A1_multiplicative (c γ₁ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    ∀ V : Submodule ℚ (Noise N m n k → ℚ), V ≤ targetSpan k m n N R A B U →
    ∀ (r : ℕ) (p : Fin r → Noise N m n k → ℚ), (∀ i, ProdFree (p i)) →
      V ≤ Submodule.span ℚ (Set.range p) ⊔ V0 N m n k →
      c * (1 - γ₁) * (Module.finrank ℚ V : ℝ) ≤ r

/-- **A1 at W1's constant** `c = 2`, false without MinRank ≥ 2. Stronger, probably false asymptotically; not used by the
recommended chain, which uses `A1_cost` in the closed scalar pipe (`Pouw.Dimension.Pipe`). -/
def A1_W1 (γ₁ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRank2 k R → A1_multiplicative 2 γ₁ k R

end Pouw.Dimension.Assumptions
