import Definitions.Pouw.Dimension.Defs

/-!
# Theorem D's residue: the named assumption A1 (trusted)

`docs/pouw/milder-assumptions.md` §2.4. `Prop`s taken as hypotheses by the statements that use them, never `axiom`s.
Theorem D (`Pouw.Dimension.AlgBound`) prices every priced register write at 16 or more (H_16). A1 covers the
scalar-pipe products H_16 leaves out.

The survey's A2 (algebraic sufficiency of a machine's programs, on its own abstract machine) is not stated: on the
checked words it is the per-word cost itself, TT_NCP restated, so it reduces TT_NCP to nothing milder, and TTNCP_U
stays the named conjecture (`Pouw.NCP.Assumptions.TTNCP_U`). The accounting's machine model is ACCEL's
(`Verity.Accel.Machine`).
-/

namespace Pouw.Dimension.Assumptions

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension

/-- A product of two free values (elements of `V₀`: affine in the noise plus any function of `F₁`), the output of
one scalar-pipe product (survey §2.4). -/
def ProdFree {N m n k : ℕ} (φ : Noise N m n k → ℚ) : Prop :=
  ∃ ℓ ∈ V0 N m n k, ∃ ℓ' ∈ V0 N m n k, φ = ℓ * ℓ'

/-- The span of the non-final checked words of the units in `U`, with the inputs `A = g(F₁)`. -/
def targetSpan (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) :
    Submodule ℚ (Noise N m n k → ℚ) :=
  Submodule.span ℚ (Set.range fun x : ↥(U ×ˢ range (nonFinal k m n)) => word k m n N R A B x.1.1 x.1.2)

/-- **Assumption A1(c, γ₁), multiplicative complexity** (survey §2.4), for the NCP instance `(k, R)`: for every
subspace `V` of the span of the non-final checked words, for every `A = g(F₁)`, any `r` products of free values whose
span together with `V₀` contains `V` number at least `c·(1 − γ₁)·dim V`. That is `L(V) ≥ c·(1 − γ₁)·dim V` over
`V₀`. `c` is a parameter; W1 fixes it at 2 (`A1_W1`). With `c·(1 − γ₁) > 1` it implies MinRank ≥ 2
(`Pouw.Dimension.A1MinRank`), so use it with MinRank as a hypothesis. -/
def A1_multiplicative (c γ₁ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    ∀ V : Submodule ℚ (Noise N m n k → ℚ), V ≤ targetSpan k m n N R A B U →
    ∀ (r : ℕ) (p : Fin r → Noise N m n k → ℚ), (∀ i, ProdFree (p i)) →
      V ≤ Submodule.span ℚ (Set.range p) ⊔ V0 N m n k →
      c * (1 - γ₁) * (Module.finrank ℚ V : ℝ) ≤ r

/-- **A1 at W1's constant** `c = 2`, under checkpoint independence (for a dependent `R` it is a false
hypothesis) and MinRank ≥ 2 (without it A1(2) is false). A ruling prices sub-word extraction
like PRMT (16 per word), so the packed-field route is gone and a scalar-pipe product costs 8 = 16/2.

Stronger, probably false asymptotically; not used by the recommended chain. Additions are free here (`a1.md`,
Brockett–Dobkin; false at MinRank 2 to 4). The recommended chain uses `A1_cost` in the closed scalar
pipe (`Pouw.Dimension.Pipe`). -/
def A1_W1 (γ₁ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRank2 k R → A1_multiplicative 2 γ₁ k R

end Pouw.Dimension.Assumptions
