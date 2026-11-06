import Definitions.Pouw.Dimension.Defs
import Definitions.Pouw.Assumptions.Dimension.Assumptions
import Definitions.Pouw.Assumptions.NCP.Chain

/-!
# Theorem D's residue: pinned statements (trusted)

A1's witness, and Theorem D for the protocol's checked values, which are taken mod `2^32` (`Pouw.NCP.Chain`). The
probabilistic lifting (`Lift_MinRank`) is open.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension.Assumptions

/-- **A1's witness**: under checkpoint independence, `A1(c, γ₁)` holds whenever `c·(1 − γ₁) ≤ 1`. That is Theorem D's
own count, one dimension per function, whether a product or not. The assumption's content is `c = 2`. -/
def A1Witness : Prop :=
  ∀ (k : ℕ) (R : Mat) (c γ₁ : ℝ), CheckpointIndependent k 16 R → c * (1 - γ₁) ≤ 1 → A1_multiplicative c γ₁ k R

/-- **Theorem D for the protocol's checked values** (mod `2^32`, which are `Params.Y`'s by `paramsYLink`): for a
signed-permutation `R`, `k ≤ 2^16`, `A = g(F₁)` int7 at every `F₁` and `B` int7 on the units in `U`, under checkpoint
independence and H_16, a program on inputs in `V₀` that outputs the wrapped non-final checked words costs at least
`16·|U|·m·n·(3k/16 − 1)`. There the wrap is the identity (`int7Int32`). The hypotheses on `R` hold jointly at the
lead (`leadIndWitness`). -/
def AlgBoundWrapped : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → SignedPerm k R → k ≤ 2 ^ 16 → U ⊆ range N →
    (∀ F, ∀ u ∈ U, ∀ i, ∀ l < k, InInt7 (A F u i l)) → (∀ u ∈ U, ∀ l < k, ∀ j, InInt7 (B u l j)) →
    (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    (∀ u ∈ U, ∀ p < nonFinal k m n, (fun ω => (Chain.wrapped k m n N R (A ω.2) B ω u p : ℚ)) ∈ slots P L) →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **A1 implies MinRank ≥ 2** (a necessary condition on `R`): if `A1(c, γ₁)` holds with `c·(1 − γ₁) > 1`, then
`MinRank2 k R`. A nonzero rank-1 combination `x·yᵀ` of the non-final checkpoint matrices makes one entry's words a
single product of free values. So A1 at W1's `c = 2` is stated under MinRank ≥ 2 (`A1_W1`); the survey's proposed
MinRank rule targets 16. -/
def A1MinRank : Prop :=
  ∀ (k : ℕ) (R : Mat) (c γ₁ : ℝ), 1 < c * (1 - γ₁) → A1_multiplicative c γ₁ k R → MinRank2 k R

end Pouw.Dimension
