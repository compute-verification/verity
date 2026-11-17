module

public import Definitions.Pouw.Assumptions.Dimension.Pipe

/-! The forest argument's statements: Theorem 1 with Corollary 2's pricing on Q-parts, and its consequences on the grid. -/

@[expose] public section

namespace Pouw.Dimension.Forest

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions

section Abstract

variable {W : Type} [AddCommGroup W] [Module ℚ W]

def Opnd (y : ℕ → W) (i : ℕ) (c : W) : Prop := c = 0 ∨ ∃ j < i, ∃ β : ℚ, c = β • y j

/-- Type P (FFMA or FMUL of two free values) or type A (FADD, or a product with a constant factor), on Q-parts. -/
def FStep (At : Set W) (y : ℕ → W) (i : ℕ) : Prop :=
  (∃ q ∈ At, ∃ c, Opnd y i c ∧ y i = q + c) ∨
    (∃ (α : ℚ) (b c : W), Opnd y i b ∧ Opnd y i c ∧ y i = α • b + c)

/-- **Rank ≥ 3 on Q-parts**: a sum of two atoms that lies in `Z` is `0`. -/
def TwoAtomFree (Z : Submodule ℚ W) (At : Set W) : Prop :=
  ∀ q ∈ At, ∀ q' ∈ At, q + q' ∈ Z → q + q' = 0

end Abstract

/-- **Theorem 1 with Corollary 2's pricing**: at 16 per priced value and 8 per FP32 step, the program costs at least
`16·d`. -/
def ForestBound : Prop :=
  ∀ (W : Type) [AddCommGroup W] [Module ℚ W] [FiniteDimensional ℚ W] (Z : Submodule ℚ W) (At : Set W),
    (∀ q ∈ At, ∀ μ : ℚ, μ • q ∈ At) → TwoAtomFree Z At →
    ∀ (y : ℕ → W) (N : ℕ) (F : Finset ℕ), F ⊆ range N → (∀ i ∈ F, FStep At y i) →
      2 * Module.finrank ℚ (Submodule.span ℚ (y '' {i | i < N ∧ y i ∈ Z})) ≤ 2 * (N - F.card) + F.card

abbrev QPart (N m n k : ℕ) : Type := (Noise N m n k → ℚ) ⧸ V0 N m n k

def atoms (N m n k : ℕ) : Set (QPart N m n k) :=
  {q | ∃ a ∈ V0 N m n k, ∃ b ∈ V0 N m n k, q = Submodule.Quotient.mk (a * b)}

def wordsQ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) :
    Submodule ℚ (QPart N m n k) :=
  Submodule.span ℚ (Set.range fun x : ↥(U ×ˢ range (nonFinal k m n)) =>
    (Submodule.Quotient.mk (word k m n N R A B x.1.1 x.1.2) : QPart N m n k))

/-- **Rank ≥ 3 on the grid, in the form Theorem 1 uses.** -/
def GridTwoAtom : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    TwoAtomFree (wordsQ k m n N R A B U) (atoms N m n k)

def A1CostQuadHolds : Prop := ∀ (k : ℕ) (R : Mat), A1_cost_quad 2 0 k R

/-- `A1_nonquad 2 0`, the pipes that merge, is all that is open of `A1_cost(2, 0)`. -/
def A1CostOfNonquad : Prop := ∀ (k : ℕ) (R : Mat), A1_nonquad 2 0 k R → A1_cost 2 0 k R

/-- **Theorem D for the quadratic closed pipe at `c_s = 8`**, with no `A1` assumption. -/
def AlgBoundQuad8 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    QuadPipe 8 P → Outputs k m n N R A B U P L →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

def familyQ {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) : Submodule ℚ (QPart N m n k) :=
  Submodule.span ℚ (Set.range fun x : ↥(U ×ˢ range (nonFinal k m n)) =>
    (Submodule.Quotient.mk (w x.1.1 x.1.2) : QPart N m n k))

def FamilyIndepQ {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) : Prop :=
  LinearIndependent ℚ (fun x : ↥(U ×ˢ range (nonFinal k m n)) =>
    (Submodule.Quotient.mk (w x.1.1 x.1.2) : QPart N m n k))

def QuadCount {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) : Prop :=
  ∀ Rs : List (Noise N m n k → ℚ), IsQuadPipe Rs →
    ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, w x.1 x.2 ∈ Rs) → 2 * S.card ≤ Rs.length

/-- **Theorem 1 for any word family**, from exactly the hypotheses its proof uses. -/
def Theorem1Family : Prop :=
  ∀ (N m n k : ℕ) (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ),
    TwoAtomFree (familyQ w U) (atoms N m n k) → FamilyIndepQ w U → QuadCount w U

/-- The grid gives `Theorem1Family`'s hypotheses to every family near the words, route U's `wordU` among them. -/
def GridNear : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (w : ℕ → ℕ → Noise N m n k → ℚ), CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    (∀ u ∈ U, ∀ p < nonFinal k m n, w u p - word k m n N R A B u p ∈ V0 N m n k) →
    TwoAtomFree (familyQ w U) (atoms N m n k) ∧ FamilyIndepQ w U

/-- At `w = word` this is `A1_cost_quad 2 0`; route U's `wordU` is such a family. -/
def A1CostQuadNear : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (w : ℕ → ℕ → Noise N m n k → ℚ), CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    (∀ u ∈ U, ∀ p < nonFinal k m n, w u p - word k m n N R A B u p ∈ V0 N m n k) → QuadCount w U

end Pouw.Dimension.Forest
