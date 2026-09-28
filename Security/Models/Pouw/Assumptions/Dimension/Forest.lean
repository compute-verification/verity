import Models.Pouw.Assumptions.Dimension.Pipe

/-!
# Theorem 1 of `a1-cost.md` (the forest argument): statements (trusted)

`a1-cost.md` §3.2–3.3, restated on the grid as the red team asked (Phase 8: P8-1, priced additions; P8-2, products of
free values, with `F₁`-functions free).

* **Q-parts** (`QPart`): the functions of the noise modulo `V₀`. An FP32 step's value is then determined by its
  operands' Q-parts: a product of two free values contributes its **atom** (`atoms`: the class of `a·b` with
  `a, b ∈ V₀`), a product with a constant factor scales, and a free operand contributes `0`.
* **The abstract count** (`ForestBound`). Values `y 0, y 1, …` in a finite-dimensional ℚ-space; the steps in `F` are
  merge-free FP32 steps (`FStep`: an atom plus an operand, or `α·b + c`), each operand `0` or a multiple of an
  earlier value; every other value is priced and arbitrary. If no nonzero element of a subspace `Z` is a sum of two
  atoms (`TwoAtomFree`, rank ≥ 3 on Q-parts), then with FP32 steps at 8 and priced values at 16 the program pays at
  least `16·dim` of the span of its values in `Z`: `2·dim ≤ 2·#priced + #FP32`. With no priced values this is
  Theorem 1, `#FP32 ≥ 2·dim`; with them it is Corollary 2 at P8-1's prices, the priced values allowed as FP32
  operands.
* **The grid** (`GridTwoAtom`): under checkpoint independence and MinRank ≥ 5, the non-final checked words modulo
  `V₀` (`wordsQ`) contain no nonzero sum of two atoms. The four-point mixed difference of an atom is a rank ≤ 2
  matrix, so two atoms give rank ≤ 4, and MinRank ≥ 5 exhibits a nonsingular 5 × 5 minor.
* Together: `A1_cost_quad 2 0` for every `k` and `R` (`A1CostQuadHolds`), Theorem D's `16·D` for the quadratic
  closed pipe at `c_s = 8` (`AlgBoundQuad8`), and TT_NCP from A2 alone into the quadratic pipe (`TTNCPQuadOfA2`).
  The open part of `A1_cost(2, 0)` is `A1_nonquad` (`A1CostOfNonquad`).
-/

namespace Pouw.Dimension.Forest

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions

section Abstract

variable {W : Type} [AddCommGroup W] [Module ℚ W]

/-- An operand of step `i`, on Q-parts: `0` (a free value) or a multiple of an earlier value. -/
def Opnd (y : ℕ → W) (i : ℕ) (c : W) : Prop := c = 0 ∨ ∃ j < i, ∃ β : ℚ, c = β • y j

/-- **A merge-free FP32 step** on Q-parts: type P, an atom plus an operand (FFMA or FMUL of two free values), or
type A, `α·b + c` (FADD, or a product with a constant factor). -/
def FStep (At : Set W) (y : ℕ → W) (i : ℕ) : Prop :=
  (∃ q ∈ At, ∃ c, Opnd y i c ∧ y i = q + c) ∨
    (∃ (α : ℚ) (b c : W), Opnd y i b ∧ Opnd y i c ∧ y i = α • b + c)

/-- **Rank ≥ 3 on Q-parts**: a sum of two atoms that lies in `Z` is `0`. -/
def TwoAtomFree (Z : Submodule ℚ W) (At : Set W) : Prop :=
  ∀ q ∈ At, ∀ q' ∈ At, q + q' ∈ Z → q + q' = 0

end Abstract

/-- **Theorem 1 with Corollary 2's pricing** (`a1-cost.md` §3.2–3.3; abstract). Values `y i` for `i < N`; the steps
in `F` are merge-free FP32 steps, the others priced (any value). If the atoms are closed under scaling and no
nonzero element of `Z` is a sum of two atoms, then `2·d ≤ 2·#priced + #FP32`, with `d` the dimension of the span of
the values that lie in `Z`: at 16 per priced value and 8 per FP32 step, the program costs at least `16·d`. -/
def ForestBound : Prop :=
  ∀ (W : Type) [AddCommGroup W] [Module ℚ W] [FiniteDimensional ℚ W] (Z : Submodule ℚ W) (At : Set W),
    (∀ q ∈ At, ∀ μ : ℚ, μ • q ∈ At) → TwoAtomFree Z At →
    ∀ (y : ℕ → W) (N : ℕ) (F : Finset ℕ), F ⊆ range N → (∀ i ∈ F, FStep At y i) →
      2 * Module.finrank ℚ (Submodule.span ℚ (y '' {i | i < N ∧ y i ∈ Z})) ≤ 2 * (N - F.card) + F.card

/-- Q-parts on the grid: the functions of the noise modulo `V₀`. -/
abbrev QPart (N m n k : ℕ) : Type := (Noise N m n k → ℚ) ⧸ V0 N m n k

/-- The atoms: products of two free values (elements of `V₀`), modulo `V₀`. -/
def atoms (N m n k : ℕ) : Set (QPart N m n k) :=
  {q | ∃ a ∈ V0 N m n k, ∃ b ∈ V0 N m n k, q = Submodule.Quotient.mk (a * b)}

/-- The span of the non-final checked words of the units in `U`, modulo `V₀`. -/
def wordsQ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) :
    Submodule ℚ (QPart N m n k) :=
  Submodule.span ℚ (Set.range fun x : ↥(U ×ˢ range (nonFinal k m n)) =>
    (Submodule.Quotient.mk (word k m n N R A B x.1.1 x.1.2) : QPart N m n k))

/-! ## Pinned statements -/

/-- **Rank ≥ 3 on the grid, in the form Theorem 1 uses** (P8-2): under checkpoint independence and MinRank ≥ 5, no
nonzero element of the checked words' span modulo `V₀` is a sum of two atoms, for every `A = g(F₁)` and `B`. -/
def GridTwoAtom : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    TwoAtomFree (wordsQ k m n N R A B U) (atoms N m n k)

/-- **`A1_cost_quad 2 0` holds** (Theorem 1 on the grid): for every `k` and `R`, under checkpoint independence and
MinRank ≥ 5, a quadratic closed FP32 pipe whose outputs include `|S|` non-final checked words has at least `2·|S|`
steps. -/
def A1CostQuadHolds : Prop := ∀ (k : ℕ) (R : Mat), A1_cost_quad 2 0 k R

/-- **The open part of `A1_cost(2, 0)`**: the pipes that merge (`A1_nonquad 2 0`) give all of it. -/
def A1CostOfNonquad : Prop := ∀ (k : ℕ) (R : Mat), A1_nonquad 2 0 k R → A1_cost 2 0 k R

/-- **Theorem D for the quadratic closed pipe at `c_s = 8`**, with no `A1` assumption: under checkpoint independence
and MinRank ≥ 5, every `QuadPipe 8` program on inputs in `V₀` that outputs the non-final checked words of the units
in `U` costs at least `16·|U|·m·n·(3k/16 − 1)`. -/
def AlgBoundQuad8 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) →
    QuadPipe 8 P → Outputs k m n N R A B U P L →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **TT_NCP from A2 alone, into the quadratic closed pipe**: `TTNCPOfQuadA2` with its `A1_cost_quad` hypothesis
discharged (`γ₁ = 0`). If a machine satisfies `A2(γ₂)` into `QuadPipe 8`, `R` is checkpoint independent with
MinRank ≥ 5, one of the machine's programs outputs the non-final checked words of the units in `U`, and
`1 − γ₀ ≤ (1 − γ₂)·(1 − 16/(3k))`, then TT_NCP's event fails for that program. For a real machine, A2 into
`QuadPipe 8` claims two things: its programs cannot beat the closed pipe on the words, and merges do not help, which
asks `A1_nonquad`'s question in program-matching form. Neither that A2 nor `A1_nonquad` implies the other. -/
def TTNCPQuadOfA2 : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (M : Machine (Noise N m n k)) (q : M.Prog) (γ₀ γ₂ : ℝ), 16 ∣ k → 0 < k →
    CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    A2_algebraicSufficiency γ₂ M (QuadPipe 8) (wordSet (word k m n N R A B) U) →
    (∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ M.outputs q) →
    γ₂ < 1 → γ₀ < 1 → 1 - γ₀ ≤ (1 - γ₂) * (1 - 16 / (3 * k : ℝ)) →
    ¬ (M.cost q / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **`TTNCPQuadOfA2`'s hypotheses hold together**: for every `k ≥ 1024` with `16 ∣ k`, every shape, `A = g(F₁)`,
`B` and unit set, one `R` and one program of `quadMachine` satisfy all of them at `γ₂ = 0` and `γ₀ = 1/2`. -/
def QuadChainConsistent : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N → ∃ R : Mat, CheckpointIndependent k 16 R ∧ MinRankGe k R 5 ∧
      A2_algebraicSufficiency 0 (quadMachine N m n k 8) (QuadPipe 8) (wordSet (word k m n N R A B) U) ∧
      (∃ q : (quadMachine N m n k 8).Prog,
        ∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ (quadMachine N m n k 8).outputs q) ∧
      1 - (1 / 2 : ℝ) ≤ (1 - 0) * (1 - 16 / (3 * k : ℝ))

/-- **`TTNCPQuadOfA2`'s hypotheses hold together at the tight constant** (red team P11-2): `QuadChainConsistent` at
`γ₂ = 0` and `γ₀ = 16/(3k)`, where the arithmetic hypothesis holds with equality. In the model the chain loses only
the first checkpoint of each column. -/
def QuadChainConsistentTight : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N → ∃ R : Mat, CheckpointIndependent k 16 R ∧ MinRankGe k R 5 ∧
      A2_algebraicSufficiency 0 (quadMachine N m n k 8) (QuadPipe 8) (wordSet (word k m n N R A B) U) ∧
      (∃ q : (quadMachine N m n k 8).Prog,
        ∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ (quadMachine N m n k 8).outputs q) ∧
      (0 : ℝ) < 1 ∧ 16 / (3 * k : ℝ) < 1 ∧
      1 - 16 / (3 * k : ℝ) = (1 - 0) * (1 - 16 / (3 * k : ℝ))

/-! ## Theorem 1 for any word family (route U: `wordU = word + an element of V₀`) -/

/-- The span of a word family's non-final values at the units in `U`, modulo `V₀` (`wordsQ` for `w = word`). -/
def familyQ {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) : Submodule ℚ (QPart N m n k) :=
  Submodule.span ℚ (Set.range fun x : ↥(U ×ˢ range (nonFinal k m n)) =>
    (Submodule.Quotient.mk (w x.1.1 x.1.2) : QPart N m n k))

/-- A word family's non-final values at the units in `U` are linearly independent modulo `V₀`. -/
def FamilyIndepQ {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) : Prop :=
  LinearIndependent ℚ (fun x : ↥(U ×ˢ range (nonFinal k m n)) =>
    (Submodule.Quotient.mk (w x.1.1 x.1.2) : QPart N m n k))

/-- Theorem 1's conclusion for a word family: every quadratic closed FP32 pipe whose outputs include the family's
values at `S ⊆ U × [0, nonFinal)` has at least `2·|S|` steps. -/
def QuadCount {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) : Prop :=
  ∀ Rs : List (Noise N m n k → ℚ), IsQuadPipe Rs →
    ∀ S ⊆ U ×ˢ range (nonFinal k m n), (∀ x ∈ S, w x.1 x.2 ∈ Rs) → 2 * S.card ≤ Rs.length

/-- **Theorem 1 for any word family**, from exactly what the proof uses: if the family's values modulo `V₀` are
linearly independent and no nonzero element of their span is a sum of two atoms, every quadratic closed pipe that
outputs `|S|` of them has at least `2·|S|` steps. -/
def Theorem1Family : Prop :=
  ∀ (N m n k : ℕ) (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ),
    TwoAtomFree (familyQ w U) (atoms N m n k) → FamilyIndepQ w U → QuadCount w U

/-- **The grid gives those hypotheses to every family near the words**: under checkpoint independence and MinRank ≥
5, every family that differs from the checked words by an element of `V₀` at each non-final position of the units
in `U` (route U's `wordU`) is linearly independent modulo `V₀` with no nonzero sum of two atoms in its span. -/
def GridNear : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (w : ℕ → ℕ → Noise N m n k → ℚ), CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    (∀ u ∈ U, ∀ p < nonFinal k m n, w u p - word k m n N R A B u p ∈ V0 N m n k) →
    TwoAtomFree (familyQ w U) (atoms N m n k) ∧ FamilyIndepQ w U

/-- **Theorem 1 for every family near the words**: under checkpoint independence and MinRank ≥ 5, a quadratic closed
pipe whose outputs include `|S|` non-final values of a family that differs from the checked words by elements of
`V₀` has at least `2·|S|` steps. At `w = word` this is `A1_cost_quad 2 0`; at route U's `wordU` it is
`A1_cost_quad_U 2 0`. -/
def A1CostQuadNear : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (w : ℕ → ℕ → Noise N m n k → ℚ), CheckpointIndependent k 16 R → MinRankGe k R 5 → U ⊆ range N →
    (∀ u ∈ U, ∀ p < nonFinal k m n, w u p - word k m n N R A B u p ∈ V0 N m n k) → QuadCount w U

end Pouw.Dimension.Forest
