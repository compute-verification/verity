import Definitions.Pouw.Dimension.Defs

/-!
# Theorem D's residue: the named assumptions A1 and A2 (trusted)

`docs/pouw/milder-assumptions.md` §2.4 and §2.5, with the red team's fixes TD-1, TD-4 and TD-5 (`red-team.md`,
Phase 6). `Prop`s taken as hypotheses by the statements that use them, never `axiom`s. Theorem D
(`Pouw.Dimension.AlgBound`) prices every priced register write at 16 or more (H_16). A1 covers the scalar-pipe
products H_16 leaves out; A2 says that a program of the machine does no better than an algebraic program that
outputs the same checked words on the whole grid.
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
(`Pouw.Dimension.A1MinRank`, TD-5), so use it with MinRank as a hypothesis. -/
def A1_multiplicative (c γ₁ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ), U ⊆ range N →
    ∀ V : Submodule ℚ (Noise N m n k → ℚ), V ≤ targetSpan k m n N R A B U →
    ∀ (r : ℕ) (p : Fin r → Noise N m n k → ℚ), (∀ i, ProdFree (p i)) →
      V ≤ Submodule.span ℚ (Set.range p) ⊔ V0 N m n k →
      c * (1 - γ₁) * (Module.finrank ℚ V : ℝ) ≤ r

/-- **A1 at W1's constant** `c = 2`, under checkpoint independence (P7-2: for a dependent `R` it is a false
hypothesis) and MinRank ≥ 2 (TD-5: without it A1(2) is false). Daniel's decision of 03:35Z prices sub-word extraction
like PRMT (16 per word), so the packed-field route is gone and a scalar-pipe product costs 8 = 16/2.

Stronger, probably false asymptotically; not used by the recommended chain. Additions are free here (`a1.md`,
Brockett–Dobkin; red team P8-3: false at MinRank 2 to 4). The recommended chain uses `A1_cost` in the closed scalar
pipe (`Pouw.Dimension.Pipe`). -/
def A1_W1 (γ₁ : ℝ) (k : ℕ) (R : Mat) : Prop :=
  CheckpointIndependent k 16 R → MinRank2 k R → A1_multiplicative 2 γ₁ k R

/-- A machine for NCP's checked words (M_4090 when instantiated): its programs, their W1 cost, and each program's
outputs, the functions of the noise it outputs correctly on the whole grid. -/
structure Machine (G : Type) where
  /-- The programs, fixed before the salt. -/
  Prog : Type
  /-- A program's W1 cost. -/
  cost : Prog → ℝ
  /-- The functions of the noise a program outputs correctly on the whole grid. -/
  outputs : Prog → Set (G → ℚ)

/-- The non-final checked words of a word family `w` at the units in `U`: the functions `w u p` with `u ∈ U` and
`p < nonFinal`. A2's target set for `w = word` (or route U's words). -/
def wordSet {N m n k : ℕ} (w : ℕ → ℕ → Noise N m n k → ℚ) (U : Finset ℕ) : Set (Noise N m n k → ℚ) :=
  {φ | ∃ u ∈ U, ∃ p < nonFinal k m n, φ = w u p}

/-- **Assumption A2(γ₂), algebraic sufficiency** (survey §2.5), for a machine `M` on an NCP instance's noise grid, a
class `Alg` of AlgM programs and a target set `W` (the checked words, `wordSet`): every program of `M` is matched
by a program of `Alg` on inputs in `V₀` that outputs every function of `W` it outputs, at a price of at most
`cost/(1 − γ₂)`. The survey's A2 takes `Alg` = AlgM_4090 (scalar products at 8); with `Alg = H16` it also contains A1.

Only the outputs in `W` are matched (red team P9B-1): a program's other outputs, such as triple products or FP32 sums
of a tensor-core word, which the model may price above their real cost, are not constrained.

This is the deterministic form (TD-4): `M.outputs` is correctness on the whole grid. The probabilistic lifting, from
a program correct at the realized noise with probability `ε` to one correct on the whole grid, is not in A2; it is a
separate node of the chain (`Lift_MinRank`, open), which needs MinRank. -/
def A2_algebraicSufficiency (γ₂ : ℝ) {N m n k : ℕ} (M : Machine (Noise N m n k))
    (Alg : List (Instr (Noise N m n k)) → Prop) (W : Set (Noise N m n k → ℚ)) : Prop :=
  ∀ q : M.Prog, ∃ (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))), (∀ f ∈ L, f ∈ V0 N m n k) ∧
    Alg P ∧ (∀ φ ∈ M.outputs q, φ ∈ W → φ ∈ slots P L) ∧ (1 - γ₂) * (cost P : ℝ) ≤ M.cost q

end Pouw.Dimension.Assumptions
