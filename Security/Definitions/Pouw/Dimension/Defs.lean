import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Definitions.Pouw.NCP.Relation

/-!
# Theorem D: the dimension bound in AlgM_16 (trusted statements)

`docs/pouw/milder-assumptions.md` §2 and §5 (a Sketch there). A program of AlgM is straight-line and noise-oblivious: a
list of instructions, and the constants of its free instructions, fixed before the salt. Its slots are functions of the
noise `ω = (E₁, F₁)` on the whole grid (`Noise`: every entry in `[−31, 32]`): first its inputs `L`, then one slot per
instruction.

* **Adaptive inputs.** `F₁` and `R` are public before `A` is committed, so `A` is any function `g(F₁)` (for the
  fixed `R` and salt), quantified. `B` does not depend on `F₁`: it is committed before `F₁` is drawn. The inputs `L`
  are any functions in `V₀` (`V0`): the constants, the noise coordinates and every function of `F₁` (so of `F₁`, `R`
  and `A`), all free.
* `free c w` outputs `c + Σ_{s < len} w_s·slot_s`, an exact ℚ-linear combination of the earlier slots and a constant,
  at no cost. This is more generous than W1, which prices adds; it gives every `E·B` term for nothing.
* `priced p f` outputs any function `f` of the noise, at price `p`. A function of the earlier slots is a function of
  the noise, so this is at least as strong as the survey's 'any function of the earlier slots'.
* **Registers.** A slot is one 32-bit register written by one instruction; an instruction that writes several
  registers is several instructions, one per register. Predicates are not slots: a free instruction reads only
  register slots, so an instruction whose output depends on a predicate other than ℚ-linearly is priced. `H16`
  prices every priced register write at 16 or more.
* **Integer semantics.** A program outputs a word when some slot's ℚ value equals the checked word's integer
  value at every grid point. The checked words are the integer ones (`Pouw.NCP.Relation.checked`); their link to
  the protocol's mod-`2^32` words is `int7Int32` (`AlgBoundWrapped`).

The targets are the non-final checked words (depth `16·(τ + 1)` for `τ < 3k/16 − 1`) of a set `U` of the `N` units
that share one `F₁`.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation

/-- An instruction over the noise grid `G`; its output is one 32-bit register slot. -/
inductive Instr (G : Type) where
  /-- `c + Σ_{s < len} w s · slot s`, free; reads register slots only, never a predicate. -/
  | free (c : ℚ) (w : ℕ → ℚ)
  /-- Any function of the noise, at a price: one register written. -/
  | priced (price : ℕ) (f : G → ℚ)

variable {G : Type}

/-- The slot an instruction appends to the slots `L`. -/
def Instr.out : Instr G → List (G → ℚ) → (G → ℚ)
  | .free c w, L => fun g => c + ∑ s ∈ range L.length, w s * (L.getD s 0) g
  | .priced _ f, _ => f

/-- An instruction's price: `0` if free. -/
def Instr.price : Instr G → ℕ
  | .free _ _ => 0
  | .priced p _ => p

/-- Whether an instruction is priced. -/
def Instr.isPriced : Instr G → Bool
  | .free _ _ => false
  | .priced _ _ => true

/-- A program's slots: its inputs `L`, then one slot per instruction, in order. -/
def slots (P : List (Instr G)) (L : List (G → ℚ)) : List (G → ℚ) :=
  P.foldl (fun S ins => S ++ [ins.out S]) L

/-- The number of priced instructions. -/
def nPriced (P : List (Instr G)) : ℕ := P.countP Instr.isPriced

/-- A program's cost: the sum of its prices. -/
def cost (P : List (Instr G)) : ℕ := (P.map Instr.price).sum

/-- **H_16**: every priced instruction, one 32-bit register written, costs at least 16. -/
def H16 (P : List (Instr G)) : Prop := ∀ ins ∈ P, ins.isPriced = true → 16 ≤ ins.price

/-- `V(L)`: the span of the constant `1` and the functions `L`. -/
def span1 (L : List (G → ℚ)) : Submodule ℚ (G → ℚ) := Submodule.span ℚ (insert 1 {f | f ∈ L})

/-- The noise coordinates as functions of the noise: `E₁[u, i, l]` and `F₁[l, j]`, each `v − 31`. -/
def coords (N m n k : ℕ) : List (Noise N m n k → ℚ) :=
  (List.finRange N).flatMap (fun u => (List.finRange m).flatMap fun i =>
      (List.finRange k).map fun l => fun ω => ((ω.1 u i l : ℕ) : ℚ) - 31) ++
    (List.finRange k).flatMap fun l => (List.finRange n).map fun j => fun ω => ((ω.2 l j : ℕ) : ℚ) - 31

/-- The functions of `F₁` alone. -/
def fnF (N m n k : ℕ) : Submodule ℚ (Noise N m n k → ℚ) :=
  Submodule.span ℚ {φ | ∀ ω ω' : Noise N m n k, ω.2 = ω'.2 → φ ω = φ ω'}

/-- `V₀`: the free values, the affine functions of the noise plus every function of `F₁`. -/
def V0 (N m n k : ℕ) : Submodule ℚ (Noise N m n k → ℚ) := span1 (coords N m n k) ⊔ fnF N m n k

/-- The non-final positions of a unit: `3k/16 − 1` checkpoints of `m·n` entries. -/
def nonFinal (k m n : ℕ) : ℕ := (3 * k / 16 - 1) * (m * n)

/-- Unit `u`'s checked word at position `p`, as a function of the noise, with the inputs `A = g(F₁)`. -/
def word (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ) :
    Noise N m n k → ℚ :=
  fun ω => (checked k m n N R (A ω.2) B ω u p : ℚ)

/-- `P` on inputs `L` outputs every non-final checked word of the units in `U`: each equals a slot at every grid
point. -/
def Outputs (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (P : List (Instr (Noise N m n k))) (L : List (Noise N m n k → ℚ)) : Prop :=
  ∀ u ∈ U, ∀ p < nonFinal k m n, word k m n N R A B u p ∈ slots P L

/-- **MinRank ≥ 2**: no combination `Σ_τ q_τ·M_τ` of the non-final checkpoint matrices is a nonzero rank-1
matrix `x·yᵀ` on `{0, …, k − 1}²`. -/
def MinRank2 (k : ℕ) (R : Mat) : Prop :=
  ∀ q x y : ℕ → ℚ,
    (∀ a < k, ∀ b < k, ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k R (16 * (t + 1)) a b : ℚ) = x a * y b) →
    (∀ a < k, x a = 0) ∨ (∀ b < k, y b = 0)

/-! ## Pinned statements -/

/-- **One dimension per priced value** (survey §2.3 step 1): `dim V(slots) ≤ dim V(inputs) + #priced`. -/
def DimLePriced : Prop :=
  ∀ (G : Type) [Fintype G] (P : List (Instr G)) (L : List (G → ℚ)),
    Module.finrank ℚ (span1 (slots P L)) ≤ Module.finrank ℚ (span1 L) + nPriced P

/-- **The four-point evaluation lemma** (survey §2.7), jointly over units and entries, for every `A = g(F₁)`: if a
combination of non-final checked words lies in `V₀`, then for every unit `u ∈ U` and entry `e`, its coefficients on
`e`'s checkpoints annihilate the checkpoint matrices: `Σ_τ c_τ·M_τ = 0`. -/
def SecondDifference : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (c : ℕ → ℕ → ℚ), U ⊆ range N →
    (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • word k m n N R A B u p) ∈ V0 N m n k →
    ∀ u ∈ U, ∀ e < m * n, ∀ a < k, ∀ b < k,
      ∑ t ∈ range (3 * k / 16 - 1), c u (t * (m * n) + e) * (ckMat k R (16 * (t + 1)) a b : ℚ) = 0

/-- **The priced count** (Theorem D without prices): under checkpoint independence, every program on inputs in `V₀`
that outputs the non-final checked words of the units in `U`, for any `A = g(F₁)`, has at least
`|U|·m·n·(3k/16 − 1)` priced instructions. -/
def PricedCount : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → Outputs k m n N R A B U P L →
    U.card * (m * n) * (3 * k / 16 - 1) ≤ nPriced P

/-- **Theorem D** (`algBound`): under checkpoint independence and H_16, every program on inputs in `V₀` that outputs
the non-final checked words of the units in `U` (sharing `F₁`, with `A = g(F₁)`) costs at least
`16·|U|·m·n·(3k/16 − 1)`. -/
def AlgBound : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))),
    CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P → Outputs k m n N R A B U P L →
    16 * (U.card * (m * n) * (3 * k / 16 - 1)) ≤ cost P

/-- **TT_NCP in AlgM_16** at `γ₀ = 16/(3k)`, in the shape of TT_NCP's event (`Pouw.NCP.Assumptions.TTNCP`, used by
`gammaFromTTNCP`): for every `γ₀ ∈ [16/(3k), 1)`, a program of AlgM_16 on inputs in `V₀` that outputs the non-final
checked words of the units in `U`, for any `A = g(F₁)`, never has `cost/(1 − γ₀) < Σ_{u ∈ U} 3·W_mm`, with
`W_mm = m·k·n`. The event is impossible, not only unlikely: correctness is on the whole grid. -/
def TTNCPAlg : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ)
    (L : List (Noise N m n k → ℚ)) (P : List (Instr (Noise N m n k))) (γ₀ : ℝ),
    16 ∣ k → 0 < k → CheckpointIndependent k 16 R → U ⊆ range N → (∀ f ∈ L, f ∈ V0 N m n k) → H16 P →
    Outputs k m n N R A B U P L → 16 / (3 * k : ℝ) ≤ γ₀ → γ₀ < 1 →
    ¬ ((cost P : ℝ) / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ))

/-- **Theorem D's hypotheses hold jointly, and its bound is attained**: for every `k ≥ 1024` with `16 ∣ k`, every
shape, `A = g(F₁)`, `B` and unit set, some `R` passes `Admissible` and is checkpoint independent (the shift by 24,
`independentWitness`), the noise coordinates lie in `V₀`, and some program on them satisfies H_16, outputs the
non-final checked words, and costs exactly `16·|U|·m·n·(3k/16 − 1)` (one priced instruction at 16 per word). -/
def AlgWitness : Prop :=
  ∀ k, 1024 ≤ k → 16 ∣ k → ∀ (m n N : ℕ) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ),
    U ⊆ range N →
    ∃ (R : Mat) (P : List (Instr (Noise N m n k))), Admissible k 16 R ∧ CheckpointIndependent k 16 R ∧
      (∀ f ∈ coords N m n k, f ∈ V0 N m n k) ∧ H16 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧
      cost P = 16 * (U.card * (m * n) * (3 * k / 16 - 1))

/-- **`γ₀ = 16/(3k)` is sharp in AlgM_16**: for every `γ₀ < 16/(3k)`, with at least one unit and `m·n > 0`, the
program of `AlgWitness` makes TT_NCP's event hold, for every `R`, `A` and `B`. -/
def AlgSharp : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) (γ₀ : ℝ),
    16 ∣ k → 0 < k → U ⊆ range N → U.Nonempty → 0 < m * n → γ₀ < 16 / (3 * k : ℝ) →
    ∃ P : List (Instr (Noise N m n k)), H16 P ∧ Outputs k m n N R A B U P (coords N m n k) ∧
      (cost P : ℝ) / (1 - γ₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ)

end Pouw.Dimension
