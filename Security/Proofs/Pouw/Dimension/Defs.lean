import Definitions.Pouw.Dimension.Defs

/-! The proof-only part of `Definitions.Pouw.Dimension.Defs`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation

variable {G : Type}

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
