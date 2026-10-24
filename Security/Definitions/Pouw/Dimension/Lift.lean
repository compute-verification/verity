import Definitions.Pouw.Dimension.Defs

/-!
# `Lift_MinRank` (red-team P7-3): a named placeholder, and MinRank ≥ ρ (trusted statements)

The chain's probabilistic node (TD-4), as the lifting worker states it (`internal/pouw/new-crypto/lifting/README.md`):
a program on inputs in `V₀` with `p` priced slots that outputs every non-final checked word correctly on a set `Z` of
noise values (whether by lottery, grinding or noise-dependent branching) has `Pr_grid[Z] ≤ ε(d)`, with `d = D − p`
the saving, under checkpoint independence and MinRank ≥ ρ. Here `ε` and `ρ` are parameters. The lifting worker proves
it on paper with L5e's exponential `ε` (`epsL5e`, stated as `LiftMinRankL5e`); in Lean only its lemmas L2, L3 and L4's
counting core are proved (`Pouw.Dimension.Lifting`).

`MinRankGe k R ρ` is `minrank.md`'s Lean-ready form of MinRank ≥ ρ: every nonzero combination of the non-final
checkpoint matrices has a `ρ × ρ` triangular minor with nonzero diagonal.

The named `Prop`s `LiftMinRank` and `LiftMinRankL5e` are in the assumptions module `Pouw.Dimension.PipeAssumptions`,
and `LiftMinRankWitness` is in `Pouw.Dimension.Pipe`.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation

/-- **MinRank ≥ ρ** (`minrank.md` §3): every combination `X = Σ_t q_t·M_t` of the non-final checkpoint matrices that
is nonzero on `{0, …, k − 1}²` has distinct rows `r i` and distinct columns `s i` (`i < ρ`) with `X[r i, s i] ≠ 0`
and `X[r i, s l] = 0` for `i < l`, a triangular minor with nonzero diagonal, hence rank ≥ ρ. -/
def MinRankGe (k : ℕ) (R : Mat) (ρ : ℕ) : Prop :=
  ∀ q : ℕ → ℚ, ∀ a < k, ∀ b < k,
    ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k R (16 * (t + 1)) a b : ℚ) ≠ 0 →
    ∃ r s : Fin ρ → ℕ, Function.Injective r ∧ Function.Injective s ∧ (∀ i, r i < k ∧ s i < k) ∧
      (∀ i, ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k R (16 * (t + 1)) (r i) (s i) : ℚ) ≠ 0) ∧
      (∀ i l, i < l → ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k R (16 * (t + 1)) (r i) (s l) : ℚ) = 0)

/-- **L5e's ε** (`lifting/README.md`): `((1 + 64^(Λ − ρ + G − 1)) / 64)^X` with
`X = ⌈d / ((n·T + k)/G + T + M·T/Λ)⌉`, where `T` is the number of non-final checkpoints per entry and `M` the number of
rows (`N·m` for `N` units sharing `F₁`, L6). -/
noncomputable def epsL5e (ρ G Λ k M n T : ℕ) (d : ℤ) : ℝ :=
  ((1 + (64 : ℝ) ^ ((Λ : ℤ) - ρ + G - 1)) / 64) ^
    ⌈(d : ℝ) / (((n * T + k : ℕ) : ℝ) / G + T + ((M * T : ℕ) : ℝ) / Λ)⌉

end Pouw.Dimension
