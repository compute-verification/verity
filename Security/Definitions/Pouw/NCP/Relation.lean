import Definitions.Pouw.NCP.Protocol
import Definitions.Pouw.NCP.Word

/-!
# Relation-free R, the reduction, and FinalOnly for NCP (trusted)

At depth `τ` the checked value of entry `(i, j)` is `ck = Σ_{l < τ} X_{il}·Y_{lj}` with `X = stackX k A E₁ R` and
`Y = stackY k B R F₁`. Its coefficient on `E₁[i, a]·F₁[b, j]` is `M_τ[a, b]` (`ckMat`), the checked value at zero input
when `E₁` is `1` only at `(0, a)` and `F₁` is `1` only at `(b, 0)`; that is `M_τ = G·J_τ·H` with `G = [I | R | −(I + R)]`
and `H = [I; −R; I − R]`.

`CheckpointIndependent k d R`: the non-final checkpoint matrices are linearly independent over `ℤ`. This is the clause
the coordinator adopted (red-team Phase 3, P3-1). It is a hypothesis of the NCP instance below, not a clause of
`Admissible`. `RelationFree k d R` (no non-final checkpoint matrix is `0`, `±M_q` or `±M_q ± M_{q'}` for earlier
checkpoints) is the weaker clause proposed after `Pouw.NCP.ExtraIdentity`; it is kept as a lemma, implied by
independence, and is not the rule: a twisted-block `R` is relation-free yet has linear relations with scalars
(red-team Phase 3).

The NCP instance of the word model (`Pouw.NCP.Word`): `N` units of `m` rows share one weight; `E₁` is per unit and
`F₁` is shared, entries in `[−31, 32]`; each unit has `3k/16` checkpoints (depths `16, 32, …, 3k`) of `m·n` entries, in
block-major order. The inputs `A u`, `B u` are arbitrary.
-/

namespace Pouw.NCP.Relation

open Finset

/-- The checked value of entry `(i, j)` at depth `τ`. -/
def ck (k : ℕ) (A B E1 F1 R : Mat) (i j τ : ℕ) : ℤ :=
  ∑ l ∈ range τ, stackX k A E1 R i l * stackY k B R F1 l j

/-- The matrix with a single `1` at `(r, c)`. -/
def unitAt (r c : ℕ) : Mat := fun x y => if x = r ∧ y = c then 1 else 0

/-- `M_τ[a, b]`, the coefficient of `E₁[i, a]·F₁[b, j]` in the checked value of `(i, j)` at depth `τ`. -/
def ckMat (k : ℕ) (R : Mat) (τ a b : ℕ) : ℤ := ck k 0 0 (unitAt 0 a) (unitAt b 0) R 0 0 τ

/-- The noise: `E₁` per unit (`m × k`) and one shared `F₁` (`k × n`); entry `v : Fin 64` stands for `v − 31`. -/
abbrev Noise (N m n k : ℕ) : Type := (Fin N → Fin m → Fin k → Fin 64) × (Fin k → Fin n → Fin 64)

/-- Unit `u`'s `E₁`. -/
def e1Of {N m n k : ℕ} (ω : Noise N m n k) (u : ℕ) : Mat := fun i l =>
  if h : u < N ∧ i < m ∧ l < k then ((ω.1 ⟨u, h.1⟩ ⟨i, h.2.1⟩ ⟨l, h.2.2⟩ : ℕ) : ℤ) - 31 else 0

/-- The shared `F₁`. -/
def f1Of {N m n k : ℕ} (ω : Noise N m n k) : Mat := fun l j =>
  if h : l < k ∧ j < n then ((ω.2 ⟨l, h.1⟩ ⟨j, h.2⟩ : ℕ) : ℤ) - 31 else 0

/-- Positions per unit: `3k/16` checkpoints of `m·n` entries. -/
def positions (k m n : ℕ) : ℕ := 3 * k / 16 * (m * n)

/-- Position `p` of unit `u`: checkpoint `p / (m n)` (depth `16·(p / (m n) + 1)`), entry `e = p % (m n)`, which is
`(i, j) = (e / n, e % n)`. -/
def checked (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ) : ℤ :=
  ck k (A u) (B u) (e1Of ω u) (f1Of ω) R (p % (m * n) / n) (p % (m * n) % n) (16 * (p / (m * n) + 1))

/-- The coefficient of `E₁[e / n, a]·F₁[b, e % n]` in position `q`'s value: `M_{τ_q}[a, b]` if `q` has entry `e`,
else `0`. -/
def coefAt (k m n : ℕ) (R : Mat) (q e a b : ℕ) : ℤ :=
  if q % (m * n) = e then ckMat k R (16 * (q / (m * n) + 1)) a b else 0

/-! ## Pinned statements -/

/-- **CheckpointIndependent** (the rule): the non-final checkpoint matrices `M_{d(p+1)}` (`d(p + 1) < 3k`) are linearly
independent over `ℤ` on `{0, …, k − 1}²`. -/
def CheckpointIndependent (k d : ℕ) (R : Mat) : Prop :=
  ∀ (n : ℕ) (c : ℕ → ℤ), d * n < 3 * k →
    (∀ a < k, ∀ b < k, ∑ p ∈ range n, c p * ckMat k R (d * (p + 1)) a b = 0) → ∀ p < n, c p = 0

end Pouw.NCP.Relation
