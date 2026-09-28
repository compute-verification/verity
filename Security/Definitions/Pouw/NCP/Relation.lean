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

/-- **RelationFree** (the weaker clause; not the rule): for every non-final checkpoint `p` (depth `d(p + 1) < 3k`), `M_{d(p+1)}` is not
`α·M_{d(q+1)} + β·M_{d(q'+1)}` with `α, β ∈ {−1, 0, 1}` and earlier `q`, `q'` (a zero coefficient drops its term; with
both zero this says `M_{d(p+1)} ≠ 0`). -/
def RelationFree (k d : ℕ) (R : Mat) : Prop :=
  ∀ p, d * (p + 1) < 3 * k → ∀ (α β : ℤ) (q q' : ℕ), (α = 0 ∨ α = 1 ∨ α = -1) → (β = 0 ∨ β = 1 ∨ β = -1) →
    (α ≠ 0 → q < p) → (β ≠ 0 → q' < p) →
    ¬ ∀ a < k, ∀ b < k,
      ckMat k R (d * (p + 1)) a b = α * ckMat k R (d * (q + 1)) a b + β * ckMat k R (d * (q' + 1)) a b

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

/-- **The reduction, grouped by entry** (every input): if a rule is an identity in the noise at position `p`, then for
every entry `e` the coefficients agree: position `p`'s coefficient is the rule applied to the operands' coefficients,
the constant dropped. -/
def ReductionOnlyIf : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (u p : ℕ) (r : Word.Rule), u < N →
    (∀ ω : Noise N m n k, checked k m n N R A B ω u p = r.target (checked k m n N R A B ω u)) →
    ∀ e < m * n, ∀ a < k, ∀ b < k,
      coefAt k m n R p e a b = r.target (fun q => coefAt k m n R q e a b) - r.target (fun _ => 0)

/-- **The reduction, converse** (zero input `A = B = 0`, where the checked value is bilinear in the noise): if the
coefficients agree for every entry and the rule's constant is `0`, the rule is an identity at `p`. -/
def ReductionIf : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (u p : ℕ) (r : Word.Rule), 0 < m * n →
    (∀ e < m * n, ∀ a < k, ∀ b < k,
      coefAt k m n R p e a b = r.target (fun q => coefAt k m n R q e a b) - r.target (fun _ => 0)) →
    r.target (fun _ => 0) = 0 →
    ∀ ω : Noise N m n k, checked k m n N R (fun _ => 0) (fun _ => 0) ω u p =
      r.target (checked k m n N R (fun _ => 0) (fun _ => 0) ω u)

/-- **FinalOnly for NCP**: with a relation-free `R`, every unit has at most `m·n` identity positions (the final ones). -/
def NCPFinalOnly : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k → RelationFree k 16 R →
    Word.Valid (fun _ => positions k m n) adv →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (checked k m n N R A B) adv u ≤ m * n

/-- **The floor for NCP** (`wordFloor` without `FinalOnly`): `N·2^−(m n·(γ₀·3k/16 − 1))`. -/
def NCPWordFloor : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (γ₀ : ℝ), 16 ∣ k → RelationFree k 16 R →
    0 ≤ γ₀ → γ₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    Word.NzNCP (fun _ => positions k m n) (checked k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - γ₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (checked k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1)))

/-- **A relation-free admissible `R` exists** (the witness for the clause): for every `k ≥ 1024` with `16 ∣ k`, some
`R` passes the three clauses of `Admissible` and is relation-free. -/
def RelationFreeWitness : Prop := ∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ RelationFree k 16 R

/-- **CheckpointIndependent** (the rule): the non-final checkpoint matrices `M_{d(p+1)}` (`d(p + 1) < 3k`) are linearly
independent over `ℤ` on `{0, …, k − 1}²`. -/
def CheckpointIndependent (k d : ℕ) (R : Mat) : Prop :=
  ∀ (n : ℕ) (c : ℕ → ℤ), d * n < 3 * k →
    (∀ a < k, ∀ b < k, ∑ p ∈ range n, c p * ckMat k R (d * (p + 1)) a b = 0) → ∀ p < n, c p = 0

/-- Independence implies relation-free. -/
def IndependentRelationFree : Prop := ∀ (k : ℕ) (R : Mat), CheckpointIndependent k 16 R → RelationFree k 16 R

/-- **FinalOnly for NCP under independence**: every unit has at most `m·n` identity positions (the final ones). -/
def NCPFinalOnlyInd : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k → CheckpointIndependent k 16 R →
    Word.Valid (fun _ => positions k m n) adv →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (checked k m n N R A B) adv u ≤ m * n

/-- **The floor for NCP under independence** (`wordFloor` without `FinalOnly`). -/
def NCPWordFloorInd : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (γ₀ : ℝ), 16 ∣ k → CheckpointIndependent k 16 R →
    0 ≤ γ₀ → γ₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    Word.NzNCP (fun _ => positions k m n) (checked k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - γ₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (checked k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1)))

end Pouw.NCP.Relation
