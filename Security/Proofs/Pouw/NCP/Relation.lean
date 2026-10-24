import Definitions.Pouw.NCP.Relation

/-! The proof-only part of `Definitions.Pouw.NCP.Relation`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.NCP.Relation

open Finset

/-- **RelationFree** (the weaker clause; not the rule): for every non-final checkpoint `p` (depth `d(p + 1) < 3k`), `M_{d(p+1)}` is not
`α·M_{d(q+1)} + β·M_{d(q'+1)}` with `α, β ∈ {−1, 0, 1}` and earlier `q`, `q'` (a zero coefficient drops its term; with
both zero this says `M_{d(p+1)} ≠ 0`). -/
def RelationFree (k d : ℕ) (R : Mat) : Prop :=
  ∀ p, d * (p + 1) < 3 * k → ∀ (α β : ℤ) (q q' : ℕ), (α = 0 ∨ α = 1 ∨ α = -1) → (β = 0 ∨ β = 1 ∨ β = -1) →
    (α ≠ 0 → q < p) → (β ≠ 0 → q' < p) →
    ¬ ∀ a < k, ∀ b < k,
      ckMat k R (d * (p + 1)) a b = α * ckMat k R (d * (q + 1)) a b + β * ckMat k R (d * (q' + 1)) a b

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
