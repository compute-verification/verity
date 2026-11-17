module

public import Definitions.Pouw.NCP.Protocol
public import Definitions.Pouw.NCP.Word

/-! Relation-free `R`, the reduction, and FinalOnly for NCP; `CheckpointIndependent` is a hypothesis of the NCP instance, not a clause of `Admissible`, and `RelationFree` is a weaker lemma. -/

@[expose] public section

namespace Pouw.NCP.Relation

open Finset

def ck (k : ℕ) (A B E1 F1 R : Mat) (i j τ : ℕ) : ℤ :=
  ∑ l ∈ range τ, stackX k A E1 R i l * stackY k B R F1 l j

def unitAt (r c : ℕ) : Mat := fun x y => if x = r ∧ y = c then 1 else 0

def ckMat (k : ℕ) (R : Mat) (τ a b : ℕ) : ℤ := ck k 0 0 (unitAt 0 a) (unitAt b 0) R 0 0 τ

/-- The weaker clause, not the rule. -/
def RelationFree (k d : ℕ) (R : Mat) : Prop :=
  ∀ p, d * (p + 1) < 3 * k → ∀ (α β : ℤ) (q q' : ℕ), (α = 0 ∨ α = 1 ∨ α = -1) → (β = 0 ∨ β = 1 ∨ β = -1) →
    (α ≠ 0 → q < p) → (β ≠ 0 → q' < p) →
    ¬ ∀ a < k, ∀ b < k,
      ckMat k R (d * (p + 1)) a b = α * ckMat k R (d * (q + 1)) a b + β * ckMat k R (d * (q' + 1)) a b

/-- Entry `v : Fin 64` stands for `v − 31`. -/
abbrev Noise (N m n k : ℕ) : Type := (Fin N → Fin m → Fin k → Fin 64) × (Fin k → Fin n → Fin 64)

def e1Of {N m n k : ℕ} (ω : Noise N m n k) (u : ℕ) : Mat := fun i l =>
  if h : u < N ∧ i < m ∧ l < k then ((ω.1 ⟨u, h.1⟩ ⟨i, h.2.1⟩ ⟨l, h.2.2⟩ : ℕ) : ℤ) - 31 else 0

def f1Of {N m n k : ℕ} (ω : Noise N m n k) : Mat := fun l j =>
  if h : l < k ∧ j < n then ((ω.2 ⟨l, h.1⟩ ⟨j, h.2⟩ : ℕ) : ℤ) - 31 else 0

def positions (k m n : ℕ) : ℕ := 3 * k / 16 * (m * n)

def checked (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ) : ℤ :=
  ck k (A u) (B u) (e1Of ω u) (f1Of ω) R (p % (m * n) / n) (p % (m * n) % n) (16 * (p / (m * n) + 1))

def coefAt (k m n : ℕ) (R : Mat) (q e a b : ℕ) : ℤ :=
  if q % (m * n) = e then ckMat k R (16 * (q / (m * n) + 1)) a b else 0

def ReductionOnlyIf : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (u p : ℕ) (r : Word.Rule), u < N →
    (∀ ω : Noise N m n k, checked k m n N R A B ω u p = r.target (checked k m n N R A B ω u)) →
    ∀ e < m * n, ∀ a < k, ∀ b < k,
      coefAt k m n R p e a b = r.target (fun q => coefAt k m n R q e a b) - r.target (fun _ => 0)

def ReductionIf : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (u p : ℕ) (r : Word.Rule), 0 < m * n →
    (∀ e < m * n, ∀ a < k, ∀ b < k,
      coefAt k m n R p e a b = r.target (fun q => coefAt k m n R q e a b) - r.target (fun _ => 0)) →
    r.target (fun _ => 0) = 0 →
    ∀ ω : Noise N m n k, checked k m n N R (fun _ => 0) (fun _ => 0) ω u p =
      r.target (checked k m n N R (fun _ => 0) (fun _ => 0) ω u)

def NCPFinalOnly : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k → RelationFree k 16 R →
    Word.Valid (fun _ => positions k m n) adv →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (checked k m n N R A B) adv u ≤ m * n

def NCPWordFloor : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (g₀ : ℝ), 16 ∣ k → RelationFree k 16 R →
    0 ≤ g₀ → g₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    Word.NzNCP (fun _ => positions k m n) (checked k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - g₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (checked k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (g₀ * (3 * k / 16 : ℕ) - 1)))

/-- The witness holds for every `k ≥ 1024` with `16 ∣ k`. -/
def RelationFreeWitness : Prop := ∀ k, 1024 ≤ k → 16 ∣ k → ∃ R, Admissible k 16 R ∧ RelationFree k 16 R

/-- The rule NCP's instance takes as a hypothesis. -/
def CheckpointIndependent (k d : ℕ) (R : Mat) : Prop :=
  ∀ (n : ℕ) (c : ℕ → ℤ), d * n < 3 * k →
    (∀ a < k, ∀ b < k, ∑ p ∈ range n, c p * ckMat k R (d * (p + 1)) a b = 0) → ∀ p < n, c p = 0

def IndependentRelationFree : Prop := ∀ (k : ℕ) (R : Mat), CheckpointIndependent k 16 R → RelationFree k 16 R

def NCPFinalOnlyInd : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k → CheckpointIndependent k 16 R →
    Word.Valid (fun _ => positions k m n) adv →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (checked k m n N R A B) adv u ≤ m * n

def NCPWordFloorInd : Prop :=
  ∀ (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary) (g₀ : ℝ), 16 ∣ k → CheckpointIndependent k 16 R →
    0 ≤ g₀ → g₀ < 1 → Word.Valid (fun _ => positions k m n) adv →
    Word.NzNCP (fun _ => positions k m n) (checked k m n N R A B) →
    pr (fun ω => (Word.cost (fun _ => positions k m n) N adv : ℝ) / (1 - g₀) <
        16 * ∑ _u ∈ Word.correctSet (fun _ => positions k m n) (checked k m n N R A B) N adv ω,
          (positions k m n : ℝ)) ≤
      N * (2 : ℝ) ^ (-((m * n : ℕ) * (g₀ * (3 * k / 16 : ℕ) - 1)))

end Pouw.NCP.Relation
