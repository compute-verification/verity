import Proofs.Pouw.NCP.RelationProofs
import Proofs.Pouw.NCP.GameProofs
import Definitions.Pouw.Guarantees.NCP

/-!
# The shift by 24: independent checkpoints, hence relation-free; FinalOnly and the floor under independence

For every non-final checkpoint `p` there is an entry `(a, b)` of `M_τp` that is nonzero while it is `0` at every
earlier checkpoint (a fresh entry). The entry, for `s` the step within the block: `(16s, 16s)` in block 1,
`(16s + 24, 16s − 24) mod k` in block 2, and `(16s + 24, 16s)` in block 3. In each, the inner sum `Σ_l X_{0l}·Y_{l0}`
has exactly two nonzero terms. Fresh entries make the checkpoint matrices triangular, hence independent
(`indep_of_fresh`: the last coefficient of a vanishing combination is read off at its fresh entry), and independence
implies relation-free (`independentRelationFree`).
-/

namespace Pouw.NCP.RelationWitness

open Finset Pouw.NCP.Relation Pouw.NCP.RelationProofs Pouw.NCP.GameProofs

/-- A term of the inner sum of `M_τ[a, b]`. -/
def tm (k : ℕ) (R : Mat) (a b l : ℕ) : ℤ := stackX k 0 (unitAt 0 a) R 0 l * stackY k 0 R (unitAt b 0) l 0

theorem xval (k : ℕ) (R : Mat) {a : ℕ} (ha : a < k) (l : ℕ) :
    stackX k 0 (unitAt 0 a) R 0 l = if l < k then (if l = a then 1 else 0) else if l < 2 * k then R a (l - k)
      else -((if l - 2 * k = a then 1 else 0) + R a (l - 2 * k)) := by
  simp only [stackX, hcat3]
  split_ifs <;> simp_all [E3, E2, mm_unitL k R ha, unitAt]

theorem yval (k : ℕ) (R : Mat) {b : ℕ} (hb : b < k) (l : ℕ) :
    stackY k 0 R (unitAt b 0) l 0 = if l < k then (if l = b then 1 else 0) else if l < 2 * k then -R (l - k) b
      else (if l - 2 * k = b then 1 else 0) - R (l - 2 * k) b := by
  simp only [stackY, vcat3]
  split_ifs <;> simp_all [F3, F2, mm_unitR k R hb, unitAt, sub_eq_add_neg]

theorem ckMat_two (k : ℕ) (R : Mat) (a b h1 h2 : ℕ) (hne : h1 ≠ h2)
    (hz : ∀ l, l ≠ h1 → l ≠ h2 → tm k R a b l = 0) (τ : ℕ) :
    ckMat k R τ a b = (if h1 < τ then tm k R a b h1 else 0) + (if h2 < τ then tm k R a b h2 else 0) := by
  have hpt : ∀ l, tm k R a b l = (if l = h1 then tm k R a b h1 else 0) + (if l = h2 then tm k R a b h2 else 0) := by
    intro l
    by_cases e1 : l = h1
    · subst e1; simp [hne]
    · by_cases e2 : l = h2
      · subst e2; simp [e1]
      · simp [e1, e2, hz l e1 e2]
  have : ckMat k R τ a b = ∑ l ∈ range τ, tm k R a b l := rfl
  rw [this, sum_congr rfl fun l _ => hpt l, sum_add_distrib, sum_ite_eq', sum_ite_eq']
  simp only [mem_range]

theorem probe1 {k s : ℕ} (hk : 1024 ≤ k) (hs : 16 * s < k) (τ : ℕ) :
    ckMat k (shiftBy 24 k) τ (16 * s) (16 * s) = (if 16 * s < τ then 1 else 0) + (if 2 * k + 16 * s < τ then -1 else 0) := by
  rw [ckMat_two k _ _ _ (16 * s) (2 * k + 16 * s) (by omega)]
  · have v1 : tm k (shiftBy 24 k) (16 * s) (16 * s) (16 * s) = 1 := by
      simp only [tm, xval k _ hs, yval k _ hs, shiftBy]; split_ifs <;> omega
    have v2 : tm k (shiftBy 24 k) (16 * s) (16 * s) (2 * k + 16 * s) = -1 := by
      simp only [tm, xval k _ hs, yval k _ hs, shiftBy]; split_ifs <;> omega
    rw [v1, v2]
  · intro l h1 h2
    simp only [tm, xval k _ hs, yval k _ hs, shiftBy]; split_ifs <;> omega

theorem probe2 {k s a b : ℕ} (hk : 1024 ≤ k) (hs : 16 * s < k) (ha : a < k) (hb : b < k)
    (hae : a = 16 * s + 24 ∨ a + k = 16 * s + 24) (hbe : b + 24 = 16 * s ∨ b + 24 = 16 * s + k) (τ : ℕ) :
    ckMat k (shiftBy 24 k) τ a b = (if k + 16 * s < τ then -1 else 0) + (if 2 * k + 16 * s < τ then 1 else 0) := by
  rw [ckMat_two k _ _ _ (k + 16 * s) (2 * k + 16 * s) (by omega)]
  · have v1 : tm k (shiftBy 24 k) a b (k + 16 * s) = -1 := by
      simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega
    have v2 : tm k (shiftBy 24 k) a b (2 * k + 16 * s) = 1 := by
      simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega
    rw [v1, v2]
  · intro l h1 h2
    simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega

theorem probe3 {k s : ℕ} (hk : 1024 ≤ k) (hs : 16 * s + 24 < k) (τ : ℕ) :
    ckMat k (shiftBy 24 k) τ (16 * s + 24) (16 * s) =
      (if 2 * k + 16 * s < τ then -1 else 0) + (if 2 * k + 16 * s + 24 < τ then 1 else 0) := by
  have ha : 16 * s + 24 < k := hs
  have hb : 16 * s < k := by omega
  rw [ckMat_two k _ _ _ (2 * k + 16 * s) (2 * k + 16 * s + 24) (by omega)]
  · have v1 : tm k (shiftBy 24 k) (16 * s + 24) (16 * s) (2 * k + 16 * s) = -1 := by
      simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega
    have v2 : tm k (shiftBy 24 k) (16 * s + 24) (16 * s) (2 * k + 16 * s + 24) = 1 := by
      simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega
    rw [v1, v2]
  · intro l h1 h2
    simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega

/-- A fresh entry at every non-final checkpoint makes `R` relation-free. -/
theorem of_fresh {k : ℕ} {R : Mat}
    (h : ∀ p, 16 * (p + 1) < 3 * k → ∃ a < k, ∃ b < k,
      ckMat k R (16 * (p + 1)) a b ≠ 0 ∧ ∀ q < p, ckMat k R (16 * (q + 1)) a b = 0) :
    RelationFree k 16 R := by
  intro p hp α β q q' _ _ hαq hβq hall
  obtain ⟨a, ha, b, hb, hne, hz⟩ := h p hp
  apply hne
  have e1 : α * ckMat k R (16 * (q + 1)) a b = 0 := by
    by_cases h0 : α = 0
    · rw [h0, zero_mul]
    · rw [hz q (hαq h0), mul_zero]
  have e2 : β * ckMat k R (16 * (q' + 1)) a b = 0 := by
    by_cases h0 : β = 0
    · rw [h0, zero_mul]
    · rw [hz q' (hβq h0), mul_zero]
  rw [hall a ha b hb, e1, e2, add_zero]

theorem fresh {k : ℕ} (hk : 1024 ≤ k) (h16 : 16 ∣ k) (p : ℕ) (hp : 16 * (p + 1) < 3 * k) :
    ∃ a < k, ∃ b < k, ckMat k (shiftBy 24 k) (16 * (p + 1)) a b ≠ 0 ∧
      ∀ q < p, ckMat k (shiftBy 24 k) (16 * (q + 1)) a b = 0 := by
  obtain ⟨c, rfl⟩ := h16
  by_cases h1 : p < c
  · refine ⟨16 * p, by omega, 16 * p, by omega, ?_, fun q hq => ?_⟩
    · rw [probe1 hk (by omega)]; split_ifs <;> omega
    · rw [probe1 hk (by omega)]; split_ifs <;> omega
  · by_cases h2 : p < 2 * c
    · obtain ⟨a, ha, hae⟩ : ∃ a, a < 16 * c ∧ (a = 16 * (p - c) + 24 ∨ a + 16 * c = 16 * (p - c) + 24) := by
        by_cases h : 16 * (p - c) + 24 < 16 * c
        · exact ⟨_, h, Or.inl rfl⟩
        · exact ⟨16 * (p - c) + 24 - 16 * c, by omega, Or.inr (by omega)⟩
      obtain ⟨b, hb, hbe⟩ : ∃ b, b < 16 * c ∧ (b + 24 = 16 * (p - c) ∨ b + 24 = 16 * (p - c) + 16 * c) := by
        by_cases h : 24 ≤ 16 * (p - c)
        · exact ⟨16 * (p - c) - 24, by omega, Or.inl (by omega)⟩
        · exact ⟨16 * (p - c) + 16 * c - 24, by omega, Or.inr (by omega)⟩
      refine ⟨a, ha, b, hb, ?_, fun q hq => ?_⟩
      · rw [probe2 hk (by omega) ha hb hae hbe]; split_ifs <;> omega
      · rw [probe2 hk (by omega) ha hb hae hbe]; split_ifs <;> omega
    · refine ⟨16 * (p - 2 * c) + 24, by omega, 16 * (p - 2 * c), by omega, ?_, fun q hq => ?_⟩
      · rw [probe3 hk (by omega)]; split_ifs <;> omega
      · rw [probe3 hk (by omega)]; split_ifs <;> omega

theorem relationFreeWitness : RelationFreeWitness := fun k hk h16 =>
  ⟨shiftBy 24 k, shiftBy24_admissible hk h16, of_fresh (fresh hk h16)⟩

theorem independentRelationFree : IndependentRelationFree := by
  intro k R hI p hp α β q q' _ _ hαq hβq hall
  have hc := hI (p + 1) (fun x => (if x = p then 1 else 0) - (if α ≠ 0 ∧ x = q then α else 0) -
    (if β ≠ 0 ∧ x = q' then β else 0)) (by omega) (fun a ha b hb => by
      simp only [sub_mul, ite_mul, one_mul, zero_mul, sum_sub_distrib, sum_ite_eq', mem_range,
        Nat.lt_succ_self, if_true]
      rw [hall a ha b hb]
      have e1 : ∑ x ∈ range (p + 1), (if α ≠ 0 ∧ x = q then α * ckMat k R (16 * (x + 1)) a b else 0) =
          α * ckMat k R (16 * (q + 1)) a b := by
        by_cases h0 : α = 0
        · simp [h0]
        · simp only [h0, ne_eq, not_false_eq_true, true_and]
          rw [sum_ite_eq']
          simp [mem_range, show q < p + 1 by have := hαq h0; omega]
      have e2 : ∑ x ∈ range (p + 1), (if β ≠ 0 ∧ x = q' then β * ckMat k R (16 * (x + 1)) a b else 0) =
          β * ckMat k R (16 * (q' + 1)) a b := by
        by_cases h0 : β = 0
        · simp [h0]
        · simp only [h0, ne_eq, not_false_eq_true, true_and]
          rw [sum_ite_eq']
          simp [mem_range, show q' < p + 1 by have := hβq h0; omega]
      rw [e1, e2]
      ring) p (Nat.lt_succ_self p)
  simp only [if_true] at hc
  have h1 : ¬ (α ≠ 0 ∧ p = q) := fun h => by have := hαq h.1; omega
  have h2 : ¬ (β ≠ 0 ∧ p = q') := fun h => by have := hβq h.1; omega
  simp [h1, h2] at hc

/-- A fresh entry at every non-final checkpoint makes the checkpoints independent (the triangular argument). -/
theorem indep_of_fresh {k : ℕ} {R : Mat}
    (h : ∀ p, 16 * (p + 1) < 3 * k → ∃ a < k, ∃ b < k,
      ckMat k R (16 * (p + 1)) a b ≠ 0 ∧ ∀ q < p, ckMat k R (16 * (q + 1)) a b = 0) :
    CheckpointIndependent k 16 R := by
  intro n
  induction n with
  | zero => intro c _ _ p hp; omega
  | succ n ih =>
    intro c hn hsum
    have hlast : c n = 0 := by
      obtain ⟨a, ha, b, hb, hne, hz⟩ := h n (by omega)
      have := hsum a ha b hb
      rw [sum_range_succ, sum_eq_zero (fun q hq => by rw [hz q (mem_range.1 hq), mul_zero]), zero_add] at this
      exact (mul_eq_zero.1 this).resolve_right hne
    have hsum' : ∀ a < k, ∀ b < k, ∑ p ∈ range n, c p * ckMat k R (16 * (p + 1)) a b = 0 := by
      intro a ha b hb
      have := hsum a ha b hb
      rwa [sum_range_succ, hlast, zero_mul, add_zero] at this
    intro p hp
    rcases Nat.lt_succ_iff_lt_or_eq.1 hp with h' | h'
    · exact ih c (by omega) hsum' p h'
    · rw [h']; exact hlast

theorem independentWitness : Pouw.Guarantees.NCP.IndependentWitness := fun k hk h16 =>
  ⟨shiftBy 24 k, shiftBy24_admissible hk h16, indep_of_fresh (fresh hk h16)⟩

theorem ncpFinalOnlyInd : NCPFinalOnlyInd := fun k m n N R A B adv h16 hI =>
  ncpFinalOnly k m n N R A B adv h16 (independentRelationFree k R hI)

theorem ncpWordFloorInd : NCPWordFloorInd := fun k m n N R A B adv γ₀ h16 hI =>
  ncpWordFloor k m n N R A B adv γ₀ h16 (independentRelationFree k R hI)

end Pouw.NCP.RelationWitness
