import Definitions.Pouw.Assumptions.Dimension.Mixing
import Proofs.Pouw.NCP.RelationWitness

/-!
# MinRank ≥ 2 for the shift by 24

`M_τ[a, b]` for the shift by 24 vanishes off the three diagonals `a = b`, `a = b + 24` and `a = b + 48` (mod `k`)
(`shift_support`), and on each diagonal its value at a checkpoint depends on the column `b` only through `b / 8`
(`shift_sib`, from the closed forms `gen0`, `gen24`, `gen48`). If a combination were `x·yᵀ` with `x(a₀) ≠ 0` and
`y(b₀) ≠ 0`, the column `b` with `b / 8 = b₀ / 8` would all carry `y(b) ≠ 0`, while row `a₀` meets at most one of
any 8 consecutive columns.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.NCP.RelationProofs Pouw.NCP.GameProofs Pouw.NCP.RelationWitness

theorem shift_support {k a b : ℕ} (hk : 1024 ≤ k) (ha : a < k) (hb : b < k) (h0 : a ≠ b) (h1 : a ≠ b + 24)
    (h2 : a + k ≠ b + 24) (h3 : a ≠ b + 48) (h4 : a + k ≠ b + 48) (τ : ℕ) :
    ckMat k (shiftBy 24 k) τ a b = 0 := by
  have : ckMat k (shiftBy 24 k) τ a b = ∑ l ∈ range τ, tm k (shiftBy 24 k) a b l := rfl
  rw [this]
  refine sum_eq_zero fun l _ => ?_
  simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega

theorem gen0 {k b : ℕ} (hk : 1024 ≤ k) (hb : b < k) (τ : ℕ) :
    ckMat k (shiftBy 24 k) τ b b = (if b < τ then 1 else 0) + (if 2 * k + b < τ then -1 else 0) := by
  rw [ckMat_two k _ _ _ b (2 * k + b) (by omega)]
  · have v1 : tm k (shiftBy 24 k) b b b = 1 := by
      simp only [tm, xval k _ hb, yval k _ hb, shiftBy]; split_ifs <;> omega
    have v2 : tm k (shiftBy 24 k) b b (2 * k + b) = -1 := by
      simp only [tm, xval k _ hb, yval k _ hb, shiftBy]; split_ifs <;> omega
    rw [v1, v2]
  · intro l h1 h2
    simp only [tm, xval k _ hb, yval k _ hb, shiftBy]; split_ifs <;> omega

theorem gen24 {k a b : ℕ} (hk : 1024 ≤ k) (ha : a < k) (hb : b < k) (hab : a = b + 24 ∨ a + k = b + 24)
    (τ : ℕ) :
    ckMat k (shiftBy 24 k) τ a b = (if 2 * k + b < τ then -1 else 0) + (if 2 * k + a < τ then 1 else 0) := by
  rw [ckMat_two k _ _ _ (2 * k + b) (2 * k + a) (by omega)]
  · have v1 : tm k (shiftBy 24 k) a b (2 * k + b) = -1 := by
      simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega
    have v2 : tm k (shiftBy 24 k) a b (2 * k + a) = 1 := by
      simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega
    rw [v1, v2]
  · intro l h1 h2
    simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega

theorem gen48 {k a b j : ℕ} (hk : 1024 ≤ k) (ha : a < k) (hb : b < k) (hj : j < k)
    (haj : a = j + 24 ∨ a + k = j + 24) (hbj : b + 24 = j ∨ b + 24 = j + k) (τ : ℕ) :
    ckMat k (shiftBy 24 k) τ a b = (if k + j < τ then -1 else 0) + (if 2 * k + j < τ then 1 else 0) := by
  rw [ckMat_two k _ _ _ (k + j) (2 * k + j) (by omega)]
  · have v1 : tm k (shiftBy 24 k) a b (k + j) = -1 := by
      simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega
    have v2 : tm k (shiftBy 24 k) a b (2 * k + j) = 1 := by
      simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega
    rw [v1, v2]
  · intro l h1 h2
    simp only [tm, xval k _ ha, yval k _ hb, shiftBy]; split_ifs <;> omega

theorem shift_sib {k : ℕ} (hk : 1024 ≤ k) (h16 : 16 ∣ k) {a0 b0 b : ℕ} (ha0 : a0 < k) (hb0 : b0 < k)
    (hb : b < k) (hbb : b / 8 = b0 / 8)
    (hd : a0 = b0 ∨ a0 = b0 + 24 ∨ a0 + k = b0 + 24 ∨ a0 = b0 + 48 ∨ a0 + k = b0 + 48) :
    ∃ a < k, ∀ t, ckMat k (shiftBy 24 k) (16 * (t + 1)) a b = ckMat k (shiftBy 24 k) (16 * (t + 1)) a0 b0 := by
  obtain ⟨c, rfl⟩ := h16
  rcases hd with h | h | h | h | h
  · refine ⟨b, hb, fun t => ?_⟩
    rw [h, gen0 hk hb, gen0 hk hb0]
    split_ifs <;> omega
  all_goals first
    | (refine ⟨if b + 24 < 16 * c then b + 24 else b + 24 - 16 * c, by split_ifs <;> omega, fun t => ?_⟩
       rw [gen24 hk (by split_ifs <;> omega) hb (by split_ifs <;> omega), gen24 hk ha0 hb0 (by omega)]
       split_ifs <;> omega)
    | (refine ⟨if b + 48 < 16 * c then b + 48 else b + 48 - 16 * c, by split_ifs <;> omega, fun t => ?_⟩
       rw [gen48 (j := if b + 24 < 16 * c then b + 24 else b + 24 - 16 * c) hk (by split_ifs <;> omega) hb
          (by split_ifs <;> omega) (by split_ifs <;> omega) (by split_ifs <;> omega),
        gen48 (j := if b0 + 24 < 16 * c then b0 + 24 else b0 + 24 - 16 * c) hk ha0 hb0
          (by split_ifs <;> omega) (by split_ifs <;> omega) (by split_ifs <;> omega)]
       split_ifs <;> omega)

theorem minRank2_shift {k : ℕ} (hk : 1024 ≤ k) (h16 : 16 ∣ k) : MinRank2 k (shiftBy 24 k) := by
  intro q x y hxy
  by_contra hne
  simp only [not_or, not_forall] at hne
  obtain ⟨⟨a0, ha0, hx0⟩, ⟨b0, hb0, hy0⟩⟩ := hne
  have hcol : ∀ b < k, y b ≠ 0 →
      (a0 = b ∨ a0 = b + 24 ∨ a0 + k = b + 24 ∨ a0 = b + 48 ∨ a0 + k = b + 48) := by
    intro b hb hyb
    by_contra hc
    simp only [not_or] at hc
    obtain ⟨c0, c1, c2, c3, c4⟩ := hc
    have h0 := hxy a0 ha0 b hb
    rw [sum_eq_zero fun t _ => by rw [shift_support hk ha0 hb c0 c1 c2 c3 c4]; simp] at h0
    exact mul_ne_zero hx0 hyb h0.symm
  have hd := hcol b0 hb0 hy0
  have hsib : ∀ b < k, b / 8 = b0 / 8 → y b ≠ 0 := by
    intro b hb hbb hyb
    obtain ⟨a, ha, he⟩ := shift_sib hk h16 ha0 hb0 hb hbb hd
    have h1 := hxy a ha b hb
    have hs : ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k (shiftBy 24 k) (16 * (t + 1)) a b : ℚ) =
        ∑ t ∈ range (3 * k / 16 - 1), q t * (ckMat k (shiftBy 24 k) (16 * (t + 1)) a0 b0 : ℚ) :=
      sum_congr rfl fun t _ => by rw [he t]
    rw [hs, hxy a0 ha0 b0 hb0, hyb, mul_zero] at h1
    exact mul_ne_zero hx0 hy0 h1
  obtain ⟨c, rfl⟩ := h16
  have e1 := hcol (8 * (b0 / 8)) (by omega) (hsib _ (by omega) (by omega))
  have e2 := hcol (8 * (b0 / 8) + 1) (by omega) (hsib _ (by omega) (by omega))
  omega

/-- **MinRank ≥ 2's witness, jointly.** -/
theorem minRank2Witness : MinRank2Witness := fun k hk h16 =>
  ⟨shiftBy 24 k, shiftBy24_admissible hk h16, indep_of_fresh (fresh hk h16), minRank2_shift hk h16⟩

end Pouw.Dimension
