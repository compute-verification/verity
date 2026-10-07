import Definitions.Pouw.NCP.Relation
import Proofs.Pouw.NCP.WordProofs

/-!
# The reduction, FinalOnly for NCP, and the floor without FinalOnly
-/

namespace Pouw.NCP.RelationProofs

open Finset Pouw.NCP.Relation

theorem stackX_split (k : ℕ) (A E R : Mat) (i l : ℕ) :
    stackX k A E R i l = stackX k A 0 R i l + stackX k 0 E R i l := by
  simp only [stackX, hcat3, E3, E2]
  split_ifs <;> simp [Pi.add_apply, Pi.neg_apply, mm, add_mul, sum_add_distrib] <;> ring

theorem stackY_split (k : ℕ) (B R F : Mat) (l j : ℕ) :
    stackY k B R F l j = stackY k B R 0 l j + stackY k 0 R F l j := by
  simp only [stackY, vcat3, F3]
  split_ifs <;> simp [Pi.add_apply, F2, mm, mul_add, sum_add_distrib]

theorem stackX_zero (k : ℕ) (R : Mat) (i l : ℕ) : stackX k 0 0 R i l = 0 := by
  simp only [stackX, hcat3, E3, E2]
  split_ifs <;> simp [mm]

theorem stackY_zero (k : ℕ) (R : Mat) (l j : ℕ) : stackY k 0 R 0 l j = 0 := by
  simp only [stackY, vcat3, F3]
  split_ifs <;> simp [F2, mm]

/-- The four-point mixed difference of the checked value is the zero-input value. -/
theorem mixdiff (k : ℕ) (A B E F R : Mat) (i j τ : ℕ) :
    ck k A B E F R i j τ - ck k A B E 0 R i j τ - ck k A B 0 F R i j τ + ck k A B 0 0 R i j τ =
      ck k 0 0 E F R i j τ := by
  simp only [ck, ← sum_sub_distrib, ← sum_add_distrib]
  refine sum_congr rfl fun l _ => ?_
  rw [stackX_split k A E, stackY_split k B R F, stackX_split k A 0, stackY_split k B R 0,
    stackX_zero, stackY_zero]
  ring

theorem stackX_unit (k : ℕ) (R : Mat) (i a x l : ℕ) :
    stackX k 0 (unitAt i a) R x l = if x = i then stackX k 0 (unitAt 0 a) R 0 l else 0 := by
  by_cases hx : x = i
  · subst hx
    simp [stackX, hcat3, E3, E2, mm, unitAt]
  · simp only [hx, if_false]
    simp only [stackX, hcat3, E3]
    split_ifs <;> simp [Pi.add_apply, E2, mm, unitAt, hx]

theorem stackY_unit (k : ℕ) (R : Mat) (b j l y : ℕ) :
    stackY k 0 R (unitAt b j) l y = if y = j then stackY k 0 R (unitAt b 0) l 0 else 0 := by
  by_cases hy : y = j
  · subst hy
    simp [stackY, vcat3, F3, F2, mm, unitAt]
  · simp only [hy, if_false]
    simp only [stackY, vcat3, F3]
    split_ifs <;> simp [Pi.add_apply, F2, mm, unitAt, hy]

/-- Locality: the coefficient of `E₁[i, a]·F₁[b, j]` appears in entry `(i, j)` only. -/
theorem ck_unit (k : ℕ) (R : Mat) (i a b j x y τ : ℕ) :
    ck k 0 0 (unitAt i a) (unitAt b j) R x y τ = if x = i ∧ y = j then ckMat k R τ a b else 0 := by
  simp only [ck, ckMat, stackX_unit k R i a x, stackY_unit k R b j _ y]
  by_cases hx : x = i <;> by_cases hy : y = j <;> simp [hx, hy]

theorem target_affine (r : Word.Rule) (x1 x2 x3 x4 : ℕ → ℤ) :
    r.target x1 - r.target x2 - r.target x3 + r.target x4 =
      r.target (fun q => x1 q - x2 q - x3 q + x4 q) - r.target (fun _ => 0) := by
  cases r <;> simp [Word.Rule.target] <;> ring

/-- The four noise points: entries `31` (value `0`), except `32` (value `1`) at `E₁[u, i, a]` if `bE` and at
`F₁[b, j]` if `bF`. -/
def pt (N m n k : ℕ) (bE bF : Bool) (u i a b j : ℕ) : Noise N m n k :=
  (fun u' i' l' => if bE = true ∧ (u' : ℕ) = u ∧ (i' : ℕ) = i ∧ (l' : ℕ) = a then 32 else 31,
   fun l' j' => if bF = true ∧ (l' : ℕ) = b ∧ (j' : ℕ) = j then 32 else 31)

theorem e1Of_pt {N m n k : ℕ} (bE bF : Bool) {u i a : ℕ} (b j : ℕ) (hu : u < N) (hi : i < m) (ha : a < k) :
    e1Of (pt N m n k bE bF u i a b j) u = if bE then unitAt i a else 0 := by
  funext x y
  cases bE <;> simp only [e1Of, pt] <;> split_ifs <;> simp_all [unitAt] <;> omega

theorem f1Of_pt {N m n k : ℕ} (bE bF : Bool) (u i a : ℕ) {b j : ℕ} (hb : b < k) (hj : j < n) :
    f1Of (pt N m n k bE bF u i a b j) = if bF then unitAt b j else 0 := by
  funext x y
  cases bF <;> simp only [f1Of, pt] <;> split_ifs <;> simp_all [unitAt] <;> omega

theorem checked_pt {k m n N : ℕ} (R : Mat) (A B : ℕ → Mat) (bE bF : Bool) {u i a b j : ℕ} (hu : u < N)
    (hi : i < m) (ha : a < k) (hb : b < k) (hj : j < n) :
    checked k m n N R A B (pt N m n k bE bF u i a b j) u = fun q =>
      ck k (A u) (B u) (if bE then unitAt i a else 0) (if bF then unitAt b j else 0) R
        (q % (m * n) / n) (q % (m * n) % n) (16 * (q / (m * n) + 1)) := by
  funext q
  simp only [checked, e1Of_pt bE bF b j hu hi ha, f1Of_pt bE bF u i a hb hj]

theorem entry_iff {m n e q : ℕ} :
    (q % (m * n) / n = e / n ∧ q % (m * n) % n = e % n) ↔ q % (m * n) = e := by
  constructor
  · rintro ⟨h1, h2⟩
    rw [← Nat.div_add_mod (q % (m * n)) n, ← Nat.div_add_mod e n, h1, h2]
  · rintro h
    rw [h]
    exact ⟨rfl, rfl⟩

theorem coef_eq {k m n : ℕ} (R : Mat) (e a b q : ℕ) :
    ck k 0 0 (unitAt (e / n) a) (unitAt b (e % n)) R (q % (m * n) / n) (q % (m * n) % n)
      (16 * (q / (m * n) + 1)) = coefAt k m n R q e a b := by
  simp only [ck_unit, coefAt, entry_iff]

theorem reductionOnlyIf : ReductionOnlyIf := by
  intro k m n N R A B u p r hu hid e he a ha b hb
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with h | h
    · simp [h] at he
    · exact h
  have hi : e / n < m := (Nat.div_lt_iff_lt_mul hn).2 he
  have hj : e % n < n := Nat.mod_lt e hn
  have h1 := hid (pt N m n k true true u (e / n) a b (e % n))
  have h2 := hid (pt N m n k true false u (e / n) a b (e % n))
  have h3 := hid (pt N m n k false true u (e / n) a b (e % n))
  have h4 := hid (pt N m n k false false u (e / n) a b (e % n))
  rw [checked_pt R A B _ _ hu hi ha hb hj] at h1 h2 h3 h4
  simp only [if_true, Bool.false_eq_true, if_false] at h1 h2 h3 h4
  rw [← coef_eq R e a b p, ← mixdiff, h1, h2, h3, h4, target_affine]
  congr 2
  funext q
  rw [mixdiff, coef_eq R]

theorem mm_unitL (k : ℕ) (R : Mat) {a : ℕ} (ha : a < k) (x : ℕ) : mm k (unitAt 0 a) R 0 x = R a x := by
  simp [mm, unitAt, ha]

theorem mm_unitR (k : ℕ) (R : Mat) {b : ℕ} (hb : b < k) (x : ℕ) : mm k R (unitAt b 0) x 0 = R x b := by
  simp [mm, unitAt, hb]

theorem sel {k : ℕ} (f : ℕ → ℤ) (hf : ∀ x, k ≤ x → f x = 0) (x : ℕ) :
    ∑ a ∈ range k, f a * (if x = a then 1 else 0) = f x := by
  simp only [mul_ite, mul_one, mul_zero]
  rw [sum_ite_eq]
  split_ifs with h
  · rfl
  · exact (hf x (by simpa using h)).symm

theorem stackX_expand (k : ℕ) (E R : Mat) (hE : ∀ i l, k ≤ l → E i l = 0) (i l : ℕ) :
    stackX k 0 E R i l = ∑ a ∈ range k, E i a * stackX k 0 (unitAt 0 a) R 0 l := by
  have hs := sel (k := k) (E i) (hE i)
  simp only [stackX, hcat3]
  split_ifs
  · simp only [Pi.add_apply, Pi.zero_apply, zero_add, unitAt, true_and]
    exact (hs l).symm
  · simp only [Pi.add_apply, Pi.zero_apply, zero_add, E2]
    rw [sum_congr rfl fun a ha => by rw [mm_unitL k R (mem_range.1 ha)]]
    rfl
  · simp only [Pi.add_apply, Pi.zero_apply, zero_add, Pi.neg_apply, E3, E2]
    rw [sum_congr rfl fun a ha => by rw [mm_unitL k R (mem_range.1 ha)]]
    simp only [unitAt, true_and, mul_neg, mul_add, sum_neg_distrib, sum_add_distrib, hs, mm]

theorem stackY_expand (k : ℕ) (F R : Mat) (hF : ∀ l j, k ≤ l → F l j = 0) (l j : ℕ) :
    stackY k 0 R F l j = ∑ b ∈ range k, F b j * stackY k 0 R (unitAt b 0) l 0 := by
  have hs := sel (k := k) (fun b => F b j) (fun x hx => hF x j hx)
  simp only [stackY, vcat3]
  split_ifs
  · simp only [Pi.add_apply, Pi.zero_apply, zero_add, unitAt, and_true]
    exact (hs l).symm
  · simp only [Pi.add_apply, Pi.zero_apply, zero_add, F2]
    rw [sum_congr rfl fun b hb => by rw [mm_unitR k R (mem_range.1 hb)]]
    simp only [mm, mul_neg, sum_neg_distrib, mul_comm]
  · simp only [Pi.add_apply, Pi.zero_apply, zero_add, F3, F2]
    rw [sum_congr rfl fun b hb => by rw [mm_unitR k R (mem_range.1 hb)]]
    simp only [unitAt, and_true, mul_add, mul_neg, sum_neg_distrib, sum_add_distrib, hs, mm, mul_comm,
      sub_eq_add_neg]

/-- The bilinear expansion of the zero-input checked value. -/
theorem ck_expand (k : ℕ) (E F R : Mat) (hE : ∀ i l, k ≤ l → E i l = 0) (hF : ∀ l j, k ≤ l → F l j = 0)
    (i j τ : ℕ) :
    ck k 0 0 E F R i j τ = ∑ a ∈ range k, ∑ b ∈ range k, E i a * F b j * ckMat k R τ a b := by
  simp only [ckMat, ck, stackX_expand k E R hE, stackY_expand k F R hF, sum_mul, mul_sum]
  rw [sum_comm]
  conv_rhs => rw [sum_comm]
  refine sum_congr rfl fun b _ => ?_
  rw [sum_comm]
  refine sum_congr rfl fun a _ => sum_congr rfl fun l _ => ?_
  ring

theorem checked_decomp {k m n N : ℕ} (R : Mat) (ω : Noise N m n k) (u : ℕ) (hmn : 0 < m * n) (q : ℕ) :
    checked k m n N R (fun _ => 0) (fun _ => 0) ω u q =
      ∑ t ∈ range (m * n) ×ˢ (range k ×ˢ range k),
        e1Of ω u (t.1 / n) t.2.1 * f1Of ω t.2.2 (t.1 % n) * coefAt k m n R q t.1 t.2.1 t.2.2 := by
  have hE : ∀ i l, k ≤ l → e1Of ω u i l = 0 := fun i l hl => by
    simp only [e1Of]; rw [dif_neg]; omega
  have hF : ∀ l j, k ≤ l → f1Of ω l j = 0 := fun l j hl => by
    simp only [f1Of]; rw [dif_neg]; omega
  rw [Finset.sum_product (range (m * n)) (range k ×ˢ range k), Finset.sum_eq_single (q % (m * n))]
  · simp only [checked, ck_expand k _ _ R hE hF, sum_product, coefAt, if_true]
  · intro e _ he
    simp [coefAt, Ne.symm he]
  · intro h
    exact absurd (mem_range.2 (Nat.mod_lt q hmn)) h

theorem target_sum (r : Word.Rule) {ι : Type} (s : Finset ι) (c : ι → ℤ) (x : ι → ℕ → ℤ) :
    r.target (fun q => ∑ t ∈ s, c t * x t q) =
      ∑ t ∈ s, c t * (r.target (x t) - r.target (fun _ => 0)) + r.target (fun _ => 0) := by
  cases r <;> simp [Word.Rule.target, mul_sum, sum_add_distrib] <;> (try ring_nf) <;>
    simp only [mul_left_comm, mul_comm, sum_add_distrib]

theorem reductionIf : ReductionIf := by
  intro k m n N R u p r hmn hrel h0 ω
  have hfun : checked k m n N R (fun _ => 0) (fun _ => 0) ω u = fun q =>
      ∑ t ∈ range (m * n) ×ˢ (range k ×ˢ range k),
        e1Of ω u (t.1 / n) t.2.1 * f1Of ω t.2.2 (t.1 % n) * coefAt k m n R q t.1 t.2.1 t.2.2 :=
    funext (checked_decomp R ω u hmn)
  rw [hfun, target_sum, h0, add_zero]
  refine sum_congr rfl fun t ht => ?_
  simp only [mem_product, mem_range] at ht
  rw [hrel t.1 ht.1 t.2.1 ht.2.1 t.2.2 ht.2.2, h0]

theorem div_lt_of_mod_eq {x p c : ℕ} (h : x < p) (hm : x % c = p % c) : x / c < p / c := by
  by_contra hc
  have h1 := Nat.div_add_mod x c
  have h2 := Nat.div_add_mod p c
  have h3 := Nat.mul_le_mul_left c (not_lt.1 hc)
  omega

theorem sgn_opt (c : Prop) [Decidable c] (s : Bool) :
    (if c then Word.sgn s else 0) = 0 ∨ (if c then Word.sgn s else 0) = 1 ∨ (if c then Word.sgn s else 0) = -1 := by
  cases s <;> split_ifs <;> simp [Word.sgn]

/-- No non-final position of a relation-free instance is an identity. -/
theorem not_identity {k m n N : ℕ} {R : Mat} {A B : ℕ → Mat} {adv : Word.Adversary} (hRF : RelationFree k 16 R)
    (hV : Word.Valid (fun _ => positions k m n) adv) {u p : ℕ} (hu : u < N) (hp : p < positions k m n)
    (hnf : 16 * (p / (m * n) + 1) < 3 * k) : ¬ Word.Identity (checked k m n N R A B) adv u p := by
  rintro ⟨hne, hid⟩
  have hred := reductionOnlyIf k m n N R A B u p (adv.rule u p) hu hid
  have hmn : 0 < m * n := by
    rcases Nat.eq_zero_or_pos (m * n) with h | h
    · simp [positions, h] at hp
    · exact h
  have he : p % (m * n) < m * n := Nat.mod_lt p hmn
  have hRd := hV u p hp
  have hq : ∀ (x : ℕ) (s : Bool), x < p →
      (if x % (m * n) = p % (m * n) then Word.sgn s else 0) ≠ 0 → x / (m * n) < p / (m * n) := by
    intro x s hx h
    split_ifs at h with hxe
    · exact div_lt_of_mod_eq hx hxe
    · exact absurd rfl h
  generalize hr : adv.rule u p = r at hne hred hRd
  cases r with
  | fresh => exact hne rfl
  | copy s x =>
    refine hRF (p / (m * n)) hnf _ 0 (x / (m * n)) 0 (sgn_opt _ s) (Or.inl rfl) (hq x s hRd)
      (fun h => absurd rfl h) fun a ha b hb => ?_
    have := hred (p % (m * n)) he a ha b hb
    simp only [coefAt, Word.Rule.target, if_true, mul_zero, sub_zero] at this
    rw [this]
    split_ifs <;> ring
  | sum s x t y =>
    refine hRF (p / (m * n)) hnf _ _ (x / (m * n)) (y / (m * n)) (sgn_opt _ s) (sgn_opt _ t)
      (hq x s hRd.1) (hq y t hRd.2) fun a ha b hb => ?_
    have := hred (p % (m * n)) he a ha b hb
    simp only [coefAt, Word.Rule.target, if_true, mul_zero, add_zero, sub_zero] at this
    rw [this]
    split_ifs <;> ring
  | const c =>
    refine hRF (p / (m * n)) hnf 0 0 0 0 (Or.inl rfl) (Or.inl rfl) (fun h => absurd rfl h)
      (fun h => absurd rfl h) fun a ha b hb => ?_
    have := hred (p % (m * n)) he a ha b hb
    simp only [coefAt, Word.Rule.target, if_true, sub_self] at this
    rw [this]
    ring

/-- An identity position of a relation-free instance is in the final checkpoint. -/
theorem final_of_identity {k m n N : ℕ} {R : Mat} {A B : ℕ → Mat} {adv : Word.Adversary} (h16 : 16 ∣ k)
    (hRF : RelationFree k 16 R) (hV : Word.Valid (fun _ => positions k m n) adv) {u p : ℕ} (hu : u < N)
    (hp : p < positions k m n) (hid : Word.Identity (checked k m n N R A B) adv u p) :
    p / (m * n) = 3 * k / 16 - 1 := by
  have hmn : 0 < m * n := by
    rcases Nat.eq_zero_or_pos (m * n) with h | h
    · simp [positions, h] at hp
    · exact h
  have hlt : p / (m * n) < 3 * k / 16 := (Nat.div_lt_iff_lt_mul hmn).2 hp
  have hf : ¬ 16 * (p / (m * n) + 1) < 3 * k := fun h => not_identity hRF hV hu hp h hid
  obtain ⟨c, rfl⟩ := h16
  generalize p / (m * n) = q at hlt hf ⊢
  omega

theorem ncpFinalOnly : NCPFinalOnly := by
  intro k m n N R A B adv h16 hRF hV u hu
  classical
  unfold Word.identCount
  refine (card_le_card_of_injOn (· % (m * n)) (t := range (m * n)) ?_ ?_).trans (by simp)
  · intro p hp
    have hp := mem_filter.1 (mem_coe.1 hp)
    rw [mem_range] at hp
    rw [mem_coe, mem_range]
    have hmn : 0 < m * n := by
      rcases Nat.eq_zero_or_pos (m * n) with h | h
      · simp [positions, h] at hp
      · exact h
    exact Nat.mod_lt p hmn
  · intro p hp p' hp' h
    have hp := mem_filter.1 (mem_coe.1 hp)
    have hp' := mem_filter.1 (mem_coe.1 hp')
    rw [mem_range] at hp hp'
    have e1 := final_of_identity h16 hRF hV hu hp.1 hp.2
    have e2 := final_of_identity h16 hRF hV hu hp'.1 hp'.2
    simp only at h
    rw [← Nat.div_add_mod p (m * n), ← Nat.div_add_mod p' (m * n), e1, e2, h]

theorem ncpWordFloor : NCPWordFloor := by
  intro k m n N R A B adv γ₀ h16 hRF hγ0 hγ1 hV hNz
  refine (WordProofs.wordBound (Noise N m n k) (fun _ => positions k m n) (checked k m n N R A B) N adv γ₀
    hγ0 hγ1 hV hNz).trans ?_
  calc ∑ u ∈ range N, (2 : ℝ) ^ (-(γ₀ * ((fun _ => positions k m n) u : ℕ) -
        Word.identCount (fun _ => positions k m n) (checked k m n N R A B) adv u))
      ≤ ∑ _u ∈ range N, (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1))) := sum_le_sum fun u hu => by
        apply Real.rpow_le_rpow_of_exponent_le (by norm_num)
        have h1 : (Word.identCount (fun _ => positions k m n) (checked k m n N R A B) adv u : ℝ) ≤ (m * n : ℕ) := by
          exact_mod_cast ncpFinalOnly k m n N R A B adv h16 hRF hV u (mem_range.1 hu)
        generalize Word.identCount (fun _ => positions k m n) (checked k m n N R A B) adv u = ic at h1 ⊢
        simp only [positions]
        push_cast at h1 ⊢
        nlinarith
    _ = N * (2 : ℝ) ^ (-((m * n : ℕ) * (γ₀ * (3 * k / 16 : ℕ) - 1))) := by
        rw [sum_const, card_range, nsmul_eq_mul]

end Pouw.NCP.RelationProofs
