import Mathlib.Tactic.IntervalCases
import Proofs.Pouw.Fp8Atom.H1TAtomProofs

/-!
# One slice: L1 to L3 (`H1TSlice`), the chain's monotonicity (`H1TChainMono`) and the range (`H1TRange`)

A slice of finite codes has `F = fOf fi` for some `fi ≤ 4`, and each of its codes is below `2^(fi + 5)` (`floorF_index`).
The forming table gives every formed code's magnitude and exponent field (`form_facts`), and the tag table every tag
code's (`tag_facts`): a tag code has exponent field `fi + 10` and significand `8 + m` for either sign. So every product
exponent is at most `fi + 13`, the grid `2^(E − 13)` is at most `2^fi`, and a tag product, `2^fi` times an integer, is
on it (L2). A tag flip changes one code's sign only, so no product's magnitude or exponent moves (L1).
-/

namespace Pouw.Fp8Atom.H1T.Proofs

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T

/-! ## Codes -/

theorem intVal_abs_le : ∀ c : Code, c.isNaN = false → |intVal c| ≤ 229376 := by decide +kernel

theorem val_eq_intVal {c : Code} (h : c.isNaN = false) : val c = intVal c / 512 := by
  rw [val, decode_eq c h]; rfl

theorem val_abs_le {c : Code} (h : c.isNaN = false) : |val c| ≤ 448 := by
  rw [val_eq_intVal h, abs_div, abs_of_pos (by norm_num : (0 : ℚ) < 512), div_le_iff₀ (by norm_num)]
  have : |(intVal c : ℚ)| ≤ 229376 := by rw [← Int.cast_abs]; exact_mod_cast intVal_abs_le c h
  linarith

theorem product_exp_eq {a b : Code} (h : (product a b).mag ≠ 0) :
    (product a b).exp = a.expOr1 + b.expOr1 - 14 := by
  unfold product at *; dsimp only at *
  split_ifs at * with h'
  · simp [Term.zero] at h
  · rfl

theorem product_mag_exp_congr {a a' b : Code} (hs : a.sig = a'.sig) (he : a.expOr1 = a'.expOr1) :
    (product a b).mag = (product a' b).mag ∧ (product a b).exp = (product a' b).exp := by
  unfold product; dsimp only; rw [hs, he]; constructor <;> split_ifs <;> rfl

/-! ## The tag tuples -/

theorem A0_facts : ∀ i : Fin 23, (A0 i).isNaN = false ∧ intVal (A0 i) % 512 = 0 ∧ 64 * 512 ≤ intVal (A0 i) := by
  decide +kernel

theorem A1_facts : ∀ i : Fin 62, (A1 i).isNaN = false ∧ intVal (A1 i) % 512 = 0 := by decide +kernel

theorem A1_sep : ∀ i i' : Fin 62, A1 i ≠ A1 i' → 4 * 512 ≤ |intVal (A1 i) - intVal (A1 i')| := by decide +kernel

theorem A0_sep : ∀ i i' : Fin 23, A0 i ≠ A0 i' → 4 * 512 ≤ |intVal (A0 i) - intVal (A0 i')| := by decide +kernel

theorem tuple_facts (j : ℕ) (i : Fin 3) : (tuple j i).isNaN = false ∧ intVal (tuple j i) % 512 = 0 := by
  unfold tuple pick3
  split_ifs
  · exact ⟨(A0_facts _).1, (A0_facts _).2.1⟩
  · exact A1_facts _
  · exact A1_facts _

theorem tuple_int (j : ℕ) (i : Fin 3) : ∃ w : ℤ, val (tuple j i) = w := by
  obtain ⟨h1, h2⟩ := tuple_facts j i
  refine ⟨intVal (tuple j i) / 512, ?_⟩
  rw [val_eq_intVal h1]
  obtain ⟨k, hk⟩ := Int.dvd_of_emod_eq_zero h2
  rw [hk, Int.mul_ediv_cancel_left _ (by norm_num)]; push_cast; ring

theorem tuple0_ge (j : ℕ) : 64 ≤ val (tuple j 0) := by
  have h := A0_facts ⟨j % 23, Nat.mod_lt _ (by norm_num)⟩
  simp only [tuple, pick3, Fin.isValue, Fin.val_zero, ite_true]
  rw [val_eq_intVal h.1, le_div_iff₀ (by norm_num)]
  exact_mod_cast h.2.2

theorem tuple_sep (j j' : ℕ) (i : Fin 3) (h : tuple j i ≠ tuple j' i) :
    4 ≤ |val (tuple j i) - val (tuple j' i)| := by
  rw [val_eq_intVal (tuple_facts j i).1, val_eq_intVal (tuple_facts j' i).1, ← sub_div, abs_div,
    abs_of_pos (by norm_num : (0 : ℚ) < 512), le_div_iff₀ (by norm_num), ← Int.cast_sub, ← Int.cast_abs]
  have : 4 * 512 ≤ |intVal (tuple j i) - intVal (tuple j' i)| := by
    unfold tuple pick3 at *
    split_ifs at * with h1 h2
    · exact A0_sep _ _ h
    · exact A1_sep _ _ h
    · exact A1_sep _ _ h
  exact_mod_cast this

/-! ## The slice's `F` -/

theorem foldl_max_ge (L : List ℚ) (a : ℚ) : a ≤ L.foldl max a := by
  induction L generalizing a with
  | nil => exact le_refl _
  | cons x L ih => exact le_trans (le_max_left a x) (ih _)

theorem le_foldl_max (L : List ℚ) (a : ℚ) : ∀ x ∈ L, x ≤ L.foldl max a := by
  induction L generalizing a with
  | nil => intro x hx; cases hx
  | cons y L ih =>
    intro x hx
    rcases List.mem_cons.1 hx with rfl | hx
    · exact le_trans (le_max_right a x) (foldl_max_ge _ _)
    · exact ih _ x hx

theorem foldl_max_mem (L : List ℚ) (a : ℚ) : L.foldl max a = a ∨ L.foldl max a ∈ L := by
  induction L generalizing a with
  | nil => left; rfl
  | cons y L ih =>
    rw [List.foldl_cons]
    rcases ih (max a y) with h | h
    · rw [h]
      rcases le_total a y with hy | hy
      · right; rw [max_eq_right hy]; exact List.mem_cons_self
      · left; exact max_eq_left hy
    · right; exact List.mem_cons_of_mem _ h

theorem fOf_eq (fi : ℕ) : fOf fi = 2 ^ ((fi : ℤ) - 2) := rfl

theorem fOf_pos (fi : ℕ) : 0 < fOf fi := two_zpow_pos _

/-- A slice of finite codes has `F = fOf fi` for some `fi ≤ 4`, and its codes are below `2^(fi + 5)`. -/
theorem floorF_index (xs : Fin 29 → Code) (hx : ∀ l, (xs l).isNaN = false) :
    ∃ fi : ℕ, fi ≤ 4 ∧ floorF (sliceMax xs) = fOf fi ∧ ∀ l, |val (xs l)| < 2 ^ (fi + 5) := by
  have hle : ∀ l, |val (xs l)| ≤ sliceMax xs := fun l =>
    le_foldl_max _ _ _ (List.mem_ofFn.2 ⟨l, rfl⟩)
  have h0 : 0 ≤ sliceMax xs := foldl_max_ge _ _
  have hmem : sliceMax xs = 0 ∨ ∃ l, sliceMax xs = |val (xs l)| := by
    rcases foldl_max_mem (List.ofFn fun l => |val (xs l)|) 0 with h | h
    · left; exact h
    · right; obtain ⟨l, hl⟩ := List.mem_ofFn.1 h; exact ⟨l, hl.symm⟩
  have h448 : sliceMax xs ≤ 448 := by
    rcases hmem with h | ⟨l, h⟩
    · rw [h]; norm_num
    · rw [h]; exact val_abs_le (hx l)
  rcases h0.lt_or_eq with hpos | hz
  · have hM : sliceMax xs ≠ 0 := hpos.ne'
    obtain ⟨b1, b2⟩ := binade_spec hpos
    have hk : binade (sliceMax xs) ≤ 8 := by
      have : (2 : ℚ) ^ binade (sliceMax xs) < 2 ^ (9 : ℤ) := by
        calc _ ≤ sliceMax xs := b1
          _ ≤ 448 := h448
          _ < _ := by norm_num
      have := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 this
      omega
    refine ⟨(binade (sliceMax xs) - 4).toNat, by omega, ?_, fun l => ?_⟩
    · unfold floorF
      rw [ite_eq_right hM, fOf_eq]
      rcases le_or_gt (binade (sliceMax xs)) 4 with h4 | h4
      · rw [show ((binade (sliceMax xs) - 4).toNat : ℤ) - 2 = -2 by omega, max_eq_right]
        · norm_num
        · calc (2 : ℚ) ^ (binade (sliceMax xs) - 6) ≤ 2 ^ (-2 : ℤ) :=
                zpow_le_zpow_right₀ one_le_two (by omega)
            _ = 1 / 4 := by norm_num
      · rw [show ((binade (sliceMax xs) - 4).toNat : ℤ) - 2 = binade (sliceMax xs) - 6 by omega, max_eq_left]
        calc (1 / 4 : ℚ) = 2 ^ (-2 : ℤ) := by norm_num
          _ ≤ _ := zpow_le_zpow_right₀ one_le_two (by omega)
    · calc |val (xs l)| ≤ sliceMax xs := hle l
        _ < 2 ^ (binade (sliceMax xs) + 1) := b2
        _ ≤ 2 ^ (((binade (sliceMax xs) - 4).toNat + 5 : ℕ) : ℤ) := zpow_le_zpow_right₀ one_le_two (by omega)
        _ = _ := zpow_natCast _ _
  · refine ⟨0, by norm_num, ?_, fun l => ?_⟩
    · rw [← hz]; simp [floorF, fOf]; norm_num
    · have := hle l; rw [← hz] at this; exact lt_of_le_of_lt this (by norm_num)

/-! ## The forming and tag tables -/

theorem val_abs_le_of {c : Code} (h : c.isNaN = false) {N : ℤ} (hN : |intVal c| ≤ N) : |val c| ≤ (N : ℚ) / 512 := by
  rw [val_eq_intVal h, abs_div, abs_of_pos (by norm_num : (0 : ℚ) < 512)]
  apply div_le_div_of_nonneg_right _ (by norm_num)
  rw [← Int.cast_abs]; exact_mod_cast hN

/-- What the proofs read from a forming vector whose code is below `2^(fi + 5)`: both codes finite, their magnitudes
(`min(144F, 448)` and `8F`, times 512) and exponent fields. -/
def formOK (v : FormVec) : Bool :=
  !(decide (|intVal (codeOf v.x)| < 2 ^ (v.fi + 14))) ||
    ((codeOf v.x1).isNaN == false && (codeOf v.x2).isNaN == false &&
      decide (|intVal (codeOf v.x1)| ≤ min (18432 * 2 ^ v.fi) 229376) &&
      decide (|intVal (codeOf v.x2)| ≤ 1024 * 2 ^ v.fi) &&
      decide ((codeOf v.x1).expOr1 ≤ v.fi + 12) && decide ((codeOf v.x2).expOr1 ≤ v.fi + 12))

theorem formTable_ok : formTable.all formOK = true := by decide +kernel

theorem codeOf_val (c : Code) : codeOf c.val = c := by ext; simp [codeOf, Nat.mod_eq_of_lt c.isLt]

theorem form_entry {x : Code} (hx : x.isNaN = false) (pos : Bool) {fi : ℕ} (hfi : fi ≤ 4) :
    ∃ v ∈ formTable, v.x = x.val ∧ v.pos = pos ∧ v.fi = fi := by
  have hk : (x.val, pos, fi) ∈ formKeys := by
    simp only [formKeys, List.mem_flatMap, List.mem_filter, List.mem_range, List.mem_map, List.mem_cons,
      List.mem_nil_iff, or_false, beq_iff_eq]
    exact ⟨x.val, ⟨x.isLt, by rw [codeOf_val]; exact hx⟩, pos, by cases pos <;> simp, fi, by omega, rfl⟩
  rw [← h1tFormTable.1, List.mem_map] at hk
  obtain ⟨v, hv, he⟩ := hk
  simp only [Prod.mk.injEq] at he
  exact ⟨v, hv, he.1, he.2.1, he.2.2⟩

theorem form_facts {x : Code} (hx : x.isNaN = false) (pos : Bool) {fi : ℕ} (hfi : fi ≤ 4)
    (hxb : |val x| < 2 ^ (fi + 5)) :
    (formReal (fOf fi) x pos).1.isNaN = false ∧ (formReal (fOf fi) x pos).2.isNaN = false ∧
    |val (formReal (fOf fi) x pos).1| ≤ ((min (18432 * 2 ^ fi) 229376 : ℤ) : ℚ) / 512 ∧
    |val (formReal (fOf fi) x pos).2| ≤ ((1024 * 2 ^ fi : ℤ) : ℚ) / 512 ∧
    (formReal (fOf fi) x pos).1.expOr1 ≤ fi + 12 ∧ (formReal (fOf fi) x pos).2.expOr1 ≤ fi + 12 := by
  obtain ⟨v, hv, h1, h2, h3⟩ := form_entry hx pos hfi
  have hr := h1tFormTable.2 v hv
  rw [h1, h2, h3, codeOf_val] at hr
  have e1 : (formReal (fOf fi) x pos).1 = codeOf v.x1 := by
    ext; rw [← hr.1]; simp [codeOf, Nat.mod_eq_of_lt (Fin.isLt _)]
  have e2 : (formReal (fOf fi) x pos).2 = codeOf v.x2 := by
    ext; rw [← hr.2]; simp [codeOf, Nat.mod_eq_of_lt (Fin.isLt _)]
  have hok := List.all_eq_true.1 formTable_ok v hv
  have hxi : |intVal (codeOf v.x)| < 2 ^ (v.fi + 14) := by
    rw [h1, codeOf_val, h3]
    rw [val_eq_intVal hx, abs_div, abs_of_pos (by norm_num : (0 : ℚ) < 512), div_lt_iff₀ (by norm_num),
      ← Int.cast_abs] at hxb
    have : ((|intVal x| : ℤ) : ℚ) < ((2 ^ (fi + 14) : ℤ) : ℚ) := by
      rw [pow_add] at hxb ⊢; push_cast at hxb ⊢; linarith
    exact_mod_cast this
  simp only [formOK, hxi, decide_true, Bool.not_true, Bool.false_or, Bool.and_eq_true, beq_iff_eq,
    decide_eq_true_eq] at hok
  obtain ⟨⟨⟨⟨⟨n1, n2⟩, b1⟩, b2⟩, x1⟩, x2⟩ := hok
  rw [e1, e2, ← h3]
  exact ⟨n1, n2, val_abs_le_of n1 b1, val_abs_le_of n2 b2, x1, x2⟩

theorem tagVal_not (F : ℚ) (neg : Bool) (m : Fin 8) : tagVal F (!neg) m = -tagVal F neg m := by
  cases neg <;> simp [tagVal]

/-- What the proofs read from a tag vector: both codes finite, significand `8 + m`, exponent field `fi + 10`. -/
def tagOK (v : TagVec) : Bool :=
  (codeOf v.c1).isNaN == false && (codeOf v.c2).isNaN == false &&
    (codeOf v.c1).sig == 8 + v.m && (codeOf v.c2).sig == 8 + v.m &&
    (codeOf v.c1).expOr1 == v.fi + 10 && (codeOf v.c2).expOr1 == v.fi + 10

theorem tagTable_ok : tagTable.all tagOK = true := by decide +kernel

theorem tag_facts {fi : ℕ} (hfi : fi ≤ 4) (neg : Bool) (m : Fin 8) :
    (enc (tagVal (fOf fi) neg m)).isNaN = false ∧ val (enc (tagVal (fOf fi) neg m)) = tagVal (fOf fi) neg m ∧
    (enc (tagVal (fOf fi) neg m)).sig = 8 + m ∧ (enc (tagVal (fOf fi) neg m)).expOr1 = fi + 10 := by
  have hk : (fi, neg, m.val) ∈ tagKeys := by
    simp only [tagKeys, List.mem_flatMap, List.mem_range, List.mem_map, List.mem_cons, List.mem_nil_iff, or_false]
    exact ⟨fi, by omega, neg, by cases neg <;> simp, m.val, m.isLt, rfl⟩
  rw [← h1tTagTable.1, List.mem_map] at hk
  obtain ⟨v, hv, he⟩ := hk
  simp only [Prod.mk.injEq] at he
  obtain ⟨h1, h2, h3⟩ := he
  have hr := h1tTagTable.2 v hv
  have hm : fin8 v.m = m := by ext; simp [fin8, h3, Nat.mod_eq_of_lt m.isLt]
  rw [h1, h2, hm] at hr
  have e1 : enc (tagVal (fOf fi) neg m) = codeOf v.c1 := by
    ext; rw [← hr.1]; simp [codeOf, Nat.mod_eq_of_lt (Fin.isLt _)]
  have hok := List.all_eq_true.1 tagTable_ok v hv
  simp only [tagOK, Bool.and_eq_true, beq_iff_eq] at hok
  obtain ⟨⟨⟨⟨⟨n1, -⟩, s1⟩, -⟩, x1⟩, -⟩ := hok
  rw [e1, h3, h1] at *
  exact ⟨n1, hr.2.2.1, s1, x1⟩

/-! ## Truncation to a grid -/

theorem truncTo_abs_le {g : ℚ} (hg : 0 < g) (y : ℚ) : |truncTo g y| ≤ |y| := by
  unfold truncTo
  have hf0 : (0 : ℚ) ≤ ⌊|y| / g⌋ := by exact_mod_cast Int.floor_nonneg.2 (div_nonneg (abs_nonneg y) hg.le)
  have hf1 : (⌊|y| / g⌋ : ℚ) * g ≤ |y| := by
    calc _ ≤ |y| / g * g := mul_le_mul_of_nonneg_right (Int.floor_le _) hg.le
      _ = |y| := div_mul_cancel₀ _ hg.ne'
  rw [abs_mul, abs_mul, abs_of_pos hg, abs_of_nonneg hf0, show |(if y < 0 then (-1 : ℚ) else 1)| = 1 by
    split_ifs <;> simp, one_mul]
  exact hf1

theorem truncTo_neg (g y : ℚ) : truncTo g (-y) = -truncTo g y := by
  unfold truncTo
  rw [abs_neg]
  rcases lt_trichotomy y 0 with h | h | h
  · rw [ite_eq_right (by linarith), ite_eq_left h]; ring
  · subst h; simp
  · rw [ite_eq_left (by linarith), ite_eq_right (not_lt.2 h.le)]; ring

theorem truncZ_int (n : ℤ) : truncZ n = n := by
  unfold truncZ
  obtain ⟨k, rfl | rfl⟩ := Int.eq_nat_or_neg n
  · simp
  · split_ifs with h
    · have : (k : ℚ) = 0 := by push_cast at h; linarith [(Nat.cast_nonneg k : (0 : ℚ) ≤ k)]
      have hk : k = 0 := by exact_mod_cast this
      simp [hk]
    · simp

theorem truncTo_int {g : ℚ} (hg : 0 < g) (n : ℤ) : truncTo g (n * g) = n * g := by
  rw [truncTo_mul hg, truncZ_int]

/-! ## The slice's lanes -/

theorem formSlice_fst_real (xs : Fin 29 → Code) (σ : SliceSalt) (l : Fin 32) (h : l.val < 29) :
    (formSlice xs σ).1 l = (formReal (floorF (sliceMax xs)) (xs ⟨l.val, h⟩) (σ.pos ⟨l.val, h⟩)).1 := by
  simp [formSlice, h]

theorem formSlice_snd_real (xs : Fin 29 → Code) (σ : SliceSalt) (l : Fin 32) (h : l.val < 29) :
    (formSlice xs σ).2 l = (formReal (floorF (sliceMax xs)) (xs ⟨l.val, h⟩) (σ.pos ⟨l.val, h⟩)).2 := by
  simp [formSlice, h]

theorem formSlice_fst_tag (xs : Fin 29 → Code) (σ : SliceSalt) (l : Fin 32) (h : ¬ l.val < 29) :
    (formSlice xs σ).1 l = enc (tagVal (floorF (sliceMax xs)) (σ.neg ⟨l.val - 29, by omega⟩)
      (σ.man ⟨l.val - 29, by omega⟩)) := by
  simp [formSlice, h]

theorem formSlice_snd_tag (xs : Fin 29 → Code) (σ : SliceSalt) (l : Fin 32) (h : ¬ l.val < 29) :
    (formSlice xs σ).2 l = enc (-tagVal (floorF (sliceMax xs)) (σ.neg ⟨l.val - 29, by omega⟩)
      (σ.man ⟨l.val - 29, by omega⟩)) := by
  simp [formSlice, h]

theorem tagVal_abs_le (F : ℚ) (hF : 0 < F) (neg : Bool) (m : Fin 8) : |tagVal F neg m| ≤ 60 * F := by
  have hm : (m : ℚ) ≤ 7 := by exact_mod_cast Nat.le_of_lt_succ m.isLt
  have hm0 : (0 : ℚ) ≤ m := Nat.cast_nonneg _
  unfold tagVal
  rw [abs_mul, abs_mul, show |(if neg then (-1 : ℚ) else 1)| = 1 by split_ifs <;> simp, one_mul,
    abs_of_pos (by positivity), abs_of_pos (by positivity)]
  nlinarith

/-- A formed slice's lanes: finite, bounded, and with exponent field at most `fi + 12`. -/
theorem lane_facts (xs : Fin 29 → Code) (hx : ∀ l, (xs l).isNaN = false) (σ : SliceSalt) {fi : ℕ} (hfi : fi ≤ 4)
    (hF : floorF (sliceMax xs) = fOf fi) (hxb : ∀ l, |val (xs l)| < 2 ^ (fi + 5)) (l : Fin 32) :
    ((formSlice xs σ).1 l).isNaN = false ∧ ((formSlice xs σ).2 l).isNaN = false ∧
    |val ((formSlice xs σ).1 l)| ≤
      (if l.val < 29 then ((min (18432 * 2 ^ fi) 229376 : ℤ) : ℚ) / 512 else 60 * fOf fi) ∧
    |val ((formSlice xs σ).2 l)| ≤ (if l.val < 29 then ((1024 * 2 ^ fi : ℤ) : ℚ) / 512 else 60 * fOf fi) ∧
    ((formSlice xs σ).1 l).expOr1 ≤ fi + 12 ∧ ((formSlice xs σ).2 l).expOr1 ≤ fi + 12 := by
  by_cases h : l.val < 29
  · rw [formSlice_fst_real xs σ l h, formSlice_snd_real xs σ l h, hF, ite_eq_left h, ite_eq_left h]
    exact form_facts (hx _) _ hfi (hxb _)
  · rw [formSlice_fst_tag xs σ l h, formSlice_snd_tag xs σ l h, hF, ite_eq_right h, ite_eq_right h, ← tagVal_not]
    obtain ⟨n1, v1, -, e1⟩ := tag_facts hfi (σ.neg ⟨l.val - 29, by omega⟩) (σ.man ⟨l.val - 29, by omega⟩)
    obtain ⟨n2, v2, -, e2⟩ := tag_facts hfi (!σ.neg ⟨l.val - 29, by omega⟩) (σ.man ⟨l.val - 29, by omega⟩)
    exact ⟨n1, n2, by rw [v1]; exact tagVal_abs_le _ (fOf_pos fi) _ _,
      by rw [v2]; exact tagVal_abs_le _ (fOf_pos fi) _ _, by omega, by omega⟩

theorem colLanes_real (b : ℕ → Code) (j τ : ℕ) (l : Fin 32) (h : l.val < 29) :
    colLanes b j τ l = b (29 * τ + l.val) := by simp [colLanes, h]

theorem colLanes_tag (b : ℕ → Code) (j τ : ℕ) (l : Fin 32) (h : ¬ l.val < 29) :
    colLanes b j τ l = tuple j ⟨l.val - 29, by omega⟩ := by simp [colLanes, h]

theorem col_facts (b : ℕ → Code) (hb : ∀ r, (b r).isNaN = false) (j τ : ℕ) (l : Fin 32) :
    (colLanes b j τ l).isNaN = false ∧ |val (colLanes b j τ l)| ≤ 448 ∧ (colLanes b j τ l).expOr1 ≤ 15 := by
  have hn : (colLanes b j τ l).isNaN = false := by
    by_cases h : l.val < 29
    · rw [colLanes_real b j τ l h]; exact hb _
    · rw [colLanes_tag b j τ l h]; exact (tuple_facts _ _).1
  exact ⟨hn, val_abs_le hn, expOr1_le _⟩

/-! ## The slice's alignment exponent and sums -/

theorem gridExp_le {a c : Fin 32 → Code} {B : ℕ} (ha : ∀ l, (a l).expOr1 ≤ B) (hc : ∀ l, (c l).expOr1 ≤ 15) :
    gridExp a c ≤ B + 1 := by
  apply maxExp_le _ _ _ (by omega)
  intro t ht hm
  obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
  rw [product_exp_eq hm]
  have := ha l; have := hc l
  omega

theorem maxExp_congr (z : ℤ) {ts ts' : List Term}
    (h : ts.map (fun t => (t.mag, t.exp)) = ts'.map (fun t => (t.mag, t.exp))) : maxExp z ts = maxExp z ts' := by
  have e : ∀ us : List Term, maxExp z us =
      (us.map fun t => (t.mag, t.exp)).foldl (fun m p => if p.1 ≠ 0 ∧ m < p.2 then p.2 else m) z := by
    intro us; unfold maxExp; rw [List.foldl_map]
  rw [e, e, h]

theorem gridExp_congr {a a' c : Fin 32 → Code} (hs : ∀ l, (a l).sig = (a' l).sig)
    (he : ∀ l, (a l).expOr1 = (a' l).expOr1) : gridExp a c = gridExp a' c := by
  unfold gridExp
  apply maxExp_congr
  rw [List.map_ofFn, List.map_ofFn]
  refine congrArg List.ofFn (funext fun l => ?_)
  obtain ⟨h1, h2⟩ := product_mag_exp_congr (b := c l) (hs l) (he l)
  show ((product (a l) (c l)).mag, (product (a l) (c l)).exp) =
    ((product (a' l) (c l)).mag, (product (a' l) (c l)).exp)
  rw [h1, h2]

theorem preSum_abs_le (a c : Fin 32 → Code) : |preSum a c| ≤ ∑ l, |val (a l)| * |val (c l)| := by
  unfold preSum
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun l _ => ?_)
  rw [← abs_mul]; exact truncTo_abs_le (two_zpow_pos _) _

theorem sum_lanes (A B : ℚ) : ∑ l : Fin 32, (if l.val < 29 then A else B) = 29 * A + 3 * B := by
  rw [Finset.sum_ite]
  simp only [Finset.sum_const, nsmul_eq_mul]
  have h1 : (Finset.univ.filter fun l : Fin 32 => l.val < 29).card = 29 := by decide +kernel
  have h2 : (Finset.univ.filter fun l : Fin 32 => ¬ l.val < 29).card = 3 := by decide +kernel
  rw [h1, h2]; push_cast; ring

theorem tagVal_eq (fi : ℕ) (neg : Bool) (m : Fin 8) :
    tagVal (fOf fi) neg m = (((if neg then -1 else 1) * (8 + m.val : ℤ) : ℤ) : ℚ) * 2 ^ fi := by
  unfold tagVal
  rw [fOf_eq, zpow_sub₀ two_ne_zero, zpow_natCast]
  split_ifs <;> push_cast <;> ring

/-- **L2**: a tag product is on the grid `2^(E − 13)` of an exponent `E ≤ fi + 13`. -/
theorem tag_on_grid {fi : ℕ} {E : ℤ} (hE : E ≤ fi + 13) (neg : Bool) (m : Fin 8) (j : ℕ) (i : Fin 3) :
    truncTo (2 ^ (E - 13)) (tagVal (fOf fi) neg m * val (tuple j i)) = tagVal (fOf fi) neg m * val (tuple j i) := by
  obtain ⟨w, hw⟩ := tuple_int j i
  have e : tagVal (fOf fi) neg m * val (tuple j i) =
      ((((if neg then -1 else 1) * (8 + m.val : ℤ)) * w * 2 ^ (fi + 13 - E).toNat : ℤ) : ℚ) * 2 ^ (E - 13) := by
    rw [tagVal_eq, hw]
    push_cast
    have : (2 : ℚ) ^ fi = 2 ^ (fi + 13 - E).toNat * 2 ^ (E - 13) := by
      rw [← zpow_natCast, ← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; omega
    rw [this]; ring
  rw [e, truncTo_int (two_zpow_pos _)]


/-! ## One slice: L1 to L3 -/

theorem block_bound (a c : Fin 32 → Code) (A B : ℚ) (ha : ∀ l, |val (a l)| ≤ if l.val < 29 then A else B)
    (hc : ∀ l, |val (c l)| ≤ 448) : |preSum a c| ≤ 29 * (A * 448) + 3 * (B * 448) := by
  rw [← sum_lanes]
  refine le_trans (preSum_abs_le a c) (Finset.sum_le_sum fun l _ => ?_)
  have h0 := abs_nonneg (val (a l))
  have h1 := abs_nonneg (val (c l))
  have h2 := ha l
  have h3 := hc l
  split_ifs at h2 ⊢ <;> nlinarith

theorem preSum_single {a a' c : Fin 32 → Code} (l0 : Fin 32) (hl : ∀ l, l ≠ l0 → a' l = a l)
    (hg : gridExp a' c = gridExp a c) :
    preSum a' c = preSum a c + (truncTo (2 ^ (gridExp a c - 13)) (val (a' l0) * val (c l0)) -
      truncTo (2 ^ (gridExp a c - 13)) (val (a l0) * val (c l0))) := by
  have key : ∀ f f' : Fin 32 → ℚ, (∀ l, l ≠ l0 → f' l = f l) → ∑ l, f' l = ∑ l, f l + (f' l0 - f l0) := by
    intro f f' h
    rw [← Finset.add_sum_erase _ f' (Finset.mem_univ l0), ← Finset.add_sum_erase _ f (Finset.mem_univ l0),
      Finset.sum_congr rfl fun l hl => h l (Finset.ne_of_mem_erase hl)]
    ring
  unfold preSum
  rw [hg]
  exact key _ _ fun l h => by rw [hl l h]

theorem formSlice_tag (xs : Fin 29 → Code) (σ : SliceSalt) (i : Fin 3) :
    (formSlice xs σ).1 ⟨29 + i.val, by omega⟩ = enc (tagVal (floorF (sliceMax xs)) (σ.neg i) (σ.man i)) ∧
    (formSlice xs σ).2 ⟨29 + i.val, by omega⟩ = enc (-tagVal (floorF (sliceMax xs)) (σ.neg i) (σ.man i)) := by
  simp [formSlice]

theorem colLanes_tag_i (b : ℕ → Code) (j τ : ℕ) (i : Fin 3) :
    colLanes b j τ ⟨29 + i.val, by omega⟩ = tuple j i := by
  simp [colLanes]

theorem negFlip_self (i : Fin 3) (σ : SliceSalt) : (negFlip i σ).neg i = !σ.neg i := by simp [negFlip]

theorem negFlip_ne {i k : Fin 3} (σ : SliceSalt) (h : k ≠ i) : (negFlip i σ).neg k = σ.neg k := by
  simp [negFlip, h]

/-- Flipping tag lane `29 + i`'s sign bit changes no other lane. -/
theorem flip_lane_ne (xs : Fin 29 → Code) (σ : SliceSalt) (i : Fin 3) (l : Fin 32)
    (h : l ≠ ⟨29 + i.val, by omega⟩) :
    (formSlice xs (negFlip i σ)).1 l = (formSlice xs σ).1 l ∧ (formSlice xs (negFlip i σ)).2 l = (formSlice xs σ).2 l := by
  by_cases hl : l.val < 29
  · rw [formSlice_fst_real _ _ _ hl, formSlice_fst_real _ _ _ hl, formSlice_snd_real _ _ _ hl,
      formSlice_snd_real _ _ _ hl]
    exact ⟨rfl, rfl⟩
  · have hk : (⟨l.val - 29, by omega⟩ : Fin 3) ≠ i := by
      intro e; apply h; apply Fin.ext
      have h1 : l.val - 29 = i.val := by rw [← e]
      have h2 := l.isLt
      show l.val = 29 + i.val
      omega
    rw [formSlice_fst_tag _ _ _ hl, formSlice_fst_tag _ _ _ hl, formSlice_snd_tag _ _ _ hl,
      formSlice_snd_tag _ _ _ hl, negFlip_ne σ hk]
    exact ⟨rfl, rfl⟩

/-- Every lane of the flipped slice has the original's significand and exponent field. -/
theorem flip_lane_sig (xs : Fin 29 → Code) (σ : SliceSalt) (i : Fin 3) {fi : ℕ} (hfi : fi ≤ 4)
    (hF : floorF (sliceMax xs) = fOf fi) (l : Fin 32) :
    ((formSlice xs (negFlip i σ)).1 l).sig = ((formSlice xs σ).1 l).sig ∧
    ((formSlice xs (negFlip i σ)).1 l).expOr1 = ((formSlice xs σ).1 l).expOr1 ∧
    ((formSlice xs (negFlip i σ)).2 l).sig = ((formSlice xs σ).2 l).sig ∧
    ((formSlice xs (negFlip i σ)).2 l).expOr1 = ((formSlice xs σ).2 l).expOr1 := by
  by_cases h : l = ⟨29 + i.val, by omega⟩
  · subst h
    obtain ⟨f1, f2⟩ := formSlice_tag xs (negFlip i σ) i
    obtain ⟨g1, g2⟩ := formSlice_tag xs σ i
    rw [f1, f2, g1, g2, hF, ← tagVal_not, ← tagVal_not, negFlip_self]
    have hm : (negFlip i σ).man i = σ.man i := rfl
    rw [hm]
    obtain ⟨-, -, s1, e1⟩ := tag_facts hfi (!σ.neg i) (σ.man i)
    obtain ⟨-, -, s2, e2⟩ := tag_facts hfi (σ.neg i) (σ.man i)
    obtain ⟨-, -, s3, e3⟩ := tag_facts hfi (!!σ.neg i) (σ.man i)
    exact ⟨by rw [s1, s2], by rw [e1, e2], by rw [s3, s1], by rw [e3, e1]⟩
  · obtain ⟨h1, h2⟩ := flip_lane_ne xs σ i l h
    rw [h1, h2]; exact ⟨rfl, rfl, rfl, rfl⟩

theorem h1tSlice : H1TSlice := by
  intro xs σ b j τ hx hb
  obtain ⟨fi, hfi, hF, hxb⟩ := floorF_index xs hx
  have hL := lane_facts xs hx σ hfi hF hxb
  have hC := col_facts b hb j τ
  have hc : ∀ l, |val (colLanes b j τ l)| ≤ 448 := fun l => (hC l).2.1
  have b1 := block_bound _ _ _ _ (fun l => (hL l).2.2.1) hc
  have b2 := block_bound _ _ _ _ (fun l => (hL l).2.2.2.1) hc
  refine ⟨?_, ?_, fun i => ?_⟩
  · rw [hF]
    interval_cases fi <;> norm_num [fOf] at b1 ⊢ <;> constructor <;> linarith
  · rw [hF]
    interval_cases fi <;> norm_num [fOf] at b2 ⊢ <;> constructor <;> linarith
  have hS := flip_lane_sig xs σ i hfi hF
  have hg1 : gridExp (formSlice xs (negFlip i σ)).1 (colLanes b j τ) = gridExp (formSlice xs σ).1 (colLanes b j τ) :=
    gridExp_congr (fun l => (hS l).1) (fun l => (hS l).2.1)
  have hg2 : gridExp (formSlice xs (negFlip i σ)).2 (colLanes b j τ) = gridExp (formSlice xs σ).2 (colLanes b j τ) :=
    gridExp_congr (fun l => (hS l).2.2.1) (fun l => (hS l).2.2.2)
  have hE1 : gridExp (formSlice xs σ).1 (colLanes b j τ) ≤ (fi : ℤ) + 13 := by
    have := gridExp_le (B := fi + 12) (fun l => (hL l).2.2.2.2.1) (fun l => (hC l).2.2); push_cast at this; omega
  have hE2 : gridExp (formSlice xs σ).2 (colLanes b j τ) ≤ (fi : ℤ) + 13 := by
    have := gridExp_le (B := fi + 12) (fun l => (hL l).2.2.2.2.2) (fun l => (hC l).2.2); push_cast at this; omega
  refine ⟨hg1, hg2, ?_, ?_⟩
  · rw [preSum_single ⟨29 + i.val, by omega⟩ (fun l h => (flip_lane_ne xs σ i l h).1) hg1,
      (formSlice_tag xs (negFlip i σ) i).1, (formSlice_tag xs σ i).1, colLanes_tag_i, hF, negFlip_self]
    have hm : (negFlip i σ).man i = σ.man i := rfl
    rw [hm, (tag_facts hfi _ _).2.1, (tag_facts hfi _ _).2.1, tag_on_grid hE1, tag_on_grid hE1, tagVal_not]
    ring
  · rw [preSum_single ⟨29 + i.val, by omega⟩ (fun l h => (flip_lane_ne xs σ i l h).2) hg2,
      (formSlice_tag xs (negFlip i σ) i).2, (formSlice_tag xs σ i).2, colLanes_tag_i, hF, negFlip_self]
    have hm : (negFlip i σ).man i = σ.man i := rfl
    rw [hm, ← tagVal_not, ← tagVal_not, Bool.not_not, (tag_facts hfi _ _).2.1, (tag_facts hfi _ _).2.1,
      tag_on_grid hE2, tag_on_grid hE2, tagVal_not]
    ring

/-! ## The chain: L7 and L4 -/

theorem h1tChainMono : H1TChainMono := by
  intro T x b j σ σ' p h
  induction p with
  | zero => exact h 0 le_rfl
  | succ p ih =>
    exact rnAdd_mono _ _ _ _ (add_le_add (ih fun t ht => h t (by omega)) (h (p + 1) le_rfl))

theorem formSlice_finite (xs : Fin 29 → Code) (hx : ∀ l, (xs l).isNaN = false) (σ : SliceSalt) (l : Fin 32) :
    ((formSlice xs σ).1 l).isNaN = false ∧ ((formSlice xs σ).2 l).isNaN = false := by
  obtain ⟨fi, hfi, hF, hxb⟩ := floorF_index xs hx
  exact ⟨(lane_facts xs hx σ hfi hF hxb l).1, (lane_facts xs hx σ hfi hF hxb l).2.1⟩

theorem trunc14_abs_le (y : ℚ) : |trunc14 y| ≤ |y| := by
  rw [trunc14_abs]
  rcases eq_or_ne y 0 with rfl | hy
  · simp [trunc14_zero]
  · rw [trunc14_pos (abs_pos.2 hy)]; exact (tpos_spec (abs_pos.2 hy)).2.1

/-- A fresh-step word of finite codes is `trunc14` of its block's sum. -/
theorem freshWord_value {T : ℕ} {x b : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (hb : ∀ r, (b r).isNaN = false)
    (j : ℕ) (σ : ℕ → SliceSalt) (p : ℕ) :
    wordQ (freshWord T x b j σ p) = some (trunc14 (if p < T then
      preSum (formSlice (realLanes x p) (σ p)).1 (colLanes b j p)
      else preSum (formSlice (realLanes x (p - T)) (σ (p - T))).2 (colLanes b j (p - T)))) := by
  have hxs : ∀ τ l, (realLanes x τ l).isNaN = false := fun τ l => hx _
  have hc : ∀ τ l, (colLanes b j τ l).isNaN = false := fun τ l => (col_facts b hb j τ l).1
  unfold freshWord
  split_ifs with h
  · exact h1tAtomValue _ _ (fun l => (formSlice_finite _ (hxs p) _ l).1) (hc p)
  · exact h1tAtomValue _ _ (fun l => (formSlice_finite _ (hxs (p - T)) _ l).2) (hc (p - T))

theorem freshVal_value {T : ℕ} {x b : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (hb : ∀ r, (b r).isNaN = false)
    (j : ℕ) (σ : ℕ → SliceSalt) (p : ℕ) :
    freshVal T x b j σ p = trunc14 (if p < T then
      preSum (formSlice (realLanes x p) (σ p)).1 (colLanes b j p)
      else preSum (formSlice (realLanes x (p - T)) (σ (p - T))).2 (colLanes b j (p - T))) := by
  unfold freshVal wordVal; rw [freshWord_value hx hb]; rfl

theorem fresh_facts {T : ℕ} {x b : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (hb : ∀ r, (b r).isNaN = false)
    (j : ℕ) (σ : ℕ → SliceSalt) (p : ℕ) :
    (wordQ (freshWord T x b j σ p)).isSome = true ∧
    |freshVal T x b j σ p| ≤ (if p < T then 6142976 else 738304) := by
  have hxs : ∀ τ l, (realLanes x τ l).isNaN = false := fun τ l => hx _
  refine ⟨by rw [freshWord_value hx hb]; rfl, ?_⟩
  rw [freshVal_value hx hb]
  split_ifs with h
  · exact le_trans (trunc14_abs_le _) (h1tSlice _ _ _ _ _ (hxs p) hb).1.1
  · exact le_trans (trunc14_abs_le _) (h1tSlice _ _ _ _ _ (hxs (p - T)) hb).2.1.1

/-- Below `2^33` the ulp is at most 512, so rounding moves a value by at most 256. -/
theorem rn_err_small {y : ℚ} (h : |y| < 2 ^ 33) : |rn y - y| ≤ 256 := by
  rcases eq_or_ne y 0 with rfl | hy
  · have : rn 0 = 0 := rn_exact ⟨0, 0, by norm_num, by simp⟩
    simp [this]
  · have hb := (binade_spec (abs_pos.2 hy)).1
    have e : (2 : ℚ) ^ (33 : ℕ) = 2 ^ (33 : ℤ) := (zpow_natCast 2 33).symm
    rw [e] at h
    have h33 : binade |y| < 33 :=
      (zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℚ) < 2)).1 (lt_of_le_of_lt hb h)
    have hu : ulp y ≤ 512 := by
      unfold ulp
      calc (2 : ℚ) ^ (binade |y| - 23) ≤ 2 ^ (9 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
        _ = 512 := by norm_num
    linarith [rn_err y]

theorem rnAdd_abs_le {r i : ℚ} (h : |r + i| < 2 ^ 33) : |rnAdd r i| ≤ |r + i| + 256 := by
  have := rn_err_small h
  have := abs_sub_abs_le_abs_sub (rn (r + i)) (r + i)
  unfold rnAdd; linarith

/-- The running words' accounting: `R_p` gains at most a block's bound plus 256 per step. -/
theorem run_bound {T : ℕ} {x b : ℕ → Code} {j : ℕ} {σ : ℕ → SliceSalt} (hT : T ≤ 1130)
    (hI : ∀ p, |freshVal T x b j σ p| ≤ (if p < T then 6142976 else 738304)) (p : ℕ) (hp : p < 2 * T) :
    |runVal T x b j σ p| ≤ (if p < T then ((p : ℚ) + 1) * 6143232 - 256
      else (T : ℚ) * 6143232 + ((p : ℚ) + 1 - T) * 738560 - 256) := by
  have hTq : (T : ℚ) ≤ 1130 := by exact_mod_cast hT
  induction p with
  | zero =>
    have hT0 : 0 < T := by omega
    have := hI 0
    rw [ite_eq_left hT0] at this ⊢
    show |freshVal T x b j σ 0| ≤ _
    norm_num; linarith
  | succ p ih =>
    have ih := ih (by omega)
    have hIp := hI (p + 1)
    show |rnAdd (runVal T x b j σ p) (freshVal T x b j σ (p + 1))| ≤ _
    have hsum := abs_add_le (runVal T x b j σ p) (freshVal T x b j σ (p + 1))
    push_cast
    by_cases h1 : p + 1 < T
    · have h0 : p < T := by omega
      rw [ite_eq_left h0] at ih
      rw [ite_eq_left h1] at hIp ⊢
      have hq : (p : ℚ) + 2 ≤ T := by exact_mod_cast h1
      have hlt : |runVal T x b j σ p + freshVal T x b j σ (p + 1)| < 2 ^ 33 := by norm_num; nlinarith
      have := rnAdd_abs_le hlt
      linarith
    · rw [ite_eq_right h1] at hIp ⊢
      by_cases h0 : p < T
      · rw [ite_eq_left h0] at ih
        have hq : (p : ℚ) + 1 = T := by exact_mod_cast (show p + 1 = T by omega)
        have hlt : |runVal T x b j σ p + freshVal T x b j σ (p + 1)| < 2 ^ 33 := by norm_num; nlinarith
        have := rnAdd_abs_le hlt
        nlinarith
      · rw [ite_eq_right h0] at ih
        have hq : (p : ℚ) + 2 ≤ 2 * T := by exact_mod_cast (show p + 2 ≤ 2 * T by omega)
        have hq' : (T : ℚ) ≤ p := by exact_mod_cast (show T ≤ p by omega)
        have hlt : |runVal T x b j σ p + freshVal T x b j σ (p + 1)| < 2 ^ 33 := by norm_num; nlinarith
        have := rnAdd_abs_le hlt
        linarith

theorem h1tRange : H1TRange := by
  intro U hU i j g p hp
  obtain ⟨hk, -, hxq, hbq⟩ := hU
  have hx : ∀ r, (U.row i r).isNaN = false := fun r => by
    unfold Unit.row; split_ifs
    · exact hxq _ _
    · rfl
  have hb : ∀ r, (U.col j r).isNaN = false := fun r => by
    unfold Unit.col; split_ifs
    · exact hbq _ _
    · rfl
  have hT : U.T ≤ 1130 := by unfold Unit.T; omega
  have hTq : (U.T : ℚ) ≤ 1130 := by exact_mod_cast hT
  have hF := fresh_facts (T := U.T) hx hb j.val (U.rowSalt g i)
  refine ⟨(hF p).1, (hF p).2, ?_⟩
  have hB := run_bound hT (fun p => (hF p).2) p hp
  split_ifs at hB with h
  · have hq : (p : ℚ) + 1 ≤ U.T := by exact_mod_cast h
    norm_num; nlinarith
  · have hq : (p : ℚ) + 1 ≤ 2 * U.T := by exact_mod_cast hp
    norm_num; nlinarith

/-- The forming table's keys cover every slice. -/
theorem h1tFormCover : H1TFormCover := floorF_index

end Pouw.Fp8Atom.H1T.Proofs
