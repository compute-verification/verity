module

public import Definitions.Pouw.PearlC.Defs
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.Fp8Atom.Fp32Proofs

/-! P2, staged and nothing pinned: Pearl-C's clean-up chain, its cancellation identities, where its rounding is and
its liveness at degenerate inputs. -/

@[expose] public section

namespace Pouw.PearlC.Peel

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.PearlC Matrix

variable {m n k r : Nat}

theorem peelU_cancel (βA : Fin m → ℚ) (βB : Fin n → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ)
    (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) :
    peelU (A' * B'ᵀ) βA βB A' B' z = (A' - rowScale βA z.NA) * (B' - rowScale βB z.NB)ᵀ := by
  simp only [peelU, peelLeft, peelRight, rowScale, Noise.NA, Noise.NB, transpose_fromCols, fromCols_mul_fromRows,
    transpose_mul, transpose_sub, transpose_neg, diagonal_transpose, Matrix.mul_sub, Matrix.sub_mul, Matrix.mul_neg,
    Matrix.mul_assoc]
  abel

theorem peelU_cancel_codes (βA : Fin m → ℚ) (βB : Fin n → ℚ) (Aq : Matrix (Fin m) (Fin k) Code)
    (Bq : Matrix (Fin n) (Fin k) Code) (z : Noise m n k r) :
    peelU (ctExact Aq Bq) βA βB (vals Aq) (vals Bq) z =
      (vals Aq - rowScale βA z.NA) * (vals Bq - rowScale βB z.NB)ᵀ :=
  peelU_cancel βA βB (vals Aq) (vals Bq) z

theorem atomMix_lines (αA βA : Fin m → ℚ) (βB : Fin n → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) (z : Noise m n k 32)
    (i : Fin m) (l : Fin k) :
    FormChain.atomMix.quantA αA βA X z i l =
        e4m3 (wordVal (hopperStep (f32Word (αA i * X i l)) (lineCodes (βA i) (z.EA i)) (z.FA l))) ∧
      ∀ t, (z.atomLines βA βB).EA i t = lineCodes (βA i) (z.EA i) t :=
  ⟨rfl, fun _ => rfl⟩

theorem peelU_cancel_atomMix (αA βA : Fin m → ℚ) (αB βB : Fin n → ℚ) (X : Matrix (Fin m) (Fin k) ℚ)
    (B : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k 32) :
    peelU (ctExact (FormChain.atomMix.quantA αA βA X z) (FormChain.atomMix.quantB αB βB B z)) (fun _ => 1)
        (fun _ => 1) (vals (FormChain.atomMix.quantA αA βA X z)) (vals (FormChain.atomMix.quantB αB βB B z))
        (z.atomLines βA βB) =
      (vals (FormChain.atomMix.quantA αA βA X z) - vals (castLines βA z.EA) * (vals z.FA)ᵀ) *
        (vals (FormChain.atomMix.quantB αB βB B z) - vals (castLines βB z.EB) * (vals z.FB)ᵀ)ᵀ := by
  rw [peelU_cancel_codes]
  simp only [rowScale, Noise.NA, Noise.NB, Noise.atomLines, diagonal_one, Matrix.one_mul]

theorem peelRight_gram (βB : Fin n → ℚ) (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) :
    (rowScale βB z.NB - B') * vals z.FA = rowScale βB (vals z.EB * z.gram) - B' * vals z.FA := by
  simp only [rowScale, Noise.NB, Noise.gram, Matrix.sub_mul, Matrix.mul_assoc]

theorem rawU_eq (Ct : Matrix (Fin m) (Fin n) ℚ) (A : Matrix (Fin m) (Fin k) ℚ) (B : Matrix (Fin n) (Fin k) ℚ)
    (NA : Matrix (Fin m) (Fin k) ℚ) (NB : Matrix (Fin n) (Fin k) ℚ) :
    rawU Ct A B NA NB = A * Bᵀ + (Ct - (A + NA) * (B + NB)ᵀ) := by
  simp only [rawU, transpose_add, Matrix.add_mul, Matrix.mul_add]
  abel

theorem rawU_cancel (A : Matrix (Fin m) (Fin k) ℚ) (B : Matrix (Fin n) (Fin k) ℚ) (NA : Matrix (Fin m) (Fin k) ℚ)
    (NB : Matrix (Fin n) (Fin k) ℚ) : rawU ((A + NA) * (B + NB)ᵀ) A B NA NB = A * Bᵀ := by
  rw [rawU_eq, sub_self, add_zero]

theorem hasSig_mul {p q : Nat} {x y : ℚ} (hx : HasSig p x) (hy : HasSig q y) : HasSig (p + q) (x * y) := by
  obtain ⟨a, e, ha, rfl⟩ := hx
  obtain ⟨b, f, hb, rfl⟩ := hy
  refine ⟨a * b, e + f, ?_, ?_⟩
  · rw [abs_mul, pow_add]
    exact mul_lt_mul'' ha hb (abs_nonneg _) (abs_nonneg _)
  · rw [zpow_add₀ two_ne_zero]; push_cast; ring

theorem hasSig_mono {p q : Nat} {x : ℚ} (h : p ≤ q) (hx : HasSig p x) : HasSig q x := by
  obtain ⟨a, e, ha, rfl⟩ := hx
  exact ⟨a, e, lt_of_lt_of_le ha (pow_le_pow_right₀ (by norm_num) h), rfl⟩

theorem representable_of_hasSig {p : Nat} {x : ℚ} (h : p ≤ 24) (hx : HasSig p x) : Representable x :=
  hasSig_mono h hx

theorem bf16_mul_representable {x y : ℚ} (hx : HasSig 8 x) (hy : HasSig 8 y) : Representable (x * y) :=
  representable_of_hasSig (by norm_num) (hasSig_mul hx hy)

theorem rn_mul_exact {p q : Nat} {x y : ℚ} (h : p + q ≤ 24) (hx : HasSig p x) (hy : HasSig q y) :
    rn (x * y) = x * y :=
  rn_exact (representable_of_hasSig h (hasSig_mul hx hy))

theorem sig_lt (c : Code) : c.sig < 16 := by
  unfold Code.sig Code.manField; split <;> omega

theorem codeVal_hasSig (c : Code) : HasSig 4 (codeVal c) := by
  have h16 : ((c.sig : ℤ)) < 16 := by exact_mod_cast sig_lt c
  unfold codeVal decode
  by_cases hN : c.isNaN
  · simp only [hN, ↓reduceIte, Option.getD_none]
    exact ⟨0, 0, by norm_num, by simp⟩
  · simp only [hN, Bool.false_eq_true, ↓reduceIte, Option.getD_some]
    refine ⟨(if c.sign then -1 else 1) * c.sig, c.lsbExp, ?_, ?_⟩
    · split_ifs <;> simpa using h16
    · split_ifs <;> push_cast <;> ring

theorem bf16Rn_hasSig (x : ℚ) : HasSig 8 (bf16Rn x) := by
  rcases eq_or_ne x 0 with rfl | hx
  · refine ⟨0, 0, by norm_num, ?_⟩
    have h : roundEven 0 = 0 := by simpa using roundEven_intCast 0
    simp [bf16Rn, h]
  have hx' : 0 < |x| := abs_pos.2 hx
  obtain ⟨-, h2⟩ := binade_spec hx'
  unfold bf16Rn ulpBf16
  generalize binade |x| = b at h2
  have hu : (0 : ℚ) < 2 ^ (b - 7) := two_zpow_pos _
  have h8 : (2 : ℚ) ^ 8 * 2 ^ (b - 7) = 2 ^ (b + 1) := by
    rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
  have hlt : |x| / 2 ^ (b - 7) < 2 ^ 8 := by rw [div_lt_iff₀ hu, h8]; exact h2
  have hn0 : 0 ≤ roundEven (|x| / 2 ^ (b - 7)) :=
    le_trans (Int.floor_nonneg.2 (div_nonneg (abs_nonneg _) hu.le)) (floor_le_roundEven _)
  have hn1 : roundEven (|x| / 2 ^ (b - 7)) ≤ 2 ^ 8 := by
    have := roundEven_le (|x| / 2 ^ (b - 7))
    have : ⌊|x| / 2 ^ (b - 7)⌋ < 2 ^ 8 := Int.floor_lt.2 (by exact_mod_cast hlt)
    omega
  generalize roundEven (|x| / 2 ^ (b - 7)) = N at hn0 hn1
  rcases hn1.lt_or_eq with h | h
  · refine ⟨(if x < 0 then -1 else 1) * N, b - 7, ?_, ?_⟩
    · split_ifs <;> simpa [abs_of_nonneg hn0] using h
    · split_ifs <;> push_cast <;> ring
  · refine ⟨if x < 0 then -1 else 1, b + 1, by split_ifs <;> norm_num, ?_⟩
    subst h
    split_ifs <;> push_cast
    · linear_combination -h8
    · linear_combination h8

/-- The sequential FP32 accumulation of `ps` into `s`: `rnAdd` in list order. -/
def accum (s : ℚ) (ps : List ℚ) : ℚ := ps.foldl rnAdd s

/-- Each step's exact sum before rounding, in order. -/
def stepSums : ℚ → List ℚ → List ℚ
  | _, [] => []
  | s, p :: ps => (s + p) :: stepSums (rnAdd s p) ps

theorem roundEven_sub_le (t : ℚ) : |(roundEven t : ℚ) - t| ≤ 1 / 2 := by
  have h1 := Int.floor_le t
  have h2 := Int.lt_floor_add_one t
  unfold roundEven; dsimp only
  split_ifs with ha hb hc <;> push_cast <;> rw [abs_le] <;> constructor <;> (try simp only [not_lt] at *) <;> linarith

theorem rn_sub_le_ulp (x : ℚ) : |rn x - x| ≤ ulp x / 2 := by
  have hu := ulp_pos x
  have hr := roundEven_sub_le (|x| / ulp x)
  have hx : |x| / ulp x * ulp x = |x| := div_mul_cancel₀ _ hu.ne'
  unfold rn
  generalize |x| / ulp x = t at hr hx
  generalize (roundEven t : ℚ) = N at hr
  generalize ulp x = u at hu hr hx
  rcases le_or_gt 0 x with h | h
  · rw [ite_eq_right (not_lt.2 h)]
    rw [abs_of_nonneg h] at hx
    rw [show 1 * N * u - x = u * (N - t) by linear_combination hx, abs_mul, abs_of_pos hu]
    nlinarith
  · rw [ite_eq_left h]
    rw [abs_of_neg h] at hx
    rw [show -1 * N * u - x = -(u * (N - t)) by linear_combination -hx, abs_neg, abs_mul, abs_of_pos hu]
    nlinarith

theorem ulp_le_abs {x : ℚ} (hx : x ≠ 0) : ulp x ≤ |x| / 2 ^ 23 := by
  rw [le_div_iff₀ (by norm_num), mul_comm, ulp, two_pow_23]
  exact (binade_spec (abs_pos.2 hx)).1

theorem rn_sub_le (x : ℚ) : |rn x - x| ≤ |x| / 2 ^ 24 := by
  rcases eq_or_ne x 0 with rfl | hx
  · rw [rn_exact ⟨0, 0, by norm_num, by simp⟩]; simp
  · calc |rn x - x| ≤ ulp x / 2 := rn_sub_le_ulp x
      _ ≤ |x| / 2 ^ 23 / 2 := by gcongr; exact ulp_le_abs hx
      _ = |x| / 2 ^ 24 := by ring

theorem rn_sub_le_ulp_of_le {x M : ℚ} (h : |x| ≤ M) : |rn x - x| ≤ ulp M / 2 := by
  rcases eq_or_ne x 0 with rfl | hx
  · rw [rn_exact ⟨0, 0, by norm_num, by simp⟩, sub_self, abs_zero]
    exact div_nonneg (ulp_pos M).le (by norm_num)
  · refine le_trans (rn_sub_le_ulp x) ?_
    have hb := binade_mono (abs_pos.2 hx) (h.trans (le_abs_self M))
    have : ulp x ≤ ulp M := zpow_le_zpow_right₀ one_le_two (by omega)
    linarith

theorem accum_error_ulp (s : ℚ) (ps : List ℚ) :
    |accum s ps - (s + ps.sum)| ≤ ((stepSums s ps).map fun y => ulp y / 2).sum := by
  induction ps generalizing s with
  | nil => simp [accum, stepSums]
  | cons p ps ih =>
    simp only [accum, List.foldl_cons, stepSums, List.map_cons, List.sum_cons] at ih ⊢
    have e := rn_sub_le_ulp (s + p)
    calc |List.foldl rnAdd (rnAdd s p) ps - (s + (p + ps.sum))|
        = |(List.foldl rnAdd (rnAdd s p) ps - (rnAdd s p + ps.sum)) + (rn (s + p) - (s + p))| := by
          rw [rnAdd]; ring_nf
      _ ≤ _ := abs_add_le _ _
      _ ≤ _ := by rw [add_comm (ulp (s + p) / 2)]; exact add_le_add (ih _) e

theorem accum_error_ulp_le (s : ℚ) (ps : List ℚ) {M : ℚ} (h : ∀ y ∈ stepSums s ps, |y| ≤ M) :
    |accum s ps - (s + ps.sum)| ≤ ps.length * (ulp M / 2) := by
  induction ps generalizing s with
  | nil => simp [accum]
  | cons p ps ih =>
    simp only [stepSums, List.mem_cons, forall_eq_or_imp] at h
    simp only [accum, List.foldl_cons, List.sum_cons, List.length_cons] at ih ⊢
    have e := rn_sub_le_ulp_of_le h.1
    calc |List.foldl rnAdd (rnAdd s p) ps - (s + (p + ps.sum))|
        = |(List.foldl rnAdd (rnAdd s p) ps - (rnAdd s p + ps.sum)) + (rn (s + p) - (s + p))| := by
          rw [rnAdd]; ring_nf
      _ ≤ _ := abs_add_le _ _
      _ ≤ ps.length * (ulp M / 2) + ulp M / 2 := add_le_add (ih _ h.2) e
      _ = _ := by push_cast; ring

theorem accum_error (s : ℚ) (ps : List ℚ) :
    |accum s ps - (s + ps.sum)| ≤ ((1 + 1 / 2 ^ 24) ^ ps.length - 1) * (|s| + (ps.map fun p => |p|).sum) := by
  induction ps generalizing s with
  | nil => simp [accum]
  | cons p ps ih =>
    simp only [accum, List.foldl_cons, List.sum_cons, List.length_cons, List.map_cons] at ih ⊢
    set c := (1 + 1 / (2 : ℚ) ^ 24) ^ ps.length with hcdef
    set P := (ps.map fun p => |p|).sum
    have hc : 1 ≤ c := one_le_pow₀ (by norm_num)
    have hP : 0 ≤ P := List.sum_nonneg (by simp)
    have e := rn_sub_le (s + p)
    have hsp : |s + p| ≤ |s| + |p| := abs_add_le s p
    have hs' : |rnAdd s p| ≤ (1 + 1 / 2 ^ 24) * (|s| + |p|) := by
      have := abs_sub_abs_le_abs_sub (rn (s + p)) (s + p)
      rw [rnAdd]; nlinarith [abs_nonneg (s + p)]
    have ih' := ih (rnAdd s p)
    calc |List.foldl rnAdd (rnAdd s p) ps - (s + (p + ps.sum))|
        = |(List.foldl rnAdd (rnAdd s p) ps - (rnAdd s p + ps.sum)) + (rn (s + p) - (s + p))| := by
          rw [rnAdd]; ring_nf
      _ ≤ _ := abs_add_le _ _
      _ ≤ (c - 1) * (|rnAdd s p| + P) + |s + p| / 2 ^ 24 := add_le_add ih' e
      _ ≤ (c - 1) * ((1 + 1 / 2 ^ 24) * (|s| + |p|) + P) + (|s| + |p|) / 2 ^ 24 := by gcongr
      _ ≤ (c * (1 + 1 / 2 ^ 24) - 1) * (|s| + (|p| + P)) := by nlinarith [abs_nonneg s, abs_nonneg p]
      _ = _ := by rw [hcdef, ← pow_succ]

/-- The peel's combine for output `(i, j)` in the K order `ord`: `C̃ i j`, then `L i t · R j t` for `t` in `ord`, each added by `rnAdd`. -/
def peelAccum (ord : List (Fin r ⊕ Fin r)) (Ct : Matrix (Fin m) (Fin n) ℚ) (L : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ)
    (R : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ) (i : Fin m) (j : Fin n) : ℚ :=
  accum (Ct i j) (ord.map fun t => L i t * R j t)

/-- A K order that lists each of the `2r` columns once. -/
def IsOrder (ord : List (Fin r ⊕ Fin r)) : Prop := ord.Nodup ∧ ∀ t, t ∈ ord

theorem sum_order {ord : List (Fin r ⊕ Fin r)} (h : IsOrder ord) (f : Fin r ⊕ Fin r → ℚ) :
    (ord.map f).sum = ∑ t, f t := by
  rw [← List.sum_toFinset f h.1]
  congr 1
  ext t; simp [h.2 t]

theorem length_order {ord : List (Fin r ⊕ Fin r)} (h : IsOrder ord) : ord.length = r + r := by
  rw [← List.toFinset_card_of_nodup h.1]
  have : ord.toFinset = Finset.univ := by ext t; simp [h.2 t]
  rw [this, Finset.card_univ, Fintype.card_sum, Fintype.card_fin]

theorem peelAccum_exact_value {ord : List (Fin r ⊕ Fin r)} (h : IsOrder ord) (Ct : Matrix (Fin m) (Fin n) ℚ)
    (L : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ) (R : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ) (i : Fin m) (j : Fin n) :
    Ct i j + (ord.map fun t => L i t * R j t).sum = (Ct + L * Rᵀ) i j := by
  rw [sum_order h, Matrix.add_apply, Matrix.mul_apply]; rfl

theorem peelAccum_error {ord : List (Fin r ⊕ Fin r)} (h : IsOrder ord) (Ct : Matrix (Fin m) (Fin n) ℚ)
    (L : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ) (R : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ) (i : Fin m) (j : Fin n) :
    |peelAccum ord Ct L R i j - (Ct + L * Rᵀ) i j| ≤
      ((1 + 1 / 2 ^ 24) ^ (r + r) - 1) * (|Ct i j| + ∑ t, |L i t * R j t|) := by
  have e := accum_error (Ct i j) (ord.map fun t => L i t * R j t)
  rw [List.length_map, length_order h, List.map_map] at e
  rw [← peelAccum_exact_value h, peelAccum, ← sum_order h (fun t => |L i t * R j t|)]
  exact e

theorem peel_error {ord : List (Fin r ⊕ Fin r)} (h : IsOrder ord) (βA : Fin m → ℚ) (βB : Fin n → ℚ)
    (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (z : Noise m n k r) (i : Fin m) (j : Fin n) :
    |peelAccum ord (ctExact Aq Bq) (peelLeft βA (vals Aq) z) (peelRight βB (vals Bq) z) i j -
        ((vals Aq - rowScale βA z.NA) * (vals Bq - rowScale βB z.NB)ᵀ) i j| ≤
      ((1 + 1 / 2 ^ 24) ^ (r + r) - 1) *
        (|ctExact Aq Bq i j| + ∑ t, |peelLeft βA (vals Aq) z i t * peelRight βB (vals Bq) z j t|) := by
  rw [← peelU_cancel_codes]
  exact peelAccum_error h _ _ _ i j

theorem peelAccum_rn_products (ord : List (Fin r ⊕ Fin r)) (Ct : Matrix (Fin m) (Fin n) ℚ)
    (L : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ) (R : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ) (i : Fin m) (j : Fin n)
    (hL : ∀ t, HasSig 8 (L i t)) (hR : ∀ t, HasSig 8 (R j t)) :
    accum (Ct i j) (ord.map fun t => rn (L i t * R j t)) = peelAccum ord Ct L R i j := by
  unfold peelAccum
  congr 1
  exact List.map_congr_left fun t _ => rn_mul_exact (by norm_num) (hL t) (hR t)

theorem ulpBf16_le {x : ℚ} (hx : x ≠ 0) : ulpBf16 x ≤ |x| / 2 ^ 7 := by
  rw [le_div_iff₀ (by norm_num), ulpBf16]
  calc (2 : ℚ) ^ (binade |x| - 7) * 2 ^ 7 = 2 ^ binade |x| := by
        rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
    _ ≤ |x| := (binade_spec (abs_pos.2 hx)).1

theorem bf16Rn_sub_le (x : ℚ) : |bf16Rn x - x| ≤ |x| / 2 ^ 8 := by
  rcases eq_or_ne x 0 with rfl | hx
  · have h : roundEven 0 = 0 := by simpa using roundEven_intCast 0
    simp [bf16Rn, h]
  have hu : 0 < ulpBf16 x := two_zpow_pos _
  have hr := roundEven_sub_le (|x| / ulpBf16 x)
  have hxu : |x| / ulpBf16 x * ulpBf16 x = |x| := div_mul_cancel₀ _ hu.ne'
  have hle := ulpBf16_le hx
  unfold bf16Rn
  generalize |x| / ulpBf16 x = t at hr hxu
  generalize (roundEven t : ℚ) = N at hr
  generalize ulpBf16 x = u at hu hr hxu hle
  have key : abs (N * u - |x|) ≤ u / 2 := by
    rw [show N * u - |x| = u * (N - t) by linear_combination hxu, abs_mul, abs_of_pos hu]; nlinarith
  rcases le_or_gt 0 x with h | h
  · rw [ite_eq_right (not_lt.2 h)]
    rw [abs_of_nonneg h] at key hle ⊢
    rw [one_mul]; linarith
  · rw [ite_eq_left h]
    rw [abs_of_neg h] at key hle ⊢
    rw [show -1 * N * u - x = -(N * u - -x) by ring, abs_neg]; linarith

theorem abs_sum_le (c : List ℚ) : |c.sum| ≤ (c.map fun p => |p|).sum := by
  induction c with
  | nil => simp
  | cons p c ih => simp only [List.sum_cons, List.map_cons]; exact (abs_add_le _ _).trans (by linarith)

/-- A combine that adds each chunk of products to the running value in one step `step`: one `rnAdd` per product, or
one wgmma step per 16 products. -/
def chunkAccum (step : ℚ → List ℚ → ℚ) (s : ℚ) (cs : List (List ℚ)) : ℚ := cs.foldl step s

/-- A step's error bound: `step a ps` is within `ε·(|a| + Σ|p|)` of `a + Σ p`. The BF16 wgmma K16 step's `ε` is not
derived here: no `ε < 1` holds on every input (`PeelExactChecks.floor_drops`). -/
def StepBound (step : ℚ → List ℚ → ℚ) (ε : ℚ) : Prop :=
  ∀ a ps, |step a ps - (a + ps.sum)| ≤ ε * (|a| + (ps.map fun p => |p|).sum)

theorem chunkAccum_error {step : ℚ → List ℚ → ℚ} {ε : ℚ} (hε : 0 ≤ ε) (h : StepBound step ε) (s : ℚ)
    (cs : List (List ℚ)) :
    |chunkAccum step s cs - (s + (cs.map List.sum).sum)| ≤
      ((1 + ε) ^ cs.length - 1) * (|s| + (cs.map fun c => (c.map fun p => |p|).sum).sum) := by
  induction cs generalizing s with
  | nil => simp [chunkAccum]
  | cons c cs ih =>
    simp only [chunkAccum, List.foldl_cons, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    set C := (1 + ε) ^ cs.length with hC
    set P := (cs.map fun c => (c.map fun p => |p|).sum).sum
    set Q := (c.map fun p => |p|).sum
    have hC1 : 1 ≤ C := one_le_pow₀ (by linarith)
    have hP : 0 ≤ P := List.sum_nonneg (by simp; intro _ _; exact List.sum_nonneg (by simp))
    have hQ : 0 ≤ Q := List.sum_nonneg (by simp)
    have hcQ : |c.sum| ≤ Q := abs_sum_le c
    have e := h s c
    have ha : |step s c| ≤ (1 + ε) * (|s| + Q) := by
      have := abs_sub_abs_le_abs_sub (step s c) (s + c.sum)
      have := abs_add_le s c.sum
      nlinarith [abs_nonneg s]
    have ih' := ih (step s c)
    calc |List.foldl step (step s c) cs - (s + (c.sum + (cs.map List.sum).sum))|
        = |(List.foldl step (step s c) cs - (step s c + (cs.map List.sum).sum)) + (step s c - (s + c.sum))| := by
          ring_nf
      _ ≤ _ := abs_add_le _ _
      _ ≤ (C - 1) * (|step s c| + P) + ε * (|s| + Q) := add_le_add ih' e
      _ ≤ (C - 1) * ((1 + ε) * (|s| + Q) + P) + ε * (|s| + Q) := by gcongr
      _ ≤ (C * (1 + ε) - 1) * (|s| + (Q + P)) := by
          nlinarith [mul_nonneg (mul_nonneg (by linarith : (0 : ℚ) ≤ C) hε) hP]
      _ = _ := by rw [hC, ← pow_succ]

/-- A step that adds its chunk exactly and rounds once to FP32. -/
def rnStep (a : ℚ) (ps : List ℚ) : ℚ := rn (a + ps.sum)

theorem rnStep_bound : StepBound rnStep (1 / 2 ^ 24) := by
  intro a ps
  have e := rn_sub_le (a + ps.sum)
  have h1 := abs_add_le a ps.sum
  have h2 := abs_sum_le ps
  unfold rnStep
  calc _ ≤ |a + ps.sum| / 2 ^ 24 := e
    _ ≤ (|a| + (ps.map fun p => |p|).sum) / 2 ^ 24 := by gcongr; linarith
    _ = _ := by ring

/-- The peel's combine for output `(i, j)` in the steps `cs`: `C̃ i j`, then each step's products `L i t · R j t` added
by `step`. For Π, `cs` splits the pinned K order of the `2r` columns into wgmma K16 steps (4 for `r = 32`). -/
def peelSteps (step : ℚ → List ℚ → ℚ) (cs : List (List (Fin r ⊕ Fin r))) (Ct : Matrix (Fin m) (Fin n) ℚ)
    (L : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ) (R : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ) (i : Fin m) (j : Fin n) : ℚ :=
  chunkAccum step (Ct i j) (cs.map fun c => c.map fun t => L i t * R j t)

theorem peelSteps_error {step : ℚ → List ℚ → ℚ} {ε : ℚ} (hε : 0 ≤ ε) (h : StepBound step ε)
    {cs : List (List (Fin r ⊕ Fin r))} (hcs : IsOrder cs.flatten) (Ct : Matrix (Fin m) (Fin n) ℚ)
    (L : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ) (R : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ) (i : Fin m) (j : Fin n) :
    |peelSteps step cs Ct L R i j - (Ct + L * Rᵀ) i j| ≤
      ((1 + ε) ^ cs.length - 1) * (|Ct i j| + ∑ t, |L i t * R j t|) := by
  have e := chunkAccum_error hε h (Ct i j) (cs.map fun c => c.map fun t => L i t * R j t)
  have h1 : ((cs.map fun c => c.map fun t => L i t * R j t).map List.sum).sum =
      (cs.flatten.map fun t => L i t * R j t).sum := by
    rw [List.map_flatten, List.sum_flatten]
  have h2 : ((cs.map fun c => c.map fun t => L i t * R j t).map fun c => (c.map fun p => |p|).sum).sum =
      (cs.flatten.map fun t => |L i t * R j t|).sum := by
    rw [List.map_flatten, List.sum_flatten, List.map_map, List.map_map]
    congr 1; apply List.map_congr_left; intro c _; simp [List.map_map, Function.comp_def]
  rw [List.length_map, h1, h2, sum_order hcs (fun t => |L i t * R j t|)] at e
  rw [← peelAccum_exact_value hcs]
  exact e

theorem bf16_mul_sub_le (x y : ℚ) : |bf16Rn x * bf16Rn y - x * y| ≤ (2 / 2 ^ 8 + 1 / 2 ^ 16) * |x * y| := by
  have e1 := bf16Rn_sub_le x
  have e2 := bf16Rn_sub_le y
  have hs : bf16Rn x * bf16Rn y - x * y =
      x * (bf16Rn y - y) + (bf16Rn x - x) * y + (bf16Rn x - x) * (bf16Rn y - y) := by ring
  rw [hs, abs_mul x y]
  calc _ ≤ |x * (bf16Rn y - y)| + |(bf16Rn x - x) * y| + |(bf16Rn x - x) * (bf16Rn y - y)| := abs_add_three _ _ _
    _ = |x| * |bf16Rn y - y| + |bf16Rn x - x| * |y| + |bf16Rn x - x| * |bf16Rn y - y| := by
        rw [abs_mul, abs_mul, abs_mul]
    _ ≤ |x| * (|y| / 2 ^ 8) + |x| / 2 ^ 8 * |y| + |x| / 2 ^ 8 * (|y| / 2 ^ 8) := by gcongr
    _ = _ := by ring

/-- A matrix cast to BF16 entry by entry: the peel's factors as the BF16 wgmma reads them. -/
def bf16Mat {p q : Type} (M : Matrix p q ℚ) : Matrix p q ℚ := M.map bf16Rn

theorem peelSteps_bf16_error {step : ℚ → List ℚ → ℚ} {ε : ℚ} (hε : 0 ≤ ε) (h : StepBound step ε)
    {cs : List (List (Fin r ⊕ Fin r))} (hcs : IsOrder cs.flatten) (Ct : Matrix (Fin m) (Fin n) ℚ)
    (L : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ) (R : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ) (i : Fin m) (j : Fin n) :
    |peelSteps step cs Ct (bf16Mat L) (bf16Mat R) i j - (Ct + L * Rᵀ) i j| ≤
      ((1 + ε) ^ cs.length - 1) * (|Ct i j| + ∑ t, |bf16Rn (L i t) * bf16Rn (R j t)|) +
        (2 / 2 ^ 8 + 1 / 2 ^ 16) * ∑ t, |L i t * R j t| := by
  have e := peelSteps_error hε h hcs Ct (bf16Mat L) (bf16Mat R) i j
  have hc : |(Ct + bf16Mat L * (bf16Mat R)ᵀ) i j - (Ct + L * Rᵀ) i j| ≤
      (2 / 2 ^ 8 + 1 / 2 ^ 16) * ∑ t, |L i t * R j t| := by
    simp only [Matrix.add_apply, Matrix.mul_apply, transpose_apply, bf16Mat, Matrix.map_apply, add_sub_add_left_eq_sub,
      ← Finset.sum_sub_distrib, Finset.mul_sum]
    exact (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun t _ => bf16_mul_sub_le _ _)
  simp only [bf16Mat, Matrix.map_apply] at e
  calc _ ≤ |peelSteps step cs Ct (bf16Mat L) (bf16Mat R) i j - (Ct + bf16Mat L * (bf16Mat R)ᵀ) i j| +
        |(Ct + bf16Mat L * (bf16Mat R)ᵀ) i j - (Ct + L * Rᵀ) i j| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add e hc

theorem peel_error_pinned {step : ℚ → List ℚ → ℚ} {ε : ℚ} (hε : 0 ≤ ε) (h : StepBound step ε)
    {cs : List (List (Fin r ⊕ Fin r))} (hcs : IsOrder cs.flatten) (βA : Fin m → ℚ) (βB : Fin n → ℚ)
    (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) (z : Noise m n k r) (i : Fin m) (j : Fin n) :
    |peelSteps step cs (ctExact Aq Bq) (bf16Mat (peelLeft βA (vals Aq) z)) (bf16Mat (peelRight βB (vals Bq) z)) i j -
        ((vals Aq - rowScale βA z.NA) * (vals Bq - rowScale βB z.NB)ᵀ) i j| ≤
      ((1 + ε) ^ cs.length - 1) * (|ctExact Aq Bq i j| +
          ∑ t, |bf16Rn (peelLeft βA (vals Aq) z i t) * bf16Rn (peelRight βB (vals Bq) z j t)|) +
        (2 / 2 ^ 8 + 1 / 2 ^ 16) * ∑ t, |peelLeft βA (vals Aq) z i t * peelRight βB (vals Bq) z j t| := by
  rw [← peelU_cancel_codes]
  exact peelSteps_bf16_error hε h hcs _ _ _ i j


/-- Pearl's `E_A` clean-up term, `(β·E_A)·((β·N_B − B̃)·F_A)ᵀ`. -/
def cleanE (βA : Fin m → ℚ) (βB : Fin n → ℚ) (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) :
    Matrix (Fin m) (Fin n) ℚ :=
  rowScale βA (vals z.EA) * ((rowScale βB z.NB - B') * vals z.FA)ᵀ

/-- Pearl's `F_B` clean-up term, `(A′·F_B)·(−β·E_B)ᵀ`. -/
def cleanF (βB : Fin n → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin m) (Fin n) ℚ :=
  (A' * vals z.FB) * (-rowScale βB (vals z.EB))ᵀ

theorem peel_split (βA : Fin m → ℚ) (βB : Fin n → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ) (B' : Matrix (Fin n) (Fin k) ℚ)
    (z : Noise m n k r) : peelLeft βA A' z * (peelRight βB B' z)ᵀ = cleanE βA βB B' z + cleanF βB A' z := by
  simp only [peelLeft, peelRight, cleanE, cleanF, transpose_fromCols, fromCols_mul_fromRows]

/-- A matrix with entry `(a, b)`'s sign flipped. -/
def flip {p q : Nat} (M : Matrix (Fin p) (Fin q) ℚ) (a : Fin p) (b : Fin q) : Matrix (Fin p) (Fin q) ℚ :=
  fun i j => if i = a ∧ j = b then -M i j else M i j

/-- The code with the other sign bit. -/
def negCode (c : Code) : Code := ⟨(c.val + 128) % 256, Nat.mod_lt _ (by decide)⟩

theorem codeVal_negCode : ∀ c : Code, codeVal (negCode c) = -codeVal c := by decide +kernel

/-- A factor with entry `(a, b)`'s sign bit flipped: another salt's factor, in any sign-symmetric noise support. -/
def flipC {p q : Nat} (M : Matrix (Fin p) (Fin q) Code) (a : Fin p) (b : Fin q) : Matrix (Fin p) (Fin q) Code :=
  fun i j => if i = a ∧ j = b then negCode (M i j) else M i j

theorem vals_flipC {p q : Nat} (M : Matrix (Fin p) (Fin q) Code) (a : Fin p) (b : Fin q) :
    vals (flipC M a b) = flip (vals M) a b := by
  ext i j
  simp only [vals, flipC, flip, Matrix.map_apply]
  split_ifs
  · exact codeVal_negCode _
  · rfl

theorem sum_flip {q : Nat} (f g : Fin q → ℚ) (t : Fin q) (h : ∀ s, g s = if s = t then -f s else f s) :
    ∑ s, g s = ∑ s, f s - 2 * f t := by
  have : ∀ s, g s = f s + (if s = t then -2 * f t else 0) := by
    intro s; rw [h s]; split_ifs with hs
    · subst hs; ring
    · ring
  simp only [this, Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
  ring

theorem flip_ne {q : Nat} {f g : Fin q → ℚ} {t : Fin q} (h : ∀ s, g s = if s = t then -f s else f s)
    (ht : f t ≠ 0) : ∑ s, g s ≠ ∑ s, f s := by
  rw [sum_flip f g t h]; intro e; apply ht; linarith

theorem rowScale_apply {p q : Nat} (β : Fin p → ℚ) (M : Matrix (Fin p) (Fin q) ℚ) (i : Fin p) (j : Fin q) :
    rowScale β M i j = β i * M i j := diagonal_mul β M i j

theorem cleanE_apply (βA : Fin m → ℚ) (βB : Fin n → ℚ) (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r)
    (i : Fin m) (j : Fin n) :
    cleanE βA βB B' z i j = ∑ s, βA i * codeVal (z.EA i s) * ((rowScale βB z.NB - B') * vals z.FA) j s := by
  rw [cleanE, Matrix.mul_apply]; simp only [rowScale_apply, transpose_apply]
  rfl

theorem cleanF_apply (βB : Fin n → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ) (z : Noise m n k r) (i : Fin m) (j : Fin n) :
    cleanF βB A' z i j = ∑ s, (A' * vals z.FB) i s * -(βB j * codeVal (z.EB j s)) := by
  rw [cleanF, Matrix.mul_apply (M := A' * vals z.FB)]; simp only [transpose_apply, Matrix.neg_apply, rowScale_apply]
  rfl

variable (fc : FormChain r)

theorem fused_live (αA βA : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) (z : Noise m n k r) (i : Fin m) (l : Fin k)
    (t : Fin r) (hA : vals (fc.quantA αA βA X z) i l ≠ 0) (hF : codeVal (z.FB l t) ≠ 0) :
    ∃ z' : Noise m n k r, z'.EA = z.EA ∧ z'.FA = z.FA ∧ z'.EB = z.EB ∧
      (vals (fc.quantA αA βA X z') * vals z'.FB) i t ≠ (vals (fc.quantA αA βA X z) * vals z.FB) i t := by
  refine ⟨{ z with FB := flipC z.FB l t }, rfl, rfl, rfl, ?_⟩
  have hq : fc.quantA αA βA X { z with FB := flipC z.FB l t } = fc.quantA αA βA X z := rfl
  have hv : vals ({ z with FB := flipC z.FB l t } : Noise m n k r).FB = flip (vals z.FB) l t := vals_flipC _ _ _
  rw [hq, hv, Matrix.mul_apply, Matrix.mul_apply]
  refine flip_ne (t := l) (fun s => ?_) (mul_ne_zero hA hF)
  simp only [flip, and_true]
  split_ifs <;> ring

theorem gramCols_live (αB βB : Fin n → ℚ) (B : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) (j : Fin n) (l : Fin k)
    (t : Fin r) (hR : (rowScale βB z.NB - vals (fc.quantB αB βB B z)) j l ≠ 0) (hF : codeVal (z.FA l t) ≠ 0) :
    ∃ z' : Noise m n k r, z'.EA = z.EA ∧ z'.EB = z.EB ∧ z'.FB = z.FB ∧
      ((rowScale βB z'.NB - vals (fc.quantB αB βB B z')) * vals z'.FA) j t ≠
        ((rowScale βB z.NB - vals (fc.quantB αB βB B z)) * vals z.FA) j t := by
  refine ⟨{ z with FA := flipC z.FA l t }, rfl, rfl, rfl, ?_⟩
  have hq : fc.quantB αB βB B { z with FA := flipC z.FA l t } = fc.quantB αB βB B z := rfl
  have hn : ({ z with FA := flipC z.FA l t } : Noise m n k r).NB = z.NB := rfl
  have hv : vals ({ z with FA := flipC z.FA l t } : Noise m n k r).FA = flip (vals z.FA) l t := vals_flipC _ _ _
  rw [hq, hn, hv, Matrix.mul_apply, Matrix.mul_apply]
  refine flip_ne (t := l) (fun s => ?_) (mul_ne_zero hR hF)
  simp only [flip, and_true]
  split_ifs <;> ring

theorem cleanE_live (βA : Fin m → ℚ) (αB βB : Fin n → ℚ) (B : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r)
    (i : Fin m) (j : Fin n) (t : Fin r) (hβ : βA i ≠ 0) (hE : codeVal (z.EA i t) ≠ 0)
    (hR : ((rowScale βB z.NB - vals (fc.quantB αB βB B z)) * vals z.FA) j t ≠ 0) :
    ∃ z' : Noise m n k r, z'.FA = z.FA ∧ z'.EB = z.EB ∧ z'.FB = z.FB ∧
      cleanE βA βB (vals (fc.quantB αB βB B z')) z' i j ≠ cleanE βA βB (vals (fc.quantB αB βB B z)) z i j := by
  refine ⟨{ z with EA := flipC z.EA i t }, rfl, rfl, rfl, ?_⟩
  have hq : fc.quantB αB βB B { z with EA := flipC z.EA i t } = fc.quantB αB βB B z := rfl
  have hn : ({ z with EA := flipC z.EA i t } : Noise m n k r).NB = z.NB := rfl
  rw [hq, cleanE_apply, cleanE_apply, hn]
  refine flip_ne (t := t) (fun s => ?_) (mul_ne_zero (mul_ne_zero hβ hE) hR)
  simp only [flipC, true_and]
  split_ifs
  · rw [codeVal_negCode]; ring
  · rfl

theorem cleanF_live (αA βA : Fin m → ℚ) (βB : Fin n → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) (z : Noise m n k r)
    (i : Fin m) (j : Fin n) (t : Fin r) (hβ : βB j ≠ 0) (hE : codeVal (z.EB j t) ≠ 0)
    (hL : (vals (fc.quantA αA βA X z) * vals z.FB) i t ≠ 0) :
    ∃ z' : Noise m n k r, z'.EA = z.EA ∧ z'.FA = z.FA ∧ z'.FB = z.FB ∧
      cleanF βB (vals (fc.quantA αA βA X z')) z' i j ≠ cleanF βB (vals (fc.quantA αA βA X z)) z i j := by
  refine ⟨{ z with EB := flipC z.EB j t }, rfl, rfl, rfl, ?_⟩
  have hq : fc.quantA αA βA X { z with EB := flipC z.EB j t } = fc.quantA αA βA X z := rfl
  rw [hq, cleanF_apply, cleanF_apply]
  refine flip_ne (t := t) (fun s => ?_) (mul_ne_zero hL (neg_ne_zero.2 (mul_ne_zero hβ hE)))
  simp only [flipC, true_and]
  split_ifs
  · rw [codeVal_negCode]; ring
  · rfl

theorem raw_cols_zero (FB : Matrix (Fin k) (Fin r) ℚ) : (0 : Matrix (Fin m) (Fin k) ℚ) * FB = 0 := Matrix.zero_mul _

/-- **The negative control**: the raw-A form's `A·N_Bᵀ` at `A = 0` and `N_A·Bᵀ` at `B = 0` are the same for any two noises. -/
theorem raw_terms_const (NA NA' : Matrix (Fin m) (Fin k) ℚ) (NB NB' : Matrix (Fin n) (Fin k) ℚ) :
    (0 : Matrix (Fin m) (Fin k) ℚ) * NBᵀ = (0 : Matrix (Fin m) (Fin k) ℚ) * NB'ᵀ ∧
      NA * (0 : Matrix (Fin n) (Fin k) ℚ)ᵀ = NA' * (0 : Matrix (Fin n) (Fin k) ℚ)ᵀ := by
  simp only [Matrix.zero_mul, transpose_zero, Matrix.mul_zero, and_self]

theorem rawU_zero_inputs (Ct : Matrix (Fin m) (Fin n) ℚ) (NA : Matrix (Fin m) (Fin k) ℚ) (NB : Matrix (Fin n) (Fin k) ℚ) :
    rawU Ct 0 0 NA NB = Ct - NA * NBᵀ := by
  simp [rawU]

/-- The noise of the witness: every factor entry the code of `1` (shape `1 × 1`, rank 1). -/
def oneNoise : Noise 1 1 1 1 := ⟨fun _ _ => e4m3 1, fun _ _ => e4m3 1, fun _ _ => e4m3 1, fun _ _ => e4m3 1⟩

/-- **The hypotheses are satisfiable at `B = 0` with `B`'s floor `σ = 1/3`, and at `x = 0` with the floor's `σ = 1`**,
with the exact chain and the real encoder, all factor entries `1`. Then `A′ = e4m3(1) = 1`, `B̃ = e4m3(1/3) = 11/32`,
and the residual is `−1/96`. -/
theorem live_witness :
    let A' := vals ((FormChain.exact 1).quantA (fun _ => 1) (fun _ => 1) 0 oneNoise)
    let B' := vals ((FormChain.exact 1).quantB (fun _ => 1) (fun _ => 1 / 3) 0 oneNoise)
    A' 0 0 ≠ 0 ∧ (A' * vals oneNoise.FB) 0 0 ≠ 0 ∧ (rowScale (fun _ => 1 / 3) oneNoise.NB - B') 0 0 ≠ 0 ∧
      ((rowScale (fun _ => 1 / 3) oneNoise.NB - B') * vals oneNoise.FA) 0 0 ≠ 0 ∧
      codeVal (oneNoise.EA 0 0) ≠ 0 ∧ codeVal (oneNoise.FA 0 0) ≠ 0 ∧ codeVal (oneNoise.EB 0 0) ≠ 0 ∧
      codeVal (oneNoise.FB 0 0) ≠ 0 := by
  decide +kernel

/-- The sparse row of the second witness: `x₀ = 1` and 63 zeros (`k = 64`). -/
def sparseRow : Matrix (Fin 1) (Fin 64) ℚ := fun _ l => if l.val = 0 then 1 else 0

/-- The second witness's noise: every factor entry the code of `1` (`m = n = 1`, `k = 64`, rank 1). -/
def oneNoise64 : Noise 1 1 64 1 := ⟨fun _ _ => e4m3 1, fun _ _ => e4m3 1, fun _ _ => e4m3 1, fun _ _ => e4m3 1⟩

theorem sparse_beta (root : ℚ → ℚ) (hroot : root (1 / 64) = 1 / 8) :
    betaOf Stat.rms root 1 1 (sparseRow 0) = 1 / 8 := by
  show 1 * 1 * root (meanSq (sparseRow 0)) = 1 / 8
  rw [show meanSq (sparseRow 0) = 1 / 64 by decide +kernel, hroot]
  norm_num

/-- **The `A` side's hypotheses hold at an in-domain sparse row**: one nonzero entry among 64 satisfies `LiveRow 1`,
its `rms` is `1/8`, and with `α = 1` and `β = 1/8` (the row's `betaOf`, `sparse_beta`), `A′ 0 0 = e4m3(9/8) ≠ 0` and
`(A′·F_B) 0 0 = 9`. -/
theorem live_witness_sparse :
    let A' := vals ((FormChain.exact 1).quantA (fun _ => 1) (fun _ => 1 / 8) sparseRow oneNoise64)
    LiveRow 1 (sparseRow 0) ∧ meanSq (sparseRow 0) = (1 / 8) ^ 2 ∧
      A' 0 0 ≠ 0 ∧ (A' * vals oneNoise64.FB) 0 0 ≠ 0 ∧ codeVal (oneNoise64.EA 0 0) ≠ 0 ∧
      codeVal (oneNoise64.FB 0 0) ≠ 0 := by
  intro A'
  refine ⟨?_, ?_⟩
  · unfold LiveRow; decide +kernel
  · decide +kernel

end Pouw.PearlC.Peel
