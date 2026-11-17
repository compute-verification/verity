module

public import ArkLib.Data.CodingTheory.JohnsonBound.Family
public import ArkLib.Data.CodingTheory.ReedSolomon.Interleaved
public import Definitions.Flock.Defs
import Proofs.Flock.Soundness.Defs

/-! At most `listBound₀` codewords lie near any level-0 table, from ArkLib's Johnson bound for interleaved Reed–Solomon
codes. -/

@[expose] public section

namespace FlockSoundness

open Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

def domainEmb (A : Arith F K) (hA : A.Correct) (d : ℕ) (hd : d ≤ 64) : Fin (2 ^ d) ↪ F :=
  ⟨fun q => A.omega q.val, fun q q' h => Fin.ext (hA.omega_injective d hd q.isLt q'.isLt h)⟩

theorem domainEmb_apply (A : Arith F K) (hA : A.Correct) (d : ℕ) (hd : d ≤ 64) (q : Fin (2 ^ d)) :
    domainEmb A hA d hd q = A.omega q.val := rfl

def tr {lanes d : ℕ} (T : Table F lanes d) : Fin (2 ^ d) → Fin (2 ^ lanes) → F := fun q ℓ => T ℓ q

/-- The novel-basis polynomials of `Arith.Correct.xhat_poly`. -/
noncomputable def xP (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) :
    Fin (2 ^ L) → Polynomial F :=
  (hA.xhat_poly L hL).choose

theorem xP_natDegree (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) (c : Fin (2 ^ L)) :
    (xP A hA L hL c).natDegree < 2 ^ L := (hA.xhat_poly L hL).choose_spec.1 c

theorem xP_linearIndependent (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) :
    LinearIndependent F (xP A hA L hL) := (hA.xhat_poly L hL).choose_spec.2.1

theorem xP_eval (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) (c : Fin (2 ^ L)) (ω : F) :
    A.Xhat c.val L ω = (xP A hA L hL c).eval ω := (hA.xhat_poly L hL).choose_spec.2.2 c ω

/-- The polynomial of one message row. -/
noncomputable def rowPoly (A : Arith F K) (hA : A.Correct) {L : ℕ} (hL : L ≤ 40)
    (row : Fin (2 ^ L) → F) : Polynomial F :=
  ∑ c, Polynomial.C (row c) * xP A hA L hL c

theorem rowPoly_eval (A : Arith F K) (hA : A.Correct) {L : ℕ} (hL : L ≤ 40) (row : Fin (2 ^ L) → F)
    (q : ℕ) : (rowPoly A hA hL row).eval (A.omega q) = A.encodeAt L row q := by
  simp only [rowPoly, Polynomial.eval_finset_sum, Polynomial.eval_mul, Polynomial.eval_C,
    Arith.encodeAt, xP_eval A hA L hL]

theorem rowPoly_natDegree (A : Arith F K) (hA : A.Correct) {L : ℕ} (hL : L ≤ 40)
    (row : Fin (2 ^ L) → F) : (rowPoly A hA hL row).natDegree < 2 ^ L := by
  have h : (rowPoly A hA hL row).natDegree ≤ 2 ^ L - 1 := by
    refine Polynomial.natDegree_sum_le_of_forall_le _ _ fun c _ => ?_
    refine (Polynomial.natDegree_C_mul_le _ _).trans ?_
    have := xP_natDegree A hA L hL c
    omega
  have : 0 < 2 ^ L := Nat.two_pow_pos L
  omega

theorem tr_encode_mem (A : Arith F K) (hA : A.Correct) {lanes cols : ℕ} (d : ℕ)
    (hcols : cols ≤ 40) (hd : d ≤ 64) (M : Message F lanes cols) :
    tr (encode A d M) ∈ ReedSolomon.Interleaved.irsCode (domainEmb A hA d hd)
      (2 ^ lanes * 2 ^ cols) (2 ^ lanes) := by
  rw [ReedSolomon.Interleaved.irsCode, Nat.mul_div_cancel_left _ (Nat.two_pow_pos lanes)]
  refine (Code.mem_moduleInterleavedCode_iff _ _ _ _ _ _).2 fun ℓ => ?_
  refine ReedSolomon.mem_code_of_polynomial_of_natDegree_lt_of_eval (rowPoly A hA hcols (M ℓ))
    (rowPoly_natDegree A hA hcols (M ℓ)) fun q => ?_
  change A.encodeAt cols (M ℓ) q.val =
    Polynomial.eval (domainEmb A hA d hd q) (rowPoly A hA hcols (M ℓ))
  rw [domainEmb_apply, rowPoly_eval A hA hcols]

theorem encode_injective (A : Arith F K) (hA : A.Correct) {lanes cols d : ℕ} (hcd : cols ≤ d)
    (hcols : cols ≤ 40) (hd : d ≤ 64) :
    Function.Injective (fun M : Message F lanes cols => tr (encode A d M)) := by
  intro M M' h
  funext ℓ c
  have hzero : rowPoly A hA hcols (M ℓ) - rowPoly A hA hcols (M' ℓ) = 0 := by
    refine Polynomial.eq_zero_of_natDegree_lt_card_of_eval_eq_zero _
      (domainEmb A hA d hd).injective (fun q => ?_) ?_
    · have := congrFun (congrFun h q) ℓ
      simp only [tr, encode] at this
      rw [domainEmb_apply, Polynomial.eval_sub, rowPoly_eval A hA hcols, rowPoly_eval A hA hcols,
        this, sub_self]
    · refine (Polynomial.natDegree_sub_le _ _).trans_lt ?_
      rw [Fintype.card_fin]
      have h1 := rowPoly_natDegree A hA hcols (M ℓ)
      have h2 := rowPoly_natDegree A hA hcols (M' ℓ)
      have h3 : 2 ^ cols ≤ 2 ^ d := Nat.pow_le_pow_right (by norm_num) hcd
      omega
  have hsum : ∑ c', (M ℓ c' - M' ℓ c') • xP A hA cols hcols c' = 0 := by
    rw [← hzero]
    simp only [rowPoly, ← Finset.sum_sub_distrib, sub_smul, Polynomial.smul_eq_C_mul, ← sub_mul,
      Polynomial.C_sub]
  have := Fintype.linearIndependent_iff.1 (xP_linearIndependent A hA cols hcols) _ hsum c
  exact sub_eq_zero.1 this

theorem relHammingDist_tr_le {lanes d : ℕ} (γ : ℝ) (T U : Table F lanes d) (h : Close γ T U) :
    (Code.relHammingDist (tr T) (tr U) : ℝ) ≤ γ := by
  classical
  rw [Code.relHammingDist_coe]
  have hn : (0 : ℝ) < Fintype.card (Fin (2 ^ d)) := by exact_mod_cast Fintype.card_pos
  rw [div_le_iff₀ hn, Fintype.card_fin]
  have hsplit := Finset.card_filter_add_card_filter_not
    (s := (Finset.univ : Finset (Fin (2 ^ d)))) (fun q => ∀ ℓ, T ℓ q = U ℓ q)
  have hdist : hammingDist (tr T) (tr U) =
      (Finset.univ.filter fun q : Fin (2 ^ d) => ¬ ∀ ℓ, T ℓ q = U ℓ q).card := by
    unfold hammingDist
    congr 1
    ext q
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, tr, ne_eq, funext_iff]
  unfold Close at h
  rw [hdist]
  rw [Finset.card_univ, Fintype.card_fin] at hsplit
  have hs : ((Finset.univ.filter fun q : Fin (2 ^ d) => ¬ ∀ ℓ, T ℓ q = U ℓ q).card : ℝ) =
      2 ^ d - (Finset.univ.filter fun q : Fin (2 ^ d) => ∀ ℓ, T ℓ q = U ℓ q).card := by
    rw [eq_sub_iff_add_eq, add_comm]
    exact_mod_cast hsplit
  have h' : ((Finset.univ.filter fun q : Fin (2 ^ d) => ∀ ℓ, T ℓ q = U ℓ q).card : ℝ) ≥
      (1 - γ) * 2 ^ d := by convert h
  rw [hs]
  push_cast
  linarith


theorem fast100_levels_ne_nil : ∀ m s, fast100 m = some s → s.levels ≠ [] := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  obtain ⟨h1, h2⟩ := hm m (by simp [h])
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; simp [mk])

theorem fast100_m : ∀ m s, fast100 m = some s → s.m = m := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  obtain ⟨h1, h2⟩ := hm m (by simp [h])
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; rfl)

theorem johnson_value (r : ℕ) : 1 / (2 * eta * (2⁻¹ : ℝ) ^ r) = ((100 * 2 ^ r : ℕ) : ℝ) := by
  unfold eta
  push_cast
  rw [inv_pow]
  have : (2 : ℝ) ^ r ≠ 0 := pow_ne_zero _ two_ne_zero
  field_simp
  ring

open Classical in
theorem card_close_le (A : Arith F K) (hA : A.Correct) (l : Level) (hl : l.Fits) (lanes : ℕ)
    (T : Table F lanes l.logLen) :
    (Finset.univ.filter fun M : Message F lanes l.logCols =>
      Close l.radius T (encode A l.logLen M)).card ≤ l.listBound := by
  rw [Level.listBound]
  have hdc : l.logLen = l.logCols + l.logInvRate := rfl
  have : NeZero (2 ^ lanes) := ⟨pow_ne_zero _ two_ne_zero⟩
  have : NeZero (2 ^ l.logLen) := ⟨pow_ne_zero _ two_ne_zero⟩
  set code := ReedSolomon.Interleaved.irsCode (domainEmb A hA l.logLen hl.2)
    (2 ^ lanes * 2 ^ l.logCols) (2 ^ lanes) with hcode
  set y := tr T
  set S := Finset.univ.filter fun M : Message F lanes l.logCols =>
    Close l.radius T (encode A l.logLen M)
  have hinj : Function.Injective (fun M : Message F lanes l.logCols => tr (encode A l.logLen M)) :=
    encode_injective A hA (by omega) hl.1 hl.2
  have hsub : ((S.image fun M => tr (encode A l.logLen M) : Finset _) : Set _) ⊆
      Code.closeCodewordsRel (code : Set (Fin (2 ^ l.logLen) → Fin (2 ^ lanes) → F)) y l.radius := by
    intro w hw
    simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe] at hw
    obtain ⟨M, hM, rfl⟩ := hw
    simp only [S, Finset.mem_filter, Finset.mem_univ, true_and] at hM
    exact Code.mem_closeCodewordsRel_iff.2
      ⟨tr_encode_mem A hA l.logLen hl.1 hl.2 M, relHammingDist_tr_le _ _ _ hM⟩
  have hcardS : (S.card : ℕ∞) ≤
      Code.Lambda (code : Set (Fin (2 ^ l.logLen) → Fin (2 ^ lanes) → F)) l.radius := by
    rw [← Finset.card_image_of_injective S hinj, ← Set.encard_coe_eq_coe_finsetCard]
    exact Code.encard_le_Lambda_of_subset_closeCodewordsRel hsub
  have hks : 0 < 2 ^ lanes * 2 ^ l.logCols / 2 ^ lanes := by
    rw [Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]; exact Nat.two_pow_pos _
  have hrate : (LinearCode.alphabetRate code : ℝ) = l.rate := by
    rw [hcode, ReedSolomon.Interleaved.alphabetRate_irsCode_eq_min,
      Nat.mul_div_cancel_left _ (Nat.two_pow_pos _), Fintype.card_fin,
      min_eq_left (Nat.pow_le_pow_right (by norm_num) (by omega)), Level.rate, hdc]
    push_cast
    rw [pow_add, inv_pow]
    have : (2 : ℝ) ^ l.logCols ≠ 0 := pow_ne_zero _ two_ne_zero
    field_simp
  have hJ := CodingTheory.irs_lambda_le_johnson_mds (domainEmb A hA l.logLen hl.2)
    (2 ^ lanes * 2 ^ l.logCols)
    (2 ^ lanes) hks eta (by unfold eta; norm_num)
  simp only at hJ
  rw [← hcode, hrate] at hJ
  have hrad : 1 - Real.sqrt l.rate - eta = l.radius := rfl
  rw [hrad, Level.rate, johnson_value, ENNReal.ofReal_natCast] at hJ
  have h1 : ((S.card : ℕ∞) : ℝ≥0∞) ≤ ((100 * 2 ^ l.logInvRate : ℕ) : ℝ≥0∞) :=
    (ENat.toENNReal_le.2 hcardS).trans hJ
  exact_mod_cast h1

open Classical in
theorem closeMsgs_card_le (A : Arith F K) (hA : A.Correct) (sch : Schedule) (m : ℕ)
    (hsch : fast100 m = some sch) (C : Oracle F) :
    (Finset.univ.filter fun M => M ∈ closeMsgs A sch C).card ≤ sch.listBound₀ := by
  obtain ⟨l0, rest, hlev⟩ : ∃ l0 rest, sch.levels = l0 :: rest := by
    cases h : sch.levels with
    | nil => exact absurd h (fast100_levels_ne_nil m sch hsch)
    | cons l0 rest => exact ⟨l0, rest, rfl⟩
  have hL : sch.level₀ = l0 := by simp [Schedule.level₀, hlev]
  have hbound : sch.listBound₀ = sch.level₀.listBound := by
    simp [Schedule.listBound₀, hlev, hL]
  rw [hbound]
  exact card_close_le A hA sch.level₀
    (hL ▸ fast100_levels_fit m sch hsch l0 (by rw [hlev]; exact List.mem_cons_self))
    sch.k0 (asTable sch C)

end FlockSoundness
