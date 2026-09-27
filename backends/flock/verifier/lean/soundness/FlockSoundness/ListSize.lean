import FlockSoundness.Defs
import ArkLib.Data.CodingTheory.JohnsonBound.Family

/-!
# The list size: at most `L₀` codewords near any level-0 table

Every phase of `rep_sound` pays a union bound over the codewords within the level-0 proximity
radius of the committed table (`closeMsgs`). This file bounds their number by `listBound₀ =
25·2^r` at rate `2^-r` (`closeMsgs_card_le`), from ArkLib's Johnson bound for interleaved
Reed–Solomon codes (`irs_lambda_le_johnson_mds`: at radius `1 − √ρ − η`, at most `1/(2ηρ)`
codewords). The bridge:

* a table, read position first (`tr`), is an interleaved word; an encoded message is a codeword of
  `irsCode` on the level's evaluation points, because each row is the evaluation of a polynomial of
  degree below `2^cols` (the novel basis, `Arith.Correct.xhat_poly`);
* "close" (`Close`: every lane agrees on a `1 − γ` fraction of positions) is relative block
  Hamming distance at most `γ`;
* encoding is injective (a nonzero polynomial of degree below `2^cols ≤ 2^d` cannot vanish on `2^d`
  distinct points), so distinct close messages are distinct close codewords.
-/

namespace FlockSoundness

open Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The level's evaluation points, as an embedding. -/
def domainEmb (A : Arith F K) (hA : A.Correct) (d : ℕ) : Fin (2 ^ d) ↪ F :=
  ⟨fun q => A.omega q.val, fun q q' h => Fin.ext (hA.omega_injective d q.isLt q'.isLt h)⟩

theorem domainEmb_apply (A : Arith F K) (hA : A.Correct) (d : ℕ) (q : Fin (2 ^ d)) :
    domainEmb A hA d q = A.omega q.val := rfl

/-- A table read position first: an interleaved word. -/
def tr {lanes d : ℕ} (T : Table F lanes d) : Fin (2 ^ d) → Fin (2 ^ lanes) → F := fun q ℓ => T ℓ q

/-- The novel-basis polynomials of `Arith.Correct.xhat_poly`. -/
noncomputable def xP (A : Arith F K) (hA : A.Correct) (L : ℕ) : Fin (2 ^ L) → Polynomial F :=
  (hA.xhat_poly L).choose

theorem xP_natDegree (A : Arith F K) (hA : A.Correct) (L : ℕ) (c : Fin (2 ^ L)) :
    (xP A hA L c).natDegree < 2 ^ L := (hA.xhat_poly L).choose_spec.1 c

theorem xP_linearIndependent (A : Arith F K) (hA : A.Correct) (L : ℕ) :
    LinearIndependent F (xP A hA L) := (hA.xhat_poly L).choose_spec.2.1

theorem xP_eval (A : Arith F K) (hA : A.Correct) (L : ℕ) (c : Fin (2 ^ L)) (ω : F) :
    A.Xhat c.val L ω = (xP A hA L c).eval ω := (hA.xhat_poly L).choose_spec.2.2 c ω

/-- The polynomial of one message row. -/
noncomputable def rowPoly (A : Arith F K) (hA : A.Correct) {L : ℕ} (row : Fin (2 ^ L) → F) :
    Polynomial F :=
  ∑ c, Polynomial.C (row c) * xP A hA L c

theorem rowPoly_eval (A : Arith F K) (hA : A.Correct) {L : ℕ} (row : Fin (2 ^ L) → F) (q : ℕ) :
    (rowPoly A hA row).eval (A.omega q) = A.encodeAt L row q := by
  simp only [rowPoly, Polynomial.eval_finset_sum, Polynomial.eval_mul, Polynomial.eval_C,
    Arith.encodeAt, xP_eval A hA]

theorem rowPoly_natDegree (A : Arith F K) (hA : A.Correct) {L : ℕ} (row : Fin (2 ^ L) → F) :
    (rowPoly A hA row).natDegree < 2 ^ L := by
  have h : (rowPoly A hA row).natDegree ≤ 2 ^ L - 1 := by
    refine Polynomial.natDegree_sum_le_of_forall_le _ _ fun c _ => ?_
    refine (Polynomial.natDegree_C_mul_le _ _).trans ?_
    have := xP_natDegree A hA L c
    omega
  have : 0 < 2 ^ L := Nat.two_pow_pos L
  omega

/-- An encoded message is an interleaved Reed–Solomon codeword on the level's points. -/
theorem tr_encode_mem (A : Arith F K) (hA : A.Correct) {lanes cols : ℕ} (d : ℕ)
    (M : Message F lanes cols) :
    tr (encode A d M) ∈ ReedSolomon.Interleaved.irsCode (domainEmb A hA d)
      (2 ^ lanes * 2 ^ cols) (2 ^ lanes) := by
  rw [ReedSolomon.Interleaved.irsCode, Nat.mul_div_cancel_left _ (Nat.two_pow_pos lanes)]
  refine (Code.mem_moduleInterleavedCode_iff _ _ _ _ _ _).2 fun ℓ => ?_
  refine ReedSolomon.mem_code_of_polynomial_of_natDegree_lt_of_eval (rowPoly A hA (M ℓ))
    (rowPoly_natDegree A hA (M ℓ)) fun q => ?_
  change A.encodeAt cols (M ℓ) q.val = Polynomial.eval (domainEmb A hA d q) (rowPoly A hA (M ℓ))
  rw [domainEmb_apply, rowPoly_eval A hA]

/-- Encoding is injective once the code has at least as many positions as coefficients. -/
theorem encode_injective (A : Arith F K) (hA : A.Correct) {lanes cols d : ℕ} (hcd : cols ≤ d) :
    Function.Injective (fun M : Message F lanes cols => tr (encode A d M)) := by
  intro M M' h
  funext ℓ c
  -- the two row polynomials agree on the `2^d` distinct points
  have hzero : rowPoly A hA (M ℓ) - rowPoly A hA (M' ℓ) = 0 := by
    refine Polynomial.eq_zero_of_natDegree_lt_card_of_eval_eq_zero _ (domainEmb A hA d).injective
      (fun q => ?_) ?_
    · have := congrFun (congrFun h q) ℓ
      simp only [tr, encode] at this
      rw [domainEmb_apply, Polynomial.eval_sub, rowPoly_eval A hA, rowPoly_eval A hA, this, sub_self]
    · refine (Polynomial.natDegree_sub_le _ _).trans_lt ?_
      rw [Fintype.card_fin]
      have h1 := rowPoly_natDegree A hA (M ℓ)
      have h2 := rowPoly_natDegree A hA (M' ℓ)
      have h3 : 2 ^ cols ≤ 2 ^ d := Nat.pow_le_pow_right (by norm_num) hcd
      omega
  -- so their coefficient vectors agree, by linear independence
  have hsum : ∑ c', (M ℓ c' - M' ℓ c') • xP A hA cols c' = 0 := by
    rw [← hzero]
    simp only [rowPoly, ← Finset.sum_sub_distrib, sub_smul, Polynomial.smul_eq_C_mul, ← sub_mul,
      Polynomial.C_sub]
  have := Fintype.linearIndependent_iff.1 (xP_linearIndependent A hA cols) _ hsum c
  exact sub_eq_zero.1 this

/-- Closeness of tables is relative block Hamming distance. -/
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


/-! ## The list size -/

/-- A `fast100` schedule has a level 0. -/
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

/-- At rate `2^-r`, `1/(2ηρ) = 25·2^r`. -/
theorem johnson_value (r : ℕ) : 1 / (2 * eta * (2⁻¹ : ℝ) ^ r) = ((25 * 2 ^ r : ℕ) : ℝ) := by
  unfold eta
  push_cast
  rw [inv_pow]
  have : (2 : ℝ) ^ r ≠ 0 := pow_ne_zero _ two_ne_zero
  field_simp
  ring

open Classical in
/-- **The list size.** At most `listBound₀ = 25·2^r` messages encode within the level-0 radius of
any table: ArkLib's Johnson bound for the level-0 interleaved Reed–Solomon code. -/
theorem closeMsgs_card_le (A : Arith F K) (hA : A.Correct) (sch : Schedule) (m : ℕ)
    (hsch : fast100 m = some sch) (C : Oracle F) :
    (Finset.univ.filter fun M => M ∈ closeMsgs A sch C).card ≤ sch.listBound₀ := by
  -- level 0
  obtain ⟨l0, rest, hlev⟩ : ∃ l0 rest, sch.levels = l0 :: rest := by
    cases h : sch.levels with
    | nil => exact absurd h (fast100_levels_ne_nil m sch hsch)
    | cons l0 rest => exact ⟨l0, rest, rfl⟩
  have hL : sch.level₀ = l0 := by simp [Schedule.level₀, hlev]
  have hbound : sch.listBound₀ = l0.listBound := by simp [Schedule.listBound₀, hlev]
  rw [hbound, Level.listBound]
  -- the level-0 code and the committed table as an interleaved word
  set d := sch.level₀.logLen with hd
  set cols := sch.level₀.logCols with hcols
  have hdc : d = cols + sch.level₀.logInvRate := rfl
  have : NeZero (2 ^ sch.k0) := ⟨pow_ne_zero _ two_ne_zero⟩
  have : NeZero (2 ^ d) := ⟨pow_ne_zero _ two_ne_zero⟩
  set code := ReedSolomon.Interleaved.irsCode (domainEmb A hA d) (2 ^ sch.k0 * 2 ^ cols)
    (2 ^ sch.k0) with hcode
  set y := tr (asTable sch C)
  set S := Finset.univ.filter fun M => M ∈ closeMsgs A sch C
  -- the close messages inject into the close codewords
  have hinj : Function.Injective (fun M : Message F sch.k0 cols => tr (encode A d M)) :=
    encode_injective A hA (by omega)
  have hsub : ((S.image fun M => tr (encode A d M) : Finset _) : Set _) ⊆
      Code.closeCodewordsRel (code : Set (Fin (2 ^ d) → Fin (2 ^ sch.k0) → F)) y
        sch.level₀.radius := by
    intro w hw
    simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe] at hw
    obtain ⟨M, hM, rfl⟩ := hw
    simp only [S, Finset.mem_filter, Finset.mem_univ, true_and, closeMsgs, Set.mem_setOf_eq] at hM
    exact Code.mem_closeCodewordsRel_iff.2
      ⟨tr_encode_mem A hA d M, relHammingDist_tr_le _ _ _ hM⟩
  have hcardS : (S.card : ℕ∞) ≤
      Code.Lambda (code : Set (Fin (2 ^ d) → Fin (2 ^ sch.k0) → F)) sch.level₀.radius := by
    rw [← Finset.card_image_of_injective S hinj, ← Set.encard_coe_eq_coe_finsetCard]
    exact Code.encard_le_Lambda_of_subset_closeCodewordsRel hsub
  -- the rate is `2^-r`, so the radius is the Johnson radius
  have hks : 0 < 2 ^ sch.k0 * 2 ^ cols / 2 ^ sch.k0 := by
    rw [Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]; exact Nat.two_pow_pos _
  have hrate : (LinearCode.alphabetRate code : ℝ) = sch.level₀.rate := by
    rw [hcode, ReedSolomon.Interleaved.alphabetRate_irsCode_eq_min,
      Nat.mul_div_cancel_left _ (Nat.two_pow_pos _), Fintype.card_fin,
      min_eq_left (Nat.pow_le_pow_right (by norm_num) (by omega)), Level.rate, hdc]
    push_cast
    rw [pow_add, inv_pow]
    have : (2 : ℝ) ^ cols ≠ 0 := pow_ne_zero _ two_ne_zero
    field_simp
  have hJ := CodingTheory.irs_lambda_le_johnson_mds (domainEmb A hA d) (2 ^ sch.k0 * 2 ^ cols)
    (2 ^ sch.k0) hks eta (by unfold eta; norm_num)
  simp only at hJ
  rw [← hcode, hrate] at hJ
  have hrad : 1 - Real.sqrt sch.level₀.rate - eta = sch.level₀.radius := rfl
  rw [hrad, Level.rate, johnson_value, ENNReal.ofReal_natCast] at hJ
  -- conclude
  have h1 : ((S.card : ℕ∞) : ℝ≥0∞) ≤ ((25 * 2 ^ sch.level₀.logInvRate : ℕ) : ℝ≥0∞) :=
    (ENat.toENNReal_le.2 hcardS).trans hJ
  rw [← hL]
  exact_mod_cast h1

end FlockSoundness
