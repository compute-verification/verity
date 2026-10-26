import Proofs.Flock.Soundness.NovelBasis
import Proofs.Flock.Soundness.LigeritoLevels
import Definitions.Flock.Accounting.Padded

/-!
# The padded level-0 code: `RS[2^d, k]` on the first `k` novel-basis monomials

C-Flock ZK's level-0 lane of `2^cols` message coefficients and `t` padding coefficients is a codeword of
`RS[2^d, k]`, `k = 2^cols + t`: the evaluation of `Σ_{c < k} a_c·X̂_c` at `d` bits (`encAt`). This
file gives that code what `ListSize`, `LigeritoLevels` and `LigeritoDecode` give a level's code:

* its codewords are exactly the polynomials of degree below `k` (`NovelBasis`), so `encAt` is in
  ArkLib's `ReedSolomon.code` and is injective, and a `K` codeword that agrees with an `F` row on
  `T` is an `F` encoding there (`exists_encAt_of_agree`);
* a `K` combination of the `k` monomials that vanishes on `k` positions is zero (`encKAt_eq_zero`);
* at most `⌊100·2^d/k⌋` row families encode within `1 − √(k/2^d) − η` of a table
  (`card_closeRows_le`, ArkLib's Johnson bound);
* `prCoin_mca_le` bounds its MCA events by `padMcaA / 2^256` (`prCoin_pad_mca_le`).
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- A lane of `k` coefficients on the first `k` monomials of the novel basis at `d` bits. -/
def encAt (A : Arith F K) (d k : ℕ) (a : Fin k → F) (q : ℕ) : F :=
  ∑ c, a c * A.Xhat c.val d (A.omega q)

/-- The same over `K`. -/
def encKAt (A : Arith F K) (d k : ℕ) (b : Fin k → K) (q : ℕ) : K :=
  ∑ c, b c * A.embed (A.Xhat c.val d (A.omega q))

variable (A : Arith F K) (hA : A.Correct)

/-- A lane's polynomial. -/
noncomputable def encPoly (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) (a : Fin k → F) : F[X] :=
  ∑ c, C (a c) * xP A hA d hd (Fin.castLE hk c)

theorem encPoly_eval (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) (a : Fin k → F) (q : ℕ) :
    (encPoly A hA d k hd hk a).eval (A.omega q) = encAt A d k a q := by
  simp only [encPoly, eval_finset_sum, eval_mul, eval_C, encAt]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [← xP_eval A hA d hd]
  rfl

theorem encPoly_natDegree (d k : ℕ) (hd : d ≤ 40) (hk : k ≤ 2 ^ d) (hk0 : 0 < k) (a : Fin k → F) :
    (encPoly A hA d k hd hk a).natDegree < k := by
  have h : (encPoly A hA d k hd hk a).natDegree ≤ k - 1 := by
    refine natDegree_sum_le_of_forall_le _ _ fun c _ => (natDegree_C_mul_le _ _).trans ?_
    have := xP_natDegree_le A hA d hd (Fin.castLE hk c)
    have := c.isLt
    simp only [Fin.coe_castLE] at *
    omega
  omega

/-- The lanes' `F` encodings are codewords of `RS[2^d, k]`. -/
theorem encAt_mem (d k : ℕ) (hd : d ≤ 40) (hd64 : d ≤ 64) (hk : k ≤ 2 ^ d) (hk0 : 0 < k)
    (a : Fin k → F) :
    (fun q : Fin (2 ^ d) => encAt A d k a q.val) ∈ ReedSolomon.code (domainEmb A hA d hd64) k :=
  ReedSolomon.mem_code_of_polynomial_of_natDegree_lt_of_eval (encPoly A hA d k hd hk a)
    (encPoly_natDegree A hA d k hd hk hk0 a) fun q => by
      rw [domainEmb_apply, encPoly_eval]

/-- A `K` combination of the `k` monomials is a codeword of `RS[2^d, k]` over `K`. -/
theorem encKAt_mem (d k : ℕ) (hd : d ≤ 40) (hd64 : d ≤ 64) (hk : k ≤ 2 ^ d) (hk0 : 0 < k)
    (b : Fin k → K) :
    (fun q : Fin (2 ^ d) => encKAt A d k b q.val) ∈ ReedSolomon.code (domainK A hA d hd64) k := by
  refine ReedSolomon.mem_code_of_polynomial_of_natDegree_lt_of_eval
    (∑ c, C (b c) * (xP A hA d hd (Fin.castLE hk c)).map A.embed) ?_ fun q => ?_
  · have h : (∑ c, C (b c) * (xP A hA d hd (Fin.castLE hk c)).map A.embed).natDegree ≤ k - 1 := by
      refine natDegree_sum_le_of_forall_le _ _ fun c _ => (natDegree_C_mul_le _ _).trans ?_
      rw [natDegree_map]
      have := xP_natDegree_le A hA d hd (Fin.castLE hk c)
      have := c.isLt
      simp only [Fin.coe_castLE] at *
      omega
    omega
  · simp only [encKAt, domainK, Function.Embedding.trans_apply, Function.Embedding.coeFn_mk,
      domainEmb_apply, eval_finset_sum, eval_mul, eval_C, eval_map_apply]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [← xP_eval A hA d hd]
    rfl

include hA in
/-- **A combination of `k` monomials vanishing on `k` positions is zero.** -/
theorem encAt_eq_zero (d k : ℕ) (hd : d ≤ 40) (hd64 : d ≤ 64) (hk : k ≤ 2 ^ d) (a : Fin k → F)
    (T : Finset (Fin (2 ^ d))) (hT : k ≤ T.card) (h : ∀ q ∈ T, encAt A d k a q.val = 0) : a = 0 := by
  classical
  by_cases hk0 : k = 0
  · subst hk0; funext c; exact c.elim0
  have hp : encPoly A hA d k hd hk a = 0 := by
    refine Polynomial.eq_zero_of_natDegree_lt_card_of_eval_eq_zero' _
      (T.image fun q => A.omega q.val) (fun ω hω => ?_) ?_
    · obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 hω
      rw [encPoly_eval]; exact h q hq
    · rw [Finset.card_image_of_injOn fun q _ q' _ hqq' =>
        Fin.ext (hA.omega_injective d hd64 q.isLt q'.isLt hqq')]
      exact lt_of_lt_of_le (encPoly_natDegree A hA d k hd hk (by omega) a) hT
  have hsum : ∑ c, a c • xP A hA d hd (Fin.castLE hk c) = 0 := by
    rw [← hp]; simp only [encPoly, Polynomial.smul_eq_C_mul]
  funext c
  exact Fintype.linearIndependent_iff.1
    ((xP_linearIndependent A hA d hd).comp _ (Fin.castLE_injective hk)) _ hsum c

include hA in
/-- **Over `K`**: a `K` combination of `k` monomials vanishing on `k` positions is zero. -/
theorem encKAt_eq_zero (d k : ℕ) (hd : d ≤ 40) (hd64 : d ≤ 64) (hk : k ≤ 2 ^ d) (b : Fin k → K)
    (T : Finset (Fin (2 ^ d))) (hT : k ≤ T.card) (h : ∀ q ∈ T, encKAt A d k b q.val = 0) :
    b = 0 := by
  classical
  set b₀ : Fin k → F := fun c => (limbs A hA (b c)).1
  set b₁ : Fin k → F := fun c => (limbs A hA (b c)).2
  have hsplit : ∀ q : ℕ, encKAt A d k b q = A.ofLimbs (encAt A d k b₀ q) (encAt A d k b₁ q) := by
    intro q
    unfold encKAt encAt
    rw [← ofLimbs_sum]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [← ofLimbs_mul_embed, ofLimbs_limbs]
  have h0 : ∀ q ∈ T, encAt A d k b₀ q.val = 0 ∧ encAt A d k b₁ q.val = 0 := by
    intro q hq
    have := h q hq
    rw [hsplit, ← ofLimbs_zero A] at this
    exact ofLimbs_inj A hA this
  have e0 := encAt_eq_zero A hA d k hd hd64 hk b₀ T hT fun q hq => (h0 q hq).1
  have e1 := encAt_eq_zero A hA d k hd hd64 hk b₁ T hT fun q hq => (h0 q hq).2
  funext c
  rw [← ofLimbs_limbs A hA (b c)]
  have := congrFun e0 c
  have := congrFun e1 c
  simp only [b₀, b₁, Pi.zero_apply] at *
  simp [*, ofLimbs_zero]

/-- **A `K` codeword agreeing with an `F` row is an `F` encoding there.** -/
theorem exists_encAt_of_agree (d k : ℕ) (hd : d ≤ 40) (hd64 : d ≤ 64) (hk : k ≤ 2 ^ d)
    (hk0 : 0 < k) (row : Fin (2 ^ d) → F) (T : Finset (Fin (2 ^ d))) (c : Fin (2 ^ d) → K)
    (hc : c ∈ ReedSolomon.code (domainK A hA d hd64) k) (hcT : ∀ q ∈ T, c q = A.embed (row q)) :
    ∃ a : Fin k → F, ∀ q ∈ T, encAt A d k a q.val = row q := by
  classical
  have : NeZero k := ⟨by omega⟩
  obtain ⟨P, hPdeg, hPc⟩ := ReedSolomon.mem_code_iff_eval_of_ne_zero.1 hc
  obtain ⟨P₀, P₁, hP₀, hPl⟩ := poly_limbs A hA P
  obtain ⟨m, hm⟩ := xP_span_lt A hA d hd k hk P₀ (lt_of_le_of_lt hP₀ hPdeg)
  refine ⟨m, fun q hq => ?_⟩
  have h1 := hcT q hq
  rw [← hPc q] at h1
  simp only [domainK, Function.Embedding.trans_apply, Function.Embedding.coeFn_mk,
    domainEmb_apply] at h1
  rw [hPl, ← ofLimbs_embed] at h1
  have h2 := (ofLimbs_inj A hA h1).1
  rw [← h2, ← hm]
  simp only [encAt, eval_finset_sum, eval_mul, eval_C]
  refine Finset.sum_congr rfl fun c' _ => ?_
  rw [← xP_eval A hA d hd]
  rfl

/-! ## The list size -/

/-- `γ`-close rows: the table's rows agree with the rows' encodings on a common `1 − γ` fraction
of the positions. -/
def CloseRows (d k : ℕ) (γ : ℝ) {s : ℕ} (T : Fin s → Fin (2 ^ d) → F) (R : Fin s → Fin k → F) :
    Prop :=
  ((Finset.univ.filter fun q : Fin (2 ^ d) => ∀ ℓ, T ℓ q = encAt A d k (R ℓ) q.val).card : ℝ) ≥
    (1 - γ) * 2 ^ d

include hA in
theorem encAt_injective (d k : ℕ) (hd : d ≤ 40) (hd64 : d ≤ 64) (hk : k ≤ 2 ^ d) :
    Function.Injective fun (a : Fin k → F) (q : Fin (2 ^ d)) => encAt A d k a q.val := by
  intro a a' h
  have hz := encAt_eq_zero A hA d k hd hd64 hk (a - a') Finset.univ
    (by rw [Finset.card_univ, Fintype.card_fin]; exact hk) fun q _ => by
      have := congrFun h q
      simp only at this
      simp only [encAt, Pi.sub_apply, sub_mul, Finset.sum_sub_distrib]
      unfold encAt at this
      rw [this, sub_self]
  exact sub_eq_zero.1 hz

include hA in
open Classical in
/-- **The list size.** At most `⌊100·2^d/k⌋` families of `s` rows encode within `1 − √(k/2^d) − η`
of any table: ArkLib's Johnson bound for the interleaved `RS[2^d, k]`. -/
theorem card_closeRows_le (d k : ℕ) (hd : d ≤ 40) (hd64 : d ≤ 64) (hk : k ≤ 2 ^ d) (hk0 : 0 < k)
    {s : ℕ} (hs : 0 < s) (T : Fin s → Fin (2 ^ d) → F) :
    (Finset.univ.filter fun R : Fin s → Fin k → F =>
      CloseRows A d k (1 - Real.sqrt ((k : ℝ) / 2 ^ d) - Accounting.eta) T R).card ≤ 100 * 2 ^ d / k := by
  have : NeZero s := ⟨by omega⟩
  have : NeZero (2 ^ d) := ⟨pow_ne_zero _ two_ne_zero⟩
  set γ := 1 - Real.sqrt ((k : ℝ) / 2 ^ d) - Accounting.eta with hγ
  set code := ReedSolomon.Interleaved.irsCode (domainEmb A hA d hd64) (s * k) s with hcode
  set y : Fin (2 ^ d) → Fin s → F := fun q ℓ => T ℓ q
  set S := Finset.univ.filter fun R : Fin s → Fin k → F => CloseRows A d k γ T R
  let enc : (Fin s → Fin k → F) → Fin (2 ^ d) → Fin s → F := fun R q ℓ => encAt A d k (R ℓ) q.val
  have hinj : Function.Injective enc := by
    intro R R' h
    funext ℓ
    exact encAt_injective A hA d k hd hd64 hk (funext fun q => congrFun (congrFun h q) ℓ)
  have hmem : ∀ R, enc R ∈ code := by
    intro R
    rw [hcode, ReedSolomon.Interleaved.irsCode, Nat.mul_div_cancel_left _ (by omega)]
    refine (Code.mem_moduleInterleavedCode_iff _ _ _ _ _ _).2 fun ℓ => ?_
    exact encAt_mem A hA d k hd hd64 hk hk0 (R ℓ)
  have hdist : ∀ R ∈ S, (Code.relHammingDist y (enc R) : ℝ) ≤ γ := by
    intro R hR
    have hR' := (Finset.mem_filter.1 hR).2
    unfold CloseRows at hR'
    rw [Code.relHammingDist_coe]
    have hn : (0 : ℝ) < Fintype.card (Fin (2 ^ d)) := by exact_mod_cast Fintype.card_pos
    rw [div_le_iff₀ hn, Fintype.card_fin]
    have hsplit := Finset.card_filter_add_card_filter_not
      (s := (Finset.univ : Finset (Fin (2 ^ d)))) (fun q => ∀ ℓ, T ℓ q = encAt A d k (R ℓ) q.val)
    have hd' : hammingDist y (enc R) =
        (Finset.univ.filter fun q : Fin (2 ^ d) => ¬ ∀ ℓ, T ℓ q = encAt A d k (R ℓ) q.val).card := by
      unfold hammingDist
      congr 1
      ext q
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, ne_eq, funext_iff, y, enc]
    rw [hd']
    rw [Finset.card_univ, Fintype.card_fin] at hsplit
    have hs' : ((Finset.univ.filter fun q : Fin (2 ^ d) =>
        ¬ ∀ ℓ, T ℓ q = encAt A d k (R ℓ) q.val).card : ℝ) =
        2 ^ d - (Finset.univ.filter fun q : Fin (2 ^ d) =>
          ∀ ℓ, T ℓ q = encAt A d k (R ℓ) q.val).card := by
      rw [eq_sub_iff_add_eq, add_comm]
      exact_mod_cast hsplit
    rw [hs']
    push_cast
    linarith [hR']
  have hsub : ((S.image enc : Finset _) : Set _) ⊆
      Code.closeCodewordsRel (code : Set (Fin (2 ^ d) → Fin s → F)) y γ := by
    intro w hw
    simp only [Finset.coe_image, Set.mem_image, Finset.mem_coe] at hw
    obtain ⟨R, hR, rfl⟩ := hw
    exact Code.mem_closeCodewordsRel_iff.2 ⟨hmem R, hdist R hR⟩
  have hcardS : (S.card : ℕ∞) ≤ Code.Lambda (code : Set (Fin (2 ^ d) → Fin s → F)) γ := by
    rw [← Finset.card_image_of_injective S hinj, ← Set.encard_coe_eq_coe_finsetCard]
    exact Code.encard_le_Lambda_of_subset_closeCodewordsRel hsub
  have hks : 0 < s * k / s := by rw [Nat.mul_div_cancel_left _ (by omega)]; exact hk0
  have hrate : (LinearCode.alphabetRate code : ℝ) = (k : ℝ) / 2 ^ d := by
    rw [hcode, ReedSolomon.Interleaved.alphabetRate_irsCode_eq_min,
      Nat.mul_div_cancel_left _ (by omega), Fintype.card_fin, min_eq_left hk]
    push_cast
    rfl
  have hJ := CodingTheory.irs_lambda_le_johnson_mds (domainEmb A hA d hd64) (s * k) s hks Accounting.eta
    (by unfold Accounting.eta; norm_num)
  simp only at hJ
  rw [← hcode, hrate] at hJ
  have hval : 1 / (2 * Accounting.eta * ((k : ℝ) / 2 ^ d)) = 100 * 2 ^ d / k := by
    unfold Accounting.eta
    have : (k : ℝ) ≠ 0 := by exact_mod_cast hk0.ne'
    field_simp
    ring
  rw [hval] at hJ
  have h1 : ((S.card : ℕ∞) : ℝ≥0∞) ≤ ENNReal.ofReal (100 * 2 ^ d / k) :=
    (ENat.toENNReal_le.2 hcardS).trans hJ
  have h2 : (S.card : ℝ) ≤ 100 * 2 ^ d / k := by
    have hnn : (0 : ℝ) ≤ 100 * 2 ^ d / k := by positivity
    have h1' : ((S.card : ℕ) : ℝ≥0∞) ≤ ENNReal.ofReal (100 * 2 ^ d / k) := by simpa using h1
    simpa using ENNReal.toReal_le_of_le_ofReal hnn h1'
  have h3 : (S.card : ℝ) * k ≤ 100 * 2 ^ d := by
    have hkpos : (0 : ℝ) < k := by exact_mod_cast hk0
    rwa [le_div_iff₀ hkpos] at h2
  rw [Nat.le_div_iff_mul_le hk0]
  exact_mod_cast h3

/-! ## The MCA bound on the padded code -/

theorem padRate_pos (l : Level) (t : ℕ) : 0 < l.padRate t := by
  unfold Level.padRate
  have : (0 : ℝ) < ((2 ^ l.logCols + t : ℕ) : ℝ) := by
    have := Nat.two_pow_pos l.logCols; exact_mod_cast (by omega : 0 < 2 ^ l.logCols + t)
  positivity

theorem padRadius_lt_johnson (l : Level) (t : ℕ) :
    l.padRadius t <
      1 - ((((2 ^ l.logCols + t - 1 : ℕ) : ℝ) / (2 : ℝ) ^ l.logLen) ^ ((1 : ℝ) / 2)) := by
  have hlt : ((2 ^ l.logCols + t - 1 : ℕ) : ℝ) / (2 : ℝ) ^ l.logLen < l.padRate t := by
    unfold Level.padRate
    apply div_lt_div_of_pos_right _ (by positivity)
    have : 1 ≤ 2 ^ l.logCols + t := by have := Nat.two_pow_pos l.logCols; omega
    exact_mod_cast (by omega : 2 ^ l.logCols + t - 1 < 2 ^ l.logCols + t)
  have hsq : (((2 ^ l.logCols + t - 1 : ℕ) : ℝ) / (2 : ℝ) ^ l.logLen) ^ ((1 : ℝ) / 2) <
      Real.sqrt (l.padRate t) := by
    rw [Real.sqrt_eq_rpow]
    exact Real.rpow_lt_rpow (by positivity) hlt (by norm_num)
  unfold Level.padRadius Accounting.eta
  linarith

/-- **The MCA bound on the padded code.** Whatever two words the prover fixes, the MCA
event of `RS[2^d, 2^cols + t]` over `K` at the padded radius has probability at most
`padMcaA / 2^256`. -/
theorem prCoin_pad_mca_le (l : Level) (t : ℕ)
    (hd : l.logLen ≤ 64) (hk : 1 < 2 ^ l.logCols + t) (hr : 0 < l.padRadius t)
    (U : Fin 2 → Fin (2 ^ l.logLen) → K) :
    prCoin (fun γ : K => IsMCA (AffineLineGenerator K)
      (ReedSolomon.code (domainK A hA l.logLen hd) (2 ^ l.logCols + t)) γ U (l.padRadius t)) ≤
      ENNReal.ofReal (l.padMcaA t / qK) := by
  have hJ := padRadius_lt_johnson l t
  have h := prCoin_mca_le (domainK A hA l.logLen hd) (2 ^ l.logCols + t)
    (l.padRadius t).toNNReal hk
    (by rw [Fintype.card_fin, Real.coe_toNNReal _ hr.le]; push_cast; exact hJ) U
  rw [Real.coe_toNNReal _ hr.le] at h
  refine h.trans (le_of_eq ?_)
  rw [Fintype.card_fin, hA.card_K]
  unfold Level.padMcaA qK
  push_cast
  congr 3
  norm_num

end FlockSoundness
