import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Definitions.Pouw.Fp8Atom.H1TPinned
import Proofs.Pouw.Fp8Atom.H1TChecks
import Proofs.Pouw.Fp8Atom.Fp32Proofs

/-!
# Proofs of H-1T's pinned statements

The conformance statements assemble the kernel-evaluated parts of `Pouw.Fp8Atom.H1TChecks`; the chain's running words
follow by induction from its local checks. The structural statements unfold the definitions. `RnErr` is
`roundEven`'s half-unit error scaled by the ulp. `RnAddMovesHalf` follows from it: an absorbed summand below one ulp
of `r` leaves `r + i` in `r`'s binade or below, so `|i| ≤ ulp (r + i) / 2 ≤ ulp r / 2`; a summand of an ulp or more
is `RnAddMoves`.
-/

namespace Pouw.Fp8Atom.H1T.Proofs

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T

/-! ## Conformance -/

theorem h1tFormTable : H1TFormTable := by
  refine ⟨by decide +kernel, fun v hv => ?_⟩
  have h := List.all_eq_true.mp Checks.formTable_all v hv
  simpa [FormVec.holds] using h

theorem h1tTagTable : H1TTagTable := by
  refine ⟨by decide +kernel, fun v hv => ?_⟩
  have h := List.all_eq_true.mp Checks.tagTable_all v hv
  simpa [TagVec.holds, and_assoc] using h

theorem h1tSaltConformance : H1TSaltConformance := by
  refine ⟨by decide +kernel, fun v hv => ?_⟩
  have h := List.all_eq_true.mp Checks.saltVectors_all v hv
  simpa [SaltVec.holds] using h

theorem rowVec_chain {v : RowVec} (h : v.holds = true) : ∀ p < 2 * v.T,
    freshWord v.T v.xs v.bs v.j v.σ p = v.I.getD p 0 ∧
      wordQ (v.R.getD p 0) = some (runVal v.T v.xs v.bs v.j v.σ p) := by
  unfold RowVec.holds at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, beq_iff_eq, Option.isSome_iff_exists] at h
  obtain ⟨⟨hI, h0⟩, hR⟩ := h
  have key : ∀ p < 2 * v.T, wordQ (v.R.getD p 0) = some (runVal v.T v.xs v.bs v.j v.σ p) := by
    intro p
    induction p with
    | zero =>
      intro hp
      obtain ⟨⟨hw, ⟨x, hx⟩⟩, -⟩ := hI 0 hp
      simp only [runVal, freshVal, wordVal, hw, hx, h0, Option.getD_some]
    | succ p ih =>
      intro hp
      obtain ⟨⟨hw, ⟨y, hy⟩⟩, -⟩ := hI (p + 1) hp
      have hr := hR p (by omega)
      rw [ih (by omega), hy] at hr
      simp only [runVal, freshVal, wordVal, hw, hy, Option.getD_some, hr, Option.bind_some, Option.map_some]
  exact fun p hp => ⟨(hI p hp).1.1, key p hp⟩

theorem h1tChainConformance : H1TChainConformance :=
  ⟨by decide +kernel, fun v hv => rowVec_chain (List.all_eq_true.mp Checks.rowVectors_all v hv)⟩

/-! ## The salt fields -/

theorem h1tReadsFields : H1TReadsFields := by
  intro U g g' h c
  have e : U.rowSalt g c.1 = U.rowSalt g' c.1 := by
    funext τ
    unfold Unit.rowSalt
    split_ifs
    · exact h _ _
    · rfl
  unfold Unit.word
  rw [e]

theorem shiftRight_small {w b : ℕ} (hw : w < 128) (hb : 7 ≤ b) : w >>> b = 0 := by
  rw [Nat.shiftRight_eq_div_pow]
  apply Nat.div_eq_of_lt
  calc w < 2 ^ 7 := by norm_num; exact hw
    _ ≤ 2 ^ b := Nat.pow_le_pow_right (by norm_num) hb

/-- Raw words below `2^7` have the fields of the zero salt: every bit `extract` reads is bit 7 or above. -/
theorem extract_small {r : RawSlice} (h : ∀ ℓ, (r ℓ).val < 128) : extract r = extract rawZero := by
  have hb : ∀ ℓ b, 7 ≤ b → (r ℓ).val >>> b = 0 := fun ℓ b hb => shiftRight_small (h ℓ) hb
  have hz : ∀ ℓ b, (rawZero ℓ).val >>> b = 0 := fun _ _ => by simp [rawZero]
  unfold extract
  congr 1
  · funext l
    simp only [bitAt, hz, hb _ _ (show 7 ≤ (if l.val % 2 = 0 then 15 else 31) by split <;> omega)]
  · funext i
    simp only [pick3, bitAt, hz, hb _ 31 (by norm_num), hb _ 15 (by norm_num)]
  · funext i
    simp only [pick3, bits3, hz, hb _ 23 (by norm_num), hb _ 7 (by norm_num)]

theorem h1tNotRawWord : H1TNotRawWord := by
  intro U c i τ ℓ e
  let one : RawSlice := Function.update rawZero ℓ ⟨1, by norm_num⟩
  let g0 : U.Salt := fun _ _ => rawZero
  let g1 : U.Salt := fun i' τ' => if i' = i ∧ τ' = τ then one else rawZero
  have hone : ∀ ℓ', (one ℓ').val < 128 := by
    intro ℓ'
    by_cases hℓ : ℓ' = ℓ
    · subst hℓ; simp [one]
    · simp [one, Function.update_of_ne hℓ, rawZero]
  have hf : ∀ i' τ', extract (g0 i' τ') = extract (g1 i' τ') := by
    intro i' τ'
    simp only [g0, g1]
    split_ifs
    · exact (extract_small hone).symm
    · rfl
  have hw := h1tReadsFields U g0 g1 hf c
  have e0 := congrFun e g0
  have e1 := congrFun e g1
  rw [hw, e1] at e0
  simp [g0, g1, one, rawZero] at e0

/-! ## The tag tuples -/

theorem h1tTupleInjective : H1TTupleInjective := by
  intro j j' hj hj' e
  have e0 := congrFun e 0
  have e1 := congrFun e 1
  have e2 := congrFun e 2
  simp only [tuple, pick3, A0, A1, Fin.isValue, Fin.val_zero, Fin.val_one, Fin.val_two] at e0 e1 e2
  norm_num at e0 e1 e2
  split_ifs at e1 e2 <;> omega

theorem h1tTupleWrap : H1TTupleWrap := by
  intro j
  have h1 : (j + 88412) % 23 = j % 23 := by omega
  have h2 : (j + 88412) / 23 % 62 = j / 23 % 62 := by omega
  have h3 : (j + 88412) / 1426 % 62 = j / 1426 % 62 := by omega
  simp only [tuple, h1, h2, h3]

theorem freshWord_col {T : ℕ} {x b b' : ℕ → Code} {j j' : ℕ} {σ : ℕ → SliceSalt}
    (h : ∀ τ, colLanes b j τ = colLanes b' j' τ) (p : ℕ) : freshWord T x b j σ p = freshWord T x b' j' σ p := by
  unfold freshWord; rw [h, h]

theorem runVal_col {T : ℕ} {x b b' : ℕ → Code} {j j' : ℕ} {σ : ℕ → SliceSalt}
    (h : ∀ τ, colLanes b j τ = colLanes b' j' τ) (p : ℕ) : runVal T x b j σ p = runVal T x b' j' σ p := by
  induction p with
  | zero => simp only [runVal, freshVal, freshWord_col h]
  | succ p ih => simp only [runVal, freshVal, freshWord_col h, ih]

theorem h1tWrapDuplicate : H1TWrapDuplicate := by
  intro U j j' hj hb i a
  have hc : ∀ τ, colLanes (U.col j) j.val τ = colLanes (U.col j') j'.val τ := by
    intro τ
    funext l
    unfold colLanes
    split_ifs
    · simp only [Unit.col, hb]
    · rw [hj, h1tTupleWrap]
  funext g
  cases a with
  | inl p => simp only [Unit.word, freshVal, freshWord_col hc]
  | inr p => simp only [Unit.word, runVal_col hc]

/-! ## Counting -/

/-- The credited `R` positions `1 ≤ p ≤ 2T − 3` are `2T − 3` of them. -/
def posEquiv (T : ℕ) : {p : Fin (2 * T) // 1 ≤ p.val ∧ p.val + 3 ≤ 2 * T} ≃ Fin (2 * T - 3) where
  toFun p := ⟨p.1.val - 1, by omega⟩
  invFun q := ⟨⟨q.val + 1, by omega⟩, by simp only; omega⟩
  left_inv p := by ext; simp only; omega
  right_inv q := by ext; simp

theorem h1tPosCard : H1TPosCard := by
  intro T hT
  rw [Fintype.card_sum, Fintype.card_fin, Fintype.card_congr (posEquiv T), Fintype.card_fin]
  omega

theorem h1tNonVacuous : H1TNonVacuous := by
  refine ⟨⟨1, 32768, 88412, fun _ _ => ⟨0x38, by norm_num⟩, fun _ _ => ⟨0x38, by norm_num⟩⟩,
    ⟨by norm_num, by norm_num, fun _ _ => rfl, fun _ _ => rfl⟩, rfl, rfl, rfl, rfl, ?_⟩
  exact h1tPosCard 1130 (by norm_num)

theorem halfOfInvolution : HalfOfInvolution := by
  intro α _ ψ P hψ hP
  classical
  set A := Finset.univ.filter fun a => P a = true
  have hdisj : Disjoint A (A.image ψ) := by
    rw [Finset.disjoint_left]
    intro a ha hb'
    obtain ⟨b, hb, rfl⟩ := Finset.mem_image.1 hb'
    simp only [A, Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    rw [hP b hb] at ha
    exact Bool.false_ne_true ha
  have hcard : (A.image ψ).card = A.card := Finset.card_image_of_injective _ hψ.injective
  have hu := Finset.card_union_of_disjoint hdisj
  have hle : (A ∪ A.image ψ).card ≤ Fintype.card α := Finset.card_le_univ _
  omega

/-! ## The FP32 add -/

theorem roundEven_err (m : ℚ) : |(roundEven m : ℚ) - m| ≤ 1 / 2 := by
  have h1 := Int.floor_le m
  have h2 := Int.lt_floor_add_one m
  unfold roundEven
  dsimp only
  split_ifs with a b c <;> (try simp only [not_lt] at *) <;> push_cast <;> rw [abs_le] <;> constructor <;> linarith

theorem rp_err (t : ℚ) : |rp t - t| ≤ ulp t / 2 := by
  have hu := ulp_pos t
  have e : rp t - t = ((roundEven (t / ulp t) : ℚ) - t / ulp t) * ulp t := by
    unfold rp; rw [sub_mul, div_mul_cancel₀ _ hu.ne']
  rw [e, abs_mul, abs_of_pos hu]
  have := roundEven_err (t / ulp t)
  nlinarith

theorem rn_err (y : ℚ) : |rn y - y| ≤ ulp y / 2 := by
  rw [rn_eq]
  rcases lt_or_ge y 0 with h | h
  · rw [ite_eq_left h, abs_of_neg h]
    have := rp_err (-y)
    rw [ulp_neg] at this
    calc |(-1) * rp (-y) - y| = |rp (-y) - -y| := by rw [← abs_neg]; congr 1; ring
      _ ≤ _ := this
  · rw [ite_eq_right (not_lt.2 h), abs_of_nonneg h, one_mul]
    exact rp_err y

theorem rnErr : RnErr := rn_err

/-- Below one ulp of a positive representable `r`, `r + i` is in `r`'s binade or below. -/
theorem ulp_add_le_pos {r i : ℚ} (hr : Representable r) (h0 : 0 < r) (hi : |i| < ulp r) :
    ulp (r + i) ≤ ulp r := by
  obtain ⟨n, hn, n1, n2⟩ := pos_sig hr h0
  have hu := ulp_pos r
  have hk1 : (2 : ℚ) ^ (binade r + 1) = 2 ^ 24 * ulp r := by rw [ulp, abs_of_pos h0, two_pow_24]
  obtain ⟨i1, i2⟩ := abs_lt.1 hi
  have n2' : (n : ℚ) + 1 ≤ 2 ^ 24 := by exact_mod_cast (show n + 1 ≤ 2 ^ 24 by omega)
  have n1' : (2 : ℚ) ^ 23 ≤ n := by exact_mod_cast n1
  have hx : 0 < r + i := by nlinarith [mul_le_mul_of_nonneg_right n1' hu.le]
  have hhi : r + i < 2 ^ (binade r + 1) := by rw [hk1]; nlinarith [mul_le_mul_of_nonneg_right n2' hu.le]
  have hb : binade (r + i) ≤ binade r := by
    have := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 (lt_of_le_of_lt (binade_spec hx).1 hhi)
    omega
  unfold ulp
  rw [abs_of_pos hx, abs_of_pos h0]
  exact zpow_le_zpow_right₀ one_le_two (by omega)

theorem ulp_add_le {r i : ℚ} (hr : Representable r) (h0 : r ≠ 0) (hi : |i| < ulp r) : ulp (r + i) ≤ ulp r := by
  rcases h0.lt_or_gt with h | h
  · have := ulp_add_le_pos (representable_neg hr) (neg_pos.2 h) (i := -i) (by rwa [ulp_neg, abs_neg])
    rwa [show -r + -i = -(r + i) by ring, ulp_neg, ulp_neg] at this
  · exact ulp_add_le_pos hr h hi

theorem rnAdd_moves_half (r i : ℚ) (hr : Representable r) (h0 : r ≠ 0) (hi : ulp r < 2 * |i|) : rnAdd r i ≠ r := by
  rcases le_or_gt (ulp r) |i| with h | h
  · exact rnAdd_moves r i hr h0 h
  intro e
  have herr := rn_err (r + i)
  unfold rnAdd at e
  rw [e, show r - (r + i) = -i by ring, abs_neg] at herr
  have := ulp_add_le hr h0 h
  linarith

theorem rnAddMovesHalf : RnAddMovesHalf := rnAdd_moves_half

end Pouw.Fp8Atom.H1T.Proofs
