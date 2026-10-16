import Proofs.Flock.Recursive.HoloExec

/-!
# Piece (a), R's opening count: the registered opening is sound at unique decoding

After both reps' sparse phases, `Model.regPhase` opens their two registered claims together (`opening`, the ring
switches and `γ`) and runs Ligerito at R's registered schedule on the registered word `w` (`regOpening`). Level 0 of
`fast100`'s schedule at `mReg` holds R's stacked bits exactly at rate at most `1/2` (`fast100_uniqueDecoding`), so a
message within `(1 − ρ)/2 − η` of `w` is the only one there. `regOpen_sound`: where no such message packs a table on
which both claims hold, the opening accepts with probability at most `regOpenError`, with no list (the reps' level 0 is
at the list-decoding radius; R's is not).

Ligerito's level 0 is `ligerito_sound_at`'s at the unique-decoding radius `udRadius`: its doom is over the (at most one)
message within `udRadius` of `w` (`card_closeL_ud_le`), the folds' MCA events stay at the Johnson radius, and the level's
queries catch a fold `udRadius`-far from it at a cost of `udQueryError` (`udQueryError_eq`). Every later level is
`ligerito_sound`'s.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge.Recursion
open Game Model Accounting
open scoped ENNReal

/-- **`fast100`'s schedule at `M` holds `M` bits at level 0 at rate `1/2`**: level 0 has `k₀` lanes of `2^logCols`
packed elements of 128 bits, `k₀ + logCols + 7 = M`, for every `M` it covers. -/
theorem fast100_uniqueDecoding (M : ℕ) (s : Accounting.Schedule) (hs : Accounting.fast100 M = some s) :
    s.levels ≠ [] ∧ s.m = M ∧ s.level₀.lanes = s.k0 ∧ s.k0 + s.level₀.logCols + 7 = M ∧
      1 ≤ s.level₀.logInvRate := by
  by_cases hm : M < 40
  · have key : ∀ M : Fin 40, ∀ s ∈ Accounting.fast100 M.val, s.levels ≠ [] ∧ s.m = M.val ∧
        s.level₀.lanes = s.k0 ∧ s.k0 + s.level₀.logCols + 7 = M.val ∧ 1 ≤ s.level₀.logInvRate := by
      decide +kernel
    exact key ⟨M, hm⟩ s hs
  · rw [Refine.model_fast100_none M (by omega)] at hs; cases hs

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **R's opening**, as `regPhase` runs it after both sparse phases: the two registered claims opened together, then
Ligerito at R's schedule `rsch` on the registered word `w`. -/
def regOpening (A : Arith F K) (sh : SparseShape) (hreg : 7 ≤ sh.mReg) (rsch : Schedule) (c₁ c₂ : RegClaim F sh)
    (w : Oracle F) : Game Bool := do
  match ← opening A hreg [c₁.toClaim, c₂.toClaim] with
  | none => pure false
  | some op => ligerito A rsch op.T op.b w

/-! ## The unique-decoding radius -/

/-- Level `l`'s unique-decoding radius `(1 − ρ)/2 − η`. -/
noncomputable def udRadius (l : Level) : ℝ := (1 - l.rate) / 2 - eta

theorem udRadius_le (l : Level) : udRadius l ≤ l.radius := by
  have hr : 0 ≤ l.rate := by unfold Level.rate; positivity
  have hs := Real.sq_sqrt hr
  unfold udRadius Level.radius
  nlinarith [sq_nonneg (1 - Real.sqrt l.rate)]

theorem udQueryError_eq (l : Level) : udQueryError l = (1 - udRadius l) ^ l.queries := by
  unfold udQueryError udRadius
  congr 1
  ring

/-- **At most one message is within `udRadius` of a table**: two such agree on more than `2^logCols` positions. -/
theorem card_closeL_ud_le (A : Arith F K) (hA : A.Correct) (l : Level) (hf : l.Fits) (lanes : ℕ)
    (C : Oracle F) : (closeL A (udRadius l) lanes l.logCols l.logLen C).card ≤ 1 := by
  classical
  have hagree : ∀ M ∈ closeL A (udRadius l) lanes l.logCols l.logLen C,
      ∃ T : Finset (Fin (2 ^ l.logLen)), (T.card : ℝ) ≥ (1 - udRadius l) * 2 ^ l.logLen ∧
        ∀ q ∈ T, ∀ ℓ, tbl lanes l.logLen C ℓ q = Model.encode A l.logLen M ℓ q := by
    intro M hM
    simp only [closeL, Finset.mem_filter, Finset.mem_univ, true_and] at hM
    unfold Model.Close at hM
    exact ⟨_, hM, fun q hq => by simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hq; exact hq⟩
  refine Finset.card_le_one.2 fun M hM M' hM' => ?_
  obtain ⟨T, hT, hTM⟩ := hagree M hM
  obtain ⟨T', hT', hTM'⟩ := hagree M' hM'
  have hcap : 2 ^ l.logCols < (T ∩ T').card := by
    have hu : ((T ∪ T').card : ℝ) + (T ∩ T').card = T.card + T'.card := by
      exact_mod_cast Finset.card_union_add_card_inter T T'
    have hle : ((T ∪ T').card : ℝ) ≤ 2 ^ l.logLen := by
      have := Finset.card_le_univ (T ∪ T')
      rw [Fintype.card_fin] at this
      exact_mod_cast this
    have hlen : (2 : ℝ) ^ l.logLen * l.rate = 2 ^ l.logCols := by
      unfold Level.rate Level.logLen
      rw [pow_add, inv_pow, mul_assoc, mul_inv_cancel₀ (pow_ne_zero _ two_ne_zero), mul_one]
    have h2 : (0 : ℝ) < 2 ^ l.logLen := by positivity
    have heta : Accounting.eta = 1 / 200 := rfl
    unfold udRadius at hT hT'
    rw [heta] at hT hT'
    have : (2 : ℝ) ^ l.logCols < (T ∩ T').card := by nlinarith
    exact_mod_cast this
  funext ℓ c
  have hp : rowPoly A hA hf.1 (M ℓ) - rowPoly A hA hf.1 (M' ℓ) = 0 := by
    refine Polynomial.eq_zero_of_natDegree_lt_card_of_eval_eq_zero' _
      ((T ∩ T').image fun q => A.omega q.val) (fun ω hω => ?_) ?_
    · obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 hω
      have e := (hTM q (Finset.mem_inter.1 hq).1 ℓ).symm.trans (hTM' q (Finset.mem_inter.1 hq).2 ℓ)
      rw [Polynomial.eval_sub, rowPoly_eval A hA hf.1, rowPoly_eval A hA hf.1]
      exact sub_eq_zero.2 e
    · rw [Finset.card_image_of_injOn fun q _ q' _ hqq' =>
          Fin.ext (hA.omega_injective l.logLen hf.2 q.isLt q'.isLt hqq')]
      exact lt_of_le_of_lt (Polynomial.natDegree_sub_le _ _)
        ((max_lt (rowPoly_natDegree A hA hf.1 _) (rowPoly_natDegree A hA hf.1 _)).trans hcap)
  have hsum : ∑ c', (M ℓ c' - M' ℓ c') • xP A hA l.logCols hf.1 c' = 0 := by
    rw [← hp]
    simp only [rowPoly, ← Finset.sum_sub_distrib, sub_smul, Polynomial.smul_eq_C_mul, ← sub_mul]
  have := Fintype.linearIndependent_iff.1 (xP_linearIndependent A hA l.logCols hf.1) _ hsum c
  exact sub_eq_zero.1 this

/-! ## R's opening -/

/-- **R's opening is sound at unique decoding**: at `fast100`'s schedule for the stacked table, if no message within
level 0's unique-decoding radius of `w` packs a table on which both claims hold, the opening accepts with probability at
most `regOpenError`. -/
theorem regOpen_sound (A : Arith F K) (hA : A.Correct) (sh : SparseShape) (rsch : Schedule)
    (hf : Accounting.fast100 sh.mReg = some rsch) (hreg : 7 ≤ sh.mReg) (c₁ c₂ : RegClaim F sh) (w : Oracle F)
    (hdoom : ∀ M : Message F rsch.k0 rsch.level₀.logCols,
      Model.Close ((1 - rsch.level₀.rate) / 2 - eta) (asTable rsch w) (Model.encode A rsch.level₀.logLen M) →
        ¬ (c₁.toClaim.Holds (witnessOf A sh.mReg rsch.k0 rsch.level₀.logCols M) ∧
          c₂.toClaim.Holds (witnessOf A sh.mReg rsch.k0 rsch.level₀.logCols M))) :
    value (fun b => b = true) (regOpening A sh hreg rsch c₁ c₂ w) ≤ ENNReal.ofReal (regOpenError rsch) := by
  classical
  obtain ⟨-, hm, -, -, -⟩ := fast100_uniqueDecoding sh.mReg rsch hf
  have hshape := fast100_shape sh.mReg rsch hf
  have hfit : rsch.level₀.Fits := (levelGood_spec (levelsChained_head_good hshape.2.2.2)).2.1
  set S₀ := closeL A (udRadius rsch.level₀) rsch.k0 rsch.level₀.logCols rsch.level₀.logLen w with hS₀def
  have hS₀ : S₀.card ≤ 1 := card_closeL_ud_le A hA rsch.level₀ hfit rsch.k0 w
  let S : Statement := ⟨sh.mReg, 6, by omega, le_rfl, ⟨0, Nat.two_pow_pos _⟩, 0, 0, []⟩
  let D : Option (OpenOut F) → Prop := fun o => ¬ ∃ M ∈ S₀, ∃ op : OpenOut F, o = some op ∧
    (∑ y ∈ Finset.range (2 ^ (rsch.m - 7)), flat M y * op.b y) = op.T
  have hopen : value (fun o => ¬ D o) (opening A hreg [c₁.toClaim, c₂.toClaim]) ≤
      ((7 * [c₁.toClaim, c₂.toClaim].length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F := by
    refine le_trans (value_mono (q := fun o => ∃ M ∈ S₀, EM S rsch M o) (fun o ho => ?_) _)
      ((value_exists_finset_le _ _ _).trans ?_)
    · have ho' : ∃ M ∈ S₀, ∃ op : OpenOut F, o = some op ∧
          (∑ y ∈ Finset.range (2 ^ (rsch.m - 7)), flat M y * op.b y) = op.T :=
        Classical.byContradiction ho
      obtain ⟨M, hM, op, hop, h⟩ := ho'
      refine ⟨M, hM, op, hop, ?_⟩
      rw [← h]
      show ∑ y ∈ Finset.range (2 ^ (sh.mReg - 7)), packedAt rsch M y * op.b y = _
      rw [hm]
      rfl
    · calc ∑ M ∈ S₀, value (EM S rsch M) (opening A hreg [c₁.toClaim, c₂.toClaim])
          ≤ ∑ _M ∈ S₀, ((7 * [c₁.toClaim, c₂.toClaim].length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F :=
            Finset.sum_le_sum fun M hM => value_EM_le A hA S rsch hreg [c₁.toClaim, c₂.toClaim] M (by
              have hc : Model.Close ((1 - rsch.level₀.rate) / 2 - eta) (asTable rsch w)
                  (Model.encode A rsch.level₀.logLen M) := by
                rw [hS₀def] at hM
                simp only [closeL, Finset.mem_filter, Finset.mem_univ, true_and] at hM
                exact hM
              have hn := hdoom M hc
              by_contra hall
              push Not at hall
              exact hn ⟨hall _ (by simp), hall _ (by simp)⟩)
        _ ≤ 1 * (((7 * [c₁.toClaim, c₂.toClaim].length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F) := by
            rw [Finset.sum_const, nsmul_eq_mul]
            gcongr
            exact_mod_cast hS₀
        _ = _ := one_mul _
  have hu : (0 : ℝ) ≤ udQueryError rsch.level₀ := by
    unfold udQueryError Level.rate Accounting.eta; positivity
  unfold regOpening
  refine (value_bind_le (opening A hreg [c₁.toClaim, c₂.toClaim]) _ _ D
    (ENNReal.ofReal (1 / qF + udQueryError rsch.level₀ + ligeritoError rsch.levels))
    fun o hD => ?_).trans ?_
  · rcases o with _ | op
    · show value (fun b => b = true) (Game.ret false) ≤ _
      simp
    · have hL : 1 ≤ rsch.level₀.listBound := by
        unfold Level.listBound; have := Nat.one_le_two_pow (n := rsch.level₀.logInvRate); omega
      refine (ligerito_sound_at A hA rsch hshape op.T op.b w (udRadius rsch.level₀) (udRadius_le _) 1 hS₀ hL
        (udQueryError rsch.level₀) hu
        ((udQueryError_eq _).symm.le.trans (le_add_of_nonneg_left (queryError_nonneg _)))
        fun M hM h => hD ⟨M, hM, op, rfl, h⟩).trans_eq ?_
      rw [Nat.cast_one]
  · have hq : (0 : ℝ) < qF := by unfold qF; positivity
    have hr : 0 ≤ ligeritoError rsch.levels :=
      ligeritoError_nonneg _ (levelsChained_ok hshape.2.2.2)
    have hsplit : regOpenError rsch = ((15 : ℕ) : ℝ) / qF +
        (1 / qF + udQueryError rsch.level₀ + ligeritoError rsch.levels) := by
      unfold regOpenError
      rw [show 7 * Flock.RecVStar.STACKED_CLAIMS + 2 = 16 from rfl]
      push_cast
      ring
    have h15 : ((7 * [c₁.toClaim, c₂.toClaim].length + 1 : ℕ) : ℝ≥0∞) / Fintype.card F =
        ENNReal.ofReal (((15 : ℕ) : ℝ) / qF) := by
      rw [ofReal_nat_div_qF', hA.card_F]
      norm_num
    have h1 : (0 : ℝ) ≤ 1 / qF := by positivity
    have h15' : (0 : ℝ) ≤ ((15 : ℕ) : ℝ) / qF := by positivity
    rw [hsplit, ENNReal.ofReal_add h15' (add_nonneg (add_nonneg h1 hu) hr), ← h15]
    exact add_le_add hopen le_rfl

end FlockC3.Fork
