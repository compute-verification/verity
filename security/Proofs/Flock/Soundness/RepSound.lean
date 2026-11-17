module

public import Definitions.Flock.Accounting.Fast100
public import Definitions.Flock.Defs
public import Definitions.Flock.Session
public import Mathlib.CategoryTheory.Category.Basic
public import Proofs.Flock.Soundness.LigeritoPhase
import Proofs.Flock.Soundness.Accounting.Fast100
import Proofs.Flock.Soundness.Accounting.Numbers
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.LincheckPhase
import Proofs.Flock.Soundness.OpeningPhase

/-! One rep is sound (`rep_sound`): from a doomed start it accepts with probability at most `repError`, the sum of its
four phases' errors. -/

@[expose] public section

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

noncomputable def εZc (s : Schedule) (st : Stmt) : ℝ :=
  (((s.listBound₀ + 1) * (s.m - 13) + s.listBound₀ * 127 + s.listBound₀ * (2 * (s.m - 6))
      + s.listBound₀ * (s.m - st.kLog) : ℕ) : ℝ) / qF

noncomputable def εLc (s : Schedule) (st : Stmt) : ℝ :=
  ((s.listBound₀ * (2 + 2 * (st.kLog - 6) + 63) : ℕ) : ℝ) / qF

noncomputable def εOp (s : Schedule) (st : Stmt) : ℝ :=
  ((s.listBound₀ * (7 * st.nClaims + 1) : ℕ) : ℝ) / qF

noncomputable def εLig (s : Schedule) : ℝ := (s.listBound₀ : ℝ) / qF + ligeritoError s.levels

theorem repError_eq (s : Schedule) (st : Stmt) :
    repError s st = εZc s st + εLc s st + εOp s st + εLig s := by
  unfold repError piopError εZc εLc εOp εLig
  push_cast
  ring

theorem εZc_nonneg (s : Schedule) (st : Stmt) : 0 ≤ εZc s st := by unfold εZc qF; positivity
theorem εLc_nonneg (s : Schedule) (st : Stmt) : 0 ≤ εLc s st := by unfold εLc qF; positivity
theorem εOp_nonneg (s : Schedule) (st : Stmt) : 0 ≤ εOp s st := by unfold εOp qF; positivity
theorem εLig_nonneg (s : Schedule) (hs : ∀ l ∈ s.levels, l.Ok) : 0 ≤ εLig s := by
  unfold εLig qF
  have := ligeritoError_nonneg s.levels hs
  positivity

theorem ofReal_nat_div_qF (N : ℕ) : ENNReal.ofReal ((N : ℝ) / qF) = (N : ℝ≥0∞) / 2 ^ 128 := by
  unfold qF
  rw [ENNReal.ofReal_div_of_pos (by positivity), ENNReal.ofReal_natCast, ENNReal.ofReal_pow (by norm_num),
    ENNReal.ofReal_ofNat]

theorem zerocheck_phase (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m) (mPts : ℕ) (C : Oracle F)
    (extra : List (Claim F S.m)) (hdoom : RepDoomed A S sch C extra) :
    value (fun o => ¬ DZc A S sch C extra o) (zerocheck A S.m) ≤
      ENNReal.ofReal (εZc sch (S.shape mPts)) := by
  classical
  refine (zerocheck_phase_of_card A hA S sch hm C extra hdoom _
    (closeMsgs_card_le A hA sch S.m hsch C)).trans (le_of_eq ?_)
  rw [hA.card_F, εZc, ofReal_nat_div_qF, fast100_m S.m sch hsch]
  simp only [Statement.shape]
  rw [← mul_div_assoc, ENNReal.div_add_div_same]
  congr 1
  have key : S.m - 13 + sch.listBound₀ * (S.m - 13 + 127 + 2 * (S.m - 6) + (S.m - S.kLog)) =
      (sch.listBound₀ + 1) * (S.m - 13) + sch.listBound₀ * 127 +
        sch.listBound₀ * (2 * (S.m - 6)) + sch.listBound₀ * (S.m - S.kLog) := by ring
  · rw [← key]
    push_cast
    ring
  · norm_num

theorem lincheck_phase (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (mPts : ℕ) (C : Oracle F) (extra : List (Claim F S.m))
    (zc : ZcOut F S.m) (hzc : DZc A S sch C extra (some zc)) :
    value (fun o => ¬ DLc A S sch C extra zc o) (lincheck A S zc) ≤
      ENNReal.ofReal (εLc sch (S.shape mPts)) := by
  classical
  refine (lincheck_phase_of_card A hA S sch C extra zc hzc _
    (closeMsgs_card_le A hA sch S.m hsch C)).trans (le_of_eq ?_)
  rw [hA.card_F, εLc, ofReal_nat_div_qF]
  simp only [Statement.shape]
  push_cast
  rw [mul_div_assoc]
  norm_num

theorem opening_phase (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (h7 : 7 ≤ S.m) (mPts : ℕ) (C : Oracle F)
    (extra : List (Claim F S.m)) (hextra : extra.length = 2 * S.regions.length)
    (zc : ZcOut F S.m) (ab : Claim F S.m) (hab : DLc A S sch C extra zc (some ab)) :
    value (fun o => ¬ DOp A S sch C o) (opening A h7 (ab :: cClaim A zc :: extra)) ≤
      ENNReal.ofReal (εOp sch (S.shape mPts)) := by
  classical
  have hdoom : ∀ M ∈ closeMsgs A sch C, ∃ c ∈ ab :: cClaim A zc :: extra,
      ¬ c.Holds (wit A S sch M) := by
    intro M hM
    by_contra hall
    push Not at hall
    exact hab M hM ⟨hall ab (by simp), hall _ (by simp), fun e he => hall e (by simp [he])⟩
  have hlen : (ab :: cClaim A zc :: extra).length = (S.shape mPts).nClaims := by
    simp [Stmt.nClaims, Statement.shape, hextra]; ring
  refine (opening_phase_of_card A hA S sch h7 C _ hdoom _
    (closeMsgs_card_le A hA sch S.m hsch C)).trans (le_of_eq ?_)
  rw [hlen, hA.card_F, εOp, ofReal_nat_div_qF]
  push_cast
  rw [mul_div_assoc]
  norm_num

theorem ligerito_phase (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (C : Oracle F)
    (op : OpenOut F) (hop : DOp A S sch C (some op)) :
    value (fun b => b = true) (ligerito A sch op.T op.b C) ≤ ENNReal.ofReal (εLig sch) := by
  classical
  have hshape := fast100_shape S.m sch hsch
  have hm := fast100_m S.m sch hsch
  refine (ligerito_sound A hA sch hshape op.T op.b C fun M hM => ?_).trans (le_of_eq ?_)
  · have hmem : M ∈ closeMsgs A sch C := by
      simp only [closeL, Finset.mem_filter, Finset.mem_univ, true_and] at hM
      exact hM
    have := hop M hmem
    rw [hm]
    exact this
  · have hL : sch.listBound₀ = sch.level₀.listBound := by
      obtain ⟨hlen, -⟩ := hshape
      cases h : sch.levels with
      | nil => rw [h] at hlen; simp at hlen
      | cons l rest => simp [Schedule.listBound₀, Schedule.level₀, h]
    rw [εLig, hL]

theorem rep_sound (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m)
    (mPts : ℕ) (C : Oracle F) (extra : List (Claim F S.m))
    (hextra : extra.length = 2 * S.regions.length) (hdoom : RepDoomed A S sch C extra) :
    value (fun b => b = true) (rep A S sch hm C extra) ≤
      ENNReal.ofReal (repError sch (S.shape mPts)) := by
  have hok := fast100_levels_ok S.m sch hsch
  have hZc := εZc_nonneg sch (S.shape mPts)
  have hLc := εLc_nonneg sch (S.shape mPts)
  have hOp := εOp_nonneg sch (S.shape mPts)
  have hLig := εLig_nonneg sch hok
  rw [repError_eq, ENNReal.ofReal_add (by positivity) hLig, ENNReal.ofReal_add (by positivity) hOp,
    ENNReal.ofReal_add hZc hLc]
  unfold rep
  simp only [bind_eq]
  refine (value_bind_le _ _ _ (DZc A S sch C extra)
    (ENNReal.ofReal (εLc sch (S.shape mPts)) + ENNReal.ofReal (εOp sch (S.shape mPts))
      + ENNReal.ofReal (εLig sch)) (fun o ho => ?_)).trans ?_
  · cases o with
    | none => simp
    | some zc =>
      simp only
      refine (value_bind_le _ _ _ (DLc A S sch C extra zc)
        (ENNReal.ofReal (εOp sch (S.shape mPts)) + ENNReal.ofReal (εLig sch))
        (fun o₂ ho₂ => ?_)).trans ?_
      · cases o₂ with
        | none => simp
        | some ab =>
          simp only
          refine (value_bind_le _ _ _ (DOp A S sch C) (ENNReal.ofReal (εLig sch))
            (fun o₃ ho₃ => ?_)).trans ?_
          · cases o₃ with
            | none => simp
            | some op => exact ligerito_phase A hA S sch hsch C op ho₃
          · exact add_le_add (opening_phase A hA S sch hsch _ mPts C extra hextra zc ab ho₂) le_rfl
      · rw [← add_assoc]
        exact add_le_add (add_le_add (lincheck_phase A hA S sch hsch mPts C extra zc ho) le_rfl)
          le_rfl
  · calc value (fun o => ¬ DZc A S sch C extra o) (zerocheck A S.m)
          + (ENNReal.ofReal (εLc sch (S.shape mPts)) + ENNReal.ofReal (εOp sch (S.shape mPts))
            + ENNReal.ofReal (εLig sch))
        ≤ ENNReal.ofReal (εZc sch (S.shape mPts))
          + (ENNReal.ofReal (εLc sch (S.shape mPts)) + ENNReal.ofReal (εOp sch (S.shape mPts))
            + ENNReal.ofReal (εLig sch)) :=
          add_le_add (zerocheck_phase A hA S sch hsch hm mPts C extra hdoom) le_rfl
      _ = _ := by ring

end FlockSoundness
