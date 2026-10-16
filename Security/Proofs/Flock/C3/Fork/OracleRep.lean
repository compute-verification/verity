import Proofs.Flock.C3.Fork.OracleFold
import Proofs.Flock.C3.Fork.OracleGame
import Proofs.Flock.Soundness.RepSound

/-!
# Piece (a), the oracle count: one rep with its sparse phase

At a registered table `R`, a rep with hidden units (`holoRep` on the public statement) whose prover sends the hidden
fold `h = hiddenFold` is `rep` on the hidden statement `R.toStatement τ regs`: the zerocheck, the opening and Ligerito
are the same games, and `holoLincheck` then checks what `lincheck` checks (`holoFold_eq`), so `holoLincheck_le`
simulates `lincheck` and `rep_sound`'s phases bound the rep (`holoRep_honest_le`, `repError`). A prover sending another
`h` is caught by the rep's sparse phase (`sparse_sound`), unless its registered claim fails on `R`'s cells; so a rep
followed by its sparse phase hands on a claim true on `R` with probability at most `repError + sparseError`
(`holoRep_stage_le`).
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge.Recursion
open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **One rep's sparse phase** on the rep's outputs, as `regPhase` runs it; no claim where the rep rejected. -/
def stage (A : Arith F K) (S : Statement) (sh : SparseShape) (units : Fin (2 ^ sh.sLog) → Finset ℕ)
    (τ : Fin (2 ^ sh.sLog) → ℕ) :
    Option (ZcOut F S.m × HoloLcOut F S.m (S.kLog - 6)) → Game (Option (RegClaim F sh))
  | none => pure none
  | some (zc, lc) => sparsePhase A sh (holoPub S sh units zc lc) τ lc.h

variable (A : Arith F K) (hA : A.Correct) (cls : HoloClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
  (τ : Fin (2 ^ cls.sLog) → Fin (2 ^ cls.tLog)) (R : RegTable cls)
  (hunits : ∀ σ, ∀ u ∈ cls.units σ, u < 2 ^ (cls.kLog - cls.slotLog))

include hA hunits in
/-- **`holoLincheck` with the hidden fold is `lincheck` on the hidden statement**: it ends with the hidden fold as `h`
and an `ab` claim in `Q` no more often than `lincheck` on `R.toStatement τ regs` ends in `Q`. -/
theorem holoLincheck_le (zc : ZcOut F cls.m) (Q : Option (Claim F cls.m) → Prop) :
    value (fun o => ∃ lc, o = some lc ∧ lc.h = hiddenFold A cls regs τ R zc lc.alpha lc.ts ∧ Q (some lc.ab))
      (holoLincheck A (cls.pubStatement regs) zc) ≤ value Q (lincheck A (R.toStatement τ regs) zc) := by
  unfold holoLincheck lincheck
  simp only [bind_eq, draw, recv, bind_coin, bind_send, bind_ret, pure_eq]
  refine eq_le_eq (value_coin _ _ _) (value_coin _ _ _) (Finset.sum_le_sum fun α _ => mul_le_mul_right ?_ _)
  refine eq_le_eq (value_coin _ _ _) (value_coin _ _ _) (Finset.sum_le_sum fun β _ => mul_le_mul_right ?_ _)
  refine Discharge.ZkSession.value_bind_mono _ fun tc => ?_
  refine eq_le_eq (value_send _ _ _) (value_send _ _ _) (iSup_mono fun zP => ?_)
  refine (value_send _ _ _).trans_le (iSup_le fun h => ?_)
  refine eq_le_eq (value_coin _ _ _) (value_coin _ _ _) (Finset.sum_le_sum fun r' _ => mul_le_mul_right ?_ _)
  refine value_ret_le fun ⟨lc, hlc, hh, hQ⟩ => ?_
  obtain ⟨hchk, rfl⟩ := of_ite_some hlc
  refine ite_some_of ?_ hQ
  refine Eq.trans (Finset.sum_congr rfl fun s _ => ?_) hchk
  exact congrArg (· * zP s) ((holoFold_eq A hA cls regs τ R hunits zc α β tc.1 s).trans
    (congrArg (_ + ·) (congrFun hh s).symm))

include hA hunits in
/-- **A rep whose prover sends the hidden fold is a rep on the hidden statement**: from a doomed start on
`R.toStatement τ regs`, it accepts with the hidden fold as `h` with probability at most `repError`. -/
theorem holoRep_honest_le (sch : Schedule) (hsch : fast100 cls.m = some sch) (mPts : ℕ) (C₀ : Oracle F)
    (extra : List (Claim F cls.m)) (hextra : extra.length = 2 * regs.length)
    (hdoom : RepDoomed A (R.toStatement τ regs) sch C₀ extra) :
    value (fun o => ∃ zc lc, o = some (zc, lc) ∧ lc.h = hiddenFold A cls regs τ R zc lc.alpha lc.ts)
      (holoRep A (cls.pubStatement regs) sch cls.hm C₀ extra) ≤
      ENNReal.ofReal (repError sch ((R.toStatement τ regs).shape mPts)) := by
  classical
  set S' := R.toStatement τ regs
  have hok := fast100_levels_ok cls.m sch hsch
  have hZc := εZc_nonneg sch (S'.shape mPts)
  have hLc := εLc_nonneg sch (S'.shape mPts)
  have hOp := εOp_nonneg sch (S'.shape mPts)
  have hLig := εLig_nonneg sch hok
  rw [repError_eq, ENNReal.ofReal_add (by positivity) hLig, ENNReal.ofReal_add (by positivity) hOp,
    ENNReal.ofReal_add hZc hLc]
  unfold holoRep
  simp only [bind_eq]
  refine (value_bind_le _ _ _ (fun o : Option (ZcOut F cls.m) => DZc A S' sch C₀ extra o)
    (ENNReal.ofReal (εLc sch (S'.shape mPts)) + ENNReal.ofReal (εOp sch (S'.shape mPts))
      + ENNReal.ofReal (εLig sch)) (fun o ho => ?_)).trans ?_
  · cases o with
    | none => exact le_of_eq_of_le (value_ret_of_not fun ⟨_, _, h, _⟩ => by cases h) zero_le
    | some zc =>
      simp only
      refine (value_bind_le _ _ _
        (fun o : Option (HoloLcOut F cls.m (cls.kLog - 6)) => ∀ lc, o = some lc →
          lc.h = hiddenFold A cls regs τ R zc lc.alpha lc.ts → DLc A S' sch C₀ extra zc (some lc.ab))
        (ENNReal.ofReal (εOp sch (S'.shape mPts)) + ENNReal.ofReal (εLig sch))
        (fun o₂ ho₂ => ?_)).trans ?_
      · cases o₂ with
        | none => exact le_of_eq_of_le (value_ret_of_not fun ⟨_, _, h, _⟩ => by cases h) zero_le
        | some lc =>
          simp only
          by_cases hh : lc.h = hiddenFold A cls regs τ R zc lc.alpha lc.ts
          · refine (value_bind_le _ _ _ (DOp A S' sch C₀) (ENNReal.ofReal (εLig sch))
              (fun o₃ ho₃ => ?_)).trans ?_
            · cases o₃ with
              | none => exact le_of_eq_of_le (value_ret_of_not fun ⟨_, _, h, _⟩ => by cases h) zero_le
              | some op =>
                refine le_of_eq_of_le (value_bind_ret _ _ _) (le_trans (value_mono (q := fun b => b = true)
                  (fun b hb => ?_) _) (ligerito_phase A hA S' sch hsch C₀ op ho₃))
                cases b
                · obtain ⟨_, _, h, _⟩ := hb
                  cases h
                · rfl
            · exact add_le_add (opening_phase A hA S' sch hsch _ mPts C₀ extra hextra zc lc.ab
                (ho₂ lc rfl hh)) le_rfl
          · refine (value_bind_le _ _ _ (fun _ => True) 0 (fun o₃ _ => ?_)).trans ?_
            · cases o₃ with
              | none => exact (value_ret_of_not fun ⟨_, _, h, _⟩ => by cases h).le
              | some op =>
                refine ((value_bind_ret _ _ _).trans ((value_congr (q := fun _ => False) (fun b => ?_) _).trans
                  (value_false _))).le
                refine iff_of_false (fun ⟨_, _, h, h'⟩ => ?_) id
                cases b
                · cases h
                · cases h
                  exact hh h'
            · simp [value_false]
      · rw [← add_assoc]
        refine add_le_add (add_le_add ?_ le_rfl) le_rfl
        refine le_trans (value_mono (fun o ho₂ => ?_) _)
          ((holoLincheck_le A hA cls regs τ R hunits zc (fun o => ¬ DLc A S' sch C₀ extra zc o)).trans
            (lincheck_phase A hA S' sch hsch mPts C₀ extra zc ho))
        push Not at ho₂
        exact ho₂
  · calc value (fun o => ¬ DZc A S' sch C₀ extra o) (zerocheck A cls.m)
          + (ENNReal.ofReal (εLc sch (S'.shape mPts)) + ENNReal.ofReal (εOp sch (S'.shape mPts))
            + ENNReal.ofReal (εLig sch))
        ≤ ENNReal.ofReal (εZc sch (S'.shape mPts))
          + (ENNReal.ofReal (εLc sch (S'.shape mPts)) + ENNReal.ofReal (εOp sch (S'.shape mPts))
            + ENNReal.ofReal (εLig sch)) :=
          add_le_add (zerocheck_phase A hA S' sch hsch cls.hm mPts C₀ extra hdoom) le_rfl
      _ = _ := by ring

include hA hunits in
/-- **One rep and its sparse phase**: from a doomed start on `R.toStatement τ regs`, the rep's sparse phase hands on a
registered claim that holds on `R`'s cells with probability at most `repError + sparseError`. -/
theorem holoRep_stage_le (sch : Schedule) (hsch : fast100 cls.m = some sch) (mPts : ℕ) (C₀ : Oracle F)
    (extra : List (Claim F cls.m)) (hextra : extra.length = 2 * regs.length)
    (hdoom : RepDoomed A (R.toStatement τ regs) sch C₀ extra) :
    value (fun o => ∃ c, o = some c ∧ c.toClaim.Holds R.bits)
      ((holoRep A (cls.pubStatement regs) sch cls.hm C₀ extra).bind
        (stage A (cls.pubStatement regs) cls.shape cls.units fun σ => (τ σ).val)) ≤
      ENNReal.ofReal (repError sch ((R.toStatement τ regs).shape mPts) + sparseError cls.shape) := by
  classical
  have hsp : 0 ≤ sparseError cls.shape := by unfold sparseError qF; positivity
  rw [ENNReal.ofReal_add (repError_nonneg _ _ (fast100_levels_ok cls.m sch hsch)) hsp]
  refine (value_bind_le _ _ _ (fun o : Option (ZcOut F cls.m × HoloLcOut F cls.m (cls.kLog - 6)) =>
    ∀ zc lc, o = some (zc, lc) → lc.h ≠ hiddenFold A cls regs τ R zc lc.alpha lc.ts)
    (ENNReal.ofReal (sparseError cls.shape)) (fun o ho => ?_)).trans (add_le_add ?_ le_rfl)
  · rcases o with _ | ⟨zc, lc⟩
    · exact le_of_eq_of_le (value_ret_of_not fun ⟨_, h, _⟩ => by cases h) zero_le
    · exact sparse_sound A hA cls.shape _ _ (fun σ => (τ σ).isLt) R.bits lc.h (ho zc lc rfl)
  · refine le_trans (value_mono (fun o ho => ?_) _)
      (holoRep_honest_le A hA cls regs τ R hunits sch hsch mPts C₀ extra hextra hdoom)
    push Not at ho
    exact ho

end FlockC3.Fork
