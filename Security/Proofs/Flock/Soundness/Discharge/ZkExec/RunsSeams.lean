import Proofs.Flock.Soundness.Discharge.ZkExec.RunsStatements

/-!
# The `--zk` schedule's dimensions

`zk_schedule_seams`: `Zk.schedule m` gives fast100's dimensions with level 0 restratified at `q_0` queries. The kernel
checks every `m < 40`; above 35 `Zk.queries0` refuses.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine

theorem zk_schedule_seams : ZkScheduleSeams := by
  intro m sch tPad h
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ x ∈ (Zk.schedule m.val).toOption,
        (decide (25 ≤ m.val ∧ m.val ≤ 35 ∧ x.1.m = m.val ∧ 1 ≤ x.1.k0 ∧ x.2 = 2 * (x.1.levels.getD 0 default).q ∧
            DimSeam x.1) &&
          (List.range x.1.levels.size).all (fun i => decide (LevelSeam (x.1.levels.getD i default) ∧
            Accounting.ceilLog2 (x.1.levels.getD i default).q ≠ 1)) &&
          x.1.levels.all fun lv => decide (lv.cols < 64 ∧ 0 < lv.q ∧ lv.q < 512 ∧ lv.cap ≤ 8)) = true := by
      decide +kernel
    have k := key ⟨m, hm⟩ (sch, tPad) (by rw [h]; rfl)
    rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_iff] at k
    obtain ⟨⟨⟨h25, h35, hsm, hk0, ht, hdim⟩, hL⟩, hS⟩ := k
    have hL' := List.all_eq_true.1 hL
    refine ⟨h25, h35, hsm, hk0, ht, hdim, fun i hi => ?_, fun i hi => ?_, fun lv hlv => ?_⟩
    · exact (of_decide_eq_true (hL' i (List.mem_range.2 hi))).1
    · exact (of_decide_eq_true (hL' i (List.mem_range.2 hi))).2
    · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hlv
      simpa using Array.all_eq_true.1 hS i hi
  · exfalso
    have hq : Zk.queries0 m = throw s!"m = {m}: no fast100 schedule" := by
      unfold Zk.queries0
      rw [if_neg (by omega), if_neg (by simp only [beq_iff_eq]; omega), if_neg (by simp only [beq_iff_eq]; omega),
        if_neg (by simp only [beq_iff_eq]; omega), if_neg (by omega), if_neg (by omega)]
    rw [Zk.schedule, hq] at h
    cases h

end FlockSoundness.Discharge.ZkExec
