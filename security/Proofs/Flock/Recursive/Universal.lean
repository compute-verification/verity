module

public import Proofs.Flock.Recursive.Premises
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Refine.Setup

/-! The compile premise `CompiledForkSound` at an executable statement's class, from the compile lemma at `fast100`'s schedules. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

theorem fast100_some {m : ℕ} (h1 : 22 ≤ m) (h2 : m ≤ 35) : ∃ sch, Accounting.fast100 m = some sch := by
  interval_cases m <;> exact ⟨_, rfl⟩

theorem compiledForkSound_of_fast100 {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (cls : HClass) (regs : List (FlockSoundness.Model.Region cls.kLog (cls.m - cls.kLog)))
    (sch : Accounting.Schedule) (hsch : Accounting.fast100 cls.m = some sch)
    (hlay : ∀ M : Hidden cls, (M.toStatement regs).LinkLayout cls.ptLocal cls.mPts) :
    Flock.Premises.CompiledForkSound T CT cls regs sch := by
  let _ : Inhabited Refine.Dig := ⟨Refine.H512 []⟩
  exact flock_inner_sound_compiled T CT execArith execArith_correct Refine.H512 Refine.enc512 Refine.leafPlain cls regs
    sch hsch hlay

/-- **The class of an executable statement**: its `m`, `k_log` and pin, its matrices (`Refine.matA`, `Refine.matB`, as
`Refine.stmtOf` reads them) as the public part, a hidden template of `nnz` entries at the unit slots `units` of
`2^slotLog`, and the link layout at `PT_LOCAL` and the statement's link points. -/
def HClass.ofStmt (st : Flock.Stmt) (hwf : Refine.StmtWF st) (hm : 13 ≤ st.m) (slotLog : ℕ) (units : Finset ℕ)
    (nnz : ℕ) : HClass where
  m := st.m
  kLog := st.c.kLog
  hk := hwf.hk
  hk6 := hwf.hk6
  hm := hm
  pin := ⟨st.pin, hwf.pin⟩
  A₀pub := fun i c => Refine.matA st i.val c.val
  B₀pub := fun i c => Refine.matB st i.val c.val
  slotLog := slotLog
  units := units
  ptLocal := Flock.PT_LOCAL
  mPts := st.mPts
  nnz := nnz

theorem compiledForkSound_ofStmt {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hwf : Refine.StmtWF st) (hr : Refine.RegionsWF st) (hm : 13 ≤ st.m)
    (s₀ : Flock.Schedule) (hs₀ : Flock.fast100 st.m = some s₀) (slotLog : ℕ) (units : Finset ℕ) (nnz : ℕ) :
    Flock.Premises.CompiledForkSound T CT (HClass.ofStmt st hwf hm slotLog units nnz) (Refine.regionsOf st)
      (Refine.schedOf s₀) := by
  have h := Refine.fast100_schedOf st.m
  rw [hs₀] at h
  exact compiledForkSound_of_fast100 T CT (HClass.ofStmt st hwf hm slotLog units nnz) (Refine.regionsOf st)
    (Refine.schedOf s₀) h.symm fun _ => Refine.stmtOf_linkLayout st hwf hr

theorem hm_of_schedule {st : Flock.Stmt} {p : Flock.Schedule × ℕ} (h : Flock.Zk.schedule st.wideLeaf st.m = .ok p) :
    13 ≤ st.m := by
  have := (ZkHidden.schedule_range h).1
  omega

theorem compiledForkSound_ofSchedule {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hwf : Refine.StmtWF st) (hr : Refine.RegionsWF st)
    {p : Flock.Schedule × ℕ} (h : Flock.Zk.schedule st.wideLeaf st.m = .ok p) (slotLog : ℕ) (units : Finset ℕ) (nnz : ℕ) :
    ∃ sch, Flock.Premises.CompiledForkSound T CT (HClass.ofStmt st hwf (hm_of_schedule h) slotLog units nnz)
      (Refine.regionsOf st) sch := by
  obtain ⟨s₀, hs₀⟩ := ZkHidden.fast100_of_schedule h
  exact ⟨_, compiledForkSound_ofStmt T CT st hwf hr (hm_of_schedule h) s₀ hs₀ slotLog units nnz⟩

end FlockSoundness.Discharge.Recursion
