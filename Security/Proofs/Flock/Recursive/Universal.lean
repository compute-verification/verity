import Proofs.Flock.Recursive.Premises
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched

/-!
# Recursion: the compile premise at a class (gap 4, its first part)

`RecursiveAudit` took `_hin : Flock.Premises.CompiledForkSound T CT cls regs sch` at the class it ranged over; it now
takes the statement, and `compiledForkSound_execClassAt` (`ExecClass`) discharges the premise at its class. Gap 4
fixes the class at `UniversalUnit_v1{G, N_IN, N_OUT}`'s, and there the premise is the compile lemma's conclusion:

* `fast100_some`: `fast100 m` is a schedule for every `22 ≤ m ≤ 35`.
* `compiledForkSound_of_fast100`: `CompiledForkSound` at a class whose schedule is `fast100`'s and whose link layout
  holds at every hidden template (`LinkLayout`): `flock_inner_sound_compiled` as stated, at the executable's arithmetic
  (`execArith_correct`) and the inner statement's plain SHA-512 leaves (`Refine.leafPlain`).
* `HClass.ofStmt`: the class of an executable statement: its `m`, `k_log`, pin and matrices (`Refine.stmtOf`'s) as the
  public part, with a hidden template's geometry (`slotLog`, `units`, `nnz`) given, the link layout at `PT_LOCAL` and the
  statement's link points.
* `compiledForkSound_ofStmt`, `compiledForkSound_ofSchedule`: `CompiledForkSound` there, at the statement's regions
  (`Refine.regionsOf`) and its `fast100` schedule. `LinkLayout` reads only the regions and the block bits, so it is
  `Refine.stmtOf_linkLayout`'s at every hidden template. What is left are the executable's own checks of the statement:
  `StmtWF` and `RegionsWF` (what setupH's or setupHidden's acceptance gives) and `Zk.schedule st.m` accepting.
-/

namespace FlockSoundness.Discharge.Recursion

/-- `fast100` has a schedule at every `m` from 22 to 35. -/
theorem fast100_some {m : ℕ} (h1 : 22 ≤ m) (h2 : m ≤ 35) : ∃ sch, Accounting.fast100 m = some sch := by
  interval_cases m <;> exact ⟨_, rfl⟩

/-- **The compile premise at a class whose schedule is `fast100`'s and whose layout holds**, from
`flock_inner_sound_compiled`. -/
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

/-- **The compile premise at an executable statement's class**, its regions and its `fast100` schedule. -/
theorem compiledForkSound_ofStmt {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hwf : Refine.StmtWF st) (hr : Refine.RegionsWF st) (hm : 13 ≤ st.m)
    (s₀ : Flock.Schedule) (hs₀ : Flock.fast100 st.m = some s₀) (slotLog : ℕ) (units : Finset ℕ) (nnz : ℕ) :
    Flock.Premises.CompiledForkSound T CT (HClass.ofStmt st hwf hm slotLog units nnz) (Refine.regionsOf st)
      (Refine.schedOf s₀) := by
  have h := Refine.fast100_schedOf st.m
  rw [hs₀] at h
  exact compiledForkSound_of_fast100 T CT (HClass.ofStmt st hwf hm slotLog units nnz) (Refine.regionsOf st)
    (Refine.schedOf s₀) h.symm fun _ => Refine.stmtOf_linkLayout st hwf hr

/-- `Zk.schedule` accepts only an `m` of at least 13 (`HClass.hm`). -/
theorem hm_of_schedule {st : Flock.Stmt} {p : Flock.Schedule × ℕ} (h : Flock.Zk.schedule st.m = .ok p) :
    13 ≤ st.m := by
  have := (ZkHidden.schedule_range h).1
  omega

/-- **The compile premise at an executable statement's class, from the verifier's own checks**: the statement's
`StmtWF` and `RegionsWF`, and `Zk.schedule` accepting its `m`. -/
theorem compiledForkSound_ofSchedule {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)]
    [∀ k, Nonempty (CT k)] (st : Flock.Stmt) (hwf : Refine.StmtWF st) (hr : Refine.RegionsWF st)
    {p : Flock.Schedule × ℕ} (h : Flock.Zk.schedule st.m = .ok p) (slotLog : ℕ) (units : Finset ℕ) (nnz : ℕ) :
    ∃ sch, Flock.Premises.CompiledForkSound T CT (HClass.ofStmt st hwf (hm_of_schedule h) slotLog units nnz)
      (Refine.regionsOf st) sch := by
  obtain ⟨s₀, hs₀⟩ := ZkHidden.fast100_of_schedule h
  exact ⟨_, compiledForkSound_ofStmt T CT st hwf hr (hm_of_schedule h) s₀ hs₀ slotLog units nnz⟩

end FlockSoundness.Discharge.Recursion
