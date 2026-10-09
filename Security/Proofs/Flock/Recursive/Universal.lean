import Proofs.Flock.Recursive.Premises

/-!
# Recursion: the compile premise at a class (gap 4, its first part)

`RecursiveAudit` takes `_hin : Flock.Premises.CompiledForkSound T CT cls regs sch` at the class it ranges over. Gap 4
fixes the class at `UniversalUnit_v1{G, N_IN, N_OUT}`'s, and there the premise is the compile lemma's conclusion:

* `fast100_some`: `fast100 m` is a schedule for every `22 ≤ m ≤ 35`.
* `compiledForkSound_of_fast100`: `CompiledForkSound` at a class whose schedule is `fast100`'s and whose link layout
  holds at every hidden template (`LinkLayout`): `flock_inner_sound_compiled` as stated, at the executable's arithmetic
  (`execArith_correct`) and hm96-sha512. What it leaves at the universal unit's class, `22 ≤ m ≤ 35` and the layout, are
  facts about that class, not hypotheses of the property.
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
  letI : Inhabited Refine.Dig := ⟨Refine.H512 []⟩
  exact flock_inner_sound_compiled T CT execArith execArith_correct Refine.H512 Refine.enc512 Refine.leaf512 cls regs
    sch hsch hlay

end FlockSoundness.Discharge.Recursion
