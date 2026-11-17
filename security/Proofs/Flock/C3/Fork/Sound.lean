module

public import Proofs.Flock.C3.Fork.Runs
public import Proofs.Flock.Recursive.HoloExec
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
import Proofs.Flock.Recursive.ForkRuns
import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Refine.Setup

/-! The executable's parameters (`holoAt st hp`) meet `holo_runs_sound`'s hypotheses. -/

@[expose] public section

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

theorem holoAt_fast100 {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st)) :
    Accounting.fast100 (holoAt st hp).cls.m = some (holoAt st hp).sch := by
  obtain ⟨-, -, _, hs⟩ := ClassOk.of_shape h
  obtain ⟨s₀, hs₀⟩ := ZkHidden.fast100_of_schedule hs
  have hsch : Accounting.fast100 st.m = some (schedAt st.m) := by
    have e := Refine.fast100_schedOf st.m
    rw [hs₀] at e
    unfold schedAt
    rw [hs₀]
    exact e.symm
  have e : execClassAt st = ExecClass.ofStmt st h := dif_pos h
  show Accounting.fast100 (execClassAt st).cls.m = some (execClassAt st).sch
  rw [e]
  exact hsch

theorem holoAt_linkLayout {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st)) :
    ∀ R : RegTable (holoAt st hp).cls,
      (R.toStatement (holoAt st hp).τ (holoAt st hp).regs).LinkLayout (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts := by
  obtain ⟨hwf, hr, -⟩ := ClassOk.of_shape h
  have hl : ∀ ec : ExecClass, ec = ExecClass.ofStmt st h →
      (∀ R ∈ ec.regs, R.free.Nodup ∧ R.free.length ≤ ec.cls.ptLocal) ∧
        ec.cls.ptLocal + (ec.cls.m - ec.cls.kLog) ≤ ec.cls.mPts := by
    rintro _ rfl
    exact Refine.stmtOf_linkLayout st hwf hr
  have e : execClassAt st = ExecClass.ofStmt st h := dif_pos h
  exact fun _ => hl _ e

theorem holoAt_rsch (st : Flock.Stmt) (hp : HoloParams) :
    (holoAt st hp).rsch.sch.levels = [] ∨
      Accounting.fast100 (holoAt st hp).cls.shape.mReg = some (holoAt st hp).rsch.sch := by
  have key : ∀ (M : ℕ) (s : FlockSoundness.Accounting.Schedule), (s = default ∨ s = schedAt M) →
      s.levels = [] ∨ Accounting.fast100 M = some s := by
    rintro M s (rfl | rfl)
    · exact Or.inl rfl
    · obtain h | ⟨s₀, h⟩ : Flock.fast100 M = none ∨ ∃ s₀, Flock.fast100 M = some s₀ := by
        cases Flock.fast100 M <;> simp
      · unfold schedAt
        rw [h]
        exact Or.inl rfl
      · have e := Refine.fast100_schedOf M
        rw [h] at e
        unfold schedAt
        rw [h]
        exact Or.inr e.symm
  refine key (holoAt st hp).cls.shape.mReg (holoAt st hp).rsch.sch ?_
  have e : (holoAt st hp).rsch.sch =
      if hp.EllOk (execClassAt st).cls.kLog then schedAt (holoAt st hp).cls.shape.mReg else default := rfl
  by_cases hl : hp.EllOk (execClassAt st).cls.kLog
  · rw [e, if_pos hl]
    exact Or.inr rfl
  · rw [e, if_neg hl]
    exact Or.inl rfl

theorem holoAt_units (st : Flock.Stmt) (hp : HoloParams) :
    ∀ σ, ∀ u ∈ (holoAt st hp).cls.units σ, u < 2 ^ ((holoAt st hp).cls.kLog - (holoAt st hp).cls.slotLog) := by
  intro σ u hu
  have hu' : u ∈ (hp.units σ).filter fun u =>
      u < 2 ^ ((execClassAt st).cls.kLog - (if hp.EllOk (execClassAt st).cls.kLog then hp.ell else 6)) := hu
  exact (Finset.mem_filter.1 hu').2

theorem holoAtτ_fast100 {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st))
    (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) :
    Accounting.fast100 (holoAtτ st hp τv).cls.m = some (holoAtτ st hp τv).sch :=
  holoAt_fast100 hp h

theorem holoAtτ_linkLayout {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st))
    (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) :
    ∀ R : RegTable (holoAtτ st hp τv).cls,
      (R.toStatement (holoAtτ st hp τv).τ (holoAtτ st hp τv).regs).LinkLayout (holoAtτ st hp τv).cls.ptLocal
        (holoAtτ st hp τv).cls.mPts := by
  obtain ⟨hwf, hr, -⟩ := ClassOk.of_shape h
  have hl : ∀ ec : ExecClass, ec = ExecClass.ofStmt st h →
      (∀ R ∈ ec.regs, R.free.Nodup ∧ R.free.length ≤ ec.cls.ptLocal) ∧
        ec.cls.ptLocal + (ec.cls.m - ec.cls.kLog) ≤ ec.cls.mPts := by
    rintro _ rfl
    exact Refine.stmtOf_linkLayout st hwf hr
  have e : execClassAt st = ExecClass.ofStmt st h := dif_pos h
  exact fun _ => hl _ e

theorem holoAtτ_rsch (st : Flock.Stmt) (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) :
    (holoAtτ st hp τv).rsch.sch.levels = [] ∨
      Accounting.fast100 (holoAtτ st hp τv).cls.shape.mReg = some (holoAtτ st hp τv).rsch.sch :=
  holoAt_rsch st hp

theorem holoAtτ_units (st : Flock.Stmt) (hp : HoloParams) (τv : Fin (2 ^ hp.sLog) → Fin (2 ^ hp.tLog)) :
    ∀ σ, ∀ u ∈ (holoAtτ st hp τv).cls.units σ,
      u < 2 ^ ((holoAtτ st hp τv).cls.kLog - (holoAtτ st hp τv).cls.slotLog) :=
  holoAt_units st hp

end FlockC3.Fork
