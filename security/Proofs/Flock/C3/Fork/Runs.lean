module

public import Proofs.Flock.C3.Fork.Coupling
public import Proofs.Flock.C3.Fork.OracleTable
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
import Proofs.Flock.Recursive.CompileRuns
import Proofs.Flock.Soundness.Game

/-! Candidate 3's table on one prover's runs (`holo_runs_sound`), reading no registrant's openings. -/

@[expose] public section

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

theorem holo_runs_sound (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo)
    (hsch : Accounting.fast100 e.cls.m = some e.sch)
    (hlay : ∀ R : RegTable e.cls, (R.toStatement e.τ e.regs).LinkLayout e.cls.ptLocal e.cls.mPts)
    (hR : e.rsch.sch.levels = [] ∨ Accounting.fast100 e.cls.shape.mReg = some e.rsch.sch)
    (hunits : ∀ σ, ∀ u ∈ e.cls.units σ, u < 2 ^ (e.cls.kLog - e.cls.slotLog))
    (x : HoloReg F D) (hno : ¬ (holoInnerCL T CT A H E Lf e).holds x)
    (G : List (List (SMsg T) × ((k : Fin r) → CT k)) → Prop) (s : Strategy (holoInnerCL T CT A H E Lf e).game) :
    (∃ t t' o o', Game.Outcome (holoInnerCL T CT A H E Lf e).game s t ∧
      Game.Outcome (holoInnerCL T CT A H E Lf e).game s t' ∧ G t ∧ G t' ∧
      PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t) o ∧
      PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t') o' ∧
      ClashHL H E Lf e.sch e.rsch.sch x o o') ∨
    prob (fun t => (holoInnerCL T CT A H E Lf e).accepts x t ∧ G t) (holoInnerCL T CT A H E Lf e).game s ≤
      ENNReal.ofReal (holoError e) := by
  by_cases hc : ∃ t t' o o', Game.Outcome (holoInnerCL T CT A H E Lf e).game s t ∧
      Game.Outcome (holoInnerCL T CT A H E Lf e).game s t' ∧ G t ∧ G t' ∧
      PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t) o ∧
      PlaysT (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) (schedSteps T CT 0 t') o' ∧
      ClashHL H E Lf e.sch e.rsch.sch x o o'
  · exact Or.inl hc
  right
  let g := holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
    e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap
  let W : OutHL F D Sl → Prop := fun o =>
    ∃ t, Game.Outcome (holoInnerCL T CT A H E Lf e).game s t ∧ G t ∧ PlaysT g (schedSteps T CT 0 t) o
  have hW : ∀ o o', W o → W o' → ¬ ClashHL H E Lf e.sch e.rsch.sch x o o' := by
    rintro o o' ⟨t, ht, hG, ho⟩ ⟨t', ht', hG', ho'⟩ hcl
    exact hc ⟨t, t', o, o', ht, ht', hG, hG', ho, ho', hcl⟩
  have horc : ∀ σ : Strategy (holoTableOf A e.cls e.regs e.sch e.rsch e.τ x.w),
      prob (fun out => out.1 = true) (holoTableOf A e.cls e.regs e.sch e.rsch e.τ x.w) σ ≤
        ENNReal.ofReal (holoError e) := by
    intro σ
    refine le_trans (prob_mono (q := fun out : Bool × Oracle F => out.1 = true ∧ ¬ ∃ R : RegTable e.cls,
      e.rsch.Close A x.w R ∧ Committed A (R.toStatement e.τ e.regs) e.sch out.2) ?_ _ σ)
      (holoTable_sound A hA e hsch hlay hR hunits x.w σ)
    rintro out h
    refine ⟨h, ?_⟩
    rintro ⟨R, hcl, hcm⟩
    exact hno (show ∃ R : RegTable e.cls, e.rsch.Close A x.w R ∧ ∃ z, (R.toStatement e.τ e.regs).Satisfies z from
      ⟨R, hcl, satisfiable_of_committed hcm⟩)
  have hval := holoTable_value_le_oracle A H E Lf e hsch hR x (holoError e) horc W hW
  calc prob (fun t => (holoInnerCL T CT A H E Lf e).accepts x t ∧ G t) (holoInnerCL T CT A H E Lf e).game s
      ≤ prob (fun t => ∃ a, PlaysT g (schedSteps T CT 0 t) a ∧ (a.accepted = true ∧ W a))
          (holoInnerCL T CT A H E Lf e).game s := by
        refine prob_mono_outcome _ s fun t ht h => ?_
        obtain ⟨⟨a, ha, hacc⟩, hG⟩ := h
        exact ⟨a, ha, hacc, t, ht, hG, ha⟩
    _ ≤ value (fun a : OutHL F D Sl => a.accepted = true ∧ W a) g :=
        sched_prob_le_value T CT g (fun a : OutHL F D Sl => a.accepted = true ∧ W a) s
    _ ≤ _ := hval

end FlockC3.Fork
