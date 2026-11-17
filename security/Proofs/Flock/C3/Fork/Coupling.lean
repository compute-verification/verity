module

public import Proofs.Flock.C3.Fork.CouplingInh
public import Proofs.Flock.C3.Fork.CouplingProb
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

/-! The coupling: candidate 3's compiled table against its oracle table (`holoTable_value_le_oracle`). -/

@[expose] public section

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

theorem holoTable_value_le_oracle (A : Arith F K) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo)
    (hsch : Accounting.fast100 e.cls.m = some e.sch)
    (hR : e.rsch.sch.levels = [] ∨ Accounting.fast100 e.cls.shape.mReg = some e.rsch.sch) (x : HoloReg F D) (ε : ℝ)
    (horc : ∀ σ : Strategy (holoTableOf A e.cls e.regs e.sch e.rsch e.τ x.w),
      prob (fun out => out.1 = true) (holoTableOf A e.cls e.regs e.sch e.rsch e.τ x.w) σ ≤ ENNReal.ofReal ε)
    (W : OutHL F D Sl → Prop) (hW : ∀ o o', W o → W o' → ¬ ClashHL H E Lf e.sch e.rsch.sch x o o') :
    value (fun o : OutHL F D Sl => o.accepted = true ∧ W o)
      (holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) ≤ ENNReal.ofReal ε := by
  obtain ⟨hlen, hk0, -, -⟩ := fast100_shape _ _ hsch
  have hsh : e.rsch.sch.levels = [] ∨
      2 ≤ e.rsch.sch.levels.length ∧ (e.rsch.sch.levels.headD default).lanes = e.rsch.sch.k0 := by
    rcases hR with h | h
    · exact Or.inl h
    · obtain ⟨h1, h2, -, -⟩ := fast100_shape _ _ h
      exact Or.inr ⟨h1, h2⟩
  exact value_le_of_forall_prob
    (allInh_holoTableCL A H E Lf (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
      e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.cap) fun σ =>
    (holoTableCL_prob_le A H E Lf (e.cls.pubStatement e.regs) e.sch hlen hk0 e.cls.hm e.cls.ptLocal e.cls.mPts
      e.cls.shape e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch hsh x W hW σ).trans
    (value_le_of_forall_prob
      (allInh_holoTable A (e.cls.pubStatement e.regs) e.sch e.cls.hm e.cls.ptLocal e.cls.mPts e.cls.shape
        e.cls.units (fun σ => (e.τ σ).val) e.rsch.sch x.w) horc)

end FlockC3.Fork
