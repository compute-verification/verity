import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.C3.Fork.CouplingValue

/-!
# Piece (a), the coupling: candidate 3's compiled table against its oracle table

`holoTable_value_le_oracle`, `table_prob_clashFree`'s analogue at `Model.holoTableCL`: on a set `W` of outcomes no two
of which clash (`ClashHL`), the compiled table ends accepted in `W` no more often than some prover of the oracle table
(`holoTableOf`) at the registered word makes it accept. As there, the oracle prover sends `W`'s tables: `tab₀W` at the
session's cap and `tabRW` at each cap a rep's or R's stream records. R's level 0 is the registered word itself, since no
run of `W` opens it off `w` (`RegClash`). It needs `CompiledL`'s lockstep (`lockTableL`, `compiled_le_bad_on`,
`bad_zero_of_fixed`) with R's stream as a third table, whose level 0 is fixed rather than sent, and `regPhase` between
the reps and the openings.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-- **(a1l) Candidate 3's compiled table on a clash-free set of outcomes** is accepted no more often than the oracle
table: if every prover makes `holoTableOf` at the registered word accept with probability at most `ε`, the compiled
table's value on accepted outcomes in a set `W` no two of which clash is at most `ε`. -/
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
  sorry

end FlockC3.Fork
