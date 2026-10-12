import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.Recursive.Universal

/-!
# Piece (a): candidate 3's table is sound, in fork form

`CompiledForkSound`'s analogue at `Model.holoTableCL` (`Recursion.holoInnerCL`): at a statement no table near the
registered word makes true, every inner prover has a fork whose two runs clash (`clashPickHL`: the reps, R's stream, or
R's level 0 opened to a row other than the registered word's), or the table accepts with probability at most
`holoError`. `holoForkSound_holoAt` is that at the executable's parameters (`holoAt st hp`), the form
`RecursiveAuditC3`'s proof reads.

* `holo_inner_sound_compiled`: the general lemma, `flock_inner_sound_compiled`'s analogue. Its proof composes the
  reps' compile lemma (`CompileRuns`, two reps on one root as in `tableCL`), the sparse phase's soundness at each rep
  (each coin a degree over `qF` at the level-0 list bound, `sparseError`), and R's registered opening at the unique-
  decoding radius (`regOpenError`), whose level 0 is the registered word `w` that `HoloReg.Opens` defines from `cap`.
* `holoAt_fast100`, `holoAt_linkLayout`, `holoAt_rsch`: the executable parameters meet its hypotheses, as
  `compiledForkSound_execClassAt` does for `execClassAt`.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Model Accounting Exec StrictCR
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type) [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]

/-- **Candidate 3's compiled table is sound in fork form**: at the reps' schedule `fast100`'s, a link layout at every
table, and R's registered schedule either empty (R rejects) or decoding level 0 uniquely, the inner protocol
`holoInnerCL` is sound with the pick `clashPickHL` and the error `holoError`. -/
theorem holo_inner_sound_compiled (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo)
    (hsch : Accounting.fast100 e.cls.m = some e.sch)
    (hlay : ∀ R : RegTable e.cls, (R.toStatement e.τ e.regs).LinkLayout e.cls.ptLocal e.cls.mPts)
    (hR : e.rsch.sch.levels = [] ∨ e.rsch.UniqueDecoding) :
    InnerSoundForkOn H (holoInnerCL T CT A H E Lf e) (clashPickHL T CT A H E Lf e)
      (ENNReal.ofReal (holoError e)) := by
  sorry

/-- **The reps' schedule at the executable's parameters is `fast100`'s**, at a statement whose class checks pass. -/
theorem holoAt_fast100 {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st)) :
    Accounting.fast100 (holoAt st hp).cls.m = some (holoAt st hp).sch := by
  sorry

/-- **Every table's statement at the executable's parameters has the link layout**, at a statement whose class checks
pass: the regions and link points are `execClassAt st`'s (`Refine.stmtOf_linkLayout`), and no table changes them. -/
theorem holoAt_linkLayout {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st)) :
    ∀ R : RegTable (holoAt st hp).cls,
      (R.toStatement (holoAt st hp).τ (holoAt st hp).regs).LinkLayout (holoAt st hp).cls.ptLocal
        (holoAt st hp).cls.mPts := by
  sorry

/-- **R's registered schedule at the executable's parameters rejects or decodes uniquely**: `fast100`'s at `mReg`, whose
level 0 has rate `1/2` and holds the stacked table exactly, or the default, whose levels are empty. -/
theorem holoAt_rsch (st : Flock.Stmt) (hp : HoloParams) :
    (holoAt st hp).rsch.sch.levels = [] ∨ (holoAt st hp).rsch.UniqueDecoding := by
  sorry

/-- **Candidate 3's table at the executable's parameters is sound in fork form**, at a statement whose class checks
pass: `holo_inner_sound_compiled` at `execArith`, SHA-512 with plain leaves and `holoAt st hp`. -/
theorem holoForkSound_holoAt {st : Flock.Stmt} (hp : HoloParams) (h : ClassOk (shapeOf st)) :
    InnerSoundForkOn Refine.H512 (holoInnerAt T CT st hp)
      (clashPickHL T CT execArith Refine.H512 Refine.enc512 Refine.leafPlain (holoAt st hp))
      (ENNReal.ofReal (holoError (holoAt st hp))) := by
  let _ : Inhabited Refine.Dig := ⟨Refine.H512 []⟩
  exact holo_inner_sound_compiled T CT execArith execArith_correct Refine.H512 Refine.enc512 Refine.leafPlain
    (holoAt st hp) (holoAt_fast100 hp h) (holoAt_linkLayout hp h) (holoAt_rsch st hp)

end FlockC3.Fork
