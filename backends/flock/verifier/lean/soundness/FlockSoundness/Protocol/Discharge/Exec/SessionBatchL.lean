import FlockSoundness.Protocol.Discharge.Exec.SessionSoundL

/-!
# Several salted tables in one batched session

`SessionBatch.lean` over a leaf scheme `Lf` (`sessionBL`). For each table `j`, everything after `Commit` is a batch
with table `j` at position `j` (`sessAfterB_gamesBL`), so `session_knowledge_soundL` applies to it: `joint_le_BL`,
`joint_le_BL'` (`Pr[the session accepts] · failProbBL ≤ ksBoundBL`) and the audit's `cover`
(`one_le_fail_add_BL`, `one_le_fail_add_BL'`).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type} {mPts : ℕ}

/-- The session strategy after `Commit`, seen with table `j` at its position. -/
noncomputable def stratBL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (ts : List (TabSpec mPts)) (j : Fin ts.length) (τ : Strategy (sessionBL A H E Lf ts)) :
    Strategy (sessAfterL A H E Lf (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
      ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j)) :=
  cast (congrArg Strategy (sessAfterB_gamesBL A H E Lf ts τ.1 j)) τ.2

/-! ## Per table: the extractor's probabilities and the knowledge bound -/

end FlockSoundness.Discharge.Exec
