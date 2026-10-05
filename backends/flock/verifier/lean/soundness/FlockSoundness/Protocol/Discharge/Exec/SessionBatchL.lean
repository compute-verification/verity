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

omit [Fintype K] in
theorem sessAfterB_eqL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (T : TabSpec mPts) (cap₀ : Cap D) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) :
    sessAfterBL (pre ++ tabAfterL A H E Lf T cap₀ :: post) =
      sessAfterL A H E Lf T.S T.sch T.hm T.ptLocal mPts pre post cap₀ := by
  simp only [sessAfterBL, sessAfterL, List.map_append, List.map_cons]
  rfl

omit [Fintype K] in
/-- **Table `j`'s place in the session**: everything after `Commit` is a batch with table `j` at
position `j`. -/
theorem sessAfterB_gamesBL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (ts : List (TabSpec mPts)) (caps : Fin ts.length → Cap D) (j : Fin ts.length) :
    sessAfterBL (gamesBL A H E Lf ts caps) =
      sessAfterL A H E Lf (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
        ((gamesBL A H E Lf ts caps).take j) ((gamesBL A H E Lf ts caps).drop (j + 1)) (caps j) := by
  have hs : gamesBL A H E Lf ts caps = (gamesBL A H E Lf ts caps).take j ++
      tabAfterL A H E Lf (ts.get j) (caps j) :: (gamesBL A H E Lf ts caps).drop (j + 1) :=
    ofFn_split (fun i => tabAfterL A H E Lf (ts.get i) (caps i)) j
  exact (congrArg sessAfterBL hs).trans (sessAfterB_eqL A H E Lf (ts.get j) (caps j) _ _)

/-- The session strategy after `Commit`, seen with table `j` at its position. -/
noncomputable def stratBL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (ts : List (TabSpec mPts)) (j : Fin ts.length) (τ : Strategy (sessionBL A H E Lf ts)) :
    Strategy (sessAfterL A H E Lf (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
      ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j)) :=
  cast (congrArg Strategy (sessAfterB_gamesBL A H E Lf ts τ.1 j)) τ.2

/-! ## Per table: the extractor's probabilities and the knowledge bound -/

end FlockSoundness.Discharge.Exec
