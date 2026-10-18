import Proofs.Flock.Claims.Complete.ZkTables.Scope

/-!
# The masked skeleton's goals discharged, at several tables

`Zk/Discharge.lean`'s goals at every honest masked run of any number of tables, from the scope (`Case.tables`):

* `tables_goal` (`TablesZk`, §3): one statement per table.
* `schedules_goal` (`SchedulesZk`): `Zk.schedule` has a schedule at every table's `m` (`zkSchedule_ok`).
* `proofCount_goal` (`ProofCountZk`): the masked session returns one proof file per rep stream (`sessionPZk_proofs`).
-/

open Flock FlockDeploy

namespace FlockComplete.ZkTables

/-- **One statement per table** (§3): the goal `TablesZk`. -/
theorem tables_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : TablesZk H rel := by
  intro o
  obtain ⟨hpos, -, hn, -⟩ := o.tables
  unfold Run.tabs tabsOf
  simp only [Array.size_map, hn, bne_self_eq_false, Bool.false_or, beq_eq_false_iff_ne]
  omega

/-- **Each table has its `--zk` schedule**: the goal `SchedulesZk`. -/
theorem schedules_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    SchedulesZk H rel := by
  intro o st hst
  obtain ⟨-, -, -, hall⟩ := o.tables
  unfold Run.tabs tabsOf at hst
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
  simp only [Array.size_map] at hj
  obtain ⟨h1, h2, -⟩ := hall j hj
  simp only [Array.getElem_map, Setup.ofCircuit]
  exact zkSchedule_ok h1 h2

/-- **One proof file per rep stream**: the goal `ProofCountZk`. -/
theorem proofCount_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    ProofCountZk H rel := by
  intro o
  have := sessionPZk_proofs (Claims.hOf H) (fun j => (Claims.coinSeq o.r j).bytes) o.spec (tabsOf o.sts)
    (MaskRank.masksOf o.sts) (traces o.sts o.w) o.ω #[] _ _ o.ran
  simp [this]

end FlockComplete.ZkTables
