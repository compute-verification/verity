import FlockSoundness.Discharge.Exec.Event
import FlockSoundness.Refine.Setup
import FlockSoundness.Protocol.Discharge.Exec.Setup

/-!
# flock-verify's acceptance sets up each table with `Stmt.setupH`

`Accepts I t` sets up the session with `Stmt.setupTables`. Under a tag without hidden outputs
(`tags.hiddenOutputs = false`; with them it calls `Stmt.setupHidden`), each of its tables is `Stmt.setupH`'s statement
(`setupTables_ok`): for one table, at the inputs and the record's draw; for `count > 1`, under
`Tags.tableIdentity` and at a subset draw of the table's part. So every fact proved of `setupH`'s statements
(`setupH_wf`, `setupH_layout`, `placement_of_setupH`) holds of each table flock-verify accepts with
(`accepts_setupH`).

`setupH_public` records the two checks `setupH` makes on the public file that the layout's registered values read:
`HmRow.checkPublic st.c st.pub` passed, and `st.regions` is `HmRow.regions st.c st.pub`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Stmt}

theorem tagsAt_hm96Rows (tags : Tags) (count j : ℕ) : (tagsAt tags count j).hm96Rows = tags.hm96Rows := by
  unfold tagsAt; split_ifs <;> rfl

/-- **flock-verify's acceptance sets up each table with `setupH`**: when `Accepts I t`, every table's statement is
`setupH`'s, under the table's tags and at some draw, the record's for a one-table session. -/
theorem accepts_setupH {I : Inputs} {t : Transcript} (hh : I.tags.hiddenOutputs = false) (h : Accepts I t) :
    ∃ spec sts, setupOf I t.record = .ok (spec, sts) ∧
      Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ d,
        Stmt.setupH (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables d I.partition I.program =
          .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record) := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  exact ⟨spec, sts, hs, hv, setupTables_setupH hh hs⟩

set_option maxHeartbeats 4000000 in
/-- **`setupH`'s checks on the public file** (§16.2, §16.4): the population's (or the draw's) public file passed
`HmRow.checkPublic`, and the statement's regions are `HmRow.regions` of it. -/
theorem setupH_public {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) :
    HmRow.checkPublic st.c st.pub = .ok () ∧ HmRow.regions st.c st.pub = .ok st.regions := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨by assumption, by assumption⟩

end FlockSoundness.Discharge.Exec
