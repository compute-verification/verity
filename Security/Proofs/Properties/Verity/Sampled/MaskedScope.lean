import Proofs.Properties.Verity.Recursive.BaseSetup
import Proofs.Flock.Claims.DeployTables
import Proofs.Flock.Claims.KS.PlainStep
import Proofs.Flock.Claims.KS.Range
import Proofs.Flock.Claims.Scope.Setup

/-!
# One-stage v2's session is in Flock's masked scope

One-stage v2 (`SampledProofs.OneStageZk`) sets its one session up with `sessionAt`, at one table (`inputsOf`'s
`count := 1`). Where that is `.ok (spec, sts)`, the session is in the scope `FlockDeploy.ZkIn` reads
(`sessionAt_inScope`), and each bound is a refusal v2's setup makes:

* one table: `Stmt.setupTables` makes `max 1 count` tables (`FlockDeploy.setupTables_size`);
* `25 ≤ m ≤ 35`: `sessionAt` runs `Zk.shapeOk` on each table, whose `Zk.queries0 st.m` refuses every other `m`;
* the range: each table's `setupOne` runs `checkInRange` on its `k_log`, its regions and its link points
  (`FlockDeploy.KS.setupOne_range`), which are the session's `m_pts` (`setupTables_pointsAny`);
* two link points: `Stmt.spec` sets `points := 2` (`setupTables_pointsAny`).

`buildSessionOf` sets the session up on `Stmt.setupTables` under either output form: under hidden outputs on
`ProvedScope.circuitOf`'s parse, which is `parseCommit`'s (`Stmt.setupTables_eq`).
-/

open Lean (Json)
open Verity.Recursive.BaseSetup (bind_ok)

namespace SampledProofs.OneStageZk

private theorem setupTables_of_buildSessionOf {tags : Flock.Tags} {cB pB : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {count : ℕ} {draw partition : Option Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hb : buildSessionOf (Flock.ProvedScope.circuitOf tags cB tables) tags cB pB coins tables count draw partition
      program registered = .ok (spec, sts)) :
    Flock.Stmt.setupTables tags cB pB coins count tables draw partition program = .ok (spec, sts) := by
  unfold buildSessionOf at hb
  have hh : tags.hm96Rows = true := by
    refine Classical.byContradiction fun hne => ?_
    simp [hne, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hb
  simp only [hh, Bool.not_true, Bool.false_eq_true, ↓reduceIte, pure_bind] at hb
  split at hb
  · next hhid =>
    obtain ⟨x, hs, hb⟩ := bind_ok hb
    obtain ⟨⟨⟩, -, hb⟩ := bind_ok hb
    obtain ⟨⟨⟩, -, hb⟩ := bind_ok hb
    obtain rfl : x = (spec, sts) := Except.ok.inj hb
    rw [Flock.Stmt.setupTables_eq]
    simpa only [Flock.ProvedScope.circuitOf, hhid, ↓reduceIte] using hs
  · obtain ⟨x, hs, hb⟩ := bind_ok hb
    obtain ⟨⟨⟩, -, hb⟩ := bind_ok hb
    obtain ⟨⟨⟩, -, hb⟩ := bind_ok hb
    obtain rfl : x = (spec, sts) := Except.ok.inj hb
    exact hs

/-- **v2's session is in Flock's masked scope**: where `sessionAt` sets a one-table session up at `(spec, sts)`, it has
one table, at `25 ≤ m ≤ 35`, with `k_log ≤ 27`, at most `1024` regions and `m_pts ≤ 64`, at at most two link points. -/
theorem sessionAt_inScope {I : Flock.Witness.Inputs} {rd : Reads} {draw : Option Json} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt} (hc : I.count = 1) (h : sessionAt I rd draw = .ok (spec, sts)) :
    sts.size = 1 ∧ (∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35) ∧
      (∀ st ∈ sts.toList, st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1024 ∧ spec.mPts ≤ 64) ∧ spec.points ≤ 2 := by
  unfold sessionAt at h
  dsimp only at h
  obtain ⟨⟨⟩, -, h⟩ := bind_ok h
  obtain ⟨⟨⟩, -, h⟩ := bind_ok h
  obtain ⟨r, hb, h⟩ := bind_ok h
  obtain ⟨⟨⟩, -, h⟩ := bind_ok h
  obtain ⟨⟨⟩, hsh, h⟩ := bind_ok h
  obtain ⟨⟨⟩, -, h⟩ := bind_ok h
  obtain rfl : r = (spec, sts) := Except.ok.inj h
  have hs := setupTables_of_buildSessionOf hb
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hs
  obtain ⟨hp2, hpts⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_pointsAny hs
  refine ⟨by rw [FlockDeploy.setupTables_size hs, hc, Nat.max_self], fun st hst => ?_, fun st hst => ?_, by omega⟩
  all_goals obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
  · have hq : Flock.Zk.shapeOk sts[j] = .ok () := Flock.Readable.array_forM_ok hsh j hj
    unfold Flock.Zk.shapeOk at hq
    obtain ⟨_, hq, -⟩ := bind_ok hq
    exact FlockDeploy.KS.queries0_range hq
  · obtain ⟨d, hd, -⟩ := hall j hj
    obtain ⟨hr, hwf⟩ := FlockDeploy.KS.setupOne_range hd
    obtain ⟨hk, hreg, h64⟩ := (Flock.checkInRange_ok _ _ _).1 hr
    have hm := hwf.mPts
    have hpj := hpts j hj
    exact ⟨hk, hreg, by omega⟩

end SampledProofs.OneStageZk
