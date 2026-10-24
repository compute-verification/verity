import Proofs.Flock.Claims.ZK.PrefixLive
import Proofs.Flock.Claims.DeployK
import Flock.Firewall.Frames

/-!
# A live session's link points

`live_links`: at a `Live` statement, the table of every rep stream has the session's link points, of `PT_LOCAL` to 64
coordinates at two points, so the link fits in `Frames.COIN_MAX` words. These are the setup facts
`ofLoop_progChecks_of` takes as `hP`.
-/

namespace Claims.ZKMal.Masked

open _root_.Flock FlockDeploy FlockComplete

/-- **At a `Live` statement every rep stream's table has the session's link points**, nonzero and within
`Frames.COIN_MAX` at the session's two points (`setupTables_pointsAny`, and each table's `checkInRange` at
`PT_LOCAL + nbl` coordinates, as `live_bounds`). -/
theorem live_links {x : Msg} (hx : Live x) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hsx : setupOf true x = some (s, spec, sts)) {k : ℕ} (hk : k < spec.repStreams.size) :
    sts[k / 2]!.mPts = spec.mPts ∧ 0 < spec.mPts ∧
      spec.points * spec.mPts ≤ _root_.Flock.Firewall.Frames.COIN_MAX := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk Readable.relation) Readable.relation (readOf true) rfl
    (fun _ _ h => readOf_zk h) x).1 hx.1
  have hi := (setupOf_some hsx).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := s.read) hp
  obtain ⟨hck, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some hsx).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  obtain ⟨hpts, hmp⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_pointsAny hset
  obtain ⟨-, -, hsz, -⟩ := ZkTables.tables_of_setup hx.1 hsx
  have hk2 : k / 2 < sts.size := by
    rw [FlockDeploy.repStreams_size, hsz] at hk
    omega
  have hlo : PT_LOCAL ≤ sts[k / 2].mPts := by
    obtain ⟨d, hd, -⟩ := (FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset).2 _ hk2
    rcases Bool.eq_false_or_eq_true (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count (k / 2)).hiddenOutputs
      with hh | hh
    · rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hh] at hd
      rw [(FlockSoundness.Discharge.Hidden.setupHidden_spec' hd).mPts]
      omega
    · rw [FlockSoundness.Discharge.HiddenExec.setupOne_public hh] at hd
      rw [setupH_mPts hd]
      omega
  have h64 := (live_bounds hx hsx).2.1
  rw [getElem!_pos sts (k / 2) hk2, hmp _ hk2]
  rw [hmp _ hk2] at hlo
  refine ⟨rfl, ?_, ?_⟩
  · unfold PT_LOCAL at hlo
    omega
  · rw [hpts]
    unfold _root_.Flock.Firewall.Frames.COIN_MAX
    omega

end Claims.ZKMal.Masked
