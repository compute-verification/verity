import Proofs.Flock.Claims.ZK.Sent
import Proofs.Flock.Claims.Complete.ZkTables.Round1
import Proofs.Flock.Claims.Complete.Zk.Json.Runs

/-!
# The masked session's second message at a live statement

Round `1` of `liveRun`, the session answered by one opening, at a live statement: the schedules and the level-0
commitments return (`schedsPZk_live`, `FlockComplete.commitsPZk_caps`), and table `0`'s rep `0` is the first stream.
Where table `0`'s extra claims at the link points return, the session's second body is that rep's first zerocheck
squeeze (`FlockComplete.sessionPZk_go_round1`, its pads' commitment returning at any hash); where they throw, the session
stops after the link message (`FlockComplete.sessionPZk_go_round1_err`).

* `repStreams_head`: the first rep stream is table `0`'s rep `0`.
* `liveRun_round1`: round `1` of `liveRun` at a live statement, in those two cases.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **The first rep stream is table `0`'s rep `0`**, at a session of some table. -/
theorem repStreams_head (spec : _root_.Flock.SessionSpec) (h : 0 < spec.tables.size) :
    spec.repStreams.toList = (0, 0, s!"{spec.tables[0]!}/rep{0}") ::
      ((List.range (2 * spec.tables.size - 1)).map fun k =>
        ((k + 1) / 2, (k + 1) % 2, s!"{spec.tables[(k + 1) / 2]!}/rep{(k + 1) % 2}")) := by
  rw [FlockDeploy.repStreams_toList, show 2 * spec.tables.size = 2 * spec.tables.size - 1 + 1 by omega,
    List.range_succ_eq_map]
  simp

/-- **Round `1` of `liveRun` at a live statement**: the schedules and the level-0 commitments return, and either table
`0`'s extra claims at the link points throw and the session stops after the link message, or they return, the pads'
commitment returns, and the session's second body is table `0` rep `0`'s first zerocheck squeeze, from the domain's
transcript with the digest, the level-0 cap and the pads' root absorbed. -/
theorem liveRun_round1 (H : Msg → Msg) (hH : Claims.hOf H = _root_.Flock.Sha512.hash) {x : Msg} (hx : Live x)
    (w : Claims.Flock.Wit) (ω : FlockComplete.TapeZk) (ops : List (Msg × Msg)) (hl : ops.length = 1) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) :
    ∃ schs cms, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs ∧
      (FlockComplete.commitsPZk (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) (FlockDeploy.tabsOf (stsOf x))
        schs (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x))
          (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask))) ω :
        _root_.Flock.V _) = .ok cms ∧
      (((∃ e, (FlockDeploy.tabsOf (stsOf x))[0]!.extra ((Array.range spec.points).map fun p =>
          (Array.range spec.mPts).map fun j => FlockComplete.coinOf (coinOfD ops[0]!.2) (p * spec.mPts + j)) =
            .error e) ∧
        (∃ e, (liveRun x w ω ops spec).1 = .error e) ∧ (liveRun x w ω ops spec).2.size ≤ ops.length) ∨
      ∃ extras ic, (FlockDeploy.tabsOf (stsOf x))[0]!.extra ((Array.range spec.points).map fun p =>
          (Array.range spec.mPts).map fun j => FlockComplete.coinOf (coinOfD ops[0]!.2) (p * spec.mPts + j)) =
            .ok extras ∧
        (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
          (FlockComplete.padsOf ω 0 0 (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[0]!.m
            (FlockDeploy.tabsOf (stsOf x))[0]!.kLog schs[0]!.1.k0)) (ω.innerSalt 0 0) : _root_.Flock.V _) = .ok ic ∧
        ops.length < (liveRun x w ω ops spec).2.size ∧
        (liveRun x w ω ops spec).2[ops.length]! =
          Claims.toMsg (_root_.Flock.Transcript.framed (FlockComplete.Replay.ZcZ.t₁
            (FlockDeploy.tabsOf (stsOf x))[0]!.digest (FlockComplete.flat cms[0]!.cap) ic.root
            (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6)) := by
  obtain ⟨h0, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  obtain ⟨schs, hs, -⟩ := schedsPZk_live hx
  have hms : ∀ i < (FlockDeploy.tabsOf (stsOf x)).size,
      (FlockDeploy.tabsOf (stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  obtain ⟨cms, hcm, -, -⟩ := FlockComplete.commitsPZk_caps (FlockDeploy.tabsOf (stsOf x)) schs
    (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x))
      (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask))) ω hms
  refine ⟨schs, cms, hs, hcm, ?_⟩
  have hc : (FlockComplete.commitsPZk (_root_.Flock.Hashes.ofSha512 (Claims.hOf H)) (FlockDeploy.tabsOf (stsOf x))
      schs (FlockComplete.tablesP (FlockDeploy.tabsOf (stsOf x))
        (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask))) ω :
      _root_.Flock.V _) = .ok cms := by
    rw [hH]
    exact hcm
  have hrs := repStreams_head spec (by omega)
  have hrun : liveRun x w ω ops spec = (FlockComplete.sessionPZk
      (FlockComplete.asksS fun k => (ops.map fun p => coinOfD p.2).getD k [])
      (_root_.Flock.Hashes.ofSha512 (Claims.hOf H)) spec (FlockDeploy.tabsOf (stsOf x))
      (FlockComplete.MaskRank.masksOf (stsOf x))
      (FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask)) ω).go #[] := by
    rw [hH]
    rfl
  have hr0 : (ops.map fun p => coinOfD p.2).getD 0 [] = coinOfD ops[0]!.2 := by
    obtain ⟨o, rfl⟩ : ∃ o, ops = [o] := List.length_eq_one_iff.1 hl
    rfl
  rw [hrun, hl]
  rw [← hr0]
  generalize FlockDeploy.tabsOf (stsOf x) = tabs at hs hc hcm ⊢
  generalize FlockComplete.ZkTables.traces (stsOf x) (FlockComplete.ZkTables.maskWOf x w ω.mask) = trs at hc hcm ⊢
  cases he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => FlockComplete.coinOf ((ops.map fun p => coinOfD p.2).getD 0 [])
        (p * spec.mPts + j)) with
  | error e =>
    refine .inl ⟨⟨e, rfl⟩, ?_⟩
    have hg := @FlockComplete.sessionPZk_go_round1_err H (fun k => (ops.map fun p => coinOfD p.2).getD k []) spec
      tabs (FlockComplete.MaskRank.masksOf (stsOf x)) trs ω _ _ hs hc _ _ hrs e
    rw [hg he]
    exact ⟨⟨e, rfl⟩, by simp⟩
  | ok extras =>
    obtain ⟨ic, hi⟩ := FlockComplete.JsonRT.innerCommitP_ok _root_.Flock.Sha512.hash
      (FlockComplete.padsOf ω 0 0 (_root_.Flock.Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0))
      (ω.innerSalt 0 0)
    refine .inr ⟨extras, ic, rfl, hi, ?_⟩
    have hi' : (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 (Claims.hOf H))
        (FlockComplete.padsOf ω 0 0 (_root_.Flock.Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0))
        (ω.innerSalt 0 0) : _root_.Flock.V _) = .ok ic := by
      rw [hH]
      exact hi
    have hg := @FlockComplete.sessionPZk_go_round1 H (fun k => (ops.map fun p => coinOfD p.2).getD k []) spec tabs
      (FlockComplete.MaskRank.masksOf (stsOf x)) trs ω _ _ hs hc _ _ hrs extras
    exact hg he hi'

end Masked

end Claims.ZKMal
