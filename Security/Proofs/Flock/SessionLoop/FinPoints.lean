import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.Claims.ZK.FrameRankFault
import Proofs.Flock.Claims.ZK.FrameLinks
import Proofs.Flock.VBridge.Family.Structure

/-!
# (b) `FinPoints`: the link points at the opened coins

* **`Loop.viewPoints_eq`**: at a `Finish` `FinishIs` reads, at a statement in the masked scope with a witness, the link
  points the session reads at the opened coins are the ones the auditor saw (`pointsOf`): point `p`'s word `j` is element
  `p·mPts + j` of the `Commit`'s coin, which the loop requests `points·mPts` words of (`loopNr x 0`), within
  `COIN_MAX` (`live_links`), and which the opened coin holds (`wordsAt_slot`).
* **`Loop.finPoints_of`**: so `FinPoints` holds where each table's regions carry bytes of their free bits' size
  (`FlockVBridge.Family.RegionBytes`, `extraClaims_exists`). That is not a premise of `SessionLoopZKDeployed`: the
  extra claims (`Stmt.regionValue`) refuse a region whose bytes are not `2^|free|` bits, while the loop's `Finish`
  carries only the claims' coordinates (`Drive.extras`, `Finish.claimsOf`) and never computes their values.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **The link points at the opened coins are the auditor's**: at a statement in the masked scope with a witness, where
the openings are `refK x` long, the session's link points at the coins they open are `pointsOf`'s. -/
theorem viewPoints_eq (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    {ops : List (Msg × Msg)} (hl : ops.length = refK x) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (m₀ ρ₀ : Msg) :
    viewPoints x ops spec = pointsOf x ((m₀, ρ₀) :: ops) spec := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hK := Masked.refK_eq hsx
  have hpos : 0 < ops.length := by omega
  obtain ⟨hts, -, htab, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hrs : spec.repStreams.size = 2 * spec.tables.size := by
    have e := congrArg List.length (FlockDeploy.repStreams_toList spec)
    simpa using e
  obtain ⟨-, -, hpc⟩ := Masked.live_links hx hsx (k := 0) (by omega)
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hsz : (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).size = spec.points * spec.mPts := by
    have e := wordsAt_size x ops (s := ops.length) (p := 0) hpos le_rfl
    rw [List.take_length] at e
    rw [e, hl0, Nat.min_eq_left hpc]
  unfold viewPoints pointsOf
  refine Array.map_congr_left fun p hp => Array.map_congr_left fun j hj => ?_
  have hp' : p < spec.points := by simpa using hp
  have hj' : j < spec.mPts := by simpa using hj
  have hlt : p * spec.mPts + j < (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).size := by
    rw [hsz]
    calc p * spec.mPts + j < p * spec.mPts + spec.mPts := by omega
      _ = (p + 1) * spec.mPts := by rw [Nat.succ_mul]
      _ ≤ spec.points * spec.mPts := Nat.mul_le_mul_right _ hp'
  have hw := wordsAt_slot Masked.dataE x ops hl (s := ops.length) (p := 0) (e := p * spec.mPts + j) hpos le_rfl
    (by rw [List.take_length]; exact hlt)
  rw [List.take_length] at hw
  have hge : (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).getD (p * spec.mPts + j) 0 =
      (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0)[p * spec.mPts + j]! := by
    simp [Array.getD, hlt, getElem!_pos]
  exact hw.symm.trans hge.symm

/-- **(b) where the regions carry their bytes**: at a statement in the masked scope with a witness whose tables' regions
each carry bytes of their free bits' size, the link points at the opened coins are the auditor's and each table's extra
claims at them return. -/
theorem finPoints_of (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hb : ∀ j (hj : j < (Masked.stsOf x).size), FlockVBridge.Family.RegionBytes (Masked.stsOf x)[j]) :
    FinPoints L₀ src real wk x w := by
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx
  refine ⟨viewPoints_eq x w hsc hR hctx.1 hsx m₀ ρ₀, fun k hk => ?_⟩
  obtain ⟨-, -, htab, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hk2 : k / 2 < (Masked.stsOf x).size := by omega
  have hk2' : k / 2 < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hk2
  rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) hk2']
  simp only [FlockDeploy.tabsOf, Array.getElem_map]
  exact FlockVBridge.Family.extraClaims_exists _ _ (hb (k / 2) hk2)

end Loop

end Claims.ZKServer
