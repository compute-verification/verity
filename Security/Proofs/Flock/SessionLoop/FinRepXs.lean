import Proofs.Flock.SessionLoop.FinRepView
import Proofs.Flock.SessionLoop.FinPoints

/-!
# (c.3)'s `FinRepXs`: the extra claims' coordinates

* **`Loop.extraClaims_xs`**: the claims `Stmt.extraClaims` returns are at `Drive.extras`' coordinates, region-major,
  then point by point, each region's point past its six skip coordinates (`FlockSoundness.Refine.extraClaims_ok`).
* **`Loop.linkPoints_wordsAt`**: at a statement in the masked scope with a witness, the link points a rep's request
  reads off the `Commit`'s coin (`Drive.linkPoints`, as `repInOf` does) are the session's at the opened coins
  (`viewPoints_eq`): the coin holds `points · mPts` words, and every table has the session's `mPts > 0`
  (`Masked.live_links`).
* **`Loop.finRepXs`**: so `FinRepXs` holds there.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **`extraClaims`' claims are at `Drive.extras`' coordinates**: where `extraClaims` returns at the link points `pts`,
its claims' coordinates are `Drive.extras st pts`. -/
theorem extraClaims_xs {st : Stmt} {pts : Array (Array F128)} {extras : Array _root_.Flock.Claim}
    (h : st.extraClaims pts = .ok extras) : extras.map (·.x) = _root_.Flock.Firewall.Drive.extras st pts := by
  obtain ⟨ess, hess, heq⟩ := FlockSoundness.Refine.extraClaims_ok st pts extras h
  have inner : ∀ (reg : _root_.Flock.Region) (rs : List (Array F128)) (es : List _root_.Flock.Claim),
      List.Forall₂ (fun r e => ∃ v, st.regionValue reg r = .ok v ∧
        e = FlockSoundness.Refine.regionClaimOf st reg r v) rs es →
      es.map (·.x) = rs.map fun r => let p := st.regionPoint reg r; p.extract 6 p.size := by
    intro reg rs es hf
    induction hf with
    | nil => rfl
    | cons hre _ ih =>
      obtain ⟨v, -, rfl⟩ := hre
      rw [List.map_cons, List.map_cons]
      exact List.cons_eq_cons.mpr ⟨rfl, ih⟩
  have outer : ∀ (regs : List _root_.Flock.Region) (ess : List (List _root_.Flock.Claim)),
      List.Forall₂ (fun reg es => List.Forall₂ (fun r e => ∃ v, st.regionValue reg r = .ok v ∧
        e = FlockSoundness.Refine.regionClaimOf st reg r v) pts.toList es) regs ess →
      ess.flatten.map (·.x) = regs.flatMap fun reg =>
        (pts.map fun r => let p := st.regionPoint reg r; p.extract 6 p.size).toList := by
    intro regs ess hf
    induction hf with
    | nil => rfl
    | cons hre _ ih =>
      rw [List.flatten_cons, List.map_append, ih, inner _ _ _ hre, List.flatMap_cons, Array.toList_map]
  apply Array.toList_inj.mp
  rw [Array.toList_map, heq, outer _ _ hess]
  unfold _root_.Flock.Firewall.Drive.extras
  rw [Array.toList_flatMap]

/-- **The request's link points are the session's**: at a statement in the masked scope with a witness, where the
openings are `refK x` long, rep stream `k`'s link points off the `Commit`'s coin (`repInOf`'s) are the session's at
the opened coins. -/
theorem linkPoints_wordsAt (x : Msg) (w : Flock.Wit)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    {ops : List (Msg × Msg)} (hl : ops.length = refK x) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {k : ℕ} (hk : k < 2 * spec.tables.size)
    (m₀ ρ₀ : Msg) :
    _root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[k / 2]!
        (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0) = viewPoints x ops spec := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hK := Masked.refK_eq hsx
  have hpos : 0 < ops.length := by omega
  have hrs : spec.repStreams.size = 2 * spec.tables.size := by
    have e := congrArg List.length (FlockDeploy.repStreams_toList spec)
    simpa using e
  obtain ⟨hmp, hmp0, hpc⟩ := Masked.live_links hx hsx (k := k) (by omega)
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hsz : (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).size = spec.points * spec.mPts := by
    have e := wordsAt_size x ops (s := ops.length) (p := 0) hpos le_rfl
    rw [List.take_length] at e
    rw [e, hl0, Nat.min_eq_left hpc]
  rw [viewPoints_eq x w hsc hR hl hsx m₀ ρ₀]
  unfold _root_.Flock.Firewall.Drive.linkPoints pointsOf
  show _ = (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j =>
    (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).getD (p * spec.mPts + j) 0
  rw [hsz, hmp, Nat.mul_div_cancel _ hmp0]
  refine Array.map_congr_left fun p hp => ?_
  have hp' : p < spec.points := by simpa using hp
  have hle : (p + 1) * spec.mPts ≤ (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).size := by
    rw [hsz]
    exact Nat.mul_le_mul_right _ hp'
  rw [Nat.succ_mul] at hle
  apply Array.ext
  · simp only [Array.size_extract, Array.size_map, Array.size_range]
    rw [Nat.succ_mul]
    omega
  · intro j h1 h2
    have hj : j < spec.mPts := by simpa using h2
    have hlt : p * spec.mPts + j < (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).size := by omega
    simp only [Array.getElem_extract, Array.getElem_map, Array.getElem_range]
    simp [Array.getD, hlt]

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **`FinRepXs` in the masked scope with a witness**: rep stream `k`'s request's coordinates are `Drive.extras` at its
link points off the `Commit`'s coin (`repInOf`), which are the session's at the opened coins (`linkPoints_wordsAt`),
and the extra claims there are at those coordinates (`extraClaims_xs`). -/
theorem finRepXs (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) :
    FinRepXs L₀ src real wk x w := by
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk extras hxe tb hi q f hRep
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  obtain ⟨-, hq, -⟩ := hRep
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, -, hq⟩ := bok hq
  have hqx : q.xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[k / 2]!
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[k / 2]!
        (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) := by
    cases hq
    rfl
  rw [hqx, linkPoints_wordsAt x w hsc hR hctx.1 hsx hk m₀ ρ₀]
  obtain ⟨-, -, hT, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hj : k / 2 < (Masked.stsOf x).size := by omega
  have hj' : k / 2 < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) (k / 2) hj'] at hxe
  simp only [FlockDeploy.tabsOf, Array.getElem_map] at hxe
  have hxe' : (Masked.stsOf x)[k / 2]!.extraClaims (viewPoints x ops spec) = .ok extras := by
    rw [getElem!_pos (Masked.stsOf x) (k / 2) hj]
    exact hxe
  exact extraClaims_xs hxe'

end Loop

end Claims.ZKServer
