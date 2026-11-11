module

public import Proofs.Flock.ZkAny.RelayLoop
public import Proofs.Flock.ZkAny.RelayLoopFacts
public import Proofs.Flock.SessionLoop.Inner
public import Proofs.Flock.SessionLoop.TakenVals

/-!
# The deployed worker, and Rust's checks as the worker's

The ZK server passes a worker's value at the link and the `Finish` only where it is the ZK server's own message
(`LeanRelease.ownMsg`), and the relay sends the ZK server's own there: the ZK server computes the link and the `Finish`
itself, and the worker sends only the frame rounds' device rounds. So the model's worker is built from a device's
rounds at the frame rounds and the ZK server's own message at every other round:

* `LeanRelease.deployedWk`: the worker built so at a device `dv`.
* `LeanRelease.kindAt_link`, `LeanRelease.kindAt_finish`: the link and the `Finish` are no frame rounds.
* `LeanRelease.honestWk_deployed`, `Loop.valsWk_deployed`: the honest worker and the worker that sends the scheduled
  rounds with `vs`'s values are built so, at their devices (`ownMsg_withDevice`: the ZK server's own message doesn't read
  the device).
* **`LoopModel.link_deployed`**, **`LoopModel.finish_deployed`**: where the model's worker is `deployedWk` at some
  device, it answers the link and the `Finish` with the ZK server's own message.

Rust's checks on a worker's rep (`LeanRelease.cross`: the worker failing, the device's level-0 root, the masking
cross-checks, the end check on the returned proof and the constraints on the pads) run in `flock-circuit` on the
prover's side of the loop's connection, before a round is sent: a rep a check failed sends each later round as its
place with no device round in it (`Firewall::round`, `Firewall::scheduled`), and the checks at the `Finish` and on the
coins of a round already sent change only the rounds after it. What reaches the loop at a frame round is the round they
leave, the content `LiveChallenger::round` sends, so they are the worker's, and the release's decision is the frame's
alone:

* `LeanRelease.withoutCross`: the release with `cross` passing every value.
* `LeanRelease.crossPasses_of_cross`, **`LoopModel.Deployed.crossPasses`**: where `cross` passes every value,
  `CrossPasses` holds, so it holds by definition at the deployed loop over `withoutCross`.
-/

@[expose] public section

namespace Claims.ZKServer

namespace LeanRelease

/-- **The deployed worker** at the device `dv`: at a frame round `dv`'s value (the device round, with whether it is the
ZK server's stand-in), at every other round the ZK server's own message (`ownMsg`), which the ZK server computes and sends
itself. -/
def deployedWk {W R : Type} (L : LeanRelease W R) (dv : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (h : List (Msg × Msg)) (s : ℕ) : Option L.toSrv.Val :=
  if L.kindAt x h.length = .masked then dv x w ρ h s else some (L.ownMsg x w ρ h, false)

theorem deployedWk_own {W R : Type} (L : LeanRelease W R) (dv : L.toSrv.Worker) {x : Msg} {w : W} {ρ : R}
    {h : List (Msg × Msg)} {s : ℕ} (hk : L.kindAt x h.length ≠ .masked) :
    L.deployedWk dv x w ρ h s = some (L.ownMsg x w ρ h, false) := by
  unfold deployedWk
  exact ite_eq_right hk

theorem deployedWk_masked {W R : Type} (L : LeanRelease W R) (dv : L.toSrv.Worker) {x : Msg} {w : W} {ρ : R}
    {h : List (Msg × Msg)} {s : ℕ} (hk : L.kindAt x h.length = .masked) :
    L.deployedWk dv x w ρ h s = dv x w ρ h s := by
  unfold deployedWk
  exact ite_eq_left hk

/-- **The link is no frame round.** -/
theorem kindAt_link {W R : Type} (L : LeanRelease W R) (x : Msg) : L.kindAt x 1 ≠ .masked :=
  fun h => (kindAt_masked h).2.1 rfl

/-- **The `Finish` is no frame round.** -/
theorem kindAt_finish {W R : Type} (L : LeanRelease W R) (x : Msg) : L.kindAt x (L.last x) ≠ .masked :=
  fun h => (kindAt_masked h).2.2 rfl

/-- **The honest worker is the deployed worker** at the device sending the honest device round (`devOf`). -/
theorem honestWk_deployed {W R : Type} (L : LeanRelease W R) :
    L.honestWk = L.deployedWk fun x w ρ h _ => some (L.devOf x w ρ h, false) := by
  funext x w ρ h s
  by_cases hk : L.kindAt x h.length = .masked
  · rw [deployedWk_masked _ _ hk]
    show some (if L.kindAt x h.length = .masked then (L.devOf x w ρ h, false) else (L.ownMsg x w ρ h, false)) =
      some (L.devOf x w ρ h, false)
    rw [ite_eq_left hk]
  · rw [deployedWk_own _ _ hk]
    show some (if L.kindAt x h.length = .masked then (L.devOf x w ρ h, false) else (L.ownMsg x w ρ h, false)) =
      some (L.ownMsg x w ρ h, false)
    rw [ite_eq_right hk]

/-- **The ZK server's own message doesn't read the device.** -/
theorem ownMsg_withDevice {W R : Type} (L : LeanRelease W R)
    (d : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg) (x : Msg) (w : W) (ρ : R)
    (h : List (Msg × Msg)) : (L.withDevice d).ownMsg x w ρ h = L.ownMsg x w ρ h :=
  rfl

/-- **The release without Rust's checks beside the frame**: `L` with `cross` passing every value, Rust's checks being
the worker's. -/
def withoutCross {W R : Type} (L : LeanRelease W R) : LeanRelease W R :=
  { L with cross := fun _ _ _ _ _ _ => true }

/-- **Where `cross` passes every value, `CrossPasses` holds.** -/
theorem crossPasses_of_cross {W R : Type} {L : LeanRelease W R} (hcr : ∀ x j w ρ h v, L.cross x j w ρ h v = true) :
    L.CrossPasses := by
  intro x w ρ h v _ _ hc
  rw [hcr] at hc
  cases hc

end LeanRelease

namespace Loop

open Claims.ZKMal

/-- **The worker that sends the scheduled rounds with `vs`'s values is the deployed worker** at the device sending the
device round with `vs`'s values (`valsDev`). -/
theorem valsWk_deployed (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)) :
    valsWk vs L =
      L.deployedWk fun x w ρ h _ => some (valsDev vs L x w ρ h (computedOf (L.computed x w ρ h)), false) := by
  funext x w ρ h s
  by_cases hk : L.kindAt x h.length = .masked
  · rw [LeanRelease.deployedWk_masked _ _ hk]
    show some (if L.kindAt x h.length = .masked then
        (valsDev vs L x w ρ h (computedOf (L.computed x w ρ h)), false)
      else ((L.withDevice (valsDev vs L)).ownMsg x w ρ h, false)) =
      some (valsDev vs L x w ρ h (computedOf (L.computed x w ρ h)), false)
    rw [ite_eq_left hk]
  · rw [LeanRelease.deployedWk_own _ _ hk]
    show some (if L.kindAt x h.length = .masked then
        (valsDev vs L x w ρ h (computedOf (L.computed x w ρ h)), false)
      else ((L.withDevice (valsDev vs L)).ownMsg x w ρ h, false)) = some (L.ownMsg x w ρ h, false)
    rw [ite_eq_right hk, LeanRelease.ownMsg_withDevice]

end Loop

namespace Relay

/-- **The deployed worker answers the link with the ZK server's own message**: where the model's worker is
`deployedWk` at some device. -/
theorem LoopModel.link_deployed {W Rr : Type} {M : LoopModel W Rr} {dv : M.L.toSrv.Worker}
    (hwk : M.wk = M.L.deployedWk dv) (h : List (Msg × Msg)) (hl : h.length = 1) :
    M.wk M.x M.w M.ρ h 0 = some (M.L.ownMsg M.x M.w M.ρ h, false) := by
  rw [hwk]
  exact M.L.deployedWk_own dv (by rw [hl]; exact M.L.kindAt_link M.x)

/-- **The deployed worker answers the `Finish` with the ZK server's own message**: where the model's worker is
`deployedWk` at some device. -/
theorem LoopModel.finish_deployed {W Rr : Type} {M : LoopModel W Rr} {dv : M.L.toSrv.Worker}
    (hwk : M.wk = M.L.deployedWk dv) (h : List (Msg × Msg)) (hl : h.length = M.L.last M.x) :
    M.wk M.x M.w M.ρ h 0 = some (M.L.ownMsg M.x M.w M.ρ h, false) := by
  rw [hwk]
  exact M.L.deployedWk_own dv (by rw [hl]; exact M.L.kindAt_finish M.x)

open _root_.Flock (V SessionSpec Stmt) in
/-- **`CrossPasses` by definition**: at the deployed loop over the release with the session loop's own rounds of the
inner proof over `L₁.withoutCross`, `cross` passes every value. -/
theorem LoopModel.Deployed.crossPasses
    {L₁ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {src : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      Nat → V (Array _root_.Flock.F128 × ByteArray × Array ByteArray)}
    {fin : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray}
    {M : LoopModel Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Nonce ×
        (ZKMal.CoinLayer.session Loop.loopNr ZKMal.Masked.dataE).refP.Rand)}
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hD : LoopModel.Deployed M (LeanRelease.withFiles (LeanRelease.withLink
      (LeanRelease.withInner ZKMal.Masked.dataE L₁.withoutCross (Loop.realOf src) fin) src) src) (Loop.realOf src) s0
      spec sts) :
    M.L.CrossPasses := by
  rw [hD.rel]
  exact LeanRelease.crossPasses_of_cross fun _ _ _ _ _ _ => rfl

end Relay

end Claims.ZKServer
