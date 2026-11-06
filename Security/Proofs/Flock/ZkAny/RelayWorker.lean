module

public import Proofs.Flock.ZkAny.RelayLoop
public import Proofs.Flock.SessionLoop.TakenVals

/-!
# The deployed worker at the link and the `Finish`

The ZK server passes a worker's value at the link and the `Finish` only where it is the firewall's own message
(`LeanRelease.ownMsg`), and the relay sends the firewall's own there: the firewall computes the link and the `Finish`
itself, and the worker sends only the frame rounds' device rounds. So the model's worker is built from a device's
rounds at the frame rounds and the firewall's own message at every other round:

* `LeanRelease.deployedWk`: the worker built so at a device `dv`.
* `LeanRelease.kindAt_link`, `LeanRelease.kindAt_finish`: the link and the `Finish` are no frame rounds.
* `LeanRelease.honestWk_deployed`, `Loop.valsWk_deployed`: the honest worker and the worker that sends the scheduled
  rounds with `vs`'s values are built so, at their devices (`ownMsg_withDevice`: the firewall's own message doesn't read
  the device).
* **`LoopModel.link_deployed`**, **`LoopModel.finish_deployed`**: where the model's worker is `deployedWk` at some
  device, it answers the link and the `Finish` with the firewall's own message.
-/

@[expose] public section

namespace Claims.ZKServer

namespace LeanRelease

/-- **The deployed worker** at the device `dv`: at a frame round `dv`'s value (the device round, with whether it is the
firewall's stand-in), at every other round the firewall's own message (`ownMsg`), which the firewall computes and sends
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

/-- **The firewall's own message doesn't read the device.** -/
theorem ownMsg_withDevice {W R : Type} (L : LeanRelease W R)
    (d : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg) (x : Msg) (w : W) (ρ : R)
    (h : List (Msg × Msg)) : (L.withDevice d).ownMsg x w ρ h = L.ownMsg x w ρ h :=
  rfl

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

/-- **The deployed worker answers the link with the firewall's own message**: where the model's worker is
`deployedWk` at some device. -/
theorem LoopModel.link_deployed {W Rr : Type} {M : LoopModel W Rr} {dv : M.L.toSrv.Worker}
    (hwk : M.wk = M.L.deployedWk dv) (h : List (Msg × Msg)) (hl : h.length = 1) :
    M.wk M.x M.w M.ρ h 0 = some (M.L.ownMsg M.x M.w M.ρ h, false) := by
  rw [hwk]
  exact M.L.deployedWk_own dv (by rw [hl]; exact M.L.kindAt_link M.x)

/-- **The deployed worker answers the `Finish` with the firewall's own message**: where the model's worker is
`deployedWk` at some device. -/
theorem LoopModel.finish_deployed {W Rr : Type} {M : LoopModel W Rr} {dv : M.L.toSrv.Worker}
    (hwk : M.wk = M.L.deployedWk dv) (h : List (Msg × Msg)) (hl : h.length = M.L.last M.x) :
    M.wk M.x M.w M.ρ h 0 = some (M.L.ownMsg M.x M.w M.ρ h, false) := by
  rw [hwk]
  exact M.L.deployedWk_own dv (by rw [hl]; exact M.L.kindAt_finish M.x)

end Relay

end Claims.ZKServer
