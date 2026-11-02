import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.SessionLoop.TakenVals
import Proofs.Flock.Claims.ZK.FrameLive

/-!
# `FillRoundEq` at the honest worker

`FillRoundEq` reads only histories on which the run is not live, so it holds at every `vs` for a worker live wherever
it reads (`LeanRelease.fillRoundEq_of_live`). A release's honest worker (`LeanRelease.honestWk`, its own device rounds)
is live on a history whose rounds' coins the frame all took, where each such round passes the frame's checks
(`Loop.HonestRoundsPass`, the frame's completeness for that device; `Loop.live_honest`). Where the frame refuses a
round's coins no device round passes them (`LeanRelease.roundOk_of_refusesCoins`), so that completeness reads only
taken coins.

The release's coin stop sticks as the session loop's does (`Flock.Firewall.Session`'s `State.refuse`): once it refuses
the coins after a prefix of a history it refuses the history's own (`LeanRelease.refusesCoins_of_take`). So a history
whose last coins the frame takes had every round's coins taken, and `FillRoundEq` holds at the honest worker
(`LeanRelease.fillRoundEq_honest`).

The worker that sends the scheduled rounds with `vs'`'s values (`Loop.valsWk vs' L`) is the honest worker at the device
`Loop.valsDev vs' L`, so `FillRoundEq` holds at it where its rounds pass the frame's checks at that device
(`LeanRelease.fillRoundEq_valsWk`); at the tables' own values that is the honest worker of the guarantees
(`LeanRelease.fillRoundEq_honestVals`). Every part of those checks the Lean frame defines (its pass decision, its taking
of the round and `valsDev`'s round) reads the history only through its decoded coins at the loop's frames; the
release's `cross` is Rust's own checks.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace LeanRelease

/-- **No device round passes the frame's checks where the frame refuses the coins**: the frame takes no round there
(`takes`). -/
theorem roundOk_of_refusesCoins {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (g : List (Msg × Msg))
    (hr : L.refusesCoins x ρ g = true) : L.roundOk x w ρ g = false := by
  have ht : L.takes x ρ g (L.devOf x w ρ g) = false := by
    unfold refusesCoins at hr
    unfold takes step
    cases ha : L.answered x ρ g with
    | ok m => rw [ha] at hr; cases hr
    | error e => rfl
  unfold roundOk
  rw [ht, Bool.and_false]

/-- The frame refuses the same coins at every device: its stop reads the coins, never the device round. -/
theorem refusesCoins_withDevice {W R : Type} (L : LeanRelease W R)
    (d : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg) (x : Msg) (ρ : R)
    (g : List (Msg × Msg)) : (L.withDevice d).refusesCoins x ρ g = L.refusesCoins x ρ g :=
  rfl

end LeanRelease

namespace Loop

/-- Completeness (the frame's, for `L`'s device, which Lean does not model for the deployed device; the honest session
of `verity/ml/flock/verifier/firewall_controls.sh` checks it under `FC_FIREWALL=1`). **Each honest round passes the
frame's checks where the frame takes its coins**: at `x` and `w`, after every history of `2` to `refK x` rounds whose
last coins the frame takes, the frame round passes the frame's checks at `L`'s device round (`LeanRelease.roundOk`). -/
def HonestRoundsPass (L : LeanRelease Flock.Wit Tp) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : finS.Nonce) (t : finS.Tape) (g : List (Msg × Msg)), 2 ≤ g.length → g.length ≤ refK x →
    L.refusesCoins x (ν, t) g = false → L.roundOk x w (ν, t) g = true

/-- **The honest worker is live on a history whose rounds' coins the frame took**, after `Hello` and fewer than `refK x`
openings, at a release whose `Finish` comes after the scheduled rounds, where its rounds pass the frame's checks
(`HonestRoundsPass`): `LeanRelease.toSrv_live_honest`. -/
theorem live_honest (L : LeanRelease Flock.Wit Tp) (x : Msg) (w : Flock.Wit) (hlast : refK x < L.last x)
    (hr : HonestRoundsPass L x w) (ν : finS.Nonce) (t : finS.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hlt : ops.length < refK x)
    (hc : ∀ i, 2 ≤ i → i ≤ ((m₀, ρ₀) :: ops).length → L.refusesCoins x (ν, t) (((m₀, ρ₀) :: ops).take i) = false) :
    L.toSrv.live L.honestWk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true := by
  refine L.toSrv_live_honest x w (ν, t) κ _ (by rw [List.length_cons]; omega) fun i h2 hi => ?_
  have hi' : i ≤ ops.length + 1 := by simpa using hi
  exact hr ν t _ (by rw [List.length_take]; omega) (by rw [List.length_take]; omega) (hc i h2 hi)

end Loop

namespace LeanRelease

/-- **`FillRoundEq` at a worker live wherever it reads**, at every `vs`: after every history of
`0 < ops.length < refK x` openings whose last coins the frame takes, the worker's run is live, and `FillRoundEq` reads
only histories on which it is not. -/
theorem fillRoundEq_of_live (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit Loop.Tp) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hl : ∀ (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg)
      (ops : List (Msg × Msg)), 0 < ops.length → ops.length < refK x →
      L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs L CoinOpen.shaH wk x w :=
  fun ν t κ m₀ ρ₀ ops hp hlt hf _ hr => (Bool.false_ne_true (hf.symm.trans (hl ν t κ m₀ ρ₀ ops hp hlt hr))).elim

/-- **`FillRoundEq` at the honest worker**, at every `vs`, the tables' own values among them, at a release whose
`Finish` comes after the scheduled rounds (at `ofLoopLink`, `loop_last`), where its rounds pass the frame's checks
(`Loop.HonestRoundsPass`): every history it reads takes all its rounds' coins, as the release's coin stop sticks
(`refusesCoins_of_take`), so the honest worker is live there (`Loop.live_honest`). -/
theorem fillRoundEq_honest (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit Loop.Tp) (x : Msg) (w : Flock.Wit) (hlast : refK x < L.last x)
    (hr : Loop.HonestRoundsPass L x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs L CoinOpen.shaH L.honestWk x w :=
  fillRoundEq_of_live vs L _ x w fun ν t κ m₀ ρ₀ ops _ hlt hc =>
    Loop.live_honest L x w hlast hr ν t κ m₀ ρ₀ ops hlt fun i _ _ =>
      Bool.eq_false_of_ne_true fun h => Bool.false_ne_true (hc.symm.trans (L.refusesCoins_of_take x (ν, t) _ i h))

/-- **`FillRoundEq` at the worker that sends the scheduled rounds with `vs'`'s values** (`Loop.valsWk vs' L`), at
every `vs`, at a release whose `Finish` comes after the scheduled rounds, where its rounds pass the frame's checks at
the device `Loop.valsDev vs' L` (`Loop.HonestRoundsPass` there): `fillRoundEq_honest`'s argument, its run live there by
`Loop.live_valsWk`. -/
theorem fillRoundEq_valsWk (vs vs' : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit Loop.Tp) (x : Msg) (w : Flock.Wit) (hlast : refK x < L.last x)
    (hr : Loop.HonestRoundsPass (L.withDevice (Loop.valsDev vs' L)) x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs L CoinOpen.shaH (Loop.valsWk vs' L) x w :=
  fillRoundEq_of_live vs L _ x w fun ν t κ m₀ ρ₀ ops _ hlt hc =>
    Loop.live_valsWk vs' L x w (ν, t) κ _ (by rw [List.length_cons]; omega) fun i h2 hi => by
      have hi' : i ≤ ops.length + 1 := by simpa using hi
      refine hr ν t _ (by rw [List.length_take]; omega) (by rw [List.length_take]; omega) ?_
      rw [refusesCoins_withDevice]
      exact Bool.eq_false_of_ne_true fun h =>
        Bool.false_ne_true (hc.symm.trans (L.refusesCoins_of_take x (ν, t) _ i h))

/-- **`FillRoundEq` at the honest worker of the guarantees**, `Loop.valsWk` at the tables' own values
(`RefSession.honestVals`), at every `vs`: `fillRoundEq_valsWk` there. -/
theorem fillRoundEq_honestVals (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit Loop.Tp) (x : Msg) (w : Flock.Wit) (hlast : refK x < L.last x)
    (hr : Loop.HonestRoundsPass
      (L.withDevice (Loop.valsDev (CoinLayer.session Loop.loopNr Masked.dataE).honestVals L)) x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs L CoinOpen.shaH
      (Loop.valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals L) x w :=
  fillRoundEq_valsWk vs _ L x w hlast hr

end LeanRelease

end Claims.ZKServer
