import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.Claims.ZK.FrameLive

/-!
# `FillRoundEq` at the honest worker

`FillRoundEq` reads only histories on which the run is not live, so it holds at every `vs` for a worker live wherever
it reads (`LeanRelease.fillRoundEq_of_live`). The honest worker (`LeanRelease.honestWk`, the release's own device
rounds) is live on a history whose rounds' coins the frame all took, where each such round passes the frame's checks
(`Loop.HonestRoundsPass`, the frame's completeness for the honest device; `Loop.live_honest`). Where the frame refuses
a round's coins no device round passes them (`LeanRelease.roundOk_of_refusesCoins`), so that completeness reads only
taken coins.

The release's coin stop sticks as the session loop's does (`Flock.Firewall.Session`'s `State.refuse`): once it refuses
the coins after a prefix of a history it refuses the history's own (`LeanRelease.refusesCoins_of_take`). So a history
whose last coins the frame takes had every round's coins taken, and `FillRoundEq` holds at the honest worker
(`LeanRelease.fillRoundEq_honest`).
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

end LeanRelease

namespace Loop

/-- Completeness (the frame's, for the honest device, which Lean does not model; the honest session of
`verity/ml/flock/verifier/firewall_controls.sh` checks it under `FC_FIREWALL=1`). **Each honest round passes the frame's
checks where the frame takes its coins**: at `x` and `w`, after every history of `2` to `refK x` rounds whose last coins
the frame takes, the frame round passes the frame's checks at its honest device round (`LeanRelease.roundOk`). -/
def HonestRoundsPass (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : finS.Nonce) (t : finS.Tape) (g : List (Msg × Msg)), 2 ≤ g.length → g.length ≤ refK x →
    (LeanRelease.ofLoopLink L₀ src real).refusesCoins x (ν, t) g = false →
    (LeanRelease.ofLoopLink L₀ src real).roundOk x w (ν, t) g = true

/-- **The honest worker is live on a history whose rounds' coins the frame took**, after `Hello` and fewer than `refK x`
openings, where its rounds pass the frame's checks (`HonestRoundsPass`): `LeanRelease.toSrv_live_honest`, the `Finish`
being round `refK x + 1` (`LeanRelease.loop_last`). -/
theorem live_honest (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit)
    (hr : HonestRoundsPass L₀ src real x w)
    (ν : finS.Nonce) (t : finS.Tape) (κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hlt : ops.length < refK x)
    (hc : ∀ i, 2 ≤ i → i ≤ ((m₀, ρ₀) :: ops).length →
      (LeanRelease.ofLoopLink L₀ src real).refusesCoins x (ν, t) (((m₀, ρ₀) :: ops).take i) = false) :
    (LeanRelease.ofLoopLink L₀ src real).toSrv.live (LeanRelease.ofLoopLink L₀ src real).honestWk x w (ν, t) κ
      ((m₀, ρ₀) :: ops) = true := by
  have hlast : (LeanRelease.ofLoopLink L₀ src real).last x = refK x + 1 :=
    LeanRelease.loop_last Loop.loopNr Masked.dataE
      (LeanRelease.withFrames Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real) x
  refine (LeanRelease.ofLoopLink L₀ src real).toSrv_live_honest x w (ν, t) κ _
    (by rw [hlast, List.length_cons]; omega) fun i h2 hi => ?_
  have hi' : i ≤ ops.length + 1 := by simpa using hi
  exact hr ν t _ (by rw [List.length_take]; omega) (by rw [List.length_take]; omega) (hc i h2 hi)

end Loop

namespace LeanRelease

/-- **`FillRoundEq` at a worker live wherever it reads**, at every `vs`: after every history of
`0 < ops.length < refK x` openings whose last coins the frame takes, the worker's run is live, and `FillRoundEq` reads
only histories on which it is not. -/
theorem fillRoundEq_of_live (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hl : ∀ (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
      (ops : List (Msg × Msg)), 0 < ops.length → ops.length < refK x →
      (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
      (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w :=
  fun ν t κ m₀ ρ₀ ops hp hlt hf _ hr => (Bool.false_ne_true (hf.symm.trans (hl ν t κ m₀ ρ₀ ops hp hlt hr))).elim

/-- **`FillRoundEq` at the honest worker**, at every `vs`, the tables' own values among them, where its rounds pass the
frame's checks (`Loop.HonestRoundsPass`): every history it reads takes all its rounds' coins, as the release's coin
stop sticks (`refusesCoins_of_take`), so the honest worker is live there (`Loop.live_honest`). -/
theorem fillRoundEq_honest (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit)
    (hr : Loop.HonestRoundsPass L₀ src real x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH
      (ofLoopLink L₀ src real).honestWk x w :=
  fillRoundEq_of_live vs L₀ src real _ x w fun ν t κ m₀ ρ₀ ops _ hlt hc =>
    Loop.live_honest L₀ src real x w hr ν t κ m₀ ρ₀ ops hlt fun i _ _ =>
      Bool.eq_false_of_ne_true fun h =>
        Bool.false_ne_true (hc.symm.trans ((ofLoopLink L₀ src real).refusesCoins_of_take x (ν, t) _ i h))

end LeanRelease

end Claims.ZKServer
