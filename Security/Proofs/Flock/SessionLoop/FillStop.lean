import Proofs.Flock.SessionLoop.TakenVals

/-!
# `FillRoundEq` at the loop's own frames: the round in a stopped rep's place

`FillRoundEq` at `SessionLoopZKDeployed`'s release (`LeanRelease.ofLoopLink`) reads the round the release sends in a
stopped rep's place (`scheduled`): the loop's own round (`Loop.schedOf`) where it schedules one, and `L₀`'s at the inner
proof's rounds (`Loop.InnerAt`), as `LiveRoundEq` does (`LeanRelease.scheduled_ofLoopLink`, `Loop.schedOf_eq_none_iff`).
It reads only histories on which the worker's run is not live, at any worker.

* **`LeanRelease.outer_ne_one_taken`**: on any history of at most `refK x` openings whose last coins the frame takes,
  live or not, no zerocheck outer coin of a table at the coins the openings open is `1` (`outer_unrefused_taken`, the
  coin stop sticking). It is `release_sched`'s outer-coin hypothesis at a stopped rep's round, which no live run
  gives.
* `LeanRelease.FillSched`, `LeanRelease.FillInner`: `FillRoundEq`'s equation at the rounds the loop schedules
  (`schedOf`'s round, released as a stopped rep's) and at the inner proof's (`L₀`'s round, where `InnerAt` holds).
  **`LeanRelease.fillRoundEq_ofLoopLink`**: the two give `FillRoundEq`.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `Statement`

namespace LeanRelease

/-- **The opened coins meet `RowsOK`'s outer-coin condition on any history whose last coins the frame takes**: at a
statement the loop serves, on a history of at most `refK x` openings, live or not, no zerocheck outer coin of a table at
the coins the openings open is `1`. One whose slot is among the openings the frame refused at `1`
(`outer_unrefused_taken`); one past them is the default coin's `0`. -/
theorem outer_ne_one_taken
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    {x : Msg} (hSv : Loop.Serves x)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hr : (ofLoopLink L₀ src real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false) (j : Fin (Masked.jOf x)) :
    ∀ (r : Fin 2), ∀ y ∈ (Masked.tabC Masked.PE x j).rOuter (Masked.czExe Masked.PE x
      ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) j) r, y ≠ 1 := by
  intro r y hy
  obtain ⟨e, he, rfl⟩ := Masked.outerReads_czExe Masked.PE x _ j r y hy
  have hj := Masked.lt_of_jOf j
  rcases Nat.lt_or_ge (Masked.outerSlot x j.val r.val) ops.length with hb | hb
  · exact outer_unrefused_taken Masked.dataE (withFiles (withLink L₀ src) src) real x (Masked.live_of_lt j) hSv ρ
      m₀ ρ₀ ops hl hr hj r.isLt hb (by rw [getElem!_pos (Masked.stsOf x) j.val hj]; exact he)
  · unfold Masked.outerAt
    split
    · show ((ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2).getD
        (Masked.outerSlot x j.val r.val) default).at e ≠ 1
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by rw [List.length_map]; exact hb), Option.getD_none]
      unfold FlockDeploy.Coin.at
      split
      · show (0 : F128) ≠ 1
        decide
      · decide
    · decide

/-- **`FillRoundEq` at the rounds the loop schedules**: its equation after every history it reads at which the loop
schedules its own round `r` (`Loop.schedOf`), `r` released as a stopped rep's. -/
def FillSched (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (r : ByteArray),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    Loop.schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = some r →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) true r.data.toList =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- **`FillRoundEq` at the inner proof's rounds** (`Loop.InnerAt`): its equation after every history it reads at which the
loop schedules no round of its own, `L₀`'s round released as a stopped rep's. -/
def FillInner (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)),
    0 < ops.length → ops.length < refK x →
    (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) →
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) true
        (L₀.scheduled x w (ν, t) ((m₀, ρ₀) :: ops)
          (computedOf ((ofLoopLink L₀ src real).computed x w (ν, t) ((m₀, ρ₀) :: ops)))) =
      dropDom ((CoinLayer.session Loop.loopNr Masked.dataE).swMsgAt x
        ((CoinLayer.session Loop.loopNr Masked.dataE).wt x w) (vs x w ν) t
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2))

/-- **`FillRoundEq` from the scheduled rounds and the inner proof's**: at `SessionLoopZKDeployed`'s release, where the
loop schedules its own round (`FillSched`) and at the inner proof's rounds (`FillInner`). -/
theorem fillRoundEq_ofLoopLink (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hS : FillSched vs L₀ src real wk x w) (hI : FillInner vs L₀ src real wk x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w := by
  intro ν t κ m₀ ρ₀ ops hpos hlt hlive hok href
  rw [scheduled_ofLoopLink]
  split
  · rename_i r hsc
    exact hS ν t κ m₀ ρ₀ ops r hpos hlt hlive hok href hsc
  · rename_i hsc
    exact hI ν t κ m₀ ρ₀ ops hpos hlt hlive hok href
      ((Loop.schedOf_eq_none_iff (by omega) (by simpa using hpos)).mp hsc)

end LeanRelease

end Claims.ZKServer
