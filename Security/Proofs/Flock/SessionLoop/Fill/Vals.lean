import Proofs.Flock.SessionLoop.Fill.Sched
import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.SessionLoop.TakenVals
import Proofs.Flock.Claims.ZK.FrameLive

/-!
# What `vs` holds at a stopped round: the stand-in's values

Once a round of the session fails, the release sends each later round in the device's place as a stopped rep's
(`LeanRelease.fillOf`): the round the loop schedules (`Loop.schedOf`) or, at the inner proof's rounds, `L₀`'s
(`LeanRelease.scheduled_ofLoopLink`). `FillRoundEq` asks the switched session's message at `vs` to be that release, so it
asks `vs` to hold, at each such history's decoded coins (`Loop.finCoins`) and by pad index (`Loop.vsWord`), the values
the stopped round carries. `Loop.FillVals` says so in two parts:

* **`Loop.FillAbs`**: at each pad the round the loop schedules absorbs (`Frame.roundFree`, positions `m.pos` to `s'.pos`
  of the absorbed order), the value a stopped rep's frame absorbs there (`Masker.absorb`): `0` in the algebraic phase,
  which leaves as the pad alone, and past it level 0's opening's clear value at the coins (`LigLock.l0Clear`), which the
  firewall's own opening carries under the pads as a live rep's does.
* **`Loop.FillFc`**: at the inner proof's first round (`Loop.InnerRound … 0`), `final_c` as the frame takes it from the
  round the release sends: its `final_c'` plus `final_c`'s pad (`Loop.finalCOf`, as `Loop.valsOf` reads it).

It reads only stopped histories, so it holds at every `vs` for a worker that never stops a rep there
(`Loop.fillVals_of_live`), the honest worker wherever its rounds pass the frame's checks (`Loop.live_honest`,
`Loop.fillVals_honest`). So does `FillRoundEq` itself (`LeanRelease.fillRoundEq_of_live`,
`LeanRelease.fillRoundEq_honest`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Frame (Masker roundFree)
open _root_.Flock.Firewall.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

/-- **At a stopped round the loop schedules, `vs` holds the values a stopped rep's frame absorbs**: after every history
`FillRoundEq` reads at which the loop schedules its own round `r` (`schedOf`), where the frame takes the stream's coins at
the state `m` (`answered`) and `r`, as a stopped rep's round, leaves it at `s'` (`Frame.roundFree`), and the round is
table `j`'s rep `r'`'s (rep stream `2j + r'`), `vs`'s value at the decoded coins at the pad of each position `p` from
`m.pos` to `s'.pos` of the absorbed order is `0` below `nPiop` (the algebraic phase) and past it level 0's opening's
clear value at the coins (`LigLock.l0Clear`, at the table's extra claims at the link points). -/
def FillAbs (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : finS.Nonce) (t : finS.Tape) (κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (r : ByteArray),
    0 < ops.length → ops.length < refK x →
    (LeanRelease.ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk (finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true →
    (LeanRelease.ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = some r →
    ∀ (m : Masker) (out : ByteArray) (s' : Masker) (free : Array ℕ),
      (LeanRelease.ofLoopLink L₀ src real).answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m →
      roundFree ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := true } r
        ((LeanRelease.ofLoopLink L₀ src real).coins x ((m₀, ρ₀) :: ops).length) = .ok (out, s', free) →
      ∀ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (extras : Array _root_.Flock.Claim) (j : Fin (jOf x)) (r' : Fin 2),
        FlockDeploy.setupOf true x = some (s0, spec, stsOf x) →
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE (finCoins x ops) spec) = .ok extras →
        2 * j.val + r'.val = (LeanRelease.ofLoopLink L₀ src real).rep x ((m₀, ρ₀) :: ops).length →
        ∀ p, m.pos ≤ p → p < s'.pos →
          vsWord vs x w ν (finCoins x ops) t j r'
              ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[p]! =
            if p < ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).nPiop then 0
            else FlockComplete.LigLock.l0Clear (ligCoin (finCoins x ops)) (ligP0 x j r') (zkTab x j).sch
              (ligTab x w t j) (ligBasis x (finCoins x ops) w t j r' extras) (layE x j) ((UE.tapeZk t).extra j.val)
              r'.val ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[p]!

/-- **At the inner proof's first round, `vs`'s `final_c` is the one the frame takes from the round the release sends**:
after every history `FillRoundEq` reads at the inner proof's first round (`InnerRound … 0`), the round being table `j`'s
rep `r'`'s (rep stream `2j + r'`), `vs`'s value at the decoded coins at `final_c`'s pad is the round's `final_c'`
(`finalCOf`) plus that pad, as `valsOf` reads it. -/
def FillFc (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : finS.Nonce) (t : finS.Tape) (κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)),
    0 < ops.length → ops.length < refK x →
    (LeanRelease.ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = false →
    ((allOk (finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true →
    (LeanRelease.ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
    InnerRound (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) 0 →
    ∀ (j : Fin (jOf x)) (r' : Fin 2),
      2 * j.val + r'.val = (LeanRelease.ofLoopLink L₀ src real).rep x ((m₀, ρ₀) :: ops).length →
      vsWord vs x w ν (finCoins x ops) t j r'
          ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc =
        finalCOf ⟨((LeanRelease.ofLoopLink L₀ src real).scheduled x w (ν, t) ((m₀, ρ₀) :: ops)
            (computedOf ((LeanRelease.ofLoopLink L₀ src real).computed x w (ν, t) ((m₀, ρ₀) :: ops)))).toArray⟩ +
          ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.getD
            ((LeanRelease.ofLoopLink L₀ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc 0

/-- **`vs` holds the stand-in's values at the stopped rounds `FillRoundEq` reads**: the values a stopped rep's frame
absorbs at the rounds the loop schedules (`FillAbs`), and `final_c` at the inner proof's first round (`FillFc`). -/
def FillVals (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  FillAbs vs L₀ src real wk x w ∧ FillFc vs L₀ src real wk x w

/-- **`FillVals` holds at a worker that never stops a rep** where `FillRoundEq` reads, at every `vs`: it reads only
stopped histories. -/
theorem fillVals_of_live (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hl : ∀ (ν : finS.Nonce) (t : finS.Tape) (κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
      (ops : List (Msg × Msg)), 0 < ops.length → ops.length < refK x →
      (LeanRelease.ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) :
    FillVals vs L₀ src real wk x w :=
  ⟨fun ν t κ m₀ ρ₀ ops _ hp hlt hf => (Bool.false_ne_true (hf.symm.trans (hl ν t κ m₀ ρ₀ ops hp hlt))).elim,
    fun ν t κ m₀ ρ₀ ops hp hlt hf => (Bool.false_ne_true (hf.symm.trans (hl ν t κ m₀ ρ₀ ops hp hlt))).elim⟩

/-- **The honest worker is live wherever `FillRoundEq` reads**: where each frame round short of the `Finish` passes the
frame's checks at its honest device round (`LeanRelease.roundOk`, the frame's completeness for the honest device), the
honest worker's run (`LeanRelease.honestWk`, the release's own device rounds) is live after `Hello` and fewer than
`refK x` openings (`LeanRelease.toSrv_live_honest`, the `Finish` being round `refK x + 1`: `LeanRelease.loop_last`). -/
theorem live_honest (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit)
    (hr : ∀ (ν : finS.Nonce) (t : finS.Tape) (g : List (Msg × Msg)), 2 ≤ g.length → g.length ≤ refK x →
      (LeanRelease.ofLoopLink L₀ src real).roundOk x w (ν, t) g = true)
    (ν : finS.Nonce) (t : finS.Tape) (κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hlt : ops.length < refK x) :
    (LeanRelease.ofLoopLink L₀ src real).toSrv.live (LeanRelease.ofLoopLink L₀ src real).honestWk x w (ν, t) κ
      ((m₀, ρ₀) :: ops) = true := by
  have hlast : (LeanRelease.ofLoopLink L₀ src real).last x = refK x + 1 :=
    LeanRelease.loop_last Loop.loopNr Masked.dataE
      (LeanRelease.withFrames Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real) x
  refine (LeanRelease.ofLoopLink L₀ src real).toSrv_live_honest x w (ν, t) κ _
    (by rw [hlast, List.length_cons]; omega) fun i h2 hi => ?_
  exact hr ν t _ (by rw [List.length_take]; omega) (by rw [List.length_take]; simp only [List.length_cons] at hi; omega)

/-- **`FillVals` at the honest worker**, at every `vs`, the tables' own values (`RefSession.honestVals`) among them,
where its rounds pass the frame's checks (`live_honest`). -/
theorem fillVals_honest (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit)
    (hr : ∀ (ν : finS.Nonce) (t : finS.Tape) (g : List (Msg × Msg)), 2 ≤ g.length → g.length ≤ refK x →
      (LeanRelease.ofLoopLink L₀ src real).roundOk x w (ν, t) g = true) :
    FillVals vs L₀ src real (LeanRelease.ofLoopLink L₀ src real).honestWk x w :=
  fillVals_of_live vs L₀ src real _ x w fun ν t κ m₀ ρ₀ ops _ hlt => live_honest L₀ src real x w hr ν t κ m₀ ρ₀ ops hlt

end Loop

namespace LeanRelease

/-- **`FillRoundEq` at a worker that never stops a rep** where it reads, at every `vs`: it reads only stopped
histories. -/
theorem fillRoundEq_of_live (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hl : ∀ (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
      (ops : List (Msg × Msg)), 0 < ops.length → ops.length < refK x →
      (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w :=
  fun ν t κ m₀ ρ₀ ops hp hlt hf _ _ => (Bool.false_ne_true (hf.symm.trans (hl ν t κ m₀ ρ₀ ops hp hlt))).elim

/-- **`FillRoundEq` at the honest worker**, at every `vs`, the tables' own values among them, where its rounds pass the
frame's checks (`Loop.live_honest`): it never stops a rep. -/
theorem fillRoundEq_honest (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit)
    (hr : ∀ (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (g : List (Msg × Msg)), 2 ≤ g.length → g.length ≤ refK x →
      (ofLoopLink L₀ src real).roundOk x w (ν, t) g = true) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH
      (ofLoopLink L₀ src real).honestWk x w :=
  fillRoundEq_of_live vs L₀ src real _ x w fun ν t κ m₀ ρ₀ ops _ hlt =>
    Loop.live_honest L₀ src real x w hr ν t κ m₀ ρ₀ ops hlt

end LeanRelease

end Claims.ZKServer
