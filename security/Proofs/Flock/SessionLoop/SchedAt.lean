module

public import Proofs.Flock.SessionLoop.CutPiopLive
public import Proofs.Flock.SessionLoop.PiopState
public import Proofs.Flock.SessionLoop.TakenVals
public import Proofs.Flock.SessionLoop.RepSrc
public import Proofs.Flock.SessionLoop.FinCommits
public import Proofs.Flock.SessionLoop.FinRounds

/-!
# A rep's scheduled round after one history

What `LiveSched`'s equation reads at each of a rep's scheduled rounds, after one history `(m₀, ρ₀) :: ops`:

* **`Loop.TakesValsAt`**: `TakesVals`'s conclusion at that history (`Loop.takesVals_iff`), the worker's free words
  there being `vs`'s values.
* **The switched session's message there** (`Loop.swMsgAt_enc`, `Loop.enc_finS_of`): before the openings' end,
  `swMsgAt` sends round `ops.length` of the session at the opened coins, which where the coins show no refusal by it is
  `sessionCut`'s body there, tagged `0`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (SessionSpec)

namespace Loop

/-- **The worker's free words are `vs`'s values after one history** `(m₀, ρ₀) :: ops` at the tape `(ν, t)` and the check
coins `κ`: `TakesVals`'s conclusion there, wherever `LiveRoundEq` reads the history (`0 < ops.length < refK x` openings,
live, whose openings pass under the tape's nonce), the frame takes the stream's coins and the worker's round is rep
stream `2j + r`'s. -/
def TakesValsAt (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce)
    (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) : Prop :=
  ∀ (v : Msg) (m : _root_.Flock.Firewall.Frame.Masker) (out : ByteArray) (s : _root_.Flock.Firewall.Frame.Masker)
    (fr : Array ℕ) (j : Fin (Masked.jOf x)) (r : Fin 2),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    L.answered x (ν, t) ((m₀, ρ₀) :: ops) = .ok m →
    _root_.Flock.Firewall.Frame.roundFree (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false } ⟨v.toArray⟩
      (L.coins x (ops.length + 1)) = .ok (out, s, fr) →
    L.rep x (ops.length + 1) = 2 * j.val + r.val →
    ∀ k < fr.size,
      (_root_.Flock.Firewall.Frame.words ⟨v.toArray⟩ fr[k]! 1)[0]! =
        vsWord vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
          (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r
          (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[m.pos + k]!

/-- `TakesVals` is `TakesValsAt` after every history. -/
theorem takesVals_iff (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) :
    TakesVals vs L wk x w ↔ ∀ ν t κ m₀ ρ₀ ops, TakesValsAt vs L wk x w ν t κ m₀ ρ₀ ops :=
  Iff.rfl

/-- **The session's round `n ≥ 1` at the deployed tables where the coins show no refusal by it**: `sessionCut`'s body
`n` on the cut, tagged `0`, where the cut session reached it (`enc_finS`). -/
theorem enc_finS_of {x : Msg} (c : Fin (refK x) → finS.Coin) {n : ℕ} (hn : n ≠ 0)
    (cut : ∀ j, ((finS.C x j).T (finS.cz x c j)).tab.Cut) {s : FlockDeploy.Ins} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, Masked.stsOf x)) (hR : ¬ Masked.Refused x c n)
    (hlt : n < (Masked.runCut x c n spec (Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x c) cut)).2.size) :
    finS.enc x c n cut =
      0 :: (Masked.runCut x c n spec (Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x c) cut)).2[n]! := by
  rw [enc_finS_eq]
  unfold Masked.encStop
  rw [ite_eq_right hR]
  unfold Masked.encExe
  rw [ite_eq_right hn, hsx]
  dsimp only
  rw [ite_eq_left hlt]

/-- **The switched session's message after the openings `cs`**, before the openings' end: its round `cs.length` at the
coins `cs` open, on the switched cut. -/
theorem swMsgAt_enc (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (ν : finS.Nonce) (t : finS.Tape)
    (cs : List finS.Coin) (hlt : cs.length < refK x) :
    finS.swMsgAt x (finS.wt x w) (vs x w ν) t cs =
      finS.enc x (finS.pad (K := refK x) cs) cs.length
        (finS.swCutOf x (finS.pad (K := refK x) cs) (finS.wt x w) (vs x w ν (finS.pad (K := refK x) cs)) t) := by
  unfold Claims.ZKMal.RefSession.swMsgAt
  exact ite_eq_left hlt

end Loop

end Claims.ZKServer
