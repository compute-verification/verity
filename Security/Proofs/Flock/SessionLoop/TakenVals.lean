import Proofs.Flock.SessionLoop.Statement
import Proofs.Flock.Claims.ZK.FinishSwMkAny
import Proofs.Flock.Claims.ZK.FinishSwFrame
import Proofs.Flock.Claims.ZK.FrameFree
import Proofs.Flock.Claims.ZK.L0PinLive
import Proofs.Flock.Claims.ZK.PadsRoot

/-!
# The values the frame took, as the switched session's values

The session loop's frame takes each value of a rep's algebraic phase as the free word the worker sent there, and
releases it masked, `v + h_i` under the rep's pad `h_i` (`Frame.Masker.absorb`); the loop reads the value back as the
released word plus the pad (`Loop.valsOf`). The switched session at the deployed tables (`CoinLayer.session Loop.loopNr
Masked.dataE`) masks the values `vs` gives at the opened coins the same way (`Masked.swSrcE_masked_pad_any`). This file
states the switched session's side by pad index, as the firewall's words:

* **`Loop.vsWord`**: table `j`'s rep `r` value `k` of `vs` at the opened coins, at the table's base on the tape (its
  mask slot and extra lanes, which a table's values may read), as the firewall's word; `0` from `n_E` on.
* **`Loop.RowsOK`**: each table's rows hold at `vs` at its base, at every nonce, opened coins and tape. Where they fail
  the switched session commits its pads tree over the moved pads (`Table.swPads`), while the frame commits it over the
  drawn ones, so a rep's first round, which carries the pads root, is the switched message only where they hold.
* **`Loop.TakesVals`**: at every live round, each word the frame frees in the worker's round is `vs`'s value there.
* **`Loop.TakesFinalC`**: at the inner proof's round 0, which the frame frees nothing of, the worker's `final_c'` plus
  its pad is `vs`'s `final_c`.
* **`Loop.swSrcE_masked_vsWord`**: below `n_E`, the switched source's masked value is `vsWord` plus the session's pad
  (`padsOf`), with no premise on the rows.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace Loop

/-- **`vs`'s value at pad `k`** of table `j`'s rep `r`, at the opened coins `c` and at table `j`'s base on the tape `τ`
(its mask slot and extra lanes), as the firewall's word; `0` from `n_E` on. -/
noncomputable def vsWord (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (k : ℕ) :
    _root_.Flock.F128 :=
  if hk : k < (tabC PE x j).z.nE then
    FlockLevel3.GF128.val (vs x w ν c j r (wtOf PE x w j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 ⟨k, hk⟩)
  else 0

/-- **The rows hold at `vs`**: at every nonce, opened coins and tape, each table's rows hold at `vs`'s values at the
table's base (`Table.RowsAt`). -/
def RowsOK (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)),
    ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, vs x w ν c j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1

/-- **The worker's free words are `vs`'s values**: after every history `LiveRoundEq` reads (`0 < ops.length < refK x`
openings, live, whose openings pass under the tape's nonce), where the frame takes the stream's coins (`answered`) and
the worker's round `v` of rep stream `2j + r`, each word the frame frees in `v` (`Frame.roundFree`: the algebraic phase's
values, `v`'s `k`-th at the absorbed order's place `m.pos + k`) is `vs`'s value at that pad index (`vsWord`), at the coins
the openings open and the tape. -/
def TakesVals (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (m : _root_.Flock.Firewall.Frame.Masker)
    (out : ByteArray) (s : _root_.Flock.Firewall.Frame.Masker) (fr : Array ℕ) (j : Fin (jOf x)) (r : Fin 2),
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

/-- **The worker's `final_c'` is `vs`'s `final_c` under its pad**: after every history `LiveRoundEq` reads, where the
worker's round `v` of rep stream `2j + r` is the inner proof's round 0 (its label frame, then the 16-byte `final_c'`
between its headers, the only shape `Frame.innerOwn` takes there), `final_c'` plus the rep's pad at `final_c`'s index
(the value the frame took there, `Loop.valsOf`) is `vs`'s value at that index (`vsWord`). -/
def TakesFinalC (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (v : Msg) (w' : ByteArray) (j : Fin (jOf x)) (r : Fin 2),
    0 < ops.length → ops.length < refK x → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true →
    ((allOk ((CoinLayer.session Loop.loopNr Masked.dataE).refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval
      CoinOpen.shaH).1 = true →
    wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false) →
    w'.size = 16 →
    v = (_root_.Flock.Firewall.Frame.labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ _root_.Flock.Transcript.header 3 1 1 ++
      w' ++ _root_.Flock.Transcript.header 4 1 1).data.toList →
    L.rep x (ops.length + 1) = 2 * j.val + r.val →
    finalCOf ⟨v.toArray⟩ + (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.getD (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc 0 =
      vsWord vs x w ν ((CoinLayer.session Loop.loopNr Masked.dataE).pad
        (ops.map fun p => (CoinLayer.session Loop.loopNr Masked.dataE).decOf p.2)) t j r
        (L.cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc

/-- **The switched source's masked value below `n_E` is `vsWord` plus the session's pad**, whatever the rows. -/
theorem swSrcE_masked_vsWord (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit)
    (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {k : ℕ}
    (hk : k < (tabC PE x j).z.nE) :
    ((swSrcE x c w (vs x w ν c) τ j.val).rep r.val).masked k =
      vsWord vs x w ν c τ j r k + (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! := by
  rw [swSrcE_masked_pad_any x c w (vs x w ν c) τ j r hk]
  unfold vsWord
  split
  · rfl
  · contradiction

end Loop

end Claims.ZKServer
