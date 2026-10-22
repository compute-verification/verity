import Proofs.Flock.Claims.ZK.FinishSwMk
import Proofs.Flock.Claims.ZK.FrameTape
import Flock.Firewall.Finish

/-!
# `hmk` at the firewall's own request

`FinishSwMk`'s `swSrcE_hmk` reads the workers' values from arrays by pad index and the pads from the session's tape
(`padsOf (UE.tapeZk τ)`). The firewall's `Finish` builds rep `r` of table `j` from its request (`Finish.RepIn`): the
values that passed the firewall (`vals`) and the frame's draws (`draws`), whose pads are `Finish.padsOf`. So where the
arrays are the request's passed values, its draws are the release's frame draws on the tape (`LeanRelease.drawsAt`), and
the session's tape has, at that rep, the pads of the tape of those draws (`tapeOfDraws`), `repCut_fw`'s `hmk` holds at
the request's own `vals` and `padsOf draws`.

* `Finish.padsOf_h`: a request's pads are its draws' `h`, as `drawsPads` has them.
* **`Masked.swSrcE_hmk_frame`**: `hmk` at the switched source, at the firewall's request: below `h_ab`, rep `r`'s masked
  values are `q.vals[k]! + (Finish.padsOf q.draws).h[k]!` (`padsOf_get`, `padsOf_drawsAt`).
-/

/-- A request's pads are its draws' `h`. -/
theorem Flock.Firewall.Finish.padsOf_h (d : _root_.Flock.Firewall.Frame.Draws) :
    (_root_.Flock.Firewall.Finish.padsOf d).h = d.h := rfl

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **`repCut_fw`'s `hmk` at the switched source, at the firewall's request `q`** for table `j`'s rep `r`, frame rep
`rep j r` of the release `L`: where the workers' arrays there are `q`'s passed values, `q`'s draws are the frame's on
the tape `ρ` (of `drawBytes` bytes, at table `j`'s layout), and the session's tape has, at that rep, the pads of the tape
of the frames' draws, then where table `j`'s rows hold at its base, rep `r`'s masked values below `h_ab` are
`q.vals + (Finish.padsOf q.draws).h`. -/
theorem swSrcE_hmk_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (q : _root_.Flock.Firewall.Finish.RepIn)
    (hA : A j r = q.vals) (hq : q.draws = L.drawsAt x ρ (rep j.val r.val))
    (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (htape : (UE.tapeZk τ).h j.val r.val =
      (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep).h j.val r.val)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsOfArr x c A j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    ∀ k < (layE x j).hab, ((swSrcE x c w (valsOfArr x c A) τ j.val).rep r.val).masked k =
      q.vals[k]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]! := by
  intro k hk
  have hlen : (layE x j).len = (layE x j).tw + 1 := rfl
  have htw : (layE x j).tw = (layE x j).hab + 3 := rfl
  rw [swSrcE_hmk x c w A τ j r hrows k hk, hA]
  congr 1
  rw [FlockComplete.padsOf_get (UE.tapeZk τ) j.val r.val (layE x j) k (by omega) (by omega) (by omega), htape,
    ← FlockComplete.padsOf_get (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep) j.val r.val (layE x j) k
      (by omega) (by omega) (by omega),
    ← hlay, Claims.ZKServer.padsOf_drawsAt L x ρ t₀ rep j.val r.val hz, hq]
  rfl

end Masked
end Claims.ZKMal
