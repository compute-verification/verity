import Proofs.Flock.Claims.ZK.FinishSwMk
import Proofs.Flock.Claims.ZK.FrameTape
import Flock.Firewall.Finish

/-!
# `hmk` at the firewall's own request

`FinishSwMk`'s `swSrcE_hmk` and `swSrcE_l0` read the firewall's clear values by pad index (`valsFw`) and the pads from
the session's tape (`padsOf (UE.tapeZk τ)`). The firewall's `Finish` builds rep `r` of table `j` from its request
(`Finish.RepIn`): the values that passed the firewall (`vals`, `zp 64` words) and the frame's draws (`draws`), whose
pads are `Finish.padsOf`. So where the passed values are the request's, its draws are the release's frame draws on the
tape (`LeanRelease.drawsAt`), and the session's tape has, at that rep, the pads of the tape of those draws
(`tapeOfDraws`), `repCut_fw`'s `hmk` holds at the request's own `vals` and `padsOf draws`.

* `Finish.padsOf_h`: a request's pads are its draws' `h`, as `drawsPads` has them.
* `Masked.pads_frame`: the session's pads are then the request's below `h_ab` (`padsOf_get`, `padsOf_drawsAt`).
* **`Masked.swSrcE_hmk_frame`**: `hmk` at the switched source, at the firewall's request: below `zp 64`, rep `r`'s
  masked values are `q.vals[k]! + (Finish.padsOf q.draws).h[k]!`.
* **`Masked.swSrcE_l0_frame`**: from `zp 64` to `h_ab` they are level 0's opening's values plus the request's pads.
-/

/-- A request's pads are its draws' `h`. -/
theorem Flock.Firewall.Finish.padsOf_h (d : _root_.Flock.Firewall.Frame.Draws) :
    (_root_.Flock.Firewall.Finish.padsOf d).h = d.h := rfl

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **The session's pads are the request's below `h_ab`**: where `q`'s draws are the frame's on the tape `ρ` (of
`drawBytes` bytes, at table `j`'s layout) for frame rep `rep j r` of the release `L`, and the session's tape has, at that
rep, the pads of the tape of the frames' draws (`tapeOfDraws`), the session's pad array for table `j`'s rep `r`
(`padsOf`) is `Finish.padsOf q.draws`'s below `h_ab`. -/
theorem pads_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (q : _root_.Flock.Firewall.Finish.RepIn)
    (hq : q.draws = L.drawsAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep).h j.val r.val k) :
    ∀ k < (layE x j).hab, (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! =
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]! := by
  intro k hk
  have hlen : (layE x j).len = (layE x j).tw + 1 := rfl
  have htw : (layE x j).tw = (layE x j).hab + 3 := rfl
  rw [FlockComplete.padsOf_get (UE.tapeZk τ) j.val r.val (layE x j) k (by omega) (by omega) (by omega), htape k hk,
    ← FlockComplete.padsOf_get (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep) j.val r.val (layE x j) k
      (by omega) (by omega) (by omega),
    ← hlay, Claims.ZKServer.padsOf_drawsAt L x ρ t₀ rep j.val r.val hz, hq]
  rfl

/-- **`repCut_fw`'s `hmk` at the switched source, at the firewall's request `q`** for table `j`'s rep `r`, frame rep
`rep j r` of the release `L`: where the passed values there are `q`'s and the pads are the request's (`pads_frame`'s
premises), then where table `j`'s rows hold at its base, rep `r`'s masked values below `zp 64` are
`q.vals + (Finish.padsOf q.draws).h`. -/
theorem swSrcE_hmk_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (q : _root_.Flock.Firewall.Finish.RepIn)
    (hA : A j r = q.vals) (hq : q.draws = L.drawsAt x ρ (rep j.val r.val))
    (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep).h j.val r.val k)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    ∀ k < (layE x j).zp 64, ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).masked k =
      q.vals[k]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]! := by
  intro k hk
  rw [swSrcE_hmk x c w A B τ j r hrows k hk, hA,
    pads_frame L ρ t₀ rep x τ j r q hq hlay hz htape k (lt_trans hk (zp64_lt_hab x j))]

/-- **Level 0's opening at the switched source, at the firewall's request `q`**: at `pads_frame`'s premises, where table
`j`'s rows hold at its base, rep `r`'s masked values from `zp 64` to `h_ab` are `B j r` plus
`(Finish.padsOf q.draws).h`. -/
theorem swSrcE_l0_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (q : _root_.Flock.Firewall.Finish.RepIn)
    (hq : q.draws = L.drawsAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep).h j.val r.val k)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    ∀ k, (layE x j).zp 64 ≤ k → k < (layE x j).hab → ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).masked k =
      B j r k + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]! := by
  intro k hk0 hk
  rw [swSrcE_l0 x c w A B τ j r hrows k hk0 hk, pads_frame L ρ t₀ rep x τ j r q hq hlay hz htape k hk]

end Masked
end Claims.ZKMal
