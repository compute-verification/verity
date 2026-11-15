module

public import Proofs.Flock.SessionLoop.L0Fixed
public import Proofs.Flock.Claims.ZK.L0Pin

/-!
# Level 0's bodies at the switched source, under `L0Fixed`

The switched session sends level 0's bodies at the pads `padsL0` its source fixes (`Masked.runCut_cutL0_of_ret`); the
frame releases them at the session's pads. Where `vs`'s level-0 values are the ZK server's (`Loop.L0Fixed`), the two
agree from `zp 64` to `h_ab`, the only pads a body reads:

* `LigLock.l0BodyAt_pads_congr`: level 0's bodies read their pads only from `zp 64` to `h_ab`.
* **`Loop.padsL0_vs`**: under `L0Fixed`, the switched source's `padsL0` at the opened coins, the tape's packed table and
  extra lanes and the rep's basis are the session's pads (`padsOf`) from `zp 64` to `h_ab`.
* **`Loop.l0BodyAt_vs`**: so level 0's bodies at those pads are its bodies at the session's pads.
-/

@[expose] public section

namespace FlockComplete.LigLock

open Flock

/-- **Level 0's bodies read their pads only from `zp 64` to `h_ab`**: the out-of-domain value's, the lane messages' and
the pair's sums'. -/
theorem l0BodyAt_pads_congr {coin : ℕ → ℕ → F128} {p0 : ℕ} {P : List UInt8} {T : F128} {cap : ByteArray}
    {sch : Schedule} {t b : Array F128} {lay : Zk.PadLayout} {h h' : Array F128} {rep : ℕ}
    {extra : ℕ → ℕ → F128}
    {nonce : ℕ → ℕ → UInt64} (hlk : lay.k0 = sch.k0)
    (hh : ∀ k, lay.zp 64 ≤ k → k < lay.hab → h[k]! = h'[k]!) (i : ℕ) :
    l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce i =
      l0BodyAt coin p0 P T cap sch t b lay h' rep extra nonce i := by
  have hk : ∀ k, lay.zp 64 ≤ k → k < lay.zp 64 + lay.nOod0 + 4 * lay.k0 + 4 → h[k]! = h'[k]! := hh
  unfold l0BodyAt l0Body
  split_ifs with h0 h1 h2 h3
  · rfl
  · rw [hk (lay.lgOod 0) (by simp only [Zk.PadLayout.lgOod]; omega)
      (by simp only [Zk.PadLayout.lgOod]; omega)]
  · simp only [l0Msg, maskMsg]
    rw [hk (lay.lgMsg (i - 2) 0) (by simp only [Zk.PadLayout.lgMsg]; omega)
        (by simp only [Zk.PadLayout.lgMsg]; omega),
      hk (lay.lgMsg (i - 2) 1) (by simp only [Zk.PadLayout.lgMsg]; omega)
        (by simp only [Zk.PadLayout.lgMsg]; omega),
      hk (lay.lgMsg (i - 2) 2) (by simp only [Zk.PadLayout.lgMsg]; omega)
        (by simp only [Zk.PadLayout.lgMsg]; omega),
      hk (lay.lgMsg (i - 2) 3) (by simp only [Zk.PadLayout.lgMsg]; omega)
        (by simp only [Zk.PadLayout.lgMsg]; omega)]
  · rw [hk (lay.lgE 0 0) (by simp only [Zk.PadLayout.lgE]; omega) (by simp only [Zk.PadLayout.lgE]; omega),
      hk (lay.lgE 0 1) (by simp only [Zk.PadLayout.lgE]; omega) (by simp only [Zk.PadLayout.lgE]; omega),
      hk (lay.lgE 1 0) (by simp only [Zk.PadLayout.lgE]; omega) (by simp only [Zk.PadLayout.lgE]; omega),
      hk (lay.lgE 1 1) (by simp only [Zk.PadLayout.lgE]; omega) (by simp only [Zk.PadLayout.lgE]; omega)]
  · rfl

end FlockComplete.LigLock

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace Loop

/-- **Under `L0Fixed` the switched source's level-0 pads are the session's**: from `zp 64` to `h_ab`, `padsL0` at the
opened coins from the rep's PIOP's end, the tape's packed table and extra lanes and the rep's basis is the source's
masked value (`vsWord` plus the session's pad, `swSrcE_masked_vsWord`) plus `l0Clear`, which is `vsWord` there. -/
theorem padsL0_vs {vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals} {x : Msg} {w : Flock.Wit}
    (hL0 : L0Fixed vs x w) (ν : CoinLayer.Nonce) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) {n : ℕ}
    (hn : ¬ Refused x c n) {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    {extras : Array _root_.Flock.Claim} (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {k : ℕ} (hlo : (layE x j).zp 64 ≤ k) (hhi : k < (layE x j).hab) :
    (FlockComplete.LigLock.padsL0 (ligCoin c) (ligP0 x j r) (zkTab x j).sch (ligTab x w τ j)
      (ligBasis x c w τ j r extras) (layE x j) ((UE.tapeZk τ).extra j.val) r.val
      ((swSrcE x c w (vs x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ)) τ j.val).rep r.val))[k]! =
      (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! := by
  unfold FlockComplete.LigLock.padsL0
  rw [FlockSoundness.Discharge.Composed.Zerocheck.get_map_range _ hhi]
  try dsimp only
  rw [swSrcE_masked_vsWord vs x w ν c τ j r (k := k) (by rw [nE_eq_hab]; exact hhi),
    hL0 ν τ c n hn s spec hs j r extras he k hlo hhi,
    _root_.Flock.F128.add_comm _ (FlockComplete.padsOf _ _ _ _).h[k]!,
    FlockSoundness.Discharge.Composed.ZerocheckMasked.add_add_self]

/-- **Under `L0Fixed` level 0's bodies at the switched source's pads are its bodies at the session's pads**
(`padsL0_vs`, `l0BodyAt_pads_congr`), at any pending bytes, transcript, cap and nonce. -/
theorem l0BodyAt_vs {vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals} {x : Msg} {w : Flock.Wit}
    (hL0 : L0Fixed vs x w) (ν : CoinLayer.Nonce) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) {n : ℕ}
    (hn : ¬ Refused x c n) {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    {extras : Array _root_.Flock.Claim} (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (P : List UInt8) (T : _root_.Flock.F128) (cap : ByteArray) (nonce : ℕ → ℕ → UInt64) (i : ℕ) :
    FlockComplete.LigLock.l0BodyAt (ligCoin c) (ligP0 x j r) P T cap (zkTab x j).sch (ligTab x w τ j)
      (ligBasis x c w τ j r extras) (layE x j)
      (FlockComplete.LigLock.padsL0 (ligCoin c) (ligP0 x j r) (zkTab x j).sch (ligTab x w τ j)
        (ligBasis x c w τ j r extras) (layE x j) ((UE.tapeZk τ).extra j.val) r.val
        ((swSrcE x c w (vs x w ν c ((CoinLayer.session loopNr Masked.dataE).seenOf x τ)) τ j.val).rep r.val))
      r.val ((UE.tapeZk τ).extra j.val) nonce i =
    FlockComplete.LigLock.l0BodyAt (ligCoin c) (ligP0 x j r) P T cap (zkTab x j).sch (ligTab x w τ j)
      (ligBasis x c w τ j r extras) (layE x j) (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h
      r.val ((UE.tapeZk τ).extra j.val) nonce i :=
  FlockComplete.LigLock.l0BodyAt_pads_congr rfl
    (fun _ hlo hhi => padsL0_vs hL0 ν τ c hn hs j r he hlo hhi) i

end Loop

end Claims.ZKServer
