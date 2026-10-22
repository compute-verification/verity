import Proofs.Flock.Claims.ZK.FinishSwSrc
import Proofs.Flock.Claims.ZK.PiopSrcSv

/-!
# The switched source's switched vectors

`FinishSrc`'s `repCut_fw` asks of its source, at the firewall's zerocheck and lincheck, the switched vectors of the
claims (`hpiop`'s second conjunct). At the switched prover's source (`swSrcE`) they are the honest source's: the
switched pre-final view moves only pads, masked values and `Y₁`, and its `ŝ` (`ν + maskHom` at the mask slot) and region
claims read the witness and the mask slot alone.

* `Table.swPre_sHat`, `Table.swPre_sReg`: the switched pre-final view's `ŝ` and region claims are the honest view's, on
  either branch of `swPre` (`Table.onVals` copies `ν` and the region claims, and `Table.shift` keeps the mask slot).
* `Table.cut_sHat_congr`: a cut's `ŝ` reads the view's `ŝ` alone.
* `Masked.swCutE_sHat`, `Masked.swCutE_sReg`: so the switched cut's are the honest cut's.
* `Masked.repSrcCut_sv_congr`: a rep source's switched vectors read the cut's `ŝ` and region claims alone.
* `Masked.swSrcE_sv`: the switched source's switched vectors are the honest source's, at every table and rep.
* **`Masked.swSrc_sv`**: `src_sv` at the switched source, at every worker's values.
-/

set_option smartUnfolding false in
/-- The switched pre-final view's `ŝ` is the honest view's. -/
theorem FlockSoundness.ZK.Table.swPre_sHat {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    (T.swPre w v ω).sHat = (T.preView w ω).sHat := by
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1
  · rw [FlockSoundness.ZK.Table.swPre, if_pos hr]
    rfl
  · rw [FlockSoundness.ZK.Table.swPre, if_neg hr]
    rfl

set_option smartUnfolding false in
/-- The switched pre-final view's region claims are the honest view's. -/
theorem FlockSoundness.ZK.Table.swPre_sReg {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    (T.swPre w v ω).sReg = (T.preView w ω).sReg := by
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1
  · rw [FlockSoundness.ZK.Table.swPre, if_pos hr]
    rfl
  · rw [FlockSoundness.ZK.Table.swPre, if_neg hr]
    rfl

/-- A cut's `ŝ` reads the view's `ŝ` alone. -/
theorem FlockSoundness.ZK.Table.cut_sHat_congr {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (S : T.Sent) {P P' : T.PreView} (h : P.sHat = P'.sHat) :
    (T.cut S P).sHat = (T.cut S P').sHat := by
  unfold FlockSoundness.ZK.Table.cut
  rw [h]

/-- A cut's region claims are the view's. -/
theorem FlockSoundness.ZK.Table.cut_sReg {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (S : T.Sent) (P : T.PreView) :
    (T.cut S P).sReg = P.sReg := rfl

/-- The switched pre-final view's `ŝ`, leaves by position, is the honest one's at any leaves. -/
theorem FlockSoundness.Discharge.ZkView.Coined.swPreWith_sHat {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : FlockSoundness.Discharge.ZkView.Base z Hi)
    (lv lv' : FlockSoundness.Discharge.ZkView.LeafPos z → ByteArray) :
    (T.swPreWith w v b lv).sHat = (T.preWith w b lv').sHat :=
  T.tab.swPre_sHat w v (T.randOf b 0)

/-- The switched pre-final view's region claims, leaves by position, are the honest one's at any leaves. -/
theorem FlockSoundness.Discharge.ZkView.Coined.swPreWith_sReg {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : FlockSoundness.Discharge.ZkView.Base z Hi)
    (lv lv' : FlockSoundness.Discharge.ZkView.LeafPos z → ByteArray) :
    (T.swPreWith w v b lv).sReg = (T.preWith w b lv').sReg :=
  T.tab.swPre_sReg w v (T.randOf b 0)

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- The switched cut's `ŝ` is the honest cut's. -/
theorem swCutE_sHat (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).sHat = (cutE x c w τ j).sHat := by
  unfold swCutE cutE swCutAt cutAt
  exact Table.cut_sHat_congr _ _ (ZkView.Coined.swPreWith_sHat _ _ _ _ _ _)

/-- The switched cut's region claims are the honest cut's. -/
theorem swCutE_sReg (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).sReg = (cutE x c w τ j).sReg := by
  unfold swCutE cutE swCutAt cutAt
  simp only [Table.cut_sReg]
  exact ZkView.Coined.swPreWith_sReg _ _ _ _ _ _

/-- A rep source's switched vectors read the cut's `ŝ` and region claims alone. -/
theorem repSrcCut_sv_congr (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (cut cut' : ∀ j, ((tabC P x j).T (cz j)).tab.Cut) (j : Fin (jOf x)) (r : Fin 2)
    (h1 : (cut j).sHat = (cut' j).sHat) (h2 : (cut j).sReg = (cut' j).sReg) :
    (repSrcCut P x cz cut j r).sv = (repSrcCut P x cz cut' j r).sv := by
  funext k
  simp only [repSrcCut, h1, h2]

/-- **The switched source's switched vectors are the honest source's**, at every table and rep. -/
theorem swSrcE_sv (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (i r : ℕ) :
    ((swSrcE x c w v τ i).rep r).sv = ((srcE x c w τ i).rep r).sv := by
  unfold swSrcE srcE tabSrcCut
  by_cases hi : i < jOf x
  · by_cases hr : r < 2
    · simp only [dif_pos hi, dif_pos hr]
      exact repSrcCut_sv_congr _ _ _ _ _ _ _ (swCutE_sHat x c w v τ _) (swCutE_sReg x c w v τ _)
    · simp only [dif_pos hi, dif_neg hr]
  · simp only [dif_neg hi]

section Switch

variable {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {w : Claims.Flock.Wit} (hw : RZk x w) (τ : UE.Tape)
  (j : Fin (jOf x)) (r : Fin 2)

include hw

/-- **`src_sv` at the switched source**: wherever the rep's ring-switch body `k` is sent before the stop, the switched
source's switched vector `k` is the prover's of claim `k`, at every worker's values `v`. -/
theorem swSrc_sv (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (hx : Live x) {n : ℕ}
    (hn : ¬ Refused x c n) {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    {extras : Array _root_.Flock.Claim} (he : (stAt x j).extraClaims (ptsE c spec) = .ok extras) :
    ∀ k < (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
        (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
        (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m (trsE x w (UE.tapeZk τ))[j.val]!
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
          (lcE c j r)) ++ extras).size,
      repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 3 + ((stAt x j).circuitFold.kLog - 6) + k ≤ n →
      ((swSrcE x c w v τ j.val).rep r.val).sv k =
        FlockComplete.switchVec (trsE x w (UE.tapeZk τ))[j.val]!
          (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
            (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
            (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m (trsE x w (UE.tapeZk τ))[j.val]!
              (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!)
              (lcE c j r)) ++ extras)[k]! := by
  intro k hk hkn
  rw [swSrcE_sv]
  exact src_sv c hw τ j r hx hn hsx he k hk hkn

end Switch

end Masked
end Claims.ZKMal
