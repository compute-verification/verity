import Proofs.Flock.Claims.ZK.FinishSwLig
import Proofs.Flock.Claims.ZK.FinishSwSv

/-!
# Level 0's opening at the switched source, at the coins

`FinishSwLig`'s `swSrcE_ligHyp_frame` leaves the switched source's restart `y₁'`, `ȳ` and levels ≥ 1's salts as
premises. The switched pre-final view moves only pads, masked values and `Y₁`; `y₁'` reads the witness lanes, the mask
slot and the extra lanes, `ȳ` the lane paddings, and levels ≥ 1 `y₁'` and the seeds (`Table.view`), none a clear value
or a pad. So those are the honest source's, and the honest source's are what the session computes at the coins
(`ligY1E`, `ligYbarE`, `ligSaltE`).

* `Table.swPre_y1`, `Table.swPre_ybar`, `Table.swPre_lig`: the switched pre-final view's are the honest view's, on
  either branch of `swPre` (`Table.onVals` copies the lanes and `lig`, and `Table.shift` moves the pads alone).
* `Masked.swCutE_y1`, `swCutE_ybar`, `swCutE_lig`: so the switched cut's are the honest cut's.
* `Masked.swSrcE_y1`, `swSrcE_ybar`, `swSrcE_hiSalt`: and the switched source's are the honest source's, at every table
  and rep.
* **`Masked.swSrcE_ligHyp_E`**: `LigHyp` at the switched source at the deployed rep's coins, the packed table, the
  basis of the PIOP's switched vectors and the tape's extra lanes, lane paddings and salts, at the request's pads,
  where level 0's part of the firewall's clear values is `l0Clear` there.
-/

set_option smartUnfolding false in
/-- The switched pre-final view's `y₁'` is the honest view's. -/
theorem FlockSoundness.ZK.Table.swPre_y1 {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    (T.swPre w v ω).y1 = (T.preView w ω).y1 := by
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_left hr]
    rfl
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_right hr]
    rfl

set_option smartUnfolding false in
/-- The switched pre-final view's `ȳ` is the honest view's. -/
theorem FlockSoundness.ZK.Table.swPre_ybar {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    (T.swPre w v ω).ybar = (T.preView w ω).ybar := by
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_left hr]
    rfl
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_right hr]
    rfl

set_option smartUnfolding false in
/-- The switched pre-final view's levels ≥ 1 are the honest view's. -/
theorem FlockSoundness.ZK.Table.swPre_lig {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    (T.swPre w v ω).lig = (T.preView w ω).lig := by
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_left hr]
    rfl
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_right hr]
    rfl

/-- A cut's `y₁'` reads the view's `y₁'` alone. -/
theorem FlockSoundness.ZK.Table.cut_y1_congr {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (S : T.Sent) {P P' : T.PreView} (h : P.y1 = P'.y1) :
    (T.cut S P).y1 = (T.cut S P').y1 := by
  unfold FlockSoundness.ZK.Table.cut
  rw [h]

/-- A cut's `ȳ` is the view's. -/
theorem FlockSoundness.ZK.Table.cut_ybar {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (S : T.Sent) (P : T.PreView) :
    (T.cut S P).ybar = P.ybar := rfl

/-- A cut's levels ≥ 1 read the view's alone. -/
theorem FlockSoundness.ZK.Table.cut_lig_congr {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (S : T.Sent) {P P' : T.PreView} (h : P.lig = P'.lig) :
    (T.cut S P).lig = (T.cut S P').lig := by
  unfold FlockSoundness.ZK.Table.cut
  rw [h]

theorem FlockSoundness.Discharge.ZkView.Coined.swPreWith_y1 {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : FlockSoundness.Discharge.ZkView.Base z Hi)
    (lv lv' : FlockSoundness.Discharge.ZkView.LeafPos z → ByteArray) :
    (T.swPreWith w v b lv).y1 = (T.preWith w b lv').y1 :=
  T.tab.swPre_y1 w v (T.randOf b 0)

theorem FlockSoundness.Discharge.ZkView.Coined.swPreWith_ybar {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : FlockSoundness.Discharge.ZkView.Base z Hi)
    (lv lv' : FlockSoundness.Discharge.ZkView.LeafPos z → ByteArray) :
    (T.swPreWith w v b lv).ybar = (T.preWith w b lv').ybar :=
  T.tab.swPre_ybar w v (T.randOf b 0)

theorem FlockSoundness.Discharge.ZkView.Coined.swPreWith_lig {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : FlockSoundness.Discharge.ZkView.Base z Hi)
    (lv lv' : FlockSoundness.Discharge.ZkView.LeafPos z → ByteArray) :
    (T.swPreWith w v b lv).lig = (T.preWith w b lv').lig :=
  T.tab.swPre_lig w v (T.randOf b 0)

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

theorem swCutE_y1 (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).y1 = (cutE x c w τ j).y1 := by
  unfold swCutE cutE swCutAt cutAt
  exact Table.cut_y1_congr _ _ (ZkView.Coined.swPreWith_y1 _ _ _ _ _ _)

theorem swCutE_ybar (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).ybar = (cutE x c w τ j).ybar := by
  unfold swCutE cutE swCutAt cutAt
  simp only [Table.cut_ybar]
  exact ZkView.Coined.swPreWith_ybar _ _ _ _ _ _

theorem swCutE_lig (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).lig = (cutE x c w τ j).lig := by
  unfold swCutE cutE swCutAt cutAt
  exact Table.cut_lig_congr _ _ (ZkView.Coined.swPreWith_lig _ _ _ _ _ _)

/-- A rep source's `y₁'`, `ȳ` and levels ≥ 1 salts read the cut's `y₁'`, `ȳ` and levels ≥ 1 alone. -/
theorem repSrcCut_restart_congr (P : Composed.LigProver) (x : Msg) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest)
    (cut cut' : ∀ j, ((tabC P x j).T (cz j)).tab.Cut) (j : Fin (jOf x)) (r : Fin 2)
    (h1 : (cut j).y1 = (cut' j).y1) (h2 : (cut j).ybar = (cut' j).ybar) (h3 : (cut j).lig = (cut' j).lig) :
    (repSrcCut P x cz cut j r).y1 = (repSrcCut P x cz cut' j r).y1 ∧
      (repSrcCut P x cz cut j r).ybar = (repSrcCut P x cz cut' j r).ybar ∧
      (repSrcCut P x cz cut j r).hiSalt = (repSrcCut P x cz cut' j r).hiSalt := by
  refine ⟨?_, ?_, ?_⟩ <;> simp only [repSrcCut, h1, h2, h3]

/-- **The switched source's `y₁'`, `ȳ` and levels ≥ 1 salts are the honest source's**, at every table and rep. -/
theorem swSrcE_restart (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (i r : ℕ) :
    ((swSrcE x c w v τ i).rep r).y1 = ((srcE x c w τ i).rep r).y1 ∧
      ((swSrcE x c w v τ i).rep r).ybar = ((srcE x c w τ i).rep r).ybar ∧
      ((swSrcE x c w v τ i).rep r).hiSalt = ((srcE x c w τ i).rep r).hiSalt := by
  unfold swSrcE srcE tabSrcCut
  by_cases hi : i < jOf x
  · by_cases hr : r < 2
    · simp only [dif_pos hi, dif_pos hr]
      exact repSrcCut_restart_congr _ _ _ _ _ _ _ (swCutE_y1 x c w v τ _) (swCutE_ybar x c w v τ _)
        (swCutE_lig x c w v τ _)
    · simp only [dif_pos hi, dif_neg hr]
      exact ⟨trivial, trivial, trivial⟩
  · simp only [dif_neg hi]
    exact ⟨trivial, trivial, trivial⟩

/-- **`repCut_fw`'s `hlig` at the switched source, at the deployed rep's coins**: at a live statement, a witness, a
tape, opened coins not refused by round `n`, the session's setup, table `j`, rep `r` and the table's extra claims, and
at `pads_frame`'s premises at the firewall's request `q`, where table `j`'s rows hold at its base and level 0's part of
the firewall's clear values is `l0Clear` at the coins from `ligP0`, the packed table and the basis of the PIOP's
switched vectors, rep `r`'s switched source holds level 0's opening (`LigHyp`) at the request's pads and the tape's
extra lanes, lane paddings and salts. -/
theorem swSrcE_ligHyp_E {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (hw : RZk x w) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hn : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (extras : Array _root_.Flock.Claim) (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (q : _root_.Flock.Firewall.Finish.RepIn)
    (hB : B j r = FlockComplete.LigLock.l0Clear (ligCoin c) (ligP0 x j r) (zkTab x j).sch (ligTab x w τ j)
      (ligBasis x c w τ j r extras) (layE x j) ((UE.tapeZk τ).extra j.val) r.val)
    (hq : q.draws = L.drawsAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep).h j.val r.val k)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    FlockComplete.LigLock.LigHyp (ligCoin c) n (ligP0 x j r) (ligSt x j).merkle (zkTab x j).sch (ligTab x w τ j)
      (ligBasis x c w τ j r extras) (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).h r.val (zkTab x j).tPad
      ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val)
      ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val) := by
  obtain ⟨h25, -, hsm, hk0, -, hdim, -⟩ := ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  obtain ⟨hn7, h2, -⟩ := hdim
  obtain ⟨ey1, eybar, esalt⟩ := swSrcE_restart x c w (valsFw x c A B) τ j.val r.val
  have hy1 := ligY1E x hx w hw τ c n hn s spec hs j r extras he
  have hybar := ligYbarE x hx w hw τ c n hn s spec hs j r extras he
  have hsalt := ligSaltE hiSaltE x hx w hw τ c n hn s spec hs j r extras he
  dsimp only at hy1 hybar hsalt
  refine swSrcE_ligHyp_frame L ρ t₀ rep x c w A B τ j r q (ligCoin c) n (ligP0 x j r) (ligSt x j).merkle
    (zkTab x j).sch (ligTab x w τ j) (ligBasis x c w τ j r extras) r.val (zkTab x j).tPad ((UE.tapeZk τ).extra j.val)
    ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val) hB hq hlay hz htape hrows hk0 (by omega) h2 rfl
    ?_ ?_ ?_
  · intro a b
    rw [ey1]
    exact hy1 a b
  · intro a b
    rw [eybar]
    exact hybar a b
  · intro a b
    rw [esalt]
    exact hsalt a b

end Masked
end Claims.ZKMal
