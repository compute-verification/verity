import Proofs.Flock.Claims.ZK.FinishSwLigE
import Proofs.Flock.Claims.ZK.FinishInner
import Proofs.Flock.Claims.ZK.InnerY

/-!
# The inner proof at the switched source

`repCut_fw`'s `hinner` asks the source for the inner proof's values (`InnerSrc`) at the firewall's pads. The switched
pre-final view keeps `ρ_in`, `σ_in` on either branch of `swPre` (`Table.shift` moves `h_tx`, `h_ty` against the exposed
pads), and `Y₂` (the padding's pads are not moved); where the rows hold its `Y₁` is the honest view's too. The firewall's
pads are the request's draws, which are the tape's wherever the tape's pads of table `j`'s rep `r` are the frame's.

* `Table.innerMsg_padShift`: the triple's two messages don't move under `padShift`.
* `Table.swPre_inner`, `Table.swPre_Y2`: the switched pre-final view's are the honest view's; `Table.swPre_Y1` where the
  rows hold.
* `Masked.swSrcE_inner`, `Masked.swSrcE_y`: so the switched source's `ρ_in`, `σ_in` are the honest source's, and its `Y`
  where table `j`'s rows hold at its base.
* `Masked.pads_frame_eq`: the firewall's pads (`Finish.padsOf` of the request's draws) are the honest pads.
* **`Masked.swSrcE_innerSrc_frame`**: `InnerSrc` at the switched source at the firewall's pads and the deployed rep's
  coins, at the inner proof's slot, from the batched constraint (which `repCut_fw` hands `hinner`: the firewall's own
  inner proof checked it, `innerP_batched`) and `τ`'s commitment.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- The triple's two messages `ρ_in`, `σ_in` are the same on pads moved by `padShift`. -/
theorem FlockSoundness.ZK.Table.innerMsg_padShift {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (p q : W) (u : T.Mask) (R : T.XLanes)
    (h : Fin 2 → (Fin T.nE → F) × F × F) (r : Fin 2) :
    T.innerMsg r ((h + T.padShift p q u R) r) = T.innerMsg r (h r) := by
  simp only [FlockSoundness.ZK.Table.innerMsg, Pi.add_apply, Prod.fst_add, Prod.snd_add,
    FlockSoundness.ZK.Table.padShift]
  refine Prod.ext ?_ ?_ <;> simp only [] <;> ring

set_option smartUnfolding false in
/-- The switched pre-final view's `ρ_in`, `σ_in` are the honest view's, on either branch of `swPre`. -/
theorem FlockSoundness.ZK.Table.swPre_inner {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    (T.swPre w v ω).inner = (T.preView w ω).inner := by
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_left hr]
    rfl
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_right hr]
    funext r
    exact T.onVals.innerMsg_padShift (w, v) (w, T.vals) ω.1 ω.2.1.2.1 ω.2.2.1 r

set_option smartUnfolding false in
/-- The switched pre-final view's `Y₂` is the honest view's. -/
theorem FlockSoundness.ZK.Table.swPre_Y2 {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    (T.swPre w v ω).Y2 = (T.preView w ω).Y2 := by
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_left hr]
    rfl
  · rw [FlockSoundness.ZK.Table.swPre, ite_eq_right hr]
    rfl

set_option smartUnfolding false in
/-- Where the rows hold, the switched pre-final view's `Y₁` is the honest view's. -/
theorem FlockSoundness.ZK.Table.swPre_Y1 {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand)
    (hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1) :
    (T.swPre w v ω).Y1 = (T.preView w ω).Y1 := by
  rw [FlockSoundness.ZK.Table.swPre, ite_eq_left hr]
  rfl

/-- A cut's `ρ_in`, `σ_in` are the view's. -/
theorem FlockSoundness.ZK.Table.cut_inner {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (S : T.Sent) (P : T.PreView) :
    (T.cut S P).inner = P.inner := rfl

/-- A cut's `Y₂` is the view's. -/
theorem FlockSoundness.ZK.Table.cut_Y2 {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (S : T.Sent) (P : T.PreView) :
    (T.cut S P).Y2 = P.Y2 := rfl

/-- A cut's `Y₁` reads the view's `Y₁` alone. -/
theorem FlockSoundness.ZK.Table.cut_Y1_congr {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (S : T.Sent) {P P' : T.PreView} (h : P.Y1 = P'.Y1) :
    (T.cut S P).Y1 = (T.cut S P').Y1 := by
  unfold FlockSoundness.ZK.Table.cut
  rw [h]

theorem FlockSoundness.Discharge.ZkView.Coined.swPreWith_inner {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : FlockSoundness.Discharge.ZkView.Base z Hi)
    (lv lv' : FlockSoundness.Discharge.ZkView.LeafPos z → ByteArray) :
    (T.swPreWith w v b lv).inner = (T.preWith w b lv').inner :=
  T.tab.swPre_inner w v (T.randOf b 0)

theorem FlockSoundness.Discharge.ZkView.Coined.swPreWith_Y2 {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : FlockSoundness.Discharge.ZkView.Base z Hi)
    (lv lv' : FlockSoundness.Discharge.ZkView.LeafPos z → ByteArray) :
    (T.swPreWith w v b lv).Y2 = (T.preWith w b lv').Y2 :=
  T.tab.swPre_Y2 w v (T.randOf b 0)

theorem FlockSoundness.Discharge.ZkView.Coined.swPreWith_Y1 {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (b : FlockSoundness.Discharge.ZkView.Base z Hi)
    (lv lv' : FlockSoundness.Discharge.ZkView.LeafPos z → ByteArray)
    (hr : T.tab.onVals.RowsAt (w, v) (T.randOf b 0).1 (T.randOf b 0).2.1.2.1) :
    (T.swPreWith w v b lv).Y1 = (T.preWith w b lv').Y1 :=
  T.tab.swPre_Y1 w v (T.randOf b 0) hr

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

theorem swCutE_inner (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).inner = (cutE x c w τ j).inner := by
  unfold swCutE cutE swCutAt cutAt
  simp only [Table.cut_inner]
  exact ZkView.Coined.swPreWith_inner _ _ _ _ _ _

theorem swCutE_Y2 (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).Y2 = (cutE x c w τ j).Y2 := by
  unfold swCutE cutE swCutAt cutAt
  simp only [Table.cut_Y2]
  exact ZkView.Coined.swPreWith_Y2 _ _ _ _ _ _

theorem swCutE_Y1 (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x))
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    (swCutE x c w v τ j).Y1 = (cutE x c w τ j).Y1 := by
  unfold swCutE cutE swCutAt cutAt
  exact Table.cut_Y1_congr _ _ (ZkView.Coined.swPreWith_Y1 _ _ _ _ _ _ hrows)

/-- Table `j`'s rep `r` switched source is `repSrcCut` of the switched cut. -/
theorem swSrcE_rep (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    (swSrcE x c w v τ j.val).rep r.val = repSrcCut PE x (czExe PE x c) (swCutE x c w v τ) j r := by
  unfold swSrcE tabSrcCut
  rw [dite_eq_left j.isLt]
  exact dite_eq_left r.isLt

/-- Table `j`'s rep `r` honest source is `repSrcCut` of the honest cut. -/
theorem srcE_rep (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) :
    (srcE x c w τ j.val).rep r.val = repSrcCut PE x (czExe PE x c) (cutE x c w τ) j r := by
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  exact dite_eq_left r.isLt

/-- **The switched source's `ρ_in`, `σ_in` are the honest source's.** -/
theorem swSrcE_inner (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    ((swSrcE x c w v τ j.val).rep r.val).rhoIn = ((srcE x c w τ j.val).rep r.val).rhoIn ∧
      ((swSrcE x c w v τ j.val).rep r.val).sigmaIn = ((srcE x c w τ j.val).rep r.val).sigmaIn := by
  have h := congrFun (swCutE_inner x c w v τ j) r
  rw [swSrcE_rep, srcE_rep]
  refine ⟨?_, ?_⟩ <;> simp only [repSrcCut, h]

/-- **Where table `j`'s rows hold at its base, the switched source's `Y` is the honest source's.** -/
theorem swSrcE_y (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) :
    ((swSrcE x c w v τ j.val).rep r.val).y = ((srcE x c w τ j.val).rep r.val).y := by
  have h1 := congrFun (swCutE_Y1 x c w v τ j hrows) r
  have h2 := congrFun (swCutE_Y2 x c w v τ j) r
  rw [swSrcE_rep, srcE_rep]
  first
    | (simp only [repSrcCut, h1, h2]; done)
    | (unfold repSrcCut; dsimp only; rw [h1, h2])

/-- Table `j`'s pads layout as the setup's and as the circuit's. -/
theorem innerLayE_layE (x : Msg) (j : Fin (jOf x)) : innerLayE x j = layE x j := by
  have hj := lt_of_jOf j
  have htab : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj)]
    simp [FlockDeploy.tabsOf]
    rfl
  show _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
    (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0 = _
  rw [htab]
  rfl

/-- **The firewall's pads are the honest pads**: at frame rep `rep j r` of the release `L`, on draws of `drawBytes`
bytes, where the tape's pads of table `j`'s rep `r` are those on the tape of the release's draws, the request's pads
(`Finish.padsOf` of its draws, as the analysis's) are the tape's at the table's layout. -/
theorem pads_frame_eq {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (q : _root_.Flock.Firewall.Finish.RepIn)
    (hq : q.draws = L.drawsAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (hpads : FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) =
      FlockComplete.padsOf (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep) j.val r.val (layE x j)) :
    (_root_.Flock.Firewall.Finish.padsOf q.draws).toC = FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) := by
  have hs : q.draws.tauSalt.size = _root_.Flock.Zk.TAU_SALT_BYTES := by
    rw [hq]
    exact Claims.ZKServer.tauSalt_size (L.lay x (rep j.val r.val)) (L.draws ρ (rep j.val r.val))
      (by rw [hz]; exact Claims.ZKServer.tau_le_drawBytes _)
  rw [hpads, ← hlay, Claims.ZKServer.padsOf_drawsAt L x ρ t₀ rep j.val r.val hz, ← hq]
  simp only [_root_.Flock.Firewall.Finish.padsOf, _root_.Flock.Firewall.Finish.PadsZk.toC,
    Claims.ZKServer.drawsPads]
  congr 1
  exact Claims.ZKServer.fitBytes_of_size _ _ hs

/-- The honest cut sends rep `r`'s `Y₁` once `β`'s round is seen. -/
theorem cutE_Y1_ne {x : Msg} (hx : Live x) {c : Fin (refK x) → FlockDeploy.Coin} {n : ℕ} (hn : ¬ Refused x c n)
    {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) (h : betaSlot x j.val r.val + 1 ≤ n) :
    (cutE x c w τ j).Y1 r ≠ none := by
  have hpass := famPasses_of (pointReads_czExe PE) (coinReads_czExe PE) hx hn hs j r .inner h
  have e : (cutE x c w τ j).Y1 r = some _ := ite_eq_left hpass
  rw [e]
  exact Option.some_ne_none _

/-- `β`'s round is two past the inner proof's slot. -/
theorem betaSlot_inner {x : Msg} (hx : Live x) {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x))
    (j : Fin (jOf x)) (r : Fin 2) :
    betaSlot x j.val r.val = repSlot x (2 * j.val + r.val) (innerOff 0) + 2 := by
  have hj := lt_of_jOf j
  obtain ⟨y⟩ := hx.2 j.val hj
  have hst : (stsOf x)[(2 * j.val + r.val) / 2]! = (stsOf x)[j.val] := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega]
    exact getElem!_pos (stsOf x) j.val hj
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  show repSlot x _ _ = repSlot x _ _ + 2
  rw [repSlot_eq hs hz, repSlot_eq hs hz]
  simp only [innerOff]
  omega

/-- **`repCut_fw`'s `hinner` at the switched source, at the deployed rep's coins**: at a live statement, a witness, a
tape, opened coins not refused by round `n` and the session's setup, at table `j`'s rep `r` inner proof begun at its
slot (`L₀`), where the firewall's pads are the request's draws and the tape's pads are those draws' (`pads_frame_eq`'s
premises) and table `j`'s rows hold at its base, the switched source carries the inner proof's values (`InnerSrc`) at
the firewall's pads and the constraints `cons`, given the batched constraint and `τ`'s commitment there. -/
theorem swSrcE_innerSrc_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hn : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (q : _root_.Flock.Firewall.Finish.RepIn)
    (hq : q.draws = L.drawsAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.Firewall.Frame.drawBytes (L.lay x (rep j.val r.val)))
    (hpads : FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) =
      FlockComplete.padsOf (Claims.ZKServer.tapeOfDraws t₀ (L.lay x) (L.draws ρ) rep) j.val r.val (layE x j))
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    (cons : Array (String × _root_.Flock.Zk.Lin)) {L₀ : ℕ} (hL : L₀ = repSlot x (2 * j.val + r.val) (innerOff 0))
    (hbat : L₀ + 1 < n →
      ((FlockComplete.Replay.InZ.lin (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC cons
          (FlockComplete.SimInner.cutCoins (ligCoin c) L₀)).c +
        _root_.Flock.dot (FlockComplete.Replay.InZ.lin (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC
          cons (FlockComplete.SimInner.cutCoins (ligCoin c) L₀)).a
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h).isZero = true)
    (htau : L₀ + 1 < n → ((swSrcE x c w v τ j.val).rep r.val).tauCom =
      FlockComplete.Replay.InZ.tcOf _root_.Flock.Sha512.hash
        (FlockComplete.Replay.InZ.tau (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC cons
          (FlockComplete.SimInner.cutCoins (ligCoin c) L₀))
        (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.tauSalt) :
    FlockComplete.SimInner.InnerSrc (ligCoin c) n _root_.Flock.Sha512.hash (layE x j)
      (_root_.Flock.Firewall.Finish.padsOf q.draws).toC cons ((swSrcE x c w v τ j.val).rep r.val) L₀ := by
  have hP := pads_frame_eq L ρ t₀ rep x τ j r q hq hlay hz hpads
  have hβ := betaSlot_inner hx hs j r
  refine ⟨?_, hbat, htau, ?_⟩
  · intro _
    rw [hP, ← innerLayE_layE, (swSrcE_inner x c w v τ j r).1, (swSrcE_inner x c w v τ j r).2]
    exact innerTripleE x c w τ j r hL
  · intro hlt _
    rw [hP, ← innerLayE_layE, swSrcE_y x c w v τ j r hrows]
    exact innerYE hs c w τ j r hL (cutE_Y1_ne hx hn hs w τ j r (by omega))

end Masked
end Claims.ZKMal
