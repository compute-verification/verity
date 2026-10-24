import Proofs.Flock.Claims.ZK.InnerTauVal
import Proofs.Flock.Claims.ZK.InnerBatched
import Proofs.Flock.Claims.ZK.PiopSrcLc

/-!
# `TauRep`: the inner proof's first five rows

At `InnerAt`, `cons` is `Zk.replay`'s three constraints, the two ring-switch rows, then `ligeritoConstraints`. The
replay reads the masked proof `proofR` under the transcript's coins, and both are the model's: the proof's masked
values are the view's (`PiopSrcVals`, `PiopSrcLc`) and the coins are `czExe`'s (`Replay.replay_coins`). So rows 0–4
are `rowsOf`'s `rowCEval`, `rowAB`, `rowLC`, `rowRsAb` and `rowRsC` by coefficient.

* **`viewMasked`**: the view's masked message at a layout index is the cut source's.
* **`coinsAt_respOf`**, **`lcCoinsAt_respOf`**: the replay's zerocheck and lincheck coins on `respOf c` are `zcCoins`
  and `lcCoins` on `slotAt c`.
* **`zc_eq`**, **`lcRounds_eq`**, **`zp_eq`**: the masked proof's zerocheck, lincheck rounds and `z_partial` are
  `zcOf`, `lcRoundsOf` and `zPartialOf` on the view.
* **`folded_unique`**: two folds that realize one statement fold alike, so the executable's folded entries are
  `combvOf`.
* **`innerPre_cutE`**: the inner run answered by `asksCut` that returns is the one answered by `asksS`.
* **`tauRep`**: `TauRep`; **`innerTauVal_of_tauLig`**: `TauLig → InnerTauVal`.
-/

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf)
namespace Masked
open FlockComplete Flock.With

/-- The view's masked message `k` of rep `r` is the cut source's (`cutV_viewsAt`). -/
theorem viewMasked (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) (k : ℕ) :
    (Composed.msgAt ((viewE x c w τ j).pre.2.1 r) k).val = ((srcE x c w τ j.val).rep r.val).masked k := by
  have hV := (RefSession.cutV_viewsAt (session (fun _ _ => 0) (tableData PE QE UE)) x c (wtOf PE x w) τ).trans
    (cutOf_eq (fun _ _ => 0) x c w τ)
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show _ = (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (cutE x c w τ) j ⟨r.val, hr⟩
    else default).masked k
  rw [dite_eq_left r.isLt]
  show _ = (if h : k < ((tabC PE x j).T (czExe PE x c j)).tab.nE then
      FlockLevel3.GF128.val ((cutE x c w τ j).masked r ⟨k, h⟩) else 0)
  rw [← hV]
  unfold Composed.msgAt
  split_ifs with h₁ h₂
  all_goals first | rfl | (exfalso; contradiction) | exact FlockLevel3.GF128.zero_eq

/-- The replay's zerocheck coins from slot `L` on, answered by `respOf c`, are `zcCoins` on `slotAt c`. -/
theorem coinsAt_respOf {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (L : ℕ) :
    Replay.Zc.coinsAt (respOf c) L = zcCoins (fun k i => slotAt c k i) L := by
  simp only [Replay.Zc.coinsAt, zcCoins, coinOf_respOf]

/-- The same for the lincheck's coins (`lcCoins`). -/
theorem lcCoinsAt_respOf {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (L n : ℕ) :
    Replay.Lc.coinsAt (respOf c) L n = lcCoins (fun k i => slotAt c k i) L n := by
  simp only [Replay.Lc.coinsAt, lcCoins, coinOf_respOf]

/-- **The masked zerocheck proof** the prover sends is `zcOf` on the view's masked values. -/
theorem zc_eq {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {w : Claims.Flock.Wit} (hw : RZk x w) (τ : UE.Tape)
    {s spec} (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2) :
    Replay.ZcZ.proof (layE x j) (padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h (stAt x j).m (zcE c j r)
        (trsE x w (UE.tapeZk τ))[j.val]! =
      Composed.zcOf (zkTab x j).table (stAt x j) ((viewE x c w τ j).pre.2.1 r) := by
  unfold Replay.ZcZ.proof Composed.zcOf
  congr 1
  · refine Array.ext (by simp [Replay.ZcZ.r1ab]) fun i h1 h2 => ?_
    have hi : i < 64 := by simpa [Replay.ZcZ.r1ab] using h1
    simp only [Replay.ZcZ.r1ab, Array.getElem_map, Array.getElem_range]
    rw [viewMasked]
    exact (src_r1ab c hw τ hsx j r hi).symm
  · refine Array.ext (by simp [Replay.ZcZ.r1c]) fun i h1 h2 => ?_
    have hi : i < 64 := by simpa [Replay.ZcZ.r1c] using h1
    simp only [Replay.ZcZ.r1c, Array.getElem_map, Array.getElem_range]
    rw [viewMasked]
    exact (src_r1c c hw τ hsx j r hi).symm
  · refine (arr_eq_pushes (by simp) fun k hk => ?_).symm
    rw [getElem!_pos _ k (by simpa using hk), Array.getElem_map, Array.getElem_range, viewMasked, viewMasked]
    exact Prod.ext (src_ml0 c hw τ hsx j r hk) (src_ml1 c hw τ hsx j r hk)
  · rw [viewMasked]
    exact (src_fa c hw τ hsx j r).symm
  · rw [viewMasked]
    exact (src_fb c hw τ hsx j r).symm
  · rw [viewMasked]
    exact (src_fc c hw τ hsx j r).symm

/-- **The masked lincheck rounds** are `lcRoundsOf` on the view, where the circuit's fold succeeds. -/
theorem lcRounds_eq {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {w : Claims.Flock.Wit} (hw : RZk x w)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {full : Array _root_.Flock.F128}
    (hfold : (stAt x j).circuitFold.fold ((lcE c j r).ca 0)
      (_root_.Flock.skipWeights (Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)).z)
      ((Replay.Zc.out (zkTab x j).table.S.m (zcE c j r) ((trsE x w (UE.tapeZk τ))[j.val]!)).rho.extract 0
        ((stAt x j).circuitFold.kLog - 6)) = .ok full) :
    Replay.pushes (Replay.LcZ.msg (layE x j) (padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h (stAt x j).circuitFold
        (stAt x j).m (trsE x w (UE.tapeZk τ))[j.val]!
        (Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!) full (lcE c j r))
      ((stAt x j).circuitFold.kLog - 6) =
    Composed.lcRoundsOf (zkTab x j).table (stAt x j) ((viewE x c w τ j).pre.2.1 r) := by
  refine (arr_eq_pushes (by simp [Composed.lcRoundsOf]; rfl) fun k hk => ?_).symm
  have hk' : k < (stAt x j).c.kLog - 6 := hk
  rw [Composed.lcRoundsOf, getElem!_pos _ k (by simpa using hk'), Array.getElem_map, Array.getElem_range, viewMasked,
    viewMasked]
  exact Prod.ext (src_lc0 c hw τ j r hfold hk') (src_lc1 c hw τ j r hfold hk')

/-- **The masked `z_partial`** is `zPartialOf` on the view. -/
theorem zp_eq {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) {w : Claims.Flock.Wit} (hw : RZk x w)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) :
    Replay.LcZ.zp (layE x j) (padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h (stAt x j).circuitFold
        (stAt x j).m (trsE x w (UE.tapeZk τ))[j.val]!
        (Replay.Zc.out (stAt x j).m (zcE c j r) (trsE x w (UE.tapeZk τ))[j.val]!) (lcE c j r) =
      Composed.zPartialOf (zkTab x j).table (stAt x j) ((viewE x c w τ j).pre.2.1 r) := by
  refine Array.ext (by simp [Replay.LcZ.zp, Composed.zPartialOf]) fun i h1 h2 => ?_
  have hi : i < 64 := by simpa [Replay.LcZ.zp] using h1
  simp only [Replay.LcZ.zp, Composed.zPartialOf, Array.getElem_map, Array.getElem_range]
  rw [viewMasked]
  exact (src_zp c hw τ j r hi).symm

/-- `addConst` moves only the constant. -/
theorem addConst_a (l : _root_.Flock.Zk.Lin) (k : _root_.Flock.F128) : (l.addConst k).a = l.a := rfl

/-- Two folds that realize one statement fold the same arguments to the same array. -/
theorem folded_unique (S : FlockSoundness.Model.Statement) (cf cf' : _root_.Flock.CircuitFold)
    (hc : FlockSoundness.Refine.FoldRealizes cf S) (hc' : FlockSoundness.Refine.FoldRealizes cf' S)
    (α β : _root_.Flock.F128) (lw ρ ts w w' : Array _root_.Flock.F128) (hlw : lw.size = 64)
    (hρ : ρ.size = S.kLog - 6) (h : cf.folded α β lw ρ ts = .ok w) (h' : cf'.folded α β lw ρ ts = .ok w') :
    w = w' := by
  obtain ⟨v, hv, hvsz, rfl⟩ := hc.folded _ _ _ _ _ w h
  obtain ⟨v', hv', hvsz', rfl⟩ := hc'.folded _ _ _ _ _ w' h'
  have hvv : v = v' := by
    refine Array.ext (hvsz.trans hvsz'.symm) fun i h1 h2 => ?_
    rw [← getElem!_pos v i h1, ← getElem!_pos v' i h2]
    exact FlockSoundness.Refine.fOf_injective
      ((hc.fold α lw ρ v hlw hρ hv ⟨i, hvsz ▸ h1⟩).trans
        (hc'.fold α lw ρ v' hlw hρ hv' ⟨i, hvsz ▸ h1⟩).symm)
  rw [hvv, hc.pin, hc'.pin]

/-- **The inner run answered by `asksCut` that returns** is the one answered by `asksS` at the same coins
(`InnerBatched`'s private `innerPre_cut`, stated here for `TauRep` and `TauLig`). -/
theorem innerPre_cutE {coin : ℕ → ℕ → _root_.Flock.F128} {n : ℕ} {resp : ℕ → Msg}
    (hr : ∀ k, coinOf (resp k) = coin k) (hs : _root_.Flock.Hashes FlockComplete.PS) (st : _root_.Flock.Setup)
    (sch : _root_.Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : FlockComplete.Trace)
    (t : Array _root_.Flock.F128) (cm0 : Commit) (extras : Array _root_.Flock.Claim) (pads : PadsZk)
    (ic : InnerCommit) (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List _root_.Flock.F128)) (words : List ℕ) :
    (cutS.st _root_.Flock.Transcript).R
      (SimInner.innerPre (asksCut coin n) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words)
      (SimInner.innerPre (asksS resp) hs st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce
        prior words) := by
  unfold SimInner.innerPre piopPZk rankCheckP
  sim_walk [zerocheckPZk_cut hr, lincheckPZk_cut hr, ringSwitchP_cut hr, ligeritoPZk_cut hr, cutS_reflT, cutS_refl]

/-- **`TauRep`**: at `InnerAt`, `cons`' first five rows are `rowsOf`'s, each of the layout's length. The run is the
session's (`innerPre_cutE`), `innerPre_split` splits it into the replay and level 0, and the replay's rows
(`replay_zerocheck`, `replay_lincheck`, `replay_values`) read the masked proof (`zc_eq`, `lcRounds_eq`, `zp_eq`) under
`czExe`'s coins (`replay_coins`), with the folded entries `combvOf` (`folded_unique`). -/
theorem tauRep : TauRep := by
  intro x w τ c n j r cons L hA hn
  obtain ⟨hL, hw, hnr, s, spec, hsx, dom, cm0, ic, extras, hex, l, hl, o, t', l', hrun, rfl, -⟩ := hA
  have hj := lt_of_jOf j
  unfold innerPreE ioE innerPadsE innerLayE at hrun
  have hrun' := innerPre_cutE (coinOf_respOf c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ l _ _ hrun
  rw [tabsOf_ofCircuit x j] at hrun' hex
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hL.1 hsx
  have hm14 : 14 ≤ (_root_.Flock.Setup.ofCircuit (stAt x j)).m := by
    show 14 ≤ (stAt x j).m
    exact le_trans (by norm_num) (hall j.val hj).1
  have hmerk : (_root_.Flock.Setup.ofCircuit (stAt x j)).merkle = _root_.Flock.MerkleScheme.hm96Sha512 :=
    (hall j.val hj).2.2
  obtain ⟨full, lL, lg, zl, rec, tI, lI, rp, x', lgc, hfold, -, hlig, hre, hlgc, ho⟩ :=
    FlockComplete.ZkTables.innerPre_split (respOf c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hm14 _ l hrun'
  rw [ho]
  rw [hmerk] at hlig
  have hpre := (Replay.Rs.prefix_loop (i := 0) _ (Nat.zero_le _)).trans
    (Replay.rec_pre (JsonRT.ligeritoPZk_rec (respOf c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) _ _ _ _ hlig)
  have hR := (Replay.Agrees.self tI.rounds 0).of_prefix hpre
  obtain ⟨hz, hr, hρ, hα, hβ, hτ, hr'⟩ :=
    Replay.replay_coins (_root_.Flock.Transcript.init #[] dom) rfl (_root_.Flock.Transcript.init tI.rounds dom) rfl hR
  have h3 := ZkExec.replay_cons_size _ _ _ _ hre
  obtain ⟨h0, h1, -, -⟩ := Composed.ZerocheckMasked.replay_zerocheck _ _ _ _ hm14 hre
  obtain ⟨combv, hcomb, h2, habv, -⟩ := Composed.Lincheck.replay_lincheck _ _ _ _ hm14 hre
  obtain ⟨-, -, hcv⟩ := Composed.RingSwitch.replay_values _ _ _ _ hre
  have hS : (zkTab x j).table.S = FlockSoundness.Refine.stmtOf (stAt x j) (zkTab x j).wf.1
      (FlockSoundness.Refine.regionsOf (stAt x j)) := rfl
  have hlc : ZkTables.lcCoinsR (respOf c) (_root_.Flock.Setup.ofCircuit (stAt x j)) l.size = lcE c j r := by
    show Replay.Lc.coinsAt (respOf c) _ _ = _
    rw [lcCoinsAt_respOf, hl]
    rfl
  have ez : Composed.ZerocheckMasked.zAt (_root_.Flock.Transcript.init tI.rounds dom) =
      Composed.zcF (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    rw [hz, coinsAt_respOf, hl]
    rfl
  have er : _root_.Flock.pinned ++ Composed.ZerocheckMasked.rOuterAt (_root_.Flock.Transcript.init tI.rounds dom) =
      Composed.rF (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    rw [hr, coinsAt_respOf, hl, rF_czExe c hsx j r]
    rfl
  have eρ : Composed.ZerocheckMasked.rhoAt (_root_.Flock.Transcript.init tI.rounds dom)
      ((_root_.Flock.Setup.ofCircuit (stAt x j)).m - 6) =
      Composed.rhoF (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    rw [rhoF_out c j r (trsE x w (UE.tapeZk τ))[j.val]!]
    show _ = Replay.pushes (zcE c j r).ρ ((stAt x j).m - 6)
    exact Replay.rhoAt_eq _ _ _ fun k hk => by rw [hρ k hk, coinsAt_respOf, hl]
  have eρ' : Composed.Lincheck.rhoOf (_root_.Flock.Setup.ofCircuit (stAt x j)).m
      (_root_.Flock.Transcript.init tI.rounds dom) = Composed.rhoF (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    rw [rhoF_out c j r (trsE x w (UE.tapeZk τ))[j.val]!]
    show _ = Replay.pushes (zcE c j r).ρ ((stAt x j).m - 6)
    refine arr_eq_pushes (by simp [Composed.Lincheck.rhoOf]; rfl) fun k hk => ?_
    rw [Composed.Lincheck.rhoOf, getElem!_pos _ k (by rw [Array.size_ofFn]; exact hk), Array.getElem_ofFn, hρ k hk,
      coinsAt_respOf, hl]
  have eα : Composed.Lincheck.coinAt (_root_.Flock.Transcript.init tI.rounds dom)
      (3 + ((_root_.Flock.Setup.ofCircuit (stAt x j)).m - 6)) = (czExe PE x c j).rest.lcAlpha r := by
    rw [hα, hlc]
    show slotAt c (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6)) 0 =
      slotAt c (repBase x (2 * j.val + r.val) + (3 + ((stAt x j).m - 6))) 0
    exact congrArg (fun q => slotAt c q 0) (Nat.add_assoc _ _ _)
  have eβ : Composed.Lincheck.coinAt (_root_.Flock.Transcript.init tI.rounds dom)
      (4 + ((_root_.Flock.Setup.ofCircuit (stAt x j)).m - 6)) = (czExe PE x c j).rest.lcBeta r := by
    rw [hβ, hlc]
    show slotAt c (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 1) 0 =
      slotAt c (repBase x (2 * j.val + r.val) + (4 + ((stAt x j).m - 6))) 0
    exact congrArg (fun q => slotAt c q 0) (by omega)
  have eτ : Composed.Lincheck.tsOf (_root_.Flock.Setup.ofCircuit (stAt x j)).m
      (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog (_root_.Flock.Transcript.init tI.rounds dom) =
      Composed.tsF (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    rw [tsF_lc c j r]
    refine arr_eq_pushes (by simp [Composed.Lincheck.tsOf]; rfl) fun k hk => ?_
    rw [Composed.Lincheck.tsOf, getElem!_pos _ k (by rw [Array.size_ofFn]; exact hk), Array.getElem_ofFn, hτ k hk, hlc]
  have er' : Composed.Lincheck.coinAt (_root_.Flock.Transcript.init tI.rounds dom)
      (5 + ((_root_.Flock.Setup.ofCircuit (stAt x j)).m - 6) + ((_root_.Flock.Setup.ofCircuit (stAt x j)).kLog - 6)) =
      (czExe PE x c j).rest.rPrime r := by
    refine hr'.trans ?_
    rw [hlc]
    show slotAt c (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 2 + ((stAt x j).c.kLog - 6)) 0 =
      slotAt c (repBase x (2 * j.val + r.val) + (5 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6))) 0
    exact congrArg (fun q => slotAt c q 0) (by omega)
  rw [eα, eβ, show Composed.Lincheck.coinAt (_root_.Flock.Transcript.init tI.rounds dom) 2 =
    Composed.ZerocheckMasked.zAt (_root_.Flock.Transcript.init tI.rounds dom) from rfl, ez, eρ', eτ] at hcomb
  have hcv' : combv = Composed.combvOf (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    have hkm := (zkTab x j).table.S.hk
    refine folded_unique (zkTab x j).table.S _ (Composed.modelFold (zkTab x j).table.S)
      (hS ▸ FlockSoundness.Refine.stmtOf_fold (stAt x j) (zkTab x j).wf.1
        (FlockSoundness.Refine.regionsOf (stAt x j)))
      (Composed.modelFold_realizes _) _ _ _ _ _ _ _ (FlockSoundness.Refine.skipWeights_size _) ?_ hcomb rfl
    simp only [Composed.rhoF, Array.size_extract, Array.size_ofFn]
    show min ((zkTab x j).table.S.kLog - 6) ((zkTab x j).table.S.m - 6) - 0 = (zkTab x j).table.S.kLog - 6
    omega
  rw [hl] at hfold
  simp only [ZkTables.zcR, coinsAt_respOf, coinOf_respOf] at hfold
  set Ly := _root_.Flock.Zk.PadLayout.new (_root_.Flock.Setup.ofCircuit (stAt x j)).m
    (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog (zkTab x j).sch.k0 with hLy
  set P := ZkTables.proofR (respOf c) (_root_.Flock.Setup.ofCircuit (stAt x j)) (trsE x w (UE.tapeZk τ))[j.val]!
    l.size (zkTab x j).sch Ly (padsOf (UE.tapeZk τ) j.val r.val Ly).h cm0 full extras (eraseL lg) with hP
  have hpz : P.zerocheck = Composed.zcOf (zkTab x j).table (stAt x j) ((viewE x c w τ j).pre.2.1 r) := by
    rw [hP]
    simp only [ZkTables.proofR]
    rw [coinsAt_respOf, hl]
    exact zc_eq c hw τ hsx j r
  have hrd : P.lincheck.rounds = Composed.lcRoundsOf (zkTab x j).table (stAt x j) ((viewE x c w τ j).pre.2.1 r) := by
    rw [hP]
    simp only [ZkTables.proofR, Replay.LcZ.proof, ZkTables.zcR, ZkTables.lcCoinsR]
    rw [coinsAt_respOf, lcCoinsAt_respOf, hl]
    exact lcRounds_eq c hw τ j r hfold
  have hzp : P.lincheck.zPartial =
      Composed.zPartialOf (zkTab x j).table (stAt x j) ((viewE x c w τ j).pre.2.1 r) := by
    rw [hP]
    simp only [ZkTables.proofR, Replay.LcZ.proof, ZkTables.zcR, ZkTables.lcCoinsR]
    rw [coinsAt_respOf, lcCoinsAt_respOf, hl]
    exact zp_eq c hw τ j r
  have hLyE : Ly = innerLayE x j := (innerLayE_eq x j).symm
  rw [hpz, ez] at h0
  rw [hpz, ez, er, eρ] at h1
  rw [hpz, hrd, hzp, eα, eβ, eτ, hcv'] at h2
  rw [hzp, er'] at habv
  rw [hpz] at hcv
  have hsz5 : ∀ B : Array (String × _root_.Flock.Zk.Lin), B.size = 2 → (rp.cons ++ B).size = 5 := fun B hB => by
    rw [Array.size_append, h3, hB]
  refine ⟨?_, fun k hk => ?_⟩
  · rw [Array.size_append, hsz5 _ rfl]
    omega
  rw [ZkExec.append_get!_left _ _ (lt_of_lt_of_eq hk (hsz5 _ rfl).symm), ← hLyE]
  interval_cases k
  · rw [ZkExec.append_get!_left _ _ (by rw [h3]; omega), h0]
    exact ⟨Composed.size_rowCEval_form _ _ _, fun i => rfl⟩
  · rw [ZkExec.append_get!_left _ _ (by rw [h3]; omega), h1]
    exact ⟨Composed.size_rowAB_form _ _ _ _ _ _, fun i => rfl⟩
  · rw [ZkExec.append_get!_left _ _ (by rw [h3]; omega), h2]
    exact ⟨by simp [Composed.Lincheck.lcFinal, Replay.lcRunning_size], fun i => rfl⟩
  · rw [ZkExec.append_get!_right _ _ (j := 0) (by rw [h3]) (by simp)]
    show (rp.abValue.addConst _).a.size = Ly.len ∧
      ∀ i : Fin ((tabC PE x j).z.nE + 4), fOf (rp.abValue.addConst _).a[i.val]! = _
    rw [addConst_a, habv]
    exact ⟨Composed.ZerocheckMasked.size_combo _ _ _, fun i => rfl⟩
  · rw [ZkExec.append_get!_right _ _ (j := 1) (by rw [h3]) (by simp)]
    show (rp.cValue.addConst _).a.size = Ly.len ∧
      ∀ i : Fin ((tabC PE x j).z.nE + 4), fOf (rp.cValue.addConst _).a[i.val]! = _
    rw [addConst_a, hcv]
    exact ⟨by simp [_root_.Flock.Zk.Lin.masked], fun i => rfl⟩

/-- **`InnerTauVal` from level 0's lane rows.** -/
theorem innerTauVal_of_tauLig (hL : TauLig) : InnerTauVal := innerTauVal_of_rows tauRep hL

end Masked
end Claims.ZKMal
