module

public import Proofs.Flock.SessionLoop.SchedAt

/-!
# The switched session at a rep's level-0 rounds

At rep stream `2j + r`'s round `ligP0 + i`, `i ≤ k₀ + 3`, the switched session sends level 0's body `i`
(`LigLock.l0BodyAt`) at the opened coins from `ligP0`, the rep's basis (`Masked.ligBasis`) and the pads its source fixes
(`LigLock.padsL0`): the earlier reps return and the rep's PIOP returns (`Masked.runCut_cutL0_of_ret`).

* `LigLock.l0BodyAt_ne_nil`: every level-0 body ends with a squeeze.
* `Masked.runCut_l0Body`: where the coins show no refusal by round `ligP0 + i`, the cut session there sends body `i`.
  The earlier reps return (`repCut_ret`, their `β` rounds before the rep's start); the rep's rank check passes
  (`rank_at_unrefused`), so its PIOP returns (`piopCut_ret`).
* **`Loop.enc_l0`**, **`Loop.swMsgAt_l0`**: so the switched session's message there is that body, tagged `0`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.LigLock

open Flock

/-- **Every level-0 body ends with a squeeze**, so none is empty. -/
theorem l0BodyAt_ne_nil {coin : ℕ → ℕ → F128} {p0 : ℕ} {P : List UInt8} {T : F128} {cap : ByteArray}
    {sch : Schedule} {t b : Array F128} {lay : Zk.PadLayout} {h : Array F128} {rep : ℕ}
    {extra : ℕ → ℕ → F128} {nonce : ℕ → ℕ → UInt64} (i : ℕ) :
    l0BodyAt coin p0 P T cap sch t b lay h rep extra nonce i ≠ [] := by
  intro h'
  have hl := congrArg List.length h'
  unfold l0BodyAt l0Body at hl
  split_ifs at hl <;>
    simp only [List.length_append, Claims.ZKServer.enc_sq, FlockSoundness.Refine.Frame.header_length,
      List.length_nil] at hl <;> omega

end FlockComplete.LigLock

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

set_option maxHeartbeats 1000000 in
/-- **The cut session at rep `2j + r`'s level-0 round `i`, on the opened coins alone**: where the coins show no refusal
by round `ligP0 + i`, `i ≤ k₀ + 3`, the cut session stopped there sends level 0's body `i` at the opened coins from
`ligP0`, the rep's basis and the pads `padsL0` of the rep's source, at any table `t` and extra lanes `extra`. The
earlier reps return (`repCut_ret`: their `β` rounds come before the rep's start, `betaSlot_lt_repBase`); the rep's
rank check passes (`rank_at_unrefused`), so its PIOP returns (`piopCut_ret`); so `runCut_cutL0_of_ret`. -/
theorem runCut_l0Body {x : Msg} (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2) {i : ℕ} (hN : ¬ Refused x c (ligP0 x j r + i))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l))
    (hi : i ≤ (zkTab x j).sch.k0 + 3) (hK : ligP0 x j r + i ≤ refK x)
    (t : Array _root_.Flock.F128) (extra : ℕ → ℕ → _root_.Flock.F128) :
    ∃ P T cap, ligP0 x j r + i < (runCut x c (ligP0 x j r + i) spec src).2.size ∧
      (runCut x c (ligP0 x j r + i) spec src).2[ligP0 x j r + i]! =
        FlockComplete.LigLock.l0BodyAt (ligCoin c) (ligP0 x j r) P T cap (zkTab x j).sch t
          (ligBasis x c w τ j r extras) (layE x j)
          (FlockComplete.LigLock.padsL0 (ligCoin c) (ligP0 x j r) (zkTab x j).sch t (ligBasis x c w τ j r extras)
            (layE x j) extra r.val ((src j.val).rep r.val)) r.val extra (fun _ _ => 0) i := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have htab' : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stsOf x)[j.val]! := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val (by simpa [FlockDeploy.tabsOf] using hj),
      getElem!_pos (stsOf x) j.val hj]
    simp [FlockDeploy.tabsOf]
  have hmS : (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = (stsOf x)[j.val]!.m := by rw [htab']; rfl
  have hkS : (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog = (stsOf x)[j.val]!.c.kLog := by rw [htab']; rfl
  have hfS : (FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog = (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog := by
    rw [htab']; rfl
  have hm25 : 25 ≤ (stsOf x)[j.val]!.m := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact (hall _ hj).1
  have hT : FlockComplete.ZkTables.StmtT (stsOf x)[j.val]! := by
    rw [getElem!_pos (stsOf x) j.val hj]
    exact _root_.Flock.Readable.stmtT_of_scope hx.1 hsx hj
  obtain ⟨⟨hwf, -⟩, hslot⟩ := hT
  have hk6 := hwf.hk6
  have hkm := hwf.hk
  have hst : (stsOf x)[j.val]! = stAt x j := getElem!_pos _ _ _
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega, hst]
    exact (zkTab x j).hz
  have hes : extras.size = spec.points * (stsOf x)[j.val]!.regions.size := by
    rw [htab'] at hext
    rw [FlockComplete.JsonRT.extraClaims_size hext]
    simp [Nat.mul_comm, ptsE]
  have eP : FlockComplete.piopRoundsZk spec (stsOf x)[j.val]! =
      ((stsOf x)[j.val]!.m + (stsOf x)[j.val]!.c.kLog - 6) +
        (2 + spec.points * (stsOf x)[j.val]!.regions.size + 1) := rfl
  have hP0 : ligP0 x j r = repBase x (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stsOf x)[(2 * j.val + r.val) / 2]! := by
    rw [repBase_eq hsx]
    exact (repSlot_eq hsx (q := 2 * j.val + r.val) hz (fun _ => 0)).trans (Nat.add_zero _)
  rw [show (2 * j.val + r.val) / 2 = j.val by omega] at hP0
  have hR : rankSlot x j.val r.val =
      repBase x (2 * j.val + r.val) + ((stsOf x)[j.val]!.m + (stsOf x)[j.val]!.c.kLog - 6) := rfl
  have hrank0 : rankSlot x j.val 0 ≤ rankSlot x j.val r.val := by
    unfold rankSlot
    rw [repBase_eq hsx, repBase_eq hsx]
    have := FlockComplete.ZkTables.repStart_mono spec (stsOf x) (show 2 * j.val + 0 ≤ 2 * j.val + r.val by omega)
    omega
  have hK1 : repBase x (2 * j.val + r.val + 1) ≤ refK x := by
    rw [repBase_eq hsx, refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hβ := betaSlot_lt_repBase hx hsx (q := 2 * j.val + r.val) (by omega)
  rw [show (2 * j.val + r.val) / 2 = j.val by omega, show (2 * j.val + r.val) % 2 = r.val by omega] at hβ
  have eβ : betaSlot x j.val r.val = repBase x (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stsOf x)[j.val]! +
        ((zkTab x j).sch.k0 + 7 + 11 * ((zkTab x j).sch.levels.size - 1) + 2) := by
    unfold betaSlot
    rw [repSlot_eq hsx hz, ← repBase_eq hsx, show (2 * j.val + r.val) / 2 = j.val by omega]
  have hm14 : 14 ≤ (FlockDeploy.tabsOf (stsOf x))[j.val]!.m := by rw [hmS]; omega
  have hkm' : (FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog ≤ (FlockDeploy.tabsOf (stsOf x))[j.val]!.m := by
    rw [hfS, hkS, hmS]
    exact hkm
  obtain ⟨l, hl⟩ : ∃ l : Array Msg, l.size = repBase x (2 * j.val + r.val) :=
    ⟨Array.replicate _ [], Array.size_replicate⟩
  have hNb : l.size + 3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 6) +
      (3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6)) + (2 + extras.size) + 1 ≤ refK x := by
    rw [hfS, hkS, hmS, hl, hes]
    omega
  obtain ⟨res, t₃, l₃, hP, -⟩ := FlockComplete.CutFree.piopCut_ret (ligCoin c) (refK x) hm14 hkm'
    (fun a lw ρ hlw hρ => by
      rw [htab'] at hρ ⊢
      exact FlockComplete.JsonRT.fold_ok _ hk6 hslot a hlw hρ)
    (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0)
    ((src j.val).rep r.val) (FlockComplete.flat cms[j.val]!.cap) ic.root extras (priorE c j.val r.val)
    (FlockComplete.MaskRank.masksOf (stsOf x))[j.val]!
    (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8) l hNb
    (fun _ _ _ _ _ _ _ _ h₁ h₂ => by
      rw [FlockComplete.MaskRank.masksOf_getElem! hj]
      exact rank_at_unrefused hx hN hj hr2 (by omega) (by omega) (ligCoin c) (refK x) (fun _ _ _ => rfl)
        (FlockDeploy.tabsOf (stsOf x))[j.val]! hmS hkS hfS _ _ _ _
        (t₀ := _root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8) rfl hl h₁ h₂)
  have hkn : ∀ k < 2 * j.val + r.val, betaSlot x (k / 2) (k % 2) < ligP0 x j r + i := by
    intro k hk
    have h1 := betaSlot_lt_repBase hx hsx (q := k) (by omega)
    have h2 : repBase x (k + 1) ≤ repBase x (2 * j.val + r.val) := by
      rw [repBase_eq hsx, repBase_eq hsx]
      exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
    omega
  have e := runCut_cutL0_of_ret hx w τ c src hsx hs hcms j r
    (fun k hk => extra_ok hx hsx c (i := k / 2) (by omega))
    (fun k hk extras' hxe ic' _ l' hl' =>
      repCut_ret hx c hN hsx hs (k := k) (by omega) (hkn k hk) _ _ hxe ic' _ hl')
    hext hic hl hP t extra (n := ligP0 x j r + i) (Nat.le_add_right _ _) (by omega) hK
  rw [Nat.add_sub_cancel_left] at e
  refine ⟨_, _, _, ?_, e⟩
  by_contra hge
  rw [getElem!_neg (runCut x c (ligP0 x j r + i) spec src).2 (ligP0 x j r + i) hge] at e
  exact FlockComplete.LigLock.l0BodyAt_ne_nil i e.symm

end Masked

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V SessionSpec)

namespace Loop

/-- **The switched session's level-0 round `i` of rep stream `2j + r`**: where the coins show no refusal by round
`ligP0 + i`, `i ≤ k₀ + 3`, the session sends level 0's body `i` at the opened coins from `ligP0`, the tape's packed
table and extra lanes, the rep's basis and the switched source's `padsL0`, tagged `0` (`Masked.runCut_l0Body`: the
views' level-0 commitments and the pads commitment return quietly). -/
theorem enc_l0 (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (C : Fin (refK x) → finS.Coin) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hN : ¬ Masked.Refused x C (Masked.ligP0 x j r + i)) {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE C spec) = .ok extras)
    (hi : i ≤ (Masked.zkTab x j).sch.k0 + 3) (hK : Masked.ligP0 x j r + i ≤ refK x) :
    ∃ P T cap, finS.enc x C (Masked.ligP0 x j r + i) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) =
      0 :: FlockComplete.LigLock.l0BodyAt (Masked.ligCoin C) (Masked.ligP0 x j r) P T cap (Masked.zkTab x j).sch
        (Masked.ligTab x w t j) (Masked.ligBasis x C w t j r extras) (Masked.layE x j)
        (FlockComplete.LigLock.padsL0 (Masked.ligCoin C) (Masked.ligP0 x j r) (Masked.zkTab x j).sch
          (Masked.ligTab x w t j) (Masked.ligBasis x C w t j r extras) (Masked.layE x j)
          ((Masked.UE.tapeZk t).extra j.val) r.val ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val))
        r.val ((Masked.UE.tapeZk t).extra j.val) (fun _ _ => 0) i := by
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨schs, hs, -⟩ := Masked.schedsPZk_live hx
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hms : ∀ i < (FlockDeploy.tabsOf (Masked.stsOf x)).size,
      (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    intro i hi
    have hi' : i < (Masked.stsOf x).size := by simpa [FlockDeploy.tabsOf] using hi
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) i hi]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    exact (hall i hi').2.2
  let cmOf : ℕ → FlockComplete.Commit := fun i =>
    Classical.choose (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))
  have hcm : ∀ i, (FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
      _root_.Flock.Hashes V) _root_.Flock.MerkleScheme.hm96Sha512 (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i) : V _) = .ok (cmOf i) := fun i =>
    (Classical.choose_spec (FlockComplete.commit0Cut_cap (schs[i]!.1.levels.getD 0 default)
      (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))).1
  have hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i)).go l =
      (.ok ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).map cmOf), l) := by
    intro l
    have hq : FlockComplete.quietS.R
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash : _root_.Flock.Hashes V)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i))
        ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
          FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
              _root_.Flock.Hashes FlockComplete.PS)
            (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default)
            (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t i)) :=
      _root_.Flock.With.Sim.mapM_array FlockComplete.quietS _ fun i =>
        FlockComplete.CutEq.commit0Cut_quiet _ _ _ _
    rw [hq l, FlockComplete.mapM_ok_mem cmOf _ fun i hi => ?_]
    rw [hms i (by simpa using hi)]
    exact hcm i
  obtain ⟨ic, hic0⟩ := FlockComplete.padsTreeCut_ok
    (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
      _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
    ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val)
  obtain ⟨P, T, cap, hlt', hb⟩ := Masked.runCut_l0Body hx w t C (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t) hsx hs hcms
    j r hN hext (ic := ic) (fun l => by rw [FlockComplete.CutEq.padsTreeCut_quiet _ _ _ l, hic0]) hi hK
    (Masked.ligTab x w t j) ((Masked.UE.tapeZk t).extra j.val)
  have hsrc : Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x C) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) :=
    rfl
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [Masked.stsOf_half x j r]
    exact (Masked.zkTab x j).hz
  have hP0 : Masked.ligP0 x j r = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! :=
    (Masked.repSlot_eq hsx (q := 2 * j.val + r.val) hz (fun _ => 0)).trans (Nat.add_zero _)
  have hb1 : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) := by
    unfold FlockComplete.repStartZk
    omega
  have h0 : Masked.ligP0 x j r + i ≠ 0 := by omega
  refine ⟨P, T, cap, ?_⟩
  rw [enc_finS_of C h0 (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) hsx hN
    (by rw [← hsrc]; exact hlt'), ← hsrc, hb]

/-- **`enc_l0` at the session's message after the openings `cs`**: `swMsgAt` sends round `cs.length`. -/
theorem swMsgAt_l0 (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (cs : List finS.Coin) (hlt : cs.length < refK x) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : cs.length = Masked.ligP0 x j r + i)
    (hN : ¬ Masked.Refused x (finS.pad (K := refK x) cs) (Masked.ligP0 x j r + i))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE (finS.pad (K := refK x) cs) spec) =
      .ok extras)
    (hi : i ≤ (Masked.zkTab x j).sch.k0 + 3) :
    ∃ P T cap, finS.swMsgAt x (finS.wt x w) (vs x w ν) t cs =
      0 :: FlockComplete.LigLock.l0BodyAt (Masked.ligCoin (finS.pad (K := refK x) cs)) (Masked.ligP0 x j r) P T cap
        (Masked.zkTab x j).sch (Masked.ligTab x w t j) (Masked.ligBasis x (finS.pad (K := refK x) cs) w t j r extras)
        (Masked.layE x j)
        (FlockComplete.LigLock.padsL0 (Masked.ligCoin (finS.pad (K := refK x) cs)) (Masked.ligP0 x j r)
          (Masked.zkTab x j).sch (Masked.ligTab x w t j) (Masked.ligBasis x (finS.pad (K := refK x) cs) w t j r extras)
          (Masked.layE x j) ((Masked.UE.tapeZk t).extra j.val) r.val
          ((Masked.swSrcE x (finS.pad (K := refK x) cs) w (vs x w ν (finS.pad (K := refK x) cs) (finS.seenOf x t)) t j.val).rep r.val))
        r.val ((Masked.UE.tapeZk t).extra j.val) (fun _ _ => 0) i := by
  rw [swMsgAt_enc vs w ν t cs hlt, hn]
  exact enc_l0 vs w hx hsx ν t (finS.pad (K := refK x) cs) j r hN hext hi (by omega)

end Loop

end Claims.ZKServer
