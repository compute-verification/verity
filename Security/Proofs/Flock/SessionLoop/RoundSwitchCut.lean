module

public import Proofs.Flock.SessionLoop.SchedAt
public import Proofs.Flock.Claims.ZK.PiopFree

/-!
# The switched session at a rep's ring switch

At rep stream `2j + r`'s round `rankSlot + k`, `k ≤ |extras| + 2`, the switched session sends the ring switch's body `k`
(`Switch.rsBody`) at the switched source's switched vectors: its rank check passed, so its PIOP returned, having sent
the zerocheck's and lincheck's bodies and then the ring switch's.

* `FlockComplete.piopCut_rs_bodies`: a returned cut PIOP sent the ring switch's bodies after its zerocheck's and
  lincheck's, with nothing pending before them.
* `Masked.runCut_switchBody`: where the coins show no refusal by round `rankSlot + k`, the cut session there sends the
  ring switch's body `k` (the rank check passes there, `rank_at_unrefused`, so the PIOP returns, `piopCut_ret`).
* **`Loop.enc_switch`**, **`Loop.swMsgAt_switch`**: so the switched session's message there is that body, tagged `0`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete

open _root_.Flock

/-- **A returned cut PIOP sent the ring switch's bodies** after its zerocheck's `3 + (m − 6)` and lincheck's
`3 + (k_log − 6)` bodies: body `k ≤ |extras| + 2` is `rsBody` at the source's switched vectors, nothing pending before
the first (the lincheck ends on its squeeze). -/
theorem piopCut_rs_bodies {coin : ℕ → ℕ → F128} {n : ℕ} {st : Setup} (hm : 14 ≤ st.m) {lay : Zk.PadLayout}
    {src : RepSrc} {cap root : ByteArray} {extras : Array Claim} {prior : Array (List F128)} {words : List ℕ}
    {t₀ : Transcript} {l : Array Claims.Msg} {res t₃ l₃}
    (hN : l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + (2 + extras.size) + 1 ≤ n)
    (h : ((piopCut (asksCut coin n) st lay src cap root extras prior words).run t₀).go l = (.ok (res, t₃), l₃))
    {k : ℕ} (hk : k ≤ extras.size + 2) :
    l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + k < l₃.size ∧
      l₃[l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + k]! =
        Claims.ZKServer.Switch.rsBody src.sv (extras.size + 2) [] k := by
  obtain ⟨zp, lp, ss, T, sws⟩ := res
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, -, h₃⟩ := piopCut_split_rank h
  obtain ⟨-, hl₁⟩ := zerocheckCut_at coin n hm _ _ _ _ _ h₁
  obtain ⟨-, hl₂⟩ := lincheckCut_at coin n st.fold st.m lay src.masked zc h₂
  obtain ⟨res₂, t₂', L₂, e₂, -, hp₂⟩ := Claims.ZKServer.lincheckCut_bodies coin n st.fold st.m lay src.masked zc t₁ l₁
    (by omega) (RankAt.lincheckCut_fold coin n h₂)
  have e₂' := e₂.symm.trans h₂
  simp only [Prod.mk.injEq, Except.ok.injEq] at e₂'
  obtain ⟨⟨-, ht⟩, hL⟩ := e₂'
  rw [← ht, ← hL] at h₃
  rw [← hL] at hl₂
  have hsz : (abcClaims st.m st.kLog zc lc ++ extras).size = extras.size + 2 := by
    rw [Array.size_append, abcClaims_size]
    omega
  obtain ⟨res₃, t₃', L₃, e₃, hL₃, -⟩ := Claims.ZKServer.Switch.ringSwitchCut_bodies coin n src.sv
    (abcClaims st.m st.kLog zc lc ++ extras) (by rw [hsz]; omega) t₂' L₂ (by rw [hsz]; omega)
  have e₃' := e₃.symm.trans h₃
  simp only [Prod.mk.injEq, Except.ok.injEq] at e₃'
  obtain ⟨-, rfl⟩ := e₃'
  rw [hsz, hp₂, Claims.ZKServer.empty_toList] at hL₃
  have hL₃' : L₃.toList = L₂.toList ++ (List.range (extras.size + 2 + 1)).map
      (Claims.ZKServer.Switch.rsBody src.sv (extras.size + 2) []) ++ [] := by
    rw [List.append_nil]
    exact hL₃
  have hI : l.size + 3 + (st.m - 6) + (3 + (st.fold.kLog - 6)) + k = L₂.size + k := by omega
  refine ⟨?_, ?_⟩
  · have e := congrArg List.length hL₃
    simp only [Array.length_toList, List.length_append, List.length_map, List.length_range] at e
    omega
  · rw [hI]
    exact Claims.ZKServer.getElem!_of_toList_mid hL₃' (by omega)

end FlockComplete

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

set_option maxHeartbeats 1000000 in
/-- **The cut session at rep `2j + r`'s ring switch round `k`, on the opened coins alone**: where the coins show no
refusal by round `rankSlot + k`, `k ≤ |extras| + 2`, the cut session stopped there sends the ring switch's body `k` at
the rep's source's switched vectors. The rank check passes there (`rank_at_unrefused`), so the PIOP returns
(`piopCut_ret`) having sent it (`piopCut_rs_bodies`), and the session reaches it (`runCut_cutPiop_live`). -/
theorem runCut_switchBody {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin)
    (src : ℕ → FlockComplete.TabSrc) {s0 : _}
    {spec : _root_.Flock.SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash :
          _root_.Flock.Hashes FlockComplete.PS)
        (FlockDeploy.tabsOf (stsOf x))[i]!.merkle (schs[i]!.1.levels.getD 0 default) (src i)).go l = (.ok cms, l))
    (j : Fin (jOf x)) (r : Fin 2) {k : ℕ} (hN : ¬ Refused x c (rankSlot x j.val r.val + k))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((src j.val).rep r.val)).go l = (.ok ic, l))
    (hk : k ≤ extras.size + 2) :
    rankSlot x j.val r.val + k < (runCut x c (rankSlot x j.val r.val + k) spec src).2.size ∧
    (runCut x c (rankSlot x j.val r.val + k) spec src).2[rankSlot x j.val r.val + k]! =
      Claims.ZKServer.Switch.rsBody ((src j.val).rep r.val).sv (extras.size + 2) [] k := by
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
  have hβ := betaSlot_lt_repBase hx hsx (q := 2 * j.val + r.val) (by omega)
  rw [show (2 * j.val + r.val) / 2 = j.val by omega, show (2 * j.val + r.val) % 2 = r.val by omega] at hβ
  have eβ : betaSlot x j.val r.val = repBase x (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (stsOf x)[j.val]! +
        ((zkTab x j).sch.k0 + 7 + 11 * ((zkTab x j).sch.levels.size - 1) + 2) := by
    unfold betaSlot
    rw [repSlot_eq hsx hz, ← repBase_eq hsx, show (2 * j.val + r.val) / 2 = j.val by omega]
  have eP : FlockComplete.piopRoundsZk spec (stsOf x)[j.val]! =
      ((stsOf x)[j.val]!.m + (stsOf x)[j.val]!.c.kLog - 6) +
        (2 + spec.points * (stsOf x)[j.val]!.regions.size + 1) := rfl
  have hK : repBase x (2 * j.val + r.val + 1) ≤ refK x := by
    rw [repBase_eq hsx, refK_eq hsx, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono _ _ (by omega)
  have hR : rankSlot x j.val r.val =
      repBase x (2 * j.val + r.val) + ((stsOf x)[j.val]!.m + (stsOf x)[j.val]!.c.kLog - 6) := rfl
  have hrank0 : rankSlot x j.val 0 ≤ rankSlot x j.val r.val := by
    unfold rankSlot
    rw [repBase_eq hsx, repBase_eq hsx]
    have := FlockComplete.ZkTables.repStart_mono spec (stsOf x) (show 2 * j.val + 0 ≤ 2 * j.val + r.val by omega)
    omega
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
  obtain ⟨hlt, hbody⟩ := FlockComplete.piopCut_rs_bodies hm14 hNb hP hk
  have hI : rankSlot x j.val r.val + k = l.size + 3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 6) +
      (3 + ((FlockDeploy.tabsOf (stsOf x))[j.val]!.fold.kLog - 6)) + k := by
    rw [hR, hfS, hkS, hmS, hl]
    omega
  obtain ⟨hlt', he⟩ := runCut_cutPiop_live hx c src hsx hs hcms j r hN (by omega) hext hic hl hP
    (n := rankSlot x j.val r.val + k) (by omega) (by omega) (by omega)
  refine ⟨hlt', he.trans ?_⟩
  rw [hI]
  exact hbody

end Masked

end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V SessionSpec)

namespace Loop

/-- **The switched session's ring switch round `k` of rep stream `2j + r`**: where the coins show no refusal by round
`rankSlot + k`, `k ≤ |extras| + 2`, the session sends the ring switch's body `k` at the switched source's switched
vectors, tagged `0` (`Masked.runCut_switchBody`: the views' level-0 commitments and the pads commitment return
quietly). -/
theorem enc_switch (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (C : Fin (refK x) → finS.Coin) (j : Fin (Masked.jOf x)) (r : Fin 2) {k : ℕ}
    (hN : ¬ Masked.Refused x C (Masked.rankSlot x j.val r.val + k)) {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE C spec) = .ok extras)
    (hk : k ≤ extras.size + 2) :
    finS.enc x C (Masked.rankSlot x j.val r.val + k) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) =
      0 :: Switch.rsBody ((Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t j.val).rep r.val).sv (extras.size + 2) [] k := by
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
  have hrc := Masked.runCut_switchBody hx C (Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t) hsx hs hcms j r hN hext (ic := ic)
    (fun l => by rw [FlockComplete.CutEq.padsTreeCut_quiet _ _ _ l, hic0]) hk
  have hsrc : Masked.swSrcE x C w (vs x w ν C (finS.seenOf x t)) t =
      Masked.tabSrcCut Masked.PE x (Masked.czExe Masked.PE x C) (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) :=
    rfl
  have h0 : Masked.rankSlot x j.val r.val + k ≠ 0 := by
    have S := Masked.slots_order hx hsx hj hr2
    omega
  rw [enc_finS_of C h0 (finS.swCutOf x C (finS.wt x w) (vs x w ν C) t) hsx hN
    (by rw [← hsrc]; exact hrc.1), ← hsrc, hrc.2]

/-- **`enc_switch` at the session's message after the openings `cs`**: `swMsgAt` sends round `cs.length`. -/
theorem swMsgAt_switch (vs : finS.SwVals) (w : Flock.Wit) {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (ν : finS.Nonce)
    (t : finS.Tape) (cs : List finS.Coin) (hlt : cs.length < refK x) (j : Fin (Masked.jOf x)) (r : Fin 2) {k : ℕ}
    (hn : cs.length = Masked.rankSlot x j.val r.val + k)
    (hN : ¬ Masked.Refused x (finS.pad (K := refK x) cs) (Masked.rankSlot x j.val r.val + k))
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE (finS.pad (K := refK x) cs) spec) =
      .ok extras)
    (hk : k ≤ extras.size + 2) :
    finS.swMsgAt x (finS.wt x w) (vs x w ν) t cs =
      0 :: Switch.rsBody ((Masked.swSrcE x (finS.pad (K := refK x) cs) w (vs x w ν (finS.pad (K := refK x) cs) (finS.seenOf x t)) t
        j.val).rep r.val).sv (extras.size + 2) [] k := by
  rw [swMsgAt_enc vs w ν t cs hlt, hn]
  exact enc_switch vs w hx hsx ν t (finS.pad (K := refK x) cs) j r hN hext hk

end Loop

end Claims.ZKServer
