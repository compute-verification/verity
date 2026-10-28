import Proofs.Flock.Claims.ZK.FinishSwVals
import Proofs.Flock.Claims.ZK.FinishInnerAt

/-!
# The switched source's cut at its inner proof's first round, from the firewall's rep

`swSrcE_repCut_v` reads the masked values below `zp 64` at every pad, `final_c`'s included. Here `final_c`'s is left
out (`hmkV` but at `fc`): the firewall's rep returning places the cut from the switched source at the inner proof's
first round (`repCut_fw_inner_at`), which then carries the source's own masked `final_c`. The rest of the chain is
`swSrcE_repCut_v`'s; the inner proof's premise `hinner` is not needed, since the cut stops before the inner proof reads
anything past its first round.

* **`swSrcE_repCut_fc_v`**: `repCut_fw_inner_at` at the switched source at `v`.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf)

namespace Masked

set_option maxHeartbeats 1000000 in
/-- **`swSrcE_repCut_v` but at `final_c`, at the inner proof's first round**: where table `j`'s masked values below
`zp 64` but at `final_c`'s pad are the request's values plus its pads (`hmkV`) and, wherever the source's PIOP returns,
from `zp 64` to `h_ab` level 0's `l0Clear` at the coins plus the pads (`hl0`), the firewall's rep returning places the cut
from the switched source at `v`, answered after `CutEq.innerAt l.size` bodies, past its Ligerito opening with no pending
bytes, where it sends the inner proof's first body at the source's masked `final_c`. -/
theorem swSrcE_repCut_fc_v (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (hw : RZk x w) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (hAd : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) (n : ℕ)
    (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (extras : Array _root_.Flock.Claim) (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (q : _root_.Flock.Firewall.Finish.RepIn)
    (hmkV : ∀ k < (layE x j).zp 64, k ≠ (layE x j).fc → ((swSrcE x c w v τ j.val).rep r.val).masked k =
      q.vals[k]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]!)
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (hg : g = ligTab x w τ j) (cm0 : FlockComplete.Commit)
    (ic : FlockComplete.InnerCommit) (salt : ℕ → ℕ → ByteArray)
    (hsalt : ∀ i, 1 ≤ i → i < (zkTab x j).sch.levels.size → ∀ q < 2 ^ ((zkTab x j).sch.levels.getD i default).d,
      salt i q = (UE.tapeZk τ).salt j.val r.val i q)
    {t₀ : _root_.Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array Msg}
    (hl : l.size = repBase x (2 * j.val + r.val))
    (hl0 : ∀ res t₃ l₃, ((FlockComplete.piopCut (FlockComplete.asksCut (ligCoin c) n) st
        (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) ((swSrcE x c w v τ j.val).rep r.val)
        (FlockComplete.flat cm0.cap) ic.root extras (priorE c j.val r.val)
        (Composed.maskWordsOf (stsOf x)[j.val]!)).run t₀).go l = (.ok (res, t₃), l₃) →
      ∀ k, (layE x j).zp 64 ≤ k → k < (layE x j).hab → ((swSrcE x c w v τ j.val).rep r.val).masked k =
        FlockComplete.LigLock.l0Clear (ligCoin c) (ligP0 x j r) (zkTab x j).sch (ligTab x w τ j)
          (ligBasis x c w τ j r extras) (layE x j) ((UE.tapeZk τ).extra j.val) r.val k +
          (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]!)
    {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript} {l' : Array Msg}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (FlockComplete.asksCut (ligCoin c) n)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st (zkTab x j).sch (zkTab x j).tPad r.val dom g cm0
      extras (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic q.vals ((UE.tapeZk τ).extra j.val)
      ((UE.tapeZk τ).lanePad j.val) salt).run t₀).go l = (.ok (p, t), l')) :
    ∃ (t₄ : _root_.Flock.Transcript) (L : Array Msg), t₄.pending = ByteArray.empty ∧
      L.size = FlockComplete.CutEq.innerAt l.size st (zkTab x j).sch extras ∧
      ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) L.size)
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st (zkTab x j).sch r.val dom
          (FlockComplete.eraseC cm0) extras ic ((swSrcE x c w v τ j.val).rep r.val)
          (priorE c j.val r.val) (Composed.maskWordsOf (stsOf x)[j.val]!)).run t₀).go l =
        (.error "past the openings", L.push (Claims.toMsg (_root_.Flock.Transcript.framed
          (FlockComplete.Replay.InZ.t₁ (((swSrcE x c w v τ j.val).rep r.val).masked
            (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0).fc) t₄) 1 1))) := by
  subst hg
  have hnr : ¬ Refused x c n := not_refused_of_admits hx hAd n
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hstj : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm25 : 25 ≤ (stAt x j).m := (hall j.val hj).1
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hfk : (stAt x j).circuitFold.kLog = (stAt x j).c.kLog := rfl
  have hSm : st.m = (stAt x j).m := by
    subst hst
    rfl
  have hSk : st.kLog = (stAt x j).c.kLog := by
    subst hst
    rfl
  have hkL : st.fold.kLog = st.kLog := by
    subst hst
    rfl
  have hm14 : 14 ≤ st.m := by
    rw [hSm]
    omega
  have hmerk : st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    subst hst
    exact (hall j.val hj).2.2
  have hLyE : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = layE x j := by
    subst hst
    rfl
  have heS : (stAt x j).extraClaims (ptsE c spec) = .ok extras := by
    rw [tabsOf_stAt] at he
    exact he
  obtain ⟨hpts, -⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  have hpsz : (ptsE c spec).size = 2 := by
    unfold ptsE
    simp [hpts]
  have hesz : extras.size = spec.points * (stAt x j).regions.size := by
    rw [FlockDeploy.extraClaims_size heS, hpsz, hpts, Nat.mul_comm]
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  -- rep `r`'s PIOP: `m + k_log − 6` bodies of zerocheck and lincheck, then `2 + |extras| + 1` of ring switch
  have hP : ligP0 x j r =
      repBase x (2 * j.val + r.val) + ((stAt x j).m + (stAt x j).c.kLog - 6) + (2 + extras.size + 1) := by
    show repSlot x (2 * j.val + r.val) (fun _ => 0) = _
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val)]
    unfold FlockComplete.piopRoundsZk
    rw [hesz]
    omega
  -- the firewall's PIOP returned, so it sent all its bodies before the stop
  have hP0 : ligP0 x j r ≤ n := by
    obtain ⟨_, _, _, _, _, _, _, hpf⟩ := _root_.Flock.Firewall.Finish.repFwC_piopFw h
    obtain ⟨-, hl₃, hn₃⟩ := FlockComplete.piopFw_at hm14 hpf (trsE x w (UE.tapeZk τ))[j.val]!
    omega
  have hP0' := hP0
  rw [hP] at hP0'
  have hln : l.size ≤ n := by
    rw [hl]
    omega
  -- the masked values below `zp 64`
  have hmk : ∀ i < (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0).zp 64,
      i ≠ (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0).fc →
      ((swSrcE x c w v τ j.val).rep r.val).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := fun i hi hne =>
    hmkV i (by rw [← hLyE]; exact hi) (by rw [← hLyE]; exact hne)
  -- the switched vectors at the firewall's zerocheck and lincheck
  have hsv : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      ∀ k < (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size,
        ((swSrcE x c w v τ j.val).rep r.val).sv k =
          _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
            (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]! := by
    intro zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂ k hk
    obtain ⟨hzc, hl₁⟩ := FlockComplete.zerocheckCut_at (ligCoin c) n hm14 _ _ _ _ _ h₁
    obtain ⟨hlc, -⟩ := FlockComplete.lincheckCut_at (ligCoin c) n _ _ _ _ _ h₂
    have hk2 : k < 2 + extras.size := by
      rw [Array.size_append, FlockComplete.abcClaims_size] at hk
      exact hk
    subst hst
    set tr := (trsE x w (UE.tapeZk τ))[j.val]! with htr
    have hzE : FlockComplete.zcCoins (ligCoin c) l.size = zcE c j r := by
      rw [hl]
    have hlE : FlockComplete.lcCoins (ligCoin c) l₁.size
        ((_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog - 6) = lcE c j r := by
      rw [hl₁, hl]
      rfl
    rw [hzE] at hzc
    rw [hlE] at hlc
    have hab := FlockComplete.Replay.abc_x (m := (_root_.Flock.Setup.ofCircuit (stAt x j)).m)
      (kLog := (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog) (z₁ := zc) (l₁ := lc)
      (z₂ := FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
      (l₂ := FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
        (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) hzc.2.1 hzc.2.2 hlc.2
    have hxk := FlockComplete.Replay.claims_x (e := extras) (FlockComplete.abcClaims_size _ _ _ _)
      (FlockComplete.abcClaims_size _ _ _ _) hab.1 hab.2 k
    have hkE : k < (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
        (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
        (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) ++ extras).size := by
      rw [Array.size_append, FlockComplete.abcClaims_size]
      exact hk2
    have hmem : ∀ (a : Array _root_.Flock.Claim) (i : ℕ), i < a.size → a[i]! ∈ a := fun a i hi => by
      rw [getElem!_pos a i hi]
      exact Array.getElem_mem hi
    have hx6E := FlockComplete.Replay.claims_xsize' (stAt x j) hkm (by omega) tr (zcE c j r) (lcE c j r)
      (ptsE c spec) extras heS _ (hmem _ k hkE)
    have hx6 : ((_root_.Flock.abcClaims (_root_.Flock.Setup.ofCircuit (stAt x j)).m
        (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog zc lc ++ extras)[k]!).x.size = (stAt x j).m - 6 := by
      rw [hxk]
      exact hx6E
    have hpk : ligTab x w τ j = FlockComplete.packed tr (stAt x j).m := by
      show FlockComplete.packed tr (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = _
      rw [tabsOf_stAt]
      rfl
    rw [hpk, FlockComplete.switchVec_packed (by omega) tr _ hx6, FlockComplete.Replay.switchVec_x hxk]
    exact swSrc_sv c hw τ j r v hx hnr hs heS k hkE (by rw [hfk]; omega)
  -- the rank check
  have hpiop : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      FlockComplete.MaskRank.maskRankOk
          (priorE c j.val r.val ++ FlockComplete.MaskRank.roundPoints st.m st.kLog t₂.rounds)
          (Composed.maskWordsOf (stsOf x)[j.val]!) = true ∧
        ∀ k < (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size,
          ((swSrcE x c w v τ j.val).rep r.val).sv k =
            _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
              (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]! := fun zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂ =>
    ⟨rank_at_admits hx hAd hj r.isLt (ligCoin c) n (fun _ _ _ => rfl) st (by rw [hstj, hSm]) (by rw [hstj, hSk]) hkL
      _ _ _ _ ht₀ hl h₁ h₂, hsv zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂⟩
  have hSch : FlockComplete.CutEq.SchOk st (zkTab x j).sch := by
    subst hst
    exact schOk_of (zkTab x j)
  refine _root_.Flock.Firewall.Finish.repCut_fw_inner_at (ligCoin c) n st hmerk (zkTab x j).sch (zkTab x j).tPad
    r.val dom (ligTab x w τ j) cm0 extras (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic q.vals
    ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) salt
    ((swSrcE x c w v τ j.val).rep r.val) (priorE c j.val r.val)
    (Composed.maskWordsOf (stsOf x)[j.val]!) t₀ l hSch hln hkL hmk hpiop ?_ h
  -- level 0's opening: the firewall's PIOP ends at `ligP0` and switched `swsE`
  intro zp lp ss T sws t₃ l₃ h₃
  have hPC := _root_.Flock.Firewall.Finish.piopCut_fw_fc (ligCoin c) n st _ (ligTab x w τ j) _
    ((swSrcE x c w v τ j.val).rep r.val) _ _ extras (priorE c j.val r.val) (Composed.maskWordsOf (stsOf x)[j.val]!)
    hm14 (by exact Nat.le_refl _) (by rw [hkL]; exact Nat.le_refl _) hmk hln hpiop h₃
  obtain ⟨hsws, hl₃, -⟩ := FlockComplete.piopFw_at hm14 h₃ (trsE x w (UE.tapeZk τ))[j.val]!
  have hl3 : l₃.size = ligP0 x j r := by
    rw [hl₃, hl, hP, hkL, hSm, hSk, hesz]
    omega
  have hsE : sws = swsE x c w τ j r extras := by
    rw [hsws]
    unfold swsE
    rw [tabsOf_stAt, ← hst, hl, zcCoins_ligRespE, lcCoins_ligRespE, rsCoins_ligRespE]
  rw [hl3, hsE]
  have H := swSrcE_ligHyp_E_v x hx w hw τ c n hnr s spec hs j r extras he v q (hl0 _ _ _ hPC)
  unfold ligBasis at H
  rw [show ligSt x j = st from (tabsOf_stAt x j).trans hst.symm, ← hLyE] at H
  exact H.ofSalt hsalt

/-- **The inner proof's first round of table `j`'s rep `r`** is the session's slot `innerOff 0` past its PIOP: from the
rep's first round, `CutEq.innerAt` bodies at the session's table, schedule and extra claims. -/
theorem innerAt_slot {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin) {s : _}
    {spec : _root_.Flock.SessionSpec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x))
    (r : Fin 2) {extras : Array _root_.Flock.Claim}
    (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras) :
    FlockComplete.CutEq.innerAt (repBase x (2 * j.val + r.val)) (_root_.Flock.Setup.ofCircuit (stAt x j))
        (zkTab x j).sch extras =
      repSlot x (2 * j.val + r.val) (innerOff 0) := by
  have heS : (stAt x j).extraClaims (ptsE c spec) = .ok extras := by
    rw [tabsOf_stAt] at he
    exact he
  obtain ⟨hpts, -⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  have hpsz : (ptsE c spec).size = 2 := by
    unfold ptsE
    simp [hpts]
  have hesz : extras.size = spec.points * (stAt x j).regions.size := by
    rw [FlockDeploy.extraClaims_size heS, hpsz, hpts, Nat.mul_comm]
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm25 : 25 ≤ (stAt x j).m := (hall j.val hj).1
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hfm : (_root_.Flock.Setup.ofCircuit (stAt x j)).m = (stAt x j).m := rfl
  have hfk : (_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog = (stAt x j).c.kLog := rfl
  rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val)]
  unfold FlockComplete.piopRoundsZk innerOff
  simp only [FlockComplete.CutEq.innerAt, FlockComplete.CutEq.ligAt]
  rw [hesz, hfm, hfk]
  omega

end Masked
end Claims.ZKMal
