import Proofs.Flock.SessionLoop.FinRepMsgsFw
import Proofs.Flock.SessionLoop.FinPiopRet
import Proofs.Flock.SessionLoop.FinL0Vals
import Proofs.Flock.Claims.ZK.FinishFwL0
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictForms

/-!
# `FinRepLanes` with no rows

`FinRepRows` consumes `Loop.FinRepLanes`: at table `j`'s rep `r`'s run, the returned proof's level 0 (its
out-of-domain value, lane messages and pair's sums) is the workers' clear values plus the request's pads. Here it holds
outright, reading no table's rows:

* the firewall computes level 0 itself: its values are `l0Clear` at the coins from the round its PIOP ends at, on its
  packed table and its PIOP's basis, plus the request's pads (`Finish.repFwC_l0`); that PIOP ends at `ligP0` and its
  basis is `ligBasis`'s, which reads the claims' points alone (`piopFw_at`, `Rs.sws_congr`);
* the views' masked values from `zp 64` to `h_ab` are the same `l0Clear` plus the request's pads
  (`Loop.swSrcE_l0_views`), and the workers' clear values plus the tape's pads whether or not the rows hold
  (`Masked.swSrcE_masked_pad_any`); the two pads are one (`Loop.finRepPads_tape`), so the clear values are `l0Clear`.

**`Loop.finRepLanes_ret`** states it.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open FlockSoundness.Refine (fOf gOf)
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

theorem lgOod_mem (L : _root_.Flock.Zk.PadLayout) : L.zp 64 ≤ L.lgOod 0 ∧ L.lgOod 0 < L.hab := by
  simp only [_root_.Flock.Zk.PadLayout.lgOod, _root_.Flock.Zk.PadLayout.hab]
  omega

theorem lgMsg_mem (L : _root_.Flock.Zk.PadLayout) {j c : ℕ} (hj : j < L.k0) (hc : c < 4) :
    L.zp 64 ≤ L.lgMsg j c ∧ L.lgMsg j c < L.hab := by
  simp only [_root_.Flock.Zk.PadLayout.lgMsg, _root_.Flock.Zk.PadLayout.hab]
  omega

theorem lgE_mem (L : _root_.Flock.Zk.PadLayout) {i c : ℕ} (hi : i < 2) (hc : c < 2) :
    L.zp 64 ≤ L.lgE i c ∧ L.lgE i c < L.hab := by
  simp only [_root_.Flock.Zk.PadLayout.lgE, _root_.Flock.Zk.PadLayout.hab]
  omega

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

set_option maxHeartbeats 4000000 in
/-- **`FinRepLanes` with no rows**: in the masked scope, with a witness, at a statement the loop serves whose inner
round 0, level 0 and pads the source draws from the tape (`LoopInnerRound0`, `Level0Src`, `RepSrc`), at table `j`'s
rep `r`'s run the returned proof's level 0 is the workers' clear values plus the request's pads. The firewall's level 0
is `l0Clear` at the coins from `ligP0` on its table and `ligBasis` plus the pads (`repFwC_l0`, its PIOP's basis by
`piopFw_at` and the claims' points, `finRepXs`); from `zp 64` to `h_ab` the views' masked values are the same plus the
same pads (`swSrcE_l0_views`) and the clear values plus the tape's pads (`swSrcE_masked_pad_any`), and the two pads
are one (`finRepPads_tape`). -/
theorem finRepLanes_ret (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) : FinRepLanes vs L₀ src (realOf src) wk x w := by
  intro ν t κ m₀ ρ₀ ops s0 spec j r tb hi q f ic' l p tr l' hrun
  obtain ⟨hF, hsx, hRep, -, hl, hgo⟩ := hrun
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hk : 2 * j.val + r.val < 2 * spec.tables.size := by
    rw [hspec]
    omega
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hkr : (2 * j.val + r.val) % 2 = r.val := by
    have := r.isLt
    omega
  have hA := admits_of_finCtx L₀ src (realOf src) wk hServe hF
  obtain ⟨schs, hs, hschZ⟩ := Masked.schedsPZk_live hx
  have hsj : schs[j.val]! = ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := hschZ j
  -- the views' commitments and pads commitments, as the switched cut's
  obtain ⟨schsV, hsV, cmsV, hcmsV, -, hpads⟩ :=
    finCommits_of vs L₀ src (realOf src) wk x w hsc hR hS0 ν t κ m₀ ρ₀ ops hF s0 spec hsx
  have e0 : schsV = schs := Except.ok.inj (hsV.symm.trans hs)
  rw [e0] at hcmsV hpads
  obtain ⟨cms, hcms, -⟩ := cmsCut_ok vs w ν t hA schs hcmsV
  have hext := (finPoints L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx).2
  obtain ⟨extras, hex⟩ := hext _ hk
  have hex' := hex
  rw [hk2] at hex'
  obtain ⟨icV, hicV⟩ := hpads (2 * j.val + r.val) hk l
  have hic2 := padsView_all vs w ν t hicV
  obtain ⟨ic, hicC, -⟩ := icCut_ok vs w ν t hA (i := j.val) (r := r.val) j.isLt r.isLt _
    (fun l => by
      have e := hic2 l
      unfold padsView at e
      rw [hk2, hkr] at e
      exact e)
  have hprev := prevCut_views L₀ src wk x w vs hsc hR hServe hF hsx hs cms j r
  have hexts : ∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras :=
    fun k hk' => hext k (by omega)
  -- from `zp 64` to `h_ab`, the views' masked values are level 0's clear values plus the request's pads
  have hl0 := fun k (hlo : (Masked.layE x j).zp 64 ≤ k) (hhi : k < (Masked.layE x j).hab) =>
    swSrcE_l0_views L₀ src wk x w vs hsc hR hServe hL0 hE hc hF hsx hs hcms j r hexts hprev hRep hex'
      (fun l => by
        have e' := hicC l
        rw [hsj] at e'
        exact e') hl hlo hhi
  -- the table's layout, and the request's pads as the tape's
  obtain ⟨htb, hsb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hs hk hRep
  rw [hk2] at htb hsb
  have hst : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := htb.symm.trans (Masked.tabsOf_stAt x j)
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (hsb.symm.trans hsj)).1
  have hlay : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 = Masked.layE x j := by
    rw [hst, hsch]
    rfl
  have hP := finRepPads_tape L₀ src x w hS hsx j r hRep
  have hPh : (_root_.Flock.Firewall.Finish.padsOf q.draws).h =
      (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h := by
    rw [← hP]
    rfl
  -- there, the workers' clear values are level 0's clear values
  have hkey : ∀ k, (Masked.layE x j).zp 64 ≤ k → k < (Masked.layE x j).hab →
      fOf (FlockComplete.LigLock.l0Clear (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) (Masked.zkTab x j).sch
          tb.g (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j) tb.extra r.val k +
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]!) =
      Composed.msgAt (fun i => (tabAt x ops j).tab.onVals.vals r
          (Masked.wtOf Masked.PE x w j, vs x w ν (finCoins x ops) j) (randAtT x ops t j).1
          (randAtT x ops t j).2.1.2.1 i + fOf (_root_.Flock.Firewall.Finish.padsOf q.draws).h[i.val]!) k := by
    intro k hlo hhi
    have hkE : k < (Masked.tabC Masked.PE x j).z.nE := by
      rw [Masked.nE_eq_hab x j]
      exact hhi
    have e1 := hl0 k hlo hhi
    rw [hPh] at e1
    have e2 := Masked.swSrcE_masked_pad_any x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j r hkE
    have e4 : FlockLevel3.GF128.val (vs x w ν (finCoins x ops) j r (Masked.wtOf Masked.PE x w j)
        (randAtT x ops t j).1 (randAtT x ops t j).2.1.2.1 ⟨k, hkE⟩) +
          (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h[k]! =
        FlockComplete.LigLock.l0Clear (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) (Masked.zkTab x j).sch
          tb.g (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j) tb.extra r.val k +
          (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h[k]! :=
      e2.symm.trans e1
    have e3 : FlockLevel3.GF128.val (vs x w ν (finCoins x ops) j r (Masked.wtOf Masked.PE x w j)
        (randAtT x ops t j).1 (randAtT x ops t j).2.1.2.1 ⟨k, hkE⟩) =
        FlockComplete.LigLock.l0Clear (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) (Masked.zkTab x j).sch
          tb.g (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j) tb.extra r.val k := by
      rw [← Composed.ZerocheckMasked.add_add_self (FlockLevel3.GF128.val _)
        (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h[k]!, e4,
        Composed.ZerocheckMasked.add_add_self]
    unfold Composed.msgAt
    split
    · rw [← e3, FlockSoundness.Refine.fOf_add]
      rfl
    · exact absurd hkE ‹_›
  -- the table's sizes and schedule
  have hm25 : 25 ≤ (Masked.stAt x j).m := (hall j.val hj).1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hm14 : 14 ≤ tb.st.m := by
    rw [hst]
    show 14 ≤ (Masked.stAt x j).m
    omega
  have hSm : tb.st.m = (Masked.stAt x j).m := by
    rw [hst]
    rfl
  have hSk : tb.st.fold.kLog = (Masked.stAt x j).c.kLog := by
    rw [hst]
    rfl
  obtain ⟨-, -, hsm, hk0, -, ⟨hn7, -⟩, -⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_schedule_seams _ _ _ (Masked.zkTab x j).hz
  have hn : tb.sch.n ≠ 0 := by
    rw [hsch, hn7, hsm]
    omega
  have hk0' : 1 ≤ tb.sch.k0 := by
    rw [hsch]
    exact hk0
  -- the request's extra claims are at the views' points
  have hxs := finRepXs L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx _ hk extras hex tb hi q f
    hRep
  have hxsz' : (_root_.Flock.Firewall.Finish.claimsOf q.xs).size = extras.size := by
    rw [← hxs]
    simp [_root_.Flock.Firewall.Finish.claimsOf]
  have hxsz : (_root_.Flock.Firewall.Finish.claimsOf q.xs).size = spec.points * (Masked.stAt x j).regions.size := by
    have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
      intro α β y f b hb
      cases y with
      | error e => cases hb
      | ok a => exact ⟨a, rfl, hb⟩
    have hq := hRep.2.1
    unfold repInOf at hq
    obtain ⟨⟨lay, ms⟩, -, hq⟩ := bok hq
    have hqx : q.xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
          (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) := by
      cases hq
      rfl
    rw [linkPoints_wordsAt x w hsc hR hF.1 hsx hk m₀ ρ₀, hk2,
      show (Masked.stsOf x)[j.val]! = Masked.stAt x j from getElem!_pos (Masked.stsOf x) j.val hj] at hqx
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
    rw [hqx]
    unfold _root_.Flock.Firewall.Drive.extras
    rw [flatMap_size_const _ _ _ fun _ => Array.size_map]
    simp only [viewPoints, Array.size_map, Array.size_range]
    exact Nat.mul_comm _ _
  -- the rep's PIOP: `m + k_log − 6` bodies of zerocheck and lincheck, then `2 + |extras| + 1` of ring switch
  have hzq : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [Masked.stsOf_half]
    exact (Masked.zkTab x j).hz
  have hP0 : Masked.ligP0 x j r = Masked.repBase x (2 * j.val + r.val) +
      ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) +
        (2 + (_root_.Flock.Firewall.Finish.claimsOf q.xs).size + 1) := by
    show Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) = _
    rw [Masked.repSlot_eq hsx hzq, Masked.stsOf_half, ← Masked.repBase_eq hsx (2 * j.val + r.val)]
    unfold FlockComplete.piopRoundsZk
    rw [hxsz]
    omega
  -- the firewall's level 0
  obtain ⟨hO, hMs, hEs⟩ := _root_.Flock.Firewall.Finish.repFwC_l0 (Masked.ligCoin (finCoins x ops)) (refK x)
    (Masked.ligRespE (finCoins x ops)) (fun k => Masked.coinOf_ligRespE (finCoins x ops) k) tb.st tb.sch tb.tPad _ _
    tb.g tb.cm0.toC (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC
    ic'.toC q.vals tb.extra tb.lanePad _ hn hk0' (Masked.ligP0 x j r)
    (Masked.ligBasis x (finCoins x ops) w t j r extras)
    (fun zp lp ss T sws t₃ l₃ hpf => by
      obtain ⟨hsws, hl₃, -⟩ := FlockComplete.piopFw_at hm14 hpf (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]!
      refine ⟨?_, ?_⟩
      · have hsE : sws = Masked.swsE x (finCoins x ops) w t j r extras := by
          rw [hsws]
          unfold Masked.swsE
          rw [Masked.tabsOf_stAt, ← hst, hl, Masked.zcCoins_ligRespE (finCoins x ops),
            Masked.lcCoins_ligRespE (finCoins x ops), Masked.rsCoins_ligRespE (finCoins x ops), ← hxsz']
          exact FlockComplete.Replay.Rs.sws_congr
            (by simp only [Array.size_append, FlockComplete.abcClaims_size, hxsz'])
            (FlockComplete.LigLock.Fed.ptsAt_claims_x (FlockComplete.LigLock.Fed.ptsAt_out _ _ _ _ _)
              (FlockComplete.LigLock.Fed.ptsAt_out _ _ _ _ _) (fun k => claimsOf_x hxs k)) _
        rw [hsE]
        unfold Masked.ligBasis
        rw [show Masked.ligSt x j = tb.st from (Masked.tabsOf_stAt x j).trans hst.symm]
      · rw [hl₃, hl, hP0, hSm, hSk]
        omega) hgo
  rw [hlay, hsch, hkr] at hO
  rw [hlay, hsch, hkr] at hMs
  rw [hlay, hsch, hkr] at hEs
  have hgl : ∀ A B : F128, gOf (⟨A, B⟩ : _root_.Flock.F256) = FlockSoundness.execArith.ofLimbs (fOf A) (fOf B) :=
    fun A B => (FlockSoundness.Refine.gOf_limbs _).trans (FlockSoundness.Discharge.ZkExec.ofLimbs_eq _ _).symm
  have h0 := hEs 0 (by norm_num)
  rw [ite_eq_left (rfl : (0 : ℕ) = 0)] at h0
  have h1 := hEs 1 (by norm_num)
  rw [ite_eq_right (show (1 : ℕ) ≠ 0 by decide)] at h1
  exact {
    oodVal := by
      rw [hO]
      exact hkey _ (lgOod_mem (Masked.layE x j)).1 (lgOod_mem (Masked.layE x j)).2
    msg := fun i hi => by
      rw [hMs i hi]
      exact ⟨(hgl _ _).trans (congrArg₂ FlockSoundness.execArith.ofLimbs
          (hkey _ (lgMsg_mem (Masked.layE x j) hi (by norm_num : 0 < 4)).1
            (lgMsg_mem (Masked.layE x j) hi (by norm_num : 0 < 4)).2)
          (hkey _ (lgMsg_mem (Masked.layE x j) hi (by norm_num : 1 < 4)).1
            (lgMsg_mem (Masked.layE x j) hi (by norm_num : 1 < 4)).2)),
        (hgl _ _).trans (congrArg₂ FlockSoundness.execArith.ofLimbs
          (hkey _ (lgMsg_mem (Masked.layE x j) hi (by norm_num : 2 < 4)).1
            (lgMsg_mem (Masked.layE x j) hi (by norm_num : 2 < 4)).2)
          (hkey _ (lgMsg_mem (Masked.layE x j) hi (by norm_num : 3 < 4)).1
            (lgMsg_mem (Masked.layE x j) hi (by norm_num : 3 < 4)).2))⟩
    e0 := by
      rw [h0]
      exact (hgl _ _).trans (congrArg₂ FlockSoundness.execArith.ofLimbs
        (hkey _ (lgE_mem (Masked.layE x j) (by norm_num : 0 < 2) (by norm_num : 0 < 2)).1
          (lgE_mem (Masked.layE x j) (by norm_num : 0 < 2) (by norm_num : 0 < 2)).2)
        (hkey _ (lgE_mem (Masked.layE x j) (by norm_num : 0 < 2) (by norm_num : 1 < 2)).1
          (lgE_mem (Masked.layE x j) (by norm_num : 0 < 2) (by norm_num : 1 < 2)).2))
    e1 := by
      rw [h1]
      exact (hgl _ _).trans (congrArg₂ FlockSoundness.execArith.ofLimbs
        (hkey _ (lgE_mem (Masked.layE x j) (by norm_num : 1 < 2) (by norm_num : 0 < 2)).1
          (lgE_mem (Masked.layE x j) (by norm_num : 1 < 2) (by norm_num : 0 < 2)).2)
        (hkey _ (lgE_mem (Masked.layE x j) (by norm_num : 1 < 2) (by norm_num : 1 < 2)).1
          (lgE_mem (Masked.layE x j) (by norm_num : 1 < 2) (by norm_num : 1 < 2)).2)) }

end Loop

end Claims.ZKServer
