import Proofs.Flock.SessionLoop.FinPrevFw
import Proofs.Flock.SessionLoop.FinHmkRet
import Proofs.Flock.SessionLoop.FinRepPads
import Proofs.Flock.Claims.ZK.FinishSwMkAny

/-!
# `FinRepMsgs` with no rows

`FinRepRows` consumes `Loop.FinRepMsgs`: at table `j`'s rep `r`'s run, below `zp 64` the workers' clear values are the
request's. Here it holds outright, reading no table's rows:

* the cut session's own rep at stream `2j + r` returns, and so does every earlier one (`Loop.repCut_views_ret`,
  `Loop.prevCut_views`), so below `zp 64` the views' masked values are the request's plus its pads
  (`Loop.swSrcE_hmk_ret`);
* the views' masked values are the workers' clear values plus the tape's pads, whether or not the rows hold
  (`Masked.swSrcE_masked_pad_any`);
* the request's pads are the tape's (**`Loop.finRepPads_tape`**, `Masked.pads_frame_eq`), so the pads cancel.

**`Loop.finRepMsgs_ret`** states it.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **The request's pads are the tape's**: where the frame draws the tape's pads (`RepSrc`), at table `j`'s rep `r`'s
request, its pads are the tape's pads of table `j`'s rep `r` at the table's layout (`Masked.pads_frame_eq`, at the
layout the frame starts at, `lay_ofLoopLink`). -/
theorem finRepPads_tape (hS : Loop.RepSrc L₀ src x w) {ν : finS.Nonce} {t : finS.Tape} {h : List (Msg × Msg)}
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    (j : Fin (Masked.jOf x)) (r : Fin 2) {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    {q : _root_.Flock.Firewall.Finish.RepIn} {f : ByteArray}
    (hRep : RepAt L₀.draws src x w (ν, t) h spec (2 * j.val + r.val) tb hi q f) :
    (_root_.Flock.Firewall.Finish.padsOf q.draws).toC =
      FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j) := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hq := hRep.2.1
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, hst0, hq⟩ := bok hq
  have hqd : q.draws = _root_.Flock.Firewall.Frame.Draws.ofBytes lay (L₀.draws (ν, t) (2 * j.val + r.val)) := by
    cases hq
    rfl
  have hLk : (LeanRelease.ofLoopLink L₀ src (realOf src)).lay x (2 * j.val + r.val) = lay := by
    have e : Loop.startOf x (2 * j.val + r.val) =
        _root_.Flock.Firewall.Drive.start (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
      unfold Loop.startOf
      rw [hsx]
      try dsimp only
    show ((Loop.startOf x (2 * j.val + r.val)).toOption.map (·.1)).getD default = lay
    rw [e]
    rw [hst0]
    all_goals rfl
  rw [hk2] at hst0
  have hlay0 := LeanRelease.lay_ofLoopLink (L₀ := L₀) src (realOf src) j r hst0
  have hd := LeanRelease.draws_ofLoopLink (L₀ := L₀) src (realOf src)
  have hzs := hS.size (LeanRelease.ofLoopLink L₀ src (realOf src)) hd ν t j r hlay0
  have hpads := hS.pads (LeanRelease.ofLoopLink L₀ src (realOf src)) hd ν t j r hlay0 (Masked.UE.tapeZk t)
  have hq' : q.draws = (LeanRelease.ofLoopLink L₀ src (realOf src)).drawsAt x (ν, t) (Loop.frepOf j.val r.val) := by
    rw [hqd]
    show _ = _root_.Flock.Firewall.Frame.Draws.ofBytes
      ((LeanRelease.ofLoopLink L₀ src (realOf src)).lay x (2 * j.val + r.val)) (L₀.draws (ν, t) (2 * j.val + r.val))
    rw [hLk]
  exact Masked.pads_frame_eq (LeanRelease.ofLoopLink L₀ src (realOf src)) (ν, t) (Masked.UE.tapeZk t) Loop.frepOf x t
    j r q hq' hlay0 hzs hpads

set_option maxHeartbeats 4000000 in
/-- **`FinRepMsgs` with no rows**: in the masked scope, with a witness, at a statement the loop serves whose level 0
and pads the source draws from the tape (`Level0Src`, `RepSrc`), below `zp 64` the workers' clear values at table `j`'s
rep `r`'s run are the request's. The cut session's reps at streams up to `2j + r` return (`prevCut_views`,
`repCut_views_ret`), so the views' masked values there are the request's plus its pads (`swSrcE_hmk_ret`, at the views'
commitments and pads commitment, `cmsCut_ok`, `icCut_ok`, and `finPiop`'s sizes); they are the workers' clear values
plus the tape's pads at any values (`swSrcE_masked_pad_any`), and the two pads are one (`finRepPads_tape`). -/
theorem finRepMsgs_ret (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) : FinRepMsgs vs L₀ src (realOf src) wk x w := by
  intro ν t κ m₀ ρ₀ ops s0 spec j r tb hi q f ic' l p tr l' hrun
  obtain ⟨hF, hsx, hRep, -, hl, -⟩ := hrun
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
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
  obtain ⟨-, hcap, hroot⟩ := finPiop L₀ src wk x w vs hsc hR hServe hF hsx hs hcms j r hexts hprev hRep hex'
    (fun l => by
      have e' := hicC l
      rw [hsj] at e'
      exact e') hl
  have hmk := swSrcE_hmk_ret L₀ src wk x w vs hsc hR hServe hE hc hF hsx hs hcms j r hexts hprev hRep hex'
    (fun l => by
      have e' := hicC l
      rw [hsj] at e'
      exact e') hl hcap hroot
    (fun _ hex₁ ic₁ _ l₁ hl₁ =>
      repCut_views_ret L₀ src wk x w hsc hR hServe hF hsx hs hk hex₁ _root_.Flock.Sha512.hash _ ic₁ _ hl₁)
  -- the table's layout, and the request's pads as the tape's
  obtain ⟨htb, hsb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hs hk hRep
  rw [hk2] at htb hsb
  have hst : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := htb.symm.trans (Masked.tabsOf_stAt x j)
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (hsb.symm.trans hsj)).1
  have hlay : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 = Masked.layE x j := by
    rw [hst, hsch]
    rfl
  have hP := finRepPads_tape L₀ src x w hS hsx j r hRep
  intro i hi'
  have hiz : i < (Masked.layE x j).zp 64 := hi'
  have hkE : i < (Masked.tabC Masked.PE x j).z.nE := by
    rw [Masked.nE_eq_hab x j]
    exact lt_trans hiz (Masked.zp64_lt_hab x j)
  have e1 := hmk i (by rw [hlay]; exact hiz)
  rw [hP] at e1
  have e2 := Masked.swSrcE_masked_pad_any x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j r hkE
  have e4 : FlockLevel3.GF128.val (vs x w ν (finCoins x ops) j r (Masked.wtOf Masked.PE x w j)
      (randAtT x ops t j).1 (randAtT x ops t j).2.1.2.1 ⟨i, hkE⟩) +
        (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h[i]! =
      q.vals[i]! + (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h[i]! :=
    e2.symm.trans e1
  have e3 : FlockLevel3.GF128.val (vs x w ν (finCoins x ops) j r (Masked.wtOf Masked.PE x w j)
      (randAtT x ops t j).1 (randAtT x ops t j).2.1.2.1 ⟨i, hkE⟩) = q.vals[i]! := by
    rw [← Composed.ZerocheckMasked.add_add_self (FlockLevel3.GF128.val _)
      (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h[i]!, e4,
      Composed.ZerocheckMasked.add_add_self]
  unfold Composed.msgAt
  split
  · exact e3
  · exact absurd hkE ‹_›

end Loop

end Claims.ZKServer
