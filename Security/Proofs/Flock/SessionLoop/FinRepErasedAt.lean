import Proofs.Flock.SessionLoop.FinRepPads
import Proofs.Flock.SessionLoop.FinRepView
import Proofs.Flock.SessionLoop.FinRounds
import Proofs.Flock.SessionLoop.Tables
import Proofs.Flock.Claims.ZK.FinishSwVals
import Proofs.Flock.Claims.ZK.FinishClaimsX

/-!
# `FinRepErased` at one rep stream

**`Loop.finRepErased_at`**: at table `j`'s rep `r` (rep stream `2j + r`), where the session's rep on the loop's table
and request returns `(p, tr)`, the run from the views returns `(p', tr)` with `eraseZ p' = eraseZ p`. The firewall's
rep is `Masked.swSrcE_repCut_v` at the views' source, on the loop's table (`finRepTab`, `Level0Src.realOf`), request
(`RepSrc`'s draws and salts, `RepSrc.salt`) and extra claims (`repFwC_claims_x`); the view run is that cut but for
level 0's commitment (`cms_cm0`), the pads commitment (`padsView_ic`) and what only the final message reads
(`eraseS_cutSrc`), so `repCut_eraseZ` and `go_of_eraseZ` give it. What the rep stream's place in the session fixes is
taken as premises:

* `hmk`: below `zp 64`, the views' masked values are the request's values plus its pads (FINH's equation);
* `hl0`: where the views' PIOP returns, level 0's masked values are `l0Clear` plus the pads (`swSrcE_l0_live`);
* `hpri`: the prior points the view run reads from the auditor's streams are the opened coins' (`Masked.priorE`);
* `hxs`: the extra claims from the views have the request's coordinates (`finRepXs`);
* `hrows`: every table's rows hold at its base (`Loop.FinRows`'s body).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (x : Msg) (w : Flock.Wit)

/-- A run's value fixes the run but for its history. -/
theorem go_eq_of_run_fst {α β : Type} {z : α × β} {a : α} (h : z.1 = a) : z = (a, z.2) := Prod.ext h rfl

set_option maxHeartbeats 1000000 in
/-- **The rep from the views up to `eraseZ`, at one rep stream**: in the masked scope, at a witness in the relation,
where the frame draws the tape's pads (`RepSrc`) and the loop the tape's level 0 (`Level0Src`), at admitted opened
coins, the session's schedules and the views' commitments, where the session's rep on the loop's table and request at
table `j`'s rep `r` returns `(p, tr)` from `l`, the run from the views returns `(p', tr)` with `eraseZ p' = eraseZ p`,
given `hmk`, `hl0`, `hpri`, `hxs` and `hrows`. -/
theorem finRepErased_at (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w) {ν : finS.Nonce} {t : finS.Tape} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hA : finS.Admits x (finCoins x ops))
    (hrows : ∀ j : Fin (Masked.jOf x), ((finS.C x j).T (finS.cz x (finCoins x ops) j)).tab.onVals.RowsAt
      (finS.wt x w j, vs x w ν (finCoins x ops) j)
      (finS.randAt x (finCoins x ops) t j).1.1 (finS.randAt x (finCoins x ops) t j).1.2.1.2.1)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hschs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs)
    {cms : Array FlockComplete.Commit} (hcms : ∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l))
    (j : Fin (Masked.jOf x)) (r : Fin 2) (hk : 2 * j.val + r.val < 2 * spec.tables.size)
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[(2 * j.val + r.val) / 2]!.extra (viewPoints x ops spec) =
      .ok extras)
    {ic : FlockComplete.InnerCommit} {l : Array Msg}
    (hic : (padsView vs x w ν t ops schs (2 * j.val + r.val)).go l = (.ok ic, l))
    (hl : l.size = Masked.repBase x (2 * j.val + r.val))
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray} (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f)
    {ic' : _root_.Flock.Firewall.Finish.InnerCommit}
    (hic' : (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic')
    (hxs : extras.map (·.x) = q.xs)
    (hmk : ∀ i < (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64,
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!)
    (hl0 : ∀ res t₃ l₃, ((FlockComplete.piopCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
        tb.st (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog (Masked.zkTab x j).sch.k0)
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
        (FlockComplete.flat tb.cm0.toC.cap) ic'.toC.root extras (Masked.priorE (finCoins x ops) j.val r.val)
        (FlockSoundness.Discharge.Composed.maskWordsOf (Masked.stsOf x)[j.val]!)).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
        (.ok (res, t₃), l₃) →
      ∀ i, (Masked.layE x j).zp 64 ≤ i → i < (Masked.layE x j).hab →
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked i =
          FlockComplete.LigLock.l0Clear (Masked.ligCoin (finCoins x ops)) (Masked.ligP0 x j r) (Masked.zkTab x j).sch
            (Masked.ligTab x w t j) (Masked.ligBasis x (finCoins x ops) w t j r extras) (Masked.layE x j)
            ((Masked.UE.tapeZk t).extra j.val) r.val i + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[i]!)
    (hpri : FlockComplete.priorOf (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!
      ((Array.range (2 * j.val + r.val)).map (streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x))) r.val =
        Masked.priorE (finCoins x ops) j.val r.val)
    {p : _root_.Flock.Zk.ZkProof} {tr : _root_.Flock.Transcript}
    (hrun : (((_root_.Flock.Firewall.Finish.repFwC
        (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
        (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad ((2 * j.val + r.val) % 2)
        (spec.domain spec.tables[(2 * j.val + r.val) / 2]! ((2 * j.val + r.val) % 2)).toUTF8 tb.g tb.cm0.toC
        (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
        tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
        (_root_.Flock.Transcript.init #[]
          (spec.domain spec.tables[(2 * j.val + r.val) / 2]! ((2 * j.val + r.val) % 2)).toUTF8)).go l).1 =
      .ok (p, tr)) :
    ∃ p' l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms (2 * j.val + r.val) extras ic).go l = (.ok (p', tr), l') ∧
      FlockComplete.eraseZ p' = FlockComplete.eraseZ p := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hkr : (2 * j.val + r.val) % 2 = r.val := by
    have := r.isLt
    omega
  -- the session's table and schedule
  obtain ⟨htb, hsb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hschs hk hRep
  rw [hk2] at htb hsb
  have hst : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := htb.symm.trans (Masked.tabsOf_stAt x j)
  obtain ⟨schs', hs', hschZ⟩ := Masked.schedsPZk_live hx
  have hss : schs' = schs := Except.ok.inj (hs'.symm.trans hschs)
  have hsj : schs[j.val]! = ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [← hss]
    exact hschZ j
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (hsb.symm.trans hsj)).1
  have htp : tb.tPad = (Masked.zkTab x j).tPad := (Prod.mk.inj (hsb.symm.trans hsj)).2
  have hLy : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 = Masked.layE x j := by
    rw [hst, hsch]
    rfl
  -- the loop's table is the tape's
  obtain ⟨-, -, -, hg⟩ := hS0.realOf ν t j r hRep.1
  obtain ⟨hex, hlp⟩ := hS0.realOf_lanes ν t j r hRep.1
  -- the request's draws and salts are the frame's
  have hq0 := hRep.2.1
  unfold repInOf at hq0
  obtain ⟨⟨lay, ms⟩, hstart, hq0⟩ := bok hq0
  have hqd : q.draws = _root_.Flock.Firewall.Frame.Draws.ofBytes lay (L₀.draws (ν, t) (2 * j.val + r.val)) := by
    cases hq0
    rfl
  have hqh : q.hiSalts = hi := by
    cases hq0
    rfl
  rw [hk2] at hstart
  have hLk : (LeanRelease.ofLoopLink L₀ src (realOf src)).lay x (2 * j.val + r.val) = lay := by
    have e : Loop.startOf x (2 * j.val + r.val) =
        _root_.Flock.Firewall.Drive.start (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
      unfold Loop.startOf
      rw [hsx]
      try dsimp only
    show ((Loop.startOf x (2 * j.val + r.val)).toOption.map (·.1)).getD default = lay
    rw [e, hk2, hstart]
    all_goals rfl
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := L₀) src (realOf src) j r hstart
  have hd := LeanRelease.draws_ofLoopLink (L₀ := L₀) src (realOf src)
  have hz := hS.size (LeanRelease.ofLoopLink L₀ src (realOf src)) hd ν t j r hlay
  have hpads := hS.pads (LeanRelease.ofLoopLink L₀ src (realOf src)) hd ν t j r hlay (Masked.UE.tapeZk t)
  have hq' : q.draws = (LeanRelease.ofLoopLink L₀ src (realOf src)).drawsAt x (ν, t) (Loop.frepOf j.val r.val) := by
    rw [hqd]
    show _ = _root_.Flock.Firewall.Frame.Draws.ofBytes
      ((LeanRelease.ofLoopLink L₀ src (realOf src)).lay x (2 * j.val + r.val)) (L₀.draws (ν, t) (2 * j.val + r.val))
    rw [hLk]
  have hsalt : ∀ i, 1 ≤ i → i < (Masked.zkTab x j).sch.levels.size →
      ∀ p < 2 ^ ((Masked.zkTab x j).sch.levels.getD i default).d,
        (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty)) i p =
          (Masked.UE.tapeZk t).salt j.val r.val i p := by
    intro i h1 h2 p hp
    rw [hqh]
    exact hS.salt ν t j r hRep.1 i h1 h2 p hp
  -- the firewall's run at the extra claims from the views
  have hrun' := go_eq_of_run_fst hrun
  have hxe : (_root_.Flock.Firewall.Finish.claimsOf q.xs).map (·.x) = extras.map (·.x) := by
    rw [_root_.Flock.Firewall.Finish.claimsOf_map_x, hxs]
  rw [_root_.Flock.Firewall.Finish.repFwC_claims_x _ _ _ _ _ _ _ _ _ hxe] at hrun'
  rw [hk2, hkr, hsch, htp, hex, hlp] at hrun'
  have he : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE (finCoins x ops) spec) = .ok extras := by
    rw [← hk2]
    exact hext
  obtain ⟨p', hcut, hpe⟩ := Masked.swSrcE_repCut_v (LeanRelease.ofLoopLink L₀ src (realOf src)) (ν, t)
    (Masked.UE.tapeZk t) Loop.frepOf x hx w hR t (finCoins x ops) hA (refK x) s0 spec hsx j r extras he
    (vs x w ν (finCoins x ops)) q hq' hlay hz hpads (hrows j)
    (fun i hi => hmk i (by rw [hLy]; exact hi)) tb.st hst _ tb.g hg tb.cm0.toC ic'.toC
    (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty)) hsalt rfl hl hl0 hrun'
  -- the view run is that cut but for level 0's commitment, the pads commitment and the final-only data
  obtain ⟨hcap, hlays⟩ := cms_cm0 vs src x w hsc hR hS0 hsx hschs hcms hk hRep
  rw [hk2] at hcap hlays
  obtain ⟨hn, hroot, hilay⟩ := padsView_ic vs L₀ src x w hsc hR hS hrows hsx hschs hk hic hRep hic'
  have hE := repCut_eraseZ (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
    (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st (Masked.zkTab x j).sch r.val
    (spec.domain spec.tables[j.val]! r.val).toUTF8
    (cm := FlockComplete.eraseC tb.cm0.toC) (cm' := cms[j.val]!) hcap.symm hlays.symm extras hn.symm hroot.symm
    hilay.symm (src := (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
    (src' := (finSrc vs x w ν t ops j.val).rep r.val) (eraseS_cutSrc vs w ν t hA j.isLt r.isLt).symm
    (Masked.priorE (finCoins x ops) j.val r.val)
    (FlockSoundness.Discharge.Composed.maskWordsOf (Masked.stsOf x)[j.val]!)
  obtain ⟨p'', hview, hpe'⟩ := go_of_eraseZ hE hcut
  exact ⟨p'', _, by
    unfold viewRep
    rw [hk2, hkr, hpri, FlockComplete.MaskRank.masksOf_getElem! hj, htb, hsj]
    exact hview, hpe'.trans hpe⟩

end Loop

end Claims.ZKServer
