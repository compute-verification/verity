import Proofs.Flock.SessionLoop.FinHmkFc
import Proofs.Flock.SessionLoop.FinRepErasedAt
import Proofs.Flock.Claims.ZK.FinishSwFc

/-!
# The views' masked values below `zp 64`, `final_c`'s included, from the history and the firewall's rep

`Loop.swSrcE_hmk_cut` gives every masked value of the views' source below `zp 64` but `final_c`'s; `Loop.swSrcE_hmk_cut_fc`
gives `final_c`'s where the rep from the views' source, cut at its inner proof's first round, stops there. The firewall's
rep returning gives that stop (`Masked.swSrcE_repCut_fc_v`, at the masked values but `final_c`'s, which
`swSrcE_hmk_cut` supplies, so the route is not circular), at the firewall's level-0 and pads commitments; the cut at the
switched cut's commitments sends the same (`repCut_eraseZ`, `go_err_of_eraseZ`), after `innerOff 0` bodies past the PIOP
(`Masked.innerAt_slot`).

* `go_err_of_eraseZ`: runs equal up to `eraseZ` stop the same.
* **`Loop.swSrcE_hmk_fw`**: every masked value of the views' source below `zp 64` is the request's plus its pad.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

namespace Loop

/-- **Runs equal up to `eraseZ` stop the same**: where one stops with an error, the other stops with it at the same
history. -/
theorem go_err_of_eraseZ {m m' : _root_.Flock.TMW FlockComplete.PS _root_.Flock.Zk.ZkProof}
    (he : FlockComplete.eraseZ <$> m = FlockComplete.eraseZ <$> m') {t : _root_.Flock.Transcript} {l : Array Msg}
    {e : String} {L : Array Msg} (h : (m.run t).go l = (.error e, L)) : (m'.run t).go l = (.error e, L) := by
  have e' := congrArg (fun z => (z.run t).go l) he
  simp only [StateT.run_map, FlockComplete.go_map', h] at e'
  rcases h' : (m'.run t).go l with ⟨r, l''⟩
  rw [h'] at e'
  cases r with
  | error err =>
    simp only [Except.map, Prod.mk.injEq, Except.error.injEq] at e'
    obtain ⟨rfl, rfl⟩ := e'
    rfl
  | ok q => simp [Except.map] at e'

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

set_option maxHeartbeats 4000000 in
/-- **`swSrcE_hmk_fw`: the views' masked values below `zp 64` are the request's plus its pads**: at `swSrcE_hmk_cut`'s
premises, where the frame draws the tape's pads (`RepSrc`) and the loop the tape's level 0 (`Level0Src`), the extra
claims from the views have the request's coordinates (`hxs`), level 0's masked values are `l0Clear` plus the pads
wherever the views' PIOP returns (`hl0`, as `swSrcE_l0_live` gives them), the firewall's level-0 and pads commitments
match the switched cut's but for what only the final message reads (`hcm0`, `hicE`), and the session's rep on the loop's
table and request returns (`hrun`), every masked value of the views' source below `zp 64` is the request's value there
plus its pad: `final_c`'s by `swSrcE_hmk_cut_fc`, at the stop `swSrcE_repCut_fc_v` gives from the others, and the
others by `swSrcE_hmk_cut`. -/
theorem swSrcE_hmk_fw (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : _root_.Flock.V _) = .ok schs)
    {cms : Array FlockComplete.Commit}
    (hcms : ∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut Masked.hsE (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle
        (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t i)).go l = (.ok cms, l))
    (j : Fin (Masked.jOf x)) (r : Fin 2)
    (hexts : ∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (hprev : ∀ k < 2 * j.val + r.val, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras → ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut Masked.hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.m
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.kLog schs[k / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))).go l =
          (.ok ic, l)) →
      ∀ l : Array Msg, l.size = Masked.repBase x k →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x)) Masked.hsE
          (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 cms[k / 2]! extras ic
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2))
          (Masked.priorE (finCoins x ops) (k / 2) (k % 2)) (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (k + 1))
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray} (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f)
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras)
    {ic : FlockComplete.InnerCommit}
    (hic : ∀ l, (FlockComplete.padsTreeCut Masked.hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)).go l = (.ok ic, l))
    {l : Array Msg} (hl : l.size = Masked.repBase x (2 * j.val + r.val))
    {res} {tr₁ : _root_.Flock.Transcript} {l₁ : Array Msg}
    (hpiop : ((FlockComplete.piopCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
      (FlockComplete.flat cms[j.val]!.cap) ic.root extras (Masked.priorE (finCoins x ops) j.val r.val)
      (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[j.val]!).run
      (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
      (.ok (res, tr₁), l₁))
    (hcap : (FlockComplete.flat cms[j.val]!.cap).size < 2 ^ 64) (hroot : ic.root.size < 2 ^ 64)
    {ic' : _root_.Flock.Firewall.Finish.InnerCommit}
    (hxs : extras.map (·.x) = q.xs)
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
    (hcm0 : (FlockComplete.eraseC tb.cm0.toC).cap = cms[j.val]!.cap ∧
      (FlockComplete.eraseC tb.cm0.toC).layers = cms[j.val]!.layers)
    (hicE : ic'.toC.ch.size = ic.ch.size ∧ ic'.toC.root = ic.root ∧ ic'.toC.layers = ic.layers)
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
    ∀ i < (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64,
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
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
  -- the session's table and schedule
  obtain ⟨htb, hsb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hs hk hRep
  rw [hk2] at htb hsb
  have hst : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := htb.symm.trans (Masked.tabsOf_stAt x j)
  obtain ⟨schs', hs', hschZ⟩ := Masked.schedsPZk_live hx
  have hss : schs' = schs := Except.ok.inj (hs'.symm.trans hs)
  have hsj : schs[j.val]! = ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [← hss]
    exact hschZ j
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (hsb.symm.trans hsj)).1
  have htp : tb.tPad = (Masked.zkTab x j).tPad := (Prod.mk.inj (hsb.symm.trans hsj)).2
  have hLy : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 = Masked.layE x j := by
    rw [hst, hsch]
    rfl
  -- the loop's table is the tape's, and the request's salts the frame's
  obtain ⟨-, -, -, hg⟩ := hS0.realOf ν t j r hRep.1
  obtain ⟨hex, hlp⟩ := hS0.realOf_lanes ν t j r hRep.1
  have hq0 := hRep.2.1
  unfold repInOf at hq0
  obtain ⟨⟨lay, ms⟩, -, hq0⟩ := bok hq0
  have hqh : q.hiSalts = hi := by
    cases hq0
    rfl
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
  have he : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE (finCoins x ops) spec) = .ok extras :=
    hext
  -- below `zp 64` but at `final_c`, from the history
  have hmkV := swSrcE_hmk_cut L₀ src wk x w vs hsc hR hServe hE hc hF hsx hs hcms j r hexts hprev hRep hext hic hl
    hpiop hcap hroot
  -- the firewall's rep places the switched source's cut at its inner proof's first round
  obtain ⟨t₄, L, hp4, hLs, hcut⟩ := Masked.swSrcE_repCut_fc_v x hx w hR t (finCoins x ops) hA (refK x) s0 spec hsx
    j r extras he (vs x w ν (finCoins x ops)) q
    (fun k hk' hne => hmkV k (by rw [hLy]; exact hk') (by rw [hLy]; exact hne)) tb.st hst _ tb.g hg tb.cm0.toC
    ic'.toC (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty)) hsalt rfl hl hl0
    hrun'
  -- at the switched cut's commitments
  have hE' := repCut_eraseZ (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) L.size)
    (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st (Masked.zkTab x j).sch r.val
    (spec.domain spec.tables[j.val]! r.val).toUTF8
    (cm := FlockComplete.eraseC tb.cm0.toC) (cm' := cms[j.val]!) hcm0.1 hcm0.2 extras hicE.1 hicE.2.1 hicE.2.2
    (src := (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
    (src' := (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val) rfl
    (Masked.priorE (finCoins x ops) j.val r.val)
    (FlockSoundness.Discharge.Composed.maskWordsOf (Masked.stsOf x)[j.val]!)
  have hcut' := go_err_of_eraseZ hE' hcut
  intro i hi'
  by_cases hne : i = (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fc
  · subst hne
    refine swSrcE_hmk_cut_fc L₀ src wk x w vs hsc hR hServe hE hc hF hsx hs hcms j r hexts hprev hRep hext hic hl
      ⟨"past the openings", L, t₄, hp4, ?_, ?_⟩
    · rw [hLs, hl, hst]
      exact Masked.innerAt_slot hx (finCoins x ops) hsx j r he
    · rw [htb, FlockComplete.MaskRank.masksOf_getElem! hj]
      exact hcut'
  · exact hmkV i hi' hne

end Loop

end Claims.ZKServer
