import Proofs.Flock.SessionLoop.FinHmkFc
import Proofs.Flock.Claims.ZK.FinishInnerRet
import Proofs.Flock.Claims.ZK.FinishSwFc

/-!
# The views' masked values below `zp 64`, `final_c`'s included, from the cut session's own return

`Loop.swSrcE_hmk_fw` reaches `final_c` through the firewall's rep, so it needs the firewall's pads commitment to be the
switched cut's (`hicE`), and that holds only where table `j`'s rows do: elsewhere the switched pads are shifted and the
two roots differ. The cut session's own rep at stream `2j + r`, returning (`hprev`'s body at `k = 2j + r`), needs
neither: its PIOP returns (`piopCut_of_repCut`, so `swSrcE_hmk_cut` applies without `hpiop`), and the same rep cut at
its inner proof's first round stops there with that round's body (`repCut_inner_of_ret`), at `repSlot … (innerOff 0)`
(`Masked.innerAt_slot`), which is `swSrcE_hmk_cut_fc`'s premise. It reads no rows, no firewall run and no level-0 values.

* **`Loop.swSrcE_hmk_ret`**: every masked value of the views' source below `zp 64` is the request's plus its pad.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

namespace Loop

variable (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

set_option maxHeartbeats 4000000 in
/-- **`swSrcE_hmk_ret`: the views' masked values below `zp 64` are the request's plus its pads, from the cut
session's return**: at `swSrcE_hmk_cut`'s premises but `hpiop`, where the rep from the views' source at stream
`2j + r`, cut at the opened coins, returns from `l` (`hcur`, `hprev`'s body at `k = 2j + r`), every masked value of the
views' source below `zp 64` is the request's value there plus its pad: those but `final_c`'s by `swSrcE_hmk_cut`, at the
PIOP `hcur` runs (`piopCut_of_repCut`), and `final_c`'s by `swSrcE_hmk_cut_fc`, at the stop `repCut_inner_of_ret`
gives from `hcur`. -/
theorem swSrcE_hmk_ret (vs : finS.SwVals)
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
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
    (hcap : (FlockComplete.flat cms[j.val]!.cap).size < 2 ^ 64) (hroot : ic.root.size < 2 ^ 64)
    (hcur : ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[(2 * j.val + r.val) / 2]!.extra (viewPoints x ops spec) = .ok extras →
      ∀ ic,
      (∀ l, (FlockComplete.padsTreeCut Masked.hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[(2 * j.val + r.val) / 2]!.m
          (FlockDeploy.tabsOf (Masked.stsOf x))[(2 * j.val + r.val) / 2]!.kLog
            schs[(2 * j.val + r.val) / 2]!.1.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t ((2 * j.val + r.val) / 2)).rep
          ((2 * j.val + r.val) % 2))).go l = (.ok ic, l)) →
      ∀ l : Array Msg, l.size = Masked.repBase x (2 * j.val + r.val) →
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x)) Masked.hsE
          (FlockDeploy.tabsOf (Masked.stsOf x))[(2 * j.val + r.val) / 2]! schs[(2 * j.val + r.val) / 2]!.1
          ((2 * j.val + r.val) % 2)
          (spec.domain spec.tables[(2 * j.val + r.val) / 2]! ((2 * j.val + r.val) % 2)).toUTF8
          cms[(2 * j.val + r.val) / 2]! extras ic
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t ((2 * j.val + r.val) / 2)).rep
            ((2 * j.val + r.val) % 2))
          (Masked.priorE (finCoins x ops) ((2 * j.val + r.val) / 2) ((2 * j.val + r.val) % 2))
          (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[(2 * j.val + r.val) / 2]!).run
          (_root_.Flock.Transcript.init #[]
            (spec.domain spec.tables[(2 * j.val + r.val) / 2]! ((2 * j.val + r.val) % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (2 * j.val + r.val + 1)) :
    ∀ i < (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64,
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hkr : (2 * j.val + r.val) % 2 = r.val := by
    have := r.isLt
    omega
  obtain ⟨schs', hs', hschZ⟩ := Masked.schedsPZk_live hx
  have hsj : schs[j.val]! = ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [← Except.ok.inj (hs'.symm.trans hs)]
    exact hschZ j
  -- the cut session's rep at stream `2j + r` returns
  obtain ⟨p, tr, l', hret, -⟩ := hcur extras (by rw [hk2]; exact hext) ic
    (fun l' => by rw [hk2, hkr, hsj]; exact hic l') l hl
  rw [hk2, hkr, hsj] at hret
  -- so its PIOP does, and below `zp 64` but at `final_c` the masked values are the request's plus its pads
  obtain ⟨res, tr₁, l₁, hpiop⟩ := _root_.Flock.Firewall.Finish.piopCut_of_repCut _ _ _ _ _ _ _ _ _ _ _ _ _ _ hret
  have hmkV := swSrcE_hmk_cut L₀ src wk x w vs hsc hR hServe hE hc hF hsx hs hcms j r hexts hprev hRep hext hic hl
    hpiop hcap hroot
  -- and cut at its inner proof's first round it stops there, sending the source's masked `final_c`
  have hSch : FlockComplete.CutEq.SchOk (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! (Masked.zkTab x j).sch := by
    rw [Masked.tabsOf_stAt x j]
    exact Masked.schOk_of (Masked.zkTab x j)
  obtain ⟨t₄, L, hp4, hLs, hcut⟩ := _root_.Flock.Firewall.Finish.repCut_inner_of_ret _ _ _ _ _ _ _ _ _ _ _ _ _ l
    hSch hret
  have he : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE (finCoins x ops) spec) = .ok extras :=
    hext
  intro i hi'
  by_cases hne : i = (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fc
  · subst hne
    exact swSrcE_hmk_cut_fc L₀ src wk x w vs hsc hR hServe hE hc hF hsx hs hcms j r hexts hprev hRep hext hic hl
      ⟨"past the openings", L, t₄, hp4,
        by rw [hLs, hl, Masked.tabsOf_stAt x j]; exact Masked.innerAt_slot hx (finCoins x ops) hsx j r he, hcut⟩
  · exact hmkV i hi' hne

end Loop

end Claims.ZKServer
