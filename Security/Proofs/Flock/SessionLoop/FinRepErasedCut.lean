import Proofs.Flock.SessionLoop.FinRepErasedOf
import Proofs.Flock.SessionLoop.FinHmkCut

/-!
# `FinRepErased` with the views' masked values read off the history

`finRepErased_of` takes the views' masked values below `zp 64` as a premise (`hMk`). Here they come from the history
(`swSrcE_hmk_cut`) inside each stream's step, at the switched cut's commitments, pads and earlier streams, which leaves
two premises in the step's own form, each over `swSrcE_hmk_cut`'s premises:

* **`Loop.FinPiop`**: table `j`'s rep `r`'s PIOP returns from the views' source, cut at the opened coins, and the cap
  and pads root it frames are under `2^64` bytes. `swSrcE_hmk_cut` reads that return; the firewall's rep returning
  doesn't give it (`piopCut_fw` needs the masked values themselves), so it is a premise.
* **`Loop.FinFc`**: the equation at `final_c`'s pad, which `swSrcE_hmk_cut` leaves out.

**`Loop.finRepErased_of_cut`**: `finRepErased_of` with `hMk` replaced by those two.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **The views' cut PIOP returns**: at `swSrcE_hmk_cut`'s premises but its PIOP's return, table `j`'s rep `r`'s PIOP
from the views' source, cut at the opened coins, returns from a history at the rep's first round, and the level-0 cap
and pads root it frames are under `2^64` bytes. -/
def FinPiop : Prop :=
  ∀ {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)}, FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops →
  ∀ {s0 : _} {spec : SessionSpec}, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
  ∀ {schs : Array (_root_.Flock.Schedule × ℕ)},
    (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : _root_.Flock.V _) = .ok schs →
  ∀ {cms : Array FlockComplete.Commit},
    (∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut Masked.hsE (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle
        (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t i)).go l = (.ok cms, l)) →
  ∀ (j : Fin (Masked.jOf x)) (r : Fin 2),
    (∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras) →
    (∀ k < 2 * j.val + r.val, ∀ extras,
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
          (Masked.priorE (finCoins x ops) (k / 2) (k % 2))
          (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (k + 1)) →
  ∀ {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray}, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f →
  ∀ {extras : Array _root_.Flock.Claim},
    (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras →
  ∀ {ic : FlockComplete.InnerCommit},
    (∀ l, (FlockComplete.padsTreeCut Masked.hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)).go l = (.ok ic, l)) →
  ∀ {l : Array Msg}, l.size = Masked.repBase x (2 * j.val + r.val) →
    (∃ res tr₁ l₁, ((FlockComplete.piopCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
      (FlockComplete.flat cms[j.val]!.cap) ic.root extras (Masked.priorE (finCoins x ops) j.val r.val)
      (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[j.val]!).run
      (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
      (.ok (res, tr₁), l₁)) ∧
    (FlockComplete.flat cms[j.val]!.cap).size < 2 ^ 64 ∧ ic.root.size < 2 ^ 64

/-- **The views' masked value at `final_c`'s pad**: at `swSrcE_hmk_cut`'s premises, the views' source's masked value at
`final_c`'s pad is the request's value there plus its pad. -/
def FinFc : Prop :=
  ∀ {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)}, FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops →
  ∀ {s0 : _} {spec : SessionSpec}, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
  ∀ {schs : Array (_root_.Flock.Schedule × ℕ)},
    (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : _root_.Flock.V _) = .ok schs →
  ∀ {cms : Array FlockComplete.Commit},
    (∀ l, ((Array.range (FlockDeploy.tabsOf (Masked.stsOf x)).size).mapM fun i =>
      FlockComplete.commit0Cut Masked.hsE (FlockDeploy.tabsOf (Masked.stsOf x))[i]!.merkle
        (schs[i]!.1.levels.getD 0 default)
        (Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t i)).go l = (.ok cms, l)) →
  ∀ (j : Fin (Masked.jOf x)) (r : Fin 2),
    (∀ k < 2 * j.val + r.val, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras) →
    (∀ k < 2 * j.val + r.val, ∀ extras,
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
          (Masked.priorE (finCoins x ops) (k / 2) (k % 2))
          (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k / 2]!).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l =
          (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (k + 1)) →
  ∀ {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray}, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec (2 * j.val + r.val) tb hi q f →
  ∀ {extras : Array _root_.Flock.Claim},
    (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras →
  ∀ {ic : FlockComplete.InnerCommit},
    (∀ l, (FlockComplete.padsTreeCut Masked.hsE
      (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
        _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)).go l = (.ok ic, l)) →
  ∀ {l : Array Msg}, l.size = Masked.repBase x (2 * j.val + r.val) →
  ∀ {res} {tr₁ : _root_.Flock.Transcript} {l₁ : Array Msg},
    ((FlockComplete.piopCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!
      (_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
        (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0)
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)
      (FlockComplete.flat cms[j.val]!.cap) ic.root extras (Masked.priorE (finCoins x ops) j.val r.val)
      (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[j.val]!).run
      (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[j.val]! r.val).toUTF8)).go l =
      (.ok (res, tr₁), l₁) →
    (FlockComplete.flat cms[j.val]!.cap).size < 2 ^ 64 → ic.root.size < 2 ^ 64 →
    ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val).masked
        (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fc =
      q.vals[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fc]! +
        (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog
          tb.sch.k0).fc]!

set_option maxHeartbeats 1000000 in
/-- **`FinRepErased` at rep stream `k`, with the masked values from the history**: `finRepErased_step` with its `hmk`
from `swSrcE_hmk_cut` below `zp 64` but at `final_c`'s pad (`FinFc` there), at the switched cut's commitments
(`cmsCut_ok`), pads (`icCut_ok`) and earlier streams (`repCut_eraseZ`), where the views' cut PIOP returns
(`FinPiop`). -/
theorem finRepErased_step_cut (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) (hPiop : FinPiop vs L₀ src wk x w) (hFc : FinFc vs L₀ src wk x w)
    {ν : finS.Nonce} {t : finS.Tape} {κ : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Chk} {m₀ ρ₀ : Msg}
    {ops : List (Msg × Msg)} (hF : FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops)
    (hrows : ∀ j : Fin (Masked.jOf x), ((finS.C x j).T (finS.cz x (finCoins x ops) j)).tab.onVals.RowsAt
      (finS.wt x w j, vs x w ν (finCoins x ops) j)
      (finS.randAt x (finCoins x ops) t j).1.1 (finS.randAt x (finCoins x ops) t j).1.2.1.2.1)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)}
    (hs : (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs)
    {cms : Array FlockComplete.Commit} (hcms : ∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l))
    (hext : ∀ k < 2 * spec.tables.size, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (hpads : ∀ k < 2 * spec.tables.size, ∀ l, ∃ ic, (padsView vs x w ν t ops schs k).go l = (.ok ic, l))
    {k : ℕ} (hk : k < 2 * spec.tables.size)
    (hprev : ∀ k' < k, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.extra (viewPoints x ops spec) = .ok extras →
      ∀ ic l, (padsView vs x w ν t ops schs k').go l = (.ok ic, l) → l.size = Masked.repBase x k' →
        ∃ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k' extras ic).go l = (.ok (p, tr), l') ∧
          tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k' ∧ l'.size = Masked.repBase x (k' + 1))
    {extras : Array _root_.Flock.Claim}
    (hex : (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    {ic : FlockComplete.InnerCommit} {l : Array Msg}
    (hic : (padsView vs x w ν t ops schs k).go l = (.ok ic, l)) (hl : l.size = Masked.repBase x k)
    {tb : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray} {q : _root_.Flock.Firewall.Finish.RepIn}
    {f : ByteArray} (hRep : RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f)
    {ic' : _root_.Flock.Firewall.Finish.InnerCommit}
    (hic' : (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic')
    {p : _root_.Flock.Zk.ZkProof} {tr : _root_.Flock.Transcript}
    (hrun : (((_root_.Flock.Firewall.Finish.repFwC
        (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
        (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
        (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
        (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
        tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
        (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).1 =
      .ok (p, tr)) :
    ∃ p' l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p', tr), l') ∧
      FlockComplete.eraseZ p' = FlockComplete.eraseZ p := by
  have hmk : ∀ i < (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64,
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2)).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := by
    have hA := admits_of_finCtx L₀ src (realOf src) wk hServe hF
    have hjk : k / 2 < Masked.jOf x := by
      have hx : Masked.Live x := Masked.live_of hsc hR
      obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
      have hJ : Masked.jOf x = (Masked.stsOf x).size := by
        unfold Masked.jOf
        exact ite_eq_left_of_eq_true _ _ (eq_true hx)
      omega
    obtain ⟨j, r, rfl⟩ : ∃ (j : Fin (Masked.jOf x)) (r : Fin 2), k = 2 * j.val + r.val :=
      ⟨⟨k / 2, hjk⟩, ⟨k % 2, Nat.mod_lt _ (by norm_num)⟩, (Nat.div_add_mod k 2).symm⟩
    have hk2 : (2 * j.val + r.val) / 2 = j.val := by
      have := r.isLt
      omega
    have hkr : (2 * j.val + r.val) % 2 = r.val := by
      have := r.isLt
      omega
    rw [hk2, hkr]
    have hx : Masked.Live x := Masked.live_of hsc hR
    obtain ⟨schs', hs', hschZ⟩ := Masked.schedsPZk_live hx
    have hsj : schs[j.val]! = ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
      rw [← Except.ok.inj (hs'.symm.trans hs)]
      exact hschZ j
    -- the switched cut's commitments and pads
    obtain ⟨cmsC, hcmsC, -, -, -, hcm⟩ := cmsCut_ok vs w ν t hA schs hcms
    have hic2 := padsView_all vs w ν t hic
    obtain ⟨icC, hicC, -, -, -⟩ := icCut_ok vs w ν t hA (i := j.val) (r := r.val) j.isLt r.isLt _
      (fun l => by
        have e := hic2 l
        unfold padsView at e
        rw [hk2, hkr] at e
        exact e)
    -- the earlier streams from the switched cut
    have hprevC : ∀ k' < 2 * j.val + r.val, ∀ extras',
        (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.extra (viewPoints x ops spec) = .ok extras' → ∀ icX,
        (∀ l, (FlockComplete.padsTreeCut Masked.hsE
          (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.m
            (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.kLog schs[k' / 2]!.1.k0).len +
            _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
          ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k' / 2)).rep (k' % 2))).go l =
            (.ok icX, l)) →
        ∀ l : Array Msg, l.size = Masked.repBase x k' →
        ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x))
            Masked.hsE (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]! schs[k' / 2]!.1 (k' % 2)
            (spec.domain spec.tables[k' / 2]! (k' % 2)).toUTF8 cmsC[k' / 2]! extras' icX
            ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k' / 2)).rep (k' % 2))
            (Masked.priorE (finCoins x ops) (k' / 2) (k' % 2))
            (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[k' / 2]!).run
            (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k' / 2]! (k' % 2)).toUTF8)).go l =
            (.ok (p, tr), l') ∧ l'.size = Masked.repBase x (k' + 1) := by
      intro k' hk' extras' hex' icX hicX l₁ hl₁
      have hk'' : k' < 2 * spec.tables.size := by omega
      obtain ⟨ic₁, hic₁⟩ := hpads k' hk'' l₁
      have hic₁' := padsView_all vs w ν t hic₁
      obtain ⟨icC₁, hicC₁, hn₁, hroot₁, hilay₁⟩ := icCut_ok vs w ν t hA (i := k' / 2) (r := k' % 2) (by omega)
        (Nat.mod_lt _ (by norm_num)) _ (fun l => by
          have e := hic₁' l
          unfold padsView at e
          exact e)
      have hX : icX = icC₁ := by
        have e := (hicX l₁).symm.trans (hicC₁ l₁)
        exact (Except.ok.inj (Prod.mk.inj e).1)
      subst hX
      obtain ⟨p₁, tr₁, l₂, hrep, -, hsz⟩ := hprev k' hk' extras' hex' ic₁ l₁ hic₁ hl₁
      have hpri' := priorOf_view vs x w hsc hR hsx hext hpads hk'' (fun k'' hk''' => hprev k'' (by omega))
      unfold viewRep at hrep
      rw [hpri'] at hrep
      obtain ⟨p₂, hcut₂, -⟩ := go_of_eraseZ (repCut_eraseZ _ _ _ _ _ _ (hcm (k' / 2)).1.symm (hcm (k' / 2)).2.symm
        extras' hn₁ hroot₁ hilay₁ (eraseS_cutSrc vs w ν t hA (by omega) (Nat.mod_lt _ (by norm_num))) _ _) hrep
      exact ⟨p₂, tr₁, l₂, hcut₂, hsz⟩
    have hexts : ∀ k' < 2 * j.val + r.val, ∃ extras',
        (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.extra (viewPoints x ops spec) = .ok extras' :=
      fun k' hk' => hext k' (by omega)
    have hexj : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (viewPoints x ops spec) = .ok extras := by
      rw [← hk2]
      exact hex
    have hicj : ∀ l, (FlockComplete.padsTreeCut Masked.hsE
        (((_root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m
          (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.kLog (Masked.zkTab x j).sch.k0).len +
          _root_.Flock.Zk.K_PAD).nextPowerOfTwo * 2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE)
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t j.val).rep r.val)).go l =
          (.ok icC, l) := fun l => by
      have e' := hicC l
      rw [hsj] at e'
      exact e'
    obtain ⟨⟨res, tr₁, l₁, hpiop⟩, hcap, hroot⟩ := hPiop hF hsx hs hcmsC j r hexts hprevC hRep hexj hicj hl
    intro i hi
    by_cases hne : i = (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fc
    · rw [hne]
      exact hFc hF hsx hs hcmsC j r hexts hprevC hRep hexj hicj hl hpiop hcap hroot
    · exact swSrcE_hmk_cut L₀ src wk x w vs hsc hR hServe hE hc hF hsx hs hcmsC j r hexts hprevC hRep hexj hicj hl
        hpiop hcap hroot i hi hne
  exact finRepErased_step vs L₀ src wk x w hsc hR hServe hS hS0 hL0 hE hc hF hrows hsx hs hcms hext hpads hk hprev
    hex hic hl hRep hic' hmk hrun

set_option maxHeartbeats 1000000 in
/-- **`FinRepErased` from its parts, with the masked values from the history**: `finRepErased_of` with `hMk` replaced
by `FinPiop` (the views' cut PIOP returns) and `FinFc` (the equation at `final_c`'s pad), the rest of `hMk` read off
the history in each stream's step (`finRepErased_step_cut`). -/
theorem finRepErased_of_cut (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
    (hRows : ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops →
      ∀ j : Fin (Masked.jOf x), ((finS.C x j).T (finS.cz x (finCoins x ops) j)).tab.onVals.RowsAt
        (finS.wt x w j, vs x w ν (finCoins x ops) j)
        (finS.randAt x (finCoins x ops) t j).1.1 (finS.randAt x (finCoins x ops) t j).1.2.1.2.1)
    (hPiop : FinPiop vs L₀ src wk x w) (hFc : FinFc vs L₀ src wk x w) :
    FinRepErased vs L₀ src (realOf src) wk x w := by
  intro ν t κ m₀ ρ₀ ops hF s0 spec hsx schs hs cms hcms
  obtain ⟨schs', hs', cms', -, -, hpads⟩ :=
    finCommits_of vs L₀ src (realOf src) wk x w hsc hR hS0 ν t κ m₀ ρ₀ ops hF s0 spec hsx
  rw [hs] at hs'
  cases hs'
  have hext := (finPoints L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx).2
  have hrows := hRows ν t κ m₀ ρ₀ ops hF
  have hfs : (filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk = true := by
    have h5 := hF.2.2.2.2.1
    change (!(filesE src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops)).isOk) = false at h5
    simpa using h5
  unfold filesE at hfs
  rw [hsx] at hfs
  dsimp only at hfs
  cases hp : proofsOf src L₀.draws x w (ν, t) ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) with
  | error e => rw [hp] at hfs; cases hfs
  | ok files =>
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  have hAll : ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
      ∀ ic l, (padsView vs x w ν t ops schs k).go l = (.ok ic, l) → l.size = Masked.repBase x k →
        ∃ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p, tr), l') ∧
          tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k ∧ l'.size = Masked.repBase x (k + 1) := by
    intro k
    induction k using Nat.strong_induction_on with
    | _ k ih =>
    intro hk extras hex ic l hic hl
    have hprev : ∀ k' < k, ∀ extras,
        (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.extra (viewPoints x ops spec) = .ok extras →
        ∀ ic l, (padsView vs x w ν t ops schs k').go l = (.ok ic, l) → l.size = Masked.repBase x k' →
          ∃ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k' extras ic).go l = (.ok (p, tr), l') ∧
            tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k' ∧
            l'.size = Masked.repBase x (k' + 1) :=
      fun k' hk' => ih k' hk' (by omega)
    obtain ⟨tb, hi, q, hRep⟩ := proofsOf_at hp (by rw [hrs]; exact hk)
    obtain ⟨ic', p, tr, hic', hfw, -⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hRep.2.2
    have hcut := finRepCut L₀ src (realOf src) wk x w hsc ν t κ m₀ ρ₀ ops hF s0 spec hsx k hk tb hi q _ hRep ic' p tr
      hic' hfw l hl
    obtain ⟨p', l', hv, -⟩ := finRepErased_step_cut vs L₀ src wk x w hsc hR hServe hS hS0 hL0 hE hc hPiop hFc hF
      hrows hsx hs hcms hext hpads hk hprev hex hic hl hRep hic' hcut
    exact ⟨p', tr, l', hv,
      viewRep_rounds vs L₀ src (realOf src) wk hsc hR hServe hE hc hF hsx hs hcms hext hpads hk hprev hex hic hl hv,
      viewRep_size vs hsc hR ν t m₀ ρ₀ ops hsx hs cms hk hex ic hl hv⟩
  intro k hk extras hex ic l hic hl tb hi q f hRep ic' p tr hic' hrun
  exact finRepErased_step_cut vs L₀ src wk x w hsc hR hServe hS hS0 hL0 hE hc hPiop hFc hF hrows hsx hs hcms hext
    hpads hk (fun k' hk' => hAll k' (by omega)) hex hic hl hRep hic' hrun

end Loop

end Claims.ZKServer
