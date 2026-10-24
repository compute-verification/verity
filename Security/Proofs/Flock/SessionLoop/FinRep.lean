import Proofs.Flock.SessionLoop.FinishParts
import Proofs.Flock.Claims.ZK.FinishBridge

/-!
# `FinishParts`' (c), the reps' proof files, in parts

`Loop.FinRepProofs` asks that rep stream `k`'s run from the views (`Loop.viewRep`) return a proof whose file is the loop's
`k`-th, `Finish.proofFile` on the loop's table (`Loop.realOf`) and request (`Loop.repInOf`). `proofFile` runs the
firewall's rep (`repFw`, answered by the request's coins from round `0`) and encodes its proof (`proofFile_ok`), and
`repFw_cut_at` holds that run to `repFwC` answered by `asksCut` at any coins that are the request's from the rep's
first round. So (c) is three parts, each named here, none a premise of a guarantee:

* **(c.1) `Loop.FinRepCoins`**: the request's coins are the opened coins from the rep's first round
  (`Masked.repBase`), and the rep ends by the last opening.
* **(c.2) `Loop.FinRepPads`**: the rep's table commits under `hm96-sha512`, and the request's pads have the layout's
  length, `h_ab = h_a·h_b` and `w = x·y` (`repFw_cut`'s premises).
* **(c.3) `Loop.FinRepFwView`**: where the firewall's rep answered by the opened coins returns from the rep's first round,
  the rep from the views returns a proof whose file at the session's shape is the firewall's at the table's.

**`Loop.finRepProofs_of_parts`**: (c.1) to (c.3) give `FinRepProofs`.
-/

namespace Flock.Firewall.Finish

open Flock With Transcript

/-- **The port's sends are the cut's at any coins that are the port's from round `base`**: `asksOf_rel` at a coin
function that agrees with `coinsFrom base coins` from round `base` to the rep's last. -/
theorem asksOf_rel_at (coins : Array (Array F128)) (base n : ℕ) (hn : base + coins.size ≤ n)
    (coin : ℕ → ℕ → F128) (hc : ∀ k < coins.size, ∀ i, coin (base + k) i = (coins[k]!)[i]!) :
    AsksRel (bridge base) (asksOf coins).toC (FlockComplete.asksCut coin n) where
  link _ _ _ _ _ _ _ h := by cases h
  round c k l hl a k' h := by
    change ((asksOf coins).round c).run k = _ at h
    dsimp only [asksOf] at h
    rw [StateT.run_bind, StateT.run_get, pure_bind] at h
    rcases hk : coins[k]? with _ | cs
    · rw [hk] at h; cases h
    · rw [hk] at h
      dsimp only at h
      have hk' : k < coins.size := (Array.getElem?_eq_some_iff.1 hk).1
      have hcs : coins[k]! = cs := by
        rw [getElem!_pos coins k hk']
        exact (Array.getElem?_eq_some_iff.1 hk).2
      split at h
      · rw [StateT.run_bind] at h
        cases h
      · rw [StateT.run_bind, StateT.run_set, pure_bind, StateT.run_pure] at h
        cases h
        have hlt : ¬ n ≤ l.size := by omega
        rw [asksCut_round_fw, if_neg hlt]
        refine ⟨?_, ?_⟩
        · show Except.ok (coin l.size) = Except.ok (fun i => cs[i]!)
          congr 1
          funext i
          rw [hl, hc k hk' i, hcs]
        · show (l.push (Claims.toMsg c)).size = base + (k + 1)
          rw [Array.size_push, hl]
          rfl

/-- **The firewall's rep is the session's rounds from `base`, at any coins that are its own from there**: `repFw_cut` at
a coin function that agrees with the rep's coins from round `base` (`asksOf_rel_at`). -/
theorem repFw_cut_at (coins : Array (Array F128)) (base n : ℕ) (hn : base + coins.size ≤ n)
    (coin : ℕ → ℕ → F128) (hc : ∀ k < coins.size, ∀ i, coin (base + k) i = (coins[k]!)[i]!) (st : Setup)
    (hm : st.merkle = MerkleScheme.hm96Sha512) (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray)
    (g : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit) (vals : Array F128)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (hps : pads.h.size = (Zk.PadLayout.new st.m st.kLog sch.k0).len)
    (hab : pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).hab]! =
      pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fa]! * pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fb]!)
    (htw : pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).tw]! =
      pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).tx]! * pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).ty]!)
    (tr : Transcript) {p : Zk.ZkProof} {t : Transcript} {k' : ℕ}
    (h : ((repFw (asksOf coins) (Hashes.ofSha512 Sha512.hash) (fun lv g s => pure (commitFw lv g s)) st sch tPad rep
      dom g cm0 extras pads ic vals extra lanePad salt).run tr).run 0 = .ok ((p, t), k'))
    (l : Array Claims.Msg) (hl : l.size = base) :
    (((repFwC (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g
        cm0.toC extras pads.toC ic.toC vals extra lanePad salt).run tr).go l).1 = .ok (p, t) ∧
      (((repFwC (FlockComplete.asksCut coin n) (Hashes.ofSha512 Sha512.hash) st sch tPad rep dom g
        cm0.toC extras pads.toC ic.toC vals extra lanePad salt).run tr).go l).2.size = base + k' := by
  rw [repFw_eq (asksOf coins) _ st hm sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt hps hab htw
    (fun e f => throw_bind_sv e f)] at h
  exact repFwC_rel (bridge base) (asksOf_rel_at coins base n hn coin hc) (ofSha512_rel _ Sha512.hash) st sch tPad rep
    dom g cm0.toC extras pads.toC ic.toC vals extra lanePad salt tr 0 l (by rw [hl]; rfl) (p, t) k' h

/-- **A proof file that `proofFile` returns**: the request's pads commitment returned, the firewall's rep answered by
the request's coins returned at its last coin, and the file is its proof's `encZk` at `shapeZk`. -/
theorem proofFile_ok {t : Tab} {rep : ℕ} {dom : ByteArray} {q : RepIn} {f : ByteArray}
    (h : proofFile t rep dom q = .ok f) :
    ∃ ic p tr, (innerCommitP (Hashes.ofSha512 Sha512.hash) (padsOf q.draws) (saltAt q.draws.salts) : V InnerCommit) =
        .ok ic ∧
      ((repFw (asksOf q.coins) (Hashes.ofSha512 Sha512.hash) (fun lv g s => pure (commitFw lv g s)) t.st t.sch t.tPad
        rep dom t.g t.cm0 (claimsOf q.xs) (padsOf q.draws) ic q.vals t.extra t.lanePad
        (fun l => saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run (Transcript.init #[] dom)).run 0 =
        .ok ((p, tr), q.coins.size) ∧
      f = encZk (shapeZk t.st t.sch (claimsOf q.xs).size) p := by
  unfold proofFile at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i ic hic
    split at h
    · cases h
    · rename_i a hrun
      obtain ⟨⟨p, tr⟩, k⟩ := a
      by_cases hk : k = q.coins.size
      · subst hk
        refine ⟨ic, p, tr, hic, hrun, ?_⟩
        simp at h
        first
        | exact h.symm
        | exact h
      · exfalso
        simp [hk] at h

end Flock.Firewall.Finish

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **Rep stream `k`'s proof file, as the loop's `Finish` makes it**: its table and salts for levels ≥ 1 (`realOf`), its
request (`repInOf`, at the frame's draws) and the file `proofFile` returns. -/
def RepAt {R : Type} (draws : R → ℕ → ByteArray)
    (src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)) (x : Msg) (w : Flock.Wit) (ρ : R)
    (h : List (Msg × Msg)) (spec : SessionSpec) (k : ℕ) (tb : _root_.Flock.Firewall.Finish.Tab)
    (hi : Array ByteArray) (q : _root_.Flock.Firewall.Finish.RepIn) (f : ByteArray) : Prop :=
  realOf src x w ρ k = .ok (tb, hi) ∧ repInOf x h spec (Masked.stsOf x) k (draws ρ k) hi = .ok q ∧
    _root_.Flock.Firewall.Finish.proofFile tb (k % 2) (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 q = .ok f

/-- **`proofsOf`'s `k`-th file is the loop's rep stream `k`'s** (`RepAt`). -/
theorem proofsOf_at {R : Type} {draws : R → ℕ → ByteArray}
    {src : Msg → Flock.Wit → R → ℕ → V (Array F128 × ByteArray × Array ByteArray)} {x : Msg} {w : Flock.Wit} {ρ : R}
    {h : List (Msg × Msg)} {spec : SessionSpec} {files : Array ByteArray}
    (hp : proofsOf src draws x w ρ h spec (Masked.stsOf x) = .ok files) {k : ℕ} (hk : k < spec.repStreams.size) :
    ∃ tb hi q, RepAt draws src x w ρ h spec k tb hi q files[k]! := by
  unfold proofsOf at hp
  obtain ⟨hsz, hall⟩ := FlockSoundness.Discharge.Exec.range_mapM_ok hp
  have e := hall k (by rw [hsz]; exact hk)
  rw [← getElem!_pos files k (by rw [hsz]; exact hk)] at e
  simp only [bind, Except.bind] at e
  split at e
  · cases e
  · rename_i a ha
    obtain ⟨tb, hi⟩ := a
    try dsimp only at e
    split at e
    · cases e
    · rename_i q hq
      exact ⟨tb, hi, q, ha, hq, e⟩

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **(c.1) The request's coins**: at a `Finish` `FinishIs` reads, rep stream `k`'s request's coins are the opened coins
from the rep's first round (`Masked.repBase`), and the rep's rounds end by the last opening. -/
def FinRepCoins : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ k < 2 * spec.tables.size, ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
      Masked.repBase x k + q.coins.size ≤ refK x ∧
        ∀ k' < q.coins.size, ∀ i, Masked.slotAt (finCoins x ops) (Masked.repBase x k + k') i = (q.coins[k']!)[i]!

/-- **(c.2) The rep's table and pads**: at a `Finish` `FinishIs` reads, rep stream `k`'s table commits under
`hm96-sha512`, and its request's pads have the layout's length, `h_ab = h_a·h_b` and `w = x·y`. -/
def FinRepPads : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ k < 2 * spec.tables.size, ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
      tb.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 ∧
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h.size =
        (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).len ∧
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).hab]! =
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fa]! *
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).fb]! ∧
      (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).tw]! =
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).tx]! *
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[(_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).ty]!

/-- **(c.3) The firewall's rep at the views**: at a `Finish` `FinishIs` reads, at the schedules and level 0's
commitments (a) gives, rep stream `k`'s extra claims and pads commitment from the views, and a history at its first
round, where the firewall's rep on the loop's table and request (`RepAt`), answered by the opened coins, returns a proof
`p`, rep stream `k`'s run from the views returns a proof whose file at the session's shape is `p`'s at the table's. -/
def FinRepFwView : Prop :=
  ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops →
    ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
    ∀ schs, (FlockComplete.schedsPZk (FlockDeploy.tabsOf (Masked.stsOf x)) : V _) = .ok schs →
    ∀ cms, (∀ l, (commitsView vs x w ν t ops schs).go l = (.ok cms, l)) →
    ∀ k < 2 * spec.tables.size, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras →
    ∀ ic l, (padsView vs x w ν t ops schs k).go l = (.ok ic, l) → l.size = Masked.repBase x k →
    ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
    ∀ ic' p tr, (_root_.Flock.Firewall.Finish.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (_root_.Flock.Firewall.Finish.padsOf q.draws) (_root_.Flock.Firewall.Finish.saltAt q.draws.salts) :
        V _root_.Flock.Firewall.Finish.InnerCommit) = .ok ic' →
      (((_root_.Flock.Firewall.Finish.repFwC
          (FlockComplete.asksCut (fun k i => Masked.slotAt (finCoins x ops) k i) (refK x))
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) tb.st tb.sch tb.tPad (k % 2)
          (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0.toC
          (_root_.Flock.Firewall.Finish.claimsOf q.xs) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic'.toC q.vals
          tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty))).run
          (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l).1 =
        .ok (p, tr) →
      ∃ p' tr' l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k extras ic).go l = (.ok (p', tr'), l') ∧
        FlockSoundness.Discharge.ZkView.encZk
          (FlockComplete.shapeZk (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]! schs[k / 2]!.1 extras.size) p' =
        _root_.Flock.Firewall.Finish.encZk
          (_root_.Flock.Firewall.Finish.shapeZk tb.st tb.sch (_root_.Flock.Firewall.Finish.claimsOf q.xs).size) p

/-- **(c.1) to (c.3) give (c).** Rep stream `k`'s file is `proofFile`'s on the loop's table and request
(`proofsOf_at`), the firewall's rep answered by the request's coins and its proof encoded (`proofFile_ok`); answered by
the opened coins from the rep's first round instead (c.1) at its table and pads (c.2), it returns the same proof
(`repFw_cut_at`), and the rep from the views returns a proof of the same file (c.3). -/
theorem finRepProofs_of_parts
    (h : FinRepCoins L₀ src real wk x w ∧ FinRepPads L₀ src real wk x w ∧ FinRepFwView vs L₀ src real wk x w) :
    FinRepProofs vs L₀ src real wk x w := by
  obtain ⟨hCo, hPa, hFV⟩ := h
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms files hp k hk extras hxe ic l hic hl
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  obtain ⟨tb, hi, q, hR⟩ := proofsOf_at hp (by rw [hrs]; exact hk)
  obtain ⟨ic', p, tr, hic', hrun, hfe⟩ := _root_.Flock.Firewall.Finish.proofFile_ok hR.2.2
  obtain ⟨hn, hc⟩ := hCo ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk tb hi q _ hR
  obtain ⟨hm, hps, hab, htw⟩ := hPa ν t κ m₀ ρ₀ ops hctx s0 spec hsx k hk tb hi q _ hR
  have hcut := (_root_.Flock.Firewall.Finish.repFw_cut_at q.coins (Masked.repBase x k) (refK x) hn
    (fun k i => Masked.slotAt (finCoins x ops) k i) hc tb.st hm tb.sch tb.tPad (k % 2)
    (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8 tb.g tb.cm0 (_root_.Flock.Firewall.Finish.claimsOf q.xs)
    (_root_.Flock.Firewall.Finish.padsOf q.draws) ic' q.vals tb.extra tb.lanePad
    (fun l => _root_.Flock.Firewall.Finish.saltAt (q.hiSalts.getD (l - 1) ByteArray.empty)) hps hab htw _ hrun l hl).1
  obtain ⟨p', tr', l', hv, he⟩ := hFV ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl
    tb hi q _ hR ic' p tr hic' hcut
  exact ⟨p', tr', l', hv, he.trans hfe.symm⟩

end Loop

end Claims.ZKServer
