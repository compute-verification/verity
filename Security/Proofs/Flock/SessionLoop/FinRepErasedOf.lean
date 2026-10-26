import Proofs.Flock.SessionLoop.FinRepErasedAt
import Proofs.Flock.SessionLoop.FinL0Vals

/-!
# `FinRepErased` from its parts

**`Loop.finRepErased_of`**: at SessionLoopZKDeployed's premises, where every table's rows hold at its base (`hRows`,
`Loop.FinRows`'s body) and the views' masked values below `zp 64` are the request's values plus its pads (`hMk`, FINH's
equation), the run from the views is the firewall's up to `eraseZ` at every rep stream. By strong induction on the
stream: where the streams before `k` return from the views with the auditor's rounds (`viewRep_rounds`,
`viewRep_size`), rep `0`'s claim points are the opened coins' (`priorOf_view`, `repCut_points`) and they return from the
switched cut too (`repCut_eraseZ`), so level 0's masked values are `l0Clear` plus the pads (`swSrcE_l0_live`), and
`finRepErased_at` gives stream `k`; the firewall's rep at `k` returns (`proofsOf_at`, `proofFile_ok`, `finRepCut`), so
stream `k` returns from the views with the auditor's rounds.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **The prior points from the auditor's streams are the opened coins'**: where the streams before `k` return from the
views with the auditor's rounds, the claim points rep `k % 2` reads off them (`priorOf`) are `Masked.priorE`'s: none at
rep `0`, and at rep `1` rep `0`'s rounds' points, which a cut rep holds at the opened coins (`repCut_points`). -/
theorem priorOf_view (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    {ν : finS.Nonce} {t : finS.Tape} {m₀ ρ₀ : Msg} {ops : List (Msg × Msg)}
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {schs : Array (_root_.Flock.Schedule × ℕ)} {cms : Array FlockComplete.Commit}
    (hext : ∀ k < 2 * spec.tables.size, ∃ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!.extra (viewPoints x ops spec) = .ok extras)
    (hpads : ∀ k < 2 * spec.tables.size, ∀ l, ∃ ic, (padsView vs x w ν t ops schs k).go l = (.ok ic, l))
    {k : ℕ} (hk : k < 2 * spec.tables.size)
    (hprev : ∀ k' < k, ∀ extras,
      (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]!.extra (viewPoints x ops spec) = .ok extras →
      ∀ ic l, (padsView vs x w ν t ops schs k').go l = (.ok ic, l) → l.size = Masked.repBase x k' →
        ∃ p tr l', (viewRep vs x w ν t m₀ ρ₀ ops spec schs cms k' extras ic).go l = (.ok (p, tr), l') ∧
          tr.rounds = streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) k' ∧ l'.size = Masked.repBase x (k' + 1)) :
    FlockComplete.priorOf (FlockDeploy.tabsOf (Masked.stsOf x))[k / 2]!
      ((Array.range k).map (streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x))) (k % 2) =
      Masked.priorE (finCoins x ops) (k / 2) (k % 2) := by
  unfold FlockComplete.priorOf Masked.priorE
  by_cases hr : k % 2 = 1
  · rw [if_pos hr, if_pos hr]
    have hx : Masked.Live x := Masked.live_of hsc hR
    obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
    obtain ⟨extras, hex⟩ := hext (k - 1) (by omega)
    obtain ⟨ic, hic⟩ := hpads (k - 1) (by omega) (Array.replicate (Masked.repBase x (k - 1)) default)
    obtain ⟨p, tr, l', hrun, hrounds, -⟩ := hprev (k - 1) (by omega) extras hex ic _ hic (by simp)
    have hback : ((Array.range k).map (streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x))).back! =
        streamOf x ((m₀, ρ₀) :: ops) spec (Masked.stsOf x) (k - 1) := by
      obtain ⟨n, rfl⟩ : ∃ n, k = n + 1 := ⟨k - 1, by omega⟩
      rw [Nat.add_sub_cancel]
      first
      | (simp [Array.back!]; done)
      | (simp only [Array.back!, Array.size_map, Array.size_range, Nat.add_sub_cancel]
         rw [getElem!_pos _ n (by simp)]
         simp)
    have h2 : (k - 1) / 2 = k / 2 := by omega
    have h3 : 2 * (k / 2) = k - 1 := by omega
    unfold viewRep at hrun
    have hpt := Masked.repCut_points hx (finCoins x ops) (refK x) (i := (k - 1) / 2) (by omega) _ _ _ _ _ _ _ _ _
      (q := k - 1) (by simp) hrun
    rw [hback, ← hrounds, h3, ← h2]
    exact hpt
  · rw [if_neg hr, if_neg hr]

set_option maxHeartbeats 1000000 in
/-- **`FinRepErased` at rep stream `k`, from the streams before it**: where the streams before `k` return from the
views with the auditor's rounds, the run from the views at `k` is the firewall's up to `eraseZ` (`finRepErased_at`),
its `hl0` from `swSrcE_l0_live` at the switched cut (`cmsCut_ok`, `icCut_ok`, the earlier streams by `repCut_eraseZ`),
its `hpri` from `priorOf_view` and its `hxs` from `finRepXs`. -/
theorem finRepErased_step (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
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
    (hmk : ∀ i < (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64,
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2)).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!)
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
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hA := admits_of_finCtx L₀ src (realOf src) wk hServe hF
  obtain ⟨-, -, hts, -⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    exact ite_eq_left_of_eq_true _ _ (eq_true hx)
  have hxs := finRepXs L₀ src (realOf src) wk x w hsc hR ν t κ m₀ ρ₀ ops hF s0 spec hsx k hk extras hex tb hi q f hRep
  have hpri := priorOf_view vs x w hsc hR hsx hext hpads hk hprev
  have hjk : k / 2 < Masked.jOf x := by omega
  obtain ⟨j, r, rfl⟩ : ∃ (j : Fin (Masked.jOf x)) (r : Fin 2), k = 2 * j.val + r.val :=
    ⟨⟨k / 2, hjk⟩, ⟨k % 2, Nat.mod_lt _ (by norm_num)⟩, (Nat.div_add_mod k 2).symm⟩
  have hk2 : (2 * j.val + r.val) / 2 = j.val := by
    have := r.isLt
    omega
  have hkr : (2 * j.val + r.val) % 2 = r.val := by
    have := r.isLt
    omega
  have hj : j.val < (Masked.stsOf x).size := Masked.lt_of_jOf j
  rw [hk2, hkr] at hpri hmk
  refine finRepErased_at vs L₀ src x w hsc hR hS hS0 hA hrows hsx hs hcms j r hk hex hic hl hRep hic' hxs hmk ?_ hpri
    hrun
  intro res t₃ l₃ hpc i hlo hhi
  -- the loop's table and schedule, and its packed table and extra lanes
  obtain ⟨htb, -⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hs hk hRep
  rw [hk2] at htb
  obtain ⟨schs', hs', hschZ⟩ := Masked.schedsPZk_live hx
  have hsj : schs[j.val]! = ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [← Except.ok.inj (hs'.symm.trans hs)]
    exact hschZ j
  obtain ⟨-, -, -, hg⟩ := hS0.realOf ν t j r hRep.1
  obtain ⟨hexT, -⟩ := hS0.realOf_lanes ν t j r hRep.1
  -- level 0's commitments and the pads commitment from the switched cut
  obtain ⟨cmsC, hcmsC, -, -, -, hcm⟩ := cmsCut_ok vs w ν t hA schs hcms
  obtain ⟨hcap, -⟩ := cms_cm0 vs src x w hsc hR hS0 hsx hs hcms hk hRep
  rw [hk2] at hcap
  obtain ⟨-, hroot, -⟩ := padsView_ic vs L₀ src x w hsc hR hS hrows hsx hs hk hic hRep hic'
  have hic2 := padsView_all vs w ν t hic
  obtain ⟨icC, hicC, -, hrootC, -⟩ := icCut_ok vs w ν t hA (i := j.val) (r := r.val) j.isLt r.isLt _
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
      ∃ p tr l', ((FlockComplete.repCut (FlockComplete.asksCut (Masked.ligCoin (finCoins x ops)) (refK x)) Masked.hsE
          (FlockDeploy.tabsOf (Masked.stsOf x))[k' / 2]! schs[k' / 2]!.1 (k' % 2)
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
  -- the view PIOP at the switched cut's commitments
  have hmw : (FlockComplete.MaskRank.masksOf (Masked.stsOf x))[j.val]! =
      FlockSoundness.Discharge.Composed.maskWordsOf (Masked.stsOf x)[j.val]! :=
    FlockComplete.MaskRank.masksOf_getElem! hj
  have hpiopC := hpc
  rw [← htb, ← hcap, ← (hcm j.val).1, ← hroot, hrootC, ← hmw] at hpiopC
  have e := swSrcE_l0_live L₀ src wk x w vs hsc hR hServe hL0 hE hc hF hsx hs hcmsC j r
    (fun k' hk' => hext k' (by omega)) hprevC hRep (by rw [← hk2]; exact hex)
    (fun l => by
      have e' := hicC l
      rw [hsj] at e'
      exact e') hl hpiopC hlo hhi
  rw [hg, hexT] at e
  exact e

set_option maxHeartbeats 1000000 in
/-- **`FinRepErased` from its parts**: at SessionLoopZKDeployed's premises, where every table's rows hold at its base
(`hRows`, `Loop.FinRows`'s body) and the views' masked values below `zp 64` are the request's values plus its pads
(`hMk`), the run from the views is the firewall's up to `eraseZ` at every rep stream, by strong induction on the stream
(`finRepErased_step`, then `viewRep_rounds` and `viewRep_size` at the firewall's rep, `finRepCut`). -/
theorem finRepErased_of (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hS : Loop.RepSrc L₀ src x w) (hS0 : Level0Src src x w)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs)
    (hRows : ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops →
      ∀ j : Fin (Masked.jOf x), ((finS.C x j).T (finS.cz x (finCoins x ops) j)).tab.onVals.RowsAt
        (finS.wt x w j, vs x w ν (finCoins x ops) j)
        (finS.randAt x (finCoins x ops) t j).1.1 (finS.randAt x (finCoins x ops) t j).1.2.1.2.1)
    (hMk : ∀ ν t κ m₀ ρ₀ ops, FinCtx (LeanRelease.ofLoopLink L₀ src (realOf src)) wk x w ν t κ m₀ ρ₀ ops →
      ∀ s0 spec, FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x) →
      ∀ k < 2 * spec.tables.size, ∀ tb hi q f, RepAt L₀.draws src x w (ν, t) ((m₀, ρ₀) :: ops) spec k tb hi q f →
      ∀ i < (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).zp 64,
        ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2)).masked i =
          q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) :
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
  -- the views' run at each stream, with the auditor's rounds
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
    obtain ⟨p', l', hv, -⟩ := finRepErased_step vs L₀ src wk x w hsc hR hServe hS hS0 hL0 hE hc hF hrows hsx hs hcms
      hext hpads hk hprev hex hic hl hRep hic' (hMk ν t κ m₀ ρ₀ ops hF s0 spec hsx k hk tb hi q _ hRep) hcut
    exact ⟨p', tr, l', hv,
      viewRep_rounds vs L₀ src (realOf src) wk hsc hR hServe hE hc hF hsx hs hcms hext hpads hk hprev hex hic hl hv,
      viewRep_size vs hsc hR ν t m₀ ρ₀ ops hsx hs cms hk hex ic hl hv⟩
  intro k hk extras hex ic l hic hl tb hi q f hRep ic' p tr hic' hrun
  exact finRepErased_step vs L₀ src wk x w hsc hR hServe hS hS0 hL0 hE hc hF hrows hsx hs hcms hext hpads hk
    (fun k' hk' => hAll k' (by omega)) hex hic hl hRep hic' (hMk ν t κ m₀ ρ₀ ops hF s0 spec hsx k hk tb hi q f hRep)
    hrun

end Loop

end Claims.ZKServer
