module

public import Proofs.Flock.ZkAny.RelayRankStop
public import Proofs.Flock.ZkAny.RelayOpenFirst
public import Proofs.Flock.ZkAny.RelayTaken
public import Proofs.Flock.SessionLoop
public import Proofs.Flock.Claims.ZK.FrameReal
public import Proofs.Flock.Claims.Complete.ZkTables.Scope
public import Proofs.Flock.Claims.Complete.ZkTables.Table
public import Proofs.Flock.Claims.ZK.FrameBeta

/-!
# The scheduled round where the model's rank stop fails

Where the coins after the loop's state at a rep's round leave a state at which the rank stop fails, the round is the
algebraic phase's last plus one (`Ranked.repState_rankFails`), and the model's scheduled round there, on the loop's own
table, is the opening's first, or the empty round where that table or its schedule is refused:

* `Bind.sched_open`: on as many rounds' coins as the algebraic phase has, a rep's schedule so far holds one round past
  them (`Count.openStep_end`, `feed_part`).
* `Loop.tabs_of_live`: each table of a `Live` statement has `m ≥ 14`, `k_log ≥ 6` and the `hm96Sha512` scheme.
* **`LeanRelease.ofLoop_rank_sched`**: at the loop's own frames, after `e0 :: h1`, whose next round is rep stream
  `k`'s round `j + 1`, where the coin step leaves a state at which the rank stop fails, the scheduled round is the
  empty round or the opening's first, at some count: the placeholder's algebraic phase has `j + 1` rounds, the loop's
  own table's as many (`Ranked.alg_cnt`, `Sched.TabAlike`), and its schedule's round `j + 1` is the opening's first
  (`Bind.sched_open`, `Bind.sched_pcs`).
-/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock (V F128 SessionSpec Stmt)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS)

namespace Bind

open FlockComplete.LigLock.Fed (feedR sentQ)
open Loop

/-- **Past the algebraic phase a rep's schedule has a next round**: on as many rounds' coins as the algebraic phase
has, the schedule so far holds a round past them, the opening's driver asking for its first (`Count.openStep_end`,
`feed_part`). -/
theorem sched_open {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hcr : cr.size = alg.size) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hm : t.st.m < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0) (hn : t.sch.n < 2 ^ 62)
    (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) : alg.size < sched.size := by
  obtain ⟨hdrop, hnf⟩ := schedSoFar_sentQ hal hs hcr.ge
  have hl0 : (cr.extract alg.size cr.size).toList.length = 0 := by
    rw [Array.length_toList, Array.size_extract, hcr, Nat.min_self, Nat.sub_self]
  have hl := feed_part _ (cr.extract alg.size cr.size).toList _
    (Count.openStep_end t rep dom xs hi (cr.extract 0 alg.size) hdom hdg hcap hm hx hk hn hq)
    (by rw [hl0]; exact Nat.lt_of_le_of_lt (Nat.zero_le _) (openList_len t.sch (by omega))) hnf
  have e := congrArg List.length hdrop
  rw [hl, hl0, List.length_drop, List.length_map, Array.length_toList] at e
  omega

end Bind

namespace Loop

/-- **The tables of a `Live` statement** have `m ≥ 14`, `k_log ≥ 6` and the `hm96Sha512` scheme
(`FlockComplete.ZkTables.tables_of_setup`, each table's `ZkTab`). -/
theorem tabs_of_live {x : Msg} (hx : ZKMal.Masked.Live x) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts)) {j : ℕ} (hj : j < sts.size) :
    14 ≤ sts[j]!.m ∧ 6 ≤ sts[j]!.c.kLog ∧ sts[j]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
  have hso : ZKMal.Masked.stsOf x = sts := by
    unfold ZKMal.Masked.stsOf
    rw [hs]
  subst hso
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  obtain ⟨y⟩ := hx.2 j hj
  rw [getElem!_pos _ j hj]
  exact ⟨by have := (hall j hj).1; omega, y.wf.1.hk6, (hall j hj).2.2⟩

end Loop

namespace LeanRelease

open Loop

/-- **Where the model's rank stop fails, its scheduled round is the empty round or the opening's first**: at the
loop's own frames on the tables `real`, the loop's own where they answer (`Count.RealTabs`), after `e0 :: h1`, whose
next round is rep stream `k`'s round `j + 1`, at a setup passing `sizesOk` whose table `k / 2` has `m ≥ 14`,
`k_log ≥ 6` and the `hm96Sha512` scheme, where the coin step leaves `m` and the rank stop fails at `m`, the model's
scheduled round is the empty round, where the loop's own table or its schedule is refused, or the opening's first, at
some count. The coin step is the loop's state at the round (`progOf_at`) taking the coins, so `j + 1` is the
placeholder's algebraic phase's round count (`Ranked.repState_rankFails`); the loop's own table is of the placeholder's
setup (`Count.real_tabAlike`), so its algebraic phase has as many rounds (`Ranked.alg_cnt`), and its schedule so far on
the rep's `j + 1` rounds' coins has the opening's first at round `j + 1` (`Bind.sched_open`, `Bind.sched_pcs`), which
`schedOf` schedules (`schedOf_at`). -/
theorem ofLoop_rank_sched (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      Nat → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Claims.Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (vals : String → Option ByteArray)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q k j : Nat}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hz : _root_.Flock.Firewall.Session.sizesOk spec sts = true) (hR : Count.RealTabs sts (real x w ρ))
    (hq : h1.length = q + 1) (hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1))
    (hk : k < spec.repStreams.size) (hm14 : 14 ≤ sts[k / 2]!.m) (hk6 : 6 ≤ sts[k / 2]!.c.kLog)
    (hmk : sts[k / 2]!.merkle = _root_.Flock.MerkleScheme.hm96Sha512) {m : Masker}
    (hA : (ofLoop T L₀ real).answeredR x ρ (e0 :: h1) = .ok m)
    (hr : rankFails ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) m = true) :
    (ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals = [] ∨
      ∃ n', Flags.PcsRound (⟨((ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals).toArray⟩, n') := by
  obtain ⟨-, -, hF⟩ := answeredR_ok hA
  obtain ⟨cs, hcs⟩ : ∃ cs, h1.map (fun p => ZKMal.CoinLayer.rd p.2) = cs := ⟨_, rfl⟩
  have hq1 : cs.length = q + 1 := by rw [← hcs, List.length_map, hq]
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = k := by rw [hki]
  have hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1 := by rw [hki]
  rw [ofLoop_frameAnswerR_same T L₀ real x ρ e0 h1 hx hq hj, hcs] at hF
  obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := by
    rcases q with _ | q'
    · have := locate_zero (roundsOf spec sts) 0
      rw [hki] at this
      exact absurd this (Nat.succ_ne_zero _)
    · exact ⟨q', rfl⟩
  have hp := locate_pred (roundsOf spec sts) 0 q' j hj
  obtain ⟨c', t, dom, xs, hi, hph⟩ :=
    progOf_placeholder x cs hx (p := q') (by rw [hq1]) (fun ho => answer_owed_none _ ho hF)
  rw [hp, hki] at hph
  have hph' : placeholder spec sts k (wordsAt x cs 0) = .ok (c', t, dom, xs, hi) := hph
  rw [progOf_at x cs hx hq1 hj (by rw [hk1]; exact hph'), hj, hk1] at hF
  obtain ⟨lay, mask, hds, hc', hdm, hxe⟩ := placeholder_ok hph'
  obtain ⟨-, hd, hx2, -⟩ := sizesOk_at hz hk
  have hls : (wordsAt x cs 0).size = min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    rw [← hcs]
    cases h1 with
    | nil => simp at hq
    | cons p h2 => simp [wordsAt, FlockDeploy.Coin.vals]
  have hdom : dom.size < 2 ^ 64 := by rw [hdm]; exact hd
  have hxs : xs.size + 2 < 2 ^ 64 := by
    rw [hxe, Loop.extras_size]
    have := Nat.mul_le_mul_left sts[k / 2]!.regions.size
      ((Loop.linkPoints_size_le sts[k / 2]! (wordsAt x cs 0)).trans (hls.trans_le (Nat.min_le_right _ _)))
    omega
  obtain ⟨hts', ⟨_, htp⟩, -⟩ := Count.placeholder_tab hph'
  obtain ⟨hm36, hk0, hn, hlq⟩ := Count.sched_facts htp
  have htm : t.st.m = sts[k / 2]!.m := by rw [hts']; rfl
  have htk : t.st.fold.kLog = sts[k / 2]!.c.kLog := by rw [hts']; rfl
  have hmb : t.st.m < 2 ^ 64 := by rw [htm]; exact Nat.lt_trans hm36 (by decide)
  have hdg : t.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmkt : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by rw [hts']; exact hmk
  obtain ⟨alg, sched, hal, hs⟩ := repState_ok hF
  have hj' : j < sched.size + INNER_ROUNDS := by
    by_contra hc
    exact answer_owed_none _ (repState_past c' t (k % 2) dom xs hi _ _ _ hal hs (by omega)) hF
  obtain ⟨sch', _, -, hlay0, -⟩ := start_eq hds
  have hcl := (ofLoop_cfg_place T L₀ real x ρ (e0 :: h1) hx (by rw [List.length_cons]; omega) hki hph').1
  have hlay : ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)).lay =
      _root_.Flock.Zk.PadLayout.new t.st.m t.st.fold.kLog sch'.k0 := by
    rw [hcl.lay, hc']
    show lay = _
    rw [hlay0, hts']
    rfl
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by
    rw [hc']
    exact Claims.ZKServer.ofDraws_order _ _ _
  have hj1 := Ranked.repState_rankFails hcc hal hs (by rw [htm]; exact hm14) (by rw [htk]; exact hk6) hmb hdg
    (placeholder_cap hph') hdom hlay hj' hF hr
  have hsc : ∀ b, schedOf (real x w ρ) x cs = some b →
      (ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals = b.data.toList := by
    intro b hb
    show (match schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) with
      | some r => r.data.toList
      | none => L₀.scheduled x w ρ (e0 :: h1) vals) = _
    rw [hcs, hb]
  have hso := schedOf_at (real := real x w ρ) hx hq1 hki
  rcases hre : real x w ρ k with _ | ⟨tr, hr'⟩
  · refine Or.inl ((hsc .empty ?_).trans rfl)
    rw [hso]
    simp only [hre]
  have hT := Count.real_tabAlike hR hph' hre
  obtain ⟨hdgr, hmbr, -, -, -, -⟩ := Count.real_bounds hph' hT
  have hstr : tr.st = t.st := hT.1
  have hschr : tr.sch = t.sch := hT.2.1
  have hcapr : (_root_.Flock.Firewall.Finish.flat tr.cm0.cap).size < 2 ^ 64 := hT.2.2.2.2.2
  rcases hrs : schedSoFar tr (k % 2) dom xs hr'
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1)) with _ | rs
  · have hrs' := hrs
    rw [hdm, hxe] at hrs'
    refine Or.inl ((hsc .empty ?_).trans rfl)
    rw [hso]
    simp only [hre, hrs']
  obtain ⟨algr, halr⟩ := schedSoFar_ok_rounds hrs
  have hA1 := (Ranked.alg_cnt (by rw [htm]; exact hm14) (by rw [htk]; exact hk6) hmb hal).1
  have hA2 := (Ranked.alg_cnt (by rw [hstr, htm]; exact hm14) (by rw [hstr, htk]; exact hk6) hmbr halr).1
  rw [hstr] at hA2
  have hcr : ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1)).size = algr.size := by
    rw [Array.size_map, Array.size_range]
    omega
  have hk0r : 1 ≤ tr.sch.k0 := by rw [hschr]; exact hk0
  have hnr : tr.sch.n < 2 ^ 62 := by rw [hschr]; exact hn
  have hqr : ∀ i, (tr.sch.levels.getD i default).q < 2 ^ 63 := by rw [hschr]; exact hlq
  have hmkr : tr.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by rw [hstr]; exact hmkt
  have hlt := Bind.sched_open halr hrs hcr hdom hdgr hcapr hmbr hxs hk0r hnr hqr
  have hpc := Bind.sched_pcs halr hrs hcr.ge hlt hdom hdgr hcapr hmbr hmkr hxs hk0r hnr hqr
  have hj2 : j + 1 < rs.size := by omega
  have hg : rs.getD algr.size (.empty, 0) = rs[j + 1] := by
    rw [Array.getD_eq_getD_getElem?, show algr.size = j + 1 by omega, Array.getElem?_eq_getElem hj2,
      Option.getD_some]
  rw [hg] at hpc
  have hsr : schedOf (real x w ρ) x cs = some rs[j + 1].1 := by
    have hrs' := hrs
    rw [hdm, hxe] at hrs'
    rw [hso]
    simp only [hre, hrs', Array.getElem?_eq_getElem hj2, Option.map_some]
  rw [hsc _ hsr]
  exact Or.inr ⟨rs[j + 1].2, hpc⟩

end LeanRelease

end Claims.ZKServer
