module

public import Proofs.Flock.SessionLoop.RoundSwitchRank

/-!
# The frame's release at a rep's ring switch's rounds

At rep stream `2j + r`'s ring switch's round `k`, `rankSlot + k`, the loop schedules the switch's body `k` at its own
table's switched vectors, and the frame releases it as it came.

* **`LeanRelease.switch_frame`**: on a live history whose next round is the ring switch's round `k`, the loop's table
  there has the rep's setup, its zerocheck's and lincheck's claims are at the points of the openings' words from the
  rep's start, and the release is the switch's body `k` at the table's switched vectors of those claims and the extra
  claims at the link points off the first opening's words (`Bind.sched_switch_sv`, `Loop.check_rs`).
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

open Loop Ranked in
set_option maxHeartbeats 2000000 in
/-- **The frame releases a rep's ring switch's round as the loop schedules it**: at a live history whose next round is
rep stream `2j + r`'s ring-switch round `k` (`rankSlot + k`, before the rep's Ligerito rounds), on the rep's own tables
where `real` answers the loop's, the loop's table there is of the rep's setup, the zerocheck's and lincheck's claims of
the schedule are at the points of the openings' words from the rep's start (`Loop.pts_raw`), and the release is the
switch's body `k` (`Switch.rsBody`) at the table's switched vectors of those claims and the extra claims at the link
points off the first opening's words. The frame took the round's coins in the algebraic phase at `k = 0`, past it later
(`Ranked.repState_switched`); the loop scheduled the body (`Bind.sched_switch_sv`), whose coins it asks for
(`Loop.loopNr_piop`), and the check passes it as it came (`Loop.check_rs`). -/
theorem switch_frame (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoop Masked.dataE L₁ real).toSrv.Worker) {x : Msg} {w : Flock.Wit} (hx : Masked.Live x)
    (hSv : Loop.Serves x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoop Masked.dataE L₁ real).toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)))
    (hlt : ops.length < refK x)
    (hl : (ofLoop Masked.dataE L₁ real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) {v : Msg}
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {k : ℕ}
    (hn : ops.length = Masked.rankSlot x j.val r.val + k)
    (hk : ops.length < Masked.repSlot x (2 * j.val + r.val) fun _ => 0) :
    ∃ (tr : Tab) (hir : Array ByteArray) (zc : _root_.Flock.ZcOut) (lc : _root_.Flock.LcOut),
      real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) ∧
      tr.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) ∧
      FlockComplete.LigLock.Fed.PtsAt tr.st.m tr.st.fold.kLog
        (fun p e => (wordsAt x (ops.map fun p => CoinLayer.rd p.2) p)[e]!) (Masked.repBase x (2 * j.val + r.val))
        zc lc ∧
      k ≤ (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
        (Masked.stAt x j) (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0))).size + 2 ∧
      (ofLoop Masked.dataE L₁ real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
        0 :: Switch.rsBody (fun i => _root_.Flock.Firewall.Finish.switchVec tr.g
          (_root_.Flock.abcClaims tr.st.m tr.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf
            (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
              (Masked.stAt x j) (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0))))[i]!)
          ((_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
            (Masked.stAt x j) (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0))).size + 2) [] k := by
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hmk : (Masked.stAt x j).merkle = _root_.Flock.MerkleScheme.hm96Sha512 := (htabs j.val hj).2.2
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hrk : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  -- the round's place: `A + k` into the rep's rounds, `A` the algebraic phase's
  obtain ⟨A, hAd⟩ : ∃ A, A = (Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6 := ⟨_, rfl⟩
  have hRk : Masked.rankSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) + A := by
    rw [hAd]
    show Masked.repBase x (2 * j.val + r.val) + ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) = _
    rw [hstj]
  have hRS : Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) =
      Masked.repBase x (2 * j.val + r.val) + A + (3 + spec.points * (Masked.stAt x j).regions.size) := by
    rw [Masked.repSlot_eq hsx hz, hst, ← Masked.repBase_eq hsx (2 * j.val + r.val), hAd]
    unfold FlockComplete.piopRoundsZk
    try dsimp only
    omega
  have hkN : k ≤ 2 + spec.points * (Masked.stAt x j).regions.size := by omega
  have hbs1 : 1 ≤ Masked.repBase x (2 * j.val + r.val) := by
    rw [Masked.repBase_eq hsx]
    unfold FlockComplete.repStartZk
    omega
  have hpos : 0 < ops.length := by omega
  -- the extra claims: the table's regions at each link point
  obtain ⟨hmp, hmp0, hpc⟩ := Masked.live_links hx hsx hq2
  have hmp' : (Masked.stAt x j).mPts = spec.mPts := by rw [← hst]; exact hmp
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hW0 : (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0).size = spec.points * spec.mPts := by
    have e := wordsAt_size x ops (s := ops.length) (p := 0) hpos le_rfl
    rw [List.take_length] at e
    rw [e, hl0, Nat.min_eq_left hpc]
  have hXS : (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
      (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))).size =
      spec.points * (Masked.stAt x j).regions.size := by
    rw [Count.extras_size, Count.linkPoints_size, hW0, hmp', Nat.mul_div_cancel _ hmp0]
    exact Nat.mul_comm _ _
  obtain ⟨-, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  rw [hst] at hxr
  have hXb : (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
      (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))).size + 2 < 2 ^ 64 := by
    rw [Count.extras_size]
    have := Nat.mul_le_mul_left (Masked.stAt x j).regions.size
      ((Loop.linkPoints_size_le (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0)).trans
        (show _ ≤ _root_.Flock.Firewall.Frames.COIN_MAX by rw [hW0]; exact hpc))
    omega
  -- the located round
  have hbase : Masked.repBase x (2 * j.val + r.val) =
      1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
    rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
  obtain ⟨NR, hNd⟩ : ∃ NR, NR = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! +
      ((Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) := ⟨_, rfl⟩
  have hR4 : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 = NR + 4 := by
    have e1 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + ((Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6) := by
      rw [hst]; omega
    rw [hrk, repRounds_true]
    simp only [hz]
    rw [hNd]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hR0 : A + k < (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    rw [hR4, hNd, hst]
    unfold FlockComplete.piopRoundsZk
    omega
  obtain ⟨i, hp⟩ : ∃ i, ops.length = i + 2 := ⟨ops.length - 2, by omega⟩
  obtain ⟨a, ha⟩ : ∃ a, A + k = a + 1 := ⟨A + k - 1, by omega⟩
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1) = (2 * j.val + r.val, a + 1) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) (a + 1)
      (by rw [hrsl]; exact hq2) (by omega)
    rw [Nat.zero_add] at e
    rw [← e]
    congr 1
    omega
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 = a + 1 := by rw [hloc]
  -- the frame took the round's coins
  have hlast : (ofLoop Masked.dataE L₁ real).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ real) x
  have hkd : (ofLoop Masked.dataE L₁ real).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, -, -, hA, -⟩ :=
    check_of_live (ofLoop Masked.dataE L₁ real) wk x w (ν, t) κ ((m₀, ρ₀) :: ops) v hl hkd hv
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same Masked.dataE L₁ real x (ν, t) (m₀, ρ₀) ops hsx (q := i + 1) (by omega) hj1 ho ht] at hA'
  replace hA' := (bind_rankStop_ok hA').1
  -- the loop's placeholder table and its state there
  have hcs : (ops.map fun p => ZKMal.CoinLayer.rd p.2).length = i + 2 := by rw [List.length_map]; omega
  have ho' : (progOf x (ops.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun h => answer_owed_none _ h hA'
  obtain ⟨c', tb, dom, xs, his, hph0⟩ := progOf_placeholder x _ hsx hcs ho'
  rw [locate_pred _ 0 i a hj1, hloc] at hph0
  dsimp only at hph0
  have hph : placeholder spec (Masked.stsOf x) (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).1
      (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', tb, dom, xs, his) := by
    rw [hloc]
    exact hph0
  rw [progOf_at x _ hsx (q := i + 1) (by rw [List.length_map]; omega) hj1 hph, hloc] at hA'
  dsimp only at hA'
  obtain ⟨alg, -, hal, -⟩ := repState_ok hA'
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph0
  obtain ⟨hm36, hk0, hnn, hq⟩ := Count.sched_facts htp
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph0
  rw [hst] at hts' hxs
  have htm : tb.st.m = (Masked.stAt x j).m := by rw [hts']; rfl
  have htk : tb.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hts']; rfl
  have hdg : tb.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : tb.st.m < 2 ^ 64 := by rw [htm]; exact lt_of_le_of_lt hm35 (by norm_num)
  have hmkt : tb.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by rw [hts']; exact hmk
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by
    rw [hc']
    exact Claims.ZKServer.ofDraws_order _ _ _
  have hAA : alg.size = A := by
    rw [(alg_cnt (by rw [htm]; omega) (by rw [htk]; omega) hmb hal).1, htm, htk, hAd]
  -- past the ring switch's first round the frame is outside every masked phase
  have hon : k ≠ 0 → m.on = false := by
    intro hk0'
    have hA'' := hA'
    rw [show a + 1 = alg.size + k by omega] at hA''
    exact repState_switched hcc hal (by rw [htm]; omega) hmb hdg (placeholder_cap hph0) (hdm ▸ hdom') hmkt
      (by rw [hxs]; exact hXb) hk0 hnn hq hk0' (by rw [hxs, hXS]; omega)
      (by rw [Array.size_map, Array.size_range]) hA''
  -- the loop's own round there
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_live Masked.dataE L₁ real wk x w (ν, t) κ (m₀, ρ₀) ops hl hkd
    hsx (q := i + 1) (by omega)
  have hre' : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  obtain ⟨hT1, hT2, -, -, -, hcapR⟩ := Count.real_tabAlike hRT hph0 hre'
  have hs'' : schedSoFar tr ((2 * j.val + r.val) % 2)
      (spec.domain (spec.tables[(2 * j.val + r.val) / 2]!) ((2 * j.val + r.val) % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
          (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hir
      ((Array.range (a + 1)).map fun q =>
        wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (i + 1 - (a + 1) + q + 1)) = .ok sched' := by
    rw [hloc] at hs'
    exact hs'
  rw [hst] at hs''
  have hcr : ((Array.range (a + 1)).map fun q =>
      wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (i + 1 - (a + 1) + q + 1)) =
      (Array.range (A + k)).map fun q =>
        wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + q) := by
    rw [ha]
    refine Array.map_congr_left fun q _ => ?_
    congr 1
    omega
  rw [hcr] at hs''
  obtain ⟨alg', hal'⟩ := schedSoFar_ok_rounds hs''
  have htrm : tr.st.m = (Masked.stAt x j).m := by rw [hT1, htm]
  have htrk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hT1, htk]
  have hmT : 14 ≤ tr.st.m := by rw [htrm]; omega
  have hmT' : tr.st.m < 2 ^ 64 := by rw [htrm]; exact lt_of_le_of_lt hm35 (by norm_num)
  have hAA' : alg'.size = A := by
    rw [(alg_cnt hmT (by rw [htrk]; omega) hmT' hal').1, htrm, htrk, hAd]
  obtain ⟨hssz, zc, lc, hptsR, hbody⟩ := Bind.sched_switch_sv hal' hs'' hdom' (by rw [hT1]; exact hdg) hcapR hmT'
    hmT (by rw [hT1]; exact hmkt) hXb (by rw [hT2]; exact hk0) (by rw [hT2]; exact hnn) (by rw [hT2]; exact hq)
    (a := k) (by rw [Array.size_map, Array.size_range, hAA']) (by rw [hXS]; omega)
  have hlt' : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 < sched'.size := by
    rw [hj1, hssz, hAA']
    omega
  have hsc : ∀ vals, (ofLoop Masked.dataE L₁ real).scheduled x w (ν, t) ((m₀, ρ₀) :: ops) vals =
      (ByteArray.mk (Switch.rsBody (fun i => _root_.Flock.Firewall.Finish.switchVec tr.g
        (_root_.Flock.abcClaims tr.st.m tr.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf
          (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
            (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))))[i]!)
        ((_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
          (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))).size + 2) [] k).toArray
      ).data.toList := by
    intro vals
    rw [ofLoop_scheduled Masked.dataE L₁ real x w (ν, t) (m₀, ρ₀) ops vals hsx (q := i + 1) (by omega) hre hs' hlt',
      hj1, show a + 1 = alg'.size + k by omega, (hbody k le_rfl).1]
    all_goals exact (List.toList_toArray _).symm
  -- the release: the scheduled body as it came
  obtain ⟨m2, out, s', hA2, hck, hrel⟩ := release_of_live (ofLoop Masked.dataE L₁ real) wk x w (ν, t) κ
    ((m₀, ρ₀) :: ops) v hl hkd hv hsc
  have hm2 : m2 = m := Except.ok.inj (hA2.symm.trans hA)
  rw [hm2] at hck
  obtain ⟨-, -, -, h7, h8⟩ := loopNr_piop hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6)
  rw [hst, ← Masked.repBase_eq hsx (2 * j.val + r.val)] at h7 h8
  have hlen : ops.length = Masked.repBase x (2 * j.val + r.val) + 3 + ((Masked.stAt x j).m - 6) + 3 +
      ((Masked.stAt x j).c.kLog - 6) + k := by omega
  have hcoins : (ofLoop Masked.dataE L₁ real).coins x ((m₀, ρ₀) :: ops).length =
      if k < (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
        (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))).size + 2 then 7
      else (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
        (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))).size + 2 := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel, hlen, hXS]
    split_ifs with hkl
    · exact h7 k (by omega)
    · rw [show k = 2 + spec.points * (Masked.stAt x j).regions.size by omega, h8]
      omega
  have hout := Loop.check_rs (m := { m with stopped := false }) hon
    (show k ≤ (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints
      (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))).size + 2 by rw [hXS]; omega)
    (by omega) hXb (fun _ => by simp [_root_.Flock.Firewall.Finish.switchVec]) hcoins hck
  refine ⟨tr, hir, zc, lc, hre', hT1.trans hts', ?_, by rw [hXS]; omega, ?_⟩
  · exact pts_raw (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2)) (R := Masked.repBase x (2 * j.val + r.val))
      (by rw [hAA', htrm, htrk, hAd]; omega) (by rw [hAA']; omega) hptsR
  · rw [hrel, hout]

end LeanRelease

end Claims.ZKServer
