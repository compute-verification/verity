module

public import Proofs.Flock.SessionLoop.SchedAt
public import Proofs.Flock.SessionLoop.LiveInner
public import Proofs.Flock.SessionLoop.RoundLigFrameOff
public import Proofs.Flock.Claims.ZK.L0PinBind
public import Proofs.Flock.Claims.ZK.FramePhase

/-!
# The frame at a rep's Ligerito rounds from level 0's close

At rep stream `2j + r`'s Ligerito rounds from level 0's closing round (`Masked.repSlot` `k0 + 4`) up to the last before
the inner proof, the frame takes the round's coins in level 0 if in a masked phase at all at the close, whose label
leaves it, and outside every masked phase past it (`LigOff.repState_close`), so its check passes the worker's round only
as the round it schedules, and releases that round as it came (`LigOff.check_pastAt`). The scheduled round is the rep's
opening fed the history's coins (`openStep`), which past the ring switch is the cut run of `ligeritoPZk` at the rep's
real table, the switch's target and basis, zero pads, the tape's extras, lane pads and salts and no nonces
(`Bind.sched_lig`), and from the close on the closing label, good frames and a squeeze (`LigOff.sched_past`).

* **`LeanRelease.release_lig_frame`**: there, for every coin function `coin` that is the history's coins from the rep's
  opening on (`CoinAt`), `ligeritoPZk`'s cut run at `coin`, cut at the round, sends at the round the body whose release
  is the worker's round, with no leading domain frame; it squeezes at each round before it as many coins as the history's
  coins there hold; and its basis is the switch's at the rep's algebraic coins and link points.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

open Loop Ranked in
set_option maxHeartbeats 4000000 in
/-- **The frame at a rep's Ligerito rounds from level 0's close, at the loop's own tables**: `release_lig_frame` at the
release `ofLoop Masked.dataE L₁ real`, on the rep's own tables where `real` answers the loop's. The frame took the round's
coins in level 0 if anywhere and, past the close, outside every masked phase (`LigOff.repState_close`, at the
placeholder); the loop scheduled the rep's opening's round there (`Bind.sched_lig`, at the real table), which from the
close on is `PastRound`'s (`LigOff.sched_past`) and whose coins it asks for (`loopNr_rep`, the plan's count), and the
check passes it as it came (`LigOff.check_pastAt`). -/
theorem lig_frame (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoop Masked.dataE L₁ real).toSrv.Worker) {x : Msg} {w : Flock.Wit} (hx : Masked.Live x)
    (hSv : Loop.Serves x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoop Masked.dataE L₁ real).toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hRT : Count.RealTabs (Masked.stsOf x) (real x w (ν, t)))
    (hlt : ops.length < refK x)
    (hl : (ofLoop Masked.dataE L₁ real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) {v : Msg}
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : (Masked.zkTab x j).sch.k0 + 4 ≤ i)
    (h2 : i < (Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) :
    ∃ (tb : Tab) (hi : Array ByteArray) (xs : Array (Array F128)) (algC : Array (Array F128)) (base : ℕ)
      (cs : List (Array F128)),
      real x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi) ∧
      xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[j.val]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[j.val]!
          (Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) ∧
      algC.size ≤ base ∧
      Masked.repBase x (2 * j.val + r.val) + algC.size = base ∧
      (∀ a (ha : a < algC.size),
        algC[a] = Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (base - algC.size + a)) ∧
      base + cs.length = ops.length ∧
      (∀ a (ha : a < cs.length), cs[a] = Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (base + a)) ∧
      base + (xs.size + 2) + 1 = Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) ∧
      ∀ coin : ℕ → ℕ → F128, FlockComplete.LigLock.Fed.CoinAt coin base cs →
        ∃ (T : F128) (sws : Array _root_.Flock.Switched) (tr₂ : _root_.Flock.Transcript) (l₂ L : Array Claims.Msg),
          l₂.size = base + (xs.size + 2) + 1 ∧
          (∃ zc lc, FlockComplete.LigLock.Fed.PtsAt tb.st.m tb.st.fold.kLog
              (FlockComplete.LigLock.Fed.recCoin algC) 0 zc lc ∧
            sws = FlockComplete.Replay.Rs.sws
              (_root_.Flock.abcClaims tb.st.m tb.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf xs)
              ⟨fun k i => (cs.getD k #[])[i]!, fun i => (cs.getD (xs.size + 2) #[])[i]!⟩) ∧
          (((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin ops.length)
              (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) _root_.Flock.MerkleScheme.hm96Sha512 tb.sch T
              tb.cm0.toC tb.g (FlockComplete.basisVals sws (tb.st.m - 7))
              (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0)
              (Array.replicate (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).len 0) r.val tb.tPad
              tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty))
              (fun _ _ => 0)).run tr₂).go l₂).2 = L ∧
          (∀ p, l₂.size ≤ p → p < ops.length → ∀ (t' : _root_.Flock.Transcript) (kind : UInt8) (k : ℕ),
            k < 2 ^ 64 → L[p]? = some (Claims.toMsg (t'.framed kind k)) →
            (Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) p).size = k) ∧
          (ofLoop Masked.dataE L₁ real).release x (ν, t) ((m₀, ρ₀) :: ops) false v = 0 :: L[ops.length]! ∧
          dropDom (0 :: L[ops.length]!) = 0 :: L[ops.length]! := by
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
  -- the round's place: `a + 1` into the rep's rounds, past its algebraic phase's `A`
  obtain ⟨A, hAd⟩ : ∃ A, A = (Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6 := ⟨_, rfl⟩
  have hRS : ∀ f : _root_.Flock.Schedule → ℕ, Masked.repSlot x (2 * j.val + r.val) f =
      Masked.repBase x (2 * j.val + r.val) + A + (3 + spec.points * (Masked.stAt x j).regions.size) +
        f (Masked.zkTab x j).sch := by
    intro f
    rw [Masked.repSlot_eq hsx hz, hst, ← Masked.repBase_eq hsx (2 * j.val + r.val), hAd]
    unfold FlockComplete.piopRoundsZk
    try dsimp only
    omega
  have hRSi : Masked.repSlot x (2 * j.val + r.val) (fun _ => i) =
      Masked.repBase x (2 * j.val + r.val) + A + (3 + spec.points * (Masked.stAt x j).regions.size) + i :=
    hRS fun _ => i
  have hRS0 : Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) =
      Masked.repBase x (2 * j.val + r.val) + A + (3 + spec.points * (Masked.stAt x j).regions.size) + 0 :=
    hRS fun _ => 0
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
  obtain ⟨i0, hp⟩ : ∃ i0, ops.length = i0 + 2 := ⟨ops.length - 2, by omega⟩
  obtain ⟨a, ha⟩ : ∃ a, ops.length = Masked.repBase x (2 * j.val + r.val) + (a + 1) :=
    ⟨ops.length - Masked.repBase x (2 * j.val + r.val) - 1, by omega⟩
  have hR0 : a + 1 < (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    rw [hR4, hNd, hst]
    unfold FlockComplete.piopRoundsZk
    omega
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i0 + 1) = (2 * j.val + r.val, a + 1) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) (a + 1)
      (by rw [hrsl]; exact hq2) hR0
    rw [Nat.zero_add] at e
    rw [← e]
    congr 1
    omega
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i0 + 1)).2 = a + 1 := by rw [hloc]
  -- the frame took the round's coins
  have hlast : (ofLoop Masked.dataE L₁ real).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ real) x
  have hkd : (ofLoop Masked.dataE L₁ real).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, -, -, hA, -⟩ :=
    check_of_live (ofLoop Masked.dataE L₁ real) wk x w (ν, t) κ ((m₀, ρ₀) :: ops) v hl hkd hv
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same Masked.dataE L₁ real x (ν, t) (m₀, ρ₀) ops hsx (q := i0 + 1) (by omega) hj1 ho ht] at hA'
  replace hA' := (bind_rankStop_ok hA').1
  -- the loop's placeholder table and its state there
  have hcs : (ops.map fun p => ZKMal.CoinLayer.rd p.2).length = i0 + 2 := by rw [List.length_map]; omega
  have ho' : (progOf x (ops.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun h => answer_owed_none _ h hA'
  obtain ⟨c', tph, dom, xs, his, hph0⟩ := progOf_placeholder x _ hsx hcs ho'
  rw [locate_pred _ 0 i0 a hj1, hloc] at hph0
  dsimp only at hph0
  have hph : placeholder spec (Masked.stsOf x) (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i0 + 1)).1
      (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', tph, dom, xs, his) := by
    rw [hloc]
    exact hph0
  rw [progOf_at x _ hsx (q := i0 + 1) (by rw [List.length_map]; omega) hj1 hph, hloc] at hA'
  dsimp only at hA'
  obtain ⟨alg, -, hal, -⟩ := repState_ok hA'
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph0
  obtain ⟨hm36, hk0, hnn, hq⟩ := Count.sched_facts htp
  have hn0 : tph.sch.n ≠ 0 := by
    have := (Count.sched_rho htp).1
    omega
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph0
  rw [hst] at hts' hxs
  have hsch : tph.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (Except.ok.inj (htp.symm.trans hz))).1
  have hk0P : tph.sch.k0 = (Masked.zkTab x j).sch.k0 := by rw [hsch]
  have hlvP : tph.sch.levels.size = (Masked.zkTab x j).sch.levels.size := by rw [hsch]
  have htm : tph.st.m = (Masked.stAt x j).m := by rw [hts']; rfl
  have htk : tph.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hts']; rfl
  have hdg : tph.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : tph.st.m < 2 ^ 64 := by rw [htm]; exact lt_of_le_of_lt hm35 (by norm_num)
  have hmkt : tph.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by rw [hts']; exact hmk
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by
    rw [hc']
    exact Claims.ZKServer.ofDraws_order _ _ _
  have hAA : alg.size = A := by
    rw [(alg_cnt (by rw [htm]; omega) (by rw [htk]; omega) hmb hal).1, htm, htk, hAd]
  have hXSP : xs.size = spec.points * (Masked.stAt x j).regions.size := by rw [hxs]; exact hXS
  -- the rep's real table, of the placeholder's shape
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_live Masked.dataE L₁ real wk x w (ν, t) κ (m₀, ρ₀) ops hl hkd
    hsx (q := i0 + 1) (by omega)
  have hre' : real x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  have hT := Count.real_tabAlike hRT hph0 hre'
  obtain ⟨hdgR, hmbR, -, -, hpR, hdR⟩ := Count.real_bounds hph0 hT
  obtain ⟨hT1, hT2, hT3, -, -, hcapR⟩ := hT
  -- at the close the frame is in level 0 if in a masked phase at all, past it outside every masked phase
  have hdomP : dom.size < 2 ^ 64 := hdm ▸ hdom'
  have hOLP := Count.openList_length htp (c := xs.size + 2) (by omega)
  obtain ⟨hlm, honm⟩ := LigOff.repState_close hcc hal (by rw [htm]; omega) hmb hdg (placeholder_cap hph0) hdomP hmkt
    (by rw [hxs]; exact hXb) hk0 hnn hq hn0 (by rw [← hT3]; exact hpR) (by rw [← hT2]; exact hdR)
    (by rw [Array.size_map, Array.size_range]; omega) (by rw [Array.size_map, Array.size_range, hOLP]; omega)
    (by rw [Array.size_map, Array.size_range]) hA'
  -- the loop's own round there: the real table's schedule at the rep's coins from its first round
  have hs'' : schedSoFar tr ((2 * j.val + r.val) % 2)
      (spec.domain (spec.tables[(2 * j.val + r.val) / 2]!) ((2 * j.val + r.val) % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j.val + r.val) / 2]!
          (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hir
      ((Array.range (a + 1)).map fun q =>
        wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (i0 + 1 - (a + 1) + q + 1)) = .ok sched' := by
    rw [hloc] at hs'
    exact hs'
  have hcr : ((Array.range (a + 1)).map fun q =>
      wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (i0 + 1 - (a + 1) + q + 1)) =
      (Array.range (a + 1)).map fun q =>
        wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + q) := by
    refine Array.map_congr_left fun q _ => ?_
    congr 1
    omega
  have hmod : (2 * j.val + r.val) % 2 = r.val := by omega
  have hdomR := hdom'
  rw [hst, hcr, hmod] at hs''
  rw [hmod] at hdomR
  obtain ⟨cr, hcrd⟩ : ∃ cr : Array (Array F128), cr = (Array.range (a + 1)).map fun q =>
      wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + q) := ⟨_, rfl⟩
  obtain ⟨xs', hxs'd⟩ : ∃ xs' : Array (Array F128), xs' = _root_.Flock.Firewall.Drive.extras (Masked.stAt x j)
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stAt x j) (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0)) :=
    ⟨_, rfl⟩
  obtain ⟨dm, hdmd⟩ : ∃ dm : ByteArray, dm = (spec.domain (spec.tables[(2 * j.val + r.val) / 2]!) r.val).toUTF8 :=
    ⟨_, rfl⟩
  rw [← hcrd, ← hxs'd, ← hdmd] at hs''
  rw [← hdmd] at hdomR
  have hXS' : xs'.size = spec.points * (Masked.stAt x j).regions.size := by rw [hxs'd]; exact hXS
  have hXb' : xs'.size + 2 < 2 ^ 64 := by rw [hxs'd]; exact hXb
  have hcrsz : cr.size = a + 1 := by rw [hcrd, Array.size_map, Array.size_range]
  have hcrget : ∀ q (hq : q < cr.size),
      cr[q] = wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.repBase x (2 * j.val + r.val) + q) := by
    intro q hq
    subst hcrd
    simp
  obtain ⟨alg', hal'⟩ := schedSoFar_ok_rounds hs''
  have htrm : tr.st.m = (Masked.stAt x j).m := by rw [hT1, htm]
  have htrk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hT1, htk]
  have hmT : 14 ≤ tr.st.m := by rw [htrm]; omega
  have hAA' : alg'.size = A := by
    rw [(alg_cnt hmT (by rw [htrk]; omega) hmbR hal').1, htrm, htrk, hAd]
  have hk0R : 1 ≤ tr.sch.k0 := by rw [hT2]; exact hk0
  have hnR : tr.sch.n < 2 ^ 62 := by rw [hT2]; exact hnn
  have hqR : ∀ i, (tr.sch.levels.getD i default).q < 2 ^ 63 := by rw [hT2]; exact hq
  have hn0R : tr.sch.n ≠ 0 := by rw [hT2]; exact hn0
  have hmkR : tr.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by rw [hT1]; exact hmkt
  have hkR : tr.sch.k0 = (Masked.zkTab x j).sch.k0 := by rw [hT2, hsch]
  have hlvR : tr.sch.levels.size = (Masked.zkTab x j).sch.levels.size := by rw [hT2, hsch]
  have hschR : _root_.Flock.Zk.schedule (Masked.stAt x j).m = .ok (tr.sch, tp) := by rw [hT2, ← hst]; exact htp
  have hOL := Count.openList_length hschR (c := xs'.size + 2) (by omega)
  have hC1 : alg'.size + (xs'.size + 2) + 1 + tr.sch.k0 + 4 ≤ cr.size := by omega
  have hC2 : cr.size < alg'.size + (Count.openList (xs'.size + 2) tr.sch).length := by omega
  obtain ⟨hlt', hpast⟩ := LigOff.sched_past hal' hs'' hdomR hdgR hcapR hmbR hmkR hXb' hk0R hnR hqR hmT hn0R hpR hdR
    hC2 hC1 le_rfl
  -- the round's coins: the plan's count, the real schedule's
  have hsc : ∀ vals, (ofLoop Masked.dataE L₁ real).scheduled x w (ν, t) ((m₀, ρ₀) :: ops) vals =
      (sched'.getD cr.size (.empty, 0)).1.data.toList := by
    intro vals
    rw [ofLoop_scheduled Masked.dataE L₁ real x w (ν, t) (m₀, ρ₀) ops vals hsx (q := i0 + 1) (by omega) hre hs'
      (by rw [hj1, ← hcrsz]; exact hlt'), hj1, hcrsz]
  have hcoins : (ofLoop Masked.dataE L₁ real).coins x ((m₀, ρ₀) :: ops).length =
      (sched'.getD cr.size (.empty, 0)).2 := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel,
      show ops.length = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) + cr.size by
        rw [← Masked.repBase_eq hsx (2 * j.val + r.val)]; omega,
      loopNr_rep hsx hq2 (by rw [hcrsz]; exact hR0), hst]
    have hpre := Count.sched_prefix tr r.val dm xs' hir cr sched' hs'' hdomR hdgR hcapR hmbR hXb' hk0R hnR hqR
    rw [hT1, hts', hXS', Nat.add_comm (spec.points * (Masked.stAt x j).regions.size) 2] at hpre
    refine Count.getD_of_prefix (hpre.trans ?_) hlt'
    rw [Count.repPlan_list hschR]
    exact List.prefix_append _ _
  -- the release: the scheduled round as it came
  obtain ⟨m2, out, s', hA2, hck, hrel⟩ := release_of_live (ofLoop Masked.dataE L₁ real) wk x w (ν, t) κ
    ((m₀, ρ₀) :: ops) v hl hkd hv hsc
  have hm2 : m2 = m := Except.ok.inj (hA2.symm.trans hA)
  rw [hm2, hcoins] at hck
  obtain ⟨-, hout⟩ := LigOff.check_pastAt (m := { m with stopped := false }) hC1 hpast hlm
    (fun h => honm (by rw [Array.size_map, Array.size_range]; omega)) hck
  have hLsz : (cr.extract 0 alg'.size).size = alg'.size := by
    simp only [Array.size_extract]
    omega
  have hcsl : (cr.extract alg'.size cr.size).toList.length = cr.size - alg'.size := by
    simp only [Array.length_toList, Array.size_extract]
    omega
  have hcsget : ∀ a' (ha' : a' < (cr.extract alg'.size cr.size).toList.length),
      (cr.extract alg'.size cr.size).toList[a'] = wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2)
        (Masked.repBase x (2 * j.val + r.val) + alg'.size + a') := by
    intro a' ha'
    rw [Array.getElem_toList, Array.getElem_extract, hcrget]
    congr 1
    omega
  refine ⟨tr, hir, xs', cr.extract 0 alg'.size, Masked.repBase x (2 * j.val + r.val) + alg'.size,
    (cr.extract alg'.size cr.size).toList, hre', by rw [hxs'd, hstj], by omega, by omega, ?_, by omega,
    fun a' ha' => hcsget a' ha', by omega, ?_⟩
  · intro a' ha'
    rw [Array.getElem_extract, hcrget]
    congr 1
    omega
  -- at each coin function that is the history's from the opening on, Ligerito's cut log
  intro coin hcoin
  obtain ⟨-, T, sws, tr₂, l₂, L, hl2, hsws, hrun, -, hLn, hcnt⟩ := Bind.sched_lig hal' hs'' hdomR hdgR hcapR hmbR
    hmkR hXb' hk0R hnR hqR hmT (by omega) hC2 coin
    (Array.replicate (Masked.repBase x (2 * j.val + r.val) + alg'.size) ([] : Msg))
    (by rw [Array.size_replicate]; exact hcoin) (n := ops.length) (by rw [Array.size_replicate]; omega)
  rw [Array.size_replicate] at hl2 hcnt
  have hLb : L[ops.length]! = (sched'.getD cr.size (.empty, 0)).1.data.toList := by
    rw [getElem!_def, hLn]
    try rfl
  obtain ⟨fs, kk, hb, hfs, -, -, -⟩ := hpast
  have hM : ∀ f ∈ FlockComplete.LigShape.preAt (alg'.size + (xs'.size + 2) + 1 + tr.sch.k0 + 4) cr.size ++ fs,
      noDom f := by
    intro f hf
    rcases List.mem_append.1 hf with hf | hf
    · unfold FlockComplete.LigShape.preAt at hf
      split_ifs at hf
      · rw [List.mem_singleton.1 hf]
        trivial
      · cases hf
    · exact (hfs f hf).2.2
  refine ⟨T, sws, tr₂, l₂, L, hl2, hsws, hrun, ?_, ?_, ?_⟩
  · intro p hp1 hp2 t' kind k hk hLp
    have e := hcnt p (by omega) hp2 (t'.framed kind k) hLp
    rw [List.getD_eq_getElem _ _ (by omega), hcsget _ (by omega)] at e
    have hpe : Masked.repBase x (2 * j.val + r.val) + alg'.size +
        (p - (Masked.repBase x (2 * j.val + r.val) + alg'.size)) = p := by omega
    rw [hpe] at e
    rw [e]
    exact Sched.count_header t'.pending kind k hk
  · rw [hrel, hout, hLb]
  · rw [hLb, hb]
    exact dropDom_noDom _ _ _ hM

/-- **The frame at a rep's Ligerito rounds from level 0's close**: at rep stream `2j + r`'s round `i` past its PIOP
(`Masked.repSlot` `i`), from level 0's close at `k0 + 4` to the last before the inner proof, the rep's real table is
`tb` with salts `hi`, and the history's coins from round `base` on are `cs`, the rep's opening's (its ring switch's,
`γ`'s and Ligerito's), after its algebraic phase's `algC`, the coins from the rep's first round (`Masked.repBase`). For
every coin function that is `cs` from `base` on, `ligeritoPZk`'s cut run at it, cut at the round and started past the
ring switch (at Ligerito's first round, `Masked.repSlot` `0`), at the switch's target `T` and basis, the table's `cm0`,
`g`, extras, lane pads and salts, zero pads and no nonces, logs `L`, where:

* the release of the worker's round is `L`'s body at the round, tagged `0`, which has no leading domain frame;
* each body `L` holds before the round squeezes as many coins as the history's coins at its round hold;
* the switch's basis is `Rs.sws` at the rep's claims (the algebraic phase's at `algC`, then the link's at the
  history's first coin) and `cs`'s first `xs.size + 3` coins. -/
theorem release_lig_frame (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) {v : Msg} (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : (Masked.zkTab x j).sch.k0 + 4 ≤ i)
    (h2 : i < (Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) :
    ∃ (tb : Tab) (hi : Array ByteArray) (xs : Array (Array F128)) (algC : Array (Array F128)) (base : ℕ)
      (cs : List (Array F128)),
      real x w (ν, t) (2 * j.val + r.val) = .ok (tb, hi) ∧
      xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[j.val]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[j.val]!
          (Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) ∧
      algC.size ≤ base ∧
      Masked.repBase x (2 * j.val + r.val) + algC.size = base ∧
      (∀ a (ha : a < algC.size),
        algC[a] = Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (base - algC.size + a)) ∧
      base + cs.length = ops.length ∧
      (∀ a (ha : a < cs.length), cs[a] = Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) (base + a)) ∧
      base + (xs.size + 2) + 1 = Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) ∧
      ∀ coin : ℕ → ℕ → F128, FlockComplete.LigLock.Fed.CoinAt coin base cs →
        ∃ (T : F128) (sws : Array _root_.Flock.Switched) (tr₂ : _root_.Flock.Transcript) (l₂ L : Array Claims.Msg),
          l₂.size = base + (xs.size + 2) + 1 ∧
          (∃ zc lc, FlockComplete.LigLock.Fed.PtsAt tb.st.m tb.st.fold.kLog
              (FlockComplete.LigLock.Fed.recCoin algC) 0 zc lc ∧
            sws = FlockComplete.Replay.Rs.sws
              (_root_.Flock.abcClaims tb.st.m tb.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf xs)
              ⟨fun k i => (cs.getD k #[])[i]!, fun i => (cs.getD (xs.size + 2) #[])[i]!⟩) ∧
          (((FlockComplete.ligeritoPZk (FlockComplete.asksCut coin ops.length)
              (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) _root_.Flock.MerkleScheme.hm96Sha512 tb.sch T
              tb.cm0.toC tb.g (FlockComplete.basisVals sws (tb.st.m - 7))
              (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0)
              (Array.replicate (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0).len 0) r.val tb.tPad
              tb.extra tb.lanePad (fun l => _root_.Flock.Firewall.Finish.saltAt (hi.getD (l - 1) ByteArray.empty))
              (fun _ _ => 0)).run tr₂).go l₂).2 = L ∧
          (∀ p, l₂.size ≤ p → p < ops.length → ∀ (t' : _root_.Flock.Transcript) (kind : UInt8) (k : ℕ),
            k < 2 ^ 64 → L[p]? = some (Claims.toMsg (t'.framed kind k)) →
            (Loop.wordsAt x (ops.map fun p => CoinLayer.rd p.2) p).size = k) ∧
          (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v = 0 :: L[ops.length]! ∧
          dropDom (0 :: L[ops.length]!) = 0 :: L[ops.length]! := by
  have hx : Masked.Live x := Masked.live_of hSc hR
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) (Masked.lt_of_jOf j))
  exact lig_frame (withFiles (withLink L₀ src) src) real wk hx hSv ν t κ m₀ ρ₀ ops
    (by rw [hreal]; exact Loop.realTabs_of src x w (ν, t) hsx) hlt hl hv j r hn h1 h2

end LeanRelease

end Claims.ZKServer
