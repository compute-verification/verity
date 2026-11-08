module

public import Proofs.Flock.SessionLoop.RoundsPassTaken
public import Proofs.Flock.SessionLoop.RoundOffOpenBind
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# The opening's binding, taken without liveness

`LeanRelease.SchedTaken` at rep stream `2j + r`'s round after its PIOP (`Masked.repSlot` `0`), wherever the frame takes
its coins (**`LeanRelease.schedTaken_openBind`**): the frame is past the algebraic phase there
(`Bind.repState_opened`), the scheduled round on the rep's own table is the opening's binding and `z`'s squeeze
(`ofLoop_sched_unrefused`, `Bind.sched_openBind`, `Loop.loopNr_l0`), and the frame takes it (`roundFree_openBind`). This is
`openBind_frame`'s argument with the frame's answer from `refusesCoins` in liveness's place. `midAll_openBind`: the
binding's frames are none the inner proof's label.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)

theorem midAll_openBind (T : F128) (cap : ByteArray) (hcap : cap.size < 2 ^ 64) : Sched.MidAll (openBind T cap) := by
  intro f hf
  simp only [openBind, List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl | rfl | rfl
  · exact ⟨by show "flock-ligerito-basis-f256-split-v0".toUTF8.size < 2 ^ 64; decide,
      ne_inner_of_isLabel (by decide)⟩
  · exact ⟨trivial, nofun⟩
  · exact ⟨hcap, nofun⟩
  · exact ⟨by show _root_.Flock.Zk.LABEL_OPEN.toUTF8.size < 2 ^ 64; decide, ne_inner_of_isLabel (by decide)⟩

namespace LeanRelease

open Loop in
set_option maxHeartbeats 4000000 in
/-- **The frame takes the opening's binding where it takes its coins**: at a history whose next round is rep stream
`2j + r`'s round after its PIOP (`Masked.repSlot` `0`), whose coins the loop's frame takes, on the loop's own tables
where `src` samples the tape's level 0. The frame is past the algebraic phase there (`Bind.repState_opened`), the
scheduled round is the opening's binding and `z`'s squeeze on the rep's own table (`ofLoop_sched_unrefused`,
`Bind.sched_openBind`, `Loop.loopNr_l0`), and the frame takes it (`roundFree_openBind`). -/
theorem schedTaken_openBind (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hSv : Loop.Serves x) (hS0 : Loop.Level0Src src x w)
    (hST : Loop.ScheduleTotal x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hlt : ops.length < refK x)
    (href : (ofLoop Masked.dataE L₁ (Loop.realOf src)).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (j : Fin (Masked.jOf x)) (r : Fin 2) (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => 0) :
    (ofLoop Masked.dataE L₁ (Loop.realOf src)).SchedTaken x w (ν, t) ((m₀, ρ₀) :: ops) := by
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
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
  -- the round's place: `A + k` into the rep's rounds, `A` the algebraic phase's, `k` past the ring switch
  obtain ⟨A, hAd⟩ : ∃ A, A = (Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6 := ⟨_, rfl⟩
  obtain ⟨k, hkk⟩ : ∃ k, k = 3 + spec.points * (Masked.stAt x j).regions.size := ⟨_, rfl⟩
  have hRS : Masked.repSlot x (2 * j.val + r.val) (fun _ => 0) =
      Masked.repBase x (2 * j.val + r.val) + A + (3 + spec.points * (Masked.stAt x j).regions.size) := by
    rw [Masked.repSlot_eq hsx hz, hst, ← Masked.repBase_eq hsx (2 * j.val + r.val), hAd]
    unfold FlockComplete.piopRoundsZk
    try dsimp only
    omega
  have hnk : ops.length = Masked.repBase x (2 * j.val + r.val) + A + k := by rw [hn, hRS, hkk]
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
    have hbase : Masked.repBase x (2 * j.val + r.val) =
        1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
      rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
    omega
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 = a + 1 := by rw [hloc]
  -- the frame's answer
  have hlast : (ofLoop Masked.dataE L₁ (Loop.realOf src)).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ (Loop.realOf src)) x
  have hkd : (ofLoop Masked.dataE L₁ (Loop.realOf src)).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, hA⟩ := answered_of_unrefused href
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same Masked.dataE L₁ (Loop.realOf src) x (ν, t) (m₀, ρ₀) ops hsx (q := i + 1) (by omega) hj1
    ho ht] at hA'
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
    rw [(Ranked.alg_cnt (by rw [htm]; omega) (by rw [htk]; omega) hmb hal).1, htm, htk, hAd]
  -- past the ring switch the frame is outside every masked phase, the algebraic phase done
  have hA'' := hA'
  rw [show a + 1 = alg.size + (xs.size + 2) + 1 by rw [hxs, hXS]; omega] at hA''
  obtain ⟨hdA, honA, -, hl0A, hiA, -, -⟩ := Bind.repState_opened hcc hal (by rw [htm]; omega) hmb hdg
    (placeholder_cap hph0) (hdm ▸ hdom') hmkt (by rw [hxs]; exact hXb) hk0 hnn hq
    (by rw [Array.size_map, Array.size_range]) hA''
  have hlig : ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lig = true := by
    rw [(ofLoop_cfg_place Masked.dataE L₁ (Loop.realOf src) x (ν, t) ((m₀, ρ₀) :: ops) hsx (q := i + 1)
      (by simp only [List.length_cons]; omega) hloc hph0).1.lig, hc']
    obtain ⟨sch1, tp1, hsch1, hlay1⟩ := start_lay hds
    show decide (lay.k0 > 0) = true
    rw [hlay1]
    exact decide_eq_true (Count.sched_facts hsch1).2.1
  -- the loop's own round there
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_unrefused L₁ src x w hx hSv hS0 hST ν t m₀ ρ₀ ops href hkd
    hsx (q := i + 1) (by omega)
  have hre' : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  obtain ⟨hT1, hT2, -, -, -, hcapR⟩ := Count.real_tabAlike (Loop.realTabs_of src x w (ν, t) hsx) hph0 hre'
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
    rw [(Ranked.alg_cnt hmT (by rw [htrk]; omega) hmT' hal').1, htrm, htrk, hAd]
  have hsch : tb.sch = (Masked.zkTab x j).sch := (Prod.mk.inj (Except.ok.inj (htp.symm.trans hz))).1
  have htrz : tr.sch = (Masked.zkTab x j).sch := hT2.trans hsch
  have hn0r : tr.sch.n ≠ 0 := by
    rw [hT2]
    have := (Count.sched_rho htp).1
    omega
  have hn64 : tr.sch.n < 2 ^ 64 := by rw [hT2]; exact lt_trans hnn (by norm_num)
  obtain ⟨hssz, _zc, _lc, -, T, -, hbody, -⟩ := Bind.sched_openBind hal' hs'' hdom' (by rw [hT1]; exact hdg) hcapR
    hmT' (by rw [hT1]; exact hmkt) hXb (by rw [hT2]; exact hk0) (by rw [hT2]; exact hnn) (by rw [hT2]; exact hq)
    hn0r (by rw [Array.size_map, Array.size_range, hAA', hXS]; omega) hmT (p := a + 1) (by rw [hAA', hXS]; omega)
  have hlt' : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 < sched'.size := by
    rw [hj1]
    exact hssz
  have hcoins : (ofLoop Masked.dataE L₁ (Loop.realOf src)).coins x ((m₀, ρ₀) :: ops).length = tr.sch.n := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel, hn, Masked.repSlot_eq hsx hz, htrz]
    exact (loopNr_l0 hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6) (i := 0) (Nat.zero_le _)).trans
      (if_pos rfl)
  -- the frame takes it
  obtain ⟨o, hro, -⟩ := roundFree_openBind
    ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    { m with stopped := false } T (FlockComplete.flat tr.cm0.toC.cap) tr.sch.n honA hdA hlig hl0A hiA hcapR hn0r
    hn64
  have hK : ((2 : UInt8) = 2 ∨ ((2 : UInt8) = 1 ∧ tr.sch.n = 1)) ∧ ¬ (tr.sch.n = 0 ∧ (2 : UInt8) = 2) ∧
      tr.sch.n < 2 ^ 64 :=
    ⟨.inl rfl, fun h => hn0r h.1, hn64⟩
  rw [← hcoins] at hK hro
  refine ⟨m, _, 2, o, _, #[], hA, hiA, ?_, midAll_openBind T (FlockComplete.flat tr.cm0.toC.cap) hcapR, hK.1,
    hK.2.1, hK.2.2, hro⟩
  rw [ofLoop_scheduled Masked.dataE L₁ (Loop.realOf src) x w (ν, t) (m₀, ρ₀) ops _ hsx (q := i + 1) (by omega) hre hs'
    hlt', hj1, hbody, hcoins]

end LeanRelease

end Claims.ZKServer
