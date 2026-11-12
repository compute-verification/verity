module

public import Proofs.Flock.SessionLoop.RoundsPassTaken
public import Proofs.Flock.SessionLoop.RoundPiop

/-!
# A rep's PIOP round past its first, taken without liveness

`LeanRelease.SchedTaken` at rep stream `2j + r`'s PIOP round `jj ≥ 1`, wherever the frame takes its coins
(**`LeanRelease.schedTaken_piop`**): the frame is in the masked phase at the values before round `jj`
(`repState_piop`), the scheduled round on the rep's own table is the PIOP's frames and squeeze
(`ofLoop_sched_unrefused`, `Sched.rounds_piop`), and the frame takes it (`roundFree_on`). This is `piop_frame`'s
argument with the frame's answer from `refusesCoins` in liveness's place.
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

namespace LeanRelease

open Loop in
set_option maxHeartbeats 2000000 in
/-- **The frame takes a rep's PIOP round past its first where it takes its coins**: at a history whose next round is
rep stream `2j + r`'s PIOP round `jj ≥ 1`, whose coins the loop's frame takes, on the loop's own tables where `src`
samples the tape's level 0. The frame is in the masked phase at the values before round `jj` (`repState_piop`), the
scheduled round is the PIOP's frames and squeeze on the rep's own table (`ofLoop_sched_unrefused`,
`Sched.rounds_piop`), and the frame takes it (`roundFree_on`). -/
theorem schedTaken_piop (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hSv : Loop.Serves x) (hS0 : Loop.Level0Src src x w)
    (hST : Loop.ScheduleTotal x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hlt : ops.length < refK x)
    (href : (ofLoop Masked.dataE L₁ (Loop.realOf src)).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (j : Fin (Masked.jOf x)) (r : Fin 2) {jj : ℕ}
    (hn : ops.length = Masked.repBase x (2 * j.val + r.val) + jj) (h0 : jj ≠ 0)
    (hjj : jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog) :
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
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hrk : (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 =
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! := by
    unfold roundsOf
    rw [List.getD_eq_getElem _ _ (by simp only [List.length_map, List.length_range]; omega)]
    simp
  have hR0 : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
      (roundsOf spec (Masked.stsOf x)).getD (2 * j.val + r.val) 0 := by
    rw [hrk, repRounds_true]
    simp only [hz]
    rw [hst]
    have hA : piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog ≤
        (Masked.stAt x j).m + ((Masked.stAt x j).c.kLog - 6) := by
      unfold piopNR
      omega
    exact le_trans (le_trans (le_trans hA (Nat.le_add_right _ _)) (Nat.le_add_right _ _)) (Nat.le_add_right _ _)
  obtain ⟨a, rfl⟩ : ∃ a, jj = a + 1 := ⟨jj - 1, by omega⟩
  have hbase : Masked.repBase x (2 * j.val + r.val) =
      1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
    rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
  obtain ⟨i, hp⟩ : ∃ i, ops.length = i + 2 := ⟨ops.length - 2, by omega⟩
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1) = (2 * j.val + r.val, a + 1) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) (a + 1)
      (by rw [hrsl]; exact hq2) (lt_of_lt_of_le hjj hR0)
    rw [Nat.zero_add] at e
    rw [← e]
    congr 1
    omega
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 = a + 1 := by rw [hloc]
  -- the frame's answer
  have hlast : (ofLoop Masked.dataE L₁ (Loop.realOf src)).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ (Loop.realOf src)) x
  have hk : (ofLoop Masked.dataE L₁ (Loop.realOf src)).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
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
  obtain ⟨alg, sched, hal, hs⟩ := repState_ok hA'
  obtain ⟨hts', -, -⟩ := Count.placeholder_tab hph0
  rw [hst] at hts'
  have htm : tb.st.m = (Masked.stAt x j).m := by rw [hts']; rfl
  have htk' : tb.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hts']; rfl
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph0
  obtain ⟨sch', tp', hsch', hlay⟩ := start_lay hds
  rw [hz] at hsch'
  simp only [Except.ok.injEq, Prod.mk.injEq] at hsch'
  obtain ⟨rfl, rfl⟩ := hsch'
  rw [hst] at hlay
  have hlayS : lay = Masked.layE x j := hlay
  have hnM : lay.nMlv = tb.st.m - 6 := by
    rw [hlayS, htm]
    try rfl
  have hnL : lay.nLc = tb.st.fold.kLog - 6 := by
    rw [hlayS, htk']
    try rfl
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by
    rw [hc']
    exact Claims.ZKServer.ofDraws_order _ _ _
  have hjjT : a + 1 < piopNR tb.st.m tb.st.fold.kLog := by rw [htm, htk']; exact hjj
  have hjjT' : a < piopNR tb.st.m tb.st.fold.kLog := by omega
  have hb : (piopV tb.st.m tb.st.fold.kLog tb.st.digest (_root_.Flock.Firewall.Finish.flat tb.cm0.cap) Sched.root0
      (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0) (fun _ => 0) (a + 1)).length ≤ c'.order.size := by
    rw [hc', Loop.ofDraws_order, absorbed_size]
    exact le_trans (piopV_length_le lay (by omega) (by omega) hnM hnL hjjT.le) (by omega)
  have hdg : tb.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : tb.st.m < 2 ^ 64 := by rw [htm]; exact lt_of_le_of_lt hm35 (by norm_num)
  obtain ⟨-, hdom', -, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨hon, hin, -, hpos⟩ := repState_piop hcc hal hs (by omega) hmb hdg (placeholder_cap hph0)
    (hdm ▸ hdom') hjjT' hb hA'
  -- the loop's own round there
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_unrefused L₁ src x w hx hSv hS0 hST ν t m₀ ρ₀ ops href hk
    hsx (q := i + 1) (by omega)
  have hre' : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  obtain ⟨hT1, -, -⟩ := Count.real_tabAlike (Loop.realTabs_of src x w (ν, t) hsx) hph0 hre'
  have htrm : tr.st.m = (Masked.stAt x j).m := by rw [hT1, hts']; rfl
  have htrk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hT1, hts']; rfl
  have hmT : 14 ≤ tr.st.m := by rw [htrm]; omega
  have hmT' : tr.st.m < 2 ^ 64 := by rw [htrm]; exact lt_of_le_of_lt hm35 (by norm_num)
  obtain ⟨alg', hal'⟩ := schedSoFar_ok_rounds hs'
  obtain ⟨hsz', hpi'⟩ := Sched.rounds_piop hmT hmT' hal'
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal' hs'
  have hjjR : a + 1 < piopNR tr.st.m tr.st.fold.kLog := by rw [htrm, htrk]; exact hjj
  have hlt' : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 < sched'.size := by
    rw [hloc, hrest, Array.size_append, hsz']
    dsimp only
    omega
  have hscheq : (sched'.getD (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2
      (.empty, 0)).1.data.toList =
      encs (piopFr tr.st.m tr.st.fold.kLog tr.st.digest (_root_.Flock.Firewall.Finish.flat tr.cm0.cap) Sched.root0
        (_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0) (fun _ => 0) (a + 1)) ++
        (Frame.sq (piopK (a + 1)) (piopN tr.st.m (a + 1))).enc := by
    rw [hloc]
    dsimp only
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left (by rw [hsz']; exact hjjR),
      hpi' (a + 1) (by omega) hjjR]
    simp only [Option.getD_some, List.toList_toArray]
  -- the loop's coin counts there
  have hpl := Count.piop_list (kLog := (Masked.stAt x j).c.kLog) (by omega : 14 ≤ (Masked.stAt x j).m)
  have hpL := congrArg List.length hpl
  simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil] at hpL
  have hNR : ∀ jj < piopNR (Masked.stAt x j).m (Masked.stAt x j).c.kLog,
      loopNr x (Masked.repBase x (2 * j.val + r.val) + jj) = piopN (Masked.stAt x j).m jj := by
    intro jj hjj'
    unfold piopNR at hjj'
    rw [Masked.repBase_eq hsx, loopNr_plan hsx hq2 hz (by rw [hst]; exact hk6) (i := jj)
      (by unfold FlockComplete.piopRoundsZk; rw [hst]; omega),
      Count.plan_get hz (by rw [List.length_append, hst, hpL]; omega), hst,
      List.getElem?_append_left (by rw [hpL]; omega), hpl]
    exact piopN_list (by omega) hk6 (by unfold piopNR; omega)
  have hcoins : (ofLoop Masked.dataE L₁ (Loop.realOf src)).coins x ((m₀, ρ₀) :: ops).length =
      piopN tr.st.m (a + 1) := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel, hn, hNR (a + 1) hjj, htrm]
  -- the frame takes it
  have hF0 : ∀ f ∈ piopFr tr.st.m tr.st.fold.kLog tr.st.digest (_root_.Flock.Firewall.Finish.flat tr.cm0.cap)
      Sched.root0 (_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0) (fun _ => 0) (a + 1),
      midFrame f ∧ plainFrame f := by
    intro f hf
    rcases Nat.eq_zero_or_pos a with rfl | ha
    · simp [piopFr] at hf
    · obtain ⟨h1, h2⟩ := piopFr_frames _ _ _ _ _ _ _ (show 2 ≤ a + 1 by omega)
      exact ⟨h1 f hf, h2 f hf⟩
  have hKN := piopKN_ok (m := tr.st.m) (jj := a + 1) hmT hmT' (by omega)
  have hCA := (ofLoop_cfg_place Masked.dataE L₁ (Loop.realOf src) x (ν, t) ((m₀, ρ₀) :: ops) hsx (q := i + 1)
    (by simp only [List.length_cons]; omega) hloc hph0).1
  have hFl : (fvals (piopFr tr.st.m tr.st.fold.kLog tr.st.digest (_root_.Flock.Firewall.Finish.flat tr.cm0.cap)
      Sched.root0 (_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0) (fun _ => 0) (a + 1))).length =
      (fvals (piopFr tb.st.m tb.st.fold.kLog tb.st.digest (_root_.Flock.Firewall.Finish.flat tb.cm0.cap)
        Sched.root0 (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0) (fun _ => 0) (a + 1))).length := by
    rw [hT1]
    exact fvals_piopFr_length _ _ _ _ _
  have hV := piopV_length_le (m := tb.st.m) (kLog := tb.st.fold.kLog) (digest := tb.st.digest)
    (cap := _root_.Flock.Firewall.Finish.flat tb.cm0.cap) (root := Sched.root0)
    (lay := _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0) (mk := fun _ => 0) lay (by omega) (by omega)
    hnM hnL (a := a + 1 + 1) (by omega)
  rw [piopV_succ, List.length_append, ← hpos] at hV
  have hbO : ({ m with stopped := false } : Masker).pos +
      nAbs (piopFr tr.st.m tr.st.fold.kLog tr.st.digest (_root_.Flock.Firewall.Finish.flat tr.cm0.cap) Sched.root0
        (_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0) (fun _ => 0) (a + 1)) ≤
      ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size := by
    show m.pos + _ ≤ _
    rw [nAbs_fvals, hFl, hCA.order, hc', Loop.ofDraws_order, absorbed_size]
    omega
  obtain ⟨out, fr, hrf, -⟩ := roundFree_on ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    { m with stopped := false } _ (piopK (a + 1)) (piopN tr.st.m (a + 1)) (order_lt_of_alike hCA hcc) hon hin
    (fun f hf => (hF0 f hf).1) (fun f hf => (hF0 f hf).2) hKN.1 hKN.2.1 hKN.2.2 hbO
  rw [← hcoins] at hKN hrf
  refine ⟨m, _, piopK (a + 1), out, _, fr, hA, hin, ?_, midAll_of_plain (fun f hf => (hF0 f hf).1)
    (fun f hf => (hF0 f hf).2), hKN.1, hKN.2.1, hKN.2.2, hrf⟩
  rw [ofLoop_scheduled Masked.dataE L₁ (Loop.realOf src) x w (ν, t) (m₀, ρ₀) ops _ hsx (q := i + 1) (by omega) hre hs'
    hlt', hscheq, hcoins]

end LeanRelease

end Claims.ZKServer
