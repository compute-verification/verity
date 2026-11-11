module

public import Proofs.Flock.SessionLoop.RoundsPassTaken
public import Proofs.Flock.SessionLoop.RoundOff
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# A rep's first round, taken without liveness

`LeanRelease.SchedTaken` at rep stream `2j + r`'s first round (`Masked.repBase`), wherever the frame takes its coins
(**`LeanRelease.schedTaken_zcBind`**): the frame starts the rep fresh (`ofLoop_answered_at`, `freshOf_ok`,
`step_prior_fresh`), the scheduled round on the rep's own table is the zerocheck's binding behind the stream's domain
frame and `r_skip`'s squeeze (`ofLoop_sched_unrefused`, `Sched.rounds_two`, `head_encs`), and the frame takes it
(`roundFree_zcHead_out`). This is `zcBind_frame`'s argument with the frame's answer from `refusesCoins` in liveness's
place. `midAll_zcHead`: the binding's frames are none the inner proof's label.
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

theorem midAll_zcHead {dom digest cap : ByteArray} (hdom : dom.size < 2 ^ 64) (hdg : digest.size < 2 ^ 64)
    (hcap : cap.size < 2 ^ 64) :
    Sched.MidAll [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
      .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] := by
  intro f hf
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl | rfl | rfl | rfl
  · exact ⟨hdom, nofun⟩
  · exact ⟨by show "flock-r1cs-v0".toUTF8.size < 2 ^ 64; decide, ne_inner_of_isLabel (by decide)⟩
  · exact ⟨hdg, nofun⟩
  · exact ⟨hcap, nofun⟩
  · exact ⟨by show _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8.size < 2 ^ 64; decide,
      ne_inner_of_isLabel (by decide)⟩

namespace LeanRelease

open Loop in
set_option maxHeartbeats 2000000 in
/-- **The frame takes a rep's first round where it takes its coins**: at a history whose next round is rep stream
`2j + r`'s first (`Masked.repBase`), whose coins the loop's frame takes, on the loop's own tables where `src` samples the
tape's level 0. The frame starts the rep fresh (`ofLoop_answered_at`, `freshOf_ok`, `step_prior_fresh`), the scheduled
round is the binding behind the stream's domain frame on the rep's own table (`ofLoop_sched_unrefused`,
`Sched.rounds_two`, `head_encs`), and the frame takes it (`roundFree_zcHead_out`). -/
theorem schedTaken_zcBind (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hSv : Loop.Serves x) (hS0 : Loop.Level0Src src x w)
    (hST : Loop.ScheduleTotal x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hlt : ops.length < refK x)
    (href : (ofLoop Masked.dataE L₁ (Loop.realOf src)).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (j : Fin (Masked.jOf x)) (r : Fin 2) (hn : ops.length = Masked.repBase x (2 * j.val + r.val)) :
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
  have hbase : Masked.repBase x (2 * j.val + r.val) =
      1 + ((roundsOf spec (Masked.stsOf x)).take (2 * j.val + r.val)).sum := by
    rw [Masked.repBase_eq hsx, repStartZk_eq spec _ hq2.le]
  obtain ⟨q, hq⟩ : ∃ q, ops.length = q + 1 := ⟨ops.length - 1, by omega⟩
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q = (2 * j.val + r.val, 0) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j.val + r.val) 0
      (by rw [hrsl]; exact hq2) (lt_of_lt_of_le (by unfold piopNR; omega) hR0)
    rw [Nat.zero_add, Nat.add_zero] at e
    rw [← e]
    congr 1
    omega
  have hj0 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).2 = 0 := by rw [hloc]
  -- the frame's answer: the rep's fresh state
  have hlast : (ofLoop Masked.dataE L₁ (Loop.realOf src)).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ (Loop.realOf src)) x
  have hk : (ofLoop Masked.dataE L₁ (Loop.realOf src)).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, hA⟩ := answered_of_unrefused href
  have hfresh : freshOf x (ops.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m := by
    rcases ofLoop_answered_at Masked.dataE L₁ (Loop.realOf src) x (ν, t) (m₀, ρ₀) ops hA hsx hq hloc with
      ⟨j', hj', -⟩ | ⟨-, hf⟩
    · omega
    · exact hf
  obtain ⟨c', tb, dom, xs, his, alg, hph0, -, hm'⟩ :=
    freshOf_ok x _ hsx (q := q) (by rw [List.length_map]; omega) hfresh
  rw [hloc] at hph0
  dsimp only at hph0
  have hm0 : m.on = false ∧ m.done = false ∧ m.inner = none ∧ m.pos = 0 := by
    rcases hm' with ⟨-, hm1⟩ | ⟨-, st', hst', hm1⟩
    · rw [hm1]
      exact ⟨rfl, rfl, rfl, rfl⟩
    · rw [hm1]
      exact step_prior_fresh hst'
  -- the loop's own round there
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_unrefused L₁ src x w hx hSv hS0 hST ν t m₀ ρ₀ ops href hk
    hsx (q := q) hq
  have hre' : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  obtain ⟨hT1, -, -, -, -, hTcap⟩ := Count.real_tabAlike (Loop.realTabs_of src x w (ν, t) hsx) hph0 hre'
  obtain ⟨hts', -, -⟩ := Count.placeholder_tab hph0
  rw [hst] at hts'
  have htrm : tr.st.m = (Masked.stAt x j).m := by rw [hT1, hts']; rfl
  have hmT : 14 ≤ tr.st.m := by rw [htrm]; omega
  have hmT' : tr.st.m < 2 ^ 64 := by rw [htrm]; exact lt_of_le_of_lt hm35 (by norm_num)
  obtain ⟨alg', hal'⟩ := schedSoFar_ok_rounds hs'
  have h0' := Sched.rounds_two hmT hmT' hal'
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal' hs'
  obtain ⟨hsz0, -⟩ := Array.getElem?_eq_some_iff.1 h0'.1
  have hlt' : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).2 < sched'.size := by
    rw [hj0, hrest, Array.size_append]
    omega
  have hscheq : (sched'.getD (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).2
      (.empty, 0)).1.data.toList =
      encs [.dom (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 / 2]!)
          ((ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 % 2)).toUTF8,
        .label "flock-r1cs-v0".toUTF8, .bytes tr.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat tr.cm0.cap),
        .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc := by
    rw [hj0, hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hsz0, h0'.1, head_encs]
    simp only [Option.getD_some, List.toList_toArray]
  have hdom := (sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx)
    (k := (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1) (by rw [hloc]; exact hq2)).2.1
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
  have hcoins : (ofLoop Masked.dataE L₁ (Loop.realOf src)).coins x ((m₀, ρ₀) :: ops).length = 6 := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel, hn]
    simpa [piopN] using hNR 0 (by unfold piopNR; omega)
  -- the frame takes it
  have hdg : tr.st.digest.size < 2 ^ 64 := by
    rw [hT1, hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hc : ∀ p, p < ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size →
      ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order[p]! <
        ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.size :=
    Claims.ZKServer.ofDraws_order _ _ _
  obtain ⟨o, tq, hro, -⟩ := roundFree_zcHead_out
    ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    { m with stopped := false } tr.st.digest (_root_.Flock.Firewall.Finish.flat tr.cm0.cap) _ hc hm0.1 hm0.2.1
    hm0.2.2.1 hdg hTcap hdom (by show m.pos ≤ _; rw [hm0.2.2.2]; exact Nat.zero_le _)
  have hK : ((2 : UInt8) = 2 ∨ ((2 : UInt8) = 1 ∧ (6 : ℕ) = 1)) ∧ ¬ ((6 : ℕ) = 0 ∧ (2 : UInt8) = 2) ∧
      (6 : ℕ) < 2 ^ 64 :=
    ⟨.inl rfl, fun h => absurd h.1 (by decide), by norm_num⟩
  rw [← hcoins] at hK hro
  refine ⟨m, _, 2, o, tq, #[], hA, hm0.2.2.1, ?_, midAll_zcHead hdom hdg hTcap, hK.1, hK.2.1, hK.2.2, hro⟩
  rw [ofLoop_scheduled Masked.dataE L₁ (Loop.realOf src) x w (ν, t) (m₀, ρ₀) ops _ hsx (q := q) hq hre hs' hlt',
    hscheq, hcoins]

end LeanRelease

end Claims.ZKServer
