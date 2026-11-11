module

public import Proofs.Flock.SessionLoop.RoundsPassTaken
public import Proofs.Flock.SessionLoop.RoundSwitchRel
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# The ring switch's rounds, taken without liveness

`LeanRelease.SchedTaken` at rep stream `2j + r`'s ring-switch round `k` (`Masked.rankSlot + k`, before the opening's
binding), wherever the frame takes its coins (**`LeanRelease.schedTaken_switch`**): the scheduled round on the rep's own
table is the switch's body `k` (`ofLoop_sched_unrefused`, `Bind.sched_switch_sv`), and the frame takes it. Round `0`
opens with the opening's label, which the frame takes where its rank stop passed (`Ranked.label_pcs_of_rank`); past it
the frame is outside every masked phase. This is `switch_frame`'s argument with the frame's answer from `refusesCoins`
in liveness's place.

* `Loop.roundFree_pcs_ok`: the frame takes a round of the opening's label and plain frames wherever it takes the label.
* `Ranked.repState_switched_in`: `repState_switched`, the frame outside the inner proof as well.
* `LeanRelease.rs_taken`: the frame takes the switch's body `k` from a state outside the inner proof, past the masked
  phase from round `1` on and taking the opening's label at round `0`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker MaskSlot K_SKIP N_INNER LABEL_ZC LABEL_PCS maskRank claimPoints roundFree)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)

namespace Loop

/-- **The frame takes a round of the opening's label and plain frames** from a state outside the inner proof, wherever
it takes the label. -/
theorem roundFree_pcs_ok (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ) {q : Masker × ByteArray}
    (hi : s.inner = none) (hl : s.label c LABEL_PCS.toUTF8 = .ok q)
    (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) :
    ∃ out t fr, roundFree c s
      (ByteArray.mk (encs (.label LABEL_PCS.toUTF8 :: fs) ++ (Frame.sq k n).enc).toArray) n = .ok (out, t, fr) := by
  have hm' : ∀ f ∈ Frame.label LABEL_PCS.toUTF8 :: fs, midFrame f := by
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · show LABEL_PCS.toUTF8.size < 2 ^ 64
      decide
    · exact hm f hf
  have hof := (label_pcs_out c s hl).1
  have hin : q.1.inner = none := by
    cases hon : s.on with
    | false =>
      rw [label_pcs_off c s hon] at hl
      cases hl
      exact hi
    | true => exact (Flags.label_pcs hon hl).2.2.2.2.2.1.trans hi
  have hst : stepAt c (s, ByteArray.empty, #[], 0) (.label LABEL_PCS.toUTF8) =
      .ok (q.1, ByteArray.empty ++ q.2, #[], 0 + 16 + (LABEL_PCS.toUTF8.size + 15) / 16 * 16) := by
    simp only [stepAt]
    rw [hl]
    all_goals rfl
  obtain ⟨r, hr, h1, -⟩ := foldlM_stepAt_off c fs q.1 (ByteArray.empty ++ q.2) #[]
    (0 + 16 + (LABEL_PCS.toUTF8.size + 15) / 16 * 16) 0 hof hp
  have hf : (Frame.label LABEL_PCS.toUTF8 :: fs).foldlM (stepAt c) (s, ByteArray.empty, #[], 0) = .ok r := by
    rw [List.foldlM_cons, hst]
    exact hr
  rw [roundFree_encs c s _ k n (by rw [hi]; simp) hm' hk h0 hn, hf]
  obtain ⟨m, o, fr, cnt⟩ := r
  simp only at h1
  subst h1
  refine ⟨o ++ header 4 k n, { q.1 with owed := some n }, fr, ?_⟩
  simp only [bind, Except.bind, hin]
  rfl

end Loop

namespace Ranked

open Loop

/-- **Past the ring switch's first round the frame is outside every masked phase and the inner proof**:
`repState_switched`, with the flags `Flags.sched_flags` carries. -/
theorem repState_switched_in {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hm : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0)
    (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) {k : ℕ} (hk0 : k ≠ 0)
    (hkN : k ≤ xs.size + 2) (hcr : cr.size = alg.size + k) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (alg.size + k)).answer c'' cs = .ok m) :
    m.on = false ∧ m.inner = none := by
  obtain ⟨-, sched, -, hs⟩ := repState_ok ha
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk' => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk']
  obtain ⟨ha0, ha1⟩ := Sched.rounds_two hm14 hm hal
  have h1s : 1 < alg.size := (Array.getElem?_eq_some_iff.1 ha1).1
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  have hpop : cr.pop.size = alg.size + (k - 1) := by rw [Array.size_pop, hcr]; omega
  obtain ⟨hsz, sv, hsv, hB⟩ := Bind.sched_switch_at hal hs hdom hdg hcap hm hmk hx hk hn hq hpop (by omega)
  have hN0 : xs.size + 2 ≠ 0 := by omega
  have hsv' : ∀ i, (sv i).size < 2 ^ 64 := fun i => by rw [hsv]; decide
  have hp : Flags.PcsRound (sched.getD alg.size (.empty, 0)) := by
    obtain ⟨hb, h7⟩ := hB 0 (Nat.zero_le _)
    rw [Bind.openList_lt t.sch hN0 (by omega)] at h7
    rw [Nat.add_zero] at hb h7
    rw [show sched.getD alg.size (.empty, 0) = ((sched.getD alg.size (.empty, 0)).1, 7) from Prod.ext rfl h7]
    exact Switch.pcsRound_rs sv hN0 (hsv' 0) hb
  have hq' : ∀ k', 1 ≤ k' → k' ≤ alg.size + (k - 1) → k' ≠ alg.size →
      Quiet.QRound (sched.getD k' (.empty, 0)) := by
    intro k' h1 hkJ hkA
    rcases Nat.lt_or_ge k' alg.size with hkl | hkl
    · rw [hg k' hkl, Array.getElem?_eq_getElem hkl, Option.getD_some]
      exact Quiet.rounds_quiet hm hdg hcap hdom hal k' _ h1 (Array.getElem?_eq_getElem hkl)
    · obtain ⟨i, rfl⟩ : ∃ i, k' = alg.size + i := ⟨k' - alg.size, by omega⟩
      obtain ⟨hb, h7⟩ := hB i (by omega)
      rw [Bind.openList_lt t.sch hN0 (by omega)] at h7
      rw [show sched.getD (alg.size + i) (.empty, 0) = ((sched.getD (alg.size + i) (.empty, 0)).1, 7) from
        Prod.ext rfl h7]
      exact Switch.qRound_rs sv (by omega) (by omega) (hsv' i) hb
  rw [show alg.size + k = alg.size + (k - 1) + 1 by omega] at ha
  rw [repState_at c t rep dom xs hi cr0 cr ns (alg.size + (k - 1)) hal hs (by omega)] at ha
  dsimp only at ha
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr (alg.size + (k - 1)))).2
  · rw [h2] at ha
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    rcases Flags.sched_flags hc _ sched cr hdg hcap hdom h0 (by omega) hp _ hq' h2 with ho | ⟨-, hF⟩
    · exact absurd ha (answer_owed_none cs ho)
    · have e := answer_alg c'' _ cs hF.2.2.1 hF.1 ha
      rw [e]
      exact ⟨(hF.2.2.2.2 (by omega)).2.1, hF.2.2.1⟩
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

end Ranked

namespace LeanRelease

open Loop in
/-- **The frame takes the ring switch's body `k`** from state `m` after a history, of `7` coins below `N` and `N` at
`N`: from a state outside the inner proof, past the masked phase from round `1` on and taking the opening's label at
round `0`. -/
theorem rs_taken {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (g : List (Msg × Msg)) {m : Masker}
    {sv : ℕ → Array F128} {N k : ℕ} (hin : m.inner = none) (hon : k ≠ 0 → m.on = false)
    (hlab : k = 0 → ∃ q, ({ m with stopped := false } : Masker).label (L.cfg x ρ g) LABEL_PCS.toUTF8 = .ok q)
    (hkN : k ≤ N) (hN : N ≠ 0) (hN' : N < 2 ^ 64) (hs : ∀ i, (sv i).size < 2 ^ 64)
    (hn : L.coins x g.length = if k < N then 7 else N) : L.RoundTaken x ρ g m (Switch.rsBody sv N [] k) := by
  unfold RoundTaken
  rw [hn]
  by_cases hk : k < N
  · rw [if_pos hk]
    by_cases h0 : k = 0
    · subst h0
      obtain ⟨q, hq⟩ := hlab rfl
      obtain ⟨o, t, fr, hro⟩ := roundFree_pcs_ok (L.cfg x ρ g) { m with stopped := false } (rsFrames (sv 0)) 2 7
        hin hq (rsFrames_mid _ (hs 0)) (rsFrames_plain _) (.inl rfl) (by decide) (by decide)
      rw [← rsBody_zero sv hk] at hro
      refine ⟨.label LABEL_PCS.toUTF8 :: rsFrames (sv 0), 2, o, t, fr, rsBody_zero sv hk, ?_, .inl rfl, by decide,
        by decide, hro⟩
      intro f hf
      rcases List.mem_cons.1 hf with rfl | hf
      · exact ⟨by show LABEL_PCS.toUTF8.size < 2 ^ 64; decide,
          ne_inner_of_isLabel (by rw [String.toUTF8_eq_toByteArray]; exact Ranked.pcs_other.2.1)⟩
      · exact midAll_of_plain (rsFrames_mid _ (hs 0)) (rsFrames_plain _) f hf
    · obtain ⟨o, fr, hro, -⟩ := roundFree_off (L.cfg x ρ g) { m with stopped := false } (rsFrames (sv k)) 2 7
        (hon h0) hin (rsFrames_mid _ (hs k)) (rsFrames_plain _) (.inl rfl) (by decide) (by decide)
      rw [← rsBody_mid sv h0 hk] at hro
      exact ⟨rsFrames (sv k), 2, o, _, fr, rsBody_mid sv h0 hk,
        midAll_of_plain (rsFrames_mid _ (hs k)) (rsFrames_plain _), .inl rfl, by decide, by decide, hro⟩
  · obtain rfl : k = N := by omega
    rw [if_neg (lt_irrefl _)]
    obtain ⟨o, fr, hro, -⟩ := roundFree_off (L.cfg x ρ g) { m with stopped := false } [] 2 k
      (hon hN) hin (fun f hf => by simp at hf) (fun f hf => by simp at hf) (.inl rfl) (fun h => hN h.1) hN'
    rw [← rsBody_last sv k] at hro
    exact ⟨[], 2, o, _, fr, rsBody_last sv k, fun f hf => by simp at hf, .inl rfl, fun h => hN h.1, hN', hro⟩

open Loop Ranked in
set_option maxHeartbeats 4000000 in
/-- **The frame takes a rep's ring-switch round where it takes its coins**: at a history whose next round is rep stream
`2j + r`'s ring-switch round `k` (`Masked.rankSlot + k`, before the opening's binding), whose coins the loop's frame
takes, on the loop's own tables where `src` samples the tape's level 0. The scheduled round is the switch's body `k` on
the rep's own table (`ofLoop_sched_unrefused`, `Bind.sched_switch_sv`, `Loop.loopNr_piop`). At `k = 0` the frame is in
the algebraic phase at its last position (`repState_piop`, `repState_pre_pcs`) and its rank stop passed, so it takes the
opening's label (`label_pcs_of_rank`); past it the frame is outside every masked phase (`repState_switched_in`), and the
frame takes the round (`rs_taken`). -/
theorem schedTaken_switch (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hSv : Loop.Serves x) (hS0 : Loop.Level0Src src x w)
    (hST : Loop.ScheduleTotal x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hlt : ops.length < refK x)
    (href : (ofLoop Masked.dataE L₁ (Loop.realOf src)).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (j : Fin (Masked.jOf x)) (r : Fin 2) {k : ℕ} (hn : ops.length = Masked.rankSlot x j.val r.val + k)
    (hk : ops.length < Masked.repSlot x (2 * j.val + r.val) fun _ => 0) :
    (ofLoop Masked.dataE L₁ (Loop.realOf src)).SchedTaken x w (ν, t) ((m₀, ρ₀) :: ops) := by
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
  -- the frame's answer, past the rank stop
  have hlast : (ofLoop Masked.dataE L₁ (Loop.realOf src)).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ (Loop.realOf src)) x
  have hkd : (ofLoop Masked.dataE L₁ (Loop.realOf src)).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, hA⟩ := answered_of_unrefused href
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same Masked.dataE L₁ (Loop.realOf src) x (ν, t) (m₀, ρ₀) ops hsx (q := i + 1) (by omega) hj1
    ho ht] at hA'
  obtain ⟨hA', hrf⟩ := bind_rankStop_ok hA'
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
  have hm14t : 14 ≤ tb.st.m := by rw [htm]; omega
  have hk6t : 6 ≤ tb.st.fold.kLog := by rw [htk]; exact hk6
  obtain ⟨hsz, -⟩ := Sched.rounds_piop hm14t hmb hal
  obtain ⟨sch', tp', -, hlay0, -⟩ := start_eq hds
  have hnM : lay.nMlv = tb.st.m - 6 := by
    rw [hlay0, htm, hst]
    all_goals rfl
  have hnL : lay.nLc = tb.st.fold.kLog - 6 := by
    rw [hlay0, htk, hst]
    all_goals rfl
  have hbA : (piopV tb.st.m tb.st.fold.kLog tb.st.digest (_root_.Flock.Firewall.Finish.flat tb.cm0.cap) Sched.root0
      (_root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0) (fun _ => 0) alg.size).length ≤
      c'.order.size := by
    rw [hc', Loop.ofDraws_order, absorbed_size]
    exact le_trans (piopV_length_le lay hm14t hk6t hnM hnL hsz.le) (by omega)
  -- past the ring switch's first round the frame is outside every masked phase
  have hsw : k ≠ 0 → m.on = false ∧ m.inner = none := by
    intro hk0'
    have hA'' := hA'
    rw [show a + 1 = alg.size + k by omega] at hA''
    exact repState_switched_in hcc hal (by rw [htm]; omega) hmb hdg (placeholder_cap hph0) (hdm ▸ hdom') hmkt
      (by rw [hxs]; exact hXb) hk0 hnn hq hk0' (by rw [hxs, hXS]; omega)
      (by rw [Array.size_map, Array.size_range]) hA''
  have hin : m.inner = none := by
    rcases Nat.eq_zero_or_pos k with hkz | hkz
    · obtain ⟨-, sched0, -, hs0⟩ := repState_ok hA'
      exact (repState_piop hcc hal hs0 hm14t hmb hdg (placeholder_cap hph0) (hdm ▸ hdom') (a := a) (by omega)
        (by rw [show a + 1 = alg.size by omega]; exact hbA) hA').2.1
    · exact (hsw (by omega)).2
  -- at the ring switch's first round the frame takes the opening's label: its rank stop passed
  have hlab : k = 0 → ∃ q, ({ m with stopped := false } : Masker).label
      ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)) LABEL_PCS.toUTF8 = .ok q := by
    intro hkz
    subst hkz
    have hA3 := hA'
    rw [show a + 1 = alg.size by omega] at hA3
    obtain ⟨CR0, CR, -, hCR, hA3⟩ := repState_args hA3
    have hcs3 : coinsAt x ((m₀, ρ₀) :: ops) = CR.getD (alg.size - 1) #[] := by
      rw [coinsAt_words x _ ops (q := i + 1) (by omega), hCR, crMap_getD _ _ (by omega),
        show i + 1 - alg.size + (alg.size - 1) + 1 = i + 1 by omega]
    obtain ⟨hA'', hcnt, hon0, hd, hl0', hpos', hco, -, -⟩ := repState_pre_pcs hcc hal hm14t hk6t hmb hdg
      (placeholder_cap hph0) (hdm ▸ hdom') hbA hcs3 hA3
    obtain ⟨hCA, -, -, -⟩ := ofLoop_cfg_place Masked.dataE L₁ (Loop.realOf src) x (ν, t) ((m₀, ρ₀) :: ops) hsx
      (q := i + 1) (by simp only [List.length_cons]; omega) hloc hph0
    have hpN : m.pos = c'.nPiop := by
      rw [hpos', hsz, piopV_length_congr tb.st.m tb.st.fold.kLog (digest' := tb.st.digest)
          (cap' := _root_.Flock.Firewall.Finish.flat tb.cm0.cap) (root' := Sched.root0) (lay' := lay) (fun _ => 0)
          (fun _ => 0), piopV_length _ _ _ _ _ _ _ hm14t hk6t hnM hnL, hc', Loop.ofDraws_nPiop, absorbed_size]
      omega
    have hlayC : ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay =
        _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.fold.kLog sch'.k0 := by
      rw [hCA.lay, hc']
      show lay = _
      rw [hlay0, htm, htk, hst]
    obtain ⟨P, hP, -⟩ := frame_points sch'.k0 hm14t hk6t (fun k => CR.getD k #[]) m.coins (hA'' ▸ hco)
      (hA'' ▸ hcnt) (fun i e => (CR.getD i #[])[e]!) (fun _ _ _ _ => rfl)
    exact Ranked.label_pcs_of_rank (s := { m with stopped := false }) hon0 hd hl0' (hpN.trans hCA.nPiop.symm)
      (fun _ _ => ⟨P, by rw [hlayC]; exact hP⟩) hrf
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
    rw [(alg_cnt hmT (by rw [htrk]; omega) hmT' hal').1, htrm, htrk, hAd]
  obtain ⟨hssz, zc, lc, -, hbody⟩ := Bind.sched_switch_sv hal' hs'' hdom' (by rw [hT1]; exact hdg) hcapR hmT'
    hmT (by rw [hT1]; exact hmkt) hXb (by rw [hT2]; exact hk0) (by rw [hT2]; exact hnn) (by rw [hT2]; exact hq)
    (a := k) (by rw [Array.size_map, Array.size_range, hAA']) (by rw [hXS]; omega)
  have hlt' : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 < sched'.size := by
    rw [hj1, hssz, hAA']
    omega
  obtain ⟨-, -, -, h7, h8⟩ := loopNr_piop hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6)
  rw [hst, ← Masked.repBase_eq hsx (2 * j.val + r.val)] at h7 h8
  have hlen : ops.length = Masked.repBase x (2 * j.val + r.val) + 3 + ((Masked.stAt x j).m - 6) + 3 +
      ((Masked.stAt x j).c.kLog - 6) + k := by omega
  have hcoins : (ofLoop Masked.dataE L₁ (Loop.realOf src)).coins x ((m₀, ρ₀) :: ops).length =
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
  -- the frame takes it
  refine schedTaken_of hA hin ?_
  rw [ofLoop_scheduled Masked.dataE L₁ (Loop.realOf src) x w (ν, t) (m₀, ρ₀) ops _ hsx (q := i + 1) (by omega) hre hs'
    hlt', hj1, show a + 1 = alg'.size + k by omega, (hbody k le_rfl).1]
  exact rs_taken (ofLoop Masked.dataE L₁ (Loop.realOf src)) x (ν, t) ((m₀, ρ₀) :: ops) hin (fun h => (hsw h).1)
    hlab (by rw [hXS]; omega) (by omega) hXb (fun _ => by simp [_root_.Flock.Firewall.Finish.switchVec]) hcoins

end LeanRelease

end Claims.ZKServer
