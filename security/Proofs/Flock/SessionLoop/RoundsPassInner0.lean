module

public import Proofs.Flock.SessionLoop.RoundsPassInner
public import Proofs.Flock.SessionLoop.RoundsPassLig
public import Proofs.Flock.SessionLoop.RoundLigFrameOff
import all Init.Data.String.Defs

/-!
# The inner proof's round 0 passes at `valsDev`

At the end of a rep's schedule `withInner`'s session loop sends the frame's own inner round 0 (`Frame.innerSched`): the
inner proof's label, `final_c'` and a squeeze of one coin. Past level 0's close every round of the schedule leaves the
frame past the opening, outside every masked phase, past level 0's close and outside the inner proof (`LigOff.EndFlags`),
so the frame takes the inner proof's label there freeing no word; `valsDev` writes the honest `final_c'`, which is the
one `finHonest` holds, and the frame's check passes the round.

* `Loop.roundFree_own0_ok`, `Loop.ownRound0_frames`, `Loop.setFinalC_inner0_self`: past the opening and level 0's close,
  outside every masked phase and the inner proof, the frame takes its own round 0 freeing no word; the round is frames
  and a squeeze; `final_c'` set to the one it holds leaves it as it is.
* `LigOff.EndFlags`, `LigOff.endFlags_answer`, `LigOff.endFlags_round`: the flags past level 0's close; coins and a
  round of the schedule from the close on keep them, the close setting them.
* `LigOff.repState_push_flags`, `LigOff.repState_end`: at a rep's state past its close, up to the end of its schedule,
  the coins the frame takes leave it with those flags (`RhoKeep.repState_closeIn` at the close).
* `LeanRelease.ofLoop_answered_end`: at the session loop's own frames, at the end of a rep's schedule on its own table,
  the state the frame takes the coins at has them. `LeanRelease.roundOk_inner0`: at the inner proof's round 0 of
  `withInner`'s session loop on `finHonest`, the round passes the frame's checks at `valsDev honestVals`.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker isLabel roundFree INNER_ROUNDS check labelFrame wordBytes words innerSched
  ownRound passes firstDiff LABEL_PCS LABEL_ZC claimPoints innerOwn)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)

namespace Loop

/-- **The frame takes its own round 0 freeing no word**: past the opening's label and, at a layout with level 0, past
its close, outside every masked phase and the inner proof, the frame takes the inner proof's label, `final_c'` and the
squeeze of one coin. -/
theorem roundFree_own0_ok {c : Cfg} {s : Masker} {w t : ByteArray} (hw : w.size = 16) (ht : t.size = 64)
    (hi : s.inner = none) (hd : s.done = true) (hon : s.on = false) (hl : c.lig = true → s.doneL0 = true) :
    ∃ o u, roundFree c s (ownRound c s w t 1 0) 1 = .ok (o, u, #[]) := by
  obtain ⟨q, hq⟩ : ∃ q, s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 = .ok q := by
    obtain ⟨p1, p2, p3, p4⟩ := Ranked.inner_other
    cases hq : s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 with
    | ok q => exact ⟨q, rfl⟩
    | error e =>
      exfalso
      cases hlig : c.lig
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon, hlig] at hq
        all_goals cases hq
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon, hlig, hl hlig] at hq
        all_goals cases hq
  have hq0 := label_inner_zero c s hq
  have hqon := label_inner_on c s hq
  have hX := ownRound_encs (c := c) (s := s) (n := 1) hw ht (show (0 : ℕ) < 4 by decide)
  rw [show ownSq 1 0 = Frame.sq 1 1 from rfl] at hX
  have hR : ownRound c s w t 1 0 = ByteArray.mk (encs (ownFrames c s w t 0) ++ (Frame.sq 1 1).enc).toArray := by
    rw [← hX, Array.toArray_toList]
  obtain ⟨tl, htl, hpl⟩ : ∃ tl, ownFrames c s w t 0 = .label _root_.Flock.Zk.LABEL_INNER.toUTF8 :: tl ∧
      ∀ f ∈ tl, plainFrame f :=
    ⟨_, rfl, fun f hf => by rw [List.mem_singleton.mp hf]; trivial⟩
  have hst : stepAt c (s, ByteArray.empty, #[], 0) (.label _root_.Flock.Zk.LABEL_INNER.toUTF8) =
      .ok (q.1, ByteArray.empty ++ q.2, #[], 0 + 16 + (_root_.Flock.Zk.LABEL_INNER.toUTF8.size + 15) / 16 * 16) := by
    simp only [stepAt]
    rw [hq]
    all_goals rfl
  obtain ⟨r, hr, h1, -⟩ := foldlM_stepAt_off c tl q.1 (ByteArray.empty ++ q.2) #[]
    (0 + 16 + (_root_.Flock.Zk.LABEL_INNER.toUTF8.size + 15) / 16 * 16) 0 hqon hpl
  have hfr := foldlM_stepAt_off_fr c tl q.1 (ByteArray.empty ++ q.2) #[]
    (0 + 16 + (_root_.Flock.Zk.LABEL_INNER.toUTF8.size + 15) / 16 * 16) r hqon hpl hr
  have hf : (ownFrames c s w t 0).foldlM (stepAt c) (s, ByteArray.empty, #[], 0) = .ok r := by
    rw [htl, List.foldlM_cons, hst]
    exact hr
  have hO : innerOwn c q.1 0 (ByteArray.mk (encs (ownFrames c s w t 0) ++ (Frame.sq 1 1).enc).toArray) 1 = .ok () := by
    rw [← hR, ownRound_state_congr c (S := s) (s := q.1) w t 1 (k := 0) (by decide) (fun h => absurd h (by decide))
      (fun h => absurd h (by decide))]
    exact innerOwn_ownRound c q.1 w t hw ht 1 0 (by decide)
  obtain ⟨m, o, fr, cnt⟩ := r
  simp only at h1 hfr
  subst h1 hfr
  refine ⟨o ++ header 4 1 1, { q.1 with owed := some 1 }, ?_⟩
  rw [hR, roundFree_encs c s _ 1 1 (by rw [hi]; simp) (ownFrames0_mid c s w t) (.inr ⟨rfl, rfl⟩) (by decide)
    (by decide), hf]
  simp only [bind, Except.bind, hq0]
  rw [hO]
  try rfl

theorem ownRound0_frames {c : Cfg} {s : Masker} {w t : ByteArray} (hw : w.size = 16) (ht : t.size = 64) :
    Frames (ownRound c s w t 1 0).data.toList 1 :=
  ⟨ownFrames c s w t 0, 1, ownRound_encs hw ht (show (0 : ℕ) < 4 by decide), ownFrames0_mid c s w t,
    .inr ⟨rfl, rfl⟩, by decide, by decide⟩

theorem setFinalC_inner0_self (a : F128) :
    setFinalC a (inner0 a.toBytes).data.toList = (inner0 a.toBytes).data.toList := by
  unfold setFinalC
  split_ifs <;> rfl

end Loop

namespace LigOff

open Loop

/-- **Past level 0's close**: past the opening's label, outside every masked phase, past level 0's close and outside the
inner proof. -/
def EndFlags (s : Masker) : Prop :=
  s.done = true ∧ s.on = false ∧ s.l0 = false ∧ s.doneL0 = true ∧ s.inner = none

theorem endFlags_answer {c : Cfg} {s s' : Masker} {cs : Array F128} (h : EndFlags s) (ha : s.answer c cs = .ok s') :
    EndFlags s' := by
  rw [answer_alg c s cs h.2.2.2.2 h.2.2.1 ha]
  exact h

/-- **A round of the schedule from level 0's close keeps the flags past it**: the closing round, from level 0 at the
layout's last masked value with all of `ρ`'s words, past the opening and outside the inner proof, sets them, and a later
round of good frames keeps them. -/
theorem endFlags_round {c : Cfg} {C p : ℕ} {r : ByteArray × ℕ} {s : Masker} {q : ByteArray × Masker × Array ℕ}
    (hr : PastRound C p r) (hCp : C ≤ p)
    (hC : p = C → s.done = true ∧ s.l0 = true ∧ s.pos = c.order.size ∧
      s.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS ∧ s.inner = none)
    (hP : C < p → EndFlags s) (h : roundFree c s r.1 r.2 = .ok q) : EndFlags q.2.1 := by
  obtain ⟨fs, k, hb, hfs, hk, hn0, hn⟩ := hr
  have hmf : ∀ f ∈ fs, midFrame f := fun f hf => (hfs f hf).1
  have hpf : ∀ f ∈ fs, plainFrame f := fun f hf => (hfs f hf).2.1
  rw [Flags.eq_mk_encs hb] at h
  unfold FlockComplete.LigShape.preAt at h
  by_cases hpC : p = C
  · rw [ite_eq_left hpC, List.singleton_append] at h
    obtain ⟨hd, hl0, hpos, hrho, hi⟩ := hC hpC
    have hm' : ∀ f ∈ Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs, midFrame f := by
      intro f hf
      rcases List.mem_cons.1 hf with rfl | hf
      · show _root_.Flock.Zk.LABEL_CLOSE.toUTF8.size < 2 ^ 64
        decide
      · exact hmf f hf
    obtain ⟨out, fr, hro, -⟩ := roundFree_close c s fs k r.2 hl0 hpos hrho hi hm' hpf hk (fun h' => hn0 h'.1) hn
    rw [hro] at h
    cases h
    exact ⟨hd, rfl, rfl, rfl, hi⟩
  · rw [ite_eq_right hpC, List.nil_append] at h
    have hE := hP (by omega)
    obtain ⟨out, fr, hro, -⟩ := roundFree_off c s fs k r.2 hE.2.1 hE.2.2.2.2 hmf hpf hk (fun h' => hn0 h'.1) hn
    rw [hro] at h
    cases h
    exact hE

/-- **A rep's state one round on, from level 0's close**: where the coins the frame takes at the rep's state at round
`J + 1` from the close, before the schedule's end, leave it at the close in level 0 at the layout's last masked value with
all of `ρ`'s words, outside the inner proof, and past the close with the flags past it, the coins it takes at the state
one round on leave it with those flags. At the close the frame is past the opening (`Ranked.Good`, `run_good`). -/
theorem repState_push_flags {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    {cr0 : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hm : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0)
    (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) (hn0 : t.sch.n ≠ 0)
    (htp : 2 * t.tPad < 2 ^ 64) (hd : ∀ i, 64 * 2 ^ (t.sch.levels.getD i default).d < 2 ^ 64)
    (cq : Array (Array F128)) (a : Array F128) {J : ℕ} (hX : cq.size = J + 1)
    (hC : alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 ≤ J + 1)
    (hE : J + 1 < alg.size + (Count.openList (xs.size + 2) t.sch).length)
    (hS : ∀ s1, (repState c t rep dom xs hi cr0 cq ns (J + 1)).answer c (cq.getD J #[]) = .ok s1 →
      (J + 1 = alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 → s1.l0 = true ∧ s1.pos = c.order.size ∧
        s1.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS ∧ s1.inner = none) ∧
      (alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 < J + 1 → EndFlags s1))
    {N : ℕ} {cs : Array F128} {m : Masker} (hN : N = J + 2)
    (ha : (repState c t rep dom xs hi cr0 (cq.push a) ns N).answer c'' cs = .ok m) : EndFlags m := by
  subst hN
  obtain ⟨-, sched', -, hs'⟩ := repState_ok ha
  rw [Array.pop_push] at hs'
  obtain ⟨sched, hs⟩ := schedSoFar_pop_ok hs'
  obtain ⟨hj, hpast⟩ := sched_past hal hs' hdom hdg hcap hm hmk hx hk hn hq hm14 hn0 htp hd (by omega) hC
    (by omega)
  have hj0 := schedSoFar_pop_lt t rep dom xs hi hs hs' hX hj
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  obtain ⟨ha0, ha1⟩ := Sched.rounds_two hm14 hm hal
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk' => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk']
  have h1s : 1 < alg.size := (Array.getElem?_eq_some_iff.1 ha1).1
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  rw [repState_at c t rep dom xs hi cr0 cq ns J hal hs hj0] at hS
  rw [repState_succ c t rep dom xs hi cr0 cq ns a J hal hs hs' hX hj] at ha
  dsimp only at ha hS
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cq J)).2
  · rw [h2] at ha hS
    simp only [Bool.false_eq_true, ↓reduceIte] at ha hS
    have hgd := run_good hc _ sched cq hdg hcap hdom h0 (J := J) (by omega) hj0 h2
    generalize (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cq J)).1 = S at ha hS hgd
    cases h3 : (_root_.Flock.Firewall.Frame.run c S [.coins (.ok (cq.getD J #[])),
        .round (sched'.getD (J + 1) (.empty, 0)).2 (sched'.getD (J + 1) (.empty, 0)).1
          (.sched (sched'.getD (J + 1) (.empty, 0)).1)]).2
    · rw [h3] at ha
      simp only [Bool.false_eq_true, ↓reduceIte] at ha
      obtain ⟨s1, ha1, he⟩ := run_coins_cons c S _ _ h3
      rw [he] at h3 ha
      obtain ⟨hl1, hl2⟩ := hS s1 ha1
      have hg1 := (Ranked.answer_keep hgd ha1).1
      have ho1 := answer_owed ha1
      have hcl : J + 1 = alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 → s1.done = true ∧ s1.l0 = true ∧
          s1.pos = c.order.size ∧ s1.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS ∧ s1.inner = none := by
        intro hJC
        obtain ⟨e1, e2, e3, e4⟩ := hl1 hJC
        refine ⟨?_, e1, e2, e3, e4⟩
        cases hd1 : s1.done
        · have := (hg1.1 hd1).2.1
          rw [e1] at this
          cases this
        · rfl
      rcases Loop.run_sched_cons (st := { S with m := s1 }) ho1 h3 with ⟨out, sx, why, hck, he'⟩ | he'
      · rw [he'] at ha
        change sx.answer c'' cs = .ok m at ha
        refine endFlags_answer ?_ ha
        rcases Loop.check_self hck with ⟨fr, hr⟩ | ⟨fr, hr⟩
        · exact endFlags_round (q := (out, sx, fr)) hpast hC hcl hl2 hr
        · exact endFlags_round (s := { s1 with stopped := true }) (q := (out, sx, fr)) hpast hC hcl hl2 hr
      · rw [he'] at ha
        change ({ s1 with stopped := true } : Masker).answer c'' cs = .ok m at ha
        exact absurd ha (answer_owed_none (s := { s1 with stopped := true }) cs ho1)
    · rw [h3] at ha
      simp only [↓reduceIte] at ha
      exact absurd ha (answer_owed_none cs rfl)
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

/-- The flags past level 0's close at a rep's state from the round after the close, `e` rounds on. -/
theorem repState_end_from {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    {cr0 : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hm : t.st.m < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) (hk : 1 ≤ t.sch.k0)
    (hn : t.sch.n < 2 ^ 62) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 63) (hn0 : t.sch.n ≠ 0)
    (htp : 2 * t.tPad < 2 ^ 64) (hd : ∀ i, 64 * 2 ^ (t.sch.levels.getD i default).d < 2 ^ 64) {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp)) {lay : _root_.Flock.Zk.PadLayout}
    {d : _root_.Flock.Firewall.Frame.Draws} {mk : Option _root_.Flock.Firewall.Frame.MaskSlot}
    (hlay : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (hcd : c = Cfg.ofDraws lay d mk) :
    ∀ (e : ℕ) {cr : Array (Array F128)} {N : ℕ} {c'' : Cfg} {cs : Array F128} {m : Masker}, cr.size = N →
      N = alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 + 1 + e →
      N ≤ alg.size + (Count.openList (xs.size + 2) t.sch).length →
      (repState c t rep dom xs hi cr0 cr ns N).answer c'' cs = .ok m → EndFlags m
  | 0, cr, N, c'', cs, m, hN, he, hE, ha => by
    obtain ⟨cq, a, rfl⟩ : ∃ cq a, cr = Array.push cq a :=
      ⟨cr.pop, cr.back!, Array.eq_push_pop_back!_of_size_ne_zero (by omega)⟩
    rw [Array.size_push] at hN
    refine repState_push_flags hc hal hm14 hm hdg hcap hdom hmk hx hk hn hq hn0 htp hd cq a
      (J := alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3)) (by omega) (by omega) (by omega)
      (fun s1 hs1 => ⟨fun _ => ?_, fun h => absurd h (by omega)⟩) (by omega) ha
    obtain ⟨-, e2, e3, e4, e5⟩ := RhoKeep.repState_closeIn hc hal hm14 hdg hcap hdom hmk hx hz hlay hcd rfl
      (by omega) hs1
    exact ⟨e2, e4, e5, e3⟩
  | e + 1, cr, N, c'', cs, m, hN, he, hE, ha => by
    obtain ⟨cq, a, rfl⟩ : ∃ cq a, cr = Array.push cq a :=
      ⟨cr.pop, cr.back!, Array.eq_push_pop_back!_of_size_ne_zero (by omega)⟩
    rw [Array.size_push] at hN
    exact repState_push_flags hc hal hm14 hm hdg hcap hdom hmk hx hk hn hq hn0 htp hd cq a
      (J := alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 + e) (by omega) (by omega) (by omega)
      (fun s1 hs1 => ⟨fun h => absurd h (by omega), fun _ =>
        repState_end_from hc hal hm14 hm hdg hcap hdom hmk hx hk hn hq hn0 htp hd hz hlay hcd e (by omega)
          (by omega) (by omega) hs1⟩) (by omega) ha

/-- **A rep's state past its close, up to the end of its schedule**: where the loop's state at a rep's round past level
0's close, at most the end of its schedule, takes the coins before it, the frame is past the opening, outside every
masked phase, past level 0's close and outside the inner proof. -/
theorem repState_end {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    {cr0 : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp)) {lay : _root_.Flock.Zk.PadLayout}
    {d : _root_.Flock.Firewall.Frame.Draws} {mk : Option _root_.Flock.Firewall.Frame.MaskSlot}
    (hlay : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (hcd : c = Cfg.ofDraws lay d mk)
    (htp : 2 * t.tPad < 2 ^ 64) (hd : ∀ i, 64 * 2 ^ (t.sch.levels.getD i default).d < 2 ^ 64)
    {cr : Array (Array F128)} {N : ℕ} (hN : cr.size = N)
    (hC : alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 < N)
    (hE : N ≤ alg.size + (Count.openList (xs.size + 2) t.sch).length) {c'' : Cfg} {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns N).answer c'' cs = .ok m) : EndFlags m := by
  obtain ⟨hm36, hk, hn, hq⟩ := Count.sched_facts hz
  have hn0 : t.sch.n ≠ 0 := by
    have := (Count.sched_rho hz).1
    omega
  exact repState_end_from hc hal hm14 (by omega) hdg hcap hdom hmk hx hk hn hq hn0 htp hd hz hlay hcd
    (N - (alg.size + (xs.size + 2) + 1 + t.sch.k0 + 4 + 1)) hN (by omega) hE ha

end LigOff

namespace LeanRelease

open Loop in
set_option maxHeartbeats 1000000 in
/-- **At the loop's own frames, at the end of a rep's schedule the frame takes coins past level 0's close**: where the
frame takes the coins after `e0 :: h1` at `m` and the next round is rep stream `k`'s round `j + 1`, the end of its
schedule so far on its own table and salts, `m` is past the opening, outside every masked phase, past level 0's close
and outside the inner proof (`LigOff.repState_end`): the schedule there is the whole rep's, past its close. -/
theorem ofLoop_answered_end (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (hL : Masked.Live x) (hSv : Serves x)
    (hR : ∀ (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
      (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {m : Masker}
    (hA : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m) (hne : (e0 :: h1).length ≠ lastOf x)
    {cs : List FlockDeploy.Coin} (hcs : h1.map (fun p => CoinLayer.rd p.2) = cs)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q k j : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hki : CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1))
    {tr : Tab} {hr : Array ByteArray} (hreal : real x w ρ k = .ok (tr, hr)) {rsched : Array (ByteArray × ℕ)}
    (hrs0 : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (wordsAt x cs 0))) hr ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok rsched)
    (hend : rsched.size = j + 1) : LigOff.EndFlags m := by
  have hq1 : cs.length = q + 1 := by rw [← hcs, List.length_map, hq]
  have hq2 : (e0 :: h1).length = q + 2 := by simp only [List.length_cons, hq]
  have hk1 : (CoinLayer.locate (roundsOf spec sts) 0 q).1 = k := by rw [hki]
  have hj : (CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1 := by rw [hki]
  obtain ⟨m0, r0⟩ := e0
  have hcr : coinsAt x ((m0, r0) :: h1) = readCoins x cs := by
    rw [← hcs]
    exact loop_coinsRead loopNr T (withFrames T L₀ real) x m0 r0 h1
  have hF : (ofLoop T L₀ real).frameAnswer x ρ ((m0, r0) :: h1) = .ok m := by
    unfold answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswer_same T L₀ real x ρ (m0, r0) h1 hx hq hj, hcs, hcr, readCoins_eq x cs hq1] at hF
  replace hF := (bind_rankStop_ok hF).1
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
  have hqs := loop_round_in_plan T (withFrames T L₀ real) x ρ ((m0, r0) :: h1) hA hne hx hq2
  have hk : k < spec.repStreams.size := by
    have := locate_lt (roundsOf spec sts) 0 (q' + 1) hqs
    rw [hki] at this
    simpa [roundsOf] using this
  obtain ⟨lay, mask, hds, hc', hdm, hxe⟩ := placeholder_ok hph'
  obtain ⟨-, hd, hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec sts hx) hk
  have hls : (wordsAt x cs 0).size = min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    rw [← hcs]
    cases h1 with
    | nil => simp at hq
    | cons p h2 => simp [wordsAt, FlockDeploy.Coin.vals]
  have hdom : dom.size < 2 ^ 64 := by rw [hdm]; exact hd
  have hxs : xs.size + 2 < 2 ^ 64 := by
    rw [hxe, extras_size]
    have := Nat.mul_le_mul_left sts[k / 2]!.regions.size
      ((linkPoints_size_le sts[k / 2]! (wordsAt x cs 0)).trans (hls.trans_le (Nat.min_le_right _ _)))
    omega
  have hT := Count.real_tabAlike (hR ρ s0 spec sts hx) hph' hreal
  obtain ⟨hdg, hmb, hn, hlq, hpd, hdd⟩ := Count.real_bounds hph' hT
  have hR' := Sched.schedSoFar_alike tr t (k % 2) dom xs hr hi
    ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1))
    hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1 hT.2.2.2.2 hdom hdg hmb hxs hn hlq hpd hdd
  rw [← hdm, ← hxe] at hrs0
  rw [hrs0] at hR'
  cases hs' : schedSoFar t (k % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1)) with
  | error e => rw [hs'] at hR'; exact (hR' : False).elim
  | ok sched' =>
  rw [hs'] at hR'
  have hsz := roundsAt_size hR'
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds hs'
  obtain ⟨hts', ⟨tp, htp⟩, -⟩ := Count.placeholder_tab hph'
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hL.1 hx
  have hj2 : k / 2 < sts.size := by
    have := FlockDeploy.repStreams_size spec
    omega
  have hst : sts[k / 2]! = sts[k / 2] := getElem!_pos sts (k / 2) hj2
  have htm : t.st.m = sts[k / 2]!.m := by rw [hts']; rfl
  have hm14 : 14 ≤ t.st.m := by
    rw [htm, hst]
    have := (htabs (k / 2) hj2).1
    omega
  have hmkt : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [hts']
    show sts[k / 2]!.merkle = _
    rw [hst]
    exact (htabs (k / 2) hj2).2.2
  have hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp) := by rw [htm]; exact htp
  obtain ⟨sch3, tp3, hs3, hl3⟩ := start_lay hds
  rw [htp, Except.ok.injEq, Prod.mk.injEq] at hs3
  have hlay : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0 := by
    rw [hl3, ← hs3.1, hts']
    try rfl
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by
    rw [hc']
    exact Claims.ZKServer.ofDraws_order _ _ _
  have hdgt : t.st.digest.size < 2 ^ 64 := by rw [← hT.1]; exact hdg
  have htpd : 2 * t.tPad < 2 ^ 64 := by rw [← hT.2.2.1]; exact hpd
  have hdT : ∀ i, 64 * 2 ^ (t.sch.levels.getD i default).d < 2 ^ 64 := by rw [← hT.2.1]; exact hdd
  obtain ⟨hm36, hkk, hnn, hqq⟩ := Count.sched_facts hz
  have hcrs : ((Array.range (j + 1)).map fun i => wordsAt x cs (q' + 1 - (j + 1) + i + 1)).size = j + 1 := by
    rw [Array.size_map, Array.size_range]
  have hfull := Count.schedSoFar_full t (k % 2) dom xs hi _ hal hs' (by rw [hcrs]; omega) hdom hdgt
    (placeholder_cap hph') (by omega) hxs hkk hnn hqq
  have hOL := Count.openList_length hz (c := xs.size + 2) (by omega)
  exact LigOff.repState_end hcc hal hm14 hdgt (placeholder_cap hph') hdom hmkt hxs hz hlay hc' htpd hdT hcrs
    (by omega) (by omega) hF

open Loop in
set_option maxHeartbeats 1000000 in
/-- **At the inner proof's round 0 `withInner`'s session loop on `finHonest` passes the frame's checks at
`valsDev honestVals`**, wherever `cross` passes its device round: at a history at the inner proof's rounds
(`Loop.InnerAt`'s components) at the end of the rep's schedule so far, on the rep's own tables where `real` answers the
loop's, at a `Live` statement the loop serves, where the frame takes the history's coins. The round in the device's
place is the frame's own inner round 0, at one coin (`Loop.innerAt_slot`); the frame is past level 0's close outside
every masked phase and the inner proof (`ofLoop_answered_end`), so it takes the round freeing no word
(`Loop.roundFree_own0_ok`); `valsDev` writes the honest `final_c'`, which is `finHonest`'s
(`Loop.innerSched_finHonest0`). -/
theorem roundOk_inner0 (L₁ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit → Tp → List (Msg × Msg) → Array ByteArray)
    (hfin : fin = finHonest (ofLoopLink L₁ src real)) {x : Msg} {w : Flock.Wit} (hx : Masked.Live x)
    (hSv : Serves x)
    (hR : ∀ (ρ : Tp) (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (ν : finS.Nonce) (t : finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hlt : ops.length < refK x)
    (href : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) =
      false)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q k i : ℕ} {tr : Tab} {hr : Array ByteArray}
    {rsched : Array (ByteArray × ℕ)} (hxs : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : (ops.map fun p => CoinLayer.rd p.2).length = q + 1)
    (hloc : CoinLayer.locate (roundsOf spec sts) 0 q = (k, i)) (hreal : real x w (ν, t) k = .ok (tr, hr))
    (hrs : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)))
      hr ((Array.range i).map fun j => wordsAt x (ops.map fun p => CoinLayer.rd p.2) (q - i + j + 1)) = .ok rsched)
    (hend : rsched.size = i)
    (hcr : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).cross x ((m₀, ρ₀) :: ops).length w (ν, t)
      ((m₀, ρ₀) :: ops) (valsDev finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) x w
        (ν, t) ((m₀, ρ₀) :: ops) (computedOf ((ofLoopLink (withInner Masked.dataE L₁ real fin) src real).computed x w
          (ν, t) ((m₀, ρ₀) :: ops)))) = true) :
    ((ofLoopLink (withInner Masked.dataE L₁ real fin) src real).withDevice
      (valsDev finS.honestVals (ofLoopLink (withInner Masked.dataE L₁ real fin) src real))).roundOk x w (ν, t)
      ((m₀, ρ₀) :: ops) = true := by
  subst hfin
  have hsch : schedOf (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2) = none :=
    schedOf_none_of_innerAt ⟨s0, spec, sts, q, k, i, tr, hr, rsched, hxs, hq, hloc, hreal, hrs, hend.le⟩
  have hq1 : ops.length = q + 1 := by rw [← hq, List.length_map]
  obtain ⟨jf, rf, -, he4, hslot, -, -, hi0, -, hnr⟩ := Loop.innerAt_slot hx hSv hxs hq
    (by rw [List.length_map]; exact hlt) hloc (hR (ν, t) s0 spec sts hxs) hreal hrs hend.le
  have hio := Loop.innerOf_eq (real x w (ν, t)) ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    (finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)) x _ hxs hq hloc hreal hrs hend.le he4
  have hsc := scheduled_innerAt L₁ src real (finHonest (ofLoopLink L₁ src real)) x w (ν, t) (m₀, ρ₀) ops hsch hio
  obtain ⟨m, ha⟩ := answered_of_unrefused href
  have hne : ((m₀, ρ₀) :: ops).length ≠ lastOf x := by rw [lastOf_eq, List.length_cons]; omega
  obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
  obtain ⟨hdn, hon, -, hdl, hin⟩ := ofLoop_answered_end Masked.dataE
    (withFiles (withLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src) src) real x w hx
    hSv hR (ν, t) (m₀, ρ₀) ops ha hne rfl hxs hq1 hloc hreal hrs hend
  have hco : (ofLoopLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src real).coins x
      ((m₀, ρ₀) :: ops).length = loopNr x (q + 1) := by
    rw [List.length_cons, hq1]
    rfl
  have hn1 : loopNr x (q + 1) = 1 := by
    rw [hslot, hend, Nat.sub_self]
    exact hnr 0 (by decide)
  have hcfg : (ofLoopLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src real).cfg x
      (ν, t) ((m₀, ρ₀) :: ops) = (withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops) := by
    unfold cfg drawsAt
    rfl
  have hcfg1 : (ofLoopLink L₁ src real).cfg x (ν, t) ((m₀, ρ₀) :: ops) =
      (withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops) :=
    (cfg_ofLoopLink_withInner L₁ src real (finHonest (ofLoopLink L₁ src real)) x (ν, t) ((m₀, ρ₀) :: ops)).symm.trans
      hcfg
  have hR0 : ∀ vals,
      (ofLoopLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src real).scheduled x w
        (ν, t) ((m₀, ρ₀) :: ops) vals =
      (innerSched ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false }
        (finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)) 1 0).data.toList := by
    intro vals
    rw [hsc vals, hn1, show j + 1 - rsched.size = 0 by omega]
    exact congrArg (fun b : ByteArray => b.data.toList)
      (Loop.innerSched_state_congr _ _ _ (k := 0) (by decide) (fun h => absurd h (by decide))
        (fun h => absurd h (by decide)))
  have hw : (wordBytes (words ((finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)).getD 0 .empty)
      ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1)).size = 16 := by
    simp only [wordBytes_size, Loop.words_size]
  have ht : (wordBytes (words ((finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)).getD 2 .empty)
      16 4)).size = 64 := by
    simp only [wordBytes_size, Loop.words_size]
  obtain ⟨o, u, hq'⟩ := Loop.roundFree_own0_ok (c := (withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    (s := { m with stopped := false }) hw ht hin hdn hon (fun _ => hdl)
  have hF := Loop.ownRound0_frames (c := (withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
    (s := { m with stopped := false }) hw ht
  have h0 : setFinalC (vsAt finS.honestVals
        (ofLoopLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src real) x w (ν, t)
        ((m₀, ρ₀) :: ops) ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc +
        ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h.getD
          ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).lay.fc 0)
      (innerSched ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false }
        (finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)) 1 0).data.toList =
      (innerSched ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false }
        (finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)) 1 0).data.toList := by
    rw [innerSched_finHonest0, vsAt_ofLoopLink_withInner, hcfg1]
    exact setFinalC_inner0_self _
  have h2 := Loop.setTau_of_ne (m := m) (by rw [hin]; intro e; cases e) (vsTauAt finS.honestVals
    (ofLoopLink (withInner Masked.dataE L₁ real (finHonest (ofLoopLink L₁ src real))) src real) x w (ν, t)
    ((m₀, ρ₀) :: ops))
    (innerSched ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)) { m with stopped := false }
      (finHonest (ofLoopLink L₁ src real) x w (ν, t) ((m₀, ρ₀) :: ops)) 1 0).data.toList
  have hd := Loop.valsDev_own finS.honestVals _ ha hcfg (hco.trans hn1) (hR0 _) hq' hF h0 h2
  exact Loop.roundOk_valsDev_own finS.honestVals _ ha hcfg (hco.trans hn1) (hR0 _) hd hq' hcr

end LeanRelease

end Claims.ZKServer
