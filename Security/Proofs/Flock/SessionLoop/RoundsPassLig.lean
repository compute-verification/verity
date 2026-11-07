module

public import Proofs.Flock.SessionLoop.RoundsPassTaken
public import Proofs.Flock.SessionLoop.RoundLigFrame
public import Proofs.Flock.SessionLoop.RhoKeepRep
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# A rep's Ligerito rounds from level 0's close, taken without liveness

`LeanRelease.SchedTaken` at rep stream `2j + r`'s Ligerito rounds from level 0's close (`Masked.repSlot` `k₀ + 4`) to
the last before the inner proof, wherever the frame takes its coins (**`LeanRelease.schedTaken_lig`**). The scheduled
round on the rep's own table is the closing label, good frames and a squeeze at the close, and good frames and a squeeze
past it (`LigOff.sched_past`). The frame is outside the inner proof at every scheduled round (`Loop.repState_answer_inner`);
at the close it is in level 0 at the layout's last masked value with all of `ρ`'s words (`RhoKeep.repState_closeIn`),
so it takes the closing label, and past the close it is outside every masked phase (`LigOff.repState_close`). This is
`lig_frame`'s argument with the frame's answer from `refusesCoins` in liveness's place.

* `LigOff.roundFree_close_ok`: the frame takes level 0's closing round wherever the closing label passes.
* `RhoKeep.answer_rho_ok`, `RhoKeep.repState_closeIn`: at the close, the coins the frame takes leave it holding all of
  `ρ`'s words, in level 0 at the layout's last masked value.
* `Loop.repState_answer_inner`: where a rep's schedule at a table of its placeholder's shape holds a round past its
  first, the frame takes the round's coins outside the inner proof.
* `LeanRelease.lig_taken`: the frame takes a round of the schedule from the close on.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker roundFree labelFrame)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)

namespace LigOff

open Loop

/-- **Level 0's closing round, taken**: from a state outside the inner proof, in level 0 if in a masked phase at all and
there at the layout's last masked value with all of `ρ`'s words, the frame takes the closing label, then plain frames
and the squeeze. -/
theorem roundFree_close_ok (c : Cfg) (s : Masker) (fs : List Frame) (k : UInt8) (n : ℕ) (hi : s.inner = none)
    (hl : s.on = true → s.l0 = true)
    (hp : s.l0 = true → s.pos = c.order.size ∧ s.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS)
    (hm : ∀ f ∈ fs, midFrame f) (hpl : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) :
    ∃ out t fr, roundFree c s (ByteArray.mk (encs (.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs) ++
      (Frame.sq k n).enc).toArray) n = .ok (out, t, fr) := by
  obtain ⟨q, hq⟩ : ∃ q, s.label c _root_.Flock.Zk.LABEL_CLOSE.toUTF8 = .ok q := by
    cases hq : s.label c _root_.Flock.Zk.LABEL_CLOSE.toUTF8 with
    | ok q => exact ⟨q, rfl⟩
    | error e =>
      exfalso
      obtain ⟨p1, p2, p3, p4⟩ := Ranked.close_other
      cases hl0 : s.l0
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0] at hq
        cases hq
      · obtain ⟨hpos, hrho⟩ := hp hl0
        simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at hq
        cases hq
  have hm' : ∀ f ∈ Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs, midFrame f := by
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · show _root_.Flock.Zk.LABEL_CLOSE.toUTF8.size < 2 ^ 64
      decide
    · exact hm f hf
  have hof := (label_close_out hl hq).1
  have hin : q.1.inner = none := by
    refine (Sched.label_inner c s _ (fun he => ?_) hq).trans hi
    exact ne_inner_of_isLabel (b := _root_.Flock.Zk.LABEL_CLOSE.toUTF8)
      (by rw [String.toUTF8_eq_toByteArray]; exact Ranked.close_other.2.1) (by rw [he])
  have hst : stepAt c (s, ByteArray.empty, #[], 0) (.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8) =
      .ok (q.1, ByteArray.empty ++ q.2, #[], 0 + 16 + (_root_.Flock.Zk.LABEL_CLOSE.toUTF8.size + 15) / 16 * 16) := by
    simp only [stepAt]
    rw [hq]
    all_goals rfl
  obtain ⟨r, hr, h1, -⟩ := foldlM_stepAt_off c fs q.1 (ByteArray.empty ++ q.2) #[]
    (0 + 16 + (_root_.Flock.Zk.LABEL_CLOSE.toUTF8.size + 15) / 16 * 16) 0 hof hpl
  have hf : (Frame.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs).foldlM (stepAt c) (s, ByteArray.empty, #[], 0) =
      .ok r := by
    rw [List.foldlM_cons, hst]
    exact hr
  rw [roundFree_encs c s _ k n (by rw [hi]; simp) hm' hk h0 hn, hf]
  obtain ⟨m, o, fr, cnt⟩ := r
  simp only at h1
  subst h1
  refine ⟨o ++ header 4 k n, { q.1 with owed := some n }, fr, ?_⟩
  simp only [bind, Except.bind, hin]
  rfl

end LigOff

namespace RhoKeep

open Loop

/-- **`ρ₁`'s coins, taken**: from level 0 at the layout's last masked value, holding `ρ₀`'s words and owing `ρ₁`'s two
coins, coins the frame takes leave it holding all of `ρ`'s words, in its phase and at its position. -/
theorem answer_rho_ok {c : Cfg} {s m : Masker} {r0 r1 : F128} {cs : Array F128} (hn : s.owed = some 2)
    (hi : s.inner = none) (hl : s.l0 = true) (hp : s.pos = c.order.size) (hr : s.rho = #[r0, r1])
    (h : s.answer c cs = .ok m) :
    m.on = s.on ∧ m.l0 = true ∧ m.inner = none ∧ m.pos = s.pos ∧
      m.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS := by
  obtain ⟨a, b, rfl⟩ := FlockComplete.JsonRT.two_of (answer_size hn h)
  cases hd : _root_.Flock.Zk.rhoIndependent (⟨r0, r1⟩, ⟨a, b⟩) with
  | false =>
    rw [answer_rho_dependent c s r0 r1 a b hn hi hl hp hr hd] at h
    cases h
  | true =>
    rw [answer_rho_independent c s r0 r1 a b hn hi hl hp hr hd] at h
    cases h
    exact ⟨rfl, hl, hi, rfl, rfl⟩

/-- **The loop's state at a rep's close**: where the loop's state at a rep's round past `ρ₁`'s takes the round's coins
at a layout of the rep's size, the frame is in level 0's masked phase, outside the inner proof, at the layout's last
masked value with all of `ρ`'s words (`rho_step`, `answer_rho_ok`). -/
theorem repState_closeIn {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hm14 : 14 ≤ t.st.m) (hdg : t.st.digest.size < 2 ^ 64)
    (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512) (hx : xs.size + 2 < 2 ^ 64) {tp : ℕ}
    (hz : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, tp)) {lay : _root_.Flock.Zk.PadLayout}
    {d : _root_.Flock.Firewall.Frame.Draws} {mk : Option _root_.Flock.Firewall.Frame.MaskSlot}
    (hlay : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0)
    (hcd : c = Cfg.ofDraws lay d mk) (hord : c''.order.size = c.order.size)
    (hcr : cr.size = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1)).answer c'' cs =
      .ok m) :
    m.on = true ∧ m.l0 = true ∧ m.inner = none ∧ m.pos = c''.order.size ∧
      m.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS := by
  obtain ⟨-, sched, -, hs⟩ := repState_ok ha
  obtain ⟨hm36, hk, hn, hq⟩ := Count.sched_facts hz
  have hn0 : t.sch.n ≠ 0 := by have := (Count.sched_rho hz).1; omega
  have hm : t.st.m < 2 ^ 64 := by omega
  have hlk : lay.k0 = t.sch.k0 := by subst hlay; rfl
  have hood : lay.nOod0 = 1 := by subst hlay; rfl
  have hk' : 0 < lay.k0 := by omega
  have hsz' : c.nPiop + l0Off (t.sch.k0 + 2 + 1) = c.order.size := by
    have h1 := l0_size lay hk' hood (i := t.sch.k0 + 2) (by omega) (by omega)
    have h3 := l0Off_succ (i := t.sch.k0 + 2) (by omega)
    have hpi : c.nPiop = absPos lay (lay.lgOod 0) := by rw [hcd]; exact nPiop_ood lay d mk hk'
    have hord' : c.order.size = (_root_.Flock.Firewall.Frame.absorbed lay).size := by subst hcd; rfl
    rw [hpi, hord']
    omega
  have hpop : cr.pop.size = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) := by
    rw [Array.size_pop, hcr]; omega
  obtain ⟨hsz, hL⟩ := sched_l0Run hc hal hs hm14 hdg hcap hdom hmk hx hz hlay hcd (j := t.sch.k0 + 3) le_rfl hpop
    (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[])
  rw [repState_at c t rep dom xs hi cr0 cr ns (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3)) hal hs (by omega)] at ha
  dsimp only at ha
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr
      (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3)))).2
  · rw [h2] at ha
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    rw [show alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2) + 1 by
      omega, schedReqs_succ, Loop.run_append] at h2 ha
    cases hb : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
        (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr
        (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)))).2
    · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h2 ha
      have hI := hL (t.sch.k0 + 2) (by omega) le_rfl hb
      rw [if_neg (show t.sch.k0 + 2 ≠ 0 by omega),
        show l0Nc (t.sch.k0 + 2) = 2 by unfold l0Nc; split_ifs <;> first | rfl | omega] at hI
      obtain ⟨-, T, sws, -, hbody⟩ := Bind.sched_l0 hal hs hdom hdg hcap hm hmk hx hk hn hq hn0 hm14
        (j := t.sch.k0 + 3) (by omega) hpop
      obtain ⟨hb1, hc1⟩ := hbody (t.sch.k0 + 3) le_rfl
      rw [FlockComplete.LigLock.l0BodyAt_r1] at hb1
      rw [show alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2) + 1 by
        omega] at hb1 hc1
      rw [List.getD_eq_getElem?_getD,
        show xs.size + 2 + 1 + (t.sch.k0 + 3) = xs.size + 2 + 1 + (t.sch.k0 + 2 + 1) by omega,
        Count.openList_rho hz (c := xs.size + 2) (by omega) (e := 1) (by omega), Option.getD_some] at hc1
      have hR := rho_step hc (b := t.sch.k0 + 2) hI hsz' hb1 hc1 h2
      rcases hR with ho | ⟨ho, hl, hon, hin, hps, hrh⟩
      · exact absurd ha (answer_owed_none _ ho)
      obtain ⟨e1, e2, e3, e4, e5⟩ := answer_rho_ok ho hin hl (hps.trans hord.symm) hrh ha
      exact ⟨e1.trans hon, e2, e3, e4.trans (hps.trans hord.symm), e5⟩
    · rw [hb, if_pos rfl, hb] at h2
      cases h2
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

end RhoKeep

namespace Loop

/-- **A rep's frame at a scheduled round past its first is outside the inner proof**: where the rep's schedule on its
coins so far, at a table of its placeholder's shape, holds the round, the frame takes the round's coins outside the
inner proof (`repState_inner`): the placeholder's rounds there are of the table's shape (`Sched.schedSoFar_alike`), so
framed. -/
theorem repState_answer_inner {c c'' : Cfg} {t tr : Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi hr : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {j : ℕ} {rs : Array (ByteArray × ℕ)}
    (hT : Sched.TabAlike tr t) (hdom : dom.size < 2 ^ 64) (hdg : tr.st.digest.size < 2 ^ 64)
    (hm : tr.st.m < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64) (hn : tr.sch.n + 1 < 2 ^ 64)
    (hq : ∀ i, (tr.sch.levels.getD i default).q < 2 ^ 64) (hp : 2 * tr.tPad < 2 ^ 64)
    (hd : ∀ i, 64 * 2 ^ (tr.sch.levels.getD i default).d < 2 ^ 64) (hcr : cr.size = j + 1)
    (hrs : schedSoFar tr rep dom xs hr cr = .ok rs) (hlt : j + 1 < rs.size) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (j + 1)).answer c'' cs = .ok m) : m.inner = none := by
  obtain ⟨alg, sched, hal, hs⟩ := repState_ok ha
  have hR := Sched.schedSoFar_alike tr t rep dom xs hr hi cr hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1 hT.2.2.2.2 hdom hdg hm
    hx hn hq hp hd
  rw [hrs] at hR
  cases hs' : schedSoFar t rep dom xs hi cr with
  | error e => rw [hs'] at hR; exact (hR : False).elim
  | ok sched' =>
    rw [hs'] at hR
    have hj' : j + 1 < sched'.size := by rw [← roundsAt_size hR]; exact hlt
    have hj0 := schedSoFar_pop_lt t rep dom xs hi hs hs' hcr hj'
    exact Sched.answer_inner (repState_inner c t rep dom xs hi cr0 cr ns j hal hs hj0 fun k hk => by
      obtain ⟨e, hk'⟩ := schedSoFar_pop_getD hs hs' (by omega) (show k < sched.size by omega)
      rw [e]
      exact roundsAt_framed hR hk') ha

end Loop

namespace LeanRelease

open Loop in
/-- **The frame takes a round of the schedule from level 0's close on** from state `m` after a history, outside the
inner proof: the closing round where the frame is in level 0 if in a masked phase at all, there at the layout's last
masked value with all of `ρ`'s words, and a later round outside every masked phase. -/
theorem lig_taken {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (g : List (Msg × Msg)) {m : Masker}
    {C p : ℕ} {r : ByteArray × ℕ} (hCp : C ≤ p) (hr : LigOff.PastRound C p r) (hin : m.inner = none)
    (hl : m.on = true → m.l0 = true)
    (hcl : p = C → m.l0 = true →
      m.pos = (L.cfg x ρ g).order.size ∧ m.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS)
    (hon : C < p → m.on = false) (hn : L.coins x g.length = r.2) : L.RoundTaken x ρ g m r.1.data.toList := by
  obtain ⟨fs, k, hb, hfs, hk, hn0, hn64⟩ := hr
  have hmf : ∀ f ∈ fs, midFrame f := fun f hf => (hfs f hf).1
  have hpf : ∀ f ∈ fs, plainFrame f := fun f hf => (hfs f hf).2.1
  unfold RoundTaken
  rw [hn]
  unfold FlockComplete.LigShape.preAt at hb
  by_cases hpC : p = C
  · rw [if_pos hpC, List.singleton_append] at hb
    obtain ⟨o, t, fr, hro⟩ := LigOff.roundFree_close_ok (L.cfg x ρ g) { m with stopped := false } fs k r.2 hin hl
      (hcl hpC) hmf hpf hk (fun h => hn0 h.1) hn64
    refine ⟨.label _root_.Flock.Zk.LABEL_CLOSE.toUTF8 :: fs, k, o, t, fr, hb, ?_, hk, fun h => hn0 h.1, hn64, ?_⟩
    · intro f hf
      rcases List.mem_cons.1 hf with rfl | hf
      · exact ⟨by show _root_.Flock.Zk.LABEL_CLOSE.toUTF8.size < 2 ^ 64; decide,
          ne_inner_of_isLabel (by rw [String.toUTF8_eq_toByteArray]; exact Ranked.close_other.2.1)⟩
      · exact midAll_of_plain hmf hpf f hf
    · rw [hb]
      exact hro
  · rw [if_neg hpC, List.nil_append] at hb
    obtain ⟨o, fr, hro, -⟩ := roundFree_off (L.cfg x ρ g) { m with stopped := false } fs k r.2 (hon (by omega)) hin
      hmf hpf hk (fun h => hn0 h.1) hn64
    exact ⟨fs, k, o, _, fr, hb, midAll_of_plain hmf hpf, hk, fun h => hn0 h.1, hn64, by rw [hb]; exact hro⟩

open Loop Ranked in
set_option maxHeartbeats 4000000 in
/-- **The frame takes a rep's Ligerito round from level 0's close where it takes its coins**: at a history whose next
round is rep stream `2j + r`'s round `i` from level 0's close at `k₀ + 4` to the last before the inner proof
(`Masked.repSlot` `i`), whose coins the loop's frame takes, on the loop's own tables where `src` samples the tape's
level 0. The scheduled round on the rep's own table is `PastRound`'s (`ofLoop_sched_unrefused`, `LigOff.sched_past`);
the frame is outside the inner proof there (`repState_answer_inner`), at the close in level 0 at the layout's last masked
value with all of `ρ`'s words (`RhoKeep.repState_closeIn`) and past it outside every masked phase
(`LigOff.repState_close`), and takes the round (`lig_taken`). -/
theorem schedTaken_lig (L₁ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hSv : Loop.Serves x) (hS0 : Loop.Level0Src src x w)
    (hST : Loop.ScheduleTotal x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hlt : ops.length < refK x)
    (href : (ofLoop Masked.dataE L₁ (Loop.realOf src)).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : (Masked.zkTab x j).sch.k0 + 4 ≤ i)
    (h2 : i < (Masked.zkTab x j).sch.k0 + 7 + 11 * ((Masked.zkTab x j).sch.levels.size - 1)) :
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
  -- the frame's answer
  have hlast : (ofLoop Masked.dataE L₁ (Loop.realOf src)).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ (Loop.realOf src)) x
  have hkd : (ofLoop Masked.dataE L₁ (Loop.realOf src)).kindAt x ((m₀, ρ₀) :: ops).length = .masked :=
    kindAt_masked_of _ (by simp only [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  obtain ⟨m, hA⟩ := answered_of_unrefused href
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same Masked.dataE L₁ (Loop.realOf src) x (ν, t) (m₀, ρ₀) ops hsx (q := i0 + 1) (by omega) hj1
    ho ht] at hA'
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
  obtain ⟨lay, mask, hds, hc', hdm, hxe⟩ := placeholder_ok hph0
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
  obtain ⟨tr, hir, sched', hre, hs'⟩ := ofLoop_sched_unrefused L₁ src x w hx hSv hS0 hST ν t m₀ ρ₀ ops href hkd
    hsx (q := i0 + 1) (by omega)
  have hre' : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by
    rw [hloc] at hre
    exact hre
  have hT := Count.real_tabAlike (Loop.realTabs_of src x w (ν, t) hsx) hph0 hre'
  obtain ⟨hdgR, hmbR, hnR, hlqR, hpR, hdR⟩ := Count.real_bounds hph0 hT
  have hTA := hT
  obtain ⟨hT1, hT2, hT3, -, -, hcapR⟩ := hTA
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
  have hrs : schedSoFar tr ((2 * j.val + r.val) % 2) dom xs hir
      ((Array.range (a + 1)).map fun q =>
        wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (i0 + 1 - (a + 1) + q + 1)) = .ok sched' := by
    rw [hdm, hxe]
    exact hs''
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
  obtain ⟨alg', hal'⟩ := schedSoFar_ok_rounds hs''
  have htrm : tr.st.m = (Masked.stAt x j).m := by rw [hT1, htm]
  have htrk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hT1, htk]
  have hmT : 14 ≤ tr.st.m := by rw [htrm]; omega
  have hAA' : alg'.size = A := by
    rw [(alg_cnt hmT (by rw [htrk]; omega) hmbR hal').1, htrm, htrk, hAd]
  have hk0R : 1 ≤ tr.sch.k0 := by rw [hT2]; exact hk0
  have hnR' : tr.sch.n < 2 ^ 62 := by rw [hT2]; exact hnn
  have hqR : ∀ i, (tr.sch.levels.getD i default).q < 2 ^ 63 := by rw [hT2]; exact hq
  have hn0R : tr.sch.n ≠ 0 := by rw [hT2]; exact hn0
  have hmkR : tr.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by rw [hT1]; exact hmkt
  have hkR : tr.sch.k0 = (Masked.zkTab x j).sch.k0 := by rw [hT2, hsch]
  have hlvR : tr.sch.levels.size = (Masked.zkTab x j).sch.levels.size := by rw [hT2, hsch]
  have hschR : _root_.Flock.Zk.schedule (Masked.stAt x j).m = .ok (tr.sch, tp) := by rw [hT2, ← hst]; exact htp
  have hOL := Count.openList_length hschR (c := xs'.size + 2) (by omega)
  have hC1 : alg'.size + (xs'.size + 2) + 1 + tr.sch.k0 + 4 ≤ cr.size := by omega
  have hC2 : cr.size < alg'.size + (Count.openList (xs'.size + 2) tr.sch).length := by omega
  obtain ⟨hlt', hpast⟩ := LigOff.sched_past hal' hs'' hdomR hdgR hcapR hmbR hmkR hXb' hk0R hnR' hqR hmT hn0R hpR hdR
    hC2 hC1 le_rfl
  -- the round's coins: the plan's count, the real schedule's
  have hsc : ∀ vals, (ofLoop Masked.dataE L₁ (Loop.realOf src)).scheduled x w (ν, t) ((m₀, ρ₀) :: ops) vals =
      (sched'.getD cr.size (.empty, 0)).1.data.toList := by
    intro vals
    rw [ofLoop_scheduled Masked.dataE L₁ (Loop.realOf src) x w (ν, t) (m₀, ρ₀) ops vals hsx (q := i0 + 1) (by omega)
      hre hs' (by rw [hj1, ← hcrsz]; exact hlt'), hj1, hcrsz]
  have hcoins : (ofLoop Masked.dataE L₁ (Loop.realOf src)).coins x ((m₀, ρ₀) :: ops).length =
      (sched'.getD cr.size (.empty, 0)).2 := by
    show loopNr x (((m₀, ρ₀) :: ops).length - 1) = _
    rw [List.length_cons, Nat.add_sub_cancel,
      show ops.length = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) + cr.size by
        rw [← Masked.repBase_eq hsx (2 * j.val + r.val)]; omega,
      loopNr_rep hsx hq2 (by rw [hcrsz]; exact hR0), hst]
    have hpre := Count.sched_prefix tr r.val dm xs' hir cr sched' hs'' hdomR hdgR hcapR hmbR hXb' hk0R hnR' hqR
    rw [hT1, hts', hXS', Nat.add_comm (spec.points * (Masked.stAt x j).regions.size) 2] at hpre
    refine Count.getD_of_prefix (hpre.trans ?_) hlt'
    rw [Count.repPlan_list hschR]
    exact List.prefix_append _ _
  -- the frame's state: outside the inner proof, and at the close holding all of `ρ`'s words
  have hin : m.inner = none :=
    repState_answer_inner hT hdomP hdgR hmbR (by rw [hxs]; exact hXb) hnR hlqR hpR hdR
      (by rw [Array.size_map, Array.size_range]) hrs (by rw [← hcrsz]; exact hlt') hA'
  have hlayE : lay = Masked.layE x j := by
    obtain ⟨sch', tp', hs3, hl3⟩ := start_lay hds
    rw [hz, Except.ok.injEq, Prod.mk.injEq] at hs3
    rw [hl3, ← hs3.1, hst]
    try rfl
  have hlay : lay = _root_.Flock.Zk.PadLayout.new tph.st.m tph.st.kLog tph.sch.k0 := by
    rw [hlayE, hts', hsch]
    try rfl
  have htp' : _root_.Flock.Zk.schedule tph.st.m = .ok (tph.sch, tp) := by
    rw [htm, ← hst]
    exact htp
  obtain ⟨hCA, -, -, -⟩ := ofLoop_cfg_place Masked.dataE L₁ (Loop.realOf src) x (ν, t) ((m₀, ρ₀) :: ops) hsx
    (q := i0 + 1) (by simp only [List.length_cons]; omega) hloc hph0
  have hord : ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size =
      c'.order.size := by rw [hCA.order]
  have hcl : cr.size = alg'.size + (xs'.size + 2) + 1 + tr.sch.k0 + 4 → m.l0 = true →
      m.pos = ((ofLoop Masked.dataE L₁ (Loop.realOf src)).cfg x (ν, t) ((m₀, ρ₀) :: ops)).order.size ∧
        m.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS := by
    intro hC _
    have hA'' := hA'
    rw [show a + 1 = alg.size + (xs.size + 2) + 1 + (tph.sch.k0 + 3) + 1 by omega] at hA''
    obtain ⟨-, -, -, hps, hrh⟩ := RhoKeep.repState_closeIn hcc hal (by rw [htm]; omega) hdg (placeholder_cap hph0)
      hdomP hmkt (by rw [hxs]; exact hXb) htp' hlay hc' hord
      (by rw [Array.size_map, Array.size_range]) hA''
    exact ⟨hps, hrh⟩
  -- the frame takes it
  refine schedTaken_of hA hin ?_
  rw [hsc]
  exact lig_taken (ofLoop Masked.dataE L₁ (Loop.realOf src)) x (ν, t) ((m₀, ρ₀) :: ops) hC1 hpast hin hlm hcl
    (fun h => honm (by rw [Array.size_map, Array.size_range]; omega)) hcoins

end LeanRelease

end Claims.ZKServer
