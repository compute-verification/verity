module

public import Proofs.Flock.SessionLoop.TakenVals
public import Proofs.Flock.SessionLoop.Inner
import all Init.Data.String.Defs
public import Proofs.Flock.Claims.ZK.FrameInner
public import Proofs.Flock.Claims.ZK.L0PinFlags
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ZK.FrameRanked

/-!
# The session loop's frame past the inner proof's label

`LeanRelease.LiveInner` is `LiveRoundEq`'s equation at the inner proof's rounds, which the loop's frame releases itself
(`LeanRelease.withInner`). Its release side rests on the frame being outside every masked phase there, so that it frees
no word of the round and releases it as it came:

* `Loop.InnerOff`: once the inner proof's label came (`Masker.inner`), the frame is outside a masked phase (`on`), past
  the opening's label (`done`) and, at a layout with level 0, past its close (`doneL0`). Every label keeps it
  (`innerOff_label`: the inner proof's label asks for it, and no later label opens a phase), and so does every absorbed
  value, round, check, answer and step of the frame (`innerOff_step`), so `Frame.run` from `{}` keeps it
  (`innerOff_run`), and so does `Loop.repState` (`innerOff_repState`).
* `Loop.roundFree_offAt`, `Loop.check_offAt`: outside every masked phase, past the inner proof's label or not, a round
  frees no word and leaves as its frames came, so the frame's check passes a device round only where it is the
  scheduled round.
* `Loop.check_inner0`, `Loop.check_innerSched`: the inner proof's round 0 passes at any state as it came, and its later
  rounds do outside a masked phase at a row of pads within a `u64`'s count; the frame's own rounds (`Frame.innerSched`)
  pass only as themselves.
* `Loop.InnerOffAny`: the states `Loop.progOf` and `Loop.freshOf` give keep it (`innerOffAny_progOf`,
  `innerOffAny_freshOf`), and so does every answer (`innerOffAny_answer`), so the state the frame takes coins at does.
  `LeanRelease.ofLoop_answered_on`: at the session loop's own frames, once the inner proof's label came the state a
  round is checked at is outside every masked phase. `LeanRelease.ofLoop_answered_inner`: past a rep's schedule, that
  state counts the inner proof's round, below its last, at a row of pads within a `u64`'s count.
* `LeanRelease.release_of_live`, `LeanRelease.scheduled_innerAt`, `LeanRelease.release_innerAt`: at the inner proof's
  rounds (`Loop.InnerAt`) `withInner`'s session loop releases the inner proof's round `Loop.innerOf` schedules, as it
  came, after a live run's device round.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock (Stmt SessionSpec)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker isLabel roundFree INNER_ROUNDS check labelFrame wordBytes words)
open _root_.Flock.Firewall.Finish (Tab)
open FlockSoundness.Refine (Frame encs)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **Past the inner proof's label the frame is outside every masked phase**: once the label came, it is outside a
masked phase, past the opening's label and, at a layout with level 0, past level 0's close. -/
def InnerOff (c : Cfg) (s : Masker) : Prop :=
  s.inner.isSome = true → s.on = false ∧ s.done = true ∧ (c.lig = true → s.doneL0 = true)

theorem innerOff_empty (c : Cfg) : InnerOff c {} := by
  intro h
  simp at h

theorem innerOff_rel {c : Cfg} {s s' : Masker} (hs : InnerOff c s) (h : Flags.Rel s s') : InnerOff c s' := by
  obtain ⟨a1, a2, -, a4, a5, -⟩ := h
  intro hq
  rw [a5] at hq
  obtain ⟨x, y, z⟩ := hs hq
  exact ⟨a2.trans x, a1.trans y, fun hl => a4.trans (z hl)⟩

/-- The inner proof's label, where the frame takes it, leaves it outside every masked phase. -/
theorem innerOff_labelInner {c : Cfg} {s : Masker} {q : Masker × ByteArray}
    (h : s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 = .ok q) :
    q.1.on = false ∧ q.1.done = true ∧ (c.lig = true → q.1.doneL0 = true) := by
  obtain ⟨p1, p2, p3, p4⟩ := Ranked.inner_other
  cases hi : s.inner with
  | some k =>
    simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi] at h
    all_goals cases h
  | none =>
    cases hd : s.done
    · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd] at h
      all_goals cases h
    · cases hon : s.on
      · cases hlig : c.lig <;> cases hdl : s.doneL0 <;>
          simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon, hlig, hdl] at h
        all_goals cases h
        all_goals exact ⟨rfl, rfl, by simp [hdl]⟩
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
        all_goals cases h

/-- Outside every masked phase, past the opening's label and level 0's close, a label other than the inner proof's
opens none: the zerocheck's label comes only before the opening's, and level 0's opening only before its close. -/
theorem innerOff_labelPast {c : Cfg} {s : Masker} {l : ByteArray} {q : Masker × ByteArray} (hon : s.on = false)
    (hd : s.done = true) (hL : c.lig = true → s.doneL0 = true) (hI : isLabel l _root_.Flock.Zk.LABEL_INNER = false)
    (h : s.label c l = .ok q) : q.1.on = false ∧ q.1.done = true ∧ (c.lig = true → q.1.doneL0 = true) := by
  have hX : c.lig = false ∨ s.doneL0 = true := by
    cases hlig : c.lig
    · exact Or.inl rfl
    · exact Or.inr (hL hlig)
  cases hl0 : s.l0
  · rcases hX with hX | hX <;> simp [Masker.label, hon, hd, hI, hl0, hX] at h
    all_goals cases h
    all_goals exact ⟨hon, hd, hL⟩
  · cases hC : isLabel l _root_.Flock.Zk.LABEL_CLOSE
    · rcases hX with hX | hX <;> simp [Masker.label, hon, hd, hI, hl0, hC, hX] at h
      all_goals cases h
      all_goals exact ⟨hon, hd, hL⟩
    · by_cases hpos : s.pos = c.order.size
      · by_cases hrho : s.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS
        · simp [Masker.label, hon, hd, hI, hl0, hC, hpos, hrho] at h
          all_goals cases h
          all_goals exact ⟨rfl, rfl, fun _ => rfl⟩
        · simp [Masker.label, hon, hd, hI, hl0, hC, hpos, hrho] at h
          all_goals cases h
      · simp [Masker.label, hon, hd, hI, hl0, hC, hpos] at h
        all_goals cases h

theorem innerOff_label {c : Cfg} {s : Masker} {l : ByteArray} {q : Masker × ByteArray} (hs : InnerOff c s)
    (h : s.label c l = .ok q) : InnerOff c q.1 := by
  intro hq
  cases hI : isLabel l _root_.Flock.Zk.LABEL_INNER
  · have hb : l ≠ _root_.Flock.Zk.LABEL_INNER.toUTF8 := by
      rintro rfl
      simp [Ranked.isLabel_self] at hI
    rw [Sched.label_inner c s l hb h] at hq
    obtain ⟨hon, hd, hL⟩ := hs hq
    exact innerOff_labelPast hon hd hL hI h
  · rw [Ranked.eq_of_isLabel hI] at h
    exact innerOff_labelInner h

/-- **A round keeps the frame outside every masked phase past the inner proof's label**, whatever its bytes. -/
theorem innerOff_roundFree {c : Cfg} {s : Masker} {content : ByteArray} {n : ℕ} {q : ByteArray × Masker × Array ℕ}
    (hs : InnerOff c s) (h : roundFree c s content n = .ok q) : InnerOff c q.2.1 := by
  unfold roundFree at h
  by_cases hI : (s.inner == some INNER_ROUNDS) = true
  · simp only [hI, ↓reduceIte] at h
    cases h
  simp only [hI, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨r, hr, h⟩ := bindV_ok h
  have hQ := Ranked.forIn_inv (fun x : Option (ByteArray × Masker × Array ℕ) × Masker × ByteArray × Array ℕ × ℕ =>
      InnerOff c x.2.1 ∧ ∀ v, x.1 = some v → InnerOff c v.2.1) _ ?step _ _ _
    ⟨hs, fun _ hv => by cases hv⟩ hr
  · obtain ⟨o, m, out, free, pos⟩ := r
    cases o with
    | none => cases h
    | some v => cases h; exact hQ.2 _ rfl
  · intro i x y hx hy
    obtain ⟨o, m, out, free, pos⟩ := x
    obtain ⟨hk, -⟩ := hx
    dsimp only at hy
    by_cases h1 : pos + 16 > content.size
    · rw [ite_eq_left h1] at hy
      by_cases h2 : (pos == content.size) = true
      · rw [ite_eq_left h2] at hy
        cases hy
      · rw [ite_eq_right h2] at hy
        cases hy
    rw [ite_eq_right h1] at hy
    generalize (content.get! pos).toNat = op at hy
    rcases op with _ | _ | _ | _ | _ | _ | op
    all_goals simp only [] at hy
    · cases hy
    · have hy := Ranked.ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · have hy := Ranked.ite_bind_ok hy
      obtain ⟨x, hx, hy⟩ := bindV_ok hy
      cases hy
      exact ⟨innerOff_label hk hx, fun _ hv => by cases hv⟩
    · have hy := Ranked.ite_bind_ok hy
      generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h3 : (!(kind == 2 || kind == 1 && len == 1)) = true
      · rw [ite_eq_left h3] at hy
        cases hy
      rw [ite_eq_right h3] at hy
      obtain ⟨r', hr', hy⟩ := bindV_ok hy
      cases hy
      have hQ' : InnerOff c r'.1 := by
        refine Ranked.forIn_inv (fun x : Masker × ByteArray × Array ℕ => InnerOff c x.1) _ ?_ _ _ _ hk hr'
        intro i x y hx hy
        by_cases h8 : (x.1.on && decide (x.1.pos < c.nPiop)) = true
        · rw [ite_eq_left h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact innerOff_rel hx (Flags.absorb_rel ha)
        · rw [ite_eq_right h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact innerOff_rel hx (Flags.absorb_rel ha)
      exact ⟨hQ', fun _ hv => by cases hv⟩
    · generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h5 : (len == 0 && kind == 2) = true
      · rw [ite_eq_left h5] at hy
        cases hy
        exact ⟨hk, fun _ hv => by cases hv⟩
      rw [ite_eq_right h5] at hy
      by_cases h4 : (pos + 16 != content.size) = true
      · rw [ite_eq_left h4] at hy
        cases hy
      rw [ite_eq_right h4] at hy
      by_cases h6 : (len != n) = true
      · rw [ite_eq_left h6] at hy
        cases hy
      rw [ite_eq_right h6] at hy
      by_cases h7 : (!(kind == 2 || kind == 1 && n == 1)) = true
      · rw [ite_eq_left h7] at hy
        cases hy
      rw [ite_eq_right h7] at hy
      have hk' : InnerOff c { m with owed := some n } := hk
      split at hy
      · obtain ⟨u, -, hy⟩ := bindV_ok hy
        cases hy
        exact ⟨hk, fun v hv => by cases hv; exact hk'⟩
      · cases hy
        exact ⟨hk, fun v hv => by cases hv; exact hk'⟩
    · have hy := Ranked.ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · cases hy

theorem innerOff_round {c : Cfg} {s s' : Masker} {content out : ByteArray} {n : ℕ}
    (h : _root_.Flock.Firewall.Frame.round c s content n = .ok (out, s')) (hs : InnerOff c s) : InnerOff c s' := by
  obtain ⟨fr, hf⟩ := Sched.roundFree_round h
  exact innerOff_roundFree hs hf

theorem innerOff_check {c : Cfg} {s s' : Masker} {v r out : ByteArray} {n : ℕ} {why : Option String}
    (h : check c s v r n = .ok (out, s', why)) (hs : InnerOff c s) : InnerOff c s' := by
  cases why with
  | none => exact innerOff_round (check_none h) hs
  | some w => exact innerOff_round (check_some h) hs

/-- **Coins the frame takes keep it outside every masked phase past the inner proof's label**, at any configuration
they are taken at: past the label they only count the inner proof's rounds, and before it they leave it unread. -/
theorem innerOff_answer {c c' : Cfg} {s s' : Masker} {cs : Array F128} (hs : InnerOff c s)
    (h : s.answer c' cs = .ok s') : InnerOff c s' := by
  cases hi : s.inner with
  | none =>
    intro hq
    rw [Sched.answer_inner hi h] at hq
    simp at hq
  | some k =>
    obtain ⟨hon, hd, hL⟩ := hs (by rw [hi]; rfl)
    unfold Masker.answer at h
    cases ho : s.owed with
    | none => rw [ho] at h; cases h
    | some n =>
      by_cases hc : (cs.size != n) = true
      · simp only [ho, hc, ↓reduceIte] at h
        cases h
      simp only [ho, hc, Bool.false_eq_true, ↓reduceIte] at h
      simp only [hi] at h
      obtain ⟨u, -, h⟩ := bindV_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      intro _
      split <;> (try split) <;> exact ⟨hon, hd, hL⟩

open _root_.Flock.Firewall.Frame (Reply) in
/-- **A step of the frame keeps it outside every masked phase past the inner proof's label**, whatever the request. -/
theorem innerOff_step (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (q : _root_.Flock.Firewall.Frame.Req)
    (hs : InnerOff c st.m) : InnerOff c (_root_.Flock.Firewall.Frame.step c st q).2.m := by
  cases q with
  | round n v p =>
    obtain ⟨s, b⟩ := st
    cases ho : s.owed with
    | some k =>
      simp only [_root_.Flock.Firewall.Frame.step, ho]
      exact hs
    | none =>
      cases p with
      | sched r =>
        cases hc : check c s v r n with
        | error e =>
          cases hst : e.startsWith _root_.Flock.Firewall.Frame.STOP <;>
            simp only [_root_.Flock.Firewall.Frame.step, ho, hc, hst, Bool.false_eq_true, ↓reduceIte] <;>
            exact hs
        | ok x =>
          obtain ⟨out, s', why⟩ := x
          simp only [_root_.Flock.Firewall.Frame.step, ho, hc]
          exact innerOff_check hc hs
      | fail =>
        simp only [_root_.Flock.Firewall.Frame.step, ho, bind, Except.bind]
        generalize hR : _root_.Flock.Firewall.Frame.round c _ v n = R
        cases R with
        | error e =>
          cases hst : e.startsWith _root_.Flock.Firewall.Frame.STOP <;>
            simp only [hst, Bool.false_eq_true, ↓reduceIte] <;> exact hs
        | ok x =>
          obtain ⟨out, s'⟩ := x
          simp only [pure, Except.pure]
          exact innerOff_round hR hs
      | own =>
        simp only [_root_.Flock.Firewall.Frame.step, ho, bind, Except.bind]
        generalize hR : _root_.Flock.Firewall.Frame.round c _ v n = R
        cases R with
        | error e =>
          cases hst : e.startsWith _root_.Flock.Firewall.Frame.STOP <;>
            simp only [hst, Bool.false_eq_true, ↓reduceIte] <;> exact hs
        | ok x =>
          obtain ⟨out, s'⟩ := x
          cases hn : s'.inner.isNone
          · simp only [hn, pure, Except.pure, Bool.false_eq_true, ↓reduceIte]
            exact innerOff_round hR hs
          · simp only [hn, ↓reduceIte, throw, throwThe, MonadExceptOf.throw]
            split <;> exact hs
  | coins cs =>
    cases cs with
    | error e => exact hs
    | ok cs =>
      simp only [_root_.Flock.Firewall.Frame.step]
      split
      · rename_i heq
        exact innerOff_answer hs heq
      · exact hs
  | prior k b =>
    simp only [_root_.Flock.Firewall.Frame.step]
    split
    · exact hs
    · split
      · exact hs
      · exact hs
  | finish =>
    simp only [_root_.Flock.Firewall.Frame.step]
    split <;> exact hs

theorem innerOff_stepEq {c : Cfg} {st st' : _root_.Flock.Firewall.Frame.State} {q : _root_.Flock.Firewall.Frame.Req}
    {a : _root_.Flock.Firewall.Frame.Reply} (h : _root_.Flock.Firewall.Frame.step c st q = (a, st'))
    (hs : InnerOff c st.m) : InnerOff c st'.m := by
  have h1 := innerOff_step c st q hs
  rw [h] at h1
  exact h1

/-- **The frame over any requests stays outside every masked phase past the inner proof's label.** -/
theorem innerOff_run (c : Cfg) : ∀ (st : _root_.Flock.Firewall.Frame.State) (qs : List _root_.Flock.Firewall.Frame.Req),
    InnerOff c st.m → InnerOff c (_root_.Flock.Firewall.Frame.run c st qs).1.m
  | st, [], hs => hs
  | st, q :: qs, hs => by
    simp only [_root_.Flock.Firewall.Frame.run]
    have h1 := innerOff_step c st q hs
    split
    · exact h1
    · exact innerOff_run c _ qs h1

theorem innerOff_runIf (c : Cfg) (qs : List _root_.Flock.Firewall.Frame.Req) :
    InnerOff c
      (if (_root_.Flock.Firewall.Frame.run c {} qs).2 then {} else (_root_.Flock.Firewall.Frame.run c {} qs).1.m) := by
  split
  · exact innerOff_empty c
  · exact innerOff_run c {} qs (innerOff_empty c)

theorem innerOff_repState (c : Cfg) (t : Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (i : ℕ) :
    InnerOff c (repState c t rep dom xs hi cr0 cr ns i) := by
  unfold repState
  split
  · exact innerOff_empty c
  · exact innerOff_runIf c _

/-- **A round outside every masked phase, past the inner proof's label or not**: where the frame takes it, it frees no
word and releases its frames as they came, then the squeeze. -/
theorem roundFree_offAt {c : Cfg} {s : Masker} {fs : List Frame} {k : UInt8} {n : ℕ} (hon : s.on = false)
    (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok q) :
    q.2.2 = #[] ∧ q.1.data.toList = encs (maskFrames (fun _ v => v) 0 fs) ++ (Frame.sq k n).enc := by
  have hs := roundFree_ok_inner h
  obtain ⟨r, hr, h1, h2⟩ := foldlM_stepAt_off c fs s ByteArray.empty #[] 0 0 hon hp
  have hfr := foldlM_stepAt_off_fr c fs s ByteArray.empty #[] 0 r hon hp hr
  rw [roundFree_encs c s fs k n hs hm hk h0 hn, hr] at h
  obtain ⟨m, o, fr, q'⟩ := r
  simp only at h1 h2 hfr
  subst h1 hfr
  obtain ⟨a, ha, h⟩ := bindV_ok h
  cases ha
  have e := exMap_ok_eq h
  subst e
  refine ⟨rfl, ?_⟩
  dsimp only
  rw [data_append', h2, sq_enc]
  simp

/-- **Outside every masked phase a round passes only as itself**: where the frame's check passes the device round `v`
against the scheduled round `r`, past the inner proof's label or not, `v` is `r` and leaves as `r`'s frames. -/
theorem check_offAt {c : Cfg} {m s' : Masker} {fs : List Frame} {k : UInt8} {n : ℕ} {v out : ByteArray}
    (hon : m.on = false) (hm : ∀ f ∈ fs, midFrame f) (hp : ∀ f ∈ fs, plainFrame f) (hk : k = 2 ∨ (k = 1 ∧ n = 1))
    (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64)
    (h : check c m v (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok (out, s', none)) :
    v = ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray ∧
      out.data.toList = encs (maskFrames (fun _ v => v) 0 fs) ++ (Frame.sq k n).enc := by
  obtain ⟨q, hq, hsz, hd⟩ := passes_none (check_none_passes h)
  rw [(roundFree_offAt hon hm hp hk h0 hn hq).1] at hd
  have hv := eq_of_firstDiff hsz hd
  refine ⟨hv, ?_⟩
  obtain ⟨fr, hf⟩ := Sched.roundFree_round (check_none h)
  rw [hv] at hf
  exact (roundFree_offAt hon hm hp hk h0 hn hf).2

/-- The inner proof's label, where the frame takes it, leaves as its label frame. -/
theorem label_inner_out (c : Cfg) (s : Masker) {q : Masker × ByteArray}
    (h : s.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 = .ok q) :
    q.2 = labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 := by
  obtain ⟨p1, p2, p3, p4⟩ := Ranked.inner_other
  cases hi : s.inner with
  | some k =>
    simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi] at h
    cases h
  | none =>
    cases hd : s.done
    · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd] at h
      cases h
    · cases hon : s.on
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
        split at h
        · cases h
        · cases h
          exact congrArg labelFrame String.toUTF8_eq_toByteArray.symm
      · simp [Masker.label, Ranked.isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
        cases h

theorem stepAt_label_inner (c : Cfg) (r : Masker × ByteArray × Array ℕ × ℕ) {r' : Masker × ByteArray × Array ℕ × ℕ}
    (h : stepAt c r (.label _root_.Flock.Zk.LABEL_INNER.toUTF8) = .ok r') :
    r'.1.on = false ∧
      r'.2.1.data.toList = r.2.1.data.toList ++ (Frame.label _root_.Flock.Zk.LABEL_INNER.toUTF8).enc := by
  simp only [stepAt] at h
  cases hl : r.1.label c _root_.Flock.Zk.LABEL_INNER.toUTF8 with
  | error e => rw [hl] at h; cases h
  | ok ql =>
    rw [hl] at h
    cases h
    refine ⟨label_inner_on c r.1 hl, ?_⟩
    show (r.2.1 ++ ql.2).data.toList = _
    rw [data_append', label_inner_out c r.1 hl, labelFrame_data]

theorem stepAt_one_off (c : Cfg) (r : Masker × ByteArray × Array ℕ × ℕ) (v : F128) (hon : r.1.on = false)
    {r' : Masker × ByteArray × Array ℕ × ℕ} (h : stepAt c r (.one v) = .ok r') :
    r'.1.on = false ∧ r'.2.1.data.toList = r.2.1.data.toList ++ (Frame.one v).enc := by
  obtain ⟨s, o, fr, p⟩ := r
  obtain ⟨r'', hr, h1, h2⟩ := foldlM_stepAt_off c [.one v] s o fr p 0 hon
    (fun f hf => by rw [List.mem_singleton.mp hf]; trivial)
  have hf : [Frame.one v].foldlM (stepAt c) (s, o, fr, p) = .ok r' := by
    rw [List.foldlM_cons, h] <;> rfl
  obtain rfl : r'' = r' := Except.ok.inj (hr.symm.trans hf)
  refine ⟨by rw [h1]; exact hon, ?_⟩
  rw [h2]
  simp [maskFrames, encs]

/-- **The inner proof's round 0 leaves as it came**: wherever the frame takes it, it frees no word and releases the
label frame, `final_c'` and `ε`'s squeeze unmasked. -/
theorem roundFree_inner0_out {c : Cfg} {s : Masker} {w' : ByteArray} (hw : w'.size = 16) {n : ℕ}
    {q : ByteArray × Masker × Array ℕ}
    (h : roundFree c s (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1) n =
      .ok q) :
    q.2.2 = #[] ∧ q.1.data.toList =
      (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList := by
  refine ⟨roundFree_inner0 hw h, ?_⟩
  have hs := roundFree_ok_inner h
  have ht : (ByteArray.mk (Array.replicate 64 0)).size = 64 := by simp [ByteArray.size]
  have hX := ownRound_encs (c := c) (s := s) (n := n) hw ht (k := 0) (by decide)
  have hE : labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1 =
      ByteArray.mk (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0) ++ (Frame.sq 1 1).enc).toArray := by
    rw [show Frame.sq 1 1 = ownSq n 0 from rfl, ← hX, Array.toArray_toList]
    rfl
  rw [hE] at h ⊢
  have hsq := roundFree_sq hs h
  obtain ⟨-, -, hn1, -⟩ := hsq
  have hsz : (ByteArray.mk (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0) ++
      (Frame.sq 1 1).enc).toArray).size - 16 =
        (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0)).length := by
    rw [size_list, List.length_append, Frame.enc_length]
    simp [Frame.payLen]
  rw [hsz] at hn1
  have hh := (header_at (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0) ++ (Frame.sq 1 1).enc)
    (encs (ownFrames c s w' (ByteArray.mk (Array.replicate 64 0)) 0)).length (.sq 1 1) []
    (by rw [List.drop_left, List.append_nil]) (by decide)).2.2
  rw [hh] at hn1
  obtain rfl : n = 1 := hn1.symm
  rw [roundFree_encs c s _ 1 1 hs (ownFrames0_mid c s w' _) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)] at h
  obtain ⟨a, ha, h⟩ := bindV_ok h
  obtain rfl := exMap_ok_eq h
  simp only [ownFrames, List.foldlM_cons, List.foldlM_nil] at ha
  obtain ⟨b, hb, ha⟩ := bindV_ok ha
  obtain ⟨d, hd, ha⟩ := bindV_ok ha
  obtain rfl : d = a := Except.ok.inj ha
  obtain ⟨hon, hb1⟩ := stepAt_label_inner c _ hb
  obtain ⟨-, hd1⟩ := stepAt_one_off c b _ hon hd
  dsimp only
  rw [data_append', hd1, hb1, sq_enc]
  simp [ownFrames, encs]

/-- **The inner proof's round 0 passes only as itself**: where the frame's check passes the device round `v` against
the scheduled round 0, at any state, `v` is that round and leaves as it came. -/
theorem check_inner0 {c : Cfg} {m s' : Masker} {w' : ByteArray} (hw : w'.size = 16) {n : ℕ} {v out : ByteArray}
    (h : check c m v (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1) n =
      .ok (out, s', none)) :
    v = labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1 ∧
      out.data.toList =
        (labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++ w' ++ header 4 1 1).data.toList := by
  obtain ⟨q, hq, hsz, hd⟩ := passes_none (check_none_passes h)
  rw [roundFree_inner0 hw hq] at hd
  have hv := eq_of_firstDiff hsz hd
  refine ⟨hv, ?_⟩
  obtain ⟨fr, hf⟩ := Sched.roundFree_round (check_none h)
  rw [hv] at hf
  exact (roundFree_inner0_out hw hf).2

theorem ownFrames_plain (c : Cfg) (s : Masker) (w t : ByteArray) :
    ∀ {k : ℕ}, 0 < k → ∀ f ∈ ownFrames c s w t k, plainFrame f
  | 0, h, _, _ => (Nat.lt_irrefl 0 h).elim
  | 1, _, f, hf => by rw [List.mem_singleton.mp hf]; trivial
  | 2, _, f, hf => by rw [List.mem_singleton.mp hf]; trivial
  | _ + 3, _, f, hf => by rw [List.mem_singleton.mp hf]; trivial

theorem maskFrames_vec (vs : List F128) : maskFrames (fun _ v => v) 0 [Frame.vec vs] = [Frame.vec vs] := by
  simp only [maskFrames, maskVals_id]

theorem maskFrames_bytes (b : ByteArray) : maskFrames (fun _ v => v) 0 [Frame.bytes b] = [Frame.bytes b] := by
  simp only [maskFrames]

theorem maskFrames_own (c : Cfg) (s : Masker) (w t : ByteArray) :
    ∀ {k : ℕ}, 0 < k → maskFrames (fun _ v => v) 0 (ownFrames c s w t k) = ownFrames c s w t k
  | 0, h => (Nat.lt_irrefl 0 h).elim
  | 1, _ => maskFrames_vec _
  | 2, _ => maskFrames_bytes t
  | _ + 3, _ => maskFrames_vec _

/-- **The inner proof's rounds 1 to 3 pass only as themselves outside every masked phase**: where the frame's check
passes the device round `v` against the ZK server's own round `k`, at a state outside a masked phase, `v` is that round
and leaves as it came. -/
theorem check_innerAt {c : Cfg} {m s' s : Masker} {w t : ByteArray} (hw : w.size = 16) (ht : t.size = 64) {k n : ℕ}
    (hk0 : 0 < k) (hk4 : k < 4) (hon : m.on = false) (hrow : c.rowH.size < 2 ^ 64) {v out : ByteArray}
    (h : check c m v (ownRound c s w t n k) n = .ok (out, s', none)) :
    v = ownRound c s w t n k ∧ out.data.toList = (ownRound c s w t n k).data.toList := by
  obtain ⟨q, hq, -, -⟩ := passes_none (check_none_passes h)
  have hsq := roundFree_sq (roundFree_ok_inner hq) hq
  have hn : n < 2 ^ 64 := by rw [← hsq.2.2.1]; exact UInt64.toNat_lt _
  obtain ⟨k', m', hkm, hm⟩ := ownSq_sq hn k
  have hX := ownRound_encs (c := c) (s := s) (n := n) hw ht hk4
  rw [hkm] at hX
  obtain ⟨hmn, hk', h0, hn'⟩ := sqOk_frames hX hm hsq
  rw [hmn] at hX
  have e : ownRound c s w t n k = ByteArray.mk (encs (ownFrames c s w t k) ++ (Frame.sq k' n).enc).toArray := by
    rw [← hX, Array.toArray_toList]
  rw [e] at h
  obtain ⟨hv, ho⟩ := check_offAt hon (ownFrames_mid hrow s w t ht hk0) (ownFrames_plain c s w t hk0) hk' h0 hn' h
  refine ⟨hv.trans e.symm, ?_⟩
  rw [ho, maskFrames_own c s w t hk0, hX]

/-- **The frame's own inner rounds pass only as themselves**: where the frame's check passes the device round `v`
against the inner proof's round `k` as the frame schedules it (`Frame.innerSched`), at a state outside a masked phase
and a row of pads within a `u64`'s count past round 0, `v` is that round and the release is that round as it came. -/
theorem check_innerSched {c : Cfg} {m s' s : Masker} {fin : Array ByteArray} {n k : ℕ} {v out : ByteArray}
    (hk : k < INNER_ROUNDS)
    (h : check c m v (_root_.Flock.Firewall.Frame.innerSched c s fin n k) n = .ok (out, s', none))
    (hst : 0 < k → m.on = false ∧ c.rowH.size < 2 ^ 64) :
    v = _root_.Flock.Firewall.Frame.innerSched c s fin n k ∧
      out.data.toList = (_root_.Flock.Firewall.Frame.innerSched c s fin n k).data.toList := by
  have hw : (wordBytes (words (fin.getD 0 .empty) ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16)
      1)).size = 16 := by
    simp only [wordBytes_size, Loop.words_size]
  have ht : (wordBytes (words (fin.getD 2 .empty) 16 4)).size = 64 := by
    simp only [wordBytes_size, Loop.words_size]
  rcases Nat.eq_zero_or_pos k with rfl | hk0
  · have e : _root_.Flock.Firewall.Frame.innerSched c s fin n 0 =
        labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8 ++ header 3 1 1 ++
          wordBytes (words (fin.getD 0 .empty) ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1) ++
          header 4 1 1 := rfl
    rw [e] at h ⊢
    exact check_inner0 hw h
  · have e : _root_.Flock.Firewall.Frame.innerSched c s fin n k = ownRound c s
        (wordBytes (words (fin.getD 0 .empty) ((labelFrame _root_.Flock.Zk.LABEL_INNER.toUTF8).size + 16) 1))
        (wordBytes (words (fin.getD 2 .empty) 16 4)) n k := rfl
    obtain ⟨hon, hrow⟩ := hst hk0
    rw [e] at h ⊢
    exact check_innerAt hw ht hk0 hk hon hrow h

/-- `InnerOff` at some configuration. -/
def InnerOffAny (s : Masker) : Prop := ∃ c, InnerOff c s

theorem innerOffAny_answer {c' : Cfg} {s s' : Masker} {cs : Array F128} (hs : InnerOffAny s)
    (h : s.answer c' cs = .ok s') : InnerOffAny s' := by
  obtain ⟨c, hc⟩ := hs
  exact ⟨c, innerOff_answer hc h⟩

theorem innerOffAny_progOf (x : Msg) (cs : List FlockDeploy.Coin) : InnerOffAny (progOf x cs) := by
  unfold progOf
  split
  · dsimp only
    split
    · exact ⟨_, innerOff_repState _ _ _ _ _ _ _ _ _ _⟩
    · exact ⟨default, innerOff_empty _⟩
  · exact ⟨default, innerOff_empty _⟩

theorem innerOffAny_freshOf (x : Msg) (cs : List FlockDeploy.Coin) {m : Masker} (h : freshOf x cs = .ok m) :
    InnerOffAny m := by
  rcases hx : FlockDeploy.setupOf true x with _ | ⟨s0, spec, sts⟩
  · exact absurd h (freshOf_none x cs hx)
  · rcases hl : cs.length with _ | q
    · unfold freshOf at h
      rw [hx, hl] at h
      simp only at h
      cases h
      exact ⟨default, innerOff_empty _⟩
    · obtain ⟨c', t, dom, xs, hi, alg, -, -, h0 | ⟨-, st', hst, rfl⟩⟩ := freshOf_ok x cs hx hl h
      · rw [h0.2]
        exact ⟨default, innerOff_empty _⟩
      · exact ⟨c', innerOff_stepEq hst (innerOff_empty c')⟩

end Loop

namespace LeanRelease

open Loop in
/-- **At the session loop's own frames, past the inner proof's label a round is checked outside every masked phase**:
where the frame takes the stream's coins after `e0 :: h1` at `m` (`ofLoop_frameAnswer`: at `Loop.progOf` or
`Loop.freshOf` of the decoded coins) and the inner proof's label came, `m` is outside a masked phase. -/
theorem ofLoop_answered_on (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {m : Masker} (ha : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m)
    (hi : m.inner.isSome = true) : m.on = false := by
  have hF : (ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) = .ok m := by
    unfold answered at ha
    split at ha
    · exact ha
    · cases ha
  rw [ofLoop_frameAnswer T L₀ real x ρ e0 h1] at hF
  have hm : InnerOffAny m := by
    rcases takeAt_ok hF with ⟨-, hf⟩ | ⟨-, -, ha⟩ | ⟨-, -, -, hf⟩
    · exact innerOffAny_freshOf x _ hf
    · exact innerOffAny_answer (innerOffAny_progOf x _) ha
    · exact innerOffAny_freshOf x _ hf
  obtain ⟨c, hc⟩ := hm
  exact (hc hi).1

open Loop in
/-- **At the loop's own frames, past a rep's schedule the frame takes coins inside the inner proof**: where the frame
takes the coins after `e0 :: h1` at `m` and the next round is rep stream `k`'s round `j + 1`, past the end of its
schedule so far on its own table and salts (`rsched`, `Count.RealTabs`), the round is the inner proof's round
`j + 1 - rsched.size`, `m` counts it (`repState_own_inner`; past the inner proof's last round the rep's state owes no
coins, `repState_past`), and the round's configuration has a row of pads within a `u64`'s count (`Session.sizesOk`,
where the loop serves `x`). -/
theorem ofLoop_answered_inner (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (hSv : Serves x)
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
    (hlt : rsched.size < j + 1) :
    j - rsched.size < INNER_ROUNDS ∧ m.inner = some (j - rsched.size + 1) ∧
      ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)).rowH.size < 2 ^ 64 := by
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
  obtain ⟨lay, mask, hst, hc', hdm, hxe⟩ := placeholder_ok hph'
  obtain ⟨-, hd, hx2, hlay⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec sts hx) hk
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
  obtain ⟨sched, hs⟩ := schedSoFar_pop_ok hs'
  have hss : sched' = sched := schedSoFar_pop_eq (j := j) hs hs' (by simp) (by omega)
  rw [hss] at hsz
  have hrow : ((ofLoop T L₀ real).cfg x ρ ((m0, r0) :: h1)).rowH.size < 2 ^ 64 := by
    obtain ⟨-, hH, -, -⟩ := ofLoop_cfg_place T L₀ real x ρ ((m0, r0) :: h1) hx hq2 hki hph'
    rw [hH, hc', ofDraws_rowH_size]
    exact hlay lay mask hst
  obtain ⟨i, rfl⟩ : ∃ i, j = sched.size + i := ⟨j - sched.size, by omega⟩
  rw [show sched.size + i - rsched.size = i by omega]
  rcases Nat.lt_or_ge i INNER_ROUNDS with h4 | h4
  · exact ⟨h4, repState_own_inner c' _ t (k % 2) dom xs hi _ _ _ hal hs h4 hF, hrow⟩
  · exact absurd hF (answer_owed_none _ (repState_past c' t (k % 2) dom xs hi _ _ _ hal hs (by omega)))

/-- **The release of a live frame round against a fixed scheduled round**: where a live run's worker sends the device
round `v` at a frame round whose scheduled round is `r` at every value, the frame's check passes `v` against `r` at the
state the frame takes the coins at, and the release is the round the check releases, tagged `0`. -/
theorem release_of_live {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (v : Msg) (hl : L.toSrv.live wk x w ρ κ h = true)
    (hk : L.kindAt x h.length = .masked) (hv : wk x w ρ h 0 = some (v, false)) {r : ByteArray}
    (hsc : ∀ vals, L.scheduled x w ρ h vals = r.data.toList) :
    ∃ m out s', L.answered x ρ h = .ok m ∧
      check (L.cfg x ρ h) { m with stopped := false } ⟨v.toArray⟩ r (L.coins x h.length) = .ok (out, s', none) ∧
      L.release x ρ h false v = 0 :: out.data.toList := by
  obtain ⟨m, out, s', ha, hc⟩ := check_of_live L wk x w ρ κ h v hl hk hv
  rw [hsc, Array.toArray_toList] at hc
  obtain ⟨fr, hf⟩ := Sched.roundFree_round (check_none hc)
  exact ⟨m, out, s', ha, hc, release_of_roundFree L x ρ h false v ha rfl hf⟩

/-- **The round in a device round's place at the inner proof's rounds** of `withInner`'s session loop: where the loop
schedules no round of its own (`Loop.schedOf`), `withInner`'s round (`Loop.innerOf`). -/
theorem scheduled_innerAt
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray)
    (x : Msg) (w : Flock.Wit)
    (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand)
    (p : Msg × Msg) (ops : List (Msg × Msg)) {r : ByteArray}
    (hs : Loop.schedOf (real x w ρ) x (ops.map fun p => CoinLayer.rd p.2) = none)
    (hin : Loop.innerOf (real x w ρ) ((withFrames Masked.dataE L₁ real).cfg x ρ (p :: ops)) (fin x w ρ (p :: ops)) x
      (ops.map fun p => CoinLayer.rd p.2) = some r)
    (vals : String → Option ByteArray) :
    (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).scheduled x w ρ (p :: ops) vals = r.data.toList := by
  rw [scheduled_ofLoopLink, hs]
  show (match Loop.innerOf (real x w ρ) ((withFrames Masked.dataE L₁ real).cfg x ρ (p :: ops)) (fin x w ρ (p :: ops)) x
      (ops.map fun p => CoinLayer.rd p.2) with
    | some r => r.data.toList
    | none => L₁.scheduled x w ρ (p :: ops) vals) = _
  simp only [hin]

open Loop in
/-- **At the inner proof's rounds `withInner`'s session loop releases its own round as it came**: where a live run's
worker sends the device round `v` after a history at the inner proof's rounds (`Loop.InnerAt`), on the rep's own tables
where `real` answers the loop's (`hR`) at a statement the loop serves, the round in the device round's place is the
inner proof's round `Loop.innerOf` schedules, and the device round and the release are that round. Its round 0 passes
at any state (`check_inner0`), its rounds 1 to 3 outside every masked phase (`ofLoop_answered_on`, past the inner
proof's label, which the rep's state counts: `ofLoop_answered_inner`); at its round 4 the frame refuses the round
(`check_inner_done`), and past it the coins before it. -/
theorem release_innerAt
    (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand))
    (src : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (fin : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce × (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand →
      List (Msg × Msg) → Array ByteArray)
    (x : Msg) (w : Flock.Wit) (hSv : Serves x)
    (hR : ∀ (ρ : (CoinLayer.session Loop.loopNr Masked.dataE).refP.Nonce ×
        (CoinLayer.session Loop.loopNr Masked.dataE).refP.Rand) (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (wk : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.Worker)
    (ν : (CoinLayer.session Loop.loopNr Masked.dataE).Nonce) (t : (CoinLayer.session Loop.loopNr Masked.dataE).Tape)
    (κ : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (v : Msg) (h0 : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) =
      true)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false))
    (hI : Loop.InnerAt (real x w (ν, t)) x (ops.map fun p => CoinLayer.rd p.2)) :
    ∃ r, Loop.innerOf (real x w (ν, t)) ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
        (fin x w (ν, t) ((m₀, ρ₀) :: ops)) x (ops.map fun p => CoinLayer.rd p.2) = some r ∧
      (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
        0 :: r.data.toList ∧ v = r.data.toList := by
  have hsch := schedOf_none_of_innerAt hI
  obtain ⟨s0, spec, sts, q, k, i, tr, hr, rsched, hx, hq, hloc, hreal, hrs, hle⟩ := hI
  have hq1 : ops.length = q + 1 := by rw [← hq, List.length_map]
  have hne : ((m₀, ρ₀) :: ops).length ≠ lastOf x := by rw [lastOf_eq, List.length_cons]; omega
  have hlast : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE _ x
  have hkd : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).kindAt x ((m₀, ρ₀) :: ops).length =
      .masked :=
    kindAt_masked_of _ (by rw [List.length_cons]; omega) (by rw [hlast, List.length_cons]; omega)
  have e4 : INNER_ROUNDS = 4 := rfl
  rcases Nat.lt_or_ge (i - rsched.size) INNER_ROUNDS with h4 | h4
  · have hin := innerOf_eq (real x w (ν, t)) ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops))
      (fin x w (ν, t) ((m₀, ρ₀) :: ops)) x _ hx hq hloc hreal hrs hle h4
    refine ⟨_, hin, ?_⟩
    obtain ⟨m, out, s', ha, hc', hrel⟩ := release_of_live (ofLoopLink (withInner Masked.dataE L₁ real fin) src real)
      wk x w (ν, t) κ ((m₀, ρ₀) :: ops) v hl hkd hv (scheduled_innerAt L₁ src real fin x w (ν, t) (m₀, ρ₀) ops hsch hin)
    have hco : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).coins x ((m₀, ρ₀) :: ops).length =
        loopNr x (q + 1) := by
      rw [List.length_cons, hq1]
      rfl
    have hcfg : (ofLoopLink (withInner Masked.dataE L₁ real fin) src real).cfg x (ν, t) ((m₀, ρ₀) :: ops) =
        (withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops) := by
      unfold cfg drawsAt
      rfl
    rw [hco, hcfg] at hc'
    have hst : 0 < i - rsched.size → m.on = false ∧
        ((withFrames Masked.dataE L₁ real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).rowH.size < 2 ^ 64 := by
      intro hpos
      obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
      obtain ⟨-, hin', hrow⟩ := ofLoop_answered_inner Masked.dataE
        (withFiles (withLink (withInner Masked.dataE L₁ real fin) src) src) real x w hSv hR (ν, t) (m₀, ρ₀) ops ha
        hne rfl hx hq1 hloc hreal hrs (by omega)
      refine ⟨ofLoop_answered_on Masked.dataE (withFiles (withLink (withInner Masked.dataE L₁ real fin) src) src) real
        x (ν, t) (m₀, ρ₀) ops ha (by simp [hin']), ?_⟩
      rw [← hcfg]
      exact hrow
    have hcs := check_innerSched h4 hc' hst
    exact ⟨by rw [hrel, hcs.2], congrArg (fun b : ByteArray => b.data.toList) hcs.1⟩
  · exfalso
    obtain ⟨m, out, s', ha, hc'⟩ := check_of_live (ofLoopLink (withInner Masked.dataE L₁ real fin) src real) wk x w
      (ν, t) κ ((m₀, ρ₀) :: ops) v hl hkd hv
    obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
    obtain ⟨h4', hin', -⟩ := ofLoop_answered_inner Masked.dataE
      (withFiles (withLink (withInner Masked.dataE L₁ real fin) src) src) real x w hSv hR (ν, t) (m₀, ρ₀) ops ha
      hne rfl hx hq1 hloc hreal hrs (by omega)
    rw [show j - rsched.size + 1 = INNER_ROUNDS by omega] at hin'
    exact absurd hc' (check_inner_done (s := { m with stopped := false }) hin' _ _ _ _)

end LeanRelease

end Claims.ZKServer
