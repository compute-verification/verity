import Proofs.Flock.SessionLoop.TakenVals
import Proofs.Flock.SessionLoop.Inner

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
* `Loop.InnerOffAny`, `Loop.innerOffAny_answered`: the frame state a release checks a round at (`LeanRelease.answered`)
  keeps it where its transport's states do, as `Loop.progOf` and `Loop.freshOf` do (`innerOffAny_progOf`,
  `innerOffAny_freshOf`). `LeanRelease.ofLoop_answered_on`: at the session loop's own frames, once the inner proof's
  label came the state a round is checked at is outside every masked phase.
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker isLabel roundFree INNER_ROUNDS check)
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
        all_goals exact ⟨hon, hd, by simp [hdl]⟩
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
          all_goals exact ⟨rfl, hd, fun _ => rfl⟩
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
            first
              | (split <;> exact hs)
              | exact hs
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

theorem innerOff_repState (c : Cfg) (t : Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128))
    (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (i : ℕ) :
    InnerOff c (repState c t rep dom xs hi cr0 cr ns i) := by
  unfold repState
  split
  · exact innerOff_empty c
  · dsimp only
    split_ifs
    · exact innerOff_empty c
    · exact innerOff_run c {} _ (innerOff_empty c)

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

theorem innerOffAny_takeAt {n : ℕ} {same : Bool} {cN cP : Cfg} {prog : Masker} {fresh : Except String Masker}
    {cs : Array F128} {m : Masker} (hp : InnerOffAny prog) (hf : ∀ m, fresh = .ok m → InnerOffAny m)
    (h : LeanRelease.takeAt n same cN cP prog fresh cs = .ok m) : InnerOffAny m := by
  unfold LeanRelease.takeAt at h
  split_ifs at h
  · exact hf m h
  · exact innerOffAny_answer hp h
  · obtain ⟨a, -, h⟩ := bindV_ok h
    exact hf m h

/-- **The state a release checks a round at stays outside every masked phase past the inner proof's label**, where its
transport's states do. -/
theorem innerOffAny_answered {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    {m : Masker} (hp : InnerOffAny (L.prog x h)) (hf : ∀ m, L.fresh x h = .ok m → InnerOffAny m)
    (ha : L.answered x ρ h = .ok m) : InnerOffAny m := by
  unfold LeanRelease.answered at ha
  split at ha
  · unfold LeanRelease.frameAnswer at ha
    split at ha
    · exact innerOffAny_takeAt hp hf ha
    · cases ha
      exact hp
  · cases ha

end Loop

namespace LeanRelease

open Loop in
/-- **At the session loop's own frames, past the inner proof's label a round is checked outside every masked phase**:
where the frame takes the stream's coins after `h` at `m` and the inner proof's label came, `m` is outside a masked
phase. -/
theorem ofLoop_answered_on (T : CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (x : Msg) (ρ : (CoinLayer.session Loop.loopNr T).refP.Nonce × (CoinLayer.session Loop.loopNr T).refP.Rand)
    (h : List (Msg × Msg)) {m : Masker} (ha : (ofLoop T L₀ real).answered x ρ h = .ok m)
    (hi : m.inner.isSome = true) : m.on = false := by
  obtain ⟨c, hc⟩ := innerOffAny_answered (ofLoop T L₀ real) x ρ h
    (innerOffAny_progOf x (h.tail.map fun p => CoinLayer.rd p.2))
    (fun m hm => innerOffAny_freshOf x (h.tail.map fun p => CoinLayer.rd p.2) hm) ha
  exact (hc hi).1

end LeanRelease

end Claims.ZKServer
