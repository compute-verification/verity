import Proofs.Flock.Claims.ZK.FrameReal
import Proofs.Flock.Claims.ZK.FrameSched
import Proofs.Flock.Claims.ZK.FrameState

/-!
# The loop's frame state is the deployed frame's check's (`ProgChecks`)

Toward `ofLoop_progChecks`: at a scheduled round the loop's frame state after
it is the placeholder frame's check of the placeholder's scheduled round against itself, and the deployed frame's check
of any device round, at the tape's draws and its scheduled round, leaves the same state but for `stopped`:

* `Frame.run` over appended requests (`run_append`).
* `check_alike_self`: where the frame's check at `c` of a device round against the scheduled round of frames `fs`
  releases a round, the check at an alike configuration `c'` of a scheduled round of alike frames `fs'` against itself
  releases one too, from the same state, its state the same but for `stopped`.
-/

namespace Claims.ZKServer

open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Frame (Draws)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS roundFree passes check standIn)
open FlockSoundness.Refine (Frame encs)

namespace Sched

theorem alike_symm : ∀ {f f' : Frame}, Alike f f' → Alike f' f
  | .dom _, .dom _, h => (h : _ = _).symm
  | .label _, .label _, h => (h : _ = _).symm
  | .bytes _, .bytes _, h => (h : _ = _).symm
  | .one _, .one _, _ => True.intro
  | .vec _, .vec _, h => (h : _ = _).symm
  | .sq _ _, .sq _ _, h => ⟨h.1.symm, h.2.symm⟩

theorem alike_trans : ∀ {f f' f'' : Frame}, Alike f f' → Alike f' f'' → Alike f f''
  | .dom _, .dom _, .dom _, h, h' => (h : _ = _).trans h'
  | .label _, .label _, .label _, h, h' => (h : _ = _).trans h'
  | .bytes _, .bytes _, .bytes _, h, h' => (h : _ = _).trans h'
  | .one _, .one _, .one _, _, _ => True.intro
  | .vec _, .vec _, .vec _, h, h' => (h : _ = _).trans h'
  | .sq _ _, .sq _ _, .sq _ _, h, h' => ⟨h.1.trans h'.1, h.2.trans h'.2⟩

theorem forall₂_alike_symm {fs fs' : List Frame} (h : List.Forall₂ Alike fs fs') : List.Forall₂ Alike fs' fs := by
  induction h with
  | nil => exact .nil
  | cons h _ ih => exact .cons (alike_symm h) ih

theorem forall₂_alike_trans {fs fs' fs'' : List Frame} (h : List.Forall₂ Alike fs fs')
    (h' : List.Forall₂ Alike fs' fs'') : List.Forall₂ Alike fs fs'' := by
  induction h generalizing fs'' with
  | nil => cases h'; exact .nil
  | cons hab _ ih =>
    cases h' with
    | cons hbc h'' => exact .cons (alike_trans hab hbc) (ih h'')

theorem cfgAlike_symm {c c' : Cfg} (h : CfgAlike c c') : CfgAlike c' c :=
  ⟨h.lay.symm, h.order.symm, h.nPiop.symm, h.lig.symm, h.mask.symm, h.size.symm⟩

theorem relM_ok_right {α β : Type} {f : α → β} {x : V α} {b : α} (h : RelM f x (.ok b)) :
    ∃ a, x = .ok a ∧ f a = f b := by
  cases x with
  | ok a => exact ⟨a, rfl, h⟩
  | error e => exact (h : False).elim

theorem roundFree_round {c : Cfg} {s : Masker} {r : ByteArray} {n : ℕ} {out : ByteArray} {s' : Masker}
    (h : _root_.Flock.Firewall.Frame.round c s r n = .ok (out, s')) : ∃ fr, roundFree c s r n = .ok (out, s', fr) := by
  unfold _root_.Flock.Firewall.Frame.round at h
  cases hf : roundFree c s r n with
  | error e => rw [hf] at h; cases h
  | ok x =>
    obtain ⟨o, s'', fr⟩ := x
    rw [hf] at h
    cases h
    exact ⟨fr, rfl⟩

theorem with_stopped_self (m : Masker) {b : Bool} (h : m.stopped = b) : { m with stopped := b } = m := by
  subst h
  rfl

/-- **The frame's check of a round against itself, at its round's state**: where `round` takes the round `r` at `m`,
the check of `r` in its own place releases what `round` does, from `m` as it is. -/
theorem check_self_any {c : Cfg} {m : Masker} {r : ByteArray} {n : ℕ} {out : ByteArray} {s' : Masker}
    (h : _root_.Flock.Firewall.Frame.round c m r n = .ok (out, s')) : ∃ w, check c m r r n = .ok (out, s', w) := by
  cases hs : m.stopped with
  | false => exact ⟨none, check_self hs h⟩
  | true =>
    refine ⟨some "the rep is stopped", ?_⟩
    unfold check standIn
    simp only [hs, ↓reduceIte]
    rw [with_stopped_self m hs, h]
    rfl

/-- **The deployed frame's check and the placeholder's, at one state**: where the frame's check at `c`, from `m` at
any `stopped`, of a device round `v` against the scheduled round of frames `fs` releases a round, the check at an alike
configuration `c'` of the scheduled round of frames `fs'` alike to `fs`, against itself, releases a round from `m`, its
state the same but for `stopped`. Outside the inner proof (`hI`, of the placeholder's frames). -/
theorem check_alike_self {c c' : Cfg} (hc : CfgAlike c c') (m : Masker) (st : Bool) (fs fs' : List Frame) (k : UInt8)
    (n : ℕ) (hs : m.inner ≠ some INNER_ROUNDS) (hm : ∀ f ∈ fs, midFrame f) (hm' : ∀ f ∈ fs', midFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (hA : List.Forall₂ Alike fs fs')
    (hI : ∀ q, fs'.foldlM (stepAt c') (m, ByteArray.empty, #[], 0) = .ok q → q.1.inner = none)
    {v out : ByteArray} {s : Masker} {why : Option String}
    (h : check c { m with stopped := st } v (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n =
      .ok (out, s, why)) :
    ∃ out' s' w, check c' m (ByteArray.mk (encs fs' ++ (Frame.sq k n).enc).toArray)
      (ByteArray.mk (encs fs' ++ (Frame.sq k n).enc).toArray) n = .ok (out', s', w) ∧ unstop s = unstop s' := by
  have hc' := cfgAlike_symm hc
  cases why with
  | none =>
    have hr := check_none h
    have hp := check_none_passes h
    obtain ⟨fs'', hA'', hm'', rfl⟩ := passes_frames c { m with stopped := st } fs k n hs hm hk h0 hn v hp
    obtain ⟨fr, hfr⟩ := roundFree_round hr
    have hR := roundFree_alike hc' m st fs' fs'' k n hs hm' hm'' hk h0 hn
      (forall₂_alike_trans (forall₂_alike_symm hA) hA'') hI
    rw [hfr] at hR
    obtain ⟨⟨o', s', fr'⟩, hq, he⟩ := relM_ok_right hR
    obtain ⟨w, hw⟩ := check_self_any (round_of_roundFree hq)
    exact ⟨o', s', w, hw, (congrArg Prod.fst he).symm⟩
  | some w =>
    have hr := check_some h
    obtain ⟨fr, hfr⟩ := roundFree_round hr
    have hR := roundFree_alike hc' m true fs' fs k n hs hm' hm hk h0 hn (forall₂_alike_symm hA) hI
    have hfr' : roundFree c { m with stopped := true } (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n =
        .ok (out, s, fr) := hfr
    rw [hfr'] at hR
    obtain ⟨⟨o', s', fr'⟩, hq, he⟩ := relM_ok_right hR
    obtain ⟨w', hw⟩ := check_self_any (round_of_roundFree hq)
    exact ⟨o', s', w', hw, (congrArg Prod.fst he).symm⟩

theorem midFrame_alike {f f' : Frame} (h : Alike f f') (hm : midFrame f) : midFrame f' := by
  cases f <;> cases f' <;> simp only [Alike, midFrame] at h hm ⊢ <;>
    first | omega | (obtain ⟨rfl, rfl⟩ := h; exact hm)

theorem notLabel_alike {f f' : Frame} {b : ByteArray} (h : Alike f f') (hf : f ≠ .label b) : f' ≠ .label b := by
  cases f <;> cases f' <;> simp only [Alike] at h <;> first | (subst h; exact hf) | nofun

theorem midAll_alike {fs fs' : List Frame} (hA : List.Forall₂ Alike fs fs') : MidAll fs → MidAll fs' := by
  induction hA with
  | nil => exact fun h => h
  | cons ha _ ih =>
    intro hM g hg
    rcases List.mem_cons.mp hg with rfl | hg
    · exact ⟨midFrame_alike ha (hM _ List.mem_cons_self).1, notLabel_alike ha (hM _ List.mem_cons_self).2⟩
    · exact ih (fun f hf => hM f (List.mem_cons_of_mem _ hf)) g hg

/-- **Two rounds of one shape are frames**: `encs` of alike mid-frames, then one squeeze, of its kind and count. -/
theorem roundAt_frames {r r' : ByteArray × ℕ} (h : RoundAt r r') :
    ∃ (fs fs' : List Frame) (k : UInt8), r.1 = ByteArray.mk (encs fs ++ (Frame.sq k r.2).enc).toArray ∧
      r'.1 = ByteArray.mk (encs fs' ++ (Frame.sq k r.2).enc).toArray ∧ r'.2 = r.2 ∧ List.Forall₂ Alike fs fs' ∧
      MidAll fs ∧ MidAll fs' ∧ (k = 2 ∨ (k = 1 ∧ r.2 = 1)) ∧ ¬ (r.2 = 0 ∧ k = 2) ∧ r.2 < 2 ^ 64 := by
  obtain ⟨hn2, p, p', k, ⟨fs, fs', hp, hp', hA, hM⟩, h1, h1', hk, h0, hn⟩ := h
  have hL : ∀ q : ByteArray, (q ++ _root_.Flock.Transcript.header 4 k r.2).data.toList =
      q.data.toList ++ (Frame.sq k r.2).enc := fun q => by
    simp [Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]
  have e : ∀ b : ByteArray, b = ByteArray.mk b.data.toList.toArray := fun b => by rw [Array.toArray_toList]
  refine ⟨fs, fs', k, ?_, ?_, hn2.symm, hA, hM, midAll_alike hA hM, hk, fun h => h0 h.1, hn⟩
  · rw [e r.1, h1, hL, hp]
  · rw [e r'.1, h1', hL, hp']

/-! ## The inner proof starts at its label alone -/

theorem inner_label_false {b : ByteArray} (hb : b ≠ _root_.Flock.Zk.LABEL_INNER.toUTF8) :
    _root_.Flock.Firewall.Frame.isLabel b _root_.Flock.Zk.LABEL_INNER = false := by
  unfold _root_.Flock.Firewall.Frame.isLabel
  cases e : (b.data == _root_.Flock.Zk.LABEL_INNER.toUTF8.data) with
  | false => rfl
  | true =>
    exfalso
    apply hb
    have := beq_iff_eq.1 e
    cases b
    cases h : _root_.Flock.Zk.LABEL_INNER.toUTF8
    rw [h] at this
    cases this
    rfl

theorem exMap_throw {α β : Type} (f : α → β) (e : String) : (throw e : V α).map f = throw e := rfl

theorem throw_bindV {α β : Type} (e : String) (k : α → V β) : ((throw e : V α) >>= k) = throw e := rfl

/-- A label other than the inner proof's leaves the inner proof's count as it was. -/
theorem label_inner (c : Cfg) (s : Masker) (b : ByteArray) {q : Masker × ByteArray}
    (hb : b ≠ _root_.Flock.Zk.LABEL_INNER.toUTF8) (h : s.label c b = .ok q) : q.1.inner = s.inner := by
  have hl := inner_label_false hb
  have key : (s.label c b).map (fun q => q.1.inner) = (s.label c b).map (fun _ => s.inner) := by
    unfold Masker.label
    cases hm : c.mask <;>
    simp only [hl, Bool.false_eq_true, ↓reduceIte, apply_ite (Except.map _), exMap_bind, exMap_pure, exMap_throw,
      throw_bindV]
  rw [h] at key
  exact Except.ok.inj key

theorem absorb_inner (c : Cfg) (s : Masker) (v : F128) {q : Masker × F128} (h : s.absorb c v = .ok q) :
    q.1.inner = s.inner := by
  unfold Masker.absorb at h
  repeat' split at h
  all_goals first | (cases h; rfl) | cases h

theorem absorb1_inner (c : Cfg) (r r' : Masker × ByteArray × Array ℕ × ℕ) (v : F128) (h : absorb1 c r v = .ok r') :
    r'.1.inner = r.1.inner := by
  unfold absorb1 at h
  cases hb : r.1.absorb c v with
  | error e => rw [hb] at h; cases h
  | ok b => rw [hb] at h; cases h; exact absorb_inner c r.1 v hb

theorem foldlM_absorb1_inner (c : Cfg) : ∀ (vs : List F128) (r r' : Masker × ByteArray × Array ℕ × ℕ),
    vs.foldlM (absorb1 c) r = .ok r' → r'.1.inner = r.1.inner
  | [], r, r', h => by cases h; rfl
  | v :: vs, r, r', h => by
    rw [List.foldlM_cons] at h
    cases ha : absorb1 c r v with
    | error e => rw [ha] at h; cases h
    | ok q => rw [ha] at h; rw [foldlM_absorb1_inner c vs q r' h, absorb1_inner c r q v ha]

/-- A frame's step leaves the inner proof's count as it was, but at the inner proof's label. -/
theorem stepAt_inner (c : Cfg) (r r' : Masker × ByteArray × Array ℕ × ℕ) (f : Frame)
    (hf : f ≠ .label _root_.Flock.Zk.LABEL_INNER.toUTF8) (h : stepAt c r f = .ok r') : r'.1.inner = r.1.inner := by
  cases f with
  | dom b => cases h; rfl
  | label b =>
    simp only [stepAt] at h
    cases hl : r.1.label c b with
    | error e => rw [hl] at h; cases h
    | ok q => rw [hl] at h; cases h; exact label_inner c r.1 b (fun e => hf (by rw [e])) hl
  | bytes b => cases h; rfl
  | one v => have e := foldlM_absorb1_inner c [v] _ r' h; exact e
  | vec vs => have e := foldlM_absorb1_inner c vs _ r' h; exact e
  | sq k n => cases h; rfl

theorem foldlM_stepAt_inner (c : Cfg) : ∀ (fs : List Frame) (r r' : Masker × ByteArray × Array ℕ × ℕ),
    (∀ f ∈ fs, f ≠ .label _root_.Flock.Zk.LABEL_INNER.toUTF8) → fs.foldlM (stepAt c) r = .ok r' →
    r'.1.inner = r.1.inner
  | [], r, r', _, h => by cases h; rfl
  | f :: fs, r, r', hf, h => by
    rw [List.foldlM_cons] at h
    cases ha : stepAt c r f with
    | error e => rw [ha] at h; cases h
    | ok q =>
      rw [ha] at h
      rw [foldlM_stepAt_inner c fs q r' (fun g hg => hf g (List.mem_cons_of_mem _ hg)) h,
        stepAt_inner c r q f (hf f List.mem_cons_self) ha]

/-- **A scheduled round starts no inner proof**: from a state outside the inner proof, the frame's steps over a
scheduled round's frames (`MidAll`, none the inner proof's label) stay outside it. -/
theorem foldlM_midAll_inner {c : Cfg} {m : Masker} (hm : m.inner = none) {fs : List Frame} (hM : MidAll fs)
    {q : Masker × ByteArray × Array ℕ × ℕ} (h : fs.foldlM (stepAt c) (m, ByteArray.empty, #[], 0) = .ok q) :
    q.1.inner = none := by
  rw [foldlM_stepAt_inner c fs _ q (fun f hf => (hM f hf).2) h]
  exact hm

/-- **A scheduled round's form**: frames, none the inner proof's label, then its squeeze of its coin count. -/
def Framed (r : ByteArray × ℕ) : Prop :=
  ∃ (fs : List Frame) (k : UInt8), r.1 = ByteArray.mk (encs fs ++ (Frame.sq k r.2).enc).toArray ∧ MidAll fs ∧
    (k = 2 ∨ (k = 1 ∧ r.2 = 1)) ∧ ¬ (r.2 = 0 ∧ k = 2) ∧ r.2 < 2 ^ 64

theorem roundFree_inner {c : Cfg} {s : Masker} {r : ByteArray × ℕ} (hr : Framed r) (hs : s.inner = none)
    {o : ByteArray} {s' : Masker} {fr : Array ℕ} (h : roundFree c s r.1 r.2 = .ok (o, s', fr)) : s'.inner = none := by
  obtain ⟨fs, k, h1, hM, hk, h0, hn⟩ := hr
  rw [h1, roundFree_encs c s fs k r.2 (by rw [hs]; nofun) (fun f hf => (hM f hf).1) hk h0 hn] at h
  cases hf : fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0) with
  | error e => rw [hf] at h; cases h
  | ok q =>
    have hq : q.1.inner = none := by rw [foldlM_stepAt_inner c fs _ q (fun f hf => (hM f hf).2) hf]; exact hs
    rw [hf] at h
    simp only [bind, Except.bind, hq] at h
    cases h
    rfl

theorem round_inner {c : Cfg} {s : Masker} {r : ByteArray × ℕ} (hr : Framed r) (hs : s.inner = none)
    {o : ByteArray} {s' : Masker} (h : _root_.Flock.Firewall.Frame.round c s r.1 r.2 = .ok (o, s')) :
    s'.inner = none := by
  obtain ⟨fr, hf⟩ := roundFree_round h
  exact roundFree_inner hr hs hf

/-- **A framed round checked in its own place stays outside the inner proof**, whether it passes or stands in. -/
theorem check_inner {c : Cfg} {s : Masker} {r : ByteArray × ℕ} (hr : Framed r) (hs : s.inner = none)
    {o : ByteArray} {s' : Masker} {w : Option String} (h : check c s r.1 r.1 r.2 = .ok (o, s', w)) :
    s'.inner = none := by
  cases w with
  | some w => exact round_inner (s := { s with stopped := true }) hr hs (check_some h)
  | none => exact round_inner hr hs (check_none h)

/-- The coins the frame takes outside the inner proof leave it outside. -/
theorem answer_inner {c : Cfg} {s s' : Masker} {cs : Array F128} (hs : s.inner = none) (h : s.answer c cs = .ok s') :
    s'.inner = none := by
  unfold Masker.answer at h
  simp only [bind, Except.bind, pure, Except.pure, hs] at h
  repeat' split at h
  all_goals (cases h <;> rfl)

/-- A request of a rep's scheduled rounds: a framed scheduled round in the device's place, or anything but a round. -/
def SchedReq : _root_.Flock.Firewall.Frame.Req → Prop
  | .round n v (.sched r) => v = r ∧ Framed (r, n)
  | .round _ _ _ => False
  | _ => True

theorem step_inner (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (q : _root_.Flock.Firewall.Frame.Req)
    (hq : SchedReq q) (hs : st.m.inner = none) : (_root_.Flock.Firewall.Frame.step c st q).2.m.inner = none := by
  cases q with
  | round n v p =>
    cases p with
    | sched r =>
      obtain ⟨rfl, hr⟩ := hq
      simp only [_root_.Flock.Firewall.Frame.step]
      split
      · exact hs
      · split
        · rename_i heq
          exact check_inner hr hs heq
        · split
          · exact hs
          · exact hs
    | fail => exact hq.elim
    | own => exact hq.elim
  | coins cs =>
    cases cs with
    | error e => exact hs
    | ok cs =>
      simp only [_root_.Flock.Firewall.Frame.step]
      split
      · rename_i heq
        exact answer_inner hs heq
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

/-- **The frame over a rep's scheduled rounds stays outside the inner proof.** -/
theorem run_inner (c : Cfg) : ∀ (st : _root_.Flock.Firewall.Frame.State) (qs : List _root_.Flock.Firewall.Frame.Req),
    (∀ q ∈ qs, SchedReq q) → st.m.inner = none → (_root_.Flock.Firewall.Frame.run c st qs).1.m.inner = none
  | st, [], _, hs => hs
  | st, q :: qs, hq, hs => by
    simp only [_root_.Flock.Firewall.Frame.run]
    have h1 := step_inner c st q (hq q List.mem_cons_self) hs
    split
    · exact h1
    · exact run_inner c _ qs (fun q' h' => hq q' (List.mem_cons_of_mem _ h')) h1

end Sched

namespace Loop

/-- **`Frame.run` over appended requests**: over the first, then, where they did not end the rep, over the rest. -/
theorem run_append (c : Cfg) : ∀ (st : _root_.Flock.Firewall.Frame.State) (l₁ l₂ : List _root_.Flock.Firewall.Frame.Req),
    _root_.Flock.Firewall.Frame.run c st (l₁ ++ l₂) =
      if (_root_.Flock.Firewall.Frame.run c st l₁).2 then _root_.Flock.Firewall.Frame.run c st l₁
      else _root_.Flock.Firewall.Frame.run c (_root_.Flock.Firewall.Frame.run c st l₁).1 l₂
  | st, [], l₂ => rfl
  | st, q :: l₁, l₂ => by
    simp only [List.cons_append, _root_.Flock.Firewall.Frame.run]
    split
    · simp
    · exact run_append c _ l₁ l₂

/-- **A rep's requests up to its scheduled round `j`**, the scheduled rounds themselves in the device's place: the
earlier rep's coins `prior`, then each round before `j` with its coins `cr`, then round `j` (`reqs_take`). -/
def schedReqs (prior : Array F128) (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) (j : ℕ) :
    List _root_.Flock.Firewall.Frame.Req :=
  (if prior.isEmpty then [] else [.prior prior.size (_root_.Flock.Firewall.Frame.wordBytes prior)]) ++
    (List.range j).flatMap (fun k =>
      [.round (sched.getD k (.empty, 0)).2 (sched.getD k (.empty, 0)).1 (.sched (sched.getD k (.empty, 0)).1),
        .coins (.ok (cr.getD k #[]))]) ++
    [.round (sched.getD j (.empty, 0)).2 (sched.getD j (.empty, 0)).1 (.sched (sched.getD j (.empty, 0)).1)]

theorem schedReqs_succ (prior : Array F128) (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) (j : ℕ) :
    schedReqs prior sched cr (j + 1) = schedReqs prior sched cr j ++
      [.coins (.ok (cr.getD j #[])),
        .round (sched.getD (j + 1) (.empty, 0)).2 (sched.getD (j + 1) (.empty, 0)).1
          (.sched (sched.getD (j + 1) (.empty, 0)).1)] := by
  unfold schedReqs
  rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
  simp only [List.append_assoc, List.cons_append, List.nil_append]

theorem schedReqs_congr (prior : Array F128) {sched sched' : Array (ByteArray × ℕ)} {cr cr' : Array (Array F128)}
    {j : ℕ} (hs : ∀ k, k ≤ j → sched.getD k (.empty, 0) = sched'.getD k (.empty, 0))
    (hc : ∀ k, k < j → cr.getD k #[] = cr'.getD k #[]) :
    schedReqs prior sched cr j = schedReqs prior sched' cr' j := by
  unfold schedReqs
  rw [flatMap_range_congr fun k hk => by rw [hs k (Nat.le_of_lt hk), hc k hk], hs j (Nat.le_refl j)]

/-- **A rep's frame state after its scheduled round `j`**: `Frame.run` over its requests up to it (`schedReqs`), where
the schedule on the rep's coins before the last has it. -/
theorem repState_at (c : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (j : ℕ)
    {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hj : j < sched.size) :
    repState c t rep dom xs hi cr0 cr ns (j + 1) =
      let r := _root_.Flock.Firewall.Frame.run c {}
        (schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr j)
      if r.2 then {} else r.1.m := by
  unfold repState
  rw [hal]
  simp only [bind, Except.bind, hs, pure, Except.pure]
  rw [show ∀ p : ℕ, p + 2 * (j + 1) - 1 = p + 2 * j + 1 from fun p => by omega, reqs_take _ _ _ _ _ hj]
  rfl

/-- **A rep's frame state one round on**: where its rounds' coins `cr` cover its round `j` and the schedule on them
has round `j + 1`, its state after round `j + 1` on one more round's coins is its state after round `j`, then the
coins of round `j`, then round `j + 1`, the scheduled round in the device's place. -/
theorem repState_succ (c : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (a : Array F128)
    (j : ℕ) {alg sched sched' : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hs' : schedSoFar t rep dom xs hi cr = .ok sched')
    (hcr : cr.size = j + 1) (hj : j + 1 < sched'.size) :
    repState c t rep dom xs hi cr0 (cr.push a) ns (j + 2) =
      let r := _root_.Flock.Firewall.Frame.run c {}
        (schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr j)
      if r.2 then {} else
        let r' := _root_.Flock.Firewall.Frame.run c r.1
          [.coins (.ok (cr.getD j #[])),
            .round (sched'.getD (j + 1) (.empty, 0)).2 (sched'.getD (j + 1) (.empty, 0)).1
              (.sched (sched'.getD (j + 1) (.empty, 0)).1)]
        if r'.2 then {} else r'.1.m := by
  have hj0 := schedSoFar_pop_lt t rep dom xs hi hs hs' hcr hj
  have hs'' := hs'
  rw [Array.eq_push_pop_back!_of_size_ne_zero (show cr.size ≠ 0 by omega)] at hs''
  obtain ⟨u, rfl⟩ := schedSoFar_snoc t rep dom xs hi cr.pop cr.back! hs hs''
  rw [repState_at c t rep dom xs hi cr0 (cr.push a) ns (j + 1) hal (by rw [Array.pop_push]; exact hs') hj]
  have h1 : schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) (sched ++ u) (cr.push a)
      (j + 1) = schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) (sched ++ u) cr
      (j + 1) := by
    refine schedReqs_congr _ (fun _ _ => rfl) (fun k hk => ?_)
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push, show k ≠ cr.size by omega, ↓reduceIte]
  have h2 : schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) (sched ++ u) cr j =
      schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr j := by
    refine schedReqs_congr _ (fun k hk => ?_) (fun _ _ => rfl)
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_append_left (show k < sched.size by omega)]
  dsimp only
  rw [h1, schedReqs_succ, h2, run_append]
  generalize _root_.Flock.Firewall.Frame.run c {}
    (schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr j) = R
  obtain ⟨S, b⟩ := R
  cases b <;> rfl

/-- The coins the frame takes leave it owing none. -/
theorem answer_owed {c : Cfg} {s s' : Masker} {cs : Array F128} (h : s.answer c cs = .ok s') : s'.owed = none := by
  unfold Masker.answer at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals (cases h <;> rfl)

/-- **The frame's coins, then a round in its scheduled round's own place**: where the frame takes the coins `cs` and
checks the round `r` against itself, its state after both is the check's. -/
theorem run_coins_round (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (cs : Array F128) (n : ℕ) (r : ByteArray)
    {s1 s' : Masker} {out : ByteArray} {w : Option String} (ha : st.m.answer c cs = .ok s1)
    (hc : check c s1 r r n = .ok (out, s', w)) :
    _root_.Flock.Firewall.Frame.run c st [.coins (.ok cs), .round n r (.sched r)] = ({ m := s', begun := true }, false) := by
  have ho := answer_owed ha
  simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ha, ho, hc,
    _root_.Flock.Firewall.Frame.Reply.ends]
  rfl

/-- **The loop's frame state one round on, at a scheduled round**: where the frame takes round `j`'s coins at the
rep's state after it, at the state `m`, and checks the schedule's round `j + 1` against itself there, the rep's state
after round `j + 1` on one more round's coins is that check's. -/
theorem repState_step (c : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (a : Array F128)
    (j : ℕ) {alg sched sched' : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hs' : schedSoFar t rep dom xs hi cr = .ok sched')
    (hcr : cr.size = j + 1) (hj : j + 1 < sched'.size) {m s' : Masker} {out : ByteArray} {w : Option String}
    (ha : (repState c t rep dom xs hi cr0 cr ns (j + 1)).answer c (cr.getD j #[]) = .ok m)
    (hc : check c m (sched'.getD (j + 1) (.empty, 0)).1 (sched'.getD (j + 1) (.empty, 0)).1
      (sched'.getD (j + 1) (.empty, 0)).2 = .ok (out, s', w)) :
    repState c t rep dom xs hi cr0 (cr.push a) ns (j + 2) = s' := by
  rw [repState_succ c t rep dom xs hi cr0 cr ns a j hal hs hs' hcr hj]
  rw [repState_at c t rep dom xs hi cr0 cr ns j hal hs (schedSoFar_pop_lt t rep dom xs hi hs hs' hcr hj)] at ha
  dsimp only at ha ⊢
  generalize _root_.Flock.Firewall.Frame.run c {}
    (schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr j) = R at ha ⊢
  obtain ⟨S, b⟩ := R
  cases b with
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte] at ha ⊢
    rw [run_coins_round c S _ _ _ ha hc]
    rfl
  | true =>
    have e : (({} : Masker).answer c (cr.getD j #[])) = .error "coins for no round" := rfl
    simp only [↓reduceIte] at ha
    rw [e] at ha
    cases ha

/-- **The loop's frame state one scheduled round on is the deployed check's** but for `stopped`: where the frame takes
round `j`'s coins at the rep's placeholder state after it, at the state `m` outside the inner proof, and the deployed
frame's check at an alike configuration `c` of a device round `v` against a round of one shape with the placeholder's
round `j + 1` (`RoundAt`, the real table's) releases a round, the rep's placeholder state after round `j + 1` is that
check's, but for `stopped`. `hI`: the placeholder round's frames enter no inner proof. -/
theorem repState_check_step {c c' : Cfg} (hc : CfgAlike c c') (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ)
    (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ)
    (a : Array F128) (j : ℕ) {alg sched sched' : Array (ByteArray × ℕ)} {r : ByteArray × ℕ}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hs' : schedSoFar t rep dom xs hi cr = .ok sched')
    (hcr : cr.size = j + 1) (hj : j + 1 < sched'.size) (hR : Sched.RoundAt r (sched'.getD (j + 1) (.empty, 0)))
    {m : Masker} (ha : (repState c' t rep dom xs hi cr0 cr ns (j + 1)).answer c' (cr.getD j #[]) = .ok m)
    (hin : m.inner ≠ some INNER_ROUNDS)
    (hI : ∀ (fs : List Frame) (k : UInt8), Sched.MidAll fs → (sched'.getD (j + 1) (.empty, 0)).1 =
        ByteArray.mk (encs fs ++ (Frame.sq k r.2).enc).toArray →
      ∀ q, fs.foldlM (stepAt c') (m, ByteArray.empty, #[], 0) = .ok q → q.1.inner = none)
    {st : Bool} {v out : ByteArray} {s : Masker} {why : Option String}
    (h : check c { m with stopped := st } v r.1 r.2 = .ok (out, s, why)) :
    repState c' t rep dom xs hi cr0 (cr.push a) ns (j + 2) =
      { s with stopped := (repState c' t rep dom xs hi cr0 (cr.push a) ns (j + 2)).stopped } := by
  obtain ⟨fs, fs', k, h1, h1', hn2, hA, hM, hM', hk, h0, hn⟩ := Sched.roundAt_frames hR
  rw [h1] at h
  obtain ⟨out', s', w, hc', hu⟩ :=
    Sched.check_alike_self hc m st fs fs' k r.2 hin (fun f hf => (hM f hf).1) (fun f hf => (hM' f hf).1) hk h0 hn hA
      (hI fs' k hM' h1') h
  have hstep := repState_step c' t rep dom xs hi cr0 cr ns a j hal hs hs' hcr hj ha
    (by rw [h1', hn2]; exact hc')
  rw [hstep]
  exact eq_unstop hu

theorem words_size (b : ByteArray) (o n : ℕ) : (_root_.Flock.Firewall.Frame.words b o n).size = n := by
  unfold _root_.Flock.Firewall.Frame.words
  rw [Array.size_map, Array.size_range]

theorem ofBytes_h_size (l : _root_.Flock.Zk.PadLayout) (b : ByteArray) : (Draws.ofBytes l b).h.size = l.len := by
  unfold Draws.ofBytes
  dsimp only
  rw [Array.size_set!, Array.size_set!, words_size]

section OfDraws
variable (l : _root_.Flock.Zk.PadLayout) (d : Draws) (m : Option _root_.Flock.Firewall.Frame.MaskSlot)
theorem ofDraws_lay : (Cfg.ofDraws l d m).lay = l := rfl
theorem ofDraws_order : (Cfg.ofDraws l d m).order = _root_.Flock.Firewall.Frame.absorbed l := rfl
theorem ofDraws_nPiop :
    (Cfg.ofDraws l d m).nPiop = (_root_.Flock.Firewall.Frame.absorbed l).size - _root_.Flock.Firewall.Frame.nLig l := rfl
theorem ofDraws_lig : (Cfg.ofDraws l d m).lig = decide (l.k0 > 0) := rfl
theorem ofDraws_mask : (Cfg.ofDraws l d m).mask = m := rfl
theorem ofDraws_h : (Cfg.ofDraws l d m).h = d.h := rfl
end OfDraws

/-- **A rep's configurations at two draws** of one layout and mask slot are alike. -/
theorem cfgAlike_ofBytes (l : _root_.Flock.Zk.PadLayout) (b b' : ByteArray)
    (m : Option _root_.Flock.Firewall.Frame.MaskSlot) :
    CfgAlike (Cfg.ofDraws l (Draws.ofBytes l b) m) (Cfg.ofDraws l (Draws.ofBytes l b') m) where
  lay := (ofDraws_lay l _ m).trans (ofDraws_lay l _ m).symm
  order := (ofDraws_order l _ m).trans (ofDraws_order l _ m).symm
  nPiop := (ofDraws_nPiop l _ m).trans (ofDraws_nPiop l _ m).symm
  lig := (ofDraws_lig l _ m).trans (ofDraws_lig l _ m).symm
  mask := (ofDraws_mask l _ m).trans (ofDraws_mask l _ m).symm
  size := by rw [ofDraws_h, ofDraws_h, ofBytes_h_size, ofBytes_h_size]

theorem map_range_congr {β : Type} {f g : ℕ → β} {n : ℕ} (h : ∀ j, j < n → f j = g j) :
    (Array.range n).map f = (Array.range n).map g := by
  apply Array.ext (by simp)
  intro i h1 _
  simp only [Array.getElem_map, Array.getElem_range]
  exact h i (by simpa using h1)

/-- **Before a later rep stream, the stream before it**: past stream `k`'s first round, the rounds of the stream
before the located one and the located round's offset fit in the coins so far. -/
theorem locate_prev : ∀ (rs : List ℕ) (k i : ℕ), k < (ZKMal.CoinLayer.locate rs k i).1 →
    rs.getD ((ZKMal.CoinLayer.locate rs k i).1 - k - 1) 0 + (ZKMal.CoinLayer.locate rs k i).2 ≤ i
  | [], _, _, h => absurd h (Nat.lt_irrefl _)
  | c :: cs, k, i, h => by
    by_cases hc : i < c
    · rw [locate_cons_lt hc] at h
      exact absurd h (Nat.lt_irrefl _)
    · rw [locate_cons_ge hc] at h ⊢
      have hge := locate_fst_ge cs (k + 1) (i - c)
      rcases Nat.lt_or_ge (k + 1) (ZKMal.CoinLayer.locate cs (k + 1) (i - c)).1 with h1 | h1
      · have ih := locate_prev cs (k + 1) (i - c) h1
        rw [show (ZKMal.CoinLayer.locate cs (k + 1) (i - c)).1 - k - 1 =
          ((ZKMal.CoinLayer.locate cs (k + 1) (i - c)).1 - (k + 1) - 1) + 1 by omega]
        simp only [List.getD_cons_succ]
        omega
      · rw [show (ZKMal.CoinLayer.locate cs (k + 1) (i - c)).1 - k - 1 = 0 by omega]
        simp only [List.getD_cons_zero]
        have := locate_snd_le cs (k + 1) (i - c)
        omega

/-- **A rep round past its stream's first follows the stream's round before it.** -/
theorem locate_pred : ∀ (rs : List ℕ) (k i j : ℕ), (ZKMal.CoinLayer.locate rs k (i + 1)).2 = j + 1 →
    ZKMal.CoinLayer.locate rs k i = ((ZKMal.CoinLayer.locate rs k (i + 1)).1, j)
  | [], _, _, _, h => by
    simp only [ZKMal.CoinLayer.locate] at h ⊢
    simp only [Prod.mk.injEq, true_and]
    omega
  | c :: cs, k, i, j, h => by
    by_cases h1 : i + 1 < c
    · rw [locate_cons_lt h1] at h
      rw [locate_cons_lt h1, locate_cons_lt (Nat.lt_of_succ_lt h1)]
      simp only at h
      simp only [Prod.mk.injEq, true_and]
      omega
    · by_cases h0 : i < c
      · rw [locate_cons_ge h1] at h
        have := locate_snd_le cs (k + 1) (i + 1 - c)
        omega
      · rw [locate_cons_ge h1] at h ⊢
        rw [locate_cons_ge h0]
        have e : i + 1 - c = i - c + 1 := by omega
        rw [e] at h ⊢
        exact locate_pred cs (k + 1) (i - c) j h

/-- **A rep stream's first round follows another stream's round.** -/
theorem locate_first (rs : List ℕ) (k i : ℕ)
    (h : (ZKMal.CoinLayer.locate rs k (i + 1)).1 = (ZKMal.CoinLayer.locate rs k i).1) :
    (ZKMal.CoinLayer.locate rs k (i + 1)).2 ≠ 0 := by
  rw [locate_succ rs k i h]
  exact Nat.succ_ne_zero _

/-- Rep 0's frame state reads no earlier rep's coins. -/
theorem repState_cr0 {c : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr : Array (Array F128)} {ns : ℕ → ℕ} {i : ℕ} (hr : rep = 0)
    (cr0 cr0' : Array (Array F128)) :
    repState c t rep dom xs hi cr0 cr ns i = repState c t rep dom xs hi cr0' cr ns i := by
  subst hr
  unfold repState
  simp only [↓reduceIte]

theorem wordsAt_append (x : Msg) (cs ds : List FlockDeploy.Coin) {p : ℕ} (hp : p < cs.length) :
    wordsAt x (cs ++ ds) p = wordsAt x cs p := by
  unfold wordsAt
  rw [List.getElem?_append_left hp]

theorem crAt_push (f : ℕ → Array F128) (s n : ℕ) :
    (Array.range (n + 1)).map (fun j => f (s + j + 1)) = ((Array.range n).map fun j => f (s + j + 1)).push
      (f (s + n + 1)) := by
  rw [Array.range_succ, Array.map_append, Array.push_eq_append]
  congr 1
  simp

/-- **The loop's frame state one coin on**: after the decoded coins `cs ++ [d]`, the state of the rep that released
the round after `cs`, after one more of its rounds, on its rounds' coins and one more. -/
theorem progOf_snoc (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : ℕ} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1) :
    progOf x (cs ++ [d]) =
      match placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) with
      | .ok (c, t, dom, xs, hi) =>
        repState c t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
          ((Array.range ((roundsOf spec sts).getD ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 - 1) 0)).map
            fun j => wordsAt x cs (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 -
              (roundsOf spec sts).getD ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 - 1) 0 + j + 1))
          (((Array.range (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2).map
            fun j => wordsAt x cs (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1)).push
            (wordsAt x (cs ++ [d]) (q + 1)))
          (fun j => loopNr x (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1))
          ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + 1)
      | .error _ => {} := by
  have hle := locate_snd_le (roundsOf spec sts) 0 q
  have hl : (cs ++ [d]).length = q + 2 := by rw [List.length_append, List.length_singleton, hq]
  unfold progOf
  rw [hx, hl]
  simp only
  rw [wordsAt_append x cs [d] (by omega), crAt_push]
  rw [show q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 +
    (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + 1 = q + 1 by omega]
  have hw : ∀ p, p ≤ q → wordsAt x (cs ++ [d]) p = wordsAt x cs p := fun p hp => wordsAt_append x cs [d] (by omega)
  congr 1
  funext c t dom xs hi
  rw [map_range_congr (n := (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2) fun j hj => hw _ (by omega)]
  by_cases hk0 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = 0
  · exact repState_cr0 (by rw [hk0]) _ _
  · have hp := locate_prev (roundsOf spec sts) 0 q (by omega)
    rw [Nat.sub_zero] at hp
    rw [map_range_congr fun j hj => hw _ (by omega)]

/-- **The loop's frame state after the decoded coins `cs`**, where `x` sets up a session, the round after them is a
rep stream's past its first and the placeholder is `(c, t, dom, xs, hi)`: that rep's after its released rounds. -/
theorem progOf_at (x : Msg) (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1) {j : ℕ}
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1) {c : Cfg}
    {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c, t, dom, xs, hi)) :
    progOf x cs =
      repState c t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
        ((Array.range ((roundsOf spec sts).getD ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 - 1) 0)).map
          fun j => wordsAt x cs (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 -
            (roundsOf spec sts).getD ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 - 1) 0 + j + 1))
        ((Array.range (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2).map
          fun j => wordsAt x cs (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1))
        (fun j => loopNr x (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1))
        (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 := by
  have hle := locate_snd_le (roundsOf spec sts) 0 q
  obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := ⟨q - 1, by omega⟩
  have hp := locate_pred (roundsOf spec sts) 0 q' j hj
  have hl : cs.length = q' + 2 := hq
  rw [hj]
  unfold progOf
  rw [hx, hl]
  simp only [hp]
  rw [hph]
  dsimp only
  rw [show q' + 1 - (j + 1) = q' - j by omega]

theorem forall₂_getD {α β : Type} {R : α → β → Prop} {d : α} {d' : β} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ R l l' → ∀ {i : ℕ}, i < l.length → R (l.getD i d) (l'.getD i d')
  | _, _, .cons hab _, 0, _ => by simpa only [List.getD_cons_zero] using hab
  | _, _, .cons _ hl, i + 1, hi => by
    simp only [List.getD_cons_succ]
    exact forall₂_getD hl (by simp only [List.length_cons] at hi; omega)

theorem roundsAt_getD {rs rs' : Array (ByteArray × ℕ)} (h : Sched.RoundsAt rs rs') {i : ℕ} (hi : i < rs.size) :
    Sched.RoundAt (rs.getD i (.empty, 0)) (rs'.getD i (.empty, 0)) := by
  have e : ∀ a : Array (ByteArray × ℕ), a.getD i (.empty, 0) = a.toList.getD i (.empty, 0) := fun a => by
    rw [Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD, Array.getElem?_toList]
  rw [e, e]
  exact forall₂_getD h (by rw [Array.length_toList]; exact hi)

theorem roundsAt_size {rs rs' : Array (ByteArray × ℕ)} (h : Sched.RoundsAt rs rs') : rs.size = rs'.size := by
  have := List.Forall₂.length_eq h
  rwa [Array.length_toList, Array.length_toList] at this

/-- Every round of a schedule of one shape with another is framed. -/
theorem roundsAt_framed {rs rs' : Array (ByteArray × ℕ)} (h : Sched.RoundsAt rs rs') {i : ℕ} (hi : i < rs'.size) :
    Sched.Framed (rs'.getD i (.empty, 0)) := by
  obtain ⟨_, fs', k, _, h1', hn2, _, _, hM', hk, h0, hn⟩ :=
    Sched.roundAt_frames (roundsAt_getD h (by rw [roundsAt_size h]; exact hi))
  exact ⟨fs', k, by rw [h1', hn2], hM', by rw [hn2]; exact hk, by rw [hn2]; exact h0, by rw [hn2]; exact hn⟩

/-- The schedule on the coins but the last is a prefix of the schedule on them all. -/
theorem schedSoFar_pop_getD {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr : Array (Array F128)} {sched sched' : Array (ByteArray × ℕ)}
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hs' : schedSoFar t rep dom xs hi cr = .ok sched')
    (hcr : cr.size ≠ 0) {k : ℕ} (hk : k < sched.size) :
    sched.getD k (.empty, 0) = sched'.getD k (.empty, 0) ∧ k < sched'.size := by
  rw [Array.eq_push_pop_back!_of_size_ne_zero hcr] at hs'
  obtain ⟨u, rfl⟩ := schedSoFar_snoc t rep dom xs hi cr.pop cr.back! hs hs'
  refine ⟨?_, by rw [Array.size_append]; omega⟩
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk]

/-- **A rep's frame state at a scheduled round is outside the inner proof**: where the schedule so far holds round `j`
and its rounds up to it are framed, the rep's state after round `j` is (`Sched.run_inner`). -/
theorem repState_inner (c : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (j : ℕ)
    {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hj : j < sched.size)
    (hF : ∀ k, k ≤ j → Sched.Framed (sched.getD k (.empty, 0))) :
    (repState c t rep dom xs hi cr0 cr ns (j + 1)).inner = none := by
  rw [repState_at c t rep dom xs hi cr0 cr ns j hal hs hj]
  dsimp only
  generalize (if rep = 0 then (#[] : Array F128) else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) = prior
  by_cases hr : (_root_.Flock.Firewall.Frame.run c {} (schedReqs prior sched cr j)).2 = true
  · rw [if_pos hr]
  · rw [if_neg hr]
    refine Sched.run_inner c ({} : _root_.Flock.Firewall.Frame.State) (schedReqs prior sched cr j)
      (fun q hq => ?_) rfl
    unfold schedReqs at hq
    rcases List.mem_append.1 hq with hq | hq
    · rcases List.mem_append.1 hq with hq | hq
      · split at hq
        · cases hq
        · rw [List.mem_singleton] at hq; subst hq; trivial
      · obtain ⟨k, hk, hq⟩ := List.mem_flatMap.1 hq
        rw [List.mem_range] at hk
        rcases List.mem_cons.1 hq with rfl | hq
        · exact ⟨rfl, hF k (by omega)⟩
        · rw [List.mem_singleton] at hq; subst hq; trivial
    · rw [List.mem_singleton] at hq; subst hq; exact ⟨rfl, hF j (Nat.le_refl j)⟩

/-- **The loop's frame state one scheduled round on, within a rep stream, is the deployed check's** but for `stopped`:
where the round after the decoded coins `cs` is round `j + 1` of its rep stream, the schedule so far at its own table
(`rsched`) and the placeholder's (`sched'`, on the same coins) are of one shape (`RoundsAt`) and hold that round, the frame takes the last coins of `cs` at the state `m`, and the deployed frame's
check at an alike configuration of a device round against the rep's round `j + 1` releases a round, the loop's state
after `cs ++ [d]` is that check's, but for `stopped`. The state `m` is outside the inner proof, which no scheduled round
starts (`repState_inner`). -/
theorem progOf_check_step (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : ℕ} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1)
    {c c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi))
    (hc : CfgAlike c c') {j : ℕ} (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1)
    {rsched sched' : Array (ByteArray × ℕ)} (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok sched')
    (hlt : j + 1 < rsched.size) {m : Masker} (ha : (progOf x cs).answer c (wordsAt x cs q) = .ok m)
    {st : Bool} {v out : ByteArray} {s : Masker} {why : Option String}
    (h : check c { m with stopped := st } v (rsched.getD (j + 1) (.empty, 0)).1 (rsched.getD (j + 1) (.empty, 0)).2 =
      .ok (out, s, why)) :
    progOf x (cs ++ [d]) = { s with stopped := (progOf x (cs ++ [d])).stopped } := by
  have hle := locate_snd_le (roundsOf spec sts) 0 q
  rw [progOf_at x cs hx hq hj hph, hj] at ha
  rw [progOf_snoc x cs d hx hq, hph]
  dsimp only
  rw [hj]
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds hs'
  obtain ⟨sched, hs⟩ := schedSoFar_pop_ok hs'
  have hg : ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)).getD j #[] = wordsAt x cs q := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by simp), Option.getD_some, Array.getElem_map,
      Array.getElem_range]
    congr 1
    omega
  rw [answer_congr _ _ hc.lay hc.order, ← hg] at ha
  have hcr : ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)).size = j + 1 := by simp
  have hj' : j + 1 < sched'.size := by rw [← roundsAt_size hrs]; exact hlt
  have hj0 := schedSoFar_pop_lt t _ dom xs hi hs hs' hcr hj'
  have hnone : m.inner = none := Sched.answer_inner (repState_inner c' t _ dom xs hi _ _ _ j hal hs hj0
    fun k hk => by
      obtain ⟨e, hk'⟩ := schedSoFar_pop_getD hs hs' (by rw [hcr]; omega) (show k < sched.size by omega)
      rw [e]
      exact roundsAt_framed hrs hk') ha
  exact repState_check_step hc t _ dom xs hi _ _ _ _ j hal hs hs' hcr hj' (roundsAt_getD hrs hlt) ha
    (by rw [hnone]; nofun) (fun _ _ hM _ _ hq => Sched.foldlM_midAll_inner hnone hM hq) h

/-- The coins the loop's frames take after the decoded coins `cs`, `q + 1` of them, are opening `q`'s words. -/
theorem readCoins_eq (x : Msg) (cs : List FlockDeploy.Coin) {q : ℕ} (hq : cs.length = q + 1) :
    readCoins x cs = wordsAt x cs q := by
  unfold readCoins wordsAt
  rw [List.getLast?_eq_getElem?, hq, Nat.add_sub_cancel]
  cases cs[q]? <;> rfl

/-- **The scheduled round in the device's place** after the decoded coins `cs`, where `x` sets up a session, the located
rep's own table and salts are `(t, hi)` and its schedule so far at them is `sched`, which holds the rep's next round:
that round. -/
theorem schedOf_eq (real : ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) (x : Msg)
    (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1)
    {t : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    (hreal : real (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = .ok (t, hi)) {sched : Array (ByteArray × ℕ)}
    (hs : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)
      (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!)
        ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
          (wordsAt x cs 0))) hi
      ((Array.range (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2).map
        fun j => wordsAt x cs (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1)) = .ok sched)
    (hlt : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 < sched.size) :
    schedOf real x cs = some (sched.getD (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 (.empty, 0)).1 := by
  unfold schedOf
  rw [hx, hq]
  simp only [hreal, hs, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.map_some,
    Option.getD_some]

/-- Where the located rep's own table is refused, the loop has no round in the device's place: the empty round. -/
theorem schedOf_real_error (real : ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) (x : Msg)
    (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1) {e : String}
    (hreal : real (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = .error e) :
    schedOf real x cs = some .empty := by
  unfold schedOf
  rw [hx, hq]
  simp only [hreal]

/-- Where the located rep's schedule so far is refused, the loop has no round in the device's place: the empty
round. -/
theorem schedOf_sched_error (real : ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)) (x : Msg)
    (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1)
    {t : _root_.Flock.Firewall.Finish.Tab} {hi : Array ByteArray}
    (hreal : real (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = .ok (t, hi)) {e : String}
    (hs : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)
      (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!)
        ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
          (wordsAt x cs 0))) hi
      ((Array.range (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2).map
        fun j => wordsAt x cs (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1)) = .error e) :
    schedOf real x cs = some .empty := by
  unfold schedOf
  rw [hx, hq]
  simp only [hreal, hs]

/-- **Rep stream `k`'s placeholder**, where it is one: the frame's configuration at zero draws at `Drive.start`'s layout
and mask slot, the stream's domain and the extra claims at the link points. -/
theorem placeholder_ok {spec : SessionSpec} {sts : Array Stmt} {k : ℕ} {link : Array F128} {c' : Cfg}
    {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)} {hi : Array ByteArray}
    (h : placeholder spec sts k link = .ok (c', t, dom, xs, hi)) :
    ∃ lay mask, _root_.Flock.Firewall.Drive.start sts[k / 2]! = .ok (lay, mask) ∧
      c' = Cfg.ofDraws lay (Draws.ofBytes lay (zeros (_root_.Flock.Firewall.Frame.drawBytes lay))) (some mask) ∧
      dom = (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8 ∧
      xs = _root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]! link) := by
  have bok : ∀ {α β : Type} {x : V α} {f : α → V β} {b : β}, x >>= f = .ok b → ∃ a, x = .ok a ∧ f a = .ok b := by
    intro α β x f b hb
    cases x with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  unfold placeholder at h
  obtain ⟨⟨lay, mask⟩, hs, h⟩ := bok h
  obtain ⟨⟨sch, tPad⟩, -, h⟩ := bok h
  obtain ⟨l0, -, h⟩ := bok h
  obtain ⟨t0, -, h⟩ := bok h
  cases h
  exact ⟨lay, mask, hs, rfl, rfl, rfl⟩

/-- The frame refuses the empty round: it has no squeeze. -/
theorem roundFree_empty (c : Cfg) (s : Masker) (n : ℕ) : ∃ e, roundFree c s ByteArray.empty n = .error e := by
  unfold roundFree
  by_cases hi : (s.inner == some INNER_ROUNDS) = true
  · simp only [hi, ↓reduceIte]
    exact ⟨_, rfl⟩
  · simp only [hi, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size,
      ByteArray.size_empty]
    exact ⟨_, rfl⟩

/-- **The frame's check refuses the empty scheduled round**, whatever the device round and the rep's state. -/
theorem check_empty (c : Cfg) (s : Masker) (v : ByteArray) (n : ℕ) (q : ByteArray × Masker × Option String) :
    check c s v ByteArray.empty n ≠ .ok q := by
  obtain ⟨e, he⟩ := roundFree_empty c s n
  obtain ⟨e', he'⟩ := roundFree_empty c { s with stopped := true } n
  intro h
  unfold check passes _root_.Flock.Firewall.Frame.standIn _root_.Flock.Firewall.Frame.round at h
  cases hs : s.stopped <;> simp only [hs, he, he', Bool.false_eq_true, ↓reduceIte, bind, Except.bind] at h <;>
    cases h

/-- A state that owes no round's coins refuses any. -/
theorem answer_owed_none {c : Cfg} {s : Masker} (cs : Array F128) (hs : s.owed = none) {m : Masker} :
    s.answer c cs ≠ .ok m := by
  intro h
  unfold _root_.Flock.Firewall.Frame.Masker.answer at h
  rw [hs] at h
  cases h

/-- Before a rep's first round its state owes no coins. -/
theorem repState_zero_owed (c : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) :
    (repState c t rep dom xs hi cr0 cr ns 0).owed = none := by
  unfold repState
  split
  · rfl
  · dsimp only
    split_ifs <;> rfl

/-- **The loop's frame state owes coins only where its rep's placeholder is one**: after the decoded coins `cs`, `p + 2`
of them, where `x` sets up a session, the placeholder of the rep that released the last round. -/
theorem progOf_placeholder (x : Msg) (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    {p : ℕ} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hl : cs.length = p + 2)
    (ho : (progOf x cs).owed ≠ none) :
    ∃ c' t dom xs hi, placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 p).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi) := by
  cases hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 p).1 (wordsAt x cs 0) with
  | error e =>
    refine absurd ?_ ho
    unfold progOf
    rw [hx, hl]
    simp only [hph]
  | ok r =>
    obtain ⟨c', t, dom, xs, hi⟩ := r
    exact ⟨c', t, dom, xs, hi, rfl⟩

/-- Where `x` sets up no session the loop's frame state is the frame's start. -/
theorem progOf_none (x : Msg) (cs : List FlockDeploy.Coin) (hx : FlockDeploy.setupOf true x = none) :
    progOf x cs = {} := by
  unfold progOf
  rw [hx]

/-- Where `x` sets up no session the loop refuses a rep's first round. -/
theorem freshOf_none (x : Msg) (cs : List FlockDeploy.Coin) (hx : FlockDeploy.setupOf true x = none) {m : Masker} :
    freshOf x cs ≠ .ok m := by
  unfold freshOf
  rw [hx]
  nofun

/-- The first rep round is the first of its stream. -/
theorem locate_zero : ∀ (rs : List ℕ) (k : ℕ), (ZKMal.CoinLayer.locate rs k 0).2 = 0
  | [], _ => rfl
  | c :: cs, k => by
    by_cases hc : 0 < c
    · rw [locate_cons_lt hc]
    · rw [locate_cons_ge hc, Nat.zero_sub]
      exact locate_zero cs (k + 1)

/-- **The loop at a rep's first round**: where `x` sets up a session and the frame takes the round after the decoded
coins `cs`, `q + 1` of them, at `m`, the first of rep stream `k`'s, the rep's placeholder and algebraic phase are ones;
rep 0 starts at the frame's start, and rep 1 at the frame's start after `prior`, rep 0's algebraic-phase coins. -/
theorem freshOf_ok (x : Msg) (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1) {m : Masker}
    (hm : freshOf x cs = .ok m) :
    ∃ c' t dom xs hi alg, placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi) ∧
      _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom = .ok alg ∧
      (((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2 = 0 ∧ m = {}) ∨
       ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2 ≠ 0 ∧ ∃ st',
        _root_.Flock.Firewall.Frame.step c' {}
          (.prior (((Array.range ((roundsOf spec sts).getD ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 - 1) 0)).map
              fun j => wordsAt x cs (q - (roundsOf spec sts).getD
                ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 - 1) 0 + j + 1)).extract 0 alg.size
              |>.foldl (· ++ ·) #[]).size
            (_root_.Flock.Firewall.Frame.wordBytes
              (((Array.range ((roundsOf spec sts).getD ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 - 1) 0)).map
                fun j => wordsAt x cs (q - (roundsOf spec sts).getD
                  ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 - 1) 0 + j + 1)).extract 0 alg.size
                |>.foldl (· ++ ·) #[]))) = (.ok, st') ∧ m = st'.m)) := by
  unfold freshOf at hm
  rw [hx, hq] at hm
  simp only at hm
  cases hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) with
  | error e => rw [hph] at hm; cases hm
  | ok r =>
  obtain ⟨c', t, dom, xs, hi⟩ := r
  rw [hph] at hm
  simp only at hm
  cases hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom with
  | error e => rw [hal] at hm; cases hm
  | ok alg =>
  rw [hal] at hm
  simp only at hm
  refine ⟨c', t, dom, xs, hi, alg, rfl, hal, ?_⟩
  by_cases h0 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2 = 0
  · rw [if_pos h0] at hm
    cases hm
    exact Or.inl ⟨h0, rfl⟩
  · rw [if_neg h0] at hm
    refine Or.inr ⟨h0, ?_⟩
    split at hm
    · cases hm
    · split at hm
      · rename_i st' heq
        cases hm
        exact ⟨st', heq, rfl⟩
      all_goals cases hm

/-- **The algebraic phase has a round.** -/
theorem rounds_pos {st : _root_.Flock.Setup} {k0 : ℕ} {cap dom : ByteArray} {alg : Array (ByteArray × ℕ)}
    (h : _root_.Flock.Firewall.Skeleton.rounds st k0 cap dom = .ok alg) : 0 < alg.size := by
  unfold _root_.Flock.Firewall.Skeleton.rounds at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h
       rw [Array.size_set!]
       exact (Array.getElem?_eq_some_iff.1 (by assumption)).1)
    | cases h

/-- **A rep's schedule on no coins holds its first round**: the algebraic phase's. -/
theorem schedSoFar_nil_pos {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {s : Array (ByteArray × ℕ)}
    (h : schedSoFar t rep dom xs hi #[] = .ok s) : 0 < s.size := by
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds h
  have hp := rounds_pos hal
  unfold schedSoFar at h
  rw [hal] at h
  simp only [bind, Except.bind, Array.size_empty, hp, ↓reduceIte, pure, Except.pure] at h
  cases h
  exact hp

/-- An earlier rep's coins leave the frame owing what it owed. -/
theorem step_prior_owed (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (k : ℕ) (b : ByteArray) :
    (_root_.Flock.Firewall.Frame.step c st (.prior k b)).2.m.owed = st.m.owed := by
  simp only [_root_.Flock.Firewall.Frame.step]
  split
  · rfl
  · split <;> rfl

/-- The frame refuses an empty `prior`: the algebraic phase's claim points need coins. -/
theorem step_prior_zero (c : Cfg) (b : ByteArray) :
    (_root_.Flock.Firewall.Frame.step c {} (.prior 0 b)).1 ≠ .ok := by
  have hne : ∀ l : _root_.Flock.Zk.PadLayout, ∃ e,
      _root_.Flock.Firewall.Frame.claimPoints l (_root_.Flock.Firewall.Frame.words b 0 0) = .error e := by
    intro l
    unfold _root_.Flock.Firewall.Frame.claimPoints
    simp only [words_size]
    split
    · exact ⟨_, rfl⟩
    · rename_i hc
      exfalso
      apply hc
      simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq]
      right
      omega
  obtain ⟨e, he⟩ := hne c.lay
  unfold _root_.Flock.Firewall.Frame.step
  simp only [Bool.false_eq_true, ↓reduceIte, he, ne_eq, reduceCtorEq, not_false_eq_true]

/-- **The frame's first round in its scheduled round's place**, from a state owing no coins. -/
theorem run_round (c : Cfg) (st : _root_.Flock.Firewall.Frame.State) (n : ℕ) (r : ByteArray) {s' : Masker}
    {out : ByteArray} {w : Option String} (ho : st.m.owed = none) (hc : check c st.m r r n = .ok (out, s', w)) :
    _root_.Flock.Firewall.Frame.run c st [.round n r (.sched r)] = ({ m := s', begun := true }, false) := by
  simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ho, hc,
    _root_.Flock.Firewall.Frame.Reply.ends]
  rfl

/-- **An earlier rep's coins, then the frame's first round** in its scheduled round's place. -/
theorem run_prior_round (c : Cfg) (k : ℕ) (b : ByteArray) {st' : _root_.Flock.Firewall.Frame.State}
    (hp : _root_.Flock.Firewall.Frame.step c {} (.prior k b) = (.ok, st')) (n : ℕ) (r : ByteArray) {s' : Masker}
    {out : ByteArray} {w : Option String} (hc : check c st'.m r r n = .ok (out, s', w)) :
    _root_.Flock.Firewall.Frame.run c {} [.prior k b, .round n r (.sched r)] = ({ m := s', begun := true }, false) := by
  have ho : st'.m.owed = none := by
    have := step_prior_owed c {} k b
    rw [hp] at this
    exact this
  rw [_root_.Flock.Firewall.Frame.run, hp]
  exact run_round c st' n r ho hc

theorem range_zero_map {β : Type} (f : ℕ → β) : (Array.range 0).map f = #[] := by simp

/-- **The loop's frame state after a rep's first round is the frame's check's**: where the round after the decoded
coins `cs` is the first of its rep stream, the frame starts it at `m` (`freshOf`), and the placeholder's schedule on no
coins holds it, the loop's state after `cs ++ [d]` is the placeholder frame's check of the placeholder's round 0 against
itself from `m`. -/
theorem progOf_first_step (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : ℕ} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1) (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = 0) {m : Masker}
    (hm : freshOf x cs = .ok m) {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi))
    {sched' : Array (ByteArray × ℕ)}
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi #[] = .ok sched')
    {out : ByteArray} {s' : Masker} {w : Option String}
    (hc : check c' m (sched'.getD 0 (.empty, 0)).1 (sched'.getD 0 (.empty, 0)).1 (sched'.getD 0 (.empty, 0)).2 =
      .ok (out, s', w)) :
    progOf x (cs ++ [d]) = s' := by
  obtain ⟨c'', t', dom', xs', hi', alg, hph', hal, hm'⟩ := freshOf_ok x cs hx hq hm
  rw [hph] at hph'
  cases hph'
  have hj' : 0 < sched'.size := schedSoFar_nil_pos hs'
  rw [progOf_snoc x cs d hx hq, hph]
  dsimp only
  rw [hj, range_zero_map]
  have hs0 : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
      ((#[] : Array (Array F128)).push (wordsAt x (cs ++ [d]) (q + 1))).pop = .ok sched' := by
    rw [Array.pop_push]
    exact hs'
  rw [repState_at c' t _ dom xs hi _ _ _ 0 hal hs0 hj']
  dsimp only
  simp only [schedReqs, List.range_zero, List.flatMap_nil, List.append_nil, Nat.sub_zero]
  rcases hm' with ⟨h0, rfl⟩ | ⟨h0, st', hp, rfl⟩
  · simp only [h0, ↓reduceIte, Array.isEmpty_empty, List.nil_append]
    rw [run_round c' {} _ _ rfl hc]
    rfl
  · simp only [h0, ↓reduceIte]
    split
    · rename_i he
      rw [Array.isEmpty_iff_size_eq_zero.1 he] at hp
      exact absurd (congrArg Prod.fst hp) (step_prior_zero c' _)
    · have hg : sched'.getD 0 (ByteArray.empty, 0) = sched'.getInternal 0 hj' :=
        (Array.getElem_eq_getD (h := hj') _).symm
      rw [hg] at hc
      rw [List.singleton_append, run_prior_round c' _ _ hp _ _ hc]
      rfl

/-- **The loop's frame state after a rep's first round is the deployed check's** but for `stopped`: as
`progOf_check_step`, at the rep's first round, which the frame starts at `m` (`freshOf`), outside the inner proof. -/
theorem progOf_check_first (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : ℕ} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1) (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = 0) {m : Masker}
    (hm : freshOf x cs = .ok m) {c c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi))
    (hc : CfgAlike c c') {rsched sched' : Array (ByteArray × ℕ)} (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi #[] = .ok sched')
    (hlt : 0 < rsched.size) {st : Bool} {v out : ByteArray} {s : Masker} {why : Option String}
    (h : check c { m with stopped := st } v (rsched.getD 0 (.empty, 0)).1 (rsched.getD 0 (.empty, 0)).2 =
      .ok (out, s, why)) :
    progOf x (cs ++ [d]) = { s with stopped := (progOf x (cs ++ [d])).stopped } := by
  have hnone : m.inner = none := by
    obtain ⟨c'', _, _, _, _, _, hph', _, hm'⟩ := freshOf_ok x cs hx hq hm
    rcases hm' with ⟨_, rfl⟩ | ⟨_, st', hp, rfl⟩
    · rfl
    · rw [← show (_root_.Flock.Firewall.Frame.step c'' {} _).2 = st' from congrArg Prod.snd hp]
      exact Sched.step_inner c'' {} _ (by trivial) rfl
  obtain ⟨fs, fs', k, h1, h1', hn2, hA, hM, hM', hk, h0, hn⟩ := Sched.roundAt_frames (roundsAt_getD hrs hlt)
  rw [h1] at h
  obtain ⟨out', s', w, hc', hu⟩ :=
    Sched.check_alike_self hc m st fs fs' k _ (by rw [hnone]; nofun) (fun f hf => (hM f hf).1)
      (fun f hf => (hM' f hf).1) hk h0 hn hA (fun _ hq' => Sched.foldlM_midAll_inner hnone hM' hq') h
  rw [progOf_first_step x cs d hx hq hj hm hph hs' (by rw [h1', hn2]; exact hc')]
  exact eq_unstop hu

theorem getElem?_flatten_locate {β : Type} :
    ∀ (ls : List (List β)) (k₀ i : ℕ),
      ls.flatten[i]? = (ls.getD ((ZKMal.CoinLayer.locate (ls.map List.length) k₀ i).1 - k₀) [])[
        (ZKMal.CoinLayer.locate (ls.map List.length) k₀ i).2]?
  | [], k₀, i => by simp [ZKMal.CoinLayer.locate]
  | l :: ls, k₀, i => by
    rw [List.flatten_cons, List.map_cons]
    by_cases hc : i < l.length
    · rw [List.getElem?_append_left hc, locate_cons_lt hc]
      simp
    · rw [List.getElem?_append_right (by omega), locate_cons_ge hc, getElem?_flatten_locate ls (k₀ + 1) (i - l.length)]
      have hK := locate_fst_ge (ls.map List.length) (k₀ + 1) (i - l.length)
      rw [show (ZKMal.CoinLayer.locate (ls.map List.length) (k₀ + 1) (i - l.length)).1 - k₀ =
        (ZKMal.CoinLayer.locate (ls.map List.length) (k₀ + 1) (i - l.length)).1 - (k₀ + 1) + 1 by omega]
      simp

/-- **Opening `q + 1`'s coin count is its rep's planned round's**: rep round `q`, at stream `k` and round `r`
(`CoinLayer.locate`), asks `repPlan`'s `r`-th count of table `k / 2`'s statement. -/
theorem loopNr_succ {x : Msg} {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) {q k r : ℕ}
    (hl : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, r)) (hq : q < (roundsOf spec sts).sum) :
    loopNr x (q + 1) = (_root_.Flock.Firewall.Session.repPlan spec sts[k / 2]!).getD r 0 := by
  have hk : k < spec.repStreams.size := by
    have := locate_lt (roundsOf spec sts) 0 q hq
    rw [hl] at this
    simpa [roundsOf] using this
  have hi : q < (_root_.Flock.Firewall.Session.planOf spec sts).size := by rw [plan_size]; exact hq
  have e := getElem?_flatten_locate (plansOf spec sts) 0 q
  rw [plansOf_length, hl, ← planOf_toList, Array.getElem?_toList] at e
  have hg : (plansOf spec sts).getD k [] =
      (_root_.Flock.Firewall.Session.repPlan spec sts[k / 2]!).toList.map
        (_root_.Flock.Firewall.Session.repName spec k, ·) := by
    simp [plansOf, hk]
  rw [Nat.sub_zero, hg, List.getElem?_map] at e
  unfold loopNr
  rw [reqOf_eq hs]
  unfold _root_.Flock.Firewall.Session.reqAt
  simp only [Nat.add_one_ne_zero, ↓reduceIte, Nat.add_sub_cancel]
  rw [Array.getD_eq_getD_getElem?, e]
  cases hr : (_root_.Flock.Firewall.Session.repPlan spec sts[k / 2]!).toList[r]? with
  | none =>
    rw [hr] at e
    simp [hi] at e
  | some n =>
    simp only [Option.map_some, Option.getD_some]
    rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, hr]
    rfl

section InPlan

open Claims.ZKMal
open _root_.Flock.Firewall.Session (planOf reqAt streamsOk)

theorem locate_ge : ∀ (rs : List ℕ) (k i : ℕ), rs.sum ≤ i →
    CoinLayer.locate rs k i = (k + rs.length, i - rs.sum)
  | [], k, i, _ => by simp [CoinLayer.locate]
  | c :: cs, k, i, h => by
    simp only [List.sum_cons] at h
    rw [locate_cons_ge (by omega), locate_ge cs (k + 1) (i - c) (by omega)]
    simp only [List.length_cons, List.sum_cons, Prod.mk.injEq]
    exact ⟨by omega, by omega⟩

theorem open_ok {S S' : _root_.Flock.Firewall.Coins.Session} {st : String} {n : ℕ} {m c : List UInt8}
    (h : S.open st n m = .ok (S', c)) : st ∈ S.hello.spec.streams.toList ∧ S'.hello = S.hello := by
  unfold _root_.Flock.Firewall.Coins.Session.open at h
  dsimp only at h
  refine ⟨?_, ?_⟩
  · by_contra hn
    have hs : ¬ List.idxOf st S.hello.spec.streams.toList < S.hello.spec.streams.size := fun h' =>
      hn (List.idxOf_lt_length_iff.mp (by simpa using h'))
    rw [if_neg hs] at h
    cases h
  · repeat' split at h
    all_goals (cases h <;> rfl)

theorem openAll_mem : ∀ (S S' : _root_.Flock.Firewall.Coins.Session) (os : List (String × ℕ × List UInt8)),
    S.openAll os = .ok S' → ∀ o ∈ os, o.1 ∈ S.hello.spec.streams.toList
  | _, _, [], _, _, ho => by cases ho
  | S, S', (st, n, m) :: os, h, o, ho => by
    unfold _root_.Flock.Firewall.Coins.Session.openAll at h
    cases hop : S.open st n m with
    | error e => simp [hop] at h
    | ok p =>
      obtain ⟨S'', c⟩ := p
      simp only [hop] at h
      obtain ⟨hmem, hhe⟩ := open_ok hop
      rcases List.mem_cons.1 ho with rfl | ho
      · exact hmem
      · rw [← hhe]; exact openAll_mem S'' S' os h o ho

theorem start_hello {hl : _root_.Flock.Firewall.Coins.Hello} {a : List UInt8} {S : _root_.Flock.Firewall.Coins.Session}
    (h : _root_.Flock.Firewall.Coins.Session.start hl a = .ok S) : S.hello = hl := by
  unfold _root_.Flock.Firewall.Coins.Session.start at h
  split at h
  · cases h; rfl
  · cases h

theorem coinVerdict_ok {hl : _root_.Flock.Firewall.Coins.Hello} {a : List UInt8} {os : List (String × ℕ × List UInt8)}
    (h : (coinVerdict (.ok hl) a os).isOk = true) :
    ∃ S S', _root_.Flock.Firewall.Coins.Session.start hl a = .ok S ∧ S.openAll os = .ok S' ∧ S.hello = hl := by
  unfold coinVerdict at h
  cases hS : _root_.Flock.Firewall.Coins.Session.start hl a with
  | error e => simp [Except.bind, hS, Except.isOk, Except.toBool] at h
  | ok S =>
    cases hO : S.openAll os with
    | error e => simp [Except.bind, hS, hO, Except.isOk, Except.toBool] at h
    | ok S' => exact ⟨S, S', rfl, hO, start_hello hS⟩

theorem reqsFrom_mem (req : ℕ → String × ℕ) : ∀ (j : ℕ) (ms : List Msg) (i : ℕ), i < ms.length →
    ∃ o ∈ reqsFrom req j ms, o.1 = (req (j + i)).1
  | _, [], _, h => by simp at h
  | j, m :: ms, 0, _ => ⟨_, List.mem_cons_self, rfl⟩
  | j, m :: ms, i + 1, h => by
    obtain ⟨o, ho, he⟩ := reqsFrom_mem req (j + 1) ms i (by simpa using h)
    exact ⟨o, List.mem_cons_of_mem _ ho, by rw [he, show j + 1 + i = j + (i + 1) by omega]⟩

theorem reqOf_past {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) {q : ℕ} (hq : (planOf spec sts).size < q) :
    reqOf x q = ("", 0) := by
  rw [reqOf_eq hs]
  simp only [reqAt, show q ≠ 0 by omega, ↓reduceIte, Array.getD_eq_getD_getElem?,
    Array.getElem?_eq_none (show (planOf spec sts).size ≤ q - 1 by omega), Option.getD_none]

theorem repStreams_name {spec : _root_.Flock.SessionSpec} {e : ℕ × ℕ × String} (he : e ∈ spec.repStreams.toList) :
    e.2.2 ≠ "" := by
  simp only [_root_.Flock.SessionSpec.repStreams, Array.toList_flatMap, Array.toList_range, List.mem_flatMap,
    List.mem_range, Array.toList_map, List.mem_map] at he
  obtain ⟨i, _, r, _, rfl⟩ := he
  intro h
  have h4 : (toString "/rep").length = 4 := by decide
  have := congrArg String.length h
  simp only [String.length_append, String.length_empty, h4] at this
  omega

theorem empty_not_stream {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    {ct : _root_.Flock.CoinTree.Spec} (hok : streamsOk spec sts ct = true) : "" ∉ ct.streams.toList := by
  rw [(streamsOk_eq hok).1]
  intro hm
  rcases List.mem_cons.1 hm with e | hm
  · exact absurd e (by decide)
  · obtain ⟨e, hmem, he⟩ := List.mem_map.1 hm
    exact repStreams_name hmem he

end InPlan

end Loop

namespace LeanRelease

/-- **Where the frame takes the coins after `h`**: at the round's rep's first-round state; at the state the last coins
leave, where the round is the last's rep's; or, the last's rep's frame taking them, at the next rep's first-round
state. -/
theorem takeAt_ok {n : ℕ} {b : Bool} {cN cP : Cfg} {p : Masker} {f : Except String Masker}
    {cs : Array _root_.Flock.F128} {m : Masker} (h : takeAt n b cN cP p f cs = .ok m) :
    (n ≤ 2 ∧ f = .ok m) ∨ (¬ n ≤ 2 ∧ b = true ∧ p.answer cN cs = .ok m) ∨
      (¬ n ≤ 2 ∧ b = false ∧ (∃ s, p.answer cP cs = .ok s) ∧ f = .ok m) := by
  unfold takeAt at h
  by_cases hn : n ≤ 2
  · rw [if_pos hn] at h
    exact Or.inl ⟨hn, h⟩
  · rw [if_neg hn] at h
    cases b with
    | true =>
      simp only [↓reduceIte] at h
      exact Or.inr (Or.inl ⟨hn, rfl, h⟩)
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte] at h
      cases ha : p.answer cP cs with
      | error e => rw [ha] at h; cases h
      | ok s =>
        rw [ha] at h
        exact Or.inr (Or.inr ⟨hn, rfl, ⟨s, rfl⟩, h⟩)

open Loop in
/-- **The loop's frames take the coins after `e0 :: h1`** (`takeAt`): its reps' (`repOf`) states after their rounds
(`progOf`) and at their first (`freshOf`). -/
theorem ofLoop_frameAnswer (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) :
    (ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) =
      takeAt (e0 :: h1).length (decide (repOf x (e0 :: h1).length = repOf x ((e0 :: h1).length - 1)))
        ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) ((ofLoop T L₀ real).cfgAt x ρ ((e0 :: h1).length - 1))
        (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)) (freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2))
        (coinsAt x (e0 :: h1)) := by
  unfold frameAnswer
  split
  · rfl
  · rename_i heq
    simp at heq

open Loop in
theorem repOf_add_two {x : Msg} {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (i : ℕ) :
    repOf x (i + 2) = (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 i).1 := by
  unfold repOf
  rw [hx]
  simp only [Nat.add_sub_cancel]

open Loop in
/-- **At a rep's round past its first the loop's frame checks it at the state the last coins leave**: the state of
the same rep, which released the last round. -/
theorem ofLoop_frameAnswer_same (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q j : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1)
    (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1) :
    (ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) =
      (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)).answer ((ofLoop T L₀ real).cfg x ρ (e0 :: h1))
        (coinsAt x (e0 :: h1)) := by
  obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := by
    rcases q with _ | q'
    · rw [locate_zero] at hj
      omega
    · exact ⟨q', rfl⟩
  have hp := locate_pred (roundsOf spec sts) 0 q' j hj
  have e1 : (e0 :: h1).length = q' + 1 + 2 := by simp only [List.length_cons, hq]
  have hs : repOf x (e0 :: h1).length = repOf x ((e0 :: h1).length - 1) := by
    rw [e1, show q' + 1 + 2 - 1 = q' + 2 by omega, repOf_add_two hx, repOf_add_two hx, hp]
  rw [ofLoop_frameAnswer]
  unfold takeAt
  rw [if_neg (by rw [e1]; omega), if_pos (decide_eq_true hs)]

open Loop in
/-- **`ProgChecks` at a scheduled round, at the loop's own frames**: where the frame takes the coins after `h` at the
state `m`, the next round is the same rep's, round `j + 1` of rep stream `k` past `Hello` and the link, and the frame's
check of a device round `v` releases `out`, that check, of a running rep, is one `ProgChecks` asks for. The rep's
schedule so far at its own table and salts (`real`, `rsched`) and at the placeholder's (`sched'`) are of one shape and
hold the round, the configuration is alike the placeholder's, and the coin count the loop requests there (`loopNr`) is
the scheduled round's. -/
theorem ofLoop_progChecks_step (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (h : List (Msg × Msg)) (a : Msg) {m : Masker} {out : ByteArray}
    (hA : (ofLoop T L₀ real).answered x ρ h = .ok m)
    (hex : ∃ v s' why, (ofLoop T L₀ real).checkAt x w ρ h m false v = .ok (out, s', why))
    {cs : List FlockDeploy.Coin} (hcs : h.tail.map (fun p => ZKMal.CoinLayer.rd p.2) = cs)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h.length = q + 2) {k j : ℕ}
    (hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1))
    {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} (hph : placeholder spec sts k (wordsAt x cs 0) = .ok (c', t, dom, xs, hi))
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ h) c')
    {tr : _root_.Flock.Firewall.Finish.Tab} {hr : Array ByteArray} (hreal : real x w ρ k = .ok (tr, hr))
    {rsched sched' : Array (ByteArray × ℕ)}
    (hrs0 : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (wordsAt x cs 0))) hr ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok rsched)
    (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t (k % 2) dom xs hi ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) =
      .ok sched')
    (hlt : j + 1 < rsched.size) (hcoins : loopNr x (q + 1) = (rsched.getD (j + 1) (.empty, 0)).2) :
    ∃ v st s' why, (ofLoop T L₀ real).checkAt x w ρ h m st v = .ok (out, s', why) ∧
      (ofLoop T L₀ real).prog x (h ++ [(0 :: out.data.toList, a)]) =
        { s' with stopped := ((ofLoop T L₀ real).prog x (h ++ [(0 :: out.data.toList, a)])).stopped } := by
  have hco : (ofLoop T L₀ real).coins x h.length = (rsched.getD (j + 1) (.empty, 0)).2 := by
    show loopNr x (h.length - 1) = _
    rw [hq, show q + 2 - 1 = q + 1 by omega]
    exact hcoins
  obtain ⟨v, s', why, hv⟩ := hex
  refine ⟨v, false, s', why, hv, ?_⟩
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = k := by rw [hki]
  have hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1 := by rw [hki]
  cases h with
  | nil => simp only [List.length_nil] at hq; omega
  | cons e0 h1 =>
  obtain ⟨m0, r0⟩ := e0
  have hcs1 : h1.map (fun p => ZKMal.CoinLayer.rd p.2) = cs := hcs
  have hq1 : cs.length = q + 1 := by
    rw [← hcs1, List.length_map]
    simp only [List.length_cons] at hq
    omega
  have hcr : coinsAt x ((m0, r0) :: h1) = readCoins x cs := by
    rw [← hcs1]
    exact loop_coinsRead loopNr T (withFrames T L₀ real) x m0 r0 h1
  have hF : (ofLoop T L₀ real).frameAnswer x ρ ((m0, r0) :: h1) = .ok m := by
    unfold answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswer_same T L₀ real x ρ (m0, r0) h1 hx (by rw [← hcs1, List.length_map] at hq1; exact hq1) hj,
    hcs1] at hF
  have hsch : schedOf (real x w ρ) x cs = some (rsched.getD (j + 1) (.empty, 0)).1 := by
    have := schedOf_eq (real x w ρ) x cs hx hq1 (by rw [hk1]; exact hreal) (by rw [hk1, hj]; exact hrs0)
      (by rw [hj]; exact hlt)
    rwa [hj] at this
  have hv' : check ((ofLoop T L₀ real).cfg x ρ ((m0, r0) :: h1)) { m with stopped := false } ⟨v.toArray⟩
      (rsched.getD (j + 1) (.empty, 0)).1 (rsched.getD (j + 1) (.empty, 0)).2 = .ok (out, s', why) := by
    unfold checkAt at hv
    have hsc : ∀ vals, (ofLoop T L₀ real).scheduled x w ρ ((m0, r0) :: h1) vals =
        (rsched.getD (j + 1) (.empty, 0)).1.data.toList := fun vals => by
      show (match schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) with
        | some r => r.data.toList
        | none => L₀.scheduled x w ρ ((m0, r0) :: h1) vals) = _
      rw [hcs1, hsch]
    rw [hsc, Array.toArray_toList] at hv
    rw [← hco]
    exact hv
  have hcs' : (((m0, r0) :: h1) ++ [(0 :: out.data.toList, a)]).tail.map (fun p => ZKMal.CoinLayer.rd p.2) =
      cs ++ [ZKMal.CoinLayer.rd a] := by
    simp only [← hcs1, List.cons_append, List.tail_cons, List.map_append, List.map_cons, List.map_nil]
  show progOf x ((((m0, r0) :: h1) ++ [(0 :: out.data.toList, a)]).tail.map fun p => ZKMal.CoinLayer.rd p.2) =
    { s' with stopped :=
      (progOf x ((((m0, r0) :: h1) ++ [(0 :: out.data.toList, a)]).tail.map fun p => ZKMal.CoinLayer.rd p.2)).stopped }
  rw [hcs']
  rw [hcr, readCoins_eq x cs hq1] at hF
  exact progOf_check_step x cs (ZKMal.CoinLayer.rd a) hx hq1 (by rw [hk1]; exact hph) hc hj hrs
    (by rw [hk1]; exact hs') hlt hF hv'

open Loop in
/-- **`ProgChecks` at a scheduled round, from the rep's own table**: as `ofLoop_progChecks_step`, where the rep's own
table (`real`) is of the placeholder's shape (`TabAlike`, at the bounds `schedSoFar_alike` reads) and its schedule so
far holds round `j + 1`. The placeholder's schedule is then of the real one's shape (`schedSoFar_alike`), so of its
coin counts, and the configuration at the tape's draws alike the placeholder's (`cfgAlike_ofBytes`). -/
theorem ofLoop_progChecks_sched (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (h : List (Msg × Msg)) (a : Msg) {m : Masker} {out : ByteArray}
    (hA : (ofLoop T L₀ real).answered x ρ h = .ok m)
    (hex : ∃ v s' why, (ofLoop T L₀ real).checkAt x w ρ h m false v = .ok (out, s', why))
    {cs : List FlockDeploy.Coin} (hcs : h.tail.map (fun p => ZKMal.CoinLayer.rd p.2) = cs)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h.length = q + 2) {k j : ℕ}
    (hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1))
    {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} (hph : placeholder spec sts k (wordsAt x cs 0) = .ok (c', t, dom, xs, hi))
    {tr : _root_.Flock.Firewall.Finish.Tab} {hr : Array ByteArray} (hreal : real x w ρ k = .ok (tr, hr))
    (hT : Sched.TabAlike tr t) (hdom : dom.size < 2 ^ 64) (hdg : tr.st.digest.size < 2 ^ 64) (hm : tr.st.m < 2 ^ 64)
    (hxs : xs.size + 2 < 2 ^ 64) (hn : tr.sch.n + 1 < 2 ^ 64) (hlq : ∀ i, (tr.sch.levels.getD i default).q < 2 ^ 64)
    (hp : 2 * tr.tPad < 2 ^ 64) (hd : ∀ i, 64 * 2 ^ (tr.sch.levels.getD i default).d < 2 ^ 64)
    {rsched : Array (ByteArray × ℕ)}
    (hrs0 : schedSoFar tr (k % 2) dom xs hr ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) =
      .ok rsched)
    (hlt : j + 1 < rsched.size)
    (hcoins : ∀ sched', schedSoFar t (k % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok sched' → j + 1 < sched'.size →
      loopNr x (q + 1) = (sched'.getD (j + 1) (.empty, 0)).2) :
    ∃ v st s' why, (ofLoop T L₀ real).checkAt x w ρ h m st v = .ok (out, s', why) ∧
      (ofLoop T L₀ real).prog x (h ++ [(0 :: out.data.toList, a)]) =
        { s' with stopped := ((ofLoop T L₀ real).prog x (h ++ [(0 :: out.data.toList, a)])).stopped } := by
  obtain ⟨lay, mask, hst, hc', hdm, hxe⟩ := placeholder_ok hph
  have hR := Sched.schedSoFar_alike tr t (k % 2) dom xs hr hi
    ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1
    hT.2.2.2.2 hdom hdg hm hxs hn hlq hp hd
  rw [hrs0] at hR
  cases hs' : schedSoFar t (k % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) with
  | error e => rw [hs'] at hR; exact (hR : False).elim
  | ok sched' =>
  rw [hs'] at hR
  have hrk : (ofLoop T L₀ real).rep x h.length = k := by
    show repOf x h.length = k
    unfold repOf
    rw [hx, hq]
    simp only [Nat.add_sub_cancel]
    rw [hki]
  have hso : startOf x k = .ok (lay, mask) := by
    unfold startOf
    rw [hx]
    exact hst
  have hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ h) c' := by
    unfold cfg drawsAt
    rw [hrk, hc']
    have e1 : (ofLoop T L₀ real).lay x k = lay := by
      show ((startOf x k).toOption.map (·.1)).getD default = lay
      rw [hso]
      rfl
    have e2 : (ofLoop T L₀ real).slot x k = some mask := by
      show (startOf x k).toOption.map (·.2) = some mask
      rw [hso]
      rfl
    rw [e1, e2]
    exact cfgAlike_ofBytes lay _ _ (some mask)
  exact ofLoop_progChecks_step T L₀ real x w ρ h a hA hex hcs hx hq hki hph hc hreal (by rw [← hdm, ← hxe]; exact hrs0)
    hR hs' hlt ((hcoins sched' hs' (by rw [← roundsAt_size hR]; exact hlt)).trans (roundsAt_getD hR hlt).1.symm)

open Loop in
/-- **`ProgChecks` at a rep's first round, at the loop's own frames**: as `ofLoop_progChecks_step`, where the round
after `e0 :: h1` is the first of rep stream `k`, which the frame starts at `m` (`freshOf`), and the rep's schedules on
no coins, at its own table and the placeholder's, are of one shape. -/
theorem ofLoop_progChecks_first (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (a : Msg) {m : Masker} {out : ByteArray}
    (hm : freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m)
    (hex : ∃ v s' why, (ofLoop T L₀ real).checkAt x w ρ (e0 :: h1) m false v = .ok (out, s', why))
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1) {k : ℕ}
    (hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, 0))
    {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray}
    (hph : placeholder spec sts k (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi))
    (hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c')
    {tr : _root_.Flock.Firewall.Finish.Tab} {hr : Array ByteArray} (hreal : real x w ρ k = .ok (tr, hr))
    {rsched sched' : Array (ByteArray × ℕ)}
    (hrs0 : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.Firewall.Drive.extras sts[k / 2]! (_root_.Flock.Firewall.Drive.linkPoints sts[k / 2]!
        (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hr #[] = .ok rsched)
    (hrs : Sched.RoundsAt rsched sched') (hs' : schedSoFar t (k % 2) dom xs hi #[] = .ok sched')
    (hcoins : loopNr x (q + 1) = (rsched.getD 0 (.empty, 0)).2) :
    ∃ v st s' why, (ofLoop T L₀ real).checkAt x w ρ (e0 :: h1) m st v = .ok (out, s', why) ∧
      (ofLoop T L₀ real).prog x ((e0 :: h1) ++ [(0 :: out.data.toList, a)]) =
        { s' with stopped := ((ofLoop T L₀ real).prog x ((e0 :: h1) ++ [(0 :: out.data.toList, a)])).stopped } := by
  have hlt : 0 < rsched.size := schedSoFar_nil_pos hrs0
  have hco : (ofLoop T L₀ real).coins x (e0 :: h1).length = (rsched.getD 0 (.empty, 0)).2 := by
    show loopNr x ((e0 :: h1).length - 1) = _
    rw [List.length_cons, hq, Nat.add_sub_cancel]
    exact hcoins
  obtain ⟨v, s', why, hv⟩ := hex
  refine ⟨v, false, s', why, hv, ?_⟩
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = k := by rw [hki]
  have hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = 0 := by rw [hki]
  have hq1 : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  have hsch : schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) =
      some (rsched.getD 0 (.empty, 0)).1 := by
    have := schedOf_eq (real x w ρ) x _ hx hq1 (by rw [hk1]; exact hreal)
      (by rw [hk1, hj, range_zero_map]; exact hrs0) (by rw [hj]; exact hlt)
    rwa [hj] at this
  have hv' : check ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) { m with stopped := false } ⟨v.toArray⟩
      (rsched.getD 0 (.empty, 0)).1 (rsched.getD 0 (.empty, 0)).2 = .ok (out, s', why) := by
    unfold checkAt at hv
    have hsc : ∀ vals, (ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals =
        (rsched.getD 0 (.empty, 0)).1.data.toList := fun vals => by
      show (match schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) with
        | some r => r.data.toList
        | none => L₀.scheduled x w ρ (e0 :: h1) vals) = _
      rw [hsch]
    rw [hsc, Array.toArray_toList] at hv
    rw [← hco]
    exact hv
  have hcs' : ((e0 :: h1) ++ [(0 :: out.data.toList, a)]).tail.map (fun p => ZKMal.CoinLayer.rd p.2) =
      (h1.map fun p => ZKMal.CoinLayer.rd p.2) ++ [ZKMal.CoinLayer.rd a] := by
    simp only [List.cons_append, List.tail_cons, List.map_append, List.map_cons, List.map_nil]
  show progOf x (((e0 :: h1) ++ [(0 :: out.data.toList, a)]).tail.map fun p => ZKMal.CoinLayer.rd p.2) =
    { s' with stopped :=
      (progOf x (((e0 :: h1) ++ [(0 :: out.data.toList, a)]).tail.map fun p => ZKMal.CoinLayer.rd p.2)).stopped }
  rw [hcs']
  exact progOf_check_first x _ (ZKMal.CoinLayer.rd a) hx hq1 hj hm (by rw [hk1]; exact hph) hc hrs
    (by rw [hk1]; exact hs') hlt hv'

open Loop in
/-- **`ProgChecks` at a rep's first round, from the rep's own table**: as `ofLoop_progChecks_sched`, at the rep's
first round (`ofLoop_progChecks_first`). -/
theorem ofLoop_progChecks_sched0 (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) (a : Msg) {m : Masker} {out : ByteArray}
    (hm : freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m)
    (hex : ∃ v s' why, (ofLoop T L₀ real).checkAt x w ρ (e0 :: h1) m false v = .ok (out, s', why))
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1) {k : ℕ}
    (hki : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, 0))
    {c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray}
    (hph : placeholder spec sts k (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi))
    {tr : _root_.Flock.Firewall.Finish.Tab} {hr : Array ByteArray} (hreal : real x w ρ k = .ok (tr, hr))
    (hT : Sched.TabAlike tr t) (hdom : dom.size < 2 ^ 64) (hdg : tr.st.digest.size < 2 ^ 64) (hm' : tr.st.m < 2 ^ 64)
    (hxs : xs.size + 2 < 2 ^ 64) (hn : tr.sch.n + 1 < 2 ^ 64) (hlq : ∀ i, (tr.sch.levels.getD i default).q < 2 ^ 64)
    (hp : 2 * tr.tPad < 2 ^ 64) (hd : ∀ i, 64 * 2 ^ (tr.sch.levels.getD i default).d < 2 ^ 64)
    {rsched : Array (ByteArray × ℕ)} (hrs0 : schedSoFar tr (k % 2) dom xs hr #[] = .ok rsched)
    (hcoins : ∀ sched', schedSoFar t (k % 2) dom xs hi #[] = .ok sched' → 0 < sched'.size →
      loopNr x (q + 1) = (sched'.getD 0 (.empty, 0)).2) :
    ∃ v st s' why, (ofLoop T L₀ real).checkAt x w ρ (e0 :: h1) m st v = .ok (out, s', why) ∧
      (ofLoop T L₀ real).prog x ((e0 :: h1) ++ [(0 :: out.data.toList, a)]) =
        { s' with stopped := ((ofLoop T L₀ real).prog x ((e0 :: h1) ++ [(0 :: out.data.toList, a)])).stopped } := by
  obtain ⟨lay, mask, hst, hc', hdm, hxe⟩ := placeholder_ok hph
  have hlt : 0 < rsched.size := schedSoFar_nil_pos hrs0
  have hR := Sched.schedSoFar_alike tr t (k % 2) dom xs hr hi #[] hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1
    hT.2.2.2.2 hdom hdg hm' hxs hn hlq hp hd
  rw [hrs0] at hR
  cases hs' : schedSoFar t (k % 2) dom xs hi #[] with
  | error e => rw [hs'] at hR; exact (hR : False).elim
  | ok sched' =>
  rw [hs'] at hR
  have hrk : (ofLoop T L₀ real).rep x (e0 :: h1).length = k := by
    show repOf x (e0 :: h1).length = k
    rw [List.length_cons, hq, repOf_add_two hx, hki]
  have hso : startOf x k = .ok (lay, mask) := by
    unfold startOf
    rw [hx]
    exact hst
  have hc : CfgAlike ((ofLoop T L₀ real).cfg x ρ (e0 :: h1)) c' := by
    unfold cfg drawsAt
    rw [hrk, hc']
    have e1 : (ofLoop T L₀ real).lay x k = lay := by
      show ((startOf x k).toOption.map (·.1)).getD default = lay
      rw [hso]
      rfl
    have e2 : (ofLoop T L₀ real).slot x k = some mask := by
      show (startOf x k).toOption.map (·.2) = some mask
      rw [hso]
      rfl
    rw [e1, e2]
    exact cfgAlike_ofBytes lay _ _ (some mask)
  exact ofLoop_progChecks_first T L₀ real x w ρ e0 h1 a hm hex hx hq hki hph hc hreal
    (by rw [← hdm, ← hxe]; exact hrs0) hR hs'
    ((hcoins sched' hs' (by rw [← roundsAt_size hR]; exact hlt)).trans (roundsAt_getD hR hlt).1.symm)

open Loop in
/-- **Where `ProgChecks` asks anything at the loop's own frames**: where the frame takes the coins after `h`, a masked
round, and its check of some device round releases a round, `h` is past `Hello` and the `Commit`, `x` sets up a session,
and the next round is round `j` of some rep stream `k`: past its first, where the frame checks it at the state the last
coins leave (`ofLoop_frameAnswer_same`), that of the rep's placeholder; at its first, at the state the frame starts it
at (`freshOf`). Where `x` sets up no session the frame takes no coins and starts no rep (`progOf_none`, `freshOf_none`),
and at `Hello` the round's scheduled round, the empty round, the check refuses (`check_empty`). -/
theorem ofLoop_progChecks_round (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (h : List (Msg × Msg)) {m : Masker} {out : ByteArray}
    (hA : (ofLoop T L₀ real).answered x ρ h = .ok m)
    (hkd : (ofLoop T L₀ real).kindAt x h.length = .masked)
    (hex : ∃ v s' why, (ofLoop T L₀ real).checkAt x w ρ h m false v = .ok (out, s', why)) :
    ∃ e0 h1 s0 spec sts q k j, h = e0 :: h1 ∧ FlockDeploy.setupOf true x = some (s0, spec, sts) ∧
      h1.length = q + 1 ∧ ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j) ∧
      ((∃ j' c' t dom xs hi, j = j' + 1 ∧
          placeholder spec sts k (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi)) ∨
        (j = 0 ∧ freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m)) := by
  obtain ⟨-, hk1, -⟩ := kindAt_masked hkd
  obtain ⟨v, s', why, hv⟩ := hex
  cases h with
  | nil =>
    unfold checkAt at hv
    exact absurd hv (check_empty _ _ _ _ _)
  | cons e0 h1 =>
  have hF : (ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) = .ok m := by
    unfold answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswer] at hF
  obtain ⟨q, hq⟩ : ∃ q, h1.length = q + 1 :=
    ⟨h1.length - 1, by simp only [List.length_cons] at hk1; omega⟩
  have e1 : (e0 :: h1).length = q + 2 := by simp only [List.length_cons, hq]
  have hcs : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  rcases hx : FlockDeploy.setupOf true x with _ | ⟨s0, spec, sts⟩
  · exfalso
    rw [progOf_none x _ hx] at hF
    rcases takeAt_ok hF with ⟨-, hf⟩ | ⟨-, -, ha⟩ | ⟨-, -, -, hf⟩
    · exact freshOf_none x _ hx hf
    · exact answer_owed_none _ rfl ha
    · exact freshOf_none x _ hx hf
  rcases hl : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q with ⟨k, _ | j'⟩
  · refine ⟨e0, h1, s0, spec, sts, q, k, 0, rfl, rfl, hq, hl, Or.inr ⟨rfl, ?_⟩⟩
    rcases takeAt_ok hF with ⟨-, hf⟩ | ⟨hn, hb, -⟩ | ⟨-, -, -, hf⟩
    · exact hf
    · exfalso
      obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := ⟨q - 1, by rw [e1] at hn; omega⟩
      rw [e1, show q' + 1 + 2 - 1 = q' + 2 by omega, repOf_add_two hx, repOf_add_two hx] at hb
      have := locate_first (roundsOf spec sts) 0 q' (of_decide_eq_true hb)
      rw [hl] at this
      exact this rfl
    · exact hf
  · rcases takeAt_ok hF with ⟨hn, -⟩ | ⟨-, -, ha⟩ | ⟨hn, hb, -, -⟩
    · exfalso
      rw [e1] at hn
      obtain rfl : q = 0 := by omega
      have := locate_zero (roundsOf spec sts) 0
      rw [hl] at this
      exact Nat.succ_ne_zero _ this
    · have ho : (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun ho => answer_owed_none _ ho ha
      obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := by
        rcases q with _ | q'
        · have := locate_zero (roundsOf spec sts) 0
          rw [hl] at this
          exact absurd this (Nat.succ_ne_zero _)
        · exact ⟨q', rfl⟩
      have hp := locate_pred (roundsOf spec sts) 0 q' j' (by rw [hl])
      obtain ⟨c', t, dom, xs, hi, hph⟩ := progOf_placeholder x _ hx (by rw [hcs]) ho
      rw [hp, hl] at hph
      exact ⟨e0, h1, s0, spec, sts, q' + 1, k, j' + 1, rfl, rfl, hq, hl, Or.inl ⟨j', c', t, dom, xs, hi, rfl, hph⟩⟩
    · exfalso
      obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := ⟨q - 1, by rw [e1] at hn; omega⟩
      have hp := locate_pred (roundsOf spec sts) 0 q' j' (by rw [hl])
      rw [e1, show q' + 1 + 2 - 1 = q' + 2 by omega, repOf_add_two hx, repOf_add_two hx, hp, hl] at hb
      simp at hb

open Loop in
/-- **Where the loop has no round in the device's place, the frame's check refuses every device round**: after a
history past `Hello` whose round's scheduled round is the empty round (`schedOf`, where the rep's table or schedule is
refused). -/
theorem ofLoop_checkAt_empty (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg))
    (hs : schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = some .empty) (m : Masker) (st : Bool)
    (v : Msg) (q : ByteArray × Masker × Option String) :
    (ofLoop T L₀ real).checkAt x w ρ (e0 :: h1) m st v ≠ .ok q := by
  have hsc : ∀ vals, (ofLoop T L₀ real).scheduled x w ρ (e0 :: h1) vals = [] := fun vals => by
    show (match schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) with
      | some r => r.data.toList
      | none => L₀.scheduled x w ρ (e0 :: h1) vals) = _
    rw [hs]
    rfl
  unfold checkAt
  rw [hsc]
  exact check_empty _ _ _ _ _

section InPlan

open Claims.ZKMal

open Loop in
/-- **A rep round the loop's frame takes coins for is within the plan**: where the coins after `h` reach the frame
(`answered`) and the round is not the `Finish`'s, the transport's coin check opened every one of `h`'s openings, the
`Finish`'s round is the one just past the plan (`lastOf`), and past it an opening's request names no stream of the tree
(`reqOf_past`; the rep streams are `<t>/rep<r>`). -/
theorem loop_round_in_plan (T : CoinLayer.TableData)
    (L : LeanRelease Flock.Wit ((CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand))
    (x : Msg) (ρ : (CoinLayer.session loopNr T).refP.Nonce × (CoinLayer.session loopNr T).refP.Rand)
    (h : List (Msg × Msg)) {m : Masker} (hA : (loop loopNr T L).answered x ρ h = .ok m) (hne : h.length ≠ lastOf x)
    {s0 : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h.length = q + 2) : q < (roundsOf spec sts).sum := by
  by_contra hlt
  have ho : (loop loopNr T L).opens x ρ h = true := by
    unfold answered at hA
    split at hA
    · assumption
    · cases hA
  match h, hq, ho, hne with
  | [], hq, _, _ => simp at hq
  | [_], hq, _, _ => simp only [List.length_cons, List.length_nil] at hq; omega
  | (_, a) :: o :: ops, hq, ho, hne =>
    have hv : (coinVerdict (helloChkAt x ρ.1) a (reqsFrom (reqOf x) 0 ((o :: ops).map Prod.snd))).isOk = true := ho
    cases hh : helloChkAt x ρ.1 with
    | error e => rw [hh] at hv; cases hv
    | ok hl' =>
      rw [hh] at hv
      obtain ⟨s', spec', sts', ct, hs', hc, hok, hle⟩ := helloChkAt_ok hh
      rw [hx] at hs'
      simp only [Option.some.injEq, Prod.mk.injEq] at hs'
      obtain ⟨rfl, rfl, rfl⟩ := hs'
      rcases Nat.lt_or_ge (roundsOf spec sts).sum q with hlt' | hge
      · obtain ⟨S, S', _, hO, hhe⟩ := coinVerdict_ok hv
        simp only [List.length_cons] at hq
        obtain ⟨o', hmem, he⟩ := reqsFrom_mem (reqOf x) 0 ((o :: ops).map Prod.snd) q
          (by simp only [List.length_map, List.length_cons]; omega)
        have hst := openAll_mem S S' _ hO o' hmem
        rw [he, hhe, hle, Nat.zero_add, reqOf_past hx (by rw [plan_size]; omega)] at hst
        exact empty_not_stream hok hst
      · apply hne
        unfold lastOf
        rw [hx]
        simp only [hc, hok, ↓reduceIte]
        rw [plan_size, hq]
        omega

end InPlan

open Loop in
/-- **`ProgChecks` at the loop's own frames**, from what a round reads beyond the loop's own definitions. At a scheduled
round, the rep's own table, where `real` answers the loop's own (`hR`, `Count.RealTabs`), is of the placeholder's shape
(`Count.real_tabAlike`) and within the bounds `schedSoFar_alike` reads (`Count.real_bounds`; the placeholder's domain and
extra claims within a `u64`'s count, `hS`). There the plan's coin counts (`repPlan`, which the loop requests, `loopNr_succ`; the round is within the plan,
`loop_round_in_plan`) are the placeholder's schedule's (`Count.plan_sched`), where each table's link points are the
session's (`hP`: its `m_pts` the session's, nonzero, and the link within `Frames.COIN_MAX`); the loop's state there is
outside the inner proof (`repState_inner`). At a rep's first round, its round 0 there, from the state the frame starts
it at (`ofLoop_progChecks_sched0`). At the inner proof's rounds, past the rep's schedule (`hO`): `ProgChecks`' own
conclusion, which `FrameInner`'s `ofLoop_progChecks` proves. Everywhere else `ProgChecks` asks nothing
(`ofLoop_progChecks_round`), or the loop has no round in the device's place (`ofLoop_checkAt_empty`). -/
theorem ofLoop_progChecks_of (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (w : Flock.Wit)
    (hS : ∀ (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (k : ℕ) (link : Array F128) (c' : Cfg)
      (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray) (xs : Array (Array F128)) (hi : Array ByteArray),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → k < spec.repStreams.size →
      link.size ≤ _root_.Flock.Firewall.Frames.COIN_MAX → placeholder spec sts k link = .ok (c', t, dom, xs, hi) →
      dom.size < 2 ^ 64 ∧ xs.size + 2 < 2 ^ 64)
    (hR : ∀ (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
      (s0 : _) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (hP : ∀ (s0 : _) (spec : SessionSpec) (sts : Array Stmt) (k : ℕ),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → k < spec.repStreams.size →
      sts[k / 2]!.mPts = spec.mPts ∧ 0 < spec.mPts ∧ spec.points * spec.mPts ≤ _root_.Flock.Firewall.Frames.COIN_MAX)
    (hO : ∀ (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
      (h : List (Msg × Msg)) (a : Msg) (m : Masker) (out : ByteArray) (s0 : _) (spec : SessionSpec)
      (sts : Array Stmt) (q k j : ℕ) (c' : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (dom : ByteArray)
      (xs : Array (Array F128)) (hi : Array ByteArray) (tr : _root_.Flock.Firewall.Finish.Tab) (hr : Array ByteArray)
      (rsched : Array (ByteArray × ℕ)),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → h.length = q + 2 →
      ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j + 1) →
      placeholder spec sts k (wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi) →
      real x w ρ k = .ok (tr, hr) →
      schedSoFar tr (k % 2) dom xs hr ((Array.range (j + 1)).map
        fun i => wordsAt x (h.tail.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j + 1) + i + 1)) = .ok rsched →
      rsched.size ≤ j + 1 → (ofLoop T L₀ real).answered x ρ h = .ok m →
      (ofLoop T L₀ real).kindAt x h.length = .masked →
      (∃ v s' why, (ofLoop T L₀ real).checkAt x w ρ h m false v = .ok (out, s', why)) →
      ∃ v st s' why, (ofLoop T L₀ real).checkAt x w ρ h m st v = .ok (out, s', why) ∧
        (ofLoop T L₀ real).prog x (h ++ [(0 :: out.data.toList, a)]) =
          { s' with stopped := ((ofLoop T L₀ real).prog x (h ++ [(0 :: out.data.toList, a)])).stopped }) :
    ProgChecks (ofLoop T L₀ real) x w := by
  intro ρ h a m out hA hkd hex
  obtain ⟨e0, h1, s0, spec, sts, q, k, j, rfl, hx, hq, hki, hcase⟩ :=
    ofLoop_progChecks_round T L₀ real x w ρ h hA hkd hex
  have hne : (e0 :: h1).length ≠ lastOf x := (kindAt_masked hkd).2.2
  have hq2 : (e0 :: h1).length = q + 2 := by simp only [List.length_cons, hq]
  have hcsl : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = k := by rw [hki]
  obtain ⟨c', t, dom, xs, hi, hph⟩ : ∃ c' t dom xs hi,
      placeholder spec sts k (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, hi) := by
    rcases hcase with ⟨j', c', t, dom, xs, hi, -, hph⟩ | ⟨-, hm⟩
    · exact ⟨c', t, dom, xs, hi, hph⟩
    · obtain ⟨c', t, dom, xs, hi, -, hph, -⟩ := freshOf_ok x _ hx hcsl hm
      rw [hk1] at hph
      exact ⟨c', t, dom, xs, hi, hph⟩
  obtain ⟨lay, mask, hst, hc', hdm, hxe⟩ := placeholder_ok hph
  have hno : schedOf (real x w ρ) x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = some .empty → False := fun hs => by
    obtain ⟨v, s', why, hv⟩ := hex
    exact ofLoop_checkAt_empty T L₀ real x w ρ e0 h1 hs m false v _ hv
  have hqs := loop_round_in_plan T (withFrames T L₀ real) x ρ (e0 :: h1) hA hne hx hq2
  have hk : k < spec.repStreams.size := by
    have := locate_lt (roundsOf spec sts) 0 q hqs
    rw [hki] at this
    simpa [roundsOf] using this
  have hls : (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0).size =
      min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    cases h1 with
    | nil => simp at hq
    | cons p h2 => simp [wordsAt, FlockDeploy.Coin.vals]
  obtain ⟨hdom, hxs⟩ := hS s0 spec sts k _ c' t dom xs hi hx hk (by rw [hls]; exact Nat.min_le_right _ _) hph
  obtain ⟨hmp, hmp0, hpc⟩ := hP s0 spec sts k hx hk
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hx]
    rfl
  have hpts : (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0).size / sts[k / 2]!.mPts = spec.points := by
    rw [hls, hl0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0]
  cases hr0 : real x w ρ k with
  | error e => exact (hno (schedOf_real_error (real x w ρ) x _ hx hcsl (by rw [hk1]; exact hr0))).elim
  | ok r =>
  obtain ⟨tr, hr⟩ := r
  have hT := Count.real_tabAlike (hR ρ s0 spec sts hx) hph hr0
  obtain ⟨hdg, hmb, hn, hlq, hp, hd⟩ := Count.real_bounds hph hT
  have hcn : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64 := by
    rw [← hT.2.2.2.2.1]
    exact hT.2.2.2.2.2
  rcases hcase with ⟨j', -, -, -, -, -, rfl, -⟩ | ⟨rfl, hm⟩
  · have hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j' + 1 := by rw [hki]
    cases hrs : schedSoFar tr (k % 2) dom xs hr ((Array.range (j' + 1)).map
        fun i => wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - (j' + 1) + i + 1)) with
    | error e =>
      exact (hno (schedOf_sched_error (real x w ρ) x _ hx hcsl (by rw [hk1]; exact hr0) (e := e)
        (by rw [hk1, hj, ← hdm, ← hxe]; exact hrs))).elim
    | ok rsched =>
    by_cases hlt : j' + 1 < rsched.size
    · exact ofLoop_progChecks_sched T L₀ real x w ρ (e0 :: h1) a hA hex rfl hx hq2 hki hph hr0 hT hdom hdg hmb hxs
        hn hlq hp hd hrs hlt (fun sched' hs' hlt' => (loopNr_succ hx hki hqs).trans
          (Count.plan_sched spec sts k (j' + 1) _ c' t dom xs hi _ sched' hph hs' hlt' hpts hdom hcn hxs))
    · exact hO ρ (e0 :: h1) a m out s0 spec sts q k j' c' t dom xs hi tr hr rsched hx hq2 hki hph hr0 hrs (by omega)
        hA hkd hex
  · have hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = 0 := by rw [hki]
    cases hrs : schedSoFar tr (k % 2) dom xs hr #[] with
    | error e =>
      exact (hno (schedOf_sched_error (real x w ρ) x _ hx hcsl (by rw [hk1]; exact hr0) (e := e)
        (by rw [hk1, hj, range_zero_map, ← hdm, ← hxe]; exact hrs))).elim
    | ok rsched =>
    exact ofLoop_progChecks_sched0 T L₀ real x w ρ e0 h1 a hm hex hx hq hki hph hr0 hT hdom hdg hmb hxs hn hlq hp hd
      hrs (fun sched' hs' hlt' => (loopNr_succ hx hki hqs).trans
        (Count.plan_sched spec sts k 0 _ c' t dom xs hi _ sched' hph hs' hlt' hpts hdom hcn hxs))

end LeanRelease

end Claims.ZKServer
