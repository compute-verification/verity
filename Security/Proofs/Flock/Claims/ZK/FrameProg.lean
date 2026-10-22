import Proofs.Flock.Claims.ZK.FrameSched
import Proofs.Flock.Claims.ZK.FrameState

/-!
# The loop's frame state is the deployed frame's check's (Stage B, `ProgChecks`)

Toward `ofLoop_progChecks` (`note:20261022T0001Z-report-zkfin-stageb`): at a scheduled round the loop's frame state after
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

theorem midAll_alike {fs fs' : List Frame} (hA : List.Forall₂ Alike fs fs') : MidAll fs → MidAll fs' := by
  induction hA with
  | nil => exact fun h => h
  | cons ha _ ih =>
    intro hM g hg
    rcases List.mem_cons.mp hg with rfl | hg
    · exact midFrame_alike ha (hM _ List.mem_cons_self)
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
    (hI : ∀ (fs : List Frame) (k : UInt8), (sched'.getD (j + 1) (.empty, 0)).1 =
        ByteArray.mk (encs fs ++ (Frame.sq k r.2).enc).toArray →
      ∀ q, fs.foldlM (stepAt c') (m, ByteArray.empty, #[], 0) = .ok q → q.1.inner = none)
    {st : Bool} {v out : ByteArray} {s : Masker} {why : Option String}
    (h : check c { m with stopped := st } v r.1 r.2 = .ok (out, s, why)) :
    repState c' t rep dom xs hi cr0 (cr.push a) ns (j + 2) =
      { s with stopped := (repState c' t rep dom xs hi cr0 (cr.push a) ns (j + 2)).stopped } := by
  obtain ⟨fs, fs', k, h1, h1', hn2, hA, hM, hM', hk, h0, hn⟩ := Sched.roundAt_frames hR
  rw [h1] at h
  obtain ⟨out', s', w, hc', hu⟩ :=
    Sched.check_alike_self hc m st fs fs' k r.2 hin hM hM' hk h0 hn hA (hI fs' k h1') h
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

/-- **The loop's frame state one coin on, within a rep stream**: where the round after the decoded coins `cs ++ [d]` is
the same rep stream's as the round after `cs`, the state is that rep's after one more of its rounds, on its rounds'
coins and one more. -/
theorem progOf_snoc (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : ℕ} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1)
    (hk : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q + 1)).1 = (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1) :
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
  have h2 := locate_succ (roundsOf spec sts) 0 q hk
  have hle := locate_snd_le (roundsOf spec sts) 0 q
  unfold progOf
  rw [hx, List.length_append, List.length_singleton, hq]
  simp only
  rw [hk, h2, wordsAt_append x cs [d] (by omega), crAt_push]
  have hst : q + 1 - ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + 1) =
      q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 := by omega
  rw [hst, show q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 +
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

/-- **The loop's frame state after the decoded coins `cs`**, where `x` sets up a session and the placeholder is
`(c, t, dom, xs, hi)`: the located rep's after its released rounds. -/
theorem progOf_at (x : Msg) (cs : List FlockDeploy.Coin) {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1) {c : Cfg}
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
  unfold progOf
  rw [hx, hq]
  simp only [hph]

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

/-- **The loop's frame state one scheduled round on, within a rep stream, is the deployed check's** but for `stopped`:
where the round after the decoded coins `cs ++ [d]` is the same rep stream's as the round after `cs`, round `j + 1` of
it, the schedule so far at its own table (`rsched`) and the placeholder's (`sched'`, on the same coins) are of one shape
(`RoundsAt`) and hold that round, the frame takes the last coins of `cs` at the state `m` outside the inner proof, and the
deployed frame's check at an alike configuration of a device round against the rep's round `j + 1` releases a round,
the loop's state after `cs ++ [d]` is that check's, but for `stopped`. `hI`: the placeholder round's frames enter no
inner proof. -/
theorem progOf_check_step (x : Msg) (cs : List FlockDeploy.Coin) (d : FlockDeploy.Coin) {s0 : _}
    {spec : SessionSpec} {sts : Array Stmt} {q : ℕ} (hx : FlockDeploy.setupOf true x = some (s0, spec, sts))
    (hq : cs.length = q + 1)
    (hk : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 (q + 1)).1 = (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1)
    {c c' : Cfg} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray}
    (hph : placeholder spec sts (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 (wordsAt x cs 0) =
      .ok (c', t, dom, xs, hi))
    (hc : CfgAlike c c') {j : ℕ} (hj : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j + 1)
    {rsched sched' : Array (ByteArray × ℕ)} (hrs : Sched.RoundsAt rsched sched')
    (hs' : schedSoFar t ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2) dom xs hi
      ((Array.range (j + 1)).map fun i => wordsAt x cs (q - (j + 1) + i + 1)) = .ok sched')
    (hlt : j + 1 < rsched.size) {m : Masker} (ha : (progOf x cs).answer c (wordsAt x cs q) = .ok m)
    (hin : m.inner ≠ some INNER_ROUNDS)
    (hI : ∀ (fs : List Frame) (k : UInt8), (sched'.getD (j + 1) (.empty, 0)).1 =
        ByteArray.mk (encs fs ++ (Frame.sq k (rsched.getD (j + 1) (.empty, 0)).2).enc).toArray →
      ∀ q, fs.foldlM (stepAt c') (m, ByteArray.empty, #[], 0) = .ok q → q.1.inner = none)
    {st : Bool} {v out : ByteArray} {s : Masker} {why : Option String}
    (h : check c { m with stopped := st } v (rsched.getD (j + 1) (.empty, 0)).1 (rsched.getD (j + 1) (.empty, 0)).2 =
      .ok (out, s, why)) :
    progOf x (cs ++ [d]) = { s with stopped := (progOf x (cs ++ [d])).stopped } := by
  have hle := locate_snd_le (roundsOf spec sts) 0 q
  rw [progOf_at x cs hx hq hph, hj] at ha
  rw [progOf_snoc x cs d hx hq hk, hph]
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
  exact repState_check_step hc t _ dom xs hi _ _ _ _ j hal hs hs' (by simp) (by rw [← roundsAt_size hrs]; exact hlt)
    (roundsAt_getD hrs hlt) ha hin hI h

end Loop

end Claims.ZKServer
