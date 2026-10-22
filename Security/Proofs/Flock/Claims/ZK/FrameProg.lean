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

open _root_.Flock (V F128)
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
the rep's schedule so far has it. -/
theorem repState_at (c : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (j : ℕ)
    {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hj : j < sched.size) :
    repState c t rep dom xs hi cr0 cr ns (j + 1) =
      let r := _root_.Flock.Firewall.Frame.run c {}
        (schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr j)
      if r.2 then {} else r.1.m := by
  unfold repState
  rw [hal]
  simp only [bind, Except.bind, hs, pure, Except.pure]
  rw [show ∀ p : ℕ, p + 2 * (j + 1) - 1 = p + 2 * j + 1 from fun p => by omega, reqs_take _ _ _ _ _ hj]
  rfl

/-- **A rep's frame state one round on**: where its rounds' coins `cr` cover its scheduled round `j` and the schedule
so far has round `j + 1`, its state after round `j + 1` on one more round's coins is its state after round `j`, then
the coins of round `j`, then round `j + 1`, the scheduled round in the device's place. -/
theorem repState_succ (c : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : ℕ → ℕ) (a : Array F128)
    (j : ℕ) {alg sched sched' : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) (hs' : schedSoFar t rep dom xs hi (cr.push a) = .ok sched')
    (hcr : cr.size = j + 1) (hj : j + 1 < sched.size) :
    repState c t rep dom xs hi cr0 (cr.push a) ns (j + 2) =
      let r := _root_.Flock.Firewall.Frame.run c {}
        (schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr j)
      if r.2 then {} else
        let r' := _root_.Flock.Firewall.Frame.run c r.1
          [.coins (.ok (cr.getD j #[])),
            .round (sched.getD (j + 1) (.empty, 0)).2 (sched.getD (j + 1) (.empty, 0)).1
              (.sched (sched.getD (j + 1) (.empty, 0)).1)]
        if r'.2 then {} else r'.1.m := by
  obtain ⟨u, rfl⟩ := schedSoFar_snoc t rep dom xs hi cr a hs hs'
  have hj' : j + 1 < (sched ++ u).size := by rw [Array.size_append]; omega
  rw [repState_at c t rep dom xs hi cr0 (cr.push a) ns (j + 1) hal hs' hj']
  have hcg : schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) (sched ++ u) (cr.push a)
      (j + 1) = schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr (j + 1) := by
    refine schedReqs_congr _ (fun k hk => ?_) (fun k hk => ?_)
    · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_append_left (show k < sched.size by omega)]
    · simp only [Array.getD_eq_getD_getElem?, Array.getElem?_push, show k ≠ cr.size by omega, ↓reduceIte]
  dsimp only
  rw [hcg, schedReqs_succ, run_append]
  generalize _root_.Flock.Firewall.Frame.run c {}
    (schedReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr j) = R
  obtain ⟨S, b⟩ := R
  cases b <;> rfl

end Loop

end Claims.ZKServer
