import Proofs.Flock.C3.Reps.KeysInW

/-!
# Ligerito's names in w: `wKey` at its sumcheck messages, out-of-domain values and final `yr`

`ligRounds` builds Ligerito's rounds in a loop; `ligRounds_eq` writes it as folds of its two pushes (`ligF1`, a message;
`ligF2`, an out-of-domain value with a message) and its level loop (`ligLevels`, a middle level `ligMid` or the last
`ligLast`). Tracking the message and out-of-domain counters through them, `ligRounds_has` finds in its rounds every
message `j` below `k₀ + 4`, and below `k₀ + 4 + 7n + 4` at `n + 1` levels after the first, every out-of-domain value
below `3`, and below `3 + 2n` there, and there the final `yr`'s `2^(ext+1)` entries. Those rounds are w's
(`mem_wNames_of_lig`), so these keys pass `wKey` (`ligKeys_wKey`).
-/

namespace FlockC3.Reps

open Flock.RecAlgebra (Key msgNames ligRounds)

/-- `x` is an element of one of the rounds `out`. -/
def InRound (out : Array (Array Key)) (x : Key) : Prop := ∃ r ∈ out, x ∈ r

theorem InRound.push {out : Array (Array Key)} {r : Array Key} {x : Key} (h : InRound out x) :
    InRound (out.push r) x :=
  let ⟨r', h1, h2⟩ := h
  ⟨r', Array.mem_push.2 (Or.inl h1), h2⟩

theorem InRound.last {out : Array (Array Key)} {r : Array Key} {x : Key} (h : x ∈ r) : InRound (out.push r) x :=
  ⟨r, Array.mem_push.2 (Or.inr rfl), h⟩

theorem msg_mem_msgNames {j a b : ℕ} (ha : a < 2) (hb : b < 2) : Key.msg j a b ∈ msgNames j := by
  match a, b, ha, hb with
  | 0, 0, _, _ => simp [msgNames]
  | 0, 1, _, _ => simp [msgNames]
  | 1, 0, _, _ => simp [msgNames]
  | 1, 1, _, _ => simp [msgNames]

/-- `ligRounds`' push of a message, at the message counter. -/
def ligF1 (b : ℕ × Array (Array Key)) (_ : ℕ) : ℕ × Array (Array Key) :=
  (b.fst + 1, b.snd.push (msgNames b.fst))

/-- `ligRounds`' push of an out-of-domain value with a message, at the message and out-of-domain counters. -/
def ligF2 (b : ℕ × ℕ × Array (Array Key)) (_ : ℕ) : ℕ × ℕ × Array (Array Key) :=
  (b.fst + 1, b.snd.fst + 1, b.snd.snd.push (#[Key.ood b.snd.fst] ++ msgNames b.fst))

/-- `ligRounds` before its level loop. -/
def lig0 (sch : Flock.Schedule) : ℕ × ℕ × Array (Array Key) :=
  let a := List.foldl ligF2 ((List.foldl ligF1 (0, #[#[Key.T], #[Key.ood 0]]) (List.range' 0 (sch.k0 + 1))).fst, 1,
    (List.foldl ligF1 (0, #[#[Key.T], #[Key.ood 0]]) (List.range' 0 (sch.k0 + 1))).snd) (List.range' 0 2)
  (a.fst + 1, a.snd.fst, a.snd.snd.push (msgNames a.fst))

/-- A level of `ligRounds`' loop but the last. -/
def ligMid (s : ℕ × ℕ × Array (Array Key)) : ℕ × ℕ × Array (Array Key) :=
  let a := List.foldl ligF2 ((List.foldl ligF1 (s.fst, s.snd.snd) (List.range' 0 4)).fst, s.snd.fst,
    (List.foldl ligF1 (s.fst, s.snd.snd) (List.range' 0 4)).snd) (List.range' 0 2)
  (a.fst + 1, a.snd.fst, a.snd.snd.push (msgNames a.fst))

/-- The last level of `ligRounds`' loop, with the final `yr`. -/
def ligLast (ext : ℕ) (s : ℕ × ℕ × Array (Array Key)) : ℕ × ℕ × Array (Array Key) :=
  let a := List.foldl ligF1 (s.fst, s.snd.snd) (List.range' 0 3)
  (a.fst + 1, s.snd.fst, a.snd.push (msgNames a.fst ++ Array.map Key.yr (Array.range (2 <<< ext))))

/-- `ligRounds`' level loop over the levels `l`, of `R` after the first: `G1` at each but the last, `G2` there. -/
def ligLevels (R : ℕ) (G1 G2 : ℕ × ℕ × Array (Array Key) → ℕ × ℕ × Array (Array Key)) (l : List ℕ)
    (init : ℕ × ℕ × Array (Array Key)) : ℕ × ℕ × Array (Array Key) :=
  Id.run (forIn l init fun i s => if i + 1 < R then pure (ForInStep.yield (G1 s)) else pure (ForInStep.yield (G2 s)))

theorem ligRounds_eq (sch : Flock.Schedule) : ligRounds sch =
    (ligLevels (sch.levels.size - 1) ligMid (ligLast sch.ext) (List.range' 0 (sch.levels.size - 1))
      (lig0 sch)).2.2 := by
  unfold ligRounds
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, List.forIn_pure_yield_eq_foldl,
    bind_pure_comp, map_pure, pure_bind, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  rfl

theorem ligF1_foldl (l : List ℕ) (mi : ℕ) (out : Array (Array Key)) :
    (l.foldl ligF1 (mi, out)).1 = mi + l.length ∧ (∀ x, InRound out x → InRound (l.foldl ligF1 (mi, out)).2 x) ∧
      ∀ t < l.length, ∀ a < 2, ∀ b < 2, InRound (l.foldl ligF1 (mi, out)).2 (.msg (mi + t) a b) := by
  induction l generalizing mi out with
  | nil => exact ⟨rfl, fun _ h => h, fun t ht => absurd ht (Nat.not_lt_zero _)⟩
  | cons _ l ih =>
    obtain ⟨h1, h2, h3⟩ := ih (mi + 1) (out.push (msgNames mi))
    simp only [List.foldl_cons, List.length_cons, ligF1]
    refine ⟨by rw [h1]; omega, fun x hx => h2 x hx.push, fun t ht a ha b hb => ?_⟩
    rcases t with _ | t
    · exact h2 _ (InRound.last (msg_mem_msgNames ha hb))
    · have := h3 t (by omega) a ha b hb
      rwa [show mi + 1 + t = mi + (t + 1) by omega] at this

theorem ligF2_foldl (l : List ℕ) (mi oi : ℕ) (out : Array (Array Key)) :
    (l.foldl ligF2 (mi, oi, out)).1 = mi + l.length ∧ (l.foldl ligF2 (mi, oi, out)).2.1 = oi + l.length ∧
      (∀ x, InRound out x → InRound (l.foldl ligF2 (mi, oi, out)).2.2 x) ∧
      (∀ t < l.length, ∀ a < 2, ∀ b < 2, InRound (l.foldl ligF2 (mi, oi, out)).2.2 (.msg (mi + t) a b)) ∧
      ∀ t < l.length, InRound (l.foldl ligF2 (mi, oi, out)).2.2 (.ood (oi + t)) := by
  induction l generalizing mi oi out with
  | nil =>
    exact ⟨rfl, rfl, fun _ h => h, fun t ht => absurd ht (Nat.not_lt_zero _),
      fun t ht => absurd ht (Nat.not_lt_zero _)⟩
  | cons _ l ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih (mi + 1) (oi + 1) (out.push (#[Key.ood oi] ++ msgNames mi))
    simp only [List.foldl_cons, List.length_cons, ligF2]
    refine ⟨by rw [h1]; omega, by rw [h2]; omega, fun x hx => h3 x hx.push, fun t ht a ha b hb => ?_,
      fun t ht => ?_⟩
    · rcases t with _ | t
      · exact h3 _ (InRound.last (Array.mem_append_right _ (msg_mem_msgNames ha hb)))
      · have := h4 t (by omega) a ha b hb
        rwa [show mi + 1 + t = mi + (t + 1) by omega] at this
    · rcases t with _ | t
      · exact h3 _ (InRound.last (Array.mem_append_left _ (by simp)))
      · have := h5 t (by omega)
        rwa [show oi + 1 + t = oi + (t + 1) by omega] at this

theorem ligF1_range (n mi : ℕ) (out : Array (Array Key)) :
    ((List.range' 0 n).foldl ligF1 (mi, out)).1 = mi + n ∧
      (∀ x, InRound out x → InRound ((List.range' 0 n).foldl ligF1 (mi, out)).2 x) ∧
      ∀ t < n, ∀ a < 2, ∀ b < 2, InRound ((List.range' 0 n).foldl ligF1 (mi, out)).2 (.msg (mi + t) a b) := by
  have h := ligF1_foldl (List.range' 0 n) mi out
  rw [List.length_range'] at h
  exact h

theorem ligF2_range (n mi oi : ℕ) (out : Array (Array Key)) :
    ((List.range' 0 n).foldl ligF2 (mi, oi, out)).1 = mi + n ∧
      ((List.range' 0 n).foldl ligF2 (mi, oi, out)).2.1 = oi + n ∧
      (∀ x, InRound out x → InRound ((List.range' 0 n).foldl ligF2 (mi, oi, out)).2.2 x) ∧
      (∀ t < n, ∀ a < 2, ∀ b < 2,
        InRound ((List.range' 0 n).foldl ligF2 (mi, oi, out)).2.2 (.msg (mi + t) a b)) ∧
      ∀ t < n, InRound ((List.range' 0 n).foldl ligF2 (mi, oi, out)).2.2 (.ood (oi + t)) := by
  have h := ligF2_foldl (List.range' 0 n) mi oi out
  rw [List.length_range'] at h
  exact h

theorem lig0_has (sch : Flock.Schedule) : (lig0 sch).1 = sch.k0 + 4 ∧ (lig0 sch).2.1 = 3 ∧
    (∀ j < sch.k0 + 4, ∀ a < 2, ∀ b < 2, InRound (lig0 sch).2.2 (.msg j a b)) ∧
    ∀ j < 3, InRound (lig0 sch).2.2 (.ood j) := by
  obtain ⟨a1, a2, a3⟩ := ligF1_range (sch.k0 + 1) 0 #[#[Key.T], #[Key.ood 0]]
  simp only [lig0]
  generalize List.foldl ligF1 (0, #[#[Key.T], #[Key.ood 0]]) (List.range' 0 (sch.k0 + 1)) = A at a1 a2 a3 ⊢
  obtain ⟨b1, b2, b3, b4, b5⟩ := ligF2_range 2 A.fst 1 A.snd
  generalize List.foldl ligF2 (A.fst, 1, A.snd) (List.range' 0 2) = B at b1 b2 b3 b4 b5 ⊢
  refine ⟨by omega, by omega, fun j hj a ha b hb => ?_, fun j hj => ?_⟩
  · by_cases h1 : j < sch.k0 + 1
    · exact (b3 _ (by simpa using a3 j h1 a ha b hb)).push
    · by_cases h2 : j < sch.k0 + 3
      · have := b4 (j - (sch.k0 + 1)) (by omega) a ha b hb
        rw [a1, show 0 + (sch.k0 + 1) + (j - (sch.k0 + 1)) = j by omega] at this
        exact this.push
      · rw [show j = B.fst by omega]
        exact InRound.last (msg_mem_msgNames ha hb)
  · rcases j with _ | j
    · exact (b3 _ (a2 _ ⟨#[Key.ood 0], by simp, by simp⟩)).push
    · have := b5 j (by omega)
      rw [Nat.add_comm] at this
      exact this.push

theorem ligMid_has (s : ℕ × ℕ × Array (Array Key)) : (ligMid s).1 = s.1 + 7 ∧ (ligMid s).2.1 = s.2.1 + 2 ∧
    (∀ x, InRound s.2.2 x → InRound (ligMid s).2.2 x) ∧
    (∀ t < 7, ∀ a < 2, ∀ b < 2, InRound (ligMid s).2.2 (.msg (s.1 + t) a b)) ∧
    ∀ t < 2, InRound (ligMid s).2.2 (.ood (s.2.1 + t)) := by
  obtain ⟨a1, a2, a3⟩ := ligF1_range 4 s.fst s.snd.snd
  simp only [ligMid]
  generalize List.foldl ligF1 (s.fst, s.snd.snd) (List.range' 0 4) = A at a1 a2 a3 ⊢
  obtain ⟨b1, b2, b3, b4, b5⟩ := ligF2_range 2 A.fst s.snd.fst A.snd
  generalize List.foldl ligF2 (A.fst, s.snd.fst, A.snd) (List.range' 0 2) = B at b1 b2 b3 b4 b5 ⊢
  refine ⟨by omega, by omega, fun x hx => (b3 x (a2 x hx)).push, fun t ht a ha b hb => ?_,
    fun t ht => (b5 t ht).push⟩
  by_cases h1 : t < 4
  · exact (b3 _ (a3 t h1 a ha b hb)).push
  · by_cases h2 : t < 6
    · have := b4 (t - 4) (by omega) a ha b hb
      rw [a1, show s.fst + 4 + (t - 4) = s.fst + t by omega] at this
      exact this.push
    · rw [show s.fst + t = B.fst by omega]
      exact InRound.last (msg_mem_msgNames ha hb)

theorem ligLast_has (ext : ℕ) (s : ℕ × ℕ × Array (Array Key)) :
    (∀ x, InRound s.2.2 x → InRound (ligLast ext s).2.2 x) ∧
    (∀ t < 4, ∀ a < 2, ∀ b < 2, InRound (ligLast ext s).2.2 (.msg (s.1 + t) a b)) ∧
    ∀ x, (∃ j < 2 <<< ext, x = .yr j) → InRound (ligLast ext s).2.2 x := by
  obtain ⟨a1, a2, a3⟩ := ligF1_range 3 s.fst s.snd.snd
  simp only [ligLast]
  generalize List.foldl ligF1 (s.fst, s.snd.snd) (List.range' 0 3) = A at a1 a2 a3 ⊢
  refine ⟨fun x hx => (a2 x hx).push, fun t ht a ha b hb => ?_, ?_⟩
  · by_cases h1 : t < 3
    · exact (a3 t h1 a ha b hb).push
    · rw [show s.fst + t = A.fst by omega]
      exact InRound.last (Array.mem_append_left _ (msg_mem_msgNames ha hb))
  · rintro _ ⟨j, hj, rfl⟩
    exact InRound.last (Array.mem_append_right _ (Array.mem_map.2 ⟨j, Array.mem_range.2 hj, rfl⟩))

/-- **The level loop's rounds**, from level `s` of `R = s + n + 1`: `init`'s, its next `7n + 4` messages and `2n`
out-of-domain values, and the last level's `P`. -/
theorem ligLevels_has {R : ℕ} {P : Key → Prop}
    {G1 G2 : ℕ × ℕ × Array (Array Key) → ℕ × ℕ × Array (Array Key)}
    (h1 : ∀ s, (G1 s).1 = s.1 + 7 ∧ (G1 s).2.1 = s.2.1 + 2 ∧ (∀ x, InRound s.2.2 x → InRound (G1 s).2.2 x) ∧
      (∀ t < 7, ∀ a < 2, ∀ b < 2, InRound (G1 s).2.2 (.msg (s.1 + t) a b)) ∧
      ∀ t < 2, InRound (G1 s).2.2 (.ood (s.2.1 + t)))
    (h2 : ∀ s, (∀ x, InRound s.2.2 x → InRound (G2 s).2.2 x) ∧
      (∀ t < 4, ∀ a < 2, ∀ b < 2, InRound (G2 s).2.2 (.msg (s.1 + t) a b)) ∧
      ∀ x, P x → InRound (G2 s).2.2 x)
    (n : ℕ) : ∀ (s : ℕ) (init : ℕ × ℕ × Array (Array Key)), s + n + 1 = R →
      (∀ x, InRound init.2.2 x → InRound (ligLevels R G1 G2 (List.range' s (n + 1)) init).2.2 x) ∧
      (∀ t < 7 * n + 4, ∀ a < 2, ∀ b < 2,
        InRound (ligLevels R G1 G2 (List.range' s (n + 1)) init).2.2 (.msg (init.1 + t) a b)) ∧
      (∀ t < 2 * n, InRound (ligLevels R G1 G2 (List.range' s (n + 1)) init).2.2 (.ood (init.2.1 + t))) ∧
      ∀ x, P x → InRound (ligLevels R G1 G2 (List.range' s (n + 1)) init).2.2 x := by
  induction n with
  | zero =>
    intro s init hs
    have hn : ¬ s + 1 < R := by omega
    simp only [ligLevels, List.range'_succ, List.range'_zero, List.forIn_cons, List.forIn_nil, hn, ↓reduceIte,
      pure_bind]
    obtain ⟨m, g, y⟩ := h2 init
    exact ⟨m, fun t ht => g t (by omega), fun t ht => absurd ht (by omega), y⟩
  | succ n ih =>
    intro s init hs
    have hn : s + 1 < R := by omega
    unfold ligLevels
    rw [List.range'_succ, List.forIn_cons]
    simp only [hn, ↓reduceIte, pure_bind]
    obtain ⟨c1, c2, m1, g1, o1⟩ := h1 init
    obtain ⟨m, g, o, y⟩ := ih (s + 1) (G1 init) (by omega)
    refine ⟨fun x hx => m x (m1 x hx), fun t ht a ha b hb => ?_, fun t ht => ?_, y⟩
    · by_cases h7 : t < 7
      · exact m _ (g1 t h7 a ha b hb)
      · have := g (t - 7) (by omega) a ha b hb
        rwa [c1, show init.1 + 7 + (t - 7) = init.1 + t by omega] at this
    · by_cases h7 : t < 2
      · exact m _ (o1 t h7)
      · have := o (t - 2) (by omega)
        rwa [c2, show init.2.1 + 2 + (t - 2) = init.2.1 + t by omega] at this

/-- **Ligerito's rounds hold its messages, out-of-domain values and final `yr`** below the schedule's counts. -/
theorem ligRounds_has (sch : Flock.Schedule) :
    (∀ j < sch.k0 + 4, ∀ a < 2, ∀ b < 2, InRound (ligRounds sch) (.msg j a b)) ∧
    (∀ j < 3, InRound (ligRounds sch) (.ood j)) ∧
    ∀ n, sch.levels.size - 1 = n + 1 →
      (∀ j < sch.k0 + 4 + (7 * n + 4), ∀ a < 2, ∀ b < 2, InRound (ligRounds sch) (.msg j a b)) ∧
      (∀ j < 3 + 2 * n, InRound (ligRounds sch) (.ood j)) ∧
      ∀ j < 2 <<< sch.ext, InRound (ligRounds sch) (.yr j) := by
  rw [ligRounds_eq]
  obtain ⟨i1, i2, i3, i4⟩ := lig0_has sch
  generalize sch.levels.size - 1 = R
  rcases R with _ | n
  · simp only [ligLevels, List.range'_zero, List.forIn_nil]
    exact ⟨i3, i4, fun _ h => absurd h (by omega)⟩
  · obtain ⟨m, g, o, y⟩ := ligLevels_has (R := n + 1) (P := fun x => ∃ j < 2 <<< sch.ext, x = .yr j) ligMid_has
      (ligLast_has sch.ext) n 0 (lig0 sch) (by omega)
    refine ⟨fun j hj a ha b hb => m _ (i3 j hj a ha b hb), fun j hj => m _ (i4 j hj), fun n' hn => ?_⟩
    obtain rfl : n = n' := by omega
    refine ⟨fun j hj a ha b hb => ?_, fun j hj => ?_, fun j hj => y _ ⟨j, hj, rfl⟩⟩
    · by_cases h : j < sch.k0 + 4
      · exact m _ (i3 j h a ha b hb)
      · have := g (j - (sch.k0 + 4)) (by omega) a ha b hb
        rwa [i1, show sch.k0 + 4 + (j - (sch.k0 + 4)) = j by omega] at this
    · by_cases h : j < 3
      · exact m _ (i4 j h)
      · have := o (j - 3) (by omega)
        rwa [i2, show 3 + (j - 3) = j by omega] at this

theorem mem_wNames_of_lig {sh : Flock.Shape} {x : Key} (h : InRound (ligRounds sh.sched) x) :
    x ∈ Flock.RecAlgebra.wNames sh :=
  let ⟨r, hr, hx⟩ := h
  mem_wNames (by unfold Flock.RecAlgebra.rounds; exact Array.mem_append_right _ hr) hx

/-- **Ligerito's keys pass `wKey`**: its messages' and out-of-domain values' below the schedule's counts, and the final
`yr`'s at a level after the first. -/
theorem ligKeys_wKey (sh : Flock.Shape) :
    (∀ j < sh.sched.k0 + 4, ∀ a < 2, ∀ b < 2, wKey sh (.msg j a b) = true) ∧
    (∀ j < 3, wKey sh (.ood j) = true) ∧
    ∀ n, sh.sched.levels.size - 1 = n + 1 →
      (∀ j < sh.sched.k0 + 4 + (7 * n + 4), ∀ a < 2, ∀ b < 2, wKey sh (.msg j a b) = true) ∧
      (∀ j < 3 + 2 * n, wKey sh (.ood j) = true) ∧ ∀ j < 2 ^ (sh.sched.ext + 1), wKey sh (.yr j) = true := by
  have hw {x : Key} (h : InRound (ligRounds sh.sched) x) (h1 : ∀ k u, x ≠ .rsT k u) (h2 : ∀ l, x ≠ .comb l) :
      wKey sh x = true :=
    wKey_of_mem (mem_wNames_of_lig h) h1 h2
  obtain ⟨m, o, l⟩ := ligRounds_has sh.sched
  refine ⟨fun j hj a ha b hb => hw (m j hj a ha b hb) (fun _ _ e => (by cases e)) (fun _ e => (by cases e)),
    fun j hj => hw (o j hj) (fun _ _ e => (by cases e)) (fun _ e => (by cases e)), fun n hn => ?_⟩
  obtain ⟨m', o', y⟩ := l n hn
  refine ⟨fun j hj a ha b hb => hw (m' j hj a ha b hb) (fun _ _ e => (by cases e)) (fun _ e => (by cases e)),
    fun j hj => hw (o' j hj) (fun _ _ e => (by cases e)) (fun _ e => (by cases e)), fun j hj => ?_⟩
  refine hw (y j ?_) (fun _ _ e => (by cases e)) (fun _ e => (by cases e))
  rwa [Nat.shiftLeft_eq, ← Nat.pow_succ']

end FlockC3.Reps
