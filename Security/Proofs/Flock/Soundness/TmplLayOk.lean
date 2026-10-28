import Proofs.Flock.Soundness.TmplPosMap

/-!
# `layoutsM`'s layouts are in order

`layDefM` lays a Definition's calls out in order, each run of its own columns between them. `layDefM_ok` follows its
loops: the runs it pushes are nonempty, ascending and apart from the calls, and with them cover the Definition's columns
(`RunsOk`). Each call's copies sit side by side, and each run of lanes is a power of two that holds its copies (`OutOk`).
`layouts_defPos` reads that as `DefPos` for each Definition `layoutsM` lays out, and `layouts_lanes` as `LayOk.lanes`.
Each depends on what `parse` checks of each call: its copies inside its caller's columns, and its callee an earlier
Definition.
-/

open Flock Flock.Tmpl

namespace FlockSoundness.TmplFold

theorem ebo {α β ε : Type} {x : Except ε α} {f : α → Except ε β} {v : β}
    (h : (x >>= f) = .ok v) : ∃ a, x = .ok a ∧ f a = .ok v := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- A loop in `Except` that finished keeps an invariant indexed by how many elements it walked. -/
theorem forIn_idx_inv {α σ ε : Type} (l : List α) (I : Nat → σ → Prop) {f : α → σ → Except ε (ForInStep σ)}
    (hf : ∀ k (hk : k < l.length) s t, I k s → f l[k] s = .ok t → ∃ s', t = .yield s' ∧ I (k + 1) s') :
    ∀ (m : List α) (k : Nat) (s v : σ), k ≤ l.length → l.drop k = m → I k s → forIn m s f = .ok v → I l.length v
  | [], k, s, v, hkl, hm, hs, h => by
    rw [List.forIn_nil] at h
    cases h
    have : l.length ≤ k := by simpa using hm
    exact (show k = l.length by omega) ▸ hs
  | a :: m, k, s, v, hkl, hm, hs, h => by
    rw [List.forIn_cons] at h
    obtain ⟨t, ht, h⟩ := ebo h
    have hk : k < l.length := Nat.lt_of_not_le fun hne => by
      rw [List.drop_eq_nil_of_le hne] at hm
      cases hm
    have ha : l[k] = a := by
      have := congrArg List.head? hm
      rw [List.head?_drop, List.getElem?_eq_getElem hk] at this
      simpa using this
    obtain ⟨s', rfl, hs'⟩ := hf k hk s t hs (ha ▸ ht)
    refine forIn_idx_inv l I hf m (k + 1) s' v hk ?_ hs' h
    rw [← List.drop_drop, hm]
    rfl

/-- Where call `i`'s copies end. -/
def callEnd (t : Tmpl) (D : Def) (i : Nat) : Nat :=
  (D.calls.getD i default).base + (D.calls.getD i default).count * (t.defs.getD (D.calls.getD i default).def_ default).cols

/-- Runs laid out so far, below `X`: nonempty, ascending, apart from the first `k` calls, and with them covering `[0, X)`. -/
structure RunsOk (t : Tmpl) (D : Def) (runs : Array (Nat × Nat × Nat)) (X k : Nat) : Prop where
  lt : ∀ r < runs.size, (runs.getD r default).1 < (runs.getD r default).2.1
  le : ∀ r < runs.size, (runs.getD r default).2.1 ≤ X
  asc : ∀ r r', r < r' → r' < runs.size → (runs.getD r default).2.1 ≤ (runs.getD r' default).1
  cover : ∀ y < X, (∃ r < runs.size, (runs.getD r default).1 ≤ y ∧ y < (runs.getD r default).2.1) ∨ (∃ i < k, (D.calls.getD i default).base ≤ y ∧ y < callEnd t D i)
  apart : ∀ r < runs.size, ∀ i < k, (runs.getD r default).2.1 ≤ (D.calls.getD i default).base ∨ callEnd t D i ≤ (runs.getD r default).1
  size : runs.size ≤ X

theorem getD_push_lt {α : Type} (a : Array α) (v d : α) {i : Nat} (h : i < a.size) : (a.push v).getD i d = a.getD i d := by
  simp [Array.getD_eq_getD_getElem?, Array.getElem?_push, h, Nat.ne_of_lt h]

theorem getD_push_eq {α : Type} (a : Array α) (v d : α) : (a.push v).getD a.size d = v := by
  simp [Array.getD_eq_getD_getElem?]

theorem runsOk_push {t : Tmpl} {D : Def} {runs : Array (Nat × Nat × Nat)} {X k b p : Nat} (h : RunsOk t D runs X k)
    (hb : X < b) (hk : ∀ i < k, callEnd t D i ≤ X) : RunsOk t D (runs.push (X, b, p)) b k := by
  have g : ∀ r < (runs.push (X, b, p)).size, ∀ {P : Nat × Nat × Nat → Prop}, (r < runs.size → P (runs.getD r default)) →
      (r = runs.size → P (X, b, p)) → P ((runs.push (X, b, p)).getD r default) := by
    intro r hr P h1 h2
    rw [Array.size_push] at hr
    rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hr) with hr | rfl
    · rw [getD_push_lt _ _ _ hr]; exact h1 hr
    · rw [getD_push_eq]; exact h2 rfl
  refine ⟨fun r hr => g r hr (P := fun q => q.1 < q.2.1) (fun hr => h.lt r hr) (fun _ => hb),
    fun r hr => g r hr (P := fun q => q.2.1 ≤ b) (fun hr => by have := h.le r hr; omega) (fun _ => Nat.le_refl _),
    fun r r' hrr hr' => ?_, fun y hy => ?_,
    fun r hr i hi => g r hr (P := fun q => q.2.1 ≤ (D.calls.getD i default).base ∨ callEnd t D i ≤ q.1) (fun hr => h.apart r hr i hi) (fun _ => Or.inr (hk i hi)), by rw [Array.size_push]; have := h.size; omega⟩
  · have hr : r < runs.size := by rw [Array.size_push] at hr'; omega
    rw [getD_push_lt _ _ _ hr]
    exact g r' hr' (P := fun q => (runs.getD r default).2.1 ≤ q.1) (fun hr' => h.asc r r' hrr hr') (fun _ => h.le r hr)
  · by_cases hyX : y < X
    · rcases h.cover y hyX with ⟨r, hr, h1, h2⟩ | hc
      · exact Or.inl ⟨r, by rw [Array.size_push]; omega, by rw [getD_push_lt _ _ _ hr]; exact ⟨h1, h2⟩⟩
      · exact Or.inr hc
    · exact Or.inl ⟨runs.size, by rw [Array.size_push]; omega, by rw [getD_push_eq]; exact ⟨by dsimp only; omega, hy⟩⟩

theorem runsOk_lift {t : Tmpl} {D : Def} {runs : Array (Nat × Nat × Nat)} {X k : Nat} (h : RunsOk t D runs X k)
    (hX : X = (D.calls.getD k default).base) : RunsOk t D runs (callEnd t D k) (k + 1) := by
  have hXe : X ≤ callEnd t D k := by unfold callEnd; omega
  refine ⟨h.lt, fun r hr => Nat.le_trans (h.le r hr) hXe, h.asc, fun y hy => ?_, fun r hr i hi => ?_,
    Nat.le_trans h.size hXe⟩
  · by_cases hyX : y < X
    · rcases h.cover y hyX with hr | ⟨i, hi, hc⟩
      · exact Or.inl hr
      · exact Or.inr ⟨i, by omega, hc⟩
    · exact Or.inr ⟨k, by omega, by omega, hy⟩
  · rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hi) with hi | rfl
    · exact h.apart r hr i hi
    · exact Or.inl (by have := h.le r hr; omega)

/-- After the first `k` calls: the runs below `x`, one place per call, calls ascending and side by side, runs of lanes
a power of two that holds their copies. -/
structure OutOk (t : Tmpl) (D : Def) (k x : Nat) (runs : Array (Nat × Nat × Nat)) (places : Array Place) : Prop where
  runs : RunsOk t D runs x k
  size : places.size = k
  ext_le : ∀ i < k, callEnd t D i ≤ x
  asc : ∀ i i', i < i' → i' < k → callEnd t D i ≤ (D.calls.getD i' default).base
  stride : ∀ i < k, 1 < (D.calls.getD i default).count → (D.calls.getD i default).stride = (t.defs.getD (D.calls.getD i default).def_ default).cols
  x_ext : k = 0 ∧ x = 0 ∨ ∃ i < k, x = callEnd t D i
  lanes : ∀ i < k, (places.getD i default).ms ≠ 1 → (places.getD i default).cs = 1 ∧
    2 ^ (places.getD i default).ms.log2 = (places.getD i default).ms ∧ (D.calls.getD i default).count ≤ (places.getD i default).ms

theorem inner_ok {t : Tmpl} {D : Def} {k x lim : Nat} (outs : List (Nat × Nat))
    {s v : Nat × Array (Nat × Nat × Nat) × List Nat × Nat}
    (hI : x ≤ s.2.2.2 ∧ s.2.2.2 ≤ lim ∧ RunsOk t D s.2.1 s.2.2.2 k) (hk : ∀ i < k, callEnd t D i ≤ x)
    (h : (forIn outs s fun x __s =>
      if (decide (__s.snd.snd.snd ≤ x.fst) && decide (x.fst ≤ lim) && !__s.snd.snd.fst.contains x.fst) = true then
        if __s.snd.snd.snd < x.fst then
          pure (ForInStep.yield (up (__s.fst + (x.fst - __s.snd.snd.snd)) x.snd,
            __s.snd.fst.push (__s.snd.snd.snd, x.fst, __s.fst), x.fst :: __s.snd.snd.fst, x.fst))
        else
          pure (ForInStep.yield (up (__s.fst + (x.fst - __s.snd.snd.snd)) x.snd, __s.snd.fst, x.fst :: __s.snd.snd.fst, x.fst))
      else pure (ForInStep.yield (__s.fst, __s.snd.fst, __s.snd.snd.fst, __s.snd.snd.snd)) : V _) = .ok v) :
    x ≤ v.2.2.2 ∧ v.2.2.2 ≤ lim ∧ RunsOk t D v.2.1 v.2.2.2 k := by
  refine forIn_idx_inv outs (fun _ s => x ≤ s.2.2.2 ∧ s.2.2.2 ≤ lim ∧ RunsOk t D s.2.1 s.2.2.2 k) ?_ outs 0 s v
    (Nat.zero_le _) rfl hI h
  intro j hj s u hs hu
  obtain ⟨pos, runs, done, x0⟩ := s
  generalize outs[j] = q at hu
  obtain ⟨p, w⟩ := q
  dsimp only at hs hu
  by_cases hq : (decide (x0 ≤ p) && decide (p ≤ lim) && !done.contains p) = true
  · rw [if_pos hq] at hu
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hq
    by_cases hp : x0 < p
    · rw [if_pos hp] at hu
      cases hu
      exact ⟨_, rfl, by dsimp only; omega, hq.1.2, runsOk_push hs.2.2 hp fun i hi => by have := hk i hi; omega⟩
    · rw [if_neg hp] at hu
      cases hu
      have : p = x0 := by omega
      subst this
      exact ⟨_, rfl, hs⟩
  · rw [if_neg hq] at hu
    cases hu
    exact ⟨_, rfl, hs⟩

theorem count_le_nextPow2 (n : Nat) : n ≤ nextPow2 n := by
  unfold nextPow2 log2ceil
  split
  · rw [Nat.pow_zero]; omega
  · have := Nat.lt_log2_self (n := n - 1)
    omega

theorem pow_log2_nextPow2 (n : Nat) : 2 ^ (nextPow2 n).log2 = nextPow2 n := by
  unfold nextPow2
  rw [Nat.log2_two_pow]

theorem outOk_step {t : Tmpl} {D : Def} {k x : Nat} {R R0 : Array (Nat × Nat × Nat)} {places : Array Place} {c : Call}
    {P : Place} (hs : OutOk t D k x R0 places) (hR : RunsOk t D R c.base k) (hck : D.calls.getD k default = c)
    (hb : x ≤ c.base) (hst : 1 < c.count → c.stride = (t.defs.getD c.def_ default).cols)
    (hP : P.ms ≠ 1 → P.cs = 1 ∧ 2 ^ P.ms.log2 = P.ms ∧ c.count ≤ P.ms) :
    OutOk t D (k + 1) (c.base + c.count * (t.defs.getD c.def_ default).cols) R (places.push P) := by
  have hext : callEnd t D k = c.base + c.count * (t.defs.getD c.def_ default).cols := by unfold callEnd; rw [hck]
  have hR' := runsOk_lift hR (by rw [hck])
  rw [hext] at hR'
  have hle : c.base ≤ callEnd t D k := by rw [hext]; omega
  refine ⟨hR', by rw [Array.size_push, hs.size], fun i hi => ?_, fun i i' hii hi' => ?_, fun i hi => ?_,
    Or.inr ⟨k, Nat.lt_succ_self k, hext.symm⟩, fun i hi => ?_⟩
  · rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hi) with hi | rfl
    · have := hs.ext_le i hi; omega
    · exact Nat.le_of_eq hext
  · rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hi') with hi' | rfl
    · exact hs.asc i i' hii hi'
    · have := hs.ext_le i hii; rw [hck]; omega
  · rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hi) with hi | rfl
    · exact hs.stride i hi
    · rw [hck]; exact hst
  · rcases Nat.lt_or_eq_of_le (Nat.le_of_lt_succ hi) with hi | rfl
    · rw [getD_push_lt _ _ _ (hs.size ▸ hi)]; exact hs.lanes i hi
    · rw [← hs.size, getD_push_eq, hs.size, hck]; exact hP

theorem layDefM_ok {t : Tmpl} {lays : Array Lay} {mode : Mode} {paths : Array Nat} {d : Nat} {L : Lay}
    (h : layDefM t lays mode paths d = .ok L)
    (hin : ∀ i < (t.defs.getD d default).calls.size, callEnd t (t.defs.getD d default) i ≤ (t.defs.getD d default).cols) :
    RunsOk t (t.defs.getD d default) L.runs (t.defs.getD d default).cols (t.defs.getD d default).calls.size ∧
      ∃ x R, OutOk t (t.defs.getD d default) (t.defs.getD d default).calls.size x R L.places := by
  unfold layDefM at h
  obtain ⟨s1, h1, h⟩ := ebo h
  rw [← Array.forIn_toList] at h1
  have key := forIn_idx_inv (t.defs.getD d default).calls.toList
    (fun k (s : Nat × Nat × Nat × Array (Nat × Nat × Nat) × Array Place × List Nat × Bool) =>
      OutOk t (t.defs.getD d default) k s.2.1 s.2.2.2.1 s.2.2.2.2.1) ?_ _ 0 _ s1 (Nat.zero_le _) rfl ?_ h1
  · obtain ⟨pos, x, al, runs, places, done, ident⟩ := s1
    rw [Array.length_toList] at key
    dsimp only at key h
    obtain ⟨⟨pos2, runs2, done2, x02⟩, h2, h⟩ := ebo h
    have hx : x ≤ (t.defs.getD d default).cols := by
      rcases key.x_ext with ⟨-, rfl⟩ | ⟨i, hi, rfl⟩
      · exact Nat.zero_le _
      · exact hin i hi
    have hin2 := inner_ok _ ⟨Nat.le_refl x, hx, key.runs⟩ key.ext_le h2
    dsimp only at hin2 h
    by_cases hx0 : x02 < (t.defs.getD d default).cols
    · rw [if_pos hx0] at h
      cases h
      exact ⟨runsOk_push hin2.2.2 hx0 fun i hi => by have := key.ext_le i hi; omega, x, runs, key⟩
    · rw [if_neg hx0] at h
      cases h
      have : x02 = (t.defs.getD d default).cols := by omega
      exact ⟨this ▸ hin2.2.2, x, runs, key⟩
  · intro k hk s u hs hu
    obtain ⟨pos, x, al, runs, places, done, ident⟩ := s
    dsimp only at hs hu
    generalize hc : (t.defs.getD d default).calls.toList[k] = c at hu
    by_cases hb : c.base < x
    · rw [if_pos hb] at hu
      cases hu
    rw [if_neg hb] at hu
    by_cases hst : (decide (c.count > 1) && c.stride != (t.defs.getD c.def_ default).cols) = true
    · rw [if_pos hst] at hu
      cases hu
    rw [if_neg hst] at hu
    obtain ⟨⟨pos2, runs2, done2, x02⟩, h2, hu⟩ := ebo hu
    dsimp only at hu
    have hin2 := inner_ok _ ⟨Nat.le_refl x, Nat.le_of_not_lt hb, hs.runs⟩ hs.ext_le h2
    dsimp only at hin2
    have hck : (t.defs.getD d default).calls.getD k default = c := by
      have hk' : k < (t.defs.getD d default).calls.size := by simpa using hk
      rw [← hc, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hk', Option.getD_some, Array.getElem_toList]
    have hst' : 1 < c.count → c.stride = (t.defs.getD c.def_ default).cols := by
      intro h1
      simp only [Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq] at hst
      exact Classical.byContradiction fun h => hst ⟨h1, h⟩
    have hR3 : ∃ R, RunsOk t (t.defs.getD d default) R c.base k ∧
        (if x02 < c.base then runs2.push (x02, c.base, pos2) else runs2) = R := by
      by_cases hx0 : x02 < c.base
      · rw [if_pos hx0]
        exact ⟨_, runsOk_push hin2.2.2 hx0 fun i hi => by have := hs.ext_le i hi; omega, rfl⟩
      · rw [if_neg hx0]
        have : x02 = c.base := by omega
        exact ⟨_, this ▸ hin2.2.2, rfl⟩
    obtain ⟨R, hR, hReq⟩ := hR3
    by_cases hx0 : x02 < c.base
    all_goals first | rw [if_pos hx0] at hu hReq | rw [if_neg hx0] at hu hReq
    all_goals rw [hReq] at hu
    all_goals by_cases hroot : (d == t.root) = true
    all_goals first | rw [if_pos hroot] at hu | rw [if_neg hroot] at hu
    all_goals
      split at hu
      · split at hu
        · cases hu
        · cases hu
          exact ⟨_, rfl, outOk_step hs hR hck (Nat.le_of_not_lt hb) hst' fun h => absurd rfl h⟩
      · split at hu
        · cases hu
          exact ⟨_, rfl, outOk_step hs hR hck (Nat.le_of_not_lt hb) hst' fun h => absurd rfl h⟩
        · cases hu
          exact ⟨_, rfl, outOk_step hs hR hck (Nat.le_of_not_lt hb) hst' fun _ =>
            ⟨rfl, pow_log2_nextPow2 _, count_le_nextPow2 _⟩⟩
  · exact ⟨⟨by simp, by simp, by simp, fun y hy => absurd hy (Nat.not_lt_zero _), by simp, Nat.le_refl _⟩, rfl, by simp,
      by simp, by simp, Or.inl ⟨rfl, rfl⟩, by simp⟩


theorem foldlM_lays {t : Tmpl} {mode : Mode} {paths : Array Nat} : ∀ (l : List Nat) (acc res : Array Lay),
    l.foldlM (fun lays d => do let L ← layDefM t lays mode paths d; pure (lays.push L)) acc = .ok res →
      res.size = acc.size + l.length ∧ (∀ i < acc.size, res.getD i default = acc.getD i default) ∧
        ∀ j (hj : j < l.length), ∃ lays0, layDefM t lays0 mode paths l[j] = .ok (res.getD (acc.size + j) default)
  | [], acc, res, h => by
    cases h
    exact ⟨rfl, fun _ _ => rfl, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
  | a :: l, acc, res, h => by
    rw [List.foldlM_cons] at h
    obtain ⟨acc', h1, h⟩ := ebo h
    obtain ⟨L, hL, h1⟩ := ebo h1
    cases h1
    obtain ⟨hs, hp, hj⟩ := foldlM_lays l _ res h
    rw [Array.size_push] at hs hj
    refine ⟨by rw [hs, List.length_cons]; omega, fun i hi => by rw [hp i (by rw [Array.size_push]; omega), getD_push_lt _ _ _ hi],
      fun j hj' => ?_⟩
    rcases j with _ | j
    · refine ⟨acc, ?_⟩
      rw [Nat.add_zero, hp acc.size (by rw [Array.size_push]; omega), getD_push_eq]
      exact hL
    · obtain ⟨lays0, h0⟩ := hj j (by rw [List.length_cons] at hj'; omega)
      refine ⟨lays0, ?_⟩
      rw [show acc.size + (j + 1) = acc.size + 1 + j by omega]
      exact h0

/-- **Each Definition's layout is in order** (`DefPos`), for every layout `layoutsM` returns, while each call's copies sit
inside its caller's columns and call an earlier Definition, and columns and calls number below `2^64`. -/
theorem layouts_defPos {t : Tmpl} {mode : Mode} {lays : Array Lay} (h : layoutsM t mode = .ok lays)
    (hcall : ∀ d < t.defs.size, ∀ i < (t.defs.getD d default).calls.size, ((t.defs.getD d default).calls.getD i default).def_ < d ∧ ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count * (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols ≤ (t.defs.getD d default).cols)
    (hsmall : ∀ d < t.defs.size, (t.defs.getD d default).cols < 2 ^ 64 ∧ (t.defs.getD d default).calls.size < 2 ^ 64) :
    ∀ d < t.defs.size, DefPos t lays d := by
  intro d hd
  unfold layoutsM at h
  obtain ⟨-, -, hj⟩ := foldlM_lays (List.range t.defs.size) #[] lays h
  obtain ⟨lays0, h0⟩ := hj d (by rw [List.length_range]; exact hd)
  rw [List.getElem_range, Array.size_empty, Nat.zero_add] at h0
  obtain ⟨hR, x, R, hO⟩ := layDefM_ok h0 fun i hi => (hcall d hd i hi).2
  have hs := hsmall d hd
  exact ⟨hR.lt, hR.asc, hR.le, fun i i' h h' => hO.asc i i' h h', fun i hi => (hcall d hd i hi).2, hO.stride,
    hR.cover, hR.apart, by have := hR.size; omega, hs.2, fun i hi => (hcall d hd i hi).1⟩

/-- Every run of lanes `layoutsM` places is a power of two that holds its copies (`LayOk.lanes`). -/
theorem layouts_lanes {t : Tmpl} {mode : Mode} {lays : Array Lay} (h : layoutsM t mode = .ok lays)
    (hcall : ∀ d < t.defs.size, ∀ i < (t.defs.getD d default).calls.size, ((t.defs.getD d default).calls.getD i default).base + ((t.defs.getD d default).calls.getD i default).count * (t.defs.getD ((t.defs.getD d default).calls.getD i default).def_ default).cols ≤ (t.defs.getD d default).cols) :
    ∀ d i, i < (t.defs.getD d default).calls.size → ((lays.getD d default).places.getD i default).ms ≠ 1 →
      ((lays.getD d default).places.getD i default).cs = 1 ∧
        2 ^ ((lays.getD d default).places.getD i default).ms.log2 = ((lays.getD d default).places.getD i default).ms ∧
        ((t.defs.getD d default).calls.getD i default).count ≤ ((lays.getD d default).places.getD i default).ms := by
  intro d i hi
  have hd : d < t.defs.size := Nat.lt_of_not_le fun hd => by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none hd] at hi
    exact Nat.not_lt_zero _ hi
  unfold layoutsM at h
  obtain ⟨-, -, hj⟩ := foldlM_lays (List.range t.defs.size) #[] lays h
  obtain ⟨lays0, h0⟩ := hj d (by rw [List.length_range]; exact hd)
  rw [List.getElem_range, Array.size_empty, Nat.zero_add] at h0
  obtain ⟨-, x, R, hO⟩ := layDefM_ok h0 (hcall d hd)
  exact hO.lanes i hi

theorem layoutM_layouts {t : Tmpl} {mode : Mode} {lays : Array Lay} (h : layoutM t mode = .ok lays) :
    layoutsM t mode = .ok lays := by
  unfold layoutM at h
  obtain ⟨lays', h1, h⟩ := ebo h
  dsimp only at h
  split at h
  · cases h
  obtain ⟨u, -, h⟩ := ebo h
  split at h
  · cases h
  cases h
  exact h1

end FlockSoundness.TmplFold
