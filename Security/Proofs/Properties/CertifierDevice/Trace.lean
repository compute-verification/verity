import Proofs.Properties.CertifierDevice.Step

/-!
# The certifier device, proved: over a trace

Every prefix of a trace leaves a `Good` state whose egress tick count is `egressRun` of the prefix's tick views
(`prefix_good`), so event `n` is a live egress tick exactly when it is an `EgressAt`, and its bucket is determined by
`n` (`egressAt_unique`). The rows `ρ` and statuses `σ` of `Records` are the ones the device's ticks compute (`rowOf`,
`statusOf`); the open window's rows are always the `ρ` of its earlier buckets (`rows_inv`).
-/

namespace Properties.CertifierDevice.Proofs

open _root_.CertifierDevice
open NetworkCertifier (Reason Status)

/-! ## Folds -/

theorem outputs_length (st : State) (tr : List Event) : (outputs st tr).length = tr.length := by
  induction tr generalizing st with
  | nil => rfl
  | cons ev tr ih => simp [outputs, ih]

theorem outputs_get (st : State) (tr : List Event) (n : Nat) (h : n < tr.length) :
    (outputs st tr)[n]? = some (step (after st (tr.take n)) tr[n]).2 := by
  induction tr generalizing st n with
  | nil => simp at h
  | cons ev tr ih =>
    cases n with
    | zero => simp [outputs, after]
    | succ n =>
      simp only [outputs, List.getElem?_cons_succ]
      rw [ih _ n (by simpa using h)]
      simp [after, List.take_succ_cons, List.foldl_cons]

theorem outputs_lt {st : State} {tr : List Event} {n : Nat} {acts : List Action}
    (h : (outputs st tr)[n]? = some acts) : n < tr.length := by
  rw [← outputs_length st tr]
  exact (List.getElem?_eq_some_iff.1 h).1

theorem after_take_succ (st : State) (tr : List Event) (n : Nat) (h : n < tr.length) :
    after st (tr.take (n + 1)) = (step (after st (tr.take n)) tr[n]).1 := by
  unfold after
  rw [List.take_succ_eq_append_getElem h, List.foldl_append]
  rfl

/-- One tick view's effect on the egress bucket count. -/
def run (c : Config) (k : Nat) (v : Option (GridKind × Nat)) : Nat :=
  if v.map Prod.fst = some .egress ∧ k < horizon c then k + 1 else k

theorem egressRun_eq (c : Config) (vs : List (Option (GridKind × Nat))) : egressRun c vs = vs.foldl (run c) 0 := rfl

theorem not_live {c : Config} {st : State} {ev : Event}
    (hl : ¬ ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c) :
    run c st.shaper.t (tickView ev) = st.shaper.t := by
  unfold run
  rw [ite_eq_right_iff]
  rintro ⟨h1, h2⟩
  exfalso
  cases ev with
  | tick g now backlog =>
    cases g
    · exact hl ⟨now, backlog, rfl, h2⟩
    · simp [tickView] at h1
  | _ => simp [tickView] at h1

theorem step_good {c : Config} {st : State} (hg : Good c st) (ev : Event) :
    Good c (step st ev).1 ∧ (step st ev).1.shaper.t = run c st.shaper.t (tickView ev) := by
  by_cases hl : ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c
  · obtain ⟨now, backlog, rfl, ht⟩ := hl
    rw [step_egress_fst hg ht]
    obtain ⟨e1, e2, -⟩ := egress_spec hg now backlog
    refine ⟨e1, ?_⟩
    rw [e2]
    simp [run, tickView, ht]
  · obtain ⟨q, -⟩ := step_quiet hg ev hl
    exact ⟨hg.quiet q, by rw [q.2.1.2.1, not_live hl]⟩

theorem after_good {c : Config} {st : State} (hg : Good c st) (tr : List Event) :
    Good c (after st tr) ∧ (after st tr).shaper.t = (tr.map tickView).foldl (run c) st.shaper.t := by
  induction tr generalizing st with
  | nil => exact ⟨hg, rfl⟩
  | cons ev tr ih =>
    obtain ⟨g1, t1⟩ := step_good hg ev
    obtain ⟨g2, t2⟩ := ih g1
    refine ⟨g2, ?_⟩
    show (after (step st ev).1 tr).shaper.t = List.foldl (run c) (run c st.shaper.t (tickView ev)) (tr.map tickView)
    rw [t2, t1]

theorem init_t (c : Config) : (init c).shaper.t = 0 := checkAdvice_t _

/-- The state after the first `n` events. -/
abbrev pre (c : Config) (tr : List Event) (n : Nat) : State := after (init c) (tr.take n)

theorem prefix_good (c : Config) (tr : List Event) (n : Nat) :
    Good c (pre c tr n) ∧ (pre c tr n).shaper.t = egressRun c ((tr.take n).map tickView) := by
  obtain ⟨g, t⟩ := after_good (init_good c) (tr.take n)
  exact ⟨g, by rw [t, init_t]; rfl⟩

theorem pre_succ (c : Config) (tr : List Event) (n : Nat) (h : n < tr.length) :
    pre c tr (n + 1) = (step (pre c tr n) tr[n]).1 := after_take_succ _ _ _ h

theorem t_succ (c : Config) (tr : List Event) (n : Nat) (h : n < tr.length) :
    (pre c tr (n + 1)).shaper.t = run c (pre c tr n).shaper.t (tickView tr[n]) := by
  rw [pre_succ c tr n h]
  exact (step_good (prefix_good c tr n).1 _).2

theorem t_mono (c : Config) (tr : List Event) (n d : Nat) (h : n + d ≤ tr.length) :
    (pre c tr n).shaper.t ≤ (pre c tr (n + d)).shaper.t := by
  induction d with
  | zero => exact le_rfl
  | succ d ih =>
    have h' : n + d < tr.length := by omega
    refine (ih (by omega)).trans ?_
    rw [← Nat.add_assoc, t_succ c tr _ h']
    unfold run
    split <;> omega

/-! ## Egress ticks over a trace -/

theorem egressAt_lt {c : Config} {tr : List Event} {n k backlog : Nat} (h : EgressAt c tr n k backlog) :
    n < tr.length := by
  obtain ⟨⟨now, hn⟩, -, -⟩ := h
  exact (List.getElem?_eq_some_iff.1 hn).1

theorem egressAt_get {c : Config} {tr : List Event} {n k backlog : Nat} (h : EgressAt c tr n k backlog) :
    ∃ now, tr[n]'(egressAt_lt h) = .tick .egress now backlog := by
  obtain ⟨⟨now, hn⟩, -, -⟩ := h
  exact ⟨now, (List.getElem?_eq_some_iff.1 hn).2⟩

theorem egressAt_t {c : Config} {tr : List Event} {n k backlog : Nat} (h : EgressAt c tr n k backlog) :
    (pre c tr n).shaper.t = k := by
  rw [(prefix_good c tr n).2]
  exact h.2.1.symm

theorem egressAt_of {c : Config} {tr : List Event} {n : Nat} (hn : n < tr.length) {now backlog : Nat}
    (he : tr[n] = .tick .egress now backlog) (ht : (pre c tr n).shaper.t < horizon c) :
    EgressAt c tr n (pre c tr n).shaper.t backlog :=
  ⟨⟨now, by rw [List.getElem?_eq_getElem hn, he]⟩, (prefix_good c tr n).2, ht⟩

theorem egressAt_unique {c : Config} {tr : List Event} {n n' k b b' : Nat} (h : EgressAt c tr n k b)
    (h' : EgressAt c tr n' k b') : n = n' := by
  have key : ∀ {n n' b b'}, EgressAt c tr n k b → EgressAt c tr n' k b' → ¬ n < n' := by
    intro n n' b b' h h' hlt
    have h1 := t_succ c tr n (egressAt_lt h)
    obtain ⟨now, he⟩ := (egressAt_get h)
    rw [he, (egressAt_t h)] at h1
    have hk : k < horizon c := h.2.2
    simp only [run, tickView, Option.map_some, hk, and_true, ↓reduceIte] at h1
    have h2 := t_mono c tr (n + 1) (n' - (n + 1)) (by have := (egressAt_lt h'); omega)
    rw [show n + 1 + (n' - (n + 1)) = n' by omega, h1, (egressAt_t h')] at h2
    omega
  rcases Nat.lt_trichotomy n n' with hlt | heq | hgt
  · exact absurd hlt (key h h')
  · exact heq
  · exact absurd hgt (key h' h)

/-- The row an event releases from `st`, if it is an egress tick. -/
def evRow (st : State) : Event → Slots
  | .tick .egress now backlog => (egressMid st now backlog).tick.2.1
  | _ => #[]

/-- The status of the record an event publishes from `st`, if it is an egress tick at a window's last bucket. -/
def evStatus (st : State) : Event → Status
  | .tick .egress now backlog => ((egressMid st now backlog).tick.2.2.map BlockGrid.status).getD .complete
  | _ => .complete

open Classical in
/-- Egress bucket `k`'s row: the one its tick releases. -/
noncomputable def rowOf (c : Config) (tr : List Event) (k : Nat) : Slots :=
  if h : ∃ n b, EgressAt c tr n k b then
    evRow (pre c tr h.choose) (tr[h.choose]'(egressAt_lt h.choose_spec.choose_spec))
  else Array.replicate c.egress.allocation none

open Classical in
/-- Window `w`'s status: the one its last bucket's tick publishes. -/
noncomputable def statusOf (c : Config) (tr : List Event) (w : Nat) : Status :=
  if h : ∃ n b, EgressAt c tr n (w * c.egress.buckets + (c.egress.buckets - 1)) b then
    evStatus (pre c tr h.choose) (tr[h.choose]'(egressAt_lt h.choose_spec.choose_spec))
  else .complete

theorem rowOf_eq {c : Config} {tr : List Event} {n k b : Nat} (h : EgressAt c tr n k b) :
    rowOf c tr k = evRow (pre c tr n) (tr[n]'(egressAt_lt h)) := by
  have hex : ∃ n b, EgressAt c tr n k b := ⟨n, b, h⟩
  unfold rowOf
  split
  · rename_i h'
    have := egressAt_unique h'.choose_spec.choose_spec h
    subst this
    rfl
  · exact absurd hex ‹_›

theorem statusOf_eq {c : Config} {tr : List Event} {n k b : Nat} (h : EgressAt c tr n k b) (w : Nat)
    (hk : k = w * c.egress.buckets + (c.egress.buckets - 1)) :
    statusOf c tr w = evStatus (pre c tr n) (tr[n]'(egressAt_lt h)) := by
  subst hk
  have hex : ∃ n b, EgressAt c tr n (w * c.egress.buckets + (c.egress.buckets - 1)) b := ⟨n, b, h⟩
  unfold statusOf
  split
  · rename_i h'
    have := egressAt_unique h'.choose_spec.choose_spec h
    subst this
    rfl
  · exact absurd hex ‹_›

theorem rowOf_packed (c : Config) (tr : List Event) (k : Nat) : Packed c.egress (rowOf c tr k) := by
  unfold rowOf
  split
  · rename_i h
    have hE := h.choose_spec.choose_spec
    obtain ⟨now, he⟩ := (egressAt_get hE)
    rw [he]
    obtain ⟨-, -, row, status, hp, hr, -⟩ := egress_spec (prefix_good c tr h.choose).1 now
      h.choose_spec.choose
    rw [hr] at hp
    exact hp
  · exact packed_replicate _

/-! ## Arithmetic -/

theorem mod_last {t T : Nat} (h : (t + 1) % T = 0) : 0 < T ∧ t % T = T - 1 ∧ t / T * T + (T - 1) = t := by
  have hT : 0 < T := by
    rcases Nat.eq_zero_or_pos T with h0 | h0
    · rw [h0, Nat.mod_zero] at h; omega
    · exact h0
  have hm : t % T = T - 1 := by
    have hlt := Nat.mod_lt t hT
    by_contra hne
    have h1 : (t % T + 1) % T = t % T + 1 := Nat.mod_eq_of_lt (by omega)
    have h2 : (t + 1) % T = (t % T + 1) % T := by
      rw [Nat.add_mod]
      rcases Nat.lt_or_ge 1 T with h3 | h3
      · rw [Nat.mod_eq_of_lt h3]
      · have : T = 1 := by omega
        subst this; simp
    omega
  refine ⟨hT, hm, ?_⟩
  rw [← hm]
  exact Nat.div_add_mod' t T

theorem push_lt {α : Type _} {a : Array α} {x : α} {j : Nat} (h : j < a.size) : (a.push x)[j]? = a[j]? := by
  rw [Array.getElem?_push]
  simp [Nat.ne_of_lt h]

theorem push_size {α : Type _} (a : Array α) (x : α) : (a.push x)[a.size]? = some x := by simp

theorem getElem!_of {a : Array Slots} {j : Nat} {v : Slots} (h : a[j]? = some v) : a[j]! = v := by
  rw [getElem!_def, h]

/-! ## The open window's rows -/

theorem rows_inv (c : Config) (tr : List Event) (n : Nat) (hn : n ≤ tr.length) :
    ∀ j < (pre c tr n).shaper.t % c.egress.buckets, (pre c tr n).shaper.certifier.rows[j]? =
      some (rowOf c tr ((pre c tr n).shaper.t / c.egress.buckets * c.egress.buckets + j)) := by
  induction n with
  | zero =>
    intro j hj
    have : (pre c tr 0).shaper.t = 0 := by simp [pre, after, init_t]
    rw [this] at hj
    simp at hj
  | succ n ih =>
    have hn' : n < tr.length := by omega
    have ih := ih (by omega)
    have hg := (prefix_good c tr n).1
    rw [pre_succ c tr n hn']
    by_cases hl : ∃ now backlog, tr[n] = .tick .egress now backlog ∧ (pre c tr n).shaper.t < horizon c
    · obtain ⟨now, backlog, he, ht⟩ := hl
      have hE := egressAt_of hn' he ht
      rw [he, step_egress_fst hg ht]
      obtain ⟨-, e2, row, status, -, hr, -, -, -, e6⟩ := egress_spec hg now backlog
      have hrow : rowOf c tr (pre c tr n).shaper.t = row := by
        rw [rowOf_eq hE, hr]
        simp only [he, evRow]
      rw [e2]
      by_cases hc : ((pre c tr n).shaper.t + 1) % c.egress.buckets = 0
      · intro j hj
        rw [hc] at hj
        omega
      · obtain ⟨e6a, -⟩ := e6 hc
        rw [e6a, succ_mod_of_not_mod hc, succ_div_of_not_mod hc]
        intro j hj
        have hsize := hg.shaper.rows
        rcases Nat.lt_or_ge j ((pre c tr n).shaper.t % c.egress.buckets) with hj' | hj'
        · rw [push_lt (by omega)]
          exact ih j hj'
        · have : j = (pre c tr n).shaper.certifier.rows.size := by omega
          rw [this, push_size, hsize, Nat.div_add_mod', hrow]
    · have hl' : ¬ ∃ now backlog, tr[n] = .tick .egress now backlog ∧ (pre c tr n).shaper.t < horizon c := hl
      obtain ⟨q, -⟩ := step_quiet hg tr[n] hl'
      rw [q.2.1.2.1, q.2.1.1.rows]
      exact ih

/-! ## The properties -/

theorem linkWrites_append (a b : List Action) : linkWrites (a ++ b) = linkWrites a ++ linkWrites b :=
  List.filterMap_append

theorem linkWrites_nil {acts : List Action} (h : ∀ a ∈ acts, ∀ b, a ≠ .writeLink b) : linkWrites acts = [] := by
  unfold linkWrites
  rw [List.filterMap_eq_nil_iff]
  intro a ha
  cases a with
  | writeLink b => exact absurd rfl (h _ ha b)
  | _ => rfl

theorem step_writes {c : Config} {st : State} (hg : Good c st) (ev : Event) :
    linkWrites (step st ev).2 = writesAt c st.shaper.t (tickView ev) := by
  by_cases hl : ∃ now backlog, ev = .tick .egress now backlog ∧ st.shaper.t < horizon c
  · obtain ⟨now, backlog, rfl, ht⟩ := hl
    rw [step_egress_snd hg ht]
    obtain ⟨-, -, row, status, hp, -, -, hacts, -⟩ := egress_spec hg now backlog
    rw [hacts, linkWrites_append]
    have hB : linkWrites (if (st.shaper.t + 1) % c.egress.buckets = 0 then
        [.publish ⟨c.egressLink, st.shaper.t / c.egress.buckets, st.shaper.certifier.rows.push row, status⟩
          c.egress] else []) = [] := by
      split <;> rfl
    rw [hB, List.append_nil]
    simp only [tickView, writesAt, ht, true_and]
    split
    · simp [linkWrites, wire_size hp]
    · rfl
  · obtain ⟨-, hacts⟩ := step_quiet hg ev hl
    rw [linkWrites_nil (fun a ha => (hacts a ha).1)]
    cases ev with
    | tick g now backlog =>
      cases g
      · simp only [tickView, writesAt]
        rw [eq_comm, ite_eq_right_iff]
        rintro ⟨h, -⟩
        exact absurd ⟨now, backlog, rfl, h⟩ hl
      · rfl
    | _ => rfl

theorem linkSchedule_proof : LinkSchedule := by
  intro c tr
  apply List.ext_getElem?
  intro n
  rw [List.getElem?_map]
  unfold writeSchedule
  rw [List.getElem?_mapIdx]
  by_cases hn : n < tr.length
  · rw [outputs_get _ _ _ hn, List.getElem?_map, List.getElem?_eq_getElem hn]
    obtain ⟨g, t⟩ := prefix_good c tr n
    simp only [Option.map_some]
    rw [step_writes g, t, List.map_take]
  · rw [List.getElem?_eq_none (by rw [outputs_length]; omega), List.getElem?_eq_none (by simp; omega)]
    rfl

theorem contentIndependent_proof : ContentIndependent := fun c tr tr' h => by
  rw [linkSchedule_proof c tr, linkSchedule_proof c tr', h]

/-- Egress bucket `k`'s tick does `egressActions` of the rows and statuses the device's ticks compute. -/
theorem egress_acts {c : Config} {tr : List Event} {n k backlog : Nat} (hE : EgressAt c tr n k backlog) :
    (outputs (init c) tr)[n]? = some (egressActions c (rowOf c tr) (statusOf c tr) k backlog) := by
  have hn := (egressAt_lt hE)
  obtain ⟨now, he⟩ := (egressAt_get hE)
  have hg := (prefix_good c tr n).1
  have ht := (egressAt_t hE)
  have hk : k < horizon c := hE.2.2
  rw [outputs_get _ _ _ hn]
  change some (step (pre c tr n) tr[n]).2 = _
  rw [he, step_egress_snd hg (ht ▸ hk)]
  obtain ⟨-, -, row, status, -, hr, hs, hacts, -⟩ := egress_spec hg now backlog
  have hrow : rowOf c tr k = row := by rw [rowOf_eq hE, hr]; simp only [he, evRow]
  rw [hacts, ht]
  unfold egressActions
  rw [hrow]
  congr 2
  split
  · rename_i hc
    obtain ⟨hT, hm, hlast⟩ := mod_last hc
    have hst : statusOf c tr (k / c.egress.buckets) = status := by
      rw [statusOf_eq hE _ hlast.symm]
      simp only [he, evStatus]
      rw [hs (ht ▸ hc)]
      rfl
    rw [hst]
    congr 3
    have hinv := rows_inv c tr n hn.le
    rw [ht] at hinv
    have hsize : (pre c tr n).shaper.certifier.rows.size = k % c.egress.buckets := by
      rw [hg.shaper.rows, ht]
    apply Array.ext
    · rw [Array.size_push, Array.size_ofFn, hsize, hm]; omega
    · intro i h1 h2
      rw [Array.getElem_ofFn, Array.getElem_push]
      split
      · rename_i hi
        have := hinv i (by rw [← hsize]; exact hi)
        rw [Array.getElem?_eq_getElem hi] at this
        exact Option.some.inj this
      · rename_i hi
        have : i = c.egress.buckets - 1 := by rw [Array.size_ofFn] at h2; omega
        simp only [this, hlast, hrow]
  · rfl

theorem records_proof : Records := by
  intro c tr
  refine ⟨rowOf c tr, statusOf c tr, rowOf_packed c tr, fun n k backlog hE => egress_acts hE, ?_⟩
  · intro n acts hacts hnot a ha
    have hn := outputs_lt hacts
    rw [outputs_get _ _ _ hn] at hacts
    cases hacts
    have hg := (prefix_good c tr n).1
    have hl : ¬ ∃ now backlog, tr[n] = .tick .egress now backlog ∧ (pre c tr n).shaper.t < horizon c := by
      rintro ⟨now, backlog, he, ht⟩
      exact hnot ⟨_, _, egressAt_of hn he ht⟩
    exact (step_quiet hg tr[n] hl).2 a ha

theorem failClosed_proof : FailClosed := by
  intro c tr n r
  dsimp only
  intro hf m k backlog hnm hE hw
  change (pre c tr n).shaper.certifier.fault = some r at hf
  change k / c.egress.buckets = (pre c tr n).shaper.t / c.egress.buckets at hw
  change ∃ grid : Array Slots, (∀ j, (pre c tr n).shaper.t % c.egress.buckets ≤ j → j < grid.size → _) ∧ _
  have hm := (egressAt_lt hE)
  have inv : ∀ d, n + d ≤ m →
      (pre c tr (n + d)).shaper.certifier.fault = some r ∧
      (pre c tr n).shaper.t ≤ (pre c tr (n + d)).shaper.t ∧
      (pre c tr (n + d)).shaper.t / c.egress.buckets = (pre c tr n).shaper.t / c.egress.buckets ∧
      ∀ j, (pre c tr n).shaper.t % c.egress.buckets ≤ j → j < (pre c tr (n + d)).shaper.t % c.egress.buckets →
        (pre c tr (n + d)).shaper.certifier.rows[j]? = some (Array.replicate c.egress.allocation none) := by
    intro d
    induction d with
    | zero => intro _; exact ⟨hf, le_rfl, rfl, fun j h1 h2 => by rw [Nat.add_zero] at h2; omega⟩
    | succ d ih =>
      intro hd
      obtain ⟨i1, i2, i3, i4⟩ := ih (by omega)
      have hi : n + d < tr.length := by omega
      have hg := (prefix_good c tr (n + d)).1
      have hle : (pre c tr (n + d + 1)).shaper.t ≤ k := by
        have := t_mono c tr (n + d + 1) (m - (n + d + 1)) (by omega)
        rw [show n + d + 1 + (m - (n + d + 1)) = m by omega, (egressAt_t hE)] at this
        exact this
      rw [← Nat.add_assoc, pre_succ c tr _ hi]
      by_cases hl : ∃ now backlog, tr[n + d] = .tick .egress now backlog ∧
          (pre c tr (n + d)).shaper.t < horizon c
      · obtain ⟨now, b, he, ht⟩ := hl
        have hts := t_succ c tr (n + d) hi
        rw [he] at hts
        simp only [run, tickView, Option.map_some, ht, and_true, ↓reduceIte] at hts
        rw [hts] at hle
        rw [he, step_egress_fst hg ht]
        obtain ⟨-, e2, row, status, -, -, -, -, e5, e6⟩ := egress_spec hg now b
        rw [e2]
        by_cases hc : ((pre c tr (n + d)).shaper.t + 1) % c.egress.buckets = 0
        · exfalso
          have h1 := succ_div_of_mod hc
          have h2 := Nat.div_le_div_right (c := c.egress.buckets) hle
          rw [h1, i3, hw] at h2
          omega
        · obtain ⟨e6a, e6b⟩ := e6 hc
          have hrow := (e5 r i1).1
          refine ⟨e6b r i1, by omega, by rw [succ_div_of_not_mod hc, i3], ?_⟩
          rw [succ_mod_of_not_mod hc, e6a]
          intro j h1 h2
          have hsize := hg.shaper.rows
          rcases Nat.lt_or_ge j ((pre c tr (n + d)).shaper.t % c.egress.buckets) with hj | hj
          · rw [push_lt (by omega)]
            exact i4 j h1 hj
          · have : j = (pre c tr (n + d)).shaper.certifier.rows.size := by omega
            rw [this, push_size, hrow]
      · obtain ⟨q, -⟩ := step_quiet hg _ hl
        have hc := q.2.1.1.of_fault i1
        rw [q.2.1.2.1, hc]
        exact ⟨i1, i2, i3, i4⟩
  obtain ⟨i1, -, i3, i4⟩ := inv (m - n) (by omega)
  rw [show n + (m - n) = m by omega] at i1 i3 i4
  have hg := (prefix_good c tr m).1
  have ht := (egressAt_t hE)
  have hk : k < horizon c := hE.2.2
  obtain ⟨now, he⟩ := (egressAt_get hE)
  obtain ⟨-, -, row, status, -, -, -, hacts, e5, -⟩ := egress_spec hg now backlog
  obtain ⟨hrow, hstat⟩ := e5 r i1
  refine ⟨(pre c tr m).shaper.certifier.rows.push row, ?_, ?_⟩
  · intro j h1 h2
    rw [Array.size_push, hg.shaper.rows] at h2
    apply getElem!_of
    rcases Nat.lt_or_ge j ((pre c tr m).shaper.t % c.egress.buckets) with hj | hj
    · rw [push_lt (by rw [hg.shaper.rows]; exact hj)]
      exact i4 j h1 hj
    · have : j = (pre c tr m).shaper.certifier.rows.size := by rw [hg.shaper.rows]; omega
      rw [this, push_size, hrow]
  · rw [outputs_get _ _ _ hm]
    change some (step (pre c tr m) tr[m]).2 = _
    rw [he, step_egress_snd hg (ht ▸ hk)]
    rw [hacts, ht, hrow, wire_replicate]
    congr 2
    split
    · rename_i hc
      rw [hstat]
    · rfl

end Properties.CertifierDevice.Proofs
