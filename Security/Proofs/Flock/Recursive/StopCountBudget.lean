import Proofs.Flock.Recursive.StopCountStream

/-!
# A stream's sessions left unreleased, at most its failure budget

`StopCountStream` counts a stream's joint stops where at most `B` of its sessions are left with a statement not
released. Here `B` is the stream's failure budget `b`, at every run (`StopCount.run`): `openSession` refuses an open while
the stream's failures and its unended recursive sessions reach `b`, a session left with a statement not released is
unended or stopped, and each stop of one of the stream's sessions is one of its failures (rule 4's count, `failures`).

* `unrel`, `unrelStopped`: the stream's sessions with a statement not released, and those of them that stopped.
* `RecOk`, `BInv`: the record's invariant (every stop of an opened session, the stream's stopped sessions not released at
  most its failures, its sessions not released at most its budget) and the run's (the slot's orphan, too, of an opened
  session).
* `card_unreleased_le_budget`: at every run, at most `b` of a stream's sessions are left with a statement not released.
* `streamStopBudget`: the stream's joint stop as a `Pattern` at its budget, which `card_streamStops` counts.
-/

namespace Flock.StopCount

open Core.Firewall Finset

/-! ## The stream's sessions not released -/

/-- The stream's first open. -/
def firstOn (evs : Events) (t : String) : Option Open :=
  evs.findSome? fun | .opened o => if o.stream == t then some o else none | _ => none

theorem budgetOf_eq (evs : Events) (t : String) : budgetOf evs t = (firstOn evs t).bind (·.budget) := rfl

/-- Whether `s` was opened on `t` and a statement of it is not released. -/
def unrelOn (evs : Events) (t s : String) : Bool :=
  match openOf evs s with
  | some o => o.stream == t && (nextOf evs o).isSome
  | none => false

/-- The recursive sessions opened. -/
def openedSet (evs : Events) : Finset String := (openedSessions evs).toFinset

/-- **The stream's sessions with a statement not released.** -/
def unrel (evs : Events) (t : String) : Finset String := (openedSet evs).filter fun s => unrelOn evs t s = true

/-- Those of them that stopped. -/
def unrelStopped (evs : Events) (t : String) : Finset String :=
  (unrel evs t).filter fun s => (stopOf evs s).isSome = true

theorem mem_openedSet {evs : Events} {s : String} : s ∈ openedSet evs ↔ (openOf evs s).isSome = true := by
  simp only [openedSet, openedSessions, List.mem_toFinset, List.mem_eraseDups, List.mem_filterMap,
    Array.mem_toList_iff]
  constructor
  · rintro ⟨e, he, hs⟩
    cases e with
    | opened o =>
      simp only [Option.some.injEq] at hs
      subst hs
      cases h : openOf evs o.session with
      | some _ => rfl
      | none =>
        rw [openOf, Array.findSome?_eq_none_iff] at h
        have := h _ he
        simp at this
    | _ => simp at hs
  · intro h
    obtain ⟨o, ho⟩ := Option.isSome_iff_exists.mp h
    obtain ⟨e, he, hf⟩ := Array.exists_of_findSome?_eq_some ho
    refine ⟨e, he, ?_⟩
    cases e with
    | opened o' =>
      by_cases hs : o'.session = s
      · simp [hs]
      · simp [hs] at hf
    | _ => simp at hf

theorem mem_unrel {evs : Events} {t s : String} :
    s ∈ unrel evs t ↔ ∃ o, openOf evs s = some o ∧ o.stream = t ∧ (nextOf evs o).isSome = true := by
  rw [unrel, Finset.mem_filter, mem_openedSet]
  unfold unrelOn
  cases openOf evs s with
  | none => simp
  | some o => simp

theorem mem_unrelStopped {evs : Events} {t s : String} :
    s ∈ unrelStopped evs t ↔ s ∈ unrel evs t ∧ (stopOf evs s).isSome = true := mem_filter

/-- The stream's sessions not released are its unended ones and its stopped ones. -/
theorem card_unrel_le (evs : Events) (t : String) : (unrel evs t).card ≤ unended evs t + (unrelStopped evs t).card := by
  have hsub : ((unrel evs t).filter fun s => ¬(stopOf evs s).isSome = true) ⊆
      ((openedSessions evs).filter (unendedOn evs t)).toFinset := by
    intro s hs
    rw [mem_filter, mem_unrel] at hs
    obtain ⟨⟨o, ho, hst, hn⟩, hS⟩ := hs
    have hS' : stopOf evs s = none := by simpa using hS
    rw [List.mem_toFinset, List.mem_filter]
    refine ⟨?_, ?_⟩
    · have : s ∈ openedSet evs := mem_openedSet.mpr (by rw [ho]; rfl)
      simpa [openedSet] using this
    · simp [unendedOn, ho, hst, hn, hS']
  have hu : ((unrel evs t).filter fun s => ¬(stopOf evs s).isSome = true).card ≤ unended evs t :=
    (card_le_card hsub).trans (List.toFinset_card_le _)
  have hs := card_filter_add_card_filter_not (s := unrel evs t) (fun s => (stopOf evs s).isSome = true)
  unfold unrelStopped
  omega

/-! ## One more event -/

theorem firstOn_push_ne {evs : Events} {e : Event} (he : ∀ o, e ≠ .opened o) (t : String) :
    firstOn (evs.push e) t = firstOn evs t := by
  simp only [firstOn, Array.findSome?_push]
  cases e with
  | opened o => exact absurd rfl (he o)
  | _ => simp

theorem firstOn_push_open (evs : Events) (o : Open) (t : String) :
    firstOn (evs.push (.opened o)) t = (firstOn evs t).or (if o.stream == t then some o else none) := by
  simp only [firstOn, Array.findSome?_push]

theorem budgetOf_push_ne {evs : Events} {e : Event} (he : ∀ o, e ≠ .opened o) (t : String) :
    budgetOf (evs.push e) t = budgetOf evs t := by
  rw [budgetOf_eq, budgetOf_eq, firstOn_push_ne he]

theorem budgetOf_push_open_ne (evs : Events) {o : Open} {t : String} (h : o.stream ≠ t) :
    budgetOf (evs.push (.opened o)) t = budgetOf evs t := by
  rw [budgetOf_eq, budgetOf_eq, firstOn_push_open]
  simp [h]

theorem openOf_push_open (evs : Events) (o : Open) (s : String) :
    openOf (evs.push (.opened o)) s = (openOf evs s).or (if o.session == s then some o else none) := by
  simp only [openOf, Array.findSome?_push]

theorem isSome_openOf_push {evs : Events} {s : String} (e : Event) (h : (openOf evs s).isSome = true) :
    (openOf (evs.push e) s).isSome = true := by
  obtain ⟨o, ho⟩ := Option.isSome_iff_exists.mp h
  rw [openOf_push_some e ho]
  rfl

/-- A statement released stays released. -/
theorem nextOf_isSome_of_push {evs : Events} {e : Event} {o : Open} (h : (nextOf (evs.push e) o).isSome = true) :
    (nextOf evs o).isSome = true := by
  cases hn : nextOf evs o with
  | some _ => rfl
  | none =>
    exfalso
    have hall := nextOf_none hn
    have : nextOf (evs.push e) o = none := by
      rw [nextOf_eq, Array.findIdx?_eq_none_iff]
      intro x hx
      obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.mp hx
      simpa using mem_relOf_push e (hall j hj)
    rw [this] at h
    cases h

theorem unrel_push_sub {evs : Events} {e : Event} (he : ∀ o, e ≠ .opened o) (t : String) :
    unrel (evs.push e) t ⊆ unrel evs t := fun s hs => by
  rw [mem_unrel] at hs ⊢
  obtain ⟨o, ho, hst, hn⟩ := hs
  rw [openOf_push_ne he] at ho
  exact ⟨o, ho, hst, nextOf_isSome_of_push hn⟩

theorem unrel_push_eq {evs : Events} {e : Event} (h1 : ∀ o, e ≠ .opened o) (h2 : ∀ s p, e ≠ .release s p)
    (t : String) : unrel (evs.push e) t = unrel evs t := by
  ext s
  rw [mem_unrel, mem_unrel, openOf_push_ne h1]
  simp only [nextOf_push_ne h2]

theorem mem_unrel_push_open {evs : Events} {o' : Open} {t s : String} (h : s ∈ unrel (evs.push (.opened o')) t) :
    s ∈ unrel evs t ∨ (s = o'.session ∧ openOf evs s = none ∧ o'.stream = t) := by
  rw [mem_unrel] at h
  obtain ⟨o, ho, hst, hn⟩ := h
  rw [nextOf_push_ne (fun _ _ h => by cases h)] at hn
  rw [openOf_push_open] at ho
  cases h0 : openOf evs s with
  | some o0 =>
    rw [h0] at ho
    cases ho
    exact Or.inl (mem_unrel.mpr ⟨o, h0, hst, hn⟩)
  | none =>
    rw [h0, Option.none_or] at ho
    by_cases hs : o'.session = s
    · subst hs
      simp only [beq_self_eq_true, ↓reduceIte, Option.some.injEq] at ho
      subst ho
      exact Or.inr ⟨rfl, rfl, hst⟩
    · simp [hs] at ho

/-! ## The stream's failures -/

theorem Tally.n_le_step (stream : String) (b : Option ℕ) (T : Tally) (e : Event) :
    T.n ≤ (Tally.step stream b T e).n := by
  cases e <;> simp only [Tally.step] <;> (repeat' split) <;> simp

theorem Tally.lookup_step {t : String} {b : Option ℕ} {T : Tally} {k : String} (e : Event)
    (h : (T.mine.lookup k).isSome = true) : ((Tally.step t b T e).mine.lookup k).isSome = true := by
  cases e <;> simp only [Tally.step] <;> (repeat' split) <;>
    first
    | exact h
    | (simp only [List.lookup_cons]; split <;> simp_all)

theorem lookup_foldl {t : String} {b : Option ℕ} {k : String} :
    ∀ (l : List Event) (T : Tally),
      ((T.mine.lookup k).isSome = true ∨ ∃ o, Event.opened o ∈ l ∧ o.stream = t ∧ o.session = k) →
      ((l.foldl (Tally.step t b) T).mine.lookup k).isSome = true
  | [], T, h => by
    rcases h with h | ⟨o, ho, -⟩
    · exact h
    · cases ho
  | e :: l, T, h => by
    rw [List.foldl_cons]
    apply lookup_foldl l
    rcases h with h | ⟨o, ho, hs, hk⟩
    · exact Or.inl (Tally.lookup_step e h)
    · rcases List.mem_cons.mp ho with rfl | ho
      · left
        simp [Tally.step, hs, hk]
      · exact Or.inr ⟨o, ho, hs, hk⟩

theorem foldl_tally_none {t : String} {b : Option ℕ} :
    ∀ (l : List Event), (∀ e ∈ l, ∀ o, e = .opened o → o.stream ≠ t) → l.foldl (Tally.step t b) {} = {}
  | [], _ => rfl
  | e :: l, h => by
    rw [List.foldl_cons]
    have : Tally.step t b {} e = {} := by
      cases e with
      | opened o =>
        have := h _ (List.mem_cons_self ..) o rfl
        simp [Tally.step, this]
      | _ => simp [Tally.step, List.lookup]
    rw [this]
    exact foldl_tally_none l (fun e he => h e (List.mem_cons_of_mem _ he))

/-- A stream's failures only grow. -/
theorem failures_le_push (evs : Events) (e : Event) (t : String) : failures evs t ≤ failures (evs.push e) t := by
  cases hf : firstOn evs t with
  | none =>
    have h0 : failures evs t = 0 := by
      unfold failures
      rw [← Array.foldl_toList, foldl_tally_none]
      intro e he o heo hst
      subst heo
      rw [firstOn, Array.findSome?_eq_none_iff] at hf
      have := hf _ (Array.mem_toList_iff.mp he)
      simp [hst] at this
    omega
  | some f =>
    have hb : budgetOf (evs.push e) t = budgetOf evs t := by
      cases e with
      | opened o =>
        rw [budgetOf_eq, budgetOf_eq, firstOn_push_open, hf]
        rfl
      | _ => exact budgetOf_push_ne (fun _ h => by cases h) t
    unfold failures
    rw [hb, Array.foldl_push]
    exact Tally.n_le_step _ _ _ _

/-- **A stop of one of the stream's sessions is one of its failures.** -/
theorem failures_push_stop {evs : Events} {s : String} {o : Open} (ho : openOf evs s = some o) (p w : String) :
    failures (evs.push (.stop s p w)) o.stream = failures evs o.stream + 1 := by
  have hmem : Event.opened o ∈ evs ∧ o.session = s := by
    obtain ⟨e, he, hf⟩ := Array.exists_of_findSome?_eq_some ho
    cases e with
    | opened o' =>
      by_cases hs : o'.session = s
      · simp only [hs, beq_self_eq_true, ↓reduceIte, Option.some.injEq] at hf
        subst hf
        exact ⟨he, hs⟩
      · simp [hs] at hf
    | _ => simp at hf
  have hl := lookup_foldl (t := o.stream) (b := budgetOf evs o.stream) (k := s) evs.toList {}
    (Or.inr ⟨o, Array.mem_toList_iff.mpr hmem.1, rfl, hmem.2⟩)
  rw [Array.foldl_toList] at hl
  unfold failures
  rw [budgetOf_push_ne (fun _ h => by cases h), Array.foldl_push]
  simp only [Tally.step, hl, ↓reduceIte]

/-! ## The record's invariant -/

/-- **The record's invariant**: a stop is of an opened session; a stream's stopped sessions not released are at most
its failures; its sessions not released are at most its budget. -/
structure RecOk (evs : Events) : Prop where
  stopOpened : ∀ s, (stopOf evs s).isSome = true → (openOf evs s).isSome = true
  stopped : ∀ t, (unrelStopped evs t).card ≤ failures evs t
  within : ∀ t b, budgetOf evs t = some b → (unrel evs t).card ≤ b

theorem RecOk.init : RecOk #[] where
  stopOpened s hs := by simp [stopOf] at hs
  stopped t := by
    have : unrelStopped #[] t = ∅ := by
      ext x
      simp [mem_unrelStopped, mem_unrel, openOf]
    rw [this]
    exact Nat.zero_le _
  within t b hb := by simp [budgetOf] at hb

/-- An event other than an open or a stop. -/
theorem RecOk.pushPlain {evs : Events} (h : RecOk evs) {e : Event} (h1 : ∀ o, e ≠ .opened o)
    (h2 : ∀ s p w, e ≠ .stop s p w) : RecOk (evs.push e) where
  stopOpened s hs := by
    rw [stopOf_push_ne h2] at hs
    rw [openOf_push_ne h1]
    exact h.stopOpened s hs
  stopped t := by
    refine le_trans (card_le_card fun s hs => ?_) ((h.stopped t).trans (failures_le_push evs e t))
    rw [mem_unrelStopped] at hs ⊢
    rw [stopOf_push_ne h2] at hs
    exact ⟨unrel_push_sub h1 t hs.1, hs.2⟩
  within t b hb := by
    rw [budgetOf_push_ne h1] at hb
    exact (card_le_card (unrel_push_sub h1 t)).trans (h.within t b hb)

/-- A stop of an opened session. -/
theorem RecOk.pushStop {evs : Events} (h : RecOk evs) {s' : String} (ho : (openOf evs s').isSome = true)
    (p w : String) : RecOk (evs.push (.stop s' p w)) where
  stopOpened s hs := by
    rw [openOf_push_ne (fun _ h => by cases h)]
    by_cases hss : s' = s
    · subst hss
      exact ho
    · rw [stopOf_push_stop_ne p w hss] at hs
      exact h.stopOpened s hs
  stopped t := by
    have hU := unrel_push_eq (evs := evs) (e := .stop s' p w) (fun _ h => by cases h) (fun _ _ h => by cases h) t
    by_cases hm : s' ∈ unrel evs t
    · obtain ⟨o, hO, hst, -⟩ := mem_unrel.mp hm
      have hf := failures_push_stop hO p w
      rw [hst] at hf
      rw [hf]
      refine (card_le_card (t := insert s' (unrelStopped evs t)) fun x hx => ?_).trans
        ((card_insert_le _ _).trans (Nat.add_le_add_right (h.stopped t) 1))
      rw [mem_unrelStopped, hU] at hx
      rw [mem_insert, mem_unrelStopped]
      by_cases hxs : x = s'
      · exact Or.inl hxs
      · rw [stopOf_push_stop_ne p w (Ne.symm hxs)] at hx
        exact Or.inr hx
    · refine le_trans (card_le_card fun x hx => ?_) ((h.stopped t).trans (failures_le_push _ _ _))
      rw [mem_unrelStopped, hU] at hx
      rw [mem_unrelStopped]
      have hxs : s' ≠ x := fun e => hm (e ▸ hx.1)
      rw [stopOf_push_stop_ne p w hxs] at hx
      exact hx
  within t b hb := by
    rw [budgetOf_push_ne (fun _ h => by cases h)] at hb
    rw [unrel_push_eq (fun _ h => by cases h) (fun _ _ h => by cases h)]
    exact h.within t b hb

/-- **An open the budget lets through**: the stream's budget is the open's, and its failures and unended sessions are
fewer than it. -/
theorem RecOk.pushOpen {evs : Events} (h : RecOk evs) (o' : Open)
    (hb : budgetOf (evs.push (.opened o')) o'.stream = o'.budget)
    (hg : ∀ b, o'.budget = some b → failures evs o'.stream + unended evs o'.stream < b) :
    RecOk (evs.push (.opened o')) where
  stopOpened s hs := by
    rw [stopOf_push_ne (fun _ _ _ h => by cases h)] at hs
    exact isSome_openOf_push _ (h.stopOpened s hs)
  stopped t := by
    refine le_trans (card_le_card fun x hx => ?_) ((h.stopped t).trans (failures_le_push _ _ _))
    rw [mem_unrelStopped] at hx ⊢
    rw [stopOf_push_ne (fun _ _ _ h => by cases h)] at hx
    rcases mem_unrel_push_open hx.1 with hu | ⟨-, hn, -⟩
    · exact ⟨hu, hx.2⟩
    · have := h.stopOpened x hx.2
      rw [hn] at this
      cases this
  within t b hbt := by
    by_cases ht : o'.stream = t
    · subst ht
      rw [hb] at hbt
      have hlt := hg b hbt
      have hsub : unrel (evs.push (.opened o')) o'.stream ⊆ insert o'.session (unrel evs o'.stream) :=
        fun x hx => by
          rcases mem_unrel_push_open hx with hu | ⟨rfl, -, -⟩
          · exact mem_insert_of_mem hu
          · exact mem_insert_self _ _
      have h1 := card_unrel_le evs o'.stream
      have h2 := h.stopped o'.stream
      have h3 := (card_le_card hsub).trans (card_insert_le _ _)
      omega
    · rw [budgetOf_push_open_ne _ ht] at hbt
      refine le_trans (card_le_card fun x hx => ?_) (h.within t b hbt)
      rcases mem_unrel_push_open hx with hu | ⟨-, -, hst⟩
      · exact hu
      · exact absurd hst ht

/-! ## The slot's orphan -/

theorem doneAfter_push {evs : Events} {k : ℕ} (hk : k < evs.size) (e : Event) (s p : String) :
    doneAfter (evs.push e) k s p = (doneAfter evs k s p || doneEv s p e) := by
  have hk1 : k + 1 ≤ evs.size := hk
  simp only [doneAfter, Array.size_push]
  rw [Array.extract_push]
  simp only [Nat.not_succ_le_self, ↓reduceIte, hk1, Array.any_push]

theorem orphan_push_of_ne {evs : Events} {e : Event} (he : ∀ s p, e ≠ .slot s p) :
    orphan (evs.push e) = (orphan evs).bind fun x => if doneEv x.2.1 x.2.2 e then none else some x := by
  rw [orphan_eq, orphan_eq, lastSlot_push_ne he]
  cases hl : lastSlot evs with
  | none => rfl
  | some x =>
    obtain ⟨k, s, p⟩ := x
    simp only
    rw [doneAfter_push (lastSlot_lt hl)]
    cases doneAfter evs k s p <;> simp

theorem orphan_of_push {evs : Events} {e : Event} (he : ∀ s p, e ≠ .slot s p) {x : ℕ × String × String}
    (h : orphan (evs.push e) = some x) : orphan evs = some x := by
  rw [orphan_push_of_ne he] at h
  cases ho : orphan evs with
  | none => rw [ho] at h; cases h
  | some y =>
    rw [ho] at h
    simp only [Option.bind_some] at h
    split at h
    · cases h
    · exact h

/-- The slot's orphan is of an opened session. -/
def OrphanOpened (evs : Events) : Prop := ∀ k s p, orphan evs = some (k, s, p) → (openOf evs s).isSome = true

theorem OrphanOpened.push {evs : Events} (h : OrphanOpened evs) {e : Event} (he : ∀ s p, e ≠ .slot s p) :
    OrphanOpened (evs.push e) := fun k s p ho => isSome_openOf_push e (h k s p (orphan_of_push he ho))

/-- `admit` refusing records the refusal, after a stop of the session, opened, if the part was asked out of turn. -/
theorem admit_error_open {evs : Events} {s part why : String} {evs' : Events}
    (h : admit evs s part = (.error why, evs')) :
    evs' = evs.push (.refused s part) ∨
      ∃ p w, (openOf evs s).isSome = true ∧ evs' = (evs.push (.stop s p w)).push (.refused s part) := by
  cases ho : openOf evs s with
  | none =>
    left
    unfold admit admit.refuse at h
    rw [ho] at h
    simp only [Prod.mk.injEq] at h
    exact h.2.symm
  | some o =>
    rcases admit_error h with h' | ⟨p, w, h'⟩
    · exact Or.inl h'
    · exact Or.inr ⟨p, w, rfl, h'⟩

/-- An open the firewall makes: its budget the stream's, and the stream's failures and unended sessions fewer than it. -/
theorem openSession_guard (evs : Events) (stream : String) (parts : Array String) (budget : Option ℕ)
    (release : Option Core.Firewall.Schedule) :
    (openSession evs stream parts budget release).2 = evs ∨
      ∃ s, (openSession evs stream parts budget release).2 =
          evs.push (.opened { session := s, stream, parts, budget, release }) ∧
        budgetOf (evs.push (.opened { session := s, stream, parts, budget, release })) stream = budget ∧
        ∀ b, budget = some b → failures evs stream + unended evs stream < b := by
  have hgo : (openSession.go evs stream parts budget release).2 = evs ∨
      ∃ s, (openSession.go evs stream parts budget release).2 =
          evs.push (.opened { session := s, stream, parts, budget, release }) ∧
        ∀ b, budget = some b → failures evs stream + unended evs stream < b := by
    unfold openSession.go openSession.made
    cases budget with
    | none => exact Or.inr ⟨_, rfl, fun _ h => by cases h⟩
    | some b =>
      dsimp only
      by_cases h1 : failures evs stream ≥ b
      · left
        simp [h1]
      · by_cases h2 : failures evs stream + unended evs stream ≥ b
        · left
          simp [h1, h2]
        · right
          simp only [h1, h2, ↓reduceIte]
          exact ⟨_, rfl, fun b' hb' => by cases hb'; omega⟩
  unfold openSession
  dsimp only
  split
  · rename_i f hf
    split
    · exact Or.inl rfl
    · rename_i hne
      have hfb : f.budget = budget := by simpa using hne
      rcases hgo with h | ⟨s, h, hg⟩
      · exact Or.inl h
      · refine Or.inr ⟨s, h, ?_, hg⟩
        have hf' : firstOn evs stream = some f := hf
        rw [budgetOf_eq, firstOn_push_open, hf']
        exact hfb
  · rename_i hf
    rcases hgo with h | ⟨s, h, hg⟩
    · exact Or.inl h
    · refine Or.inr ⟨s, h, ?_, hg⟩
      have hf' : firstOn evs stream = none := hf
      rw [budgetOf_eq, firstOn_push_open, hf']
      simp

/-! ## The run's invariant -/

/-- **The run's invariant**: the record's, and the slot's orphan of an opened session. -/
structure BInv (evs : Events) : Prop where
  ok : RecOk evs
  orph : OrphanOpened evs

theorem BInv.init : BInv #[] where
  ok := RecOk.init
  orph k s p h := by simp [orphan] at h

theorem BInv.pushPlain {evs : Events} (h : BInv evs) {e : Event} (h1 : ∀ o, e ≠ .opened o)
    (h2 : ∀ s p w, e ≠ .stop s p w) (h3 : ∀ s p, e ≠ .slot s p) : BInv (evs.push e) :=
  ⟨h.ok.pushPlain h1 h2, h.orph.push h3⟩

theorem BInv.stop {evs : Events} (h : BInv evs) {s : String} (ho : (openOf evs s).isSome = true) (p w : String) :
    BInv (stop evs s p w).2 := by
  rw [stop_evs]
  split
  · exact h
  · exact ⟨h.ok.pushStop ho p w, h.orph.push fun _ _ h => by cases h⟩

theorem RecOk.slot {evs : Events} (h : BInv evs) (s part : String) : RecOk (slot evs s part) := by
  rcases (slot_evs evs s part).1 with he | ⟨k, s', p', w, hor, -, he⟩
  · rw [he]
    exact h.ok.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)
  · rw [he]
    exact (h.ok.pushStop (h.orph k s' p' hor) p' w).pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)

variable {M : String → ℕ → ℕ}

theorem holder_opened {σ : St} (hI : Inv M σ) {h : Holder} (hh : σ.holder = some h) :
    (openOf σ.evs h.session).isSome = true := by
  obtain ⟨o, hO, -⟩ := hI.hold h hh
  rw [hO]
  rfl

theorem BInv.seat {σ : St} (hB : BInv σ.evs) (s part : String) : BInv (step M σ (.seat s part)).evs := by
  dsimp only [step]
  split
  · exact hB
  · have hR1 := RecOk.slot hB s part
    obtain ⟨k1, hor1⟩ := orphan_slot σ.evs s part
    split
    · rename_i a evs2 hadm
      obtain ⟨o, j, hO1, -, -, hserved⟩ := Properties.Firewall.admit_ok hadm
      obtain ⟨-, -, -, rfl⟩ := Properties.Firewall.served_ok hserved
      refine ⟨hR1, fun k s' p' hor => ?_⟩
      rw [hor1] at hor
      simp only [Option.some.injEq, Prod.mk.injEq] at hor
      obtain ⟨-, rfl, -⟩ := hor
      rw [hO1]
      rfl
    · rename_i w evs2 hadm
      rcases admit_error_open hadm with he | ⟨p, w', hos, he⟩
      · subst he
        refine ⟨hR1.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h), fun k s' p' hor => ?_⟩
        rw [orphan_push_of_ne (fun _ _ h => by cases h), hor1] at hor
        simp [doneEv] at hor
      · subst he
        refine ⟨(hR1.pushStop hos p w').pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h),
          fun k s' p' hor => ?_⟩
        have h0 : orphan ((slot σ.evs s part).push (.stop s p w')) = none := by
          rw [orphan_push_of_ne (fun _ _ h => by cases h), hor1]
          simp [doneEv]
        rw [orphan_push_of_ne (fun _ _ h => by cases h), h0] at hor
        cases hor

theorem BInv.next {σ : St} (hI : Inv M σ) (hB : BInv σ.evs) : ∀ st, BInv (step M σ st).evs
  | .openRec stream parts budget release => by
    dsimp only [step]
    cases partsOk parts with
    | error _ => exact hB
    | ok u =>
      cases u
      cases budgetArg budget with
      | error _ => exact hB
      | ok u =>
        cases u
        show BInv (openSession σ.evs stream parts budget release).2
        rcases openSession_guard σ.evs stream parts budget release with h | ⟨s, h, hb, hg⟩
        · rw [h]
          exact hB
        · rw [h]
          exact ⟨hB.ok.pushOpen _ hb hg, hB.orph.push fun _ _ h => by cases h⟩
  | .innerSend s => by
    dsimp only [step]
    split <;> exact hB
  | .innerFinish s => by
    dsimp only [step]
    split
    · exact hB.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h) (fun _ _ h => by cases h)
    · exact hB
  | .innerStop s why => by
    dsimp only [step]
    split
    · rename_i hg
      rw [Bool.and_eq_true] at hg
      exact hB.stop hg.1 INNER why
    · exact hB
  | .wait s part => hB.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h) (fun _ _ h => by cases h)
  | .seat s part => hB.seat s part
  | .relay => by
    dsimp only [step]
    split
    · split <;> exact hB
    · exact hB
  | .barrier now prove ready => by
    dsimp only [step]
    split
    · rename_i h hh
      split
      · exact hB
      · split
        · exact hB
        all_goals exact hB.stop (holder_opened hI hh) _ _
    · exact hB
  | .verdict v => by
    cases v with
    | error why =>
      dsimp only [step]
      split
      · rename_i h hh
        split
        · exact hB.stop (holder_opened hI hh) _ _
        · exact hB
      · exact hB
    | ok b =>
      cases b with
      | false =>
        dsimp only [step]
        split
        · rename_i h hh
          split
          · exact hB.stop (holder_opened hI hh) _ _
          · exact hB
        · exact hB
      | true =>
        dsimp only [step]
        split
        · split
          · exact hB.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h) (fun _ _ h => by cases h)
          · exact hB
        · exact hB
  | .cut why => by
    dsimp only [step]
    split
    · rename_i h hh
      split
      · exact hB
      · exact hB.stop (holder_opened hI hh) _ _
    · exact hB
  | .die => hB

/-- **Every run keeps the record's invariant.** -/
theorem BInv.ofRun (M : String → ℕ → ℕ) (steps : List Step) : BInv (run M steps).evs := by
  have : ∀ (l : List Step) σ, Inv M σ → BInv σ.evs → BInv (l.foldl (step M) σ).evs := by
    intro l
    induction l with
    | nil => intro σ _ h; exact h
    | cons st rest ih => intro σ hI h; exact ih _ (hI.next st) (h.next hI st)
  exact this steps _ (Inv.init M) BInv.init

/-! ## The stream's sessions not released, at most its budget -/

/-- **At most `b` of a stream's sessions are left with a statement not released**, at every run, `b` the stream's
failure budget: of sessions `ss i`, distinct, each opened on the stream if opened at all. -/
theorem card_unreleased_le_budget (M : String → ℕ → ℕ) (steps : List Step) {m : ℕ} (ss : Fin m → String)
    (hinj : Function.Injective ss) {t : String}
    (hs : ∀ i o, openOf (run M steps).evs (ss i) = some o → o.stream = t) {b : ℕ}
    (hb : budgetOf (run M steps).evs t = some b) :
    (univ.filter fun i => unreleasedOf (run M steps) (ss i)).card ≤ b := by
  refine le_trans ?_ ((BInv.ofRun M steps).ok.within t b hb)
  refine card_le_card_of_injOn ss (fun i hi => ?_) (fun i _ j _ h => hinj h)
  rw [mem_coe, mem_filter] at hi
  rw [mem_coe, mem_unrel]
  obtain ⟨-, hi⟩ := hi
  unfold unreleasedOf at hi
  cases ho : openOf (run M steps).evs (ss i) with
  | none =>
    rw [ho] at hi
    cases hi
  | some o =>
    rw [ho] at hi
    exact ⟨o, rfl, hs i o ho, hi⟩

/-- **A stream's joint stop at its budget `b`**: at most `b` of its sessions stop, so its joint stop is a `Pattern` at
`b`, of which `card_streamStops` counts `Σ_{|S| ≤ b} Π_{i ∈ S} Σ_p (M_p + 1)`. -/
def streamStopBudget {m : ℕ} (N : Fin m → ℕ) (M : String → ℕ → ℕ) (steps : List Step) (ss : Fin m → String)
    (hinj : Function.Injective ss) {t : String}
    (hs : ∀ i o, openOf (run M steps).evs (ss i) = some o → o.stream = t) {b : ℕ}
    (hb : budgetOf (run M steps).evs t = some b) :
    Pattern (fun i => Σ p : Fin (N i), Fin (M (ss i) p + 1)) b :=
  streamStopIn N M steps ss b (card_unreleased_le_budget M steps ss hinj hs hb)

end Flock.StopCount
