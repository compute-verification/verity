module

public import Proofs.Flock.Recursive.StopCountStream
public import Mathlib.Data.Fin.VecNotation
import all Init.Data.Repr

/-!
# A stream's sessions left unreleased, at most its failure budget

`StopCountStream` counts a stream's joint stops where at most `B` of its sessions are left with a statement not
released. Here `B` is the stream's failure budget `b`, at every order of the record's ops (`RecOp`), and so at every run
(`StopCount.run`): `openSession` refuses an open while the stream's failures and its unended recursive sessions reach
`b`, a session left with a statement not released is unended or stopped, a stop is recorded only of an opened session
(`stop`, and `slot`'s for the slot's orphan), so each stop of one of the stream's sessions is one of its failures (rule
4's count, `failures`).

* `unrel`, `unrelStopped`: the stream's sessions with a statement not released, and those of them that stopped.
* `RecOk`: the record's invariant (every stop of an opened session, the stream's stopped sessions not released at most
  its failures, its sessions not released at most its budget), kept by each of the record's ops (`RecOk.apply`).
* `card_unrel_le_budget`: at every order of the record's ops, at most `b` of a stream's sessions have a statement not
  released.
* `card_unreleased_le_budget`: the same at every run.
* `streamStopBudget`: the stream's joint stop as a `Pattern` at its budget, which `card_streamStops` counts.
* `Flock.SecurityProofs.UnrelLeBudget`, `UnreleasedLeBudget`, `StreamStopBudget`: the three as the lock states them
  (`UnrelWithin`, `UnreleasedWithin`, `StreamStopWithin`), each with an honest witness, a stream at budget 1 whose
  first session runs through before its second opens, and a nontrivial one, a ZK server without rule 4's refusals
  (`Witness.openMade`), where a stream at budget 1 leaves two sessions unreleased.
-/

@[expose] public section

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

/-! ## Each of the record's ops keeps its invariant -/

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

/-- An open the ZK server makes: its budget the stream's, and the stream's failures and unended sessions fewer than it. -/
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

/-- `stop`: a stop only of an opened session, or nothing. -/
theorem RecOk.afterStop {evs : Events} (h : RecOk evs) (s p w : String) : RecOk (Core.Firewall.stop evs s p w).2 := by
  rw [stop_evs]
  split
  · exact h
  · rename_i hc
    have ho : (openOf evs s).isSome = true := by
      cases hO : openOf evs s with
      | none => simp [hO] at hc
      | some _ => rfl
    exact h.pushStop ho p w

/-- `slot`: the slot's orphan's stop only if it was opened, then the slot. -/
theorem RecOk.afterSlot {evs : Events} (h : RecOk evs) (s part : String) : RecOk (Core.Firewall.slot evs s part) := by
  rcases (slot_evs evs s part).1 with he | ⟨k, s', p', w, hor, hos, -, he⟩
  · rw [he]
    exact h.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)
  · rw [he]
    exact (h.pushStop hos p' w).pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)

/-- `admit`: nothing when it admits; a refusal when it refuses, after a stop of the session, opened, if out of turn. -/
theorem RecOk.afterAdmit {evs : Events} (h : RecOk evs) (s part : String) :
    RecOk (Core.Firewall.admit evs s part).2 := by
  cases hadm : Core.Firewall.admit evs s part with
  | mk r evs' =>
    cases r with
    | ok a =>
      obtain ⟨o, j, -, -, -, hserved⟩ := Properties.Firewall.admit_ok hadm
      obtain ⟨-, -, -, rfl⟩ := Properties.Firewall.served_ok hserved
      exact h
    | error why =>
      rcases admit_error_open hadm with he | ⟨p, w, hos, he⟩
      · rw [he]
        exact h.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)
      · rw [he]
        exact (h.pushStop hos p w).pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)

/-- `openSession`: an open the budget lets through, or nothing. -/
theorem RecOk.afterOpen {evs : Events} (h : RecOk evs) (stream : String) (parts : Array String)
    (budget : Option ℕ) (release : Option Core.Firewall.Schedule) :
    RecOk (openSession evs stream parts budget release).2 := by
  rcases openSession_guard evs stream parts budget release with he | ⟨s, he, hb, hg⟩
  · rw [he]
    exact h
  · rw [he]
    exact h.pushOpen _ hb hg

/-- One of the record's ops, as the ZK server calls them: `openSession`, `finish`, `stop`, `release`, `slot`,
`admit`, and the Loop's `wait`. -/
inductive RecOp where
  | openRec (stream : String) (parts : Array String) (budget : Option ℕ) (release : Option Core.Firewall.Schedule)
  | finish (s : String)
  | stop (s part why : String)
  | release (s part : String)
  | slot (s part : String)
  | admit (s part : String)
  | wait (s part : String)

/-- The record after `op`. -/
def RecOp.apply (evs : Events) : RecOp → Events
  | .openRec stream parts budget sched => (openSession evs stream parts budget sched).2
  | .finish s => Core.Firewall.finish evs s
  | .stop s part why => (Core.Firewall.stop evs s part why).2
  | .release s part => Core.Firewall.release evs s part
  | .slot s part => Core.Firewall.slot evs s part
  | .admit s part => (Core.Firewall.admit evs s part).2
  | .wait s part => evs.push (.wait s part)

/-- The record after the ops `ops`, in order, from the empty record. -/
def opsRun (ops : List RecOp) : Events := ops.foldl RecOp.apply #[]

theorem RecOk.apply {evs : Events} (h : RecOk evs) : ∀ op, RecOk (RecOp.apply evs op)
  | .openRec stream parts budget sched => h.afterOpen stream parts budget sched
  | .finish _ => h.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)
  | .stop s part why => h.afterStop s part why
  | .release _ _ => h.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)
  | .slot s part => h.afterSlot s part
  | .admit s part => h.afterAdmit s part
  | .wait _ _ => h.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)

/-- **Every order of the record's ops keeps its invariant.** -/
theorem RecOk.ofOps (ops : List RecOp) : RecOk (opsRun ops) := by
  have : ∀ (l : List RecOp) evs, RecOk evs → RecOk (l.foldl RecOp.apply evs) := by
    intro l
    induction l with
    | nil => intro evs h; exact h
    | cons op rest ih => intro evs h; exact ih _ (h.apply op)
  exact this ops _ RecOk.init

/-- **At most `b` of a stream's sessions are left with a statement not released**, at every order of the record's ops,
`b` the stream's failure budget. -/
theorem card_unrel_le_budget (ops : List RecOp) {t : String} {b : ℕ} (hb : budgetOf (opsRun ops) t = some b) :
    (unrel (opsRun ops) t).card ≤ b :=
  (RecOk.ofOps ops).within t b hb

/-! ## A run's record -/

variable {M : String → ℕ → ℕ}

/-- Each step of a run leaves the record as it was or applies the record's ops to it, so keeps its invariant. -/
theorem RecOk.next {σ : St} (hR : RecOk σ.evs) : ∀ st, RecOk (step M σ st).evs
  | .openRec stream parts budget release => by
    dsimp only [step]
    cases partsOk parts with
    | error _ => exact hR
    | ok u =>
      cases u
      cases budgetArg budget with
      | error _ => exact hR
      | ok u =>
        cases u
        exact hR.afterOpen stream parts budget release
  | .innerSend s => by
    dsimp only [step]
    split <;> exact hR
  | .innerFinish s => by
    dsimp only [step]
    split
    · exact hR.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)
    · exact hR
  | .innerStop s why => by
    dsimp only [step]
    split
    · exact hR.afterStop s INNER why
    · exact hR
  | .wait s part => hR.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)
  | .seat s part => by
    have hA := (hR.afterSlot s part).afterAdmit s part
    dsimp only [step]
    split
    · exact hR
    · split
      · rename_i a evs2 hadm
        rw [hadm] at hA
        exact hA
      · rename_i w evs2 hadm
        rw [hadm] at hA
        exact hA
  | .relay => by
    dsimp only [step]
    split
    · split <;> exact hR
    · exact hR
  | .barrier now prove ready => by
    dsimp only [step]
    split
    · split
      · exact hR
      · split
        · exact hR
        all_goals exact hR.afterStop _ _ _
    · exact hR
  | .verdict v => by
    cases v with
    | error why =>
      dsimp only [step]
      split
      · split
        · exact hR.afterStop _ _ _
        · exact hR
      · exact hR
    | ok b =>
      cases b with
      | false =>
        dsimp only [step]
        split
        · split
          · exact hR.afterStop _ _ _
          · exact hR
        · exact hR
      | true =>
        dsimp only [step]
        split
        · split
          · exact hR.pushPlain (fun _ h => by cases h) (fun _ _ _ h => by cases h)
          · exact hR
        · exact hR
  | .cut why => by
    dsimp only [step]
    split
    · split
      · exact hR
      · exact hR.afterStop _ _ _
    · exact hR
  | .die => hR

/-- **Every run keeps the record's invariant.** -/
theorem RecOk.ofRun (M : String → ℕ → ℕ) (steps : List Step) : RecOk (run M steps).evs := by
  have : ∀ (l : List Step) σ, RecOk σ.evs → RecOk (l.foldl (step M) σ).evs := by
    intro l
    induction l with
    | nil => intro σ h; exact h
    | cons st rest ih => intro σ h; exact ih _ (h.next st)
  exact this steps _ RecOk.init

/-! ## The stream's sessions not released, at most its budget -/

/-- **At most `b` of a stream's sessions are left with a statement not released**, at every run, `b` the stream's
failure budget: of sessions `ss i`, distinct, each opened on the stream if opened at all. -/
theorem card_unreleased_le_budget (M : String → ℕ → ℕ) (steps : List Step) {m : ℕ} (ss : Fin m → String)
    (hinj : Function.Injective ss) {t : String}
    (hs : ∀ i o, openOf (run M steps).evs (ss i) = some o → o.stream = t) {b : ℕ}
    (hb : budgetOf (run M steps).evs t = some b) :
    (univ.filter fun i => unreleasedOf (run M steps) (ss i)).card ≤ b := by
  refine le_trans ?_ ((RecOk.ofRun M steps).within t b hb)
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

/-! ## The bounds as the lock states them -/

/-- **At most `b` of the stream `t`'s sessions are left with a statement not released** on the record `evs`, `b` its
failure budget. -/
def UnrelWithin (evs : Events) (t : String) : Prop :=
  ∀ b, budgetOf evs t = some b → (unrel evs t).card ≤ b

/-- **At most `b` of sessions `ss i` are left with a statement not released** in `σ`, of any sessions `ss i`, distinct,
each opened on the stream `t` if opened at all, `b` its failure budget. -/
def UnreleasedWithin (σ : St) (t : String) : Prop :=
  ∀ (m : ℕ) (ss : Fin m → String), Function.Injective ss → (∀ i o, openOf σ.evs (ss i) = some o → o.stream = t) →
    ∀ b, budgetOf σ.evs t = some b → (univ.filter fun i => unreleasedOf σ (ss i)).card ≤ b

/-- **The joint stop is a `Pattern` at the stream's budget** in `σ`, at message counts `M`: of any sessions `ss i`,
distinct, each with `N i` statements and opened on the stream `t` if opened at all, at most `b` stop, `b` its failure
budget. -/
def StreamStopWithin (M : String → ℕ → ℕ) (σ : St) (t : String) : Prop :=
  ∀ (m : ℕ) (N : Fin m → ℕ) (ss : Fin m → String), Function.Injective ss →
    (∀ i o, openOf σ.evs (ss i) = some o → o.stream = t) → ∀ b, budgetOf σ.evs t = some b →
      ∃ J : Pattern (fun i => Σ p : Fin (N i), Fin (M (ss i) p + 1)) b, J.1 = streamStop N M σ ss

theorem stream_of_all {evs : Events} {s t : String} (h : ((openOf evs s).all fun o => o.stream == t) = true) :
    ∀ o, openOf evs s = some o → o.stream = t := fun o ho => by
  rw [ho] at h
  simpa using h

end Flock.StopCount

namespace Flock.Guarantees

/-- **At most `b` of a stream's sessions are left with a statement not released**, at every order of the record's ops
(`StopCount.opsRun`: `openSession`, `finish`, `stop`, `release`, `slot`, `admit` and the Loop's `wait`, as
`Firewall.Record` defines them) and every stream, `b` the stream's failure budget.

It is about the Lean record's ops, in any order, with no model of the loop. The deployed opener is Python's
`verity_flock.firewall.Firewall`, so for the deployment it rests on Python agreeing with the Lean, which the ZK server's
difftests check. -/
def UnrelLeBudget : Prop :=
  ∀ (ops : List StopCount.RecOp) (t : String), StopCount.UnrelWithin (StopCount.opsRun ops) t

/-- **At most `b` of a stream's sessions are left with a statement not released**, at every run (`StopCount.run`: the
ZK server's ops in any order, at each statement's message count `M s p`), of any sessions `ss i`, distinct, each opened on
the stream if opened at all, `b` the stream's failure budget. It is about `StopCount.run`, as `RecStopsOfRun` is. -/
def UnreleasedLeBudget : Prop :=
  ∀ (M : String → ℕ → ℕ) (steps : List StopCount.Step) (t : String), StopCount.UnreleasedWithin (StopCount.run M steps) t

/-- **A stream's joint stop at its budget**: at every run, the joint stop of a stream's sessions (`StopCount.streamStop`)
is a `Pattern` at the stream's failure budget `b`, at most `b` of its entries stops; and a `Pattern` at `b` takes
`Σ_{|S| ≤ b} Π_{i ∈ S} Σ_p (M_p + 1)` values. It is about `StopCount.run`, as `RecStopsOfRun` is. -/
def StreamStopBudget : Prop :=
  (∀ (M : String → ℕ → ℕ) (steps : List StopCount.Step) (t : String),
      StopCount.StreamStopWithin M (StopCount.run M steps) t) ∧
    ∀ (m : ℕ) (N : Fin m → ℕ) (M : String → ℕ → ℕ) (ss : Fin m → String) (b : ℕ),
      Fintype.card (StopCount.Pattern (fun i => Σ p : Fin (N i), Fin (M (ss i) p + 1)) b) =
        ∑ S ∈ Finset.univ.filter (fun S : Finset (Fin m) => S.card ≤ b), ∏ i ∈ S, ∑ p : Fin (N i), (M (ss i) p + 1)

end Flock.Guarantees

namespace Flock.SecurityProofs

open Flock.StopCount

/-- **At most `b` unreleased, at every order of the record's ops**: `card_unrel_le_budget`. -/
theorem UnrelLeBudget : Flock.Guarantees.UnrelLeBudget := fun ops _ _ hb => card_unrel_le_budget ops hb

/-- **At most `b` unreleased, at every run**: `card_unreleased_le_budget`. -/
theorem UnreleasedLeBudget : Flock.Guarantees.UnreleasedLeBudget := fun M steps _ _ ss hinj hs _ hb =>
  card_unreleased_le_budget M steps ss hinj hs hb

/-- **The joint stop at the budget, counted**: `streamStopBudget` and `card_streamStops`. -/
theorem StreamStopBudget : Flock.Guarantees.StreamStopBudget :=
  ⟨fun M steps _ _ N ss hinj hs _ hb => ⟨streamStopBudget N M steps ss hinj hs hb, rfl⟩,
    fun _ N M ss b => card_streamStops N M ss b⟩

end Flock.SecurityProofs

namespace Flock.StopCount.Witness

open Core.Firewall

/-- An honest stream at budget 1, as the record's ops: session `r0000` opened, its inner session finished, its statement
admitted in the slot and released; then `r0001` opened. -/
def honestOps : List RecOp :=
  [.openRec "st" #["a"] (some 1) none, .finish "r0000", .slot "r0000" "a", .admit "r0000" "a", .release "r0000" "a",
   .openRec "st" #["b"] (some 1) none]

/-- The same stream as a run: `r0000` runs through, its statement accepted and released, then `r0001` opens. -/
def honestStream : List Step :=
  [.openRec "st" #["a"] (some 1) none, .innerSend "r0000", .innerFinish "r0000", .seat "r0000" "a", .relay,
   .barrier 150 .ok .ok, .verdict (.ok true), .openRec "st" #["b"] (some 1) none]

/-- The stream's two sessions. -/
def two : Fin 2 → String := !["r0000", "r0001"]

/-- A ZK server without rule 4's refusals: every open is made, whatever the stream's failures and unended sessions. -/
def openMade (evs : Events) (stream : String) (parts : Array String) (budget : Option ℕ)
    (release : Option Core.Firewall.Schedule) : Events :=
  (openSession.made evs stream parts budget release).2

/-- The record's ops at `openMade`. -/
def applyMade (evs : Events) : RecOp → Events
  | .openRec stream parts budget sched => openMade evs stream parts budget sched
  | op => RecOp.apply evs op

def opsMade (ops : List RecOp) : Events := ops.foldl applyMade #[]

/-- A run's step at `openMade`. -/
def stepMade (M : String → ℕ → ℕ) (σ : St) : Step → St
  | .openRec stream parts budget release =>
    match partsOk parts, budgetArg budget with
    | .ok (), .ok () => { σ with evs := openMade σ.evs stream parts budget release }
    | _, _ => σ
  | st => step M σ st

def runMade (M : String → ℕ → ℕ) (steps : List Step) : St := steps.foldl (stepMade M) St.init

/-- Two sessions opened on a stream at budget 1, the first not ended. -/
def twoOps : List RecOp := [.openRec "st" #["a"] (some 1) none, .openRec "st" #["b"] (some 1) none]

def twoSteps : List Step := [.openRec "st" #["a"] (some 1) none, .openRec "st" #["b"] (some 1) none]

end Flock.StopCount.Witness

namespace Flock.SecurityProofs

open Core.Firewall Flock.StopCount Flock.StopCount.Witness

/-- **`UnrelLeBudget` says something at an honest stream**: at budget 1, `r0000` ran through and was released, `r0001`
opened after it, and one of the stream's sessions is left unreleased, the budget reached. -/
theorem UnrelLeBudget_honest :
    budgetOf (opsRun honestOps) "st" = some 1 ∧ unrelOn (opsRun honestOps) "st" "r0000" = false ∧
      (openOf (opsRun honestOps) "r0001").isSome = true ∧ (unrel (opsRun honestOps) "st").card = 1 := by
  decide +kernel

/-- **`UnrelLeBudget` fails at a ZK server without rule 4's refusals** (`openMade`): there a stream at budget 1 leaves two
sessions unreleased. -/
theorem UnrelLeBudget_nontrivial : ¬ UnrelWithin (opsMade twoOps) "st" := fun h =>
  absurd (h 1 (by decide +kernel)) (by decide +kernel)

/-- **`UnreleasedLeBudget` says something at an honest run**: at budget 1, `r0000` ran through, `r0001` opened after it
and is left unreleased, the budget reached. -/
theorem UnreleasedLeBudget_honest :
    budgetOf (run one honestStream).evs "st" = some 1 ∧ unreleasedOf (run one honestStream) "r0000" = false ∧
      unreleasedOf (run one honestStream) "r0001" = true ∧
      (Finset.univ.filter fun i => unreleasedOf (run one honestStream) (two i)).card = 1 := by
  decide +kernel

theorem two_injective : Function.Injective two := by decide +kernel

theorem runMade_stream : ∀ i o, openOf (runMade one twoSteps).evs (two i) = some o → o.stream = "st" := by
  have h : ∀ i, ((openOf (runMade one twoSteps).evs (two i)).all fun o => o.stream == "st") = true := by decide +kernel
  exact fun i => stream_of_all (h i)

/-- **`UnreleasedLeBudget` fails at a ZK server without rule 4's refusals** (`openMade`): there a stream at budget 1
leaves both its sessions unreleased. -/
theorem UnreleasedLeBudget_nontrivial : ¬ UnreleasedWithin (runMade one twoSteps) "st" := fun h =>
  absurd (h 2 two two_injective runMade_stream 1 (by decide +kernel)) (by decide +kernel)

/-- **`StreamStopBudget` says something at an honest run**: at budget 1, of the stream's two sessions `r0000` ran
through and `r0001`, opened after it, would stop at its inner session, one stop, the budget reached. -/
theorem StreamStopBudget_honest :
    budgetOf (run one honestStream).evs "st" = some 1 ∧
      (streamStop ![2, 2] one (run one honestStream) two 0).isNone = true ∧
      (streamStop ![2, 2] one (run one honestStream) two 1).isSome = true := by
  decide +kernel

/-- **`StreamStopBudget` fails at a ZK server without rule 4's refusals** (`openMade`): there both sessions of a stream at
budget 1 stop, which is no `Pattern` at 1. -/
theorem StreamStopBudget_nontrivial : ¬ StreamStopWithin one (runMade one twoSteps) "st" := fun h => by
  obtain ⟨J, hJ⟩ := h 2 ![2, 2] two two_injective runMade_stream 1 (by decide +kernel)
  have hc := J.2
  rw [hJ] at hc
  revert hc
  decide +kernel

end Flock.SecurityProofs
