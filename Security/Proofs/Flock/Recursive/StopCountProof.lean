import Proofs.Properties.Firewall
import Proofs.Flock.Recursive.StopCount

/-!
# A recursive session's stops, counted from the firewall's rules: the proof

`Flock.SecurityProofs.RecStopsOfRun`, by an invariant of `StopCount.step` (`Inv`) over the run's ops, on the record's
rules as `Firewall.Record` writes them and as `Proofs.Properties.Firewall` proves them (`admit_ok`, `served_ok`,
`nextOf_lt`, `stopOf_push_stop`, `afterOnTime`). What one more event does to a recursive session's open, stop, finish,
released statements and the slot's orphan comes first.
-/

namespace Flock.StopCount

open Core.Firewall

/-! ## One more event on the record -/

/-- The statements of `s` released on `evs`, as `nextOf` reads them. -/
def relOf (evs : Events) (s : String) : Array String :=
  evs.filterMap fun | .release s' p => if s' == s then some p else none | _ => none

theorem nextOf_eq (evs : Events) (o : Open) :
    nextOf evs o = o.parts.findIdx? fun p => !(relOf evs o.session).contains p := rfl

theorem openOf_push_some {evs : Events} {s : String} {o : Open} (e : Event) (h : openOf evs s = some o) :
    openOf (evs.push e) s = some o := by
  simp only [openOf, Array.findSome?_push] at h ⊢
  rw [h]
  rfl

theorem openOf_push_ne {evs : Events} {e : Event} (he : ∀ o, e ≠ .opened o) (s : String) :
    openOf (evs.push e) s = openOf evs s := by
  simp only [openOf, Array.findSome?_push]
  cases e with
  | opened o => exact absurd rfl (he o)
  | _ => exact Option.or_none

theorem stopOf_push_ne {evs : Events} {e : Event} (he : ∀ s' p w, e ≠ .stop s' p w) (s : String) :
    stopOf (evs.push e) s = stopOf evs s := by
  simp only [stopOf, Array.findSome?_push]
  cases e with
  | stop s' p w => exact absurd rfl (he s' p w)
  | _ => exact Option.or_none

theorem stopOf_push_stop_ne {evs : Events} {s s' : String} (p w : String) (h : s' ≠ s) :
    stopOf (evs.push (.stop s' p w)) s = stopOf evs s := by
  simp only [stopOf, Array.findSome?_push]
  have : (s' == s) = false := by simpa using h
  simp [this]

theorem stopOf_push_isSome {evs : Events} {s : String} (e : Event) (h : (stopOf evs s).isSome) :
    (stopOf (evs.push e) s).isSome := by
  obtain ⟨p, hp⟩ := Option.isSome_iff_exists.mp h
  simp only [stopOf, Array.findSome?_push] at hp ⊢
  rw [hp]
  rfl

theorem stopOf_push_stop_isSome (evs : Events) (s p w : String) : (stopOf (evs.push (.stop s p w)) s).isSome := by
  rw [Properties.Firewall.stopOf_push_stop]
  cases stopOf evs s <;> rfl

theorem finished_push_ne {evs : Events} {e : Event} (he : ∀ s', e ≠ .finish s') (s : String) :
    finished (evs.push e) s = finished evs s := by
  simp only [finished, Array.any_push]
  cases e with
  | finish s' => exact absurd rfl (he s')
  | _ => simp

theorem finished_push_true {evs : Events} {s : String} (e : Event) (h : finished evs s = true) :
    finished (evs.push e) s = true := by
  simp only [finished, Array.any_push] at h ⊢
  rw [h]
  rfl

theorem finished_push_finish (evs : Events) (s' s : String) :
    finished (evs.push (.finish s')) s = (finished evs s || s' == s) := by
  simp only [finished, Array.any_push]

theorem relOf_push_ne {evs : Events} {e : Event} (he : ∀ s' p, e ≠ .release s' p) (s : String) :
    relOf (evs.push e) s = relOf evs s := by
  simp only [relOf]
  rw [Array.filterMap_push (by simp)]
  cases e with
  | release s' p => exact absurd rfl (he s' p)
  | _ => rfl

theorem relOf_push_release (evs : Events) (s' p s : String) :
    relOf (evs.push (.release s' p)) s = if s' = s then (relOf evs s).push p else relOf evs s := by
  simp only [relOf]
  rw [Array.filterMap_push (by simp)]
  by_cases h : s' = s
  · subst h; simp
  · have : (s' == s) = false := by simpa using h
    simp [this, h]

theorem mem_relOf_push {evs : Events} {s x : String} (e : Event) (h : x ∈ relOf evs s) : x ∈ relOf (evs.push e) s := by
  by_cases he : ∃ s' p, e = .release s' p
  · obtain ⟨s', p, rfl⟩ := he
    rw [relOf_push_release]
    split
    · exact Array.mem_push.mpr (Or.inl h)
    · exact h
  · rw [relOf_push_ne (fun s' p h' => he ⟨s', p, h'⟩)]
    exact h

/-! ## The slot's orphan on one more event -/

theorem getElem!_push_size (evs : Events) (e : Event) : (evs.push e)[evs.size]! = e := by
  rw [getElem!_pos (evs.push e) evs.size (by simp)]
  simp

theorem getElem!_push_lt {evs : Events} (e : Event) {k : ℕ} (hk : k < evs.size) : (evs.push e)[k]! = evs[k]! := by
  rw [getElem!_pos (evs.push e) k (by simp; omega), getElem!_pos evs k hk, Array.getElem_push_lt hk]

theorem findSome?_congr {α β : Type} {f g : α → Option β} :
    ∀ {l : List α}, (∀ x ∈ l, f x = g x) → l.findSome? f = l.findSome? g
  | [], _ => rfl
  | x :: l, h => by
    simp only [List.findSome?_cons]
    rw [h x (List.mem_cons_self ..), findSome?_congr fun y hy => h y (List.mem_cons_of_mem _ hy)]

/-- The record's last `slot` event, as `orphan` finds it. -/
def lastSlot (evs : Events) : Option (ℕ × String × String) :=
  (List.range evs.size).reverse.findSome? fun k => match evs[k]! with
    | .slot s p => some (k, s, p)
    | _ => none

/-- Whether `e` is an outcome of `s` at `p`, as `orphan` reads it. -/
def doneEv (s p : String) : Event → Bool
  | .release s' p' | .refused s' p' => s' == s && p' == p
  | .stop s' _ _ => s' == s
  | _ => false

/-- Whether an outcome of `s` at `p` follows position `k`. -/
def doneAfter (evs : Events) (k : ℕ) (s p : String) : Bool := (evs.extract (k + 1) evs.size).any (doneEv s p)

theorem orphan_eq (evs : Events) :
    orphan evs = match lastSlot evs with
      | none => none
      | some (k, s, p) => if doneAfter evs k s p then none else some (k, s, p) := rfl

theorem lastSlot_lt {evs : Events} {k : ℕ} {s p : String} (h : lastSlot evs = some (k, s, p)) : k < evs.size := by
  obtain ⟨a, ha, hf⟩ := List.exists_of_findSome?_eq_some h
  have ha' : a < evs.size := by simpa using ha
  revert hf
  split
  · intro hf
    cases hf
    exact ha'
  · intro hf
    cases hf

theorem lastSlot_push_slot (evs : Events) (s p : String) : lastSlot (evs.push (.slot s p)) = some (evs.size, s, p) := by
  simp only [lastSlot, Array.size_push, List.range_succ, List.reverse_append, List.reverse_cons, List.reverse_nil,
    List.nil_append, List.cons_append, List.findSome?_cons, getElem!_push_size]

theorem lastSlot_push_ne {evs : Events} {e : Event} (he : ∀ s p, e ≠ .slot s p) :
    lastSlot (evs.push e) = lastSlot evs := by
  simp only [lastSlot, Array.size_push, List.range_succ, List.reverse_append, List.reverse_cons, List.reverse_nil,
    List.nil_append, List.cons_append, List.findSome?_cons, getElem!_push_size]
  cases e with
  | slot s p => exact absurd rfl (he s p)
  | _ =>
  try simp only
  apply findSome?_congr
  intro k hk
  have hk' : k < evs.size := by simpa using hk
  rw [getElem!_push_lt _ hk']

theorem orphan_push_slot (evs : Events) (s p : String) : orphan (evs.push (.slot s p)) = some (evs.size, s, p) := by
  rw [orphan_eq, lastSlot_push_slot]
  simp only [doneAfter, Array.size_push]
  rw [Array.extract_empty_of_size_le_start (by simp)]
  simp

/-- An event after the orphan's slot leaves it the orphan, unless it is an outcome of its session at its part. -/
theorem orphan_push_ne {evs : Events} {e : Event} (he : ∀ s p, e ≠ .slot s p) {k : ℕ} {s p : String}
    (h : orphan evs = some (k, s, p)) :
    orphan (evs.push e) = some (k, s, p) ∨ (∃ p' w, e = .stop s p' w) ∨ e = .release s p ∨ e = .refused s p := by
  rw [orphan_eq] at h ⊢
  rw [lastSlot_push_ne he]
  cases hl : lastSlot evs with
  | none => rw [hl] at h; cases h
  | some t =>
    obtain ⟨k', s', p'⟩ := t
    rw [hl] at h
    simp only at h ⊢
    have hk := lastSlot_lt hl
    cases hd : doneAfter evs k' s' p'
    · rw [hd] at h
      simp only [Bool.false_eq_true, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨h1, h2, h3⟩ := h
      subst k' s' p'
      have hk1 : k + 1 ≤ evs.size := hk
      have hpush : doneAfter (evs.push e) k s p = (doneAfter evs k s p || doneEv s p e) := by
        simp only [doneAfter, Array.size_push]
        rw [Array.extract_push]
        simp only [Nat.not_succ_le_self, ↓reduceIte, hk1, Array.any_push]
      rw [hpush, hd, Bool.false_or]
      cases e with
      | release s' p' =>
        by_cases hs : s' = s ∧ p' = p
        · obtain ⟨rfl, rfl⟩ := hs; exact Or.inr (Or.inr (Or.inl rfl))
        · left
          have : (s' == s && p' == p) = false := by
            simp only [Bool.and_eq_false_iff, beq_eq_false_iff_ne, ne_eq]
            by_cases h1 : s' = s
            · exact Or.inr fun h2 => hs ⟨h1, h2⟩
            · exact Or.inl h1
          simp [doneEv, this]
      | refused s' p' =>
        by_cases hs : s' = s ∧ p' = p
        · obtain ⟨rfl, rfl⟩ := hs; exact Or.inr (Or.inr (Or.inr rfl))
        · left
          have : (s' == s && p' == p) = false := by
            simp only [Bool.and_eq_false_iff, beq_eq_false_iff_ne, ne_eq]
            by_cases h1 : s' = s
            · exact Or.inr fun h2 => hs ⟨h1, h2⟩
            · exact Or.inl h1
          simp [doneEv, this]
      | stop s' p'' w =>
        by_cases hs : s' = s
        · subst hs; exact Or.inr (Or.inl ⟨p'', w, rfl⟩)
        · left
          have : (s' == s) = false := by simpa using hs
          simp [doneEv, this]
      | slot s' p' => exact absurd rfl (he s' p')
      | _ => left; simp [doneEv]
    · rw [hd] at h
      simp at h

/-! ## The ops' events -/

theorem openSession_evs (evs : Events) (stream : String) (parts : Array String) (budget : Option ℕ)
    (release : Option Core.Firewall.Schedule) :
    (openSession evs stream parts budget release).2 = evs ∨
      ∃ s, (openSession evs stream parts budget release).2 =
        evs.push (.opened { session := s, stream, parts, budget, release }) := by
  unfold openSession openSession.go openSession.made
  simp only
  repeat' split
  all_goals first | exact Or.inl rfl | exact Or.inr ⟨_, rfl⟩

theorem stop_evs (evs : Events) (s part why : String) :
    (stop evs s part why).2 = if (stopOf evs s).isSome then evs else evs.push (.stop s part why) := by
  by_cases h : (stopOf evs s).isSome <;> simp [Core.Firewall.stop, h]

theorem stop_evs_none {evs : Events} {s : String} (part why : String) (h : stopOf evs s = none) :
    (stop evs s part why).2 = evs.push (.stop s part why) := by
  rw [stop_evs, h]
  rfl

theorem slot_evs (evs : Events) (s part : String) :
    (slot evs s part = evs.push (.slot s part) ∨
      ∃ k s' p' w, orphan evs = some (k, s', p') ∧ stopOf evs s' = none ∧
        slot evs s part = (evs.push (.stop s' p' w)).push (.slot s part)) ∧
    ∀ k s' p', orphan evs = some (k, s', p') → (stopOf (slot evs s part) s').isSome := by
  refine ⟨?_, fun k s' p' ho => ?_⟩
  · unfold Core.Firewall.slot
    cases ho : orphan evs with
    | none => exact Or.inl rfl
    | some t =>
      obtain ⟨k, s', p'⟩ := t
      dsimp only
      cases hst : stopOf evs s' with
      | none =>
        exact Or.inr ⟨k, s', p', _, rfl, hst, rfl⟩
      | some _ => exact Or.inl rfl
  · unfold Core.Firewall.slot
    rw [ho]
    dsimp only
    apply stopOf_push_isSome
    cases hst : stopOf evs s' with
    | none => exact stopOf_push_stop_isSome evs s' p' _
    | some _ => simp only [Option.isNone_some, Bool.false_eq_true, ↓reduceIte, hst, Option.isSome_some]

/-- `admit` refusing records the refusal, after the stop at the next statement if the part was asked out of turn. -/
theorem admit_error {evs : Events} {s part why : String} {evs' : Events} (h : admit evs s part = (.error why, evs')) :
    evs' = evs.push (.refused s part) ∨ ∃ p w, evs' = (evs.push (.stop s p w)).push (.refused s part) := by
  unfold admit admit.served admit.refuse at h
  simp only at h
  repeat' split at h
  all_goals
    first
    | (simp only [Prod.mk.injEq] at h; exact Or.inl h.2.symm)
    | (simp only [Prod.mk.injEq] at h; exact Or.inr ⟨_, _, h.2.symm⟩)
    | simp at h

theorem after_go {a : Admitted} {now : ℕ} {prove ready : Core.Firewall.Run} {a' : Admitted}
    (h : Decision.after a now prove ready = .go a') : a' = a :=
  (Properties.Firewall.afterOnTime a now prove ready a' h).1

/-- What `partsOk` admits: at least one statement, and distinct ones. -/
theorem partsOk_ok {parts : Array String} (h : partsOk parts = .ok ()) :
    0 < parts.size ∧ ∀ i j (hi : i < parts.size) (hj : j < parts.size), parts[i] = parts[j] → i = j := by
  unfold partsOk at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hc
    have hne : parts.isEmpty = false := by
      revert hc; cases parts.isEmpty <;> simp
    have hd : ((List.range parts.size).all fun i =>
        (List.range parts.size).all fun j => i == j || parts[i]! != parts[j]!) = true := by
      revert hc
      cases ((List.range parts.size).all fun i =>
        (List.range parts.size).all fun j => i == j || parts[i]! != parts[j]!) <;> simp
    refine ⟨?_, fun i j hi hj hij => ?_⟩
    · have h0 : parts.size ≠ 0 := fun h0 => by
        have := Array.isEmpty_iff_size_eq_zero.mpr h0
        rw [hne] at this
        exact absurd this (by decide)
      omega
    · rw [List.all_eq_true] at hd
      have hi' := hd i (List.mem_range.mpr hi)
      rw [List.all_eq_true] at hi'
      have hij' := hi' j (List.mem_range.mpr hj)
      cases hb : (i == j)
      · rw [hb, Bool.false_or, getElem!_pos parts i hi, getElem!_pos parts j hj, hij] at hij'
        simp at hij'
      · exact beq_iff_eq.mp hb

theorem openOf_session {evs : Events} {s : String} {o : Open} (h : openOf evs s = some o) : o.session = s := by
  obtain ⟨e, -, he⟩ := Array.exists_of_findSome?_eq_some h
  revert he
  split
  · rename_i o'
    intro he
    cases hs : (o'.session == s)
    · rw [hs] at he
      cases he
    · rw [hs] at he
      cases he
      exact beq_iff_eq.mp hs
  · intro he
    cases he

theorem nextOf_push_ne {evs : Events} {e : Event} (he : ∀ s' p, e ≠ .release s' p) (o : Open) :
    nextOf (evs.push e) o = nextOf evs o := by
  rw [nextOf_eq, nextOf_eq, relOf_push_ne he]

/-- `parts[j]` released, by `nextOf`'s order: every statement before the next is released, and the next is not. -/
theorem nextOf_some {evs : Events} {o : Open} {i : ℕ} (h : nextOf evs o = some i) :
    ∃ hi : i < o.parts.size, o.parts[i] ∉ relOf evs o.session ∧
      ∀ j (hj : j < i), o.parts[j]'(Nat.lt_trans hj hi) ∈ relOf evs o.session := by
  rw [nextOf_eq] at h
  obtain ⟨hi, hp, hlt⟩ := Array.findIdx?_eq_some_iff_getElem.mp h
  refine ⟨hi, fun hm => ?_, fun j hj => ?_⟩
  · have : (relOf evs o.session).contains o.parts[i] = true := Array.contains_iff_mem.mpr hm
    rw [this] at hp
    cases hp
  · have := hlt j hj
    cases hc : (relOf evs o.session).contains (o.parts[j]'(Nat.lt_trans hj hi))
    · rw [hc] at this; exact absurd rfl this
    · exact Array.contains_iff_mem.mp hc

theorem nextOf_none {evs : Events} {o : Open} (h : nextOf evs o = none) :
    ∀ j (hj : j < o.parts.size), o.parts[j] ∈ relOf evs o.session := by
  rw [nextOf_eq] at h
  intro j hj
  have := Array.findIdx?_eq_none_iff.mp h o.parts[j] (Array.getElem_mem hj)
  cases hc : (relOf evs o.session).contains o.parts[j]
  · rw [hc] at this; cases this
  · exact Array.contains_iff_mem.mp hc

theorem nOf_some {σ : St} {s : String} {o : Open} (h : openOf σ.evs s = some o) : nOf σ s = o.parts.size + 1 := by
  simp [nOf, h]

theorem nOf_none {σ : St} {s : String} (h : openOf σ.evs s = none) : nOf σ s = 0 := by
  simp [nOf, h]

/-! ## The invariant -/

/-- The record `evs'` leaves every recursive session's open, finish and releases as `evs` has them, and keeps its stop. -/
structure Same (evs evs' : Events) : Prop where
  open_ : ∀ s, openOf evs' s = openOf evs s
  fin : ∀ s, finished evs' s = finished evs s
  rel : ∀ s, relOf evs' s = relOf evs s
  stop : ∀ s, (stopOf evs s).isSome → (stopOf evs' s).isSome

theorem Same.refl (evs : Events) : Same evs evs := ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl, fun _ h => h⟩

theorem Same.trans {a b c : Events} (h1 : Same a b) (h2 : Same b c) : Same a c :=
  ⟨fun s => (h2.open_ s).trans (h1.open_ s), fun s => (h2.fin s).trans (h1.fin s),
    fun s => (h2.rel s).trans (h1.rel s), fun s h => h2.stop s (h1.stop s h)⟩

theorem Same.push {evs : Events} {e : Event} (h1 : ∀ o, e ≠ .opened o) (h2 : ∀ s, e ≠ .finish s)
    (h3 : ∀ s p, e ≠ .release s p) : Same evs (evs.push e) :=
  ⟨openOf_push_ne h1, finished_push_ne h2, relOf_push_ne h3, fun _ h => stopOf_push_isSome e h⟩

theorem Same.nextOf {evs evs' : Events} (h : Same evs evs') (o : Open) : nextOf evs' o = nextOf evs o := by
  rw [nextOf_eq, nextOf_eq, h.rel]

theorem Same.pushStop (evs : Events) (s p w : String) : Same evs (evs.push (.stop s p w)) :=
  Same.push (fun _ h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)

theorem Same.pushRefused (evs : Events) (s p : String) : Same evs (evs.push (.refused s p)) :=
  Same.push (fun _ h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)

theorem Same.pushSlot (evs : Events) (s p : String) : Same evs (evs.push (.slot s p)) :=
  Same.push (fun _ h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)

theorem Same.pushWait (evs : Events) (s p : String) : Same evs (evs.push (.wait s p)) :=
  Same.push (fun _ h => by cases h) (fun _ h => by cases h) (fun _ _ h => by cases h)

theorem Same.slot (evs : Events) (s part : String) : Same evs (slot evs s part) := by
  rcases (slot_evs evs s part).1 with h | ⟨k, s', p', w, -, -, h⟩
  · rw [h]; exact Same.pushSlot _ _ _
  · rw [h]; exact (Same.pushStop _ _ _ _).trans (Same.pushSlot _ _ _)

/-- Statement `j` of `s` is done in `σ`: released, or `s` stopped, or, the slot free, the slot's orphan is of `s`, whose
stop the next `slot` records. -/
def Done (σ : St) (s : String) (j : ℕ) : Prop :=
  (∃ o, openOf σ.evs s = some o ∧ ∃ hj : j < o.parts.size, o.parts[j] ∈ relOf σ.evs s) ∨
    (stopOf σ.evs s).isSome ∨ (σ.holder = none ∧ ∃ k p, orphan σ.evs = some (k, s, p))

/-- The holder `h` in `σ`: admitted at its recursive session's next statement, after its inner session finished and
before any stop, its slot the record's last with no outcome after it, its statement admitted once. -/
def Holds (σ : St) (h : Holder) : Prop :=
  ∃ o, openOf σ.evs h.session = some o ∧
    ∃ hj : h.admitted.index < o.parts.size, o.parts[h.admitted.index] = h.part ∧
    nextOf σ.evs o = some h.admitted.index ∧ finished σ.evs h.session = true ∧ stopOf σ.evs h.session = none ∧
    h.admitted.releaseMs = releaseMs o.release h.admitted.index ∧
    (∃ k, orphan σ.evs = some (k, h.session, h.part)) ∧ σ.adm h.session h.stmt = 1

/-- **The invariant of a run.** -/
structure Inv (M : String → ℕ → ℕ) (σ : St) : Prop where
  le : ∀ s p, σ.sent s p ≤ M s p
  past : ∀ s p, nOf σ s ≤ p → σ.sent s p = 0
  pre : ∀ s q p, q < p → σ.sent s p ≠ 0 → σ.sent s q = M s q
  fin : ∀ s, finished σ.evs s = true → σ.sent s 0 = M s 0
  rel : ∀ s x, x ∈ relOf σ.evs s → ∃ o, openOf σ.evs s = some o ∧ finished σ.evs s = true ∧
    ∃ j, ∃ hj : j < o.parts.size, o.parts[j] = x ∧ σ.sent s (j + 1) = M s (j + 1)
  parts : ∀ s o, openOf σ.evs s = some o → 0 < o.parts.size ∧
    ∀ i j (hi : i < o.parts.size) (hj : j < o.parts.size), o.parts[i] = o.parts[j] → i = j
  hold : ∀ h, σ.holder = some h → Holds σ h
  admLe : ∀ s p, σ.adm s p ≤ 1
  adm0 : ∀ s, σ.adm s 0 = 0
  admDone : ∀ s j, σ.adm s (j + 1) = 1 →
    (∃ h, σ.holder = some h ∧ h.session = s ∧ h.admitted.index = j) ∨ Done σ s j
  leftAt : ∀ s j t, σ.left s (j + 1) = some t → ∃ o, openOf σ.evs s = some o ∧
    (o.release = none ∨ releaseMs o.release j = some t)

theorem Inv.init (M : String → ℕ → ℕ) : Inv M St.init where
  le _ _ := Nat.zero_le _
  past _ _ _ := rfl
  pre _ _ _ _ h := absurd rfl h
  fin _ h := by simp [finished, St.init] at h
  rel _ _ h := by simp [relOf, St.init] at h
  parts _ _ h := by simp [openOf, St.init] at h
  hold _ h := by cases h
  admLe _ _ := Nat.zero_le _
  adm0 _ := rfl
  admDone _ _ h := by cases h
  leftAt _ _ _ h := by cases h

/-- What carries over to `σ'` from `σ`, its record the same up to stops and its counts and times as they were. -/
theorem Inv.same {M : String → ℕ → ℕ} {σ σ' : St} (hI : Inv M σ) (he : Same σ.evs σ'.evs) (hs : σ'.sent = σ.sent)
    (hl : σ'.left = σ.left) (hold : ∀ h, σ'.holder = some h → Holds σ' h) (admLe : ∀ s p, σ'.adm s p ≤ 1)
    (adm0 : ∀ s, σ'.adm s 0 = 0)
    (admDone : ∀ s j, σ'.adm s (j + 1) = 1 →
      (∃ h, σ'.holder = some h ∧ h.session = s ∧ h.admitted.index = j) ∨ Done σ' s j) : Inv M σ' where
  le s p := hs ▸ hI.le s p
  past s p h := by
    rw [hs]
    apply hI.past
    simpa [nOf, he.open_] using h
  pre s q p := hs ▸ hI.pre s q p
  fin s h := hs ▸ hI.fin s (he.fin s ▸ h)
  rel s x h := by
    rw [he.rel] at h
    obtain ⟨o, ho, hf, j, hj, hx, hm⟩ := hI.rel s x h
    exact ⟨o, (he.open_ s).trans ho, (he.fin s).trans hf, j, hj, hx, hs ▸ hm⟩
  parts s o h := hI.parts s o ((he.open_ s).symm.trans h)
  hold := hold
  admLe := admLe
  adm0 := adm0
  admDone := admDone
  leftAt s j t h := by
    rw [hl] at h
    obtain ⟨o, ho, hr⟩ := hI.leftAt s j t h
    exact ⟨o, (he.open_ s).trans ho, hr⟩

/-- A done statement stays done across a record the same up to stops, the slot's holder as it was, if the record's orphan
stays or its session stops. -/
theorem Done.same {σ σ' : St} {s : String} {j : ℕ} (hd : Done σ s j) (he : Same σ.evs σ'.evs)
    (hh : σ'.holder = σ.holder)
    (ho : ∀ k p, σ.holder = none → orphan σ.evs = some (k, s, p) →
      orphan σ'.evs = some (k, s, p) ∨ (stopOf σ'.evs s).isSome) : Done σ' s j := by
  rcases hd with ⟨o, hO, hj, hm⟩ | hst | ⟨hn, k, p, hor⟩
  · exact Or.inl ⟨o, (he.open_ s).trans hO, hj, (he.rel s).symm ▸ hm⟩
  · exact Or.inr (Or.inl (he.stop s hst))
  · rcases ho k p hn hor with h | h
    · exact Or.inr (Or.inr ⟨hh.trans hn, k, p, h⟩)
    · exact Or.inr (Or.inl h)

end Flock.StopCount
