module

public import Proofs.Properties.Firewall
public import Proofs.Flock.Recursive.StopCount
import all Init.Data.Repr

/-!
# A recursive session's stops, counted from the ZK server's rules: the proof

`Flock.SecurityProofs.RecStopsOfRun`, by an invariant of `StopCount.step` (`Inv`) over the run's ops, on the record's
rules as `Firewall.Record` writes them and as `Proofs.Properties.Firewall` proves them (`admit_ok`, `served_ok`,
`nextOf_lt`, `stopOf_push_stop`, `afterOnTime`). What one more event does to a recursive session's open, stop, finish,
released statements and the slot's orphan comes first. Its witnesses come last: `RecStopsOfRun_honest`, an honest run
that runs through and releases its statement at its release time, and `RecStopsOfRun_nontrivial`, a ZK server without
`admit` (`Witness.stepAll`), where what reached the auditor is no cut at the run's stop.
-/

@[expose] public section

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
    (stop evs s part why).2 =
      if (openOf evs s).isNone || (stopOf evs s).isSome then evs else evs.push (.stop s part why) := by
  by_cases h : (openOf evs s).isNone || (stopOf evs s).isSome <;> simp only [Core.Firewall.stop, h] <;> rfl

theorem stop_evs_none {evs : Events} {s : String} (part why : String) (ho : (openOf evs s).isSome = true)
    (h : stopOf evs s = none) : (stop evs s part why).2 = evs.push (.stop s part why) := by
  obtain ⟨o, ho'⟩ := Option.isSome_iff_exists.mp ho
  rw [stop_evs, h, ho']
  rfl

theorem slot_evs (evs : Events) (s part : String) :
    (slot evs s part = evs.push (.slot s part) ∨
      ∃ k s' p' w, orphan evs = some (k, s', p') ∧ (openOf evs s').isSome = true ∧ stopOf evs s' = none ∧
        slot evs s part = (evs.push (.stop s' p' w)).push (.slot s part)) ∧
    ∀ k s' p', orphan evs = some (k, s', p') → (openOf evs s').isSome = true →
      (stopOf (slot evs s part) s').isSome := by
  refine ⟨?_, fun k s' p' ho hos => ?_⟩
  · unfold Core.Firewall.slot
    cases ho : orphan evs with
    | none => exact Or.inl rfl
    | some t =>
      obtain ⟨k, s', p'⟩ := t
      dsimp only
      cases hst : stopOf evs s' with
      | none =>
        cases hos : (openOf evs s').isSome with
        | true =>
          exact Or.inr ⟨k, s', p', s!"the slot's holder ended before its outcome (slot {k})", rfl, hos, hst,
            by simp only [Option.isNone_none, Bool.and_self, ↓reduceIte]⟩
        | false => exact Or.inl rfl
      | some _ => exact Or.inl (by simp)
  · unfold Core.Firewall.slot
    rw [ho]
    dsimp only
    apply stopOf_push_isSome
    cases hst : stopOf evs s' with
    | none =>
      simp only [hos, Option.isNone_none, Bool.and_self, ↓reduceIte]
      exact stopOf_push_stop_isSome evs s' p' _
    | some _ => simp only [Option.isNone_some, Bool.and_false, Bool.false_eq_true, ↓reduceIte, hst, Option.isSome_some]

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
  rcases (slot_evs evs s part).1 with h | ⟨k, s', p', w, -, -, -, h⟩
  · rw [h]; exact Same.pushSlot _ _ _
  · rw [h]; exact (Same.pushStop _ _ _ _).trans (Same.pushSlot _ _ _)

/-- Statement `j` of `s` is done in `σ`: released, or `s` stopped, or, the slot free, the slot's orphan is of `s`,
opened, whose stop the next `slot` records. -/
def Done (σ : St) (s : String) (j : ℕ) : Prop :=
  (∃ o, openOf σ.evs s = some o ∧ ∃ hj : j < o.parts.size, o.parts[j] ∈ relOf σ.evs s) ∨
    (stopOf σ.evs s).isSome ∨
      (σ.holder = none ∧ (openOf σ.evs s).isSome = true ∧ ∃ k p, orphan σ.evs = some (k, s, p))

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

/-- What carries over to `σ'` from `σ`, its record the same up to stops and its counts as they were. -/
theorem Inv.same {M : String → ℕ → ℕ} {σ σ' : St} (hI : Inv M σ) (he : Same σ.evs σ'.evs) (hs : σ'.sent = σ.sent)
    (hl : ∀ s j t, σ'.left s (j + 1) = some t → ∃ o, openOf σ.evs s = some o ∧
      (o.release = none ∨ releaseMs o.release j = some t))
    (hold : ∀ h, σ'.holder = some h → Holds σ' h) (admLe : ∀ s p, σ'.adm s p ≤ 1)
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
    obtain ⟨o, ho, hr⟩ := hl s j t h
    exact ⟨o, (he.open_ s).trans ho, hr⟩

/-- A done statement stays done on a record that keeps the session's open, releases and stop, if the slot's orphan stays
(the slot still free) or the session stops. -/
theorem Done.grow {σ σ' : St} {s : String} {j : ℕ} (hd : Done σ s j)
    (hO : ∀ o, openOf σ.evs s = some o → openOf σ'.evs s = some o)
    (hR : ∀ x, x ∈ relOf σ.evs s → x ∈ relOf σ'.evs s)
    (hS : (stopOf σ.evs s).isSome → (stopOf σ'.evs s).isSome)
    (ho : ∀ k p, σ.holder = none → (openOf σ.evs s).isSome = true → orphan σ.evs = some (k, s, p) →
      (σ'.holder = none ∧ orphan σ'.evs = some (k, s, p)) ∨ (stopOf σ'.evs s).isSome) : Done σ' s j := by
  rcases hd with ⟨o, hO', hj, hm⟩ | hst | ⟨hn, hos, k, p, hor⟩
  · exact Or.inl ⟨o, hO o hO', hj, hR _ hm⟩
  · exact Or.inr (Or.inl (hS hst))
  · rcases ho k p hn hos hor with ⟨hn', h⟩ | h
    · obtain ⟨o, ho'⟩ := Option.isSome_iff_exists.mp hos
      exact Or.inr (Or.inr ⟨hn', by rw [hO o ho']; rfl, k, p, h⟩)
    · exact Or.inr (Or.inl h)

theorem Done.same {σ σ' : St} {s : String} {j : ℕ} (hd : Done σ s j) (he : Same σ.evs σ'.evs)
    (ho : ∀ k p, σ.holder = none → (openOf σ.evs s).isSome = true → orphan σ.evs = some (k, s, p) →
      (σ'.holder = none ∧ orphan σ'.evs = some (k, s, p)) ∨ (stopOf σ'.evs s).isSome) : Done σ' s j :=
  hd.grow (fun _ h => (he.open_ s).trans h) (fun _ h => (he.rel s).symm ▸ h) (he.stop s) ho

/-- The holder stays the holder on a record that keeps its session's open, next statement, finish, stop and orphan. -/
theorem Holds.grow {σ σ' : St} {h : Holder} (hh : Holds σ h)
    (hO : ∀ o, openOf σ.evs h.session = some o → openOf σ'.evs h.session = some o)
    (hN : ∀ o, nextOf σ'.evs o = nextOf σ.evs o)
    (hF : finished σ.evs h.session = true → finished σ'.evs h.session = true)
    (hS : stopOf σ'.evs h.session = stopOf σ.evs h.session)
    (ho : ∀ k, orphan σ.evs = some (k, h.session, h.part) → orphan σ'.evs = some (k, h.session, h.part))
    (ha : σ'.adm h.session h.stmt = σ.adm h.session h.stmt) : Holds σ' h := by
  obtain ⟨o, hO', hj, hp, hn, hf, hst, hr, ⟨k, hor⟩, hadm⟩ := hh
  exact ⟨o, hO o hO', hj, hp, (hN o).trans hn, hF hf, hS.trans hst, hr, ⟨k, ho k hor⟩, ha.trans hadm⟩

theorem orphan_push_keep {evs : Events} {e : Event} {k : ℕ} {s p : String} (h : orphan evs = some (k, s, p))
    (h1 : ∀ s p, e ≠ .slot s p) (h2 : ∀ p' w, e ≠ .stop s p' w) (h3 : e ≠ .release s p) (h4 : e ≠ .refused s p) :
    orphan (evs.push e) = some (k, s, p) := by
  rcases orphan_push_ne h1 h with h | ⟨p', w, h⟩ | h | h
  · exact h
  · exact absurd h (h2 p' w)
  · exact absurd h h3
  · exact absurd h h4

theorem orphan_slot (evs : Events) (s part : String) : ∃ k, orphan (slot evs s part) = some (k, s, part) := by
  rcases (slot_evs evs s part).1 with h | ⟨k, s', p', w, -, -, -, h⟩
  · exact ⟨_, h ▸ orphan_push_slot _ _ _⟩
  · exact ⟨_, h ▸ orphan_push_slot _ _ _⟩

theorem bump_eq (f : String → ℕ → ℕ) (s : String) (p : ℕ) : bump f s p s p = f s p + 1 := by
  simp [bump]

theorem bump_ne {f : String → ℕ → ℕ} {s s' : String} {p p' : ℕ} (h : ¬(s' = s ∧ p' = p)) :
    bump f s p s' p' = f s' p' := by
  simp [bump, h]

theorem put_eq {α : Type} (f : String → ℕ → α) (s : String) (p : ℕ) (a : α) : put f s p a s p = a := by
  simp [put]

theorem put_ne {α : Type} {f : String → ℕ → α} {s s' : String} {p p' : ℕ} {a : α} (h : ¬(s' = s ∧ p' = p)) :
    put f s p a s' p' = f s' p' := by
  simp [put, h]

/-! ## Each op keeps the invariant -/

variable {M : String → ℕ → ℕ}

theorem Inv.wait {σ : St} (hI : Inv M σ) (s part : String) : Inv M (step M σ (.wait s part)) := by
  show Inv M { σ with evs := σ.evs.push (.wait s part) }
  have he := Same.pushWait σ.evs s part
  have hor : ∀ k s' p', orphan σ.evs = some (k, s', p') →
      orphan (σ.evs.push (.wait s part)) = some (k, s', p') := fun _ _ _ h =>
    orphan_push_keep h (fun _ _ h => by cases h) (fun _ _ h => by cases h) (fun h => by cases h) (fun h => by cases h)
  refine hI.same he rfl (fun s j t h => hI.leftAt s j t h) (fun h hh => ?_) hI.admLe hI.adm0 (fun s' j ha => ?_)
  · exact (hI.hold h hh).grow (fun o h => (he.open_ _).trans h) he.nextOf (fun h => (he.fin _).trans h)
      (stopOf_push_ne (fun _ _ _ h => by cases h) _) (fun k => hor k _ _) rfl
  · rcases hI.admDone s' j ha with hh | hd
    · exact Or.inl hh
    · exact Or.inr (hd.same he fun k p hn _ h => Or.inl ⟨hn, hor _ _ _ h⟩)

theorem Inv.pushOpened {σ : St} (hI : Inv M σ) (o' : Open) (hp : partsOk o'.parts = .ok ()) :
    Inv M { σ with evs := σ.evs.push (.opened o') } := by
  have hO : ∀ s o, openOf σ.evs s = some o → openOf (σ.evs.push (.opened o')) s = some o :=
    fun _ _ h => openOf_push_some _ h
  have hOn : ∀ s o, openOf (σ.evs.push (.opened o')) s = some o → openOf σ.evs s = some o ∨ o = o' := by
    intro s o h
    cases h0 : openOf σ.evs s with
    | some o0 => rw [openOf_push_some _ h0] at h; exact Or.inl h
    | none =>
      right
      simp only [openOf, Array.findSome?_push] at h h0
      rw [h0] at h
      simp only [Option.none_or] at h
      cases hs : (o'.session == s)
      · rw [hs] at h; cases h
      · rw [hs] at h; cases h; rfl
  have hF : ∀ s, finished (σ.evs.push (.opened o')) s = finished σ.evs s :=
    finished_push_ne (fun _ h => by cases h)
  have hR : ∀ s, relOf (σ.evs.push (.opened o')) s = relOf σ.evs s := relOf_push_ne (fun _ _ h => by cases h)
  have hS : ∀ s, stopOf (σ.evs.push (.opened o')) s = stopOf σ.evs s := stopOf_push_ne (fun _ _ _ h => by cases h)
  have hN : ∀ o, nextOf (σ.evs.push (.opened o')) o = nextOf σ.evs o := nextOf_push_ne (fun _ _ h => by cases h)
  have hor : ∀ k s' p', orphan σ.evs = some (k, s', p') →
      orphan (σ.evs.push (.opened o')) = some (k, s', p') := fun _ _ _ h =>
    orphan_push_keep h (fun _ _ h => by cases h) (fun _ _ h => by cases h) (fun h => by cases h) (fun h => by cases h)
  exact {
    le := hI.le
    past := fun s p h => by
      cases h0 : openOf σ.evs s with
      | some o =>
        apply hI.past
        have := nOf_some (σ := { σ with evs := σ.evs.push (.opened o') }) (hO s o h0)
        rw [nOf_some h0]
        omega
      | none => exact hI.past s p (by rw [nOf_none h0]; exact Nat.zero_le _)
    pre := hI.pre
    fin := fun s h => hI.fin s ((hF s).symm.trans h)
    rel := fun s x h => by
      rw [hR] at h
      obtain ⟨o, ho, hf, j, hj, hx, hm⟩ := hI.rel s x h
      exact ⟨o, hO s o ho, (hF s).trans hf, j, hj, hx, hm⟩
    parts := fun s o h => by
      rcases hOn s o h with h | rfl
      · exact hI.parts s o h
      · exact partsOk_ok hp
    hold := fun h hh => (hI.hold h hh).grow (hO _) hN (fun h => (hF _).trans h) (hS _) (fun k => hor k _ _) rfl
    admLe := hI.admLe
    adm0 := hI.adm0
    admDone := fun s j ha => by
      rcases hI.admDone s j ha with hh | hd
      · exact Or.inl hh
      · exact Or.inr (hd.grow (hO s) (fun x h => (hR s).symm ▸ h) (fun h => (hS s).symm ▸ h)
          fun k p hn _ h => Or.inl ⟨hn, hor _ _ _ h⟩)
    leftAt := fun s j t h => by
      obtain ⟨o, ho, hr⟩ := hI.leftAt s j t h
      exact ⟨o, hO s o ho, hr⟩ }

theorem Inv.openRec {σ : St} (hI : Inv M σ) (stream : String) (parts : Array String) (budget : Option ℕ)
    (release : Option Core.Firewall.Schedule) : Inv M (step M σ (.openRec stream parts budget release)) := by
  dsimp only [step]
  cases hp : partsOk parts with
  | error _ => exact hI
  | ok u =>
    cases u
    cases hb : budgetArg budget with
    | error _ => exact hI
    | ok u =>
      cases u
      show Inv M { σ with evs := (openSession σ.evs stream parts budget release).2 }
      rcases openSession_evs σ.evs stream parts budget release with h | ⟨sid, h⟩
      · rw [h]; exact hI
      · rw [h]; exact hI.pushOpened _ hp

theorem Inv.innerSend {σ : St} (hI : Inv M σ) (s : String) : Inv M (step M σ (.innerSend s)) := by
  dsimp only [step]
  split
  · rename_i hg
    rw [Bool.and_eq_true, Bool.and_eq_true] at hg
    obtain ⟨⟨hO, hF⟩, hlt⟩ := hg
    have hF' : finished σ.evs s = false := by simpa using hF
    have hlt' : σ.sent s 0 < M s 0 := of_decide_eq_true hlt
    obtain ⟨o, hO'⟩ := Option.isSome_iff_exists.mp hO
    exact {
      le := fun s' p => by
        by_cases hq : s' = s ∧ p = 0
        · obtain ⟨rfl, rfl⟩ := hq
          show bump σ.sent s' 0 s' 0 ≤ _
          rw [bump_eq]; exact hlt'
        · show bump σ.sent s 0 s' p ≤ _
          rw [bump_ne hq]; exact hI.le s' p
      past := fun s' p h => by
        show bump σ.sent s 0 s' p = 0
        by_cases hq : s' = s ∧ p = 0
        · obtain ⟨rfl, rfl⟩ := hq
          have := nOf_some (σ := σ) hO'
          simp only [nOf] at h this
          omega
        · rw [bump_ne hq]; exact hI.past s' p h
      pre := fun s' q p hqp hne => by
        show bump σ.sent s 0 s' q = M s' q
        have hp0 : ¬(s' = s ∧ p = 0) := fun h => by omega
        change bump σ.sent s 0 s' p ≠ 0 at hne
        rw [bump_ne hp0] at hne
        by_cases hq : s' = s ∧ q = 0
        · obtain ⟨rfl, rfl⟩ := hq
          have := hI.pre s' 0 p hqp hne
          omega
        · rw [bump_ne hq]; exact hI.pre s' q p hqp hne
      fin := fun s' h => by
        show bump σ.sent s 0 s' 0 = M s' 0
        have hs : ¬(s' = s ∧ 0 = 0) := fun ⟨hs, _⟩ => by subst hs; rw [hF'] at h; cases h
        rw [bump_ne hs]; exact hI.fin s' h
      rel := fun s' x h => by
        obtain ⟨o, ho, hf, j, hj, hx, hm⟩ := hI.rel s' x h
        refine ⟨o, ho, hf, j, hj, hx, ?_⟩
        show bump σ.sent s 0 s' (j + 1) = _
        rw [bump_ne (fun h => by omega)]; exact hm
      parts := hI.parts
      hold := hI.hold
      admLe := hI.admLe
      adm0 := hI.adm0
      admDone := hI.admDone
      leftAt := hI.leftAt }
  · exact hI

theorem Inv.innerFinish {σ : St} (hI : Inv M σ) (s : String) : Inv M (step M σ (.innerFinish s)) := by
  dsimp only [step]
  split
  · rename_i hg
    rw [Bool.and_eq_true, Bool.and_eq_true] at hg
    obtain ⟨⟨-, -⟩, heq⟩ := hg
    have heq' : σ.sent s 0 = M s 0 := of_decide_eq_true heq
    have hO : ∀ s' o, openOf σ.evs s' = some o → openOf (finish σ.evs s) s' = some o := fun _ _ h =>
      openOf_push_some _ h
    have hO' : ∀ s', openOf (finish σ.evs s) s' = openOf σ.evs s' := openOf_push_ne (fun _ h => by cases h)
    have hR : ∀ s', relOf (finish σ.evs s) s' = relOf σ.evs s' := relOf_push_ne (fun _ _ h => by cases h)
    have hS : ∀ s', stopOf (finish σ.evs s) s' = stopOf σ.evs s' := stopOf_push_ne (fun _ _ _ h => by cases h)
    have hN : ∀ o, nextOf (finish σ.evs s) o = nextOf σ.evs o := nextOf_push_ne (fun _ _ h => by cases h)
    have hF : ∀ s', finished σ.evs s' = true → finished (finish σ.evs s) s' = true := fun _ h =>
      finished_push_true _ h
    have hor : ∀ k s' p', orphan σ.evs = some (k, s', p') → orphan (finish σ.evs s) = some (k, s', p') :=
      fun _ _ _ h => orphan_push_keep h (fun _ _ h => by cases h) (fun _ _ h => by cases h) (fun h => by cases h)
        (fun h => by cases h)
    exact {
      le := hI.le
      past := fun s' p h => hI.past s' p (by simpa [nOf, hO'] using h)
      pre := hI.pre
      fin := fun s' h => by
        change finished (σ.evs.push (.finish s)) s' = true at h
        rw [finished_push_finish] at h
        by_cases hs : s = s'
        · subst hs; exact heq'
        · have : (s == s') = false := by simpa using hs
          rw [this, Bool.or_false] at h
          exact hI.fin s' h
      rel := fun s' x h => by
        change x ∈ relOf (finish σ.evs s) s' at h
        rw [hR] at h
        obtain ⟨o, ho, hf, j, hj, hx, hm⟩ := hI.rel s' x h
        exact ⟨o, hO s' o ho, hF s' hf, j, hj, hx, hm⟩
      parts := fun s' o h => hI.parts s' o ((hO' s').symm.trans h)
      hold := fun h hh => (hI.hold h hh).grow (hO _) hN (hF _) (hS _) (fun k => hor k _ _) rfl
      admLe := hI.admLe
      adm0 := hI.adm0
      admDone := fun s' j ha => by
        rcases hI.admDone s' j ha with hh | hd
        · exact Or.inl hh
        · exact Or.inr (hd.grow (hO s') (fun x h => (hR s').symm ▸ h) (fun h => (hS s').symm ▸ h)
            fun k p hn _ h => Or.inl ⟨hn, hor _ _ _ h⟩)
      leftAt := fun s' j t h => by
        obtain ⟨o, ho, hr⟩ := hI.leftAt s' j t h
        exact ⟨o, hO s' o ho, hr⟩ }
  · exact hI

theorem Inv.innerStop {σ : St} (hI : Inv M σ) (s why : String) : Inv M (step M σ (.innerStop s why)) := by
  dsimp only [step]
  split
  · rename_i hg
    rw [Bool.and_eq_true] at hg
    obtain ⟨-, hF⟩ := hg
    have hF' : finished σ.evs s = false := by simpa using hF
    rw [stop_evs]
    split
    · exact hI
    · have he := Same.pushStop σ.evs s INNER why
      refine hI.same he rfl (fun s j t h => hI.leftAt s j t h) (fun h hh => ?_) hI.admLe hI.adm0
        (fun s' j ha => ?_)
      · have hh' := hI.hold h hh
        have hne : h.session ≠ s := by
          intro hs
          obtain ⟨-, -, -, -, -, hf, -⟩ := hh'
          rw [hs, hF'] at hf
          cases hf
        exact hh'.grow (fun o h => (he.open_ _).trans h) he.nextOf (fun h => (he.fin _).trans h)
          (stopOf_push_stop_ne _ _ (Ne.symm hne)) (fun k h => orphan_push_keep (e := .stop s INNER why) h
            (fun _ _ h => by cases h) (fun _ _ h => by cases h; exact hne rfl) (fun h => by cases h)
            (fun h => by cases h)) rfl
      · rcases hI.admDone s' j ha with hh | hd
        · exact Or.inl hh
        · refine Or.inr (hd.same he fun k p hn _ h => ?_)
          rcases orphan_push_ne (e := .stop s INNER why) (fun _ _ h => by cases h) h with h' | ⟨p', w, h'⟩ | h' | h'
          · exact Or.inl ⟨hn, h'⟩
          · cases h'; exact Or.inr (stopOf_push_stop_isSome _ _ _ _)
          · cases h'
          · cases h'
  · exact hI

theorem Inv.seat {σ : St} (hI : Inv M σ) (s part : String) : Inv M (step M σ (.seat s part)) := by
  dsimp only [step]
  split
  · exact hI
  · rename_i hh
    have hS1 := Same.slot σ.evs s part
    have hst1 := (slot_evs σ.evs s part).2
    obtain ⟨k1, hor1⟩ := orphan_slot σ.evs s part
    split
    · rename_i a evs2 hadm
      obtain ⟨o, j, hO1, hS1', hN1, hserved⟩ := Properties.Firewall.admit_ok hadm
      obtain ⟨hF1, hp, rfl, rfl⟩ := Properties.Firewall.served_ok hserved
      have hj := Properties.Firewall.nextOf_lt hN1
      have hO : openOf σ.evs s = some o := (hS1.open_ s).symm.trans hO1
      have hses := openOf_session hO
      have hp' : o.parts[j] = part := by rw [← getElem!_pos o.parts j hj]; exact hp
      have hadm0 : σ.adm s (j + 1) = 0 := by
        have hle := hI.admLe s (j + 1)
        rcases Nat.lt_or_ge (σ.adm s (j + 1)) 1 with h | h
        · omega
        · exfalso
          rcases hI.admDone s j (by omega) with ⟨h', hh', -⟩ | hd
          · rw [hh] at hh'; cases hh'
          · rcases hd with ⟨o', hO', hj', hm⟩ | hst | ⟨-, -, k, p, hor⟩
            · rw [hO] at hO'
              cases hO'
              obtain ⟨_, hn, _⟩ := nextOf_some hN1
              rw [hses] at hn
              exact hn ((hS1.rel s).symm ▸ hm)
            · have := hS1.stop s hst
              rw [hS1'] at this
              cases this
            · have := hst1 k s p hor (by rw [hO]; rfl)
              rw [hS1'] at this
              cases this
      refine hI.same hS1 rfl (fun s j t h => hI.leftAt s j t h) (fun h' hh' => ?_) (fun s' p => ?_) (fun s' => ?_)
        (fun s' j' ha => ?_)
      · cases hh'
        refine ⟨o, hO1, hj, hp', hN1, hF1, hS1', rfl, ⟨k1, hor1⟩, ?_⟩
        show bump σ.adm s (j + 1) s (j + 1) = 1
        rw [bump_eq, hadm0]
      · show bump σ.adm s (j + 1) s' p ≤ 1
        by_cases hq : s' = s ∧ p = j + 1
        · obtain ⟨rfl, rfl⟩ := hq
          rw [bump_eq]; omega
        · rw [bump_ne hq]; exact hI.admLe s' p
      · show bump σ.adm s (j + 1) s' 0 = 0
        have hq : ¬(s' = s ∧ 0 = j + 1) := fun h => Nat.succ_ne_zero _ h.2.symm
        rw [bump_ne hq]
        exact hI.adm0 s'
      · change bump σ.adm s (j + 1) s' (j' + 1) = 1 at ha
        by_cases hq : s' = s ∧ j' + 1 = j + 1
        · exact Or.inl ⟨_, rfl, hq.1.symm, by show j = j'; omega⟩
        · rw [bump_ne hq] at ha
          rcases hI.admDone s' j' ha with ⟨h', hh', -⟩ | hd
          · rw [hh] at hh'; cases hh'
          · exact Or.inr (hd.same hS1 fun k p _ hos hor => Or.inr (hst1 k s' p hor hos))
    · rename_i w evs2 hadm
      have hS2 : Same (slot σ.evs s part) evs2 := by
        rcases admit_error hadm with h | ⟨p, w', h⟩
        · rw [h]; exact Same.pushRefused _ _ _
        · rw [h]; exact (Same.pushStop _ _ _ _).trans (Same.pushRefused _ _ _)
      have hS := hS1.trans hS2
      refine hI.same hS rfl (fun s j t h => hI.leftAt s j t h) (fun h hh' => ?_) hI.admLe hI.adm0 (fun s' j ha => ?_)
      · change σ.holder = some h at hh'
        rw [hh] at hh'; cases hh'
      · rcases hI.admDone s' j ha with ⟨h', hh', -⟩ | hd
        · rw [hh] at hh'; cases hh'
        · exact Or.inr (hd.same hS fun k p _ hos hor => Or.inr (hS2.stop s' (hst1 k s' p hor hos)))

theorem Inv.relay {σ : St} (hI : Inv M σ) : Inv M (step M σ .relay) := by
  dsimp only [step]
  split
  · rename_i h hh
    split
    · rename_i hlt
      obtain ⟨o, hO, hj, -, hN, hF, -, -, -, -⟩ := hI.hold h hh
      have hses := openOf_session hO
      have hn := nOf_some (σ := σ) hO
      have hpre : ∀ q, q < h.stmt → σ.sent h.session q = M h.session q := by
        intro q hq
        cases q with
        | zero => exact hI.fin _ hF
        | succ j =>
          have hj' : j < h.admitted.index := by simp only [Holder.stmt] at hq; omega
          obtain ⟨_, _, hrel⟩ := nextOf_some hN
          have hm := hrel j hj'
          rw [hses] at hm
          obtain ⟨o', hO', -, j'', hj'', hx, hsent⟩ := hI.rel _ _ hm
          rw [hO] at hO'
          cases hO'
          have := (hI.parts _ o hO).2 j'' j hj'' (by omega) hx
          subst this
          exact hsent
      exact {
        le := fun s' p => by
          show bump σ.sent h.session h.stmt s' p ≤ _
          by_cases hq : s' = h.session ∧ p = h.stmt
          · obtain ⟨rfl, rfl⟩ := hq; rw [bump_eq]; exact hlt
          · rw [bump_ne hq]; exact hI.le s' p
        past := fun s' p hp' => by
          show bump σ.sent h.session h.stmt s' p = 0
          by_cases hq : s' = h.session ∧ p = h.stmt
          · obtain ⟨rfl, rfl⟩ := hq
            exfalso
            change nOf σ h.session ≤ h.stmt at hp'
            rw [hn] at hp'
            simp only [Holder.stmt] at hp'
            omega
          · rw [bump_ne hq]; exact hI.past s' p hp'
        pre := fun s' q p hqp hne => by
          show bump σ.sent h.session h.stmt s' q = M s' q
          change bump σ.sent h.session h.stmt s' p ≠ 0 at hne
          by_cases hP : s' = h.session ∧ p = h.stmt
          · obtain ⟨rfl, rfl⟩ := hP
            rw [bump_ne (fun h' => by have := h'.2; omega)]
            exact hpre q hqp
          · rw [bump_ne hP] at hne
            by_cases hQ : s' = h.session ∧ q = h.stmt
            · obtain ⟨rfl, rfl⟩ := hQ
              have := hI.pre _ _ p hqp hne
              omega
            · rw [bump_ne hQ]; exact hI.pre s' q p hqp hne
        fin := fun s' hf => by
          show bump σ.sent h.session h.stmt s' 0 = _
          have hq : ¬(s' = h.session ∧ 0 = h.stmt) := fun h' => Nat.succ_ne_zero _ h'.2.symm
          rw [bump_ne hq]
          exact hI.fin s' hf
        rel := fun s' x hx => by
          obtain ⟨o', hO', hf, j', hj', hx', hm⟩ := hI.rel s' x hx
          refine ⟨o', hO', hf, j', hj', hx', ?_⟩
          show bump σ.sent h.session h.stmt s' (j' + 1) = _
          by_cases hq : s' = h.session ∧ j' + 1 = h.stmt
          · obtain ⟨rfl, hq⟩ := hq
            rw [hq] at hm
            omega
          · rw [bump_ne hq]; exact hm
        parts := hI.parts
        hold := hI.hold
        admLe := hI.admLe
        adm0 := hI.adm0
        admDone := hI.admDone
        leftAt := hI.leftAt }
    · exact hI
  · exact hI

/-- The holder's stop, recorded: its statement is done, the slot free. -/
theorem Inv.stopHolder {σ : St} (hI : Inv M σ) {h : Holder} (hh : σ.holder = some h) (why : String) :
    Inv M { σ with evs := (stop σ.evs h.session h.part why).2, holder := none } := by
  obtain ⟨o, hO, hj, hp, hN, hF, hS, -, -, -⟩ := hI.hold h hh
  rw [stop_evs_none _ _ (by rw [hO]; rfl) hS]
  have he := Same.pushStop σ.evs h.session h.part why
  refine hI.same he rfl (fun s j t h => hI.leftAt s j t h) (fun h' hh' => by cases hh') hI.admLe hI.adm0
    (fun s j ha => ?_)
  rcases hI.admDone s j ha with ⟨h', hh', hs, -⟩ | hd
  · rw [hh] at hh'
    cases hh'
    subst hs
    exact Or.inr (Or.inr (Or.inl (stopOf_push_stop_isSome _ _ _ _)))
  · exact Or.inr (hd.same he fun k p hn _ _ => by rw [hh] at hn; cases hn)

/-- The holder's release, recorded once all its statement's messages reached the auditor: its statement is done, the
slot free. -/
theorem Inv.released {σ : St} (hI : Inv M σ) {h : Holder} (hh : σ.holder = some h)
    (heq : σ.sent h.session h.stmt = M h.session h.stmt) :
    Inv M { σ with evs := release σ.evs h.session h.part, holder := none } := by
  obtain ⟨o, hO, hj, hp, -, hF, -, -, -, -⟩ := hI.hold h hh
  have hO' : ∀ s, openOf (release σ.evs h.session h.part) s = openOf σ.evs s :=
    openOf_push_ne (fun _ h => by cases h)
  have hF' : ∀ s, finished (release σ.evs h.session h.part) s = finished σ.evs s :=
    finished_push_ne (fun _ h => by cases h)
  have hS' : ∀ s, (stopOf σ.evs s).isSome → (stopOf (release σ.evs h.session h.part) s).isSome :=
    fun _ h => stopOf_push_isSome _ h
  have hR : ∀ s x, x ∈ relOf σ.evs s → x ∈ relOf (release σ.evs h.session h.part) s :=
    fun _ _ h => mem_relOf_push _ h
  exact {
    le := hI.le
    past := fun s p hp' => hI.past s p (by simpa [nOf, hO'] using hp')
    pre := hI.pre
    fin := fun s hf => hI.fin s ((hF' s).symm.trans hf)
    rel := fun s x hx => by
      change x ∈ relOf (σ.evs.push (.release h.session h.part)) s at hx
      rw [relOf_push_release] at hx
      split at hx
      · rename_i hs
        subst hs
        rcases Array.mem_push.mp hx with hx | rfl
        · obtain ⟨o', ho', hf, j, hj', hx', hm⟩ := hI.rel _ x hx
          exact ⟨o', (hO' _).trans ho', (hF' _).trans hf, j, hj', hx', hm⟩
        · exact ⟨o, (hO' _).trans hO, (hF' _).trans hF, h.admitted.index, hj, hp, heq⟩
      · obtain ⟨o', ho', hf, j, hj', hx', hm⟩ := hI.rel s x hx
        exact ⟨o', (hO' _).trans ho', (hF' _).trans hf, j, hj', hx', hm⟩
    parts := fun s o' ho' => hI.parts s o' ((hO' s).symm.trans ho')
    hold := fun h' hh' => by cases hh'
    admLe := hI.admLe
    adm0 := hI.adm0
    admDone := fun s j ha => by
      rcases hI.admDone s j ha with ⟨h', hh', hs, hi⟩ | hd
      · rw [hh] at hh'
        cases hh'
        subst hs hi
        refine Or.inr (Or.inl ⟨o, (hO' _).trans hO, hj, ?_⟩)
        change o.parts[h.admitted.index] ∈ relOf (σ.evs.push (.release h.session h.part)) h.session
        rw [relOf_push_release]
        simp [hp]
      · exact Or.inr (hd.grow (fun o' ho' => (hO' _).trans ho') (hR s) (hS' s)
          fun k p hn _ _ => by rw [hh] at hn; cases hn)
    leftAt := fun s j t hl => by
      obtain ⟨o', ho', hr⟩ := hI.leftAt s j t hl
      exact ⟨o', (hO' _).trans ho', hr⟩ }

theorem Inv.barrier {σ : St} (hI : Inv M σ) (now : ℕ) (prove ready : Core.Firewall.Run) :
    Inv M (step M σ (.barrier now prove ready)) := by
  dsimp only [step]
  split
  · rename_i h hh
    split
    · exact hI
    · split
      next a hgo =>
        have ha := after_go hgo
        subst ha
        obtain ⟨o, hO, -, -, -, -, -, hr, -, -⟩ := hI.hold h hh
        refine hI.same (Same.refl _) rfl (fun s j t hl => ?_) (fun h' hh' => ?_) hI.admLe hI.adm0
          (fun s j ha => ?_)
        · by_cases hq : s = h.session ∧ j + 1 = h.stmt
          · obtain ⟨rfl, hq⟩ := hq
            have hj' : j = h.admitted.index := by simp only [Holder.stmt] at hq; omega
            subst hj'
            change put σ.left h.session h.stmt (some (h.admitted.releaseMs.getD now)) h.session h.stmt = some t at hl
            rw [put_eq, hr] at hl
            refine ⟨o, hO, ?_⟩
            cases hrel : o.release with
            | none => exact Or.inl rfl
            | some r =>
              right
              rw [hrel] at hl
              simpa [releaseMs] using hl
          · change put σ.left h.session h.stmt _ s (j + 1) = some t at hl
            rw [put_ne hq] at hl
            exact hI.leftAt s j t hl
        · cases hh'
          exact hI.hold h hh
        · rcases hI.admDone s j ha with ⟨h', hh', hs, hi⟩ | hd
          · rw [hh] at hh'
            cases hh'
            exact Or.inl ⟨_, rfl, hs, hi⟩
          · exact Or.inr (hd.same (Same.refl _) fun k p hn _ _ => by rw [hh] at hn; cases hn)
      all_goals exact hI.stopHolder hh _
  · exact hI

theorem Inv.verdict {σ : St} (hI : Inv M σ) (v : Except String Bool) : Inv M (step M σ (.verdict v)) := by
  cases v with
  | error why =>
    dsimp only [step]
    split
    · rename_i h hh
      split
      · exact hI.stopHolder hh why
      · exact hI
    · exact hI
  | ok b =>
    cases b with
    | false =>
      dsimp only [step]
      split
      · rename_i h hh
        split
        · exact hI.stopHolder hh _
        · exact hI
      · exact hI
    | true =>
      dsimp only [step]
      split
      · rename_i h hh
        split
        · rename_i hg
          rw [Bool.and_eq_true] at hg
          exact hI.released hh (of_decide_eq_true hg.2)
        · exact hI
      · exact hI

theorem Inv.cut {σ : St} (hI : Inv M σ) (why : String) : Inv M (step M σ (.cut why)) := by
  dsimp only [step]
  split
  · rename_i h hh
    split
    · exact hI
    · exact hI.stopHolder hh why
  · exact hI

theorem Inv.die {σ : St} (hI : Inv M σ) : Inv M (step M σ .die) := by
  show Inv M { σ with holder := none }
  refine hI.same (Same.refl _) rfl (fun s j t h => hI.leftAt s j t h) (fun h hh => by cases hh) hI.admLe hI.adm0
    (fun s j ha => ?_)
  rcases hI.admDone s j ha with ⟨h, hh, hs, -⟩ | hd
  · obtain ⟨o, hO, -, -, -, -, -, -, ⟨k, hor⟩, -⟩ := hI.hold h hh
    subst hs
    exact Or.inr (Or.inr (Or.inr ⟨rfl, by rw [hO]; rfl, k, h.part, hor⟩))
  · exact Or.inr (hd.same (Same.refl _) fun k p _ _ hor => Or.inl ⟨rfl, hor⟩)

theorem Inv.next {σ : St} (hI : Inv M σ) : ∀ st, Inv M (step M σ st)
  | .openRec stream parts budget release => hI.openRec stream parts budget release
  | .innerSend s => hI.innerSend s
  | .innerFinish s => hI.innerFinish s
  | .innerStop s why => hI.innerStop s why
  | .wait s part => hI.wait s part
  | .seat s part => hI.seat s part
  | .relay => hI.relay
  | .barrier now prove ready => hI.barrier now prove ready
  | .verdict v => hI.verdict v
  | .cut why => hI.cut why
  | .die => hI.die

/-- **Every run keeps the invariant.** -/
theorem Inv.ofRun (M : String → ℕ → ℕ) (steps : List Step) : Inv M (run M steps) := by
  have : ∀ (l : List Step) σ, Inv M σ → Inv M (l.foldl (step M) σ) := by
    intro l
    induction l with
    | nil => intro σ h; exact h
    | cons st rest ih => intro σ h; exact ih _ (h.next st)
  exact this steps _ (Inv.init M)

/-! ## The stop, read off the invariant -/

theorem recStop_none {σ : St} {s : String}
    (h : (List.finRange (nOf σ s)).find? (fun q : Fin (nOf σ s) => decide (σ.sent s q < M s q)) = none) :
    recStop M σ s = none := by
  rw [recStop, h]

theorem recStop_some {σ : St} {s : String} {q : Fin (nOf σ s)}
    (h : (List.finRange (nOf σ s)).find? (fun q : Fin (nOf σ s) => decide (σ.sent s q < M s q)) = some q) :
    recStop M σ s = some ⟨q, ⟨min (σ.sent s q) (M s q), Nat.lt_succ_of_le (Nat.min_le_right _ _)⟩⟩ := by
  rw [recStop, h]

/-- **What reached the auditor is the cut at the stop**: every message of each statement before the first one short of
its count, what reached it of that one, and nothing after it or past the session's statements. -/
theorem Inv.sent_cutAt {σ : St} (hI : Inv M σ) (s : String) (p : ℕ) :
    σ.sent s p = cutAt (fun p : Fin (nOf σ s) => M s p) (recStop M σ s) p := by
  unfold cutAt
  split
  · rename_i hp
    cases hf : (List.finRange (nOf σ s)).find? (fun q : Fin (nOf σ s) => decide (σ.sent s q < M s q)) with
    | none =>
      rw [recStop_none hf]
      have := List.find?_eq_none.mp hf ⟨p, hp⟩ (List.mem_finRange _)
      simp only [decide_eq_true_eq, Nat.not_lt] at this
      exact Nat.le_antisymm (hI.le s p) this
    | some q =>
      rw [recStop_some hf]
      obtain ⟨hq, i, hi, hiq, hmin⟩ := List.find?_eq_some_iff_getElem.mp hf
      have hlt : σ.sent s q < M s q := of_decide_eq_true hq
      have hiv : i = q.val := by
        rw [List.getElem_finRange] at hiq
        rw [← hiq]
        rfl
      show σ.sent s p = if p < q.val then M s p else if p = q.val then min (σ.sent s q) (M s q) else 0
      by_cases hpq : p < q.val
      · simp only [hpq, ↓reduceIte]
        have := hmin p (by omega)
        simp only [List.getElem_finRange, Fin.val_cast, Bool.not_eq_true', decide_eq_false_iff_not,
          Nat.not_lt] at this
        exact Nat.le_antisymm (hI.le s p) this
      · simp only [hpq, ↓reduceIte]
        by_cases hpe : p = q.val
        · simp only [hpe, ↓reduceIte]
          exact (Nat.min_eq_left (Nat.le_of_lt hlt)).symm
        · simp only [hpe, ↓reduceIte]
          refine Classical.byContradiction fun hne => ?_
          have := hI.pre s q p (by omega) hne
          omega
  · rename_i hp
    exact hI.past s p (Nat.le_of_not_lt hp)

/-- **A session that stops has a statement not released**: if every statement of `s` was released, every message of
every statement reached the auditor. -/
theorem Inv.runsThrough {σ : St} (hI : Inv M σ) (s : String) : recStop M σ s = none ∨ unreleasedOf σ s = true := by
  cases ho : openOf σ.evs s with
  | none =>
    left
    apply recStop_none
    apply List.find?_eq_none.mpr
    intro x _
    have := x.isLt
    have := nOf_none ho
    omega
  | some o =>
    cases hn : nextOf σ.evs o with
    | some i => right; simp [unreleasedOf, ho, hn]
    | none =>
      left
      have hall := nextOf_none hn
      rw [openOf_session ho] at hall
      have key : ∀ p, p < o.parts.size + 1 → σ.sent s p = M s p := by
        intro p hp
        cases p with
        | zero =>
          obtain ⟨-, -, hf, -⟩ := hI.rel s _ (hall 0 (hI.parts s o ho).1)
          exact hI.fin s hf
        | succ j =>
          have hj : j < o.parts.size := by omega
          obtain ⟨o', ho', -, j', hj', hx', hm⟩ := hI.rel s _ (hall j hj)
          rw [ho] at ho'
          cases ho'
          have := (hI.parts s o ho).2 j' j hj' hj hx'
          subst this
          exact hm
      apply recStop_none
      apply List.find?_eq_none.mpr
      intro x _
      have hx : x.val < o.parts.size + 1 := by
        have := x.isLt
        have := nOf_some ho
        omega
      simp [key x hx]

/-- **Proofs leave at their release time** where the session is scheduled. -/
theorem Inv.leftSched {σ : St} (hI : Inv M σ) (s : String) (j : ℕ) :
    schedOf σ s = none ∨ σ.left s (j + 1) = none ∨ σ.left s (j + 1) = releaseMs (schedOf σ s) j := by
  cases hl : σ.left s (j + 1) with
  | none => exact Or.inr (Or.inl rfl)
  | some t =>
    obtain ⟨o, ho, hr⟩ := hI.leftAt s j t hl
    have hsch : schedOf σ s = o.release := by simp [schedOf, ho]
    rw [hsch]
    rcases hr with hr | hr
    · exact Or.inl hr
    · exact Or.inr (Or.inr hr.symm)

end Flock.StopCount

namespace Flock.SecurityProofs

/-- **A recursive session's view is cut at its stop**: every run keeps the invariant, which reads off the stop. -/
theorem RecStopsOfRun : Flock.Guarantees.RecStopsOfRun := fun M steps s =>
  have hI := StopCount.Inv.ofRun M steps
  ⟨hI.sent_cutAt s, hI.admLe s, hI.runsThrough s, hI.leftSched s⟩

end Flock.SecurityProofs

namespace Flock.StopCount.Witness

open Core.Firewall

/-- One message a statement. -/
def one : String → ℕ → ℕ := fun _ _ => 1

/-- An honest run: a recursive session of one statement, scheduled to release it at 150 ms, whose inner session sends
its message and finishes, whose outer session takes the slot, is admitted, sends its message, passes its `Finish` at
its release time and is accepted. -/
def honest : List Step :=
  [.openRec "st" #["a"] none (some { start := 100, windows := #[50] }), .innerSend "r0000", .innerFinish "r0000",
   .seat "r0000" "a", .relay, .barrier 150 .ok .ok, .verdict (.ok true)]

/-- A ZK server without `admit`: an outer session that finds the slot free takes it, as statement `0`'s, whatever the
record says. -/
def stepAll (M : String → ℕ → ℕ) (σ : St) : Step → St
  | .seat s part =>
    match σ.holder with
    | some _ => σ
    | none => { σ with holder := some ⟨s, part, ⟨"", 0, none⟩, false⟩, adm := bump σ.adm s 1 }
  | st => step M σ st

def runAll (M : String → ℕ → ℕ) (steps : List Step) : St := steps.foldl (stepAll M) St.init

/-- At `stepAll`, an outer session takes the slot before the inner session sent anything, and sends its message. -/
def unordered : List Step := [.openRec "st" #["a"] none none, .seat "r0000" "a", .relay]

end Flock.StopCount.Witness

namespace Flock.SecurityProofs

open Core.Firewall Flock.StopCount Flock.StopCount.Witness

/-- **`RecStopsOfRun` says something at an honest run**: its session runs through (`recStop … = none`), every message of
both statements reached the auditor, nothing is left unreleased, and the statement's proofs left at its release time. -/
theorem RecStopsOfRun_honest :
    (recStop one (run one honest) "r0000").isNone = true ∧ (run one honest).sent "r0000" 0 = 1 ∧
      (run one honest).sent "r0000" 1 = 1 ∧ unreleasedOf (run one honest) "r0000" = false ∧
      (run one honest).left "r0000" 1 = releaseMs (schedOf (run one honest) "r0000") 0 ∧
      (run one honest).left "r0000" 1 = some 150 := by
  decide +kernel

/-- **`RecStopsOfRun` fails at a ZK server without `admit`**: there, statement `1`'s message reached the auditor while
none of the inner session's did, which is no cut at the run's stop. -/
theorem RecStopsOfRun_nontrivial :
    ¬ (runAll one unordered).sent "r0000" 1 =
      cutAt (fun p : Fin (nOf (runAll one unordered) "r0000") => one "r0000" p)
        (recStop one (runAll one unordered) "r0000") 1 := by
  decide +kernel

end Flock.SecurityProofs
