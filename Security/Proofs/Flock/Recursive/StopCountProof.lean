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

end Flock.StopCount
