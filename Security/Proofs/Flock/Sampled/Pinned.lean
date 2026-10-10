import Proofs.Flock.Sampled.Defs
import VStar.Menu.Pinned

/-!
# F6's pinned menus as `Menu`s

A pinned menu's entries (`VStar.Menu.Pinned`, generated from F6's `menu.json`) give a `Menu`: its shapes are the
entries' (slots, nonzero bound), a shape's inner session is `VStar.Menu.inner`'s (the largest of its entries'), and its
type bound `VStar.Menu.types`'s. The padded sessions of an audit of `k` draws stay a parameter: `menu.json` gives the
session counts at its own `K` only.

* `Menu.ofEntries`: the menu of a list of entries, given its padded sessions.
* `Menu.ofEntries_inRange`, `Menu.ofEntries_staged`: `Menu.InRange` and `Menu.Staged` of such a menu from its entries'
  inner sessions, one by one.
* `pinned_inRange`: every pinned model's menu is in the live verifier's range, whatever its padded sessions.
-/

namespace FlockSampled

open VStar.Menu (Entry)

/-- The menu of the entries `es`, with padded sessions `padded`. -/
def Menu.ofEntries (es : List Entry) (padded : ℕ → List Shape) : Menu where
  shapes := (es.map fun e => (⟨e.slots, e.bound⟩ : Shape)).dedup
  inner s := (VStar.Menu.inner es s.slots s.nnz).getD default
  types s := VStar.Menu.types es s.slots s.nnz
  padded := padded

theorem foldl_max_mem (e : Entry) (rest : List Entry) : rest.foldl Entry.max e ∈ e :: rest := by
  induction rest generalizing e with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons]
    rcases List.mem_cons.1 (ih (Entry.max e x)) with h | h
    · rw [h]; unfold Entry.max; split <;> simp
    · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h)

/-- A shape on the menu has the inner session of one of its entries. -/
theorem Menu.ofEntries_inner {es : List Entry} {padded : ℕ → List Shape} {s : Shape}
    (hs : s ∈ (Menu.ofEntries es padded).shapes) :
    ∃ e ∈ es, VStar.Menu.inner es s.slots s.nnz = some e.inner := by
  obtain ⟨e, he, rfl⟩ := List.mem_map.1 (List.mem_dedup.1 hs)
  have hat : e ∈ VStar.Menu.at_ es e.slots e.bound := List.mem_filter.2 ⟨he, by simp⟩
  unfold VStar.Menu.inner
  generalize hl : VStar.Menu.at_ es e.slots e.bound = l at hat
  match l, hat with
  | x :: rest, _ =>
    refine ⟨rest.foldl Entry.max x, ?_, rfl⟩
    have hsub : x :: rest ⊆ es := fun y hy => (List.mem_filter.1 (hl ▸ hy)).1
    exact hsub (foldl_max_mem x rest)

/-- **A menu of entries is in the live verifier's range** when each entry's inner session is: `k_log ≤ 27` and a link
point's `PT_LOCAL + (m − k_log)` coordinates at most 64. -/
theorem Menu.ofEntries_inRange {es : List Entry} (h : ∀ e ∈ es, e.kLog ≤ 27 ∧ e.m ≤ e.kLog + 40)
    (padded : ℕ → List Shape) : (Menu.ofEntries es padded).InRange := by
  intro s hs
  obtain ⟨e, he, hi⟩ := Menu.ofEntries_inner hs
  have h' := h e he
  simp only [Menu.rangeCheck, Menu.ofEntries, hi, Option.getD_some, VStar.Menu.Entry.inner]
  exact (Flock.checkInRange_ok _ _ _).2 ⟨h'.1, by omega, by simp only [Flock.PT_LOCAL]; omega⟩

/-- **V* stages a menu of entries** when it stages each entry's inner session. -/
theorem Menu.ofEntries_staged {es : List Entry} (h : ∀ e ∈ es, ∃ vs, VStar.statements e.inner = .ok vs)
    (padded : ℕ → List Shape) : (Menu.ofEntries es padded).Staged := by
  intro s hs
  obtain ⟨e, he, hi⟩ := Menu.ofEntries_inner hs
  simpa only [Menu.vstar, Menu.ofEntries, hi, Option.getD_some] using h e he

/-- **Every pinned model's menu is in the live verifier's range**, whatever its padded sessions. -/
theorem pinned_inRange : ∀ p ∈ VStar.Menu.pinned, ∀ padded, (Menu.ofEntries p.2 padded).InRange := by
  have h : ∀ p ∈ VStar.Menu.pinned, ∀ e ∈ p.2, e.kLog ≤ 27 ∧ e.m ≤ e.kLog + 40 := by decide
  exact fun p hp => Menu.ofEntries_inRange (h p hp)

end FlockSampled
