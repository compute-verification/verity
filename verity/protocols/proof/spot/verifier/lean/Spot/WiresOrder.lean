module

public import Spot.WiresSpan

/-! **The order's closed form against its units** (`Wires` "The order"). An order's range meets a range of units
exactly when it holds one of them (`mem_ranksIn_iff`), so a wire's closed form, every range meeting the bounds on its
writers below the range they are read in, is its check unit by unit (`orderedWire_eq_ref`). A well-formed order's ranges
of a template tile its units, so each of its units lies in a range (`ranksIn_ne_nil`). -/

@[expose] public section

namespace Spot.Wires

open Circuits

theorem readerUnit_lt {w : Wire} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0) {i : Nat} (hi : i < w.units) :
    i / w.cuts < w.readerUnits := by
  unfold Wire.readerUnits
  have h2 : w.units / w.cuts * w.cuts = w.units := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)
  exact (Nat.div_lt_iff_lt_mul (by omega)).2 (by rw [h2]; exact hi)

theorem mem_ranksIn {o : List UnitRange} {t : String} {lo hi k : Nat} :
    k ∈ ranksIn o t lo hi ↔ ∃ p0 n, o[k]? = some (t, p0, n) ∧ max p0 lo < min (p0 + n) hi := by
  simp only [ranksIn, List.mem_map, List.mem_filter, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨⟨s, p0, n⟩, k'⟩, ⟨hm, rfl, h⟩, rfl⟩
    exact ⟨p0, n, List.mem_zipIdx_iff_getElem?.1 hm, h⟩
  · rintro ⟨p0, n, hk, h⟩
    exact ⟨((t, p0, n), k), ⟨List.mem_zipIdx_iff_getElem?.2 hk, rfl, h⟩, rfl⟩

/-- **An order's range meets units `[lo, hi)` exactly when it holds one of them.** -/
theorem mem_ranksIn_iff {o : List UnitRange} {t : String} {lo hi k : Nat} :
    k ∈ ranksIn o t lo hi ↔ ∃ u, lo ≤ u ∧ u < hi ∧ k ∈ ranksIn o t u (u + 1) := by
  simp only [mem_ranksIn]
  constructor
  · rintro ⟨p0, n, hk, h⟩
    exact ⟨max p0 lo, by omega, by omega, p0, n, hk, by omega⟩
  · rintro ⟨u, h1, h2, p0, n, hk, h⟩
    exact ⟨p0, n, hk, by omega⟩

theorem all_ranksIn {o : List UnitRange} {t : String} {lo hi a : Nat} :
    (ranksIn o t lo hi).all (· < a) = (List.range' lo (hi - lo)).all fun v => (ranksIn o t v (v + 1)).all (· < a) := by
  rw [Bool.eq_iff_iff]
  simp only [List.all_eq_true, List.mem_range'_1, decide_eq_true_eq]
  constructor
  · intro h v hv k hk
    exact h k (mem_ranksIn_iff.2 ⟨v, hv.1, by omega, hk⟩)
  · intro h k hk
    obtain ⟨v, hv1, hv2, hk'⟩ := mem_ranksIn_iff.1 hk
    exact h v ⟨hv1, by omega⟩ k hk'

/-- **The closed form is the check unit by unit.** -/
theorem orderedWire_eq_ref (o : List UnitRange) (w : Wire) : orderedWire o w = orderedWireRef o w := by
  unfold orderedWire orderedWireRef
  congr 1
  funext ⟨b, a⟩
  dsimp only
  rw [all_ranksIn]

theorem lookup_map_pair {α β : Type} [BEq α] [LawfulBEq α] (ts : List α) (f : α → β) (t : α) :
    (ts.map fun x => (x, f x)).lookup t = if t ∈ ts then some (f t) else none := by
  induction ts with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.lookup_cons, List.mem_cons]
    by_cases h : t = x
    · subst h; simp
    · simp [h, ih, beq_false_of_ne h]

theorem rangesOf_nil {o : List UnitRange} {t : String} (h : t ∉ o.map (·.1)) : rangesOf o t = [] := by
  unfold rangesOf
  rw [List.filter_eq_nil_iff]
  rintro ⟨b, a⟩ hm
  have : b.1 ∈ o.map (·.1) := List.mem_map.2 ⟨b, List.mem_of_getElem? (List.mem_zipIdx_iff_getElem?.1 hm), rfl⟩
  simp only [beq_iff_eq]
  rintro rfl
  exact h this

theorem index_lookup (o : List UnitRange) (t : String) : ((rangeIndex o).lookup t).getD [] = rangesOf o t := by
  unfold rangeIndex
  rw [lookup_map_pair]
  by_cases h : t ∈ (o.map (·.1)).eraseDups
  · simp [h]
  · have h' : t ∉ o.map (·.1) := fun hm => h (List.mem_eraseDups.2 hm)
    simp [h, rangesOf_nil h']

theorem ranksIn_eq (o : List UnitRange) (t : String) (lo hi : Nat) :
    ((rangesOf o t).filter fun ((_, p0, n), _) => decide (max p0 lo < min (p0 + n) hi)).map (·.2) = ranksIn o t lo hi := by
  unfold rangesOf ranksIn
  rw [List.filter_filter]
  congr 2
  funext ⟨⟨s, p0, n⟩, k⟩
  simp [Bool.and_comm]

/-- **The indexed closed form is the closed form.** -/
theorem orderedWireIn_eq (o : List UnitRange) (w : Wire) : orderedWireIn (rangeIndex o) w = orderedWire o w := by
  unfold orderedWireIn orderedWire
  simp only [index_lookup]
  have h : ∀ x : UnitRange × Nat, (w.readsIn x.1 = false) ∨ x.1.1 = w.reader := by
    intro x
    by_cases hx : x.1.1 = w.reader
    · exact Or.inr hx
    · left; unfold Wire.readsIn; simp [hx]
  simp only [ranksIn_eq]
  rw [show rangesOf o w.reader = o.zipIdx.filter (fun (b, _) => b.1 == w.reader) from rfl, List.all_filter]
  congr 1
  funext ⟨b, a⟩
  rcases h (b, a) with h | h
  · simp [h]
  · have h' : b.1 = w.reader := h
    simp [h']

theorem orderCodeIx_eq (c : Wiring) : orderCodeIx c = orderCode c orderedWire := by
  unfold orderCodeIx orderCode
  cases orderOf c with
  | error r => rfl
  | ok o => simp only [orderedWireIn_eq]

theorem tiles_cover : ∀ {bs : List (Nat × Nat)} {cur units u : Nat}, tiles bs cur units = true → cur ≤ u → u < units →
    ∃ p ∈ bs, p.1 ≤ u ∧ u < p.1 + p.2
  | [], cur, units, u, h, h1, h2 => by
    simp only [tiles, beq_iff_eq] at h
    omega
  | (p0, n) :: bs, cur, units, u, h, h1, h2 => by
    simp only [tiles, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    obtain ⟨⟨rfl, hn⟩, hr⟩ := h
    by_cases hu : u < p0 + n
    · exact ⟨(p0, n), List.mem_cons_self .., h1, hu⟩
    · obtain ⟨p, hp, hp'⟩ := tiles_cover hr (by omega) h2
      exact ⟨p, List.mem_cons_of_mem _ hp, hp'⟩

theorem lookup_mem {β : Type} {a : String} {b : β} : ∀ {l : List (String × β)}, l.lookup a = some b → (a, b) ∈ l
  | [], h => by simp [List.lookup] at h
  | (k, v) :: l, h => by
    simp only [List.lookup] at h
    split at h
    · rename_i hk
      cases h
      rw [beq_iff_eq.1 hk]
      exact List.mem_cons_self ..
    · exact List.mem_cons_of_mem _ (lookup_mem h)

/-- **A well-formed order covers every unit**: each unit of a template lies in one of its ranges. -/
theorem ranksIn_ne_nil {c : Wiring} {o : List UnitRange} (h : orderOk c o = true) {t : String} {tt : Template}
    (ht : c.templates.lookup t = some tt) {u : Nat} (hu : u < tt.units) : ranksIn o t u (u + 1) ≠ [] := by
  simp only [orderOk, Bool.and_eq_true, List.all_eq_true] at h
  have hti := h.2 (t, tt) (lookup_mem ht)
  obtain ⟨p, hp, hp1, hp2⟩ := tiles_cover hti (Nat.zero_le u) hu
  rw [List.mem_mergeSort, List.mem_map] at hp
  obtain ⟨⟨s, p0, n⟩, hm, rfl⟩ := hp
  rw [List.mem_filter, beq_iff_eq] at hm
  obtain ⟨hm, rfl⟩ := hm
  obtain ⟨k, hk⟩ := List.mem_iff_getElem?.1 hm
  intro hnil
  have : k ∈ ranksIn o s u (u + 1) := mem_ranksIn.2 ⟨p0, n, hk, by simp at hp1 hp2; omega⟩
  rw [hnil] at this
  exact List.not_mem_nil this

/-- Each unit of a template has a range in a well-formed order, one meeting it. -/
theorem rangeOf_some {c : Wiring} {o : List UnitRange} (h : orderOk c o = true) {t : String} {tt : Template}
    (ht : c.templates.lookup t = some tt) {u : Nat} (hu : u < tt.units) :
    ∃ k, rangeOf o t u = some k ∧ k ∈ ranksIn o t u (u + 1) := by
  unfold rangeOf
  cases hl : ranksIn o t u (u + 1) with
  | nil => exact absurd hl (ranksIn_ne_nil h ht hu)
  | cons k l => exact ⟨k, rfl, List.mem_cons_self ..⟩

theorem orderOf_ok {c : Wiring} {o : List UnitRange} (h : orderOf c = .ok o) : orderOk c o = true := by
  unfold orderOf at h
  split at h
  · split at h
    · cases h
      assumption
    · cases h
  · cases h

end Spot.Wires
