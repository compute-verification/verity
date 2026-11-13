module

public import Spot.WiresSpan

/-! **The order's closed form against its units** (`Wires` "The order"). An order's range meets a range of units
exactly when it holds one of them (`mem_ranksIn_iff`), so a wire's closed form, every range meeting the bounds on its
writers below the range they are read in, is its check unit by unit (`orderedWire_eq_ref`). A well-formed order's ranges
of a template tile its units, so each of its units lies in a range (`ranksIn_ne_nil`). -/

@[expose] public section

namespace Spot.Wires

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
