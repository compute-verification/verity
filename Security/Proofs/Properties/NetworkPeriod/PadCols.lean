import Properties.NetworkPeriod

/-!
# `padCols` counts the input columns no port covers and no forced zero holds

`padCols` marks each port's columns in one pass (`padMark_getD`, `padMarks_getD`); `padCols_eq` states its count as the
filter over the unit's input columns with `SampledProofs.Bits.srcKey`'s search for a port, beside the rows' test for a
forced zero, so a bound on the columns no port covers and no forced zero holds reads `padCols`. It needs nothing of the
ports: a mark past the input columns marks none of them, and a column marked twice counts once.
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period

theorem padMark_size (a0 : ℕ) (l : List ℕ) (h : Array Bool) :
    (l.foldl (fun h t => h.set! (a0 + t) true) h).size = h.size := by
  induction l generalizing h with
  | nil => rfl
  | cons t l ih => simp only [List.foldl_cons]; rw [ih]; simp

theorem padMark_getD (a0 : ℕ) (l : List ℕ) (h : Array Bool) (c : ℕ) :
    (l.foldl (fun h t => h.set! (a0 + t) true) h)[c]?.getD false = true ↔
      h[c]?.getD false = true ∨ (c < h.size ∧ ∃ t ∈ l, a0 + t = c) := by
  induction l generalizing h with
  | nil => simp
  | cons t l ih =>
    simp only [List.foldl_cons]
    rw [ih]
    simp only [Array.set!_eq_setIfInBounds, Array.size_setIfInBounds, Array.getElem?_setIfInBounds, List.mem_cons,
      exists_eq_or_imp]
    by_cases h1 : a0 + t = c
    · subst h1
      by_cases h2 : a0 + t < h.size <;> simp [h2]
    · simp [h1]

theorem pad_range_add (a0 w c : ℕ) : (∃ t ∈ List.range w, a0 + t = c) ↔ a0 ≤ c ∧ c < a0 + w := by
  constructor
  · rintro ⟨t, ht, rfl⟩
    rw [List.mem_range] at ht
    omega
  · rintro ⟨h1, h2⟩
    exact ⟨c - a0, List.mem_range.2 (by omega), by omega⟩

theorem padMarks_getD (ps : List (ℕ × ℕ)) (h : Array Bool) (c : ℕ) :
    (ps.foldl (fun h q => (List.range q.2).foldl (fun h t => h.set! (q.1 + t) true) h) h)[c]?.getD false = true ↔
      h[c]?.getD false = true ∨ (c < h.size ∧ ∃ q ∈ ps, q.1 ≤ c ∧ c < q.1 + q.2) := by
  induction ps generalizing h with
  | nil => simp
  | cons q ps ih =>
    simp only [List.foldl_cons]
    rw [ih, padMark_getD, padMark_size, pad_range_add]
    constructor
    · rintro ((h1 | ⟨h2, h3⟩) | ⟨h4, q', hq', h5⟩)
      · exact Or.inl h1
      · exact Or.inr ⟨h2, q, List.mem_cons_self, h3⟩
      · exact Or.inr ⟨h4, q', List.mem_cons_of_mem _ hq', h5⟩
    · rintro (h1 | ⟨h2, q', hq', h5⟩)
      · exact Or.inl (Or.inl h1)
      · rcases List.mem_cons.1 hq' with rfl | hq'
        · exact Or.inl (Or.inr ⟨h2, h5⟩)
        · exact Or.inr ⟨h2, q', hq', h5⟩

theorem pad_not_eq {x y : Bool} {p : Prop} (hx : x = true ↔ p) (hy : y = true ↔ ¬ p) : (!x) = y := by
  cases x <;> cases y <;> simp_all

/-- **`padCols` is the count of the unit's input columns no port covers and no forced zero holds**: the columns below
`min (inWords · WORD) constPos` for which `srcKey`'s search over `Flock.portCols` finds no port, and, when the unit's
input rows were read, whose row is not empty in both A and B. -/
theorem padCols_eq (cf : Flock.Circuit) :
    padCols cf = ((List.range (min (cf.unit.inWords * Flock.WORD) cf.unit.constPos)).filter fun c =>
      ((List.range (Flock.portCols cf.unit.inGroups).size).find? fun p =>
        decide (((Flock.portCols cf.unit.inGroups).getD p (0, 0)).1 ≤ c ∧
          c < ((Flock.portCols cf.unit.inGroups).getD p (0, 0)).1 +
            ((Flock.portCols cf.unit.inGroups).getD p (0, 0)).2)).isNone &&
        (!(min (cf.unit.inWords * Flock.WORD) cf.unit.constPos ≤ cf.unit.a.rows &&
            min (cf.unit.inWords * Flock.WORD) cf.unit.constPos ≤ cf.unit.b.rows) ||
          !(cf.unit.a.row c).isEmpty || !(cf.unit.b.row c).isEmpty)).length := by
  unfold padCols
  dsimp only
  congr 1
  apply List.filter_congr
  intro c hc
  try dsimp only
  congr 1
  rw [List.mem_range] at hc
  generalize Flock.portCols cf.unit.inGroups = pc
  apply pad_not_eq (p := ∃ q ∈ pc.toList, q.1 ≤ c ∧ c < q.1 + q.2)
  · rw [Array.getD_eq_getD_getElem?, ← Array.foldl_toList, padMarks_getD]
    simp [hc]
  · rw [Option.isNone_iff_eq_none, List.find?_eq_none]
    constructor
    · rintro h ⟨q, hq, hcov⟩
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hq)
      exact h i (List.mem_range.2 hi) (by simpa [hi] using hcov)
    · intro h p hp hP
      rw [List.mem_range] at hp
      exact h ⟨pc[p], Array.getElem_mem_toList hp, by simpa [hp] using hP⟩

end Properties.NetworkPeriod
