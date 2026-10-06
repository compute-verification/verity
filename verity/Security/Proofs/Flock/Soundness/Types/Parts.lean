import Proofs.Flock.Soundness.Types.Dag

/-!
# The unit's parts, as `deriveChecked` checks them (the rows a template statement reads, for 1e's T3)

`Typed.templateOf` builds a template's statement from the unit `u = done[unit]`: its own rows (`rootOf`), its parts
`u.parts` at their bases, Δ's bindings (`u.delta`), and its own rows' entries at part columns (`crossOf`). `deriveChecked`
checks the part structure that reading needs (`partsChecked`), and these lemmas state it:
1. **The parts' regions** (`part_region`, `parts_apart`): part `(j, b)` holds the rows `[b, b + done[j].size)`, past the
   unit's own rows and inside its rows. Its callee is an earlier layout whose rows are its size, and the parts follow one
   another by base.
2. **A part's rows** (`part_rows`): each callee row reads only inside the callee, and sits at `b + c`, shifted, except the
   callee's constant, which is Δ's copy of `one`, and its bound inputs, which are Δ's bindings.
3. **Δ** (`delta_nodup`, `delta_entry`, `delta_const`, `callee_rows`): one entry per row, each the unit's constant, a
   part's constant or a part's bound input. Each callee's input rows are self rows, and its constant row is
   `[const]·[const]`.
4. **The own rows** (`own_reads`, `unit_const_row`, `order_cols`): each reads only own and part columns. So the entries
   `crossOf` takes, those at or past    `u.size`, are part columns. The constant row is `[const]·[const]`, and the order
   lists only own and part columns.
5. **A part's reads** (`part_reads`, `callee_prod`): the unit's reads cover a part's rows exactly as its callee's reads,
   shifted, cover the callee's rows. A covered callee row is a product row `a · []`, and its read's low minterms are
   callee rows. Both are about part rows only, so a flat unit's inline reads over its own rows are untouched.
-/

namespace FlockSoundness.Types.Dag

open Flock Flock.DeriveAll

theorem distinctFrom_spec : ∀ (l : List Nat) (seen : Std.HashSet Nat), distinctFrom l seen = true →
    (∀ c ∈ l, seen.contains c = false) ∧ l.Nodup
  | [], _, _ => ⟨by simp, List.nodup_nil⟩
  | c :: rest, seen, h => by
    unfold distinctFrom at h
    simp only [Bool.and_eq_true, Bool.not_eq_true'] at h
    obtain ⟨hc, hrest⟩ := h
    obtain ⟨hr1, hr2⟩ := distinctFrom_spec rest (seen.insert c) hrest
    refine ⟨fun x hx => ?_, List.nodup_cons.mpr ⟨fun hm => ?_, hr2⟩⟩
    · rcases List.mem_cons.mp hx with rfl | hx
      · exact hc
      · have := hr1 x hx
        rw [Std.HashSet.contains_insert] at this
        simp only [Bool.or_eq_false_iff] at this
        exact this.2
    · have := hr1 c hm
      rw [Std.HashSet.contains_insert] at this
      simp at this

/-- `partsChecked`, unfolded. -/
theorem partsChecked_spec {done : Array D} {unit : Nat} (h : partsChecked done unit = true) :
    (∀ p ∈ (done.getD unit default).parts, partOk done unit (done.getD unit default) p.1 p.2 = true) ∧
      (∀ pq ∈ (done.getD unit default).parts.toList.zip (done.getD unit default).parts.toList.tail,
        pq.1.2 + (done.getD pq.1.1 default).size ≤ pq.2.2) ∧
      ((done.getD unit default).delta.toList.map (·.1)).Nodup ∧
      (∀ e ∈ (done.getD unit default).delta, deltaEntryOk done (done.getD unit default) e = true) ∧
      (∀ c < (done.getD unit default).size,
        ∀ x ∈ ((done.getD unit default).rows.getD c ([], [])).1 ++ ((done.getD unit default).rows.getD c ([], [])).2,
          x < (done.getD unit default).size ∨ inPart done (done.getD unit default) x = true) ∧
      (done.getD unit default).rows.getD (done.getD unit default).const ([], []) =
        ([(done.getD unit default).const], [(done.getD unit default).const]) ∧
      (∀ c ∈ order done (done.getD unit default),
        c < (done.getD unit default).size ∨ inPart done (done.getD unit default) c = true) ∧
      ∀ p ∈ (done.getD unit default).parts, partReadsOk done (done.getD unit default) p.1 p.2 = true := by
  unfold partsChecked at h
  simp only [Bool.and_eq_true, Array.all_eq_true_iff_forall_mem, List.all_eq_true, decide_eq_true_eq,
    List.mem_range, Bool.or_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨hp, hz⟩, hd⟩, he⟩, ho⟩, hc⟩, hord⟩, hrd⟩ := h
  exact ⟨hp, hz, (distinctFrom_spec _ _ hd).2, he, ho, hc, hord, hrd⟩

variable {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
  {words : String → Option (Array Nat)} {unit : Nat} {done : Array D}

/-- **1. A part's region.** Part `(j, b)`'s callee is an earlier layout whose rows are its size; the part holds the rows
`[b, b + done[j].size)`, past the unit's own rows and inside its rows. -/
theorem part_region (hd : deriveChecked types ls info words unit = .ok done) {p : Nat × Nat}
    (hp : p ∈ (done.getD unit default).parts) :
    p.1 < unit ∧ (done.getD p.1 default).rows.size = (done.getD p.1 default).size ∧
      (done.getD unit default).size ≤ p.2 ∧ p.2 + (done.getD p.1 default).size ≤ (done.getD unit default).rows.size := by
  have h := (partsChecked_spec (deriveChecked_parts hd)).1 p hp
  unfold partOk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  exact ⟨h.1.1.1.1, h.1.1.1.2, h.1.1.2, h.1.2⟩

theorem apart_of_consecutive {u : D}
    (hz : ∀ pq ∈ u.parts.toList.zip u.parts.toList.tail, pq.1.2 + (done.getD pq.1.1 default).size ≤ pq.2.2) :
    ∀ {i k : Nat} (hik : i < k) (hk : k < u.parts.size),
      u.parts[i].2 + (done.getD u.parts[i].1 default).size ≤ u.parts[k].2 := by
  have step : ∀ m (hm : m + 1 < u.parts.size),
      u.parts[m].2 + (done.getD u.parts[m].1 default).size ≤ u.parts[m + 1].2 := by
    intro m hm
    have hlen : m < (u.parts.toList.zip u.parts.toList.tail).length := by simp; omega
    have := hz _ (List.getElem_mem hlen)
    simpa [List.getElem_zip, List.getElem_tail, Array.getElem_toList] using this
  intro i k hik hk
  induction k with
  | zero => omega
  | succ k ih =>
    rcases Nat.lt_succ_iff_lt_or_eq.mp hik with hlt | heq
    · have h1 := ih hlt (by omega)
      have h2 := step k hk
      omega
    · subst heq
      exact step i hk

/-- **1. The parts follow one another by base**: each ends at or before the next begins, so parts are disjoint and ordered
by base. -/
theorem parts_apart (hd : deriveChecked types ls info words unit = .ok done) {i k : Nat} (hik : i < k)
    (hk : k < (done.getD unit default).parts.size) :
    (done.getD unit default).parts[i].2 + (done.getD (done.getD unit default).parts[i].1 default).size ≤
      (done.getD unit default).parts[k].2 :=
  apart_of_consecutive (partsChecked_spec (deriveChecked_parts hd)).2.1 hik hk

/-- **2. A part's rows.** For part `(j, b)` and each callee row `c`: the row reads only inside the callee; the callee's
constant is Δ's copy of `one`; any other row is a bound input (Δ's binding, at one of the callee's input columns), or
sits at `b + c` as the callee's row shifted, with no Δ entry. -/
theorem part_rows (hd : deriveChecked types ls info words unit = .ok done) {p : Nat × Nat}
    (hp : p ∈ (done.getD unit default).parts) {c : Nat} (hc : c < (done.getD p.1 default).size) :
    (∀ x ∈ ((done.getD p.1 default).rows.getD c ([], [])).1 ++ ((done.getD p.1 default).rows.getD c ([], [])).2,
        x < (done.getD p.1 default).size) ∧
      (c = (done.getD p.1 default).const →
        (done.getD unit default).delta.find? (·.1 == p.2 + c) = some (p.2 + c, none)) ∧
      (c ≠ (done.getD p.1 default).const →
        (∃ s, (done.getD unit default).delta.find? (·.1 == p.2 + c) = some (p.2 + c, some s) ∧
            c ∈ (done.getD p.1 default).inCols) ∨
          ((done.getD unit default).delta.find? (·.1 == p.2 + c) = none ∧
            (done.getD unit default).rows[p.2 + c]? = some (shiftRow p.2 ((done.getD p.1 default).rows.getD c ([], []))))) := by
  have h := (partsChecked_spec (deriveChecked_parts hd)).1 p hp
  unfold partOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  obtain ⟨hw, hr⟩ := h.2 c hc
  refine ⟨fun x hx => ?_, fun hcc => ?_, fun hcc => ?_⟩
  · unfold rowWithin at hw
    simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hw
    rcases List.mem_append.mp hx with hx | hx
    · exact hw.1 x hx
    · exact hw.2 x hx
  · rw [ite_eq_left (by simpa using hcc)] at hr
    simpa using hr
  · rw [ite_eq_right (by simpa using hcc)] at hr
    split at hr
    · rename_i r s hf
      have hr' : r = p.2 + c := by simpa using Array.find?_some hf
      subst hr'
      exact Or.inl ⟨s, hf, Array.contains_iff_mem.mp hr⟩
    · cases hr
    · rename_i hf
      exact Or.inr ⟨hf, by simpa using hr⟩

/-- **3. Δ has one entry per row.** -/
theorem delta_nodup (hd : deriveChecked types ls info words unit = .ok done) :
    ((done.getD unit default).delta.toList.map (·.1)).Nodup :=
  (partsChecked_spec (deriveChecked_parts hd)).2.2.1

/-- **3. Every Δ entry** is the unit's constant, a part's constant (both `none`, copies of `one`), or a part's bound
input: a binding (`some`) at one of the callee's input columns. -/
theorem delta_entry (hd : deriveChecked types ls info words unit = .ok done) {e : Nat × Option (List Nat)}
    (he : e ∈ (done.getD unit default).delta) :
    (e.1 = (done.getD unit default).const ∧ e.2 = none) ∨
      ∃ p ∈ (done.getD unit default).parts,
        (e.1 = p.2 + (done.getD p.1 default).const ∧ e.2 = none) ∨
          (e.2.isSome ∧ ∃ x ∈ (done.getD p.1 default).inCols, e.1 = p.2 + x) := by
  have h := (partsChecked_spec (deriveChecked_parts hd)).2.2.2.1 e he
  unfold deltaEntryOk at h
  simp only [Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, Option.isNone_iff_eq_none, Array.any_eq_true'] at h
  rcases h with h | ⟨p, hp, h⟩
  · exact Or.inl h
  · refine Or.inr ⟨p, hp, ?_⟩
    rcases h with h | ⟨hs, x, hx, he'⟩
    · exact Or.inl h
    · exact Or.inr ⟨hs, x, hx, he'.symm⟩

/-- **3. The unit's constant** is Δ's copy of `one`. -/
theorem delta_const (hd : deriveChecked types ls info words unit = .ok done) (hu : unit < ls.size) :
    (done.getD unit default).delta.find? (·.1 == (done.getD unit default).const) =
      some ((done.getD unit default).const, none) := by
  have h := (deriveChecked_spec hd).1 unit hu
  cases hown : (ls.getD unit default).own with
  | derive =>
    cases ht : (ls.getD unit default).type.bind types with
    | none =>
      simp only [checkLayout, hown, ht, Bool.false_eq_true] at h
    | some t =>
      obtain ⟨-, -, -, -, hone, -⟩ := checkLayout_spec h ht
      simp only [beq_self_eq_true, Chk.oneAt, ↓reduceIte, beq_iff_eq] at hone
      exact hone
  | gen sha n vb lo => exact absurd rfl (genLayout_spec h hown).1

/-- **3. A callee's input rows are self rows, and its constant row is `[const]·[const]`**: for any layout other than the
unit, from its own check. -/
theorem callee_rows (hd : deriveChecked types ls info words unit = .ok done) {j : Nat} (hj : j < ls.size)
    (hju : j ≠ unit) :
    (∀ x ∈ (done.getD j default).inCols, (done.getD j default).rows[x]? = some ([x], [x])) ∧
      (done.getD j default).rows[(done.getD j default).const]? =
        some ([(done.getD j default).const], [(done.getD j default).const]) := by
  have h := (deriveChecked_spec hd).1 j hj
  have hk : (j == unit) = false := by simp [hju]
  cases hown : (ls.getD j default).own with
  | derive =>
    cases ht : (ls.getD j default).type.bind types with
    | none =>
      simp only [checkLayout, hown, ht, Bool.false_eq_true] at h
    | some t =>
      obtain ⟨-, -, hin, -, hone, hins, -⟩ := checkLayout_spec h ht
      rw [hk] at hone hins
      refine ⟨fun x hx => ?_, ?_⟩
      · obtain ⟨w, hw, rfl⟩ := Array.mem_iff_getElem.mp hx
        have := hins w (by omega)
        have hgd : (done.getD j default).inCols.getD w 0 = (done.getD j default).inCols[w] :=
          getD_of_getElem? (Array.getElem?_eq_getElem hw)
        rw [hgd] at this
        simp only [Chk.rowIs, Bool.and_eq_true, beq_iff_eq] at this
        exact this.1.1
      · simp only [Chk.oneAt, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, beq_iff_eq] at hone
        exact hone.1
  | gen sha n vb lo =>
    obtain ⟨-, hg⟩ := genLayout_spec h hown
    obtain ⟨hin, -, hone, -, -, -, hins⟩ := genOk_facts hg
    refine ⟨fun x hx => ?_, ?_⟩
    · have hx' : x < n := by rw [hin] at hx; simpa using hx
      have := hins x hx'
      simp only [Chk.rowIs, Bool.and_eq_true, beq_iff_eq] at this
      exact this.1.1
    · simp only [Chk.oneAt, Bool.false_eq_true, ↓reduceIte, Bool.and_eq_true, beq_iff_eq] at hone
      exact hone.1

/-- **4. The unit's own rows** read only own columns and part columns. -/
theorem own_reads (hd : deriveChecked types ls info words unit = .ok done) {c : Nat}
    (hc : c < (done.getD unit default).size) :
    ∀ x ∈ ((done.getD unit default).rows.getD c ([], [])).1 ++ ((done.getD unit default).rows.getD c ([], [])).2,
      x < (done.getD unit default).size ∨
        ∃ p ∈ (done.getD unit default).parts, p.2 ≤ x ∧ x < p.2 + (done.getD p.1 default).size := by
  intro x hx
  rcases (partsChecked_spec (deriveChecked_parts hd)).2.2.2.2.1 c hc x hx with h | h
  · exact Or.inl h
  · unfold inPart at h
    simp only [Array.any_eq_true', Bool.and_eq_true, decide_eq_true_eq] at h
    exact Or.inr h

/-- **The unit's constant row is `[const]·[const]`**, as a derived row: its root net's constant reads only itself, so
`crossOf` has no entry at it. -/
theorem unit_const_row (hd : deriveChecked types ls info words unit = .ok done) :
    (done.getD unit default).rows.getD (done.getD unit default).const ([], []) =
      ([(done.getD unit default).const], [(done.getD unit default).const]) :=
  (partsChecked_spec (deriveChecked_parts hd)).2.2.2.2.2.1

/-- **The unit's order lists only own and part columns.** -/
theorem order_cols (hd : deriveChecked types ls info words unit = .ok done) :
    ∀ c ∈ order done (done.getD unit default), c < (done.getD unit default).size ∨
      ∃ p ∈ (done.getD unit default).parts, p.2 ≤ c ∧ c < p.2 + (done.getD p.1 default).size := by
  intro c hc
  rcases (partsChecked_spec (deriveChecked_parts hd)).2.2.2.2.2.2.1 c hc with h | h
  · exact Or.inl h
  · unfold inPart at h
    simp only [Array.any_eq_true', Bool.and_eq_true, decide_eq_true_eq] at h
    exact Or.inr h

/-- **A part's rows are covered exactly by its callee's reads, shifted**: at callee row `c`, the unit's read covering
`b + c` is the callee's read covering `c`, shifted by `b`, or none. -/
theorem part_reads (hd : deriveChecked types ls info words unit = .ok done) {p : Nat × Nat}
    (hp : p ∈ (done.getD unit default).parts) {c : Nat} (hc : c < (done.getD p.1 default).size) :
    (done.getD unit default).reads.find? (·.covers (p.2 + c)) =
      ((done.getD p.1 default).reads.find? (·.covers c)).map (·.shift p.2) := by
  have h := (partsChecked_spec (deriveChecked_parts hd)).2.2.2.2.2.2.2 p hp
  unfold partReadsOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range] at h
  exact beq_iff_eq.mp (h.2 c hc).1

/-- **A callee's covered row is a product row `a · []`, and its read's low minterms are callee rows.** -/
theorem callee_prod (hd : deriveChecked types ls info words unit = .ok done) {p : Nat × Nat}
    (hp : p ∈ (done.getD unit default).parts) {c : Nat} (hc : c < (done.getD p.1 default).size) {r : ReadRec}
    (hr : (done.getD p.1 default).reads.find? (·.covers c) = some r) :
    ((done.getD p.1 default).rows.getD c ([], [])).2 = [] ∧
      ∀ l < 2 ^ r.lo, r.loTop + l < (done.getD p.1 default).size := by
  have h := (partsChecked_spec (deriveChecked_parts hd)).2.2.2.2.2.2.2 p hp
  unfold partReadsOk at h
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, Array.all_eq_true_iff_forall_mem,
    decide_eq_true_eq] at h
  have hrow := (h.2 c hc).2
  rw [hr] at hrow
  refine ⟨List.isEmpty_iff.mp hrow, fun l hl => ?_⟩
  have := h.1 r (Array.mem_of_find?_eq_some hr)
  omega

end FlockSoundness.Types.Dag
