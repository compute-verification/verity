import Proofs.Flock.Soundness.Discharge.Hidden.SpecP

/-!
# The public leaves `HmIn.parse` accepted, and the rows `HmIn.delta` keeps

* `parse_ports`, `parse_leaves`, `parse_noPorts`: with public input ports, one public leaf map per unit, each as long as
  the unit's `leaves_in`; without, no public leaf map.
* `publicCol_at`: a column `HmIn.publicCol` names is bit `t` of public leaf `jj` of a unit slot `k`, at column
  `g0.col · WORD + 16 jj + t` of the slot, inside the unit's input rows (16-bit leaves, `CheckFacts.unit_groups`).
* `keepCol`, `inDelta_toList`, `deltaEntry_inDelta`: `HmIn.delta` is `HmOut.delta` without the pairs into the columns it
  drops (every column but a public leaf's bits keeps its pairs, and every column without public input ports), so its
  level-0 entries are `HmOut.delta`'s in every kept row and zero in the others.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Refine

/-- `List.mapM` in `Except` keeps the length. -/
theorem list_mapM_length {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → l'.length = l.length
  | [], l', h => by
      simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst h
      rfl
  | a :: l, l', h => by
      rw [List.mapM_cons] at h
      obtain ⟨b, -, h⟩ := ebind_ok h
      obtain ⟨bs, hbs, h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      simp [list_mapM_length hbs]

/-- `Array.mapM` in `Except` keeps the size. -/
theorem mapM_size {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β} (h : xs.mapM f = .ok ys) :
    ys.size = xs.size := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l =>
      rw [hl] at h
      simp only [Functor.map, Except.map, Except.ok.injEq] at h
      subst h
      rw [List.size_toArray, list_mapM_length hl, Array.length_toList]

set_option maxHeartbeats 4000000 in
/-- **What `HmIn.parse` accepted**: no public input port and no public leaf map, or public input ports with one public
leaf map per unit, each as long as its `leaves_in`. -/
theorem parse_ports {c : Circuit} {pi : HmIn.Ports} (h : HmIn.parse c = .ok pi) :
    (pi.ports.isEmpty = true ∧ pi.leaves = #[]) ∨
      (pi.ports.isEmpty = false ∧ pi.leaves.size = c.upv ∧
        ∀ u < c.upv, (pi.leaves[u]!).size = (c.leavesIn.getD u #[]).size) := by
  unfold HmIn.parse at h
  dsimp only at h
  cases hj : c.metaJ.getObjVal? "public_inputs" with
  | error e =>
    rw [hj] at h
    try dsimp only at h
    try simp only [throw_bind'] at h
    repeat' walk_step
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact Or.inl ⟨rfl, rfl⟩
  | ok pj =>
    rw [hj] at h
    try dsimp only at h
    try simp only [throw_bind'] at h
    repeat' walk_step
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      have hc := ‹¬ (_ || _) = true›
      have hps := ‹¬ (Array.isEmpty (α := Lean.Json) _) = true›
      have hm := mapM_size ‹Array.mapM _ _ = Except.ok _›
      try dsimp only
      refine Or.inr ⟨?_, ?_, fun u hu => ?_⟩
      · rw [Bool.eq_false_iff]
        intro hpe
        apply hps
        rw [Array.isEmpty_iff_size_eq_zero] at hpe ⊢
        omega
      · by_contra hn
        exact hc (by simp [hn])
      · by_contra hn
        refine hc ?_
        rw [Bool.or_eq_true, List.any_eq_true]
        refine Or.inr ⟨u, List.mem_range.2 hu, ?_⟩
        rw [Bool.or_eq_true, bne_iff_ne]
        exact Or.inl hn

/-- **With public input ports, one public leaf map per unit, each as long as its `leaves_in`**. -/
theorem parse_leaves {c : Circuit} {pi : HmIn.Ports} (h : HmIn.parse c = .ok pi) (hne : pi.ports.isEmpty = false) :
    pi.leaves.size = c.upv ∧ ∀ u < c.upv, (pi.leaves[u]!).size = (c.leavesIn.getD u #[]).size := by
  rcases parse_ports h with ⟨he, -⟩ | ⟨-, h1, h2⟩
  · rw [he] at hne; cases hne
  · exact ⟨h1, h2⟩

/-- **Without public input ports, no public leaf map.** -/
theorem parse_noPorts {c : Circuit} {pi : HmIn.Ports} (h : HmIn.parse c = .ok pi) (he : pi.ports.isEmpty = true) :
    pi.leaves = #[] := by
  rcases parse_ports h with ⟨-, h1⟩ | ⟨hne, -⟩
  · exact h1
  · rw [he] at hne; cases hne

/-- A list of 16s sums to 16 per entry. -/
theorem sum_16 {l : List ℕ} (h : ∀ b ∈ l, b = 16) : l.sum = 16 * l.length := by
  rw [List.eq_replicate_iff.2 ⟨rfl, h⟩, List.sum_replicate, smul_eq_mul, List.length_replicate, Nat.mul_comm]

/-- **A column `HmIn.publicCol` names is a public leaf's bit**: bit `t` of leaf `jj` of unit slot `k`'s leaf group, a
public leaf of the slot's unit, at a column of the slot inside the unit's input rows. -/
theorem publicCol_at {c : Circuit} {pi : HmIn.Ports} (hp : HmIn.parse c = .ok pi) (hne : pi.ports.isEmpty = false)
    (C : CheckFacts c) (hU : Built c.unit) {q : ℕ} (hq : HmIn.publicCol c pi q = true) :
    ∃ k < c.unitRange.count, ∃ jj < (c.unit.inGroups.getD 0 default).bits.size, ∃ t < 16,
      q = c.unitRange.slot k + ((c.unit.inGroups.getD 0 default).col * WORD + 16 * jj + t) ∧
      (c.unit.inGroups.getD 0 default).col * WORD + 16 * jj + t < c.unit.inWords * WORD ∧
      0 ≤ (pi.leaves.getD (k % c.upv) #[]).getD jj (-1) := by
  simp only [HmIn.publicCol, Bool.and_eq_true, decide_eq_true_eq, ge_iff_le] at hq
  obtain ⟨⟨⟨hs, hk⟩, hcol⟩, hl⟩ := hq
  have h16 := C.unit_groups.2
  have hlv := C.leaves.2
  have hg0m : c.unit.inGroups.getD 0 default ∈ c.unit.inGroups.toList := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (Nat.pos_of_ne_zero C.unit_groups.1),
      Option.getD_some]
    exact Array.getElem_mem_toList _
  obtain ⟨-, -, -, -, hgr, -⟩ := built_spec hU
  obtain ⟨hb, hw⟩ := hgr _ hg0m
  have hdm := Nat.div_add_mod (q - c.unitRange.start) (2 ^ c.unitRange.slotLog)
  generalize hg0 : c.unit.inGroups.getD 0 default = g0 at hcol hl h16 hlv hb hw ⊢
  generalize (q - c.unitRange.start) / 2 ^ c.unitRange.slotLog = k at hk hl hdm
  generalize (q - c.unitRange.start) % 2 ^ c.unitRange.slotLog = a at hcol hl hdm
  have hu : k % c.upv < c.upv := Nat.mod_lt _ C.upv_pos
  have hjl : (a - g0.col * WORD) / 16 < (pi.leaves.getD (k % c.upv) #[]).size := by
    by_contra hn
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega), Option.getD_none] at hl
    omega
  have hsz : (pi.leaves.getD (k % c.upv) #[]).size = g0.bits.size := by
    obtain ⟨h1, h2⟩ := parse_leaves hp hne
    have e1 : pi.leaves.getD (k % c.upv) #[] = pi.leaves[k % c.upv]! := by
      rw [Array.getD_eq_getD_getElem?, getElem!_pos pi.leaves _ (by omega), Array.getElem?_eq_getElem (by omega),
        Option.getD_some]
    have e2 : c.leavesIn.getD (k % c.upv) #[] = c.leavesIn[k % c.upv]! := by
      rw [Array.getD_eq_getD_getElem?, getElem!_pos c.leavesIn _ (by rw [C.leaves.1]; exact hu),
        Array.getElem?_eq_getElem (by rw [C.leaves.1]; exact hu), Option.getD_some]
    rw [e1, h2 _ hu, e2, hlv _ hu]
  rw [← Array.foldl_toList, foldl_add_eq_sum, Nat.zero_add, sum_16 h16, Array.length_toList] at hb
  have hslot : c.unitRange.slot k = c.unitRange.start + 2 ^ c.unitRange.slotLog * k := by
    unfold Range.slot Range.start; ring
  have hW : (g0.col + g0.words) * WORD ≤ c.unit.inWords * WORD := Nat.mul_le_mul_right _ hw
  rw [Nat.add_mul] at hW
  have h1 := Nat.div_add_mod (a - g0.col * WORD) 16
  have h2 := Nat.mod_lt (a - g0.col * WORD) (show 0 < 16 by norm_num)
  refine ⟨k, hk, (a - g0.col * WORD) / 16, by omega, (a - g0.col * WORD) % 16, h2, ?_, by omega, hl⟩
  rw [hslot]
  omega

/-! ## The rows `HmIn.delta` keeps -/

/-- **Whether `HmIn.delta` keeps Δ's pairs into block column `q`**: every column without public input ports, else every
column but a public leaf's bits (`HmIn.publicCol`). -/
def keepCol (c : Circuit) (pi : HmIn.Ports) (q : ℕ) : Bool := pi.ports.isEmpty || !HmIn.publicCol c pi q

/-- **`HmIn.delta` is `HmOut.delta` without the pairs into the columns it drops**, on both sides. -/
theorem inDelta_toList (c : Circuit) (pi : HmIn.Ports) (pin : ℕ) :
    (HmIn.delta c pi pin).1.toList = (HmOut.delta c pin).1.toList.filter (fun p => keepCol c pi p.1) ∧
      (HmIn.delta c pi pin).2.toList = (HmOut.delta c pin).2.toList.filter (fun p => keepCol c pi p.1) := by
  unfold HmIn.delta keepCol
  generalize HmOut.delta c pin = d
  obtain ⟨da, db⟩ := d
  dsimp only
  cases he : pi.ports.isEmpty <;> simp

/-- Pairs into the rows a filter keeps: a row's count is the list's, or zero where the filter drops it. -/
theorem count_filter_fst (l : List (ℕ × ℕ)) (P : ℕ → Bool) (r j : ℕ) :
    (l.filter fun p => P p.1).count (r, j) = if P r then l.count (r, j) else 0 := by
  split
  · rename_i h
    exact List.count_filter (p := fun p : ℕ × ℕ => P p.1) (a := (r, j)) h
  · rename_i h
    exact List.count_eq_zero.2 fun hm => h (List.mem_filter.1 hm).2

/-- **`HmIn.delta`'s level-0 entries**: `HmOut.delta`'s in every row it keeps, zero in the others. -/
theorem deltaEntry_inDelta (c : Circuit) (pi : HmIn.Ports) (pin r j : ℕ) :
    deltaEntry (HmIn.delta c pi pin).1 r j =
        (if keepCol c pi r then deltaEntry (HmOut.delta c pin).1 r j else 0) ∧
      deltaEntry (HmIn.delta c pi pin).2 r j =
        if keepCol c pi r then deltaEntry (HmOut.delta c pin).2 r j else 0 := by
  obtain ⟨h1, h2⟩ := inDelta_toList c pi pin
  rw [deltaEntry_eq_count, deltaEntry_eq_count, deltaEntry_eq_count, deltaEntry_eq_count, h1, h2,
    count_filter_fst, count_filter_fst]
  constructor <;> split <;> simp

/-- At a kept row, `HmIn.delta`'s entries are `HmOut.delta`'s. -/
theorem deltaEntry_kept {c : Circuit} {pi : HmIn.Ports} {pin r : ℕ} (hk : keepCol c pi r = true) (j : ℕ) :
    deltaEntry (HmIn.delta c pi pin).1 r j = deltaEntry (HmOut.delta c pin).1 r j ∧
      deltaEntry (HmIn.delta c pi pin).2 r j = deltaEntry (HmOut.delta c pin).2 r j := by
  obtain ⟨h1, h2⟩ := deltaEntry_inDelta c pi pin r j
  rw [ite_eq_left hk] at h1 h2
  exact ⟨h1, h2⟩

/-- Without public input ports every column is kept. -/
theorem keepCol_noPorts {c : Circuit} {pi : HmIn.Ports} (he : pi.ports.isEmpty = true) (q : ℕ) : keepCol c pi q = true := by
  simp [keepCol, he]

end FlockSoundness.Discharge.Hidden
