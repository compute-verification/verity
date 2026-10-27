import Proofs.Flock.Soundness.ExecTemplate
import Proofs.Flock.Soundness.ComposePlace
import Proofs.Flock.Soundness.Types.Parts

/-!
# A typed template's unit, placed (1e for templates, T3)

A template statement places its unit `u = done[unit]` by parts: the root net (`Typed.rootOf u`, the unit's own
rows without their part columns) one slot per template instance, each placed layout's net one slot per part, and Δ
(`delta_typed`): the flat entries, then per template instance each input's copy of its message bit, each part's bound
inputs, and the root rows' entries at part columns (`crossOf`). Unit column `x` of template instance `g` is at
`Typed.col c tb g x`.

- **`TemplateLayout st tb words done u`**: the matrix-level facts of the accepted statement that this reads: the ranges, the
  root's and the parts' nets and slots, and Δ. A part's net (`PartNet`) is its callee's derived rows, and a generated
  read's lookup from the table the verifier holds.
- **`UnitShape done u`**: the facts about the unit's rows this reads: its constant row, its order within its own and its
  parts' columns, no Δ entry at an input, and a part's reads its callee's (`deriveChecked` gives each).
- **`TemplateLayout.blockFacts`**: with those, and the part lemmas from `deriveChecked … = .ok done`, the block rows
  at `col g c` are the unit's `blockRow` there, for every template instance `g`: `BlockFacts` at `Typed.col c tb g`,
  which is a table class's `placed g` (`BlockFacts.placement`).
-/

open Flock FlockLevel3

namespace FlockSoundness

open Compose Flock.DeriveAll Types.Dag

/-! ## Counting pairs by row -/

theorem count_pairs_map (r y j : ℕ) (xs : List ℕ) :
    (xs.map fun x => (r, x)).count (y, j) = if r = y then xs.count j else 0 := by
  by_cases h : r = y
  · subst h
    rw [ite_eq_left rfl]
    exact List.count_map_of_injective _ _ (fun a b hab => by simpa using hab) _
  · rw [ite_eq_right h, List.count_eq_zero]
    intro hm
    obtain ⟨x, -, hx⟩ := List.mem_map.1 hm
    exact h (congrArg Prod.fst hx)

/-- A count of `(y, j)` over entries whose row is `R e`: the count of `j` over the entries at row `y`. -/
theorem count_flatMap_rows {α : Type} (l : List α) (R : α → ℕ) (F : α → List ℕ) (y j : ℕ) :
    (l.flatMap fun e => (F e).map fun x => (R e, x)).count (y, j) =
      ((l.filter fun e => R e = y).flatMap F).count j := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.flatMap_cons, List.count_append, ih, count_pairs_map, List.filter_cons]
    by_cases h : R a = y
    · simp [h, List.count_append]
    · simp [h]

/-! ## Slots of a net's range -/

/-- Column `x` is at offset `off` of slot `s` of net `n`'s range. -/
def AtSlot (c : Circuit) (n s off x : ℕ) : Prop :=
  s < (c.rangeOf n).count ∧ off < 2 ^ (c.rangeOf n).slotLog ∧ x = (c.rangeOf n).slot s + off

/-- Over pairwise-apart ranges, a column is at one slot of one net's range. -/
theorem AtSlot.inj {c : Circuit} (hd : c.ranges.toList.Pairwise Apart) {n n' s s' off off' x : ℕ}
    (hm : c.rangeOf n ∈ c.ranges.toList) (hk : (c.rangeOf n).kind = .net n)
    (hm' : c.rangeOf n' ∈ c.ranges.toList) (hk' : (c.rangeOf n').kind = .net n')
    (h : AtSlot c n s off x) (h' : AtSlot c n' s' off' x) : n = n' ∧ s = s' ∧ off = off' := by
  obtain ⟨hs, ho, rfl⟩ := h
  obtain ⟨hs', ho', he⟩ := h'
  have h2 : InSpan (c.rangeOf n') ((c.rangeOf n).slot s + off) := by
    rw [he]; exact inSpan_of_covers (covers_slot hs' ho')
  have hR := eq_of_inSpan hd hm hm' (inSpan_of_covers (covers_slot hs ho)) h2
  have hn : n = n' := Kind.net.inj (hk.symm.trans (hR ▸ hk'))
  subst hn
  exact ⟨rfl, slot_add_inj ho ho' he⟩

/-- A range of net `ip` whose span holds a column of net `n`'s range is `n`'s range. -/
theorem AtSlot.eq_range {c : Circuit} (hd : c.ranges.toList.Pairwise Apart) {n s off x : ℕ}
    (hm : c.rangeOf n ∈ c.ranges.toList) (hk : (c.rangeOf n).kind = .net n) {r : Range} (hr : r ∈ c.ranges.toList)
    {ip : ℕ} (hki : r.kind = .net ip) {s' off' : ℕ} (hs' : s' < r.count) (ho' : off' < 2 ^ r.slotLog)
    (h : AtSlot c n s off x) (hx : x = r.slot s' + off') : r = c.rangeOf n ∧ ip = n ∧ s' = s ∧ off' = off := by
  obtain ⟨hs, ho, rfl⟩ := h
  have h2 : InSpan r ((c.rangeOf n).slot s + off) := by rw [hx]; exact inSpan_of_covers (covers_slot hs' ho')
  have hR := eq_of_inSpan hd hr hm h2 (inSpan_of_covers (covers_slot hs ho))
  subst hR
  have := Kind.net.inj (hki.symm.trans hk)
  exact ⟨rfl, this, (slot_add_inj ho' ho hx.symm).1, (slot_add_inj ho' ho hx.symm).2⟩

/-- A column of net `n`'s range is in the block. -/
theorem AtSlot.lt {c : Circuit} (hb : ∀ r ∈ c.ranges.toList, r.start + spanH r ≤ 2 ^ c.kLog) {n s off x : ℕ}
    (hm : c.rangeOf n ∈ c.ranges.toList) (h : AtSlot c n s off x) : x < 2 ^ c.kLog := by
  obtain ⟨hs, ho, rfl⟩ := h
  have := (covers_iff _ _).1 (covers_slot hs ho)
  have := count_mul_le_spanH (c.rangeOf n)
  have := hb _ hm
  omega

/-- A range with a slot inside the block has slots no larger than the block. -/
theorem AtSlot.slotLog_le {c : Circuit} (hb : ∀ r ∈ c.ranges.toList, r.start + spanH r ≤ 2 ^ c.kLog)
    {n s off x : ℕ} (hm : c.rangeOf n ∈ c.ranges.toList) (h : AtSlot c n s off x) : (c.rangeOf n).slotLog ≤ c.kLog := by
  obtain ⟨hs, -, -⟩ := h
  have h1 := hb _ hm
  have h2 := count_mul_le_spanH (c.rangeOf n)
  have h3 : 2 ^ (c.rangeOf n).slotLog ≤ (c.rangeOf n).count * 2 ^ (c.rangeOf n).slotLog :=
    Nat.le_mul_of_pos_left _ (by omega)
  have h4 : 2 ^ (c.rangeOf n).slotLog ≤ 2 ^ c.kLog := by omega
  exact (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).1 h4

/-! ## Δ's first loop at a slot -/

/-- **Δ's first loop at a column of a net's slot**: `(k, k)` and `(k, pin)` at the slot's constant `k` unless it is the
pin, and nothing at its other columns. -/
theorem constPairs_count {c : Circuit} {pin : ℕ} (hd : c.ranges.toList.Pairwise Apart)
    (hsl : ∀ r ∈ c.ranges.toList, ∀ i, r.kind = .net i → r.slotLog = (c.nets.getD i default).2.unitLog)
    (hfit : ∀ i, (c.nets.getD i default).2.constPos < 2 ^ (c.nets.getD i default).2.unitLog)
    {n s off x : ℕ} (hm : c.rangeOf n ∈ c.ranges.toList) (hk : (c.rangeOf n).kind = .net n) (hx : AtSlot c n s off x)
    (j : ℕ) :
    (constPairs c pin).count (x, j) =
      if off = (c.nets.getD n default).2.constPos ∧ x ≠ pin then
        (if j = x then 1 else 0) + (if j = pin then 1 else 0) else 0 := by
  obtain ⟨hs, hoff, rfl⟩ := hx
  have hcW : (c.nets.getD n default).2.constPos < 2 ^ (c.rangeOf n).slotLog := by
    rw [hsl _ hm n hk]; exact hfit n
  rw [constPairs, count_flatMap_apart hd hm (inSpan_of_covers (covers_slot hs hoff)) _ _ (fun a ha hax => by
    refine List.count_eq_zero.2 fun hm' => hax ?_
    obtain ⟨i, hki, s', hs', heq, -⟩ := slotPairs_fst hm'
    simp only at heq
    rw [heq]
    exact inSpan_of_covers (covers_slot hs' (by rw [hsl a ha i hki]; exact hfit i)))]
  simp only [slotPairs, hk]
  by_cases hoc : off = (c.nets.getD n default).2.constPos
  · subst hoc
    rw [count_flatMap_range _ _ hs (fun s' hs' hne => by
      refine List.count_eq_zero.2 fun hm' => hne ?_
      split at hm'
      · simp only [List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq] at hm'
        exact (slot_add_inj hcW hcW (by rcases hm' with ⟨h, -⟩ | ⟨h, -⟩ <;> exact h)).1.symm
      · cases hm')]
    by_cases hxp : (c.rangeOf n).slot s + (c.nets.getD n default).2.constPos = pin
    · have : ((c.rangeOf n).slot s + (c.nets.getD n default).2.constPos != pin) = false := by
        simp only [hxp, bne_self_eq_false]
      rw [ite_eq_right (by rw [this]; exact Bool.false_ne_true), ite_eq_right (fun h => h.2 hxp)]
      rfl
    · rw [ite_eq_left (bne_iff_ne.2 hxp), ite_eq_left ⟨rfl, hxp⟩]
      generalize (c.rangeOf n).slot s + (c.nets.getD n default).2.constPos = x at hxp ⊢
      by_cases h1 : j = x
      · subst h1
        simp [hxp]
      · by_cases h2 : j = pin
        · subst h2; simp [h1, Ne.symm h1]
        · simp [h1, h2, Ne.symm h1, Ne.symm h2]
  · rw [ite_eq_right (fun h => hoc h.1)]
    refine count_flatMap_zero fun s' _ => List.count_eq_zero.2 fun hm' => ?_
    split at hm'
    · simp only [List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq] at hm'
      exact hoc (slot_add_inj hoff hcW (by rcases hm' with ⟨h, -⟩ | ⟨h, -⟩ <;> exact h)).2
    · cases hm'

/-! ## Unit columns in the block -/

namespace Typed

theorem col_own {c : Circuit} {tb : TypedBlock} {g x : ℕ} (hx : x < tb.own) :
    Typed.col c tb g x = c.slotCol c.unitNet g 0 + x := by
  unfold Typed.col; rw [ite_eq_left hx]

/-- The last part at or before `x` is part `i` when `x` is past its base and before every later part's. -/
theorem find?_parts {parts : List (ℕ × ℕ × ℕ)} {i : ℕ} (hi : i < parts.length) {x : ℕ} (hle : parts[i].2.2 ≤ x)
    (hgt : ∀ k (hk : k < parts.length), i < k → x < parts[k].2.2) :
    parts.reverse.find? (·.2.2 ≤ x) = some parts[i] := by
  generalize he : parts[i] = e at hle ⊢
  have hsplit : parts = parts.take i ++ e :: parts.drop (i + 1) := by
    rw [← he]
    conv_lhs => rw [← List.take_append_drop i parts, List.drop_eq_getElem_cons hi]
  have hnone : (parts.drop (i + 1)).reverse.find? (·.2.2 ≤ x) = none := by
    refine List.find?_eq_none.2 fun e' he' => ?_
    rw [List.mem_reverse] at he'
    obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.1 he'
    rw [List.getElem_drop]
    have := hgt (i + 1 + k) (by simp at hk; omega) (by omega)
    simp only [decide_eq_true_eq, not_le]
    exact this
  rw [hsplit, List.reverse_append, List.reverse_cons, List.append_assoc, List.find?_append, hnone, Option.none_or,
    List.singleton_append]
  exact List.find?_cons_of_pos (by simpa using hle)

theorem col_part {c : Circuit} {tb : TypedBlock} {g x i : ℕ} (hi : i < tb.parts.size) (hown : tb.own ≤ x)
    (hle : tb.parts[i].2.2 ≤ x) (hgt : ∀ k (hk : k < tb.parts.size), i < k → x < tb.parts[k].2.2) :
    Typed.col c tb g x = c.slotCol tb.parts[i].1 g tb.parts[i].2.1 + (x - tb.parts[i].2.2) := by
  unfold Typed.col
  rw [ite_eq_right (by omega), find?_parts (parts := tb.parts.toList) (i := i) (by simpa using hi)
    (by simpa using hle) (fun k hk hik => by simpa using hgt k (by simpa using hk) hik)]
  simp

end Typed

theorem count_flatMap_rows2 {α : Type} (l : List α) (R : α → ℕ) (F : α → List ℕ) (G : ℕ → ℕ) (y j : ℕ) :
    (l.flatMap fun e => (R e, R e) :: (F e).map fun x => (R e, G x)).count (y, j) =
      ((l.filter fun e => R e = y).flatMap fun e => R e :: (F e).map G).count j := by
  have := count_flatMap_rows l R (fun e => R e :: (F e).map G) y j
  simp only [List.map_cons, List.map_map, Function.comp_def] at this
  exact this

theorem count_flatMap_rows3 {α : Type} (l : List α) (R : α → ℕ) (F : α → List ℕ) (G : ℕ → ℕ) (y j : ℕ) :
    (l.flatMap fun e => (F e).map fun x => (R e, G x)).count (y, j) =
      ((l.filter fun e => R e = y).flatMap fun e => (F e).map G).count j := by
  have := count_flatMap_rows l R (fun e => (F e).map G) y j
  simp only [List.map_map, Function.comp_def] at this
  exact this

/-- With distinct rows, the entries at row `c` are the one `find?` returns. -/
theorem filter_fst_eq_find? {β : Type} : ∀ {l : List (ℕ × β)}, (l.map (·.1)).Nodup → ∀ c,
    l.filter (fun e => e.1 = c) = (l.find? (·.1 == c)).toList
  | [], _, _ => rfl
  | a :: l, hn, c => by
    rw [List.map_cons, List.nodup_cons] at hn
    by_cases ha : a.1 = c
    · rw [List.filter_cons_of_pos (by simpa using ha), List.find?_cons_of_pos (by simpa using ha)]
      rw [List.filter_eq_nil_iff.2 fun e he h => hn.1 (by
        rw [List.mem_map]; exact ⟨e, he, by simpa [← ha] using h⟩)]
      rfl
    · rw [List.filter_cons_of_neg (by simpa using ha), List.find?_cons_of_neg (by simpa using ha)]
      exact filter_fst_eq_find? hn.2 c

theorem find?_of_mem {β : Type} {l : List (ℕ × β)} (hn : (l.map (·.1)).Nodup) {e : ℕ × β} (he : e ∈ l) :
    l.find? (·.1 == e.1) = some e := by
  have : e ∈ l.filter (fun e' => e'.1 = e.1) := List.mem_filter.2 ⟨he, by simp⟩
  rw [filter_fst_eq_find? hn] at this
  exact Option.mem_toList.1 this

theorem filter_filterMap_fst {β γ : Type} (l : List (ℕ × β)) (f : ℕ × β → Option (ℕ × γ))
    (hf : ∀ e y, f e = some y → y.1 = e.1) (c : ℕ) :
    (l.filterMap f).filter (fun y => y.1 = c) = (l.filter (fun e => e.1 = c)).filterMap f := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.filterMap_cons, List.filter_cons]
    cases hfa : f a with
    | none =>
      simp only
      rw [ih]
      split <;> simp [hfa]
    | some y =>
      have hy := hf a y hfa
      simp only [List.filter_cons, hy]
      split <;> simp [hfa, ih]

theorem filterMap_range_single {γ : Type} (f : ℕ → Option γ) {c : ℕ} (hf : ∀ r, r ≠ c → f r = none) :
    ∀ n, (List.range n).filterMap f = if c < n then (f c).toList else []
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.filterMap_append, filterMap_range_single f hf n, List.filterMap_cons,
      List.filterMap_nil]
    by_cases hn : n = c
    · subst hn
      cases f n <;> simp
    · rw [hf n hn]
      by_cases hc : c < n
      · simp [hc, show c < n + 1 by omega]
      · simp [hc, show ¬ c < n + 1 by omega]

/-! ## The typed tail's rows -/

/-- Own row `r`'s entries at part columns (`crossOf`'s filter). -/
def crossRow (u : D) (r : ℕ) : Option (ℕ × Array ℕ × Array ℕ) :=
  if ((u.rows.getD r ([], [])).1.filter (· ≥ u.size)).isEmpty && ((u.rows.getD r ([], [])).2.filter (· ≥ u.size)).isEmpty
  then none
  else some (r, ((u.rows.getD r ([], [])).1.filter (· ≥ u.size)).toArray,
    ((u.rows.getD r ([], [])).2.filter (· ≥ u.size)).toArray)

theorem crossOf_toList (u : D) : (Typed.crossOf u).toList = (List.range u.size).filterMap (crossRow u) := by
  unfold Typed.crossOf
  rw [Array.toList_filterMap, Array.toList_range]
  congr 1

/-- The cross entries at own column `c`, as the columns they read (`G` places them). -/
def crossCols (G : ℕ → ℕ) (u : D) (side : Bool) (c : ℕ) : List ℕ :=
  if c < u.size then
    ((if side then (u.rows.getD c ([], [])).1 else (u.rows.getD c ([], [])).2).filter (· ≥ u.size)).map G
  else []

theorem crossOf_at (G : ℕ → ℕ) (u : D) (side : Bool) (c : ℕ) :
    (((Typed.crossOf u).toList.filter fun e => e.1 = c).flatMap fun e =>
      (if side then e.2.1 else e.2.2).toList.map G) = crossCols G u side c := by
  have hfst : ∀ r y, crossRow u r = some y → y.1 = r := by
    intro r y h
    unfold crossRow at h
    split at h
    · cases h
    · cases h; rfl
  rw [crossOf_toList, List.filter_filterMap, filterMap_range_single _ (c := c) (fun r hr => by
    cases h : crossRow u r with
    | none => rfl
    | some y => simp [hfst r y h, hr])]
  unfold crossCols
  by_cases hc : c < u.size
  · rw [ite_eq_left hc, ite_eq_left hc]
    cases h : crossRow u c with
    | none =>
      unfold crossRow at h
      split at h
      · rename_i he
        simp only [Bool.and_eq_true, List.isEmpty_iff] at he
        cases side <;> simp only [Option.filter_none, Option.toList_none, List.flatMap_nil, Bool.false_eq_true,
          ↓reduceIte, he.1, he.2, List.map_nil]
      · cases h
    | some y =>
      have hy := hfst c y h
      unfold crossRow at h
      split at h
      · cases h
      · cases h
        cases side <;> simp [Option.filter]
  · rw [ite_eq_right hc, ite_eq_right hc]
    rfl

/-- The bound-input entries at unit column `c`, as the columns they read (`G` places them): its self entry and its
source's. -/
def bindCols (G : ℕ → ℕ) (u : D) (c : ℕ) : List ℕ :=
  match u.delta.find? (·.1 == c) with
  | some (_, some s) => G c :: s.map G
  | _ => []

theorem bind_at (G : ℕ → ℕ) (u : D) (hn : (u.delta.toList.map (·.1)).Nodup) (c : ℕ) :
    (((u.delta.filterMap fun (r, src) => src.map (r, ·.toArray)).toList.filter fun e => e.1 = c).flatMap
      fun rs => G rs.1 :: rs.2.toList.map G) = bindCols G u c := by
  rw [Array.toList_filterMap, filter_filterMap_fst _ _ (fun e y h => by
    obtain ⟨r, src⟩ := e
    cases src with
    | none => cases h
    | some s => cases h; rfl), filter_fst_eq_find? hn, Array.find?_toList]
  unfold bindCols
  cases hf : u.delta.find? (·.1 == c) with
  | none => rfl
  | some e =>
    obtain ⟨r, src⟩ := e
    have hr : r = c := by simpa using Array.find?_some hf
    subst hr
    cases src with
    | none => rfl
    | some s => simp

/-- Unit column `x` is its own or in one of its parts. -/
def InUnit (done : Array D) (u : D) (x : ℕ) : Prop :=
  x < u.size ∨ ∃ p ∈ u.parts, p.2 ≤ x ∧ x < p.2 + (done.getD p.1 default).size

/-- Unit column `x` is no constant's: not the unit's, and no Δ copy of `one`. -/
def NotConst (u : D) (x : ℕ) : Prop := x ≠ u.const ∧ u.delta.find? (·.1 == x) ≠ some (x, none)

theorem netOfD_lookup {d : D} {nm : String} {n : Net} (h : Typed.netOfD d nm = .ok n) : n.lookup = none :=
  (Net.ofRows_ok h).2.2.2.2.2.2.2.2.2.2.2.2.1

/-- **A net built from derived rows reads them back**: its entry at `(c, j)` is the count of `j` in `d`'s row `c`. -/
theorem netOfD_sparse {d : D} {nm : String} {n : Net} (h : Typed.netOfD d nm = .ok n) (hu : n.unitLog ≤ 32) (c j : ℕ) :
    sparseEntry n.a c j = ((((Typed.rowsA d).getD c #[]).toList.count j : ℕ) : ZMod 2) ∧
      sparseEntry n.b c j = ((((Typed.rowsB d).getD c #[]).toList.count j : ℕ) : ZMod 2) := by
  obtain ⟨-, ha, hb, -⟩ := Typed.netOfD_ok h
  have hw : 2 ^ n.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hu
  exact ⟨by rw [sparseEntry_ofRows ha, ofRows_row ha hw], by rw [sparseEntry_ofRows hb, ofRows_row hb hw]⟩

theorem rowsA_lt {d : D} {nm : String} {n : Net} (h : Typed.netOfD d nm = .ok n) (off : ℕ) :
    (∀ y ∈ ((Typed.rowsA d).getD off #[]).toList, y < 2 ^ n.unitLog) ∧
      ∀ y ∈ ((Typed.rowsB d).getD off #[]).toList, y < 2 ^ n.unitLog := by
  obtain ⟨-, ha, hb, -⟩ := Typed.netOfD_ok h
  have key : ∀ (rows : Array (Array ℕ)) (m : Sparse), Sparse.ofRows rows (2 ^ n.unitLog) = .ok m →
      ∀ y ∈ (rows.getD off #[]).toList, y < 2 ^ n.unitLog := by
    intro rows m hm y hy
    obtain ⟨-, -, -, hc⟩ := ofRows_spec hm
    by_cases ho : off < rows.size
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem ho, Option.getD_some] at hy
      exact hc _ (Array.getElem_mem_toList ho) y hy
    · rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega), Option.getD_none] at hy
      simp at hy
  exact ⟨key _ _ ha, key _ _ hb⟩

theorem rowsA_getD {d : D} {c : ℕ} (hc : c < d.size) :
    (Typed.rowsA d).getD c #[] = (d.rows.getD c ([], [])).1.toArray := by
  simp [Typed.rowsA, Array.getD_eq_getD_getElem?, hc]

theorem rowsB_getD {d : D} {c : ℕ} (hc : c < d.size) :
    (Typed.rowsB d).getD c #[] = (d.rows.getD c ([], [])).2.toArray := by
  simp [Typed.rowsB, Array.getD_eq_getD_getElem?, hc]

theorem rootOf_rows (u : D) (c : ℕ) :
    (Typed.rootOf u).rows.getD c ([], []) =
      ((u.rows.getD c ([], [])).1.filter (· < u.size), (u.rows.getD c ([], [])).2.filter (· < u.size)) := by
  simp only [Typed.rootOf, Array.getD_eq_getD_getElem?, Array.getElem?_map]
  cases u.rows[c]? <;> rfl

/-! ## A generated read's lookup reads what `readSide` reads -/

theorem count_readSide (ws : Array ℕ) (r : ReadRec) (c j : ℕ) :
    (readSide ws r c).count j =
      if r.loTop ≤ j ∧ j < r.loTop + 2 ^ r.lo ∧
          (ws.getD ((c - r.prod0) % 2 ^ (r.n - r.lo) * 2 ^ r.lo + (j - r.loTop)) 0).testBit
            ((c - r.prod0) / 2 ^ (r.n - r.lo)) then 1 else 0 := by
  unfold readSide
  by_cases hj : r.loTop ≤ j
  · obtain ⟨l, rfl⟩ : ∃ l, j = r.loTop + l := ⟨j - r.loTop, by omega⟩
    rw [List.count_map_of_injective _ _ (fun a b h => by simpa using h), Nat.add_sub_cancel_left]
    rw [List.Nodup.count ((List.nodup_range).filter _)]
    by_cases hl : l < 2 ^ r.lo
    · simp [hl, List.mem_filter]
    · simp [hl, List.mem_filter]
  · rw [List.count_eq_zero.2 (fun hm => by
      obtain ⟨l, -, hl⟩ := List.mem_map.1 hm
      omega), ite_eq_right (fun h => hj h.1)]

theorem lookupEntry_readSide {tb : ByteArray} {r : ReadRec} {ws : Array ℕ}
    (hws : ws = (Array.range (tb.size / 4)).map fun i => rd32 tb (4 * i)) (hsz : ws.size = 2 ^ r.n) (hlo : r.lo ≤ r.n)
    {c : ℕ} (hc : r.covers c = true) (j : ℕ) :
    lookupEntry (some ⟨tb, r.n, r.lo, r.k, r.prod0, r.loTop⟩) c j = (((readSide ws r c).count j : ℕ) : ZMod 2) := by
  simp only [ReadRec.covers, Bool.and_eq_true, decide_eq_true_eq] at hc
  obtain ⟨hc1, hc2⟩ := hc
  have hnh : 0 < 2 ^ (r.n - r.lo) := Nat.two_pow_pos _
  have hbi : (c - r.prod0) / 2 ^ (r.n - r.lo) < r.k := by
    rw [Nat.div_lt_iff_lt_mul hnh]; omega
  rw [count_readSide]
  unfold lookupEntry
  simp only
  by_cases hj : r.loTop ≤ j ∧ j < r.loTop + 2 ^ r.lo
  · have hl : j - r.loTop < 2 ^ r.lo := by omega
    have hidx : (c - r.prod0) % 2 ^ (r.n - r.lo) * 2 ^ r.lo + (j - r.loTop) < ws.size := by
      rw [hsz]
      have hm := Nat.mod_lt (c - r.prod0) hnh
      have : 2 ^ r.n = 2 ^ (r.n - r.lo) * 2 ^ r.lo := by rw [← Nat.pow_add]; congr 1; omega
      rw [this]
      calc (c - r.prod0) % 2 ^ (r.n - r.lo) * 2 ^ r.lo + (j - r.loTop)
          < (c - r.prod0) % 2 ^ (r.n - r.lo) * 2 ^ r.lo + 2 ^ r.lo := by omega
        _ = ((c - r.prod0) % 2 ^ (r.n - r.lo) + 1) * 2 ^ r.lo := by ring
        _ ≤ 2 ^ (r.n - r.lo) * 2 ^ r.lo := Nat.mul_le_mul_right _ hm
    have hget : ws.getD ((c - r.prod0) % 2 ^ (r.n - r.lo) * 2 ^ r.lo + (j - r.loTop)) 0 =
        rd32 tb (4 * ((c - r.prod0) % 2 ^ (r.n - r.lo) * 2 ^ r.lo + (j - r.loTop))) := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hidx, Option.getD_some]
      subst hws
      simp
    rw [hget]
    have hbit : (Lookup.word ⟨tb, r.n, r.lo, r.k, r.prod0, r.loTop⟩ (2 ^ r.lo) (2 ^ r.k - 1)
        ((c - r.prod0) % 2 ^ (r.n - r.lo)) (j - r.loTop)).testBit ((c - r.prod0) / 2 ^ (r.n - r.lo)) =
        (rd32 tb (4 * ((c - r.prod0) % 2 ^ (r.n - r.lo) * 2 ^ r.lo + (j - r.loTop)))).testBit
          ((c - r.prod0) / 2 ^ (r.n - r.lo)) := by
      simp only [Lookup.word, Nat.testBit_and, Nat.testBit_two_pow_sub_one, decide_eq_true hbi, Bool.and_true]
    rw [hbit]
    by_cases hb : (rd32 tb (4 * ((c - r.prod0) % 2 ^ (r.n - r.lo) * 2 ^ r.lo + (j - r.loTop)))).testBit
        ((c - r.prod0) / 2 ^ (r.n - r.lo)) = true
    · rw [ite_eq_left ⟨hc1, hc2, hj.1, hj.2, hb⟩, ite_eq_left ⟨hj.1, hj.2, hb⟩]; simp
    · rw [ite_eq_right (fun h => hb h.2.2.2.2), ite_eq_right (fun h => hb h.2.2)]; simp
  · rw [ite_eq_right (fun h => hj ⟨h.2.2.1, h.2.2.2.1⟩), ite_eq_right (fun h => hj ⟨h.1, h.2.1⟩)]; simp

theorem lookupEntry_uncovered {tb : ByteArray} {r : ReadRec} {c : ℕ} (hc : r.covers c = false) (j : ℕ) :
    lookupEntry (some ⟨tb, r.n, r.lo, r.k, r.prod0, r.loTop⟩) c j = 0 := by
  simp only [ReadRec.covers, Bool.and_eq_false_iff, decide_eq_false_iff_not, not_le, not_lt] at hc
  unfold lookupEntry
  simp only
  rw [ite_eq_right (fun h => by rcases hc with hc | hc <;> omega)]

/-! ## A part's placed net -/

/-- **A part's placed net**: its callee's derived rows (`netOfD`), with no lookup and no read, or a generated read's
lookup from the table the verifier holds for its read (`words`, its `u32` words). -/
def PartNet (words : String → Option (Array ℕ)) (d : D) (net : Net) : Prop :=
  (∃ nm, Typed.netOfD d nm = .ok { net with lookup := none }) ∧
    ((net.lookup = none ∧ d.reads = #[]) ∨
      ∃ r tb, d.reads.toList = [r] ∧ net.lookup = some ⟨tb, r.n, r.lo, r.k, r.prod0, r.loTop⟩ ∧
        (words r.table).getD #[] = (Array.range (tb.size / 4)).map (fun i => rd32 tb (4 * i)) ∧
        ((words r.table).getD #[]).size = 2 ^ r.n ∧ r.lo ≤ r.n)

/-- A callee's row `c`'s `B` side as the verifier folds it (`fullRow`): a read's table-direct side, else its derived row. -/
def partB (words : String → Option (Array ℕ)) (d : D) (c : ℕ) : List ℕ :=
  match d.reads.find? (·.covers c) with
  | some r => readSide ((words r.table).getD #[]) r c
  | none => (d.rows.getD c ([], [])).2

theorem PartNet.sparseA {words : String → Option (Array ℕ)} {d : D} {net : Net} (h : PartNet words d net)
    (hu : net.unitLog ≤ 32) (c j : ℕ) :
    sparseEntry net.a c j = ((((Typed.rowsA d).getD c #[]).toList.count j : ℕ) : ZMod 2) := by
  obtain ⟨⟨nm, hn⟩, -⟩ := h
  exact (netOfD_sparse hn hu c j).1

/-- **A part net's `B` at a row no read covers** is the callee's derived row. -/
theorem PartNet.entryB_unc {words : String → Option (Array ℕ)} {d : D} {net : Net} (h : PartNet words d net)
    (hu : net.unitLog ≤ 32) {c : ℕ} (hnc : d.reads.find? (·.covers c) = none) (j : ℕ) :
    sparseEntry net.b c j + lookupEntry net.lookup c j = ((((Typed.rowsB d).getD c #[]).toList.count j : ℕ) : ZMod 2) := by
  obtain ⟨⟨nm, hn⟩, hl⟩ := h
  have hsb : sparseEntry net.b c j = _ := (netOfD_sparse hn hu c j).2
  rw [hsb]
  rcases hl with ⟨hl, -⟩ | ⟨r, tb, hr, hl, -, -, -⟩
  · rw [hl]; simp [lookupEntry]
  · have hcv : r.covers c = false := by
      rw [← Array.find?_toList, hr] at hnc
      simpa using hnc
    rw [hl, lookupEntry_uncovered hcv, add_zero]

/-- **A part net's `B` at a callee row** is the row as the verifier folds it (`partB`): a covered row's derived `B` is
empty, and the lookup adds its table-direct side. -/
theorem PartNet.entryB {words : String → Option (Array ℕ)} {d : D} {net : Net} (h : PartNet words d net)
    (hu : net.unitLog ≤ 32) {c : ℕ} (hc : c < d.size)
    (hB : ∀ r, d.reads.find? (·.covers c) = some r → (d.rows.getD c ([], [])).2 = []) (j : ℕ) :
    sparseEntry net.b c j + lookupEntry net.lookup c j = (((partB words d c).count j : ℕ) : ZMod 2) := by
  cases hf : d.reads.find? (·.covers c) with
  | none =>
    rw [h.entryB_unc hu hf, rowsB_getD hc]
    simp [partB, hf]
  | some r =>
    obtain ⟨⟨nm, hn⟩, hl⟩ := h
    have hsb : sparseEntry net.b c j = _ := (netOfD_sparse hn hu c j).2
    rw [hsb, rowsB_getD hc, hB r hf]
    rcases hl with ⟨-, hr⟩ | ⟨r', tb, hr, hl, hws, hsz, hlo⟩
    · rw [hr] at hf; simp at hf
    · have hrr : r = r' := by
        rw [← Array.find?_toList, hr] at hf
        by_cases hcv : r'.covers c = true
        · simp [hcv] at hf; exact hf.symm
        · simp [hcv] at hf
      subst hrr
      have hcv : r.covers c = true := by
        have := Array.find?_some hf
        simpa using this
      rw [hl, lookupEntry_readSide hws hsz hlo hcv j]
      simp [partB, hf]

/-! ## The facts T3 reads -/

/-- **The accepted template statement's layout, as T3 reads it**, for the class's unit `u`: the ranges (disjoint, in the
block, of their nets' slots, `G · per_vu` slots each, one per net); the root (its net `Typed.rootOf u`'s, one slot per
template instance); each part `i` (its net that of its callee `done[u.parts[i].1]`, not the root's or the rows', at its
slot `q_i` of that net's, no two parts at one slot); the rows' circuits (not the root); Δ (the flat entries, whose other
rows are input bits of the rows' circuits, then the typed tail); the pin (a net slot's constant, whose row reads
itself); and the block's size. -/
structure TemplateLayout (st : Stmt) (tb : TypedBlock) (words : String → Option (Array ℕ)) (done : Array D) (u : D) :
    Prop where
  typed : st.c.typed = some tb
  disj : st.c.ranges.toList.Pairwise Apart
  bound : ∀ r ∈ st.c.ranges.toList, r.start + spanH r ≤ 2 ^ st.c.kLog
  slotLog : ∀ r ∈ st.c.ranges.toList, ∀ i, r.kind = .net i → r.slotLog = (st.c.nets.getD i default).2.unitLog
  count : ∀ r ∈ st.c.ranges.toList, ∀ n, r.kind = .net n → r.count = st.c.g * st.c.perVu.getD n 0
  one_range : ∀ n < st.c.nets.size, (st.c.ranges.filter (·.kind == Kind.net n)).size = 1
  fit : ∀ i, (st.c.nets.getD i default).2.constPos < 2 ^ (st.c.nets.getD i default).2.unitLog
  unitNet : st.c.unitNet < st.c.nets.size
  root_perVu : st.c.perVu.getD st.c.unitNet 0 = 1
  root : ∃ nm, Typed.netOfD (Typed.rootOf u) nm = .ok st.c.unit
  own : tb.own = u.size
  inCols : tb.inCols = u.inCols
  bind : tb.bind = u.delta.filterMap fun (r, src) => src.map (r, ·.toArray)
  cross : tb.cross = Typed.crossOf u
  /-- the unit reads no table in its own rows (`templateOf`) -/
  reads_past : ∀ r ∈ u.reads.toList, u.size ≤ r.prod0
  parts_size : tb.parts.size = u.parts.size
  part : ∀ i (hi : i < tb.parts.size) (hu : i < u.parts.size), tb.parts[i].2.2 = u.parts[i].2 ∧
    tb.parts[i].1 < st.c.nets.size ∧ tb.parts[i].1 ≠ st.c.unitNet ∧ tb.parts[i].1 ≠ st.c.shaNet ∧
    tb.parts[i].1 ≠ st.c.hmNet ∧ tb.parts[i].2.1 < st.c.perVu.getD tb.parts[i].1 0 ∧
    PartNet words (done.getD u.parts[i].1 default) (st.c.nets.getD tb.parts[i].1 default).2
  part_inj : ∀ i (hi : i < tb.parts.size) k (hk : k < tb.parts.size), tb.parts[i].1 = tb.parts[k].1 →
    tb.parts[i].2.1 = tb.parts[k].2.1 → i = k
  shaNet : st.c.shaNet < st.c.nets.size ∧ st.c.shaNet ≠ st.c.unitNet
  hmNet : st.c.hmNet < st.c.nets.size ∧ st.c.hmNet ≠ st.c.unitNet
  delta : TailQ st.c st.pin tb (st.da, st.db)
  deltaIn : ∀ x, DeltaIn st.c x → ∃ n, (n = st.c.shaNet ∨ n = st.c.hmNet) ∧
    ∃ s off, AtSlot st.c n s off x ∧ off < (st.c.nets.getD n default).2.inWords * Flock.WORD
  pin : ∃ rp ∈ st.c.ranges.toList, ∃ ip, rp.kind = .net ip ∧ 0 < rp.count ∧
    st.pin = rp.start + (st.c.nets.getD ip default).2.constPos ∧ ConstRow (st.c.nets.getD ip default).2
  kLog : st.c.kLog ≤ 32
  kLog6 : 6 ≤ st.c.kLog
  m : st.m = st.c.kLog + st.nbl

/-- **The unit's rows, as T3 reads them**, beyond the part lemmas: its constant is an own column whose row is
`[const]·[const]`; its order lists own and part columns only; no Δ entry sits at one of its inputs; a part's rows are
covered by its callee's reads, shifted; and a covered callee row is `a · []`, its read's low minterms callee rows. -/
structure UnitShape (done : Array D) (u : D) : Prop where
  const_row : u.rows.getD u.const ([], []) = ([u.const], [u.const])
  order_in : ∀ c ∈ order done u, InUnit done u c
  in_delta : ∀ x ∈ u.inCols, u.delta.find? (·.1 == x) = none
  /-- a part's rows are covered by its callee's reads, shifted -/
  part_reads : ∀ p ∈ u.parts, ∀ c < (done.getD p.1 default).size,
    u.reads.find? (·.covers (p.2 + c)) = ((done.getD p.1 default).reads.find? (·.covers c)).map (·.shift p.2)
  /-- a callee's product row is `a · []`, and its low minterms are its own rows -/
  callee_prod : ∀ p ∈ u.parts, ∀ c < (done.getD p.1 default).size, ∀ r,
    (done.getD p.1 default).reads.find? (·.covers c) = some r →
      ((done.getD p.1 default).rows.getD c ([], [])).2 = [] ∧ ∀ l < 2 ^ r.lo, r.loTop + l < (done.getD p.1 default).size

/-- **The ranges' entry at a net slot's column**: the net's entry there, within the slot. -/
theorem rangesEntry_at {st : Stmt} (hd : st.c.ranges.toList.Pairwise Apart) {n s off x : ℕ}
    (hm : st.c.rangeOf n ∈ st.c.ranges.toList) (hk : (st.c.rangeOf n).kind = .net n) (h : AtSlot st.c n s off x)
    (e : Kind → ℕ → ℕ → ZMod 2) (j : ℕ) :
    rangesEntry st e x j =
      if j / 2 ^ (st.c.rangeOf n).slotLog = (st.c.rangeOf n).pos0 + s then
        e (.net n) off (j % 2 ^ (st.c.rangeOf n).slotLog) else 0 := by
  obtain ⟨hs, ho, rfl⟩ := h
  rw [rangesEntry_of_covers hd hm (covers_slot hs ho) e j, (slot_div s ho).1, (slot_div s ho).2, hk]

/-- A slot's rows, counted: its local row `l`, below the slot size, read at the slot's columns. -/
theorem count_slot {r : Range} {s : ℕ} (l : List ℕ) (hl : ∀ y ∈ l, y < 2 ^ r.slotLog) (j : ℕ) :
    (if j / 2 ^ r.slotLog = r.pos0 + s then ((l.count (j % 2 ^ r.slotLog) : ℕ) : ZMod 2) else 0) =
      (((l.map (r.slot s + ·)).count j : ℕ) : ZMod 2) := by
  have := count_map_add_ite (B := r.pos0 + s) (Nat.two_pow_pos _) l hl j
  exact this.symm

theorem with_lookup_none {n : Net} (h : n.lookup = none) : { n with lookup := none } = n := by
  cases n; cases h; rfl

theorem size_le_slot {d : D} {nm : String} {n : Net} (h : Typed.netOfD d nm = .ok n) : d.size ≤ 2 ^ n.unitLog := by
  rw [(Typed.netOfD_ok h).1]
  exact le_trans (le_two_pow_log2ceil _) (Nat.pow_le_pow_right (by norm_num) (le_max_right _ _))

namespace TemplateLayout

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

theorem range (T : TemplateLayout st tb words done 𝒖) {n : ℕ} (hn : n < st.c.nets.size) :
    st.c.rangeOf n ∈ st.c.ranges.toList ∧ (st.c.rangeOf n).kind = .net n ∧
      (st.c.rangeOf n).count = st.c.g * st.c.perVu.getD n 0 ∧
      (st.c.rangeOf n).slotLog = (st.c.nets.getD n default).2.unitLog := by
  obtain ⟨hm, hk, -⟩ := rangeOf_spec (T.one_range n hn)
  exact ⟨hm, hk, T.count _ hm _ hk, T.slotLog _ hm _ hk⟩

/-- The root's slot in template instance `g` holds its own columns. -/
theorem at_own (T : TemplateLayout st tb words done 𝒖) {g x : ℕ} (hg : g < st.c.g) (hx : x < 𝒖.size) :
    AtSlot st.c st.c.unitNet g x (Typed.col st.c tb g x) := by
  obtain ⟨-, -, hc, hsl⟩ := T.range T.unitNet
  obtain ⟨nm, hr⟩ := T.root
  refine ⟨by rw [hc, T.root_perVu, Nat.mul_one]; exact hg, ?_, ?_⟩
  · rw [hsl]
    have := size_le_slot hr
    exact lt_of_lt_of_le hx this
  · rw [Typed.col_own (by rw [T.own]; exact hx)]
    unfold Circuit.slotCol
    rw [T.root_perVu, Nat.mul_one, Nat.add_zero]

/-- Part `i`'s slot in template instance `g` holds its callee's columns, from its base. -/
theorem at_part (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb words done 𝒖) {g x i : ℕ}
    (hg : g < st.c.g) (hi : i < tb.parts.size) (hu : i < 𝒖.parts.size) (hlo : 𝒖.parts[i].2 ≤ x)
    (hhi : x < 𝒖.parts[i].2 + (done.getD 𝒖.parts[i].1 default).size) :
    AtSlot st.c tb.parts[i].1 (g * st.c.perVu.getD tb.parts[i].1 0 + tb.parts[i].2.1) (x - 𝒖.parts[i].2)
      (Typed.col st.c tb g x) := by
  obtain ⟨hb, hn, -, -, -, hq, ⟨nm, hnet⟩, -⟩ := T.part i hi hu
  obtain ⟨-, hc, hsl⟩ := (T.range hn).2
  have hreg := part_region hd (Array.getElem_mem hu)
  refine ⟨by rw [hc]; exact Rows.mul_add_lt hg hq, ?_, ?_⟩
  · rw [hsl]
    have : (done.getD 𝒖.parts[i].1 default).size ≤ 2 ^ (st.c.nets.getD tb.parts[i].1 default).2.unitLog := by
      have := size_le_slot hnet; exact this
    omega
  · rw [Typed.col_part hi (by rw [T.own]; omega) (by rw [hb]; exact hlo) (fun k hk hik => by
      have hk' : k < 𝒖.parts.size := by rw [← T.parts_size]; exact hk
      rw [(T.part k hk hk').1]
      have := parts_apart hd hik hk'
      omega), hb]
    rfl

/-- **Where a unit column is**: an own column in the root's slot, or a part's column in the part's slot. -/
theorem at_inUnit (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb words done 𝒖)
    {g x : ℕ} (hg : g < st.c.g) (hx : InUnit done 𝒖 x) :
    (x < 𝒖.size ∧ AtSlot st.c st.c.unitNet g x (Typed.col st.c tb g x)) ∨
      ∃ i, ∃ (hi : i < tb.parts.size) (hu : i < 𝒖.parts.size), 𝒖.size ≤ x ∧ 𝒖.parts[i].2 ≤ x ∧
        x < 𝒖.parts[i].2 + (done.getD 𝒖.parts[i].1 default).size ∧
        AtSlot st.c tb.parts[i].1 (g * st.c.perVu.getD tb.parts[i].1 0 + tb.parts[i].2.1) (x - 𝒖.parts[i].2)
          (Typed.col st.c tb g x) := by
  rcases hx with hx | ⟨p, hp, hlo, hhi⟩
  · exact Or.inl ⟨hx, T.at_own hg hx⟩
  · obtain ⟨i, hu, rfl⟩ := Array.mem_iff_getElem.1 hp
    have hi : i < tb.parts.size := by rw [T.parts_size]; exact hu
    have := (part_region hd hp).2.2.1
    exact Or.inr ⟨i, hi, hu, by omega, hlo, hhi, T.at_part hd hg hi hu hlo hhi⟩

/-- A unit column is in the block. -/
theorem col_lt (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb words done 𝒖)
    {g x : ℕ} (hg : g < st.c.g) (hx : InUnit done 𝒖 x) : Typed.col st.c tb g x < 2 ^ st.c.kLog := by
  rcases T.at_inUnit hd hg hx with ⟨-, h⟩ | ⟨i, hi, hu, -, -, -, h⟩
  · exact h.lt T.bound (T.range T.unitNet).1
  · exact h.lt T.bound (T.range (T.part i hi hu).2.1).1

/-- **Unit columns are at distinct positions**, across template instances. -/
theorem col_inj (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb words done 𝒖)
    {g g' x x' : ℕ} (hg : g < st.c.g) (hg' : g' < st.c.g) (hx : InUnit done 𝒖 x) (hx' : InUnit done 𝒖 x')
    (h : Typed.col st.c tb g x = Typed.col st.c tb g' x') : g = g' ∧ x = x' := by
  have hR := fun {n} (hn : n < st.c.nets.size) => T.range hn
  rcases T.at_inUnit hd hg hx with ⟨hxo, ha⟩ | ⟨i, hi, hu, hxo, hlo, hhi, ha⟩ <;>
    rcases T.at_inUnit hd hg' hx' with ⟨hxo', ha'⟩ | ⟨i', hi', hu', hxo', hlo', hhi', ha'⟩ <;>
    rw [← h] at ha'
  · obtain ⟨-, h1, h2⟩ := ha.inj T.disj (hR T.unitNet).1 (hR T.unitNet).2.1 (hR T.unitNet).1 (hR T.unitNet).2.1 ha'
    exact ⟨h1, h2⟩
  · have hn' := (T.part i' hi' hu').2.1
    exact absurd (ha.inj T.disj (hR T.unitNet).1 (hR T.unitNet).2.1 (hR hn').1 (hR hn').2.1 ha').1.symm
      (T.part i' hi' hu').2.2.1
  · have hn := (T.part i hi hu).2.1
    exact absurd (ha.inj T.disj (hR hn).1 (hR hn).2.1 (hR T.unitNet).1 (hR T.unitNet).2.1 ha').1
      (T.part i hi hu).2.2.1
  · have hn := (T.part i hi hu).2.1
    have hn' := (T.part i' hi' hu').2.1
    obtain ⟨h1, h2, h3⟩ := ha.inj T.disj (hR hn).1 (hR hn).2.1 (hR hn').1 (hR hn').2.1 ha'
    have hq := (T.part i hi hu).2.2.2.2.2.1
    have hq' := (T.part i' hi' hu').2.2.2.2.2.1
    rw [← h1] at hq' h2
    have hgg : g = g' := by
      have e1 := Nat.add_mul_div_left tb.parts[i].2.1 g (lt_of_le_of_lt (Nat.zero_le _) hq)
      have e2 := Nat.add_mul_div_left tb.parts[i'].2.1 g' (lt_of_le_of_lt (Nat.zero_le _) hq)
      rw [Nat.div_eq_of_lt hq, Nat.zero_add] at e1
      rw [Nat.div_eq_of_lt hq', Nat.zero_add] at e2
      rw [← e1, ← e2, Nat.add_comm, Nat.mul_comm, h2, Nat.add_comm, Nat.mul_comm]
    subst hgg
    have hii := T.part_inj i hi i' hi' h1 (by omega)
    subst hii
    exact ⟨rfl, by omega⟩

/-! ## Where the typed tail writes -/

theorem callee_col (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size) {p : ℕ × ℕ}
    (hp : p ∈ 𝒖.parts) {y : ℕ} (hy : y ∈ (done.getD p.1 default).inCols) : y < (done.getD p.1 default).size := by
  have hreg := part_region hd hp
  have := ((callee_rows hd (by omega) (by omega)).1 y hy)
  rw [← hreg.2.1]
  exact (Array.getElem?_eq_some_iff.1 this).1

/-- **An input column** is a unit column, and no constant's. -/
theorem inCols_row (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (U : UnitShape done 𝒖) {x : ℕ} (hx : x ∈ 𝒖.inCols) : InUnit done 𝒖 x ∧ NotConst 𝒖 x := by
  have hnone := U.in_delta x hx
  refine ⟨U.order_in x (by unfold order; exact List.mem_append_left _ (by simpa using hx)), fun h => ?_, ?_⟩
  · rw [h, delta_const hd hul] at hnone; cases hnone
  · rw [hnone]; simp

/-- **A bound input's row** is a part's column, and no constant's; it reads its Δ source. -/
theorem bind_row (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) {rs : ℕ × Array ℕ} (hrs : rs ∈ tb.bind.toList) :
    ∃ s, 𝒖.delta.find? (·.1 == rs.1) = some (rs.1, some s) ∧ rs.2 = s.toArray ∧ InUnit done 𝒖 rs.1 ∧
      NotConst 𝒖 rs.1 := by
  rw [T.bind, Array.toList_filterMap, List.mem_filterMap] at hrs
  obtain ⟨⟨r, src⟩, he, hf⟩ := hrs
  cases src with
  | none => cases hf
  | some s =>
    simp only [Option.map_some, Option.some.injEq] at hf
    subst hf
    have hmem : (r, some s) ∈ 𝒖.delta := by simpa using he
    have hfind : 𝒖.delta.find? (·.1 == r) = some (r, some s) := by
      rw [← Array.find?_toList]; exact find?_of_mem (delta_nodup hd) he
    refine ⟨s, hfind, rfl, ?_, fun h => ?_, by rw [hfind]; simp⟩
    · rcases delta_entry hd hmem with ⟨-, h⟩ | ⟨p, hp, ⟨-, h⟩ | ⟨-, y, hy, hr⟩⟩
      · cases h
      · cases h
      · have := callee_col hd hul hp hy
        exact Or.inr ⟨p, hp, by simp only at hr; omega, by simp only at hr; omega⟩
    · simp only at h
      rw [h, delta_const hd hul] at hfind
      cases hfind

/-- **A cross entry's row** is an own column, and no constant's. -/
theorem cross_row (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb words done 𝒖)
    (U : UnitShape done 𝒖) {e : ℕ × Array ℕ × Array ℕ} (he : e ∈ tb.cross.toList) :
    e.1 < 𝒖.size ∧ NotConst 𝒖 e.1 := by
  rw [T.cross, crossOf_toList, List.mem_filterMap] at he
  obtain ⟨r, hr, hc⟩ := he
  rw [List.mem_range] at hr
  have hr1 : e.1 = r := by
    unfold crossRow at hc
    split at hc
    · cases hc
    · cases hc; rfl
  rw [hr1]
  have hne : r ≠ 𝒖.const := by
    rintro rfl
    have hf : [𝒖.const].filter (· ≥ 𝒖.size) = [] := by
      rw [List.filter_eq_nil_iff]
      intro x hx
      simp only [List.mem_singleton] at hx
      subst hx
      simp only [ge_iff_le, decide_eq_true_eq, not_le]
      exact hr
    simp only [crossRow, U.const_row, hf, List.isEmpty_nil, Bool.and_self, ↓reduceIte, reduceCtorEq] at hc
  refine ⟨hr, hne, fun hf => ?_⟩
  rcases delta_entry hd (Array.mem_of_find?_eq_some hf) with ⟨h, -⟩ | ⟨p, hp, ⟨h, -⟩ | ⟨h, -⟩⟩
  · exact hne h
  · have := (part_region hd hp).2.2.1
    simp only at h
    omega
  · simp at h

/-- **Every row the typed tail writes** is the position of a unit column that is no constant's, in its template
instance. -/
theorem tailVU_rows (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) {side : Bool} {g' : ℕ} {p : ℕ × ℕ}
    (hp : p ∈ Typed.tailVU st.c tb side g') :
    ∃ x, InUnit done 𝒖 x ∧ NotConst 𝒖 x ∧ p.1 = Typed.col st.c tb g' x := by
  unfold Typed.tailVU at hp
  simp only [List.mem_append, List.mem_flatMap] at hp
  rcases hp with (⟨w, hw, hp⟩ | ⟨rs, hrs, hp⟩) | ⟨e, he, hp⟩
  · rw [List.mem_range] at hw
    have hx : tb.inCols[w]! ∈ 𝒖.inCols := by
      rw [getElem!_pos tb.inCols w hw]
      exact (fun a (ha : a ∈ tb.inCols) => (T.inCols ▸ ha : a ∈ 𝒖.inCols)) _ (Array.getElem_mem hw)
    obtain ⟨h1, h2⟩ := inCols_row hd hul U hx
    refine ⟨_, h1, h2, ?_⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;> rfl
  · obtain ⟨s, -, -, h1, h2⟩ := T.bind_row hd hul hrs
    refine ⟨_, h1, h2, ?_⟩
    simp only [List.mem_cons, List.mem_map] at hp
    rcases hp with rfl | ⟨x, -, rfl⟩ <;> rfl
  · obtain ⟨h1, h2⟩ := T.cross_row hd U he
    refine ⟨_, Or.inl h1, h2, ?_⟩
    obtain ⟨x, -, rfl⟩ := List.mem_map.1 hp
    rfl

/-- **The typed tail writes nothing at a row no non-constant unit column is at.** -/
theorem tail_zero (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) {y : ℕ}
    (hy : ∀ g' < st.c.g, ∀ x, InUnit done 𝒖 x → NotConst 𝒖 x → Typed.col st.c tb g' x ≠ y) (side : Bool) (j : ℕ) :
    (Typed.tail st.c tb side).count (y, j) = 0 := by
  refine List.count_eq_zero.2 fun hm => ?_
  unfold Typed.tail at hm
  obtain ⟨g', hg', hm⟩ := List.mem_flatMap.1 hm
  obtain ⟨x, hx, hnc, he⟩ := T.tailVU_rows hd hul U hm
  exact hy g' (List.mem_range.1 hg') x hx hnc he.symm

/-- **The typed tail at a unit column's position**: its bound-input entries there, and its cross entries there. -/
theorem tail_count (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) {g c : ℕ} (hg : g < st.c.g) (hc : InUnit done 𝒖 c)
    (hin : c ∉ 𝒖.inCols) (side : Bool) (j : ℕ) :
    (Typed.tail st.c tb side).count (Typed.col st.c tb g c, j) =
      (bindCols (Typed.col st.c tb g) 𝒖 c).count j + (crossCols (Typed.col st.c tb g) 𝒖 side c).count j := by
  unfold Typed.tail
  rw [count_flatMap_range _ _ hg (fun g' hg' hne => List.count_eq_zero.2 fun hm => by
    obtain ⟨x, hx, -, he⟩ := T.tailVU_rows hd hul U hm
    exact hne (T.col_inj hd hg' hg hx hc he.symm).1)]
  unfold Typed.tailVU
  rw [List.count_append, List.count_append]
  have hA : ((List.range tb.inCols.size).flatMap fun w =>
      [(Typed.col st.c tb g tb.inCols[w]!, Typed.col st.c tb g tb.inCols[w]!),
        (Typed.col st.c tb g tb.inCols[w]!, Typed.copySrc st.c g w)]).count (Typed.col st.c tb g c, j) = 0 := by
    refine List.count_eq_zero.2 fun hm => ?_
    obtain ⟨w, hw, hm⟩ := List.mem_flatMap.1 hm
    rw [List.mem_range] at hw
    have hx : tb.inCols[w]! ∈ 𝒖.inCols := by
      rw [getElem!_pos tb.inCols w hw]
      exact (fun a (ha : a ∈ tb.inCols) => (T.inCols ▸ ha : a ∈ 𝒖.inCols)) _ (Array.getElem_mem hw)
    have he : Typed.col st.c tb g tb.inCols[w]! = Typed.col st.c tb g c := by
      simp only [List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq] at hm
      rcases hm with ⟨h, -⟩ | ⟨h, -⟩ <;> exact h.symm
    have := (T.col_inj hd hg hg (inCols_row hd hul U hx).1 hc he).2
    exact hin (this ▸ hx)
  rw [hA, Nat.zero_add]
  have hB := count_flatMap_rows2 tb.bind.toList (fun rs => Typed.col st.c tb g rs.1) (fun rs => rs.2.toList)
    (Typed.col st.c tb g) (Typed.col st.c tb g c) j
  have hC := count_flatMap_rows3 tb.cross.toList (fun e => Typed.col st.c tb g e.1)
    (fun e => (if side then e.2.1 else e.2.2).toList) (Typed.col st.c tb g) (Typed.col st.c tb g c) j
  rw [hB, hC]
  rw [List.filter_congr (q := fun rs => decide (rs.1 = c)) (fun rs hrs => by
      obtain ⟨-, -, -, h1, -⟩ := T.bind_row hd hul hrs
      exact decide_eq_decide.2 ⟨fun h => (T.col_inj hd hg hg h1 hc h).2, fun h => by rw [h]⟩),
    List.filter_congr (l := tb.cross.toList) (q := fun e => decide (e.1 = c)) (fun e he => by
      obtain ⟨h1, -⟩ := T.cross_row hd U he
      exact decide_eq_decide.2 ⟨fun h => (T.col_inj hd hg hg (Or.inl h1) hc h).2, fun h => by rw [h]⟩)]
  rw [T.cross, crossOf_at]
  congr 2
  rw [T.bind]
  exact bind_at _ _ (delta_nodup hd) c

/-! ## Δ at a unit column's row -/

/-- **Δ at a row no flat-statement input bit is at**: its first loop's pairs there, then the typed tail's. -/
theorem delta_count (T : TemplateLayout st tb words done 𝒖) {x : ℕ} (hx : ∀ y, DeltaIn st.c y → y ≠ x) (j : ℕ) :
    deltaEntry st.da x j =
        ((((constPairs st.c st.pin).count (x, j) + (Typed.tail st.c tb true).count (x, j) : ℕ)) : ZMod 2) ∧
      deltaEntry st.db x j =
        ((((constPairs st.c st.pin).count (x, j) + (Typed.tail st.c tb false).count (x, j) : ℕ)) : ZMod 2) := by
  obtain ⟨d, ⟨rest, hdl, hr⟩, ha, hb⟩ := T.delta
  have h0 : rest.count (x, j) = 0 := List.count_eq_zero.2 fun hm => hx _ (hr _ hm) rfl
  refine ⟨?_, ?_⟩
  · rw [deltaEntry_eq_count, show st.da.toList = _ from ha, hdl, List.count_append, List.count_append, h0,
      Nat.add_zero]
  · rw [deltaEntry_eq_count, show st.db.toList = _ from hb, hdl, List.count_append, List.count_append, h0,
      Nat.add_zero]

/-- Δ's other flat rows are in the rows' circuits' slots, so at no other net's column. -/
theorem deltaIn_ne (T : TemplateLayout st tb words done 𝒖) {n s off x : ℕ} (hn : n < st.c.nets.size)
    (hns : n ≠ st.c.shaNet) (hnh : n ≠ st.c.hmNet) (hx : AtSlot st.c n s off x) :
    ∀ y, DeltaIn st.c y → y ≠ x := by
  intro y hy hyx
  obtain ⟨n', hn', s', off', hat, -⟩ := T.deltaIn y hy
  have hn'' : n' < st.c.nets.size := by rcases hn' with rfl | rfl; exact T.shaNet.1; exact T.hmNet.1
  obtain ⟨h1, -⟩ := hat.inj T.disj (T.range hn'').1 (T.range hn'').2.1 (T.range hn).1 (T.range hn).2.1 (hyx ▸ hx)
  rcases hn' with rfl | rfl
  · exact hns h1.symm
  · exact hnh h1.symm

/-- **The block rows at a unit column's position**, counted: the slot net's row there, placed; Δ's first-loop pairs
there; and the typed tail's bound-input and cross entries there. -/
theorem entry_at (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) {g c n sl off : ℕ} {d : D} (hg : g < st.c.g)
    (hc : InUnit done 𝒖 c) (hin : c ∉ 𝒖.inCols) (hn : n < st.c.nets.size) (hns : n ≠ st.c.shaNet)
    (hnh : n ≠ st.c.hmNet) (hat : AtSlot st.c n sl off (Typed.col st.c tb g c))
    (hnet : ∃ nm, Typed.netOfD d nm = .ok { (st.c.nets.getD n default).2 with lookup := none }) (LB : List ℕ)
    (hLB : ∀ y ∈ LB, y < 2 ^ (st.c.nets.getD n default).2.unitLog)
    (hB : (st.c.nets.getD n default).2.unitLog ≤ 32 → ∀ j', sparseEntry (st.c.nets.getD n default).2.b off j' +
      lookupEntry (st.c.nets.getD n default).2.lookup off j' = ((LB.count j' : ℕ) : ZMod 2)) (j : ℕ) :
    rangesEntry st (slotEntryA st) (Typed.col st.c tb g c) j + deltaEntry st.da (Typed.col st.c tb g c) j =
        (((((Typed.rowsA d).getD off #[]).toList.map ((st.c.rangeOf n).slot sl + ·)).count j +
          (constPairs st.c st.pin).count (Typed.col st.c tb g c, j) + (bindCols (Typed.col st.c tb g) 𝒖 c).count j +
          (crossCols (Typed.col st.c tb g) 𝒖 true c).count j : ℕ) : ZMod 2) ∧
      rangesEntry st (slotEntryB st) (Typed.col st.c tb g c) j + deltaEntry st.db (Typed.col st.c tb g c) j =
        (((LB.map ((st.c.rangeOf n).slot sl + ·)).count j +
          (constPairs st.c st.pin).count (Typed.col st.c tb g c, j) + (bindCols (Typed.col st.c tb g) 𝒖 c).count j +
          (crossCols (Typed.col st.c tb g) 𝒖 false c).count j : ℕ) : ZMod 2) := by
  obtain ⟨nm, hnet⟩ := hnet
  obtain ⟨hm, hk, -, hsl⟩ := T.range hn
  have h32 : (st.c.nets.getD n default).2.unitLog ≤ 32 := by
    rw [← hsl]; exact le_trans (hat.slotLog_le T.bound hm) T.kLog
  have hs : ∀ j', sparseEntry (st.c.nets.getD n default).2.a off j' = _ := fun j' => (netOfD_sparse hnet h32 off j').1
  have hla : ∀ y ∈ ((Typed.rowsA d).getD off #[]).toList, y < 2 ^ (st.c.nets.getD n default).2.unitLog :=
    (rowsA_lt hnet off).1
  obtain ⟨hda, hdb⟩ := T.delta_count (T.deltaIn_ne hn hns hnh hat) j
  have hta := T.tail_count hd hul U hg hc hin true j
  have htb := T.tail_count hd hul U hg hc hin false j
  have hre := fun e => rangesEntry_at T.disj hm hk hat e j
  refine ⟨?_, ?_⟩
  · rw [hre, hda, hta]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.a off _ else 0) + _ = _
    rw [hs, count_slot _ (by rw [hsl]; exact hla) j]
    push_cast; ring
  · rw [hre, hdb, htb]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ +
      lookupEntry (st.c.nets.getD n default).2.lookup off _ else 0) + _ = _
    rw [hB h32, count_slot _ (by rw [hsl]; exact hLB) j]
    push_cast; ring

/-- A row of the unit whose block row doesn't read `one` reads only columns of its order. -/
theorem cols_in_order (hd : deriveChecked types ls info words unit = .ok done) {c : ℕ} (hc : c ∈ order done 𝒖)
    (hin : c ∉ 𝒖.inCols.toList) {p : List ℕ × List ℕ} (hp : ∀ one', blockRow words 𝒖 one' c = some p) :
    ∀ x ∈ p.1 ++ p.2, x ∈ order done 𝒖 := by
  intro x hx
  obtain ⟨-, -, htake, hrd, -⟩ := order_sound hd (x + 1)
  obtain ⟨ℓ, hℓ, rfl⟩ := List.getElem_of_mem hc
  have hge : 𝒖.inCols.size ≤ ℓ := by
    by_contra hlt
    apply hin
    rw [← htake]
    have := List.getElem_mem (l := (order done 𝒖).take 𝒖.inCols.size) (n := ℓ) (by rw [List.length_take]; omega)
    rwa [List.getElem_take] at this
  exact List.mem_of_mem_take (hrd ℓ hℓ hge p (hp _) x hx (by omega))

/-- An own row with no Δ entry is its derived row: no read covers an own row (`TemplateLayout.reads_past`). -/
theorem blockRow_own (T : TemplateLayout st tb words done 𝒖) {c : ℕ} (hc : c < 𝒖.size)
    (hf : 𝒖.delta.find? (·.1 == c) = none) (one' : ℕ) : blockRow words 𝒖 one' c = 𝒖.rows[c]? := by
  have hnc : 𝒖.reads.find? (·.covers c) = none := by
    rw [← Array.find?_toList]
    refine List.find?_eq_none.2 fun r hr hcv => ?_
    have := T.reads_past r hr
    simp only [ReadRec.covers, Bool.and_eq_true, decide_eq_true_eq] at hcv
    omega
  unfold blockRow fullRow
  rw [hf, hnc]

/-- **A part's other row is its callee's row as the verifier folds it**, shifted to the part's base: the part's reads are
its callee's (`UnitShape.part_reads`, and `readSide_shift`). -/
theorem blockRow_part (U : UnitShape done 𝒖) {p : ℕ × ℕ} (hp : p ∈ 𝒖.parts) {c : ℕ}
    (hc : c < (done.getD p.1 default).size) (hf : 𝒖.delta.find? (·.1 == p.2 + c) = none)
    (hrow : 𝒖.rows[p.2 + c]? = some (shiftRow p.2 ((done.getD p.1 default).rows.getD c ([], [])))) (one' : ℕ) :
    blockRow words 𝒖 one' (p.2 + c) = some (shiftRow p.2 (((done.getD p.1 default).rows.getD c ([], [])).1,
      partB words (done.getD p.1 default) c)) := by
  unfold blockRow fullRow
  rw [hf, U.part_reads p hp c hc]
  unfold partB
  cases hr : (done.getD p.1 default).reads.find? (·.covers c) with
  | none => simp only [Option.map_none]; rw [hrow]
  | some r =>
    simp only [Option.map_some, hrow, shiftRow]
    rw [readSide_shift]
    rfl

end TemplateLayout

/-! ## Counting a row's entries -/

theorem count_map_filter_split (l : List ℕ) (G : ℕ → ℕ) (n j : ℕ) :
    (l.map G).count j = ((l.filter (· < n)).map G).count j + ((l.filter (· ≥ n)).map G).count j := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    by_cases ha : a < n
    · simp only [List.map_cons, List.filter_cons, ha, decide_true, ↓reduceIte, ge_iff_le,
        show ¬ n ≤ a by omega, decide_false, Bool.false_eq_true, List.count_cons, ih]
      omega
    · simp only [List.map_cons, List.filter_cons, ha, decide_false, Bool.false_eq_true, ↓reduceIte, ge_iff_le,
        show n ≤ a by omega, decide_true, List.count_cons, ih]
      omega

/-- A constant's row: its own entry, and Δ's `(k, k)` and `(k, pin)` unless it is the pin, read the pin. -/
theorem const_sum {P : Prop} [Decidable P] (hP : P) (y pin j : ℕ) :
    ((([y].count j + (if P ∧ y ≠ pin then (if j = y then 1 else 0) + (if j = pin then 1 else 0) else 0) +
      ([] : List ℕ).count j + ([] : List ℕ).count j : ℕ)) : ZMod 2) = (([pin].count j : ℕ) : ZMod 2) := by
  by_cases hyp : y = pin
  · subst hyp; simp
  · rw [ite_eq_left ⟨hP, hyp⟩]
    by_cases h1 : j = y
    · subst h1
      simp [hyp, Ne.symm hyp]
      decide
    · by_cases h2 : j = pin
      · subst h2; simp [h1, Ne.symm h1]
      · simp [h1, h2, Ne.symm h1, Ne.symm h2]

/-- A bound input's row: its own entry and Δ's self entry cancel, and its source's entries remain. -/
theorem bind_sum (y j : ℕ) (l : List ℕ) :
    ((([y].count j + 0 + (y :: l).count j + ([] : List ℕ).count j : ℕ)) : ZMod 2) = ((l.count j : ℕ) : ZMod 2) := by
  have h : [y].count j + 0 + (y :: l).count j + ([] : List ℕ).count j = l.count j + 2 * [y].count j := by
    simp only [List.count_cons, List.count_nil]; ring
  rw [h]
  push_cast
  rw [show (2 : ZMod 2) = 0 from rfl, zero_mul, add_zero]


/-! ## The block rows at each template instance's unit -/

namespace TemplateLayout

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

theorem pin_lt {u : D} (T : TemplateLayout st tb words done u) : st.pin < 2 ^ st.c.kLog := by
  obtain ⟨rp, hrp, ip, hk, hc, hpin, -⟩ := T.pin
  have hoffW : (st.c.nets.getD ip default).2.constPos < 2 ^ rp.slotLog := by
    rw [T.slotLog rp hrp ip hk]; exact T.fit ip
  have := (covers_iff _ _).1 (covers_slot hc hoffW)
  have := count_mul_le_spanH rp
  have := T.bound rp hrp
  rw [start_eq_slot] at hpin
  omega

/-- **The accepted template statement, as the model's block R1CS**: the verifier's `m`, `k_log` and pin, the level-0
matrices it folds (`placedA`, `placedB`), and public regions `regs`. -/
def model {u : D} (T : TemplateLayout st tb words done u) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    Model.Statement where
  m := st.m
  kLog := st.c.kLog
  hk := by rw [T.m]; omega
  hk6 := T.kLog6
  pin := ⟨st.pin, T.pin_lt⟩
  A₀ := placedA st
  B₀ := placedB st
  regions := regs

/-- **The pin's row reads the pin**, in `A` and in `B`: Δ writes nothing there, no unit column that is no constant's is
at it. -/
theorem placed_pin (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) (j : ℕ) :
    rangesEntry st (slotEntryA st) st.pin j + deltaEntry st.da st.pin j = (if j = st.pin then 1 else 0) ∧
      rangesEntry st (slotEntryB st) st.pin j + deltaEntry st.db st.pin j = (if j = st.pin then 1 else 0) := by
  obtain ⟨rp, hrp, ip, hk, hc, hpin, hcr⟩ := T.pin
  have hoffW : (st.c.nets.getD ip default).2.constPos < 2 ^ rp.slotLog := by
    rw [T.slotLog rp hrp ip hk]; exact T.fit ip
  have hp : st.pin = rp.slot 0 + (st.c.nets.getD ip default).2.constPos := by rw [hpin, start_eq_slot]
  have hx : ∀ y, DeltaIn st.c y → y ≠ st.pin := by
    intro y hy hyp
    obtain ⟨n', hn', s', off', hat, hlt⟩ := T.deltaIn y hy
    have hn'' : n' < st.c.nets.size := by
      rcases hn' with rfl | rfl
      · exact T.shaNet.1
      · exact T.hmNet.1
    obtain ⟨-, rfl, -, rfl⟩ := hat.eq_range T.disj (T.range hn'').1 (T.range hn'').2.1 hrp hk hc hoffW (hyp.trans hp)
    have := hcr.1
    omega
  have hcp : (constPairs st.c st.pin).count (st.pin, j) = 0 :=
    List.count_eq_zero.2 fun hm => (constPairs_fst hm).elim fun _ ⟨_, _, _, _, _, _, hne⟩ => hne rfl
  have hty : ∀ g' < st.c.g, ∀ x, InUnit done 𝒖 x → NotConst 𝒖 x → Typed.col st.c tb g' x ≠ st.pin := by
    intro g' hg' x hxu hnc he
    rcases T.at_inUnit hd hg' hxu with ⟨-, hat⟩ | ⟨i, hi, hu, -, hlo, hhi, hat⟩
    · obtain ⟨-, rfl, -, hoff⟩ :=
        hat.eq_range T.disj (T.range T.unitNet).1 (T.range T.unitNet).2.1 hrp hk hc hoffW (he.trans hp)
      obtain ⟨nm, hr⟩ := T.root
      exact hnc.1 (hoff.symm.trans (Typed.netOfD_ok hr).2.2.2.2.1)
    · obtain ⟨-, hn, -, -, -, -, ⟨nm, hnet⟩, -⟩ := T.part i hi hu
      obtain ⟨-, rfl, -, hoff⟩ := hat.eq_range T.disj (T.range hn).1 (T.range hn).2.1 hrp hk hc hoffW (he.trans hp)
      have hcst : (st.c.nets.getD tb.parts[i].1 default).2.constPos = (done.getD 𝒖.parts[i].1 default).const :=
        (Typed.netOfD_ok hnet).2.2.2.2.1
      have hx' : x = 𝒖.parts[i].2 + (done.getD 𝒖.parts[i].1 default).const := by omega
      have hf := (part_rows hd (Array.getElem_mem hu) (c := (done.getD 𝒖.parts[i].1 default).const)
        (by omega)).2.1 rfl
      rw [← hx'] at hf
      exact hnc.2 hf
  obtain ⟨hda, hdb⟩ := T.delta_count hx j
  rw [hda, hdb, hcp, T.tail_zero hd hul U hty true j, T.tail_zero hd hul U hty false j]
  simp only [Nat.cast_zero, add_zero]
  constructor
  · conv_lhs => rw [hp]
    rw [rangesEntry_of_covers T.disj hrp (covers_slot hc hoffW) _ _, (slot_div 0 hoffW).1, (slot_div 0 hoffW).2, hk,
      ite_slot_eq (v := slotEntryA st (.net ip) (st.c.nets.getD ip default).2.constPos) (Nat.two_pow_pos _) hoffW
        hcr.2.1, hp]
    rfl
  · conv_lhs => rw [hp]
    rw [rangesEntry_of_covers T.disj hrp (covers_slot hc hoffW) _ _, (slot_div 0 hoffW).1, (slot_div 0 hoffW).2, hk,
      ite_slot_eq (v := slotEntryB st (.net ip) (st.c.nets.getD ip default).2.constPos) (Nat.two_pow_pos _) hoffW
        hcr.2.2, hp]
    rfl

/-- `rows_at` at a part's column. -/
theorem rows_part (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) {g : ℕ} (hg : g < st.c.g) {one : ℕ}
    {c : ℕ} (hin' : c ∉ 𝒖.inCols) (hcU : InUnit done 𝒖 c) {p : List ℕ × List ℕ}
    (hp : blockRow words 𝒖 one c = some p)
    (hfree : ∀ q : List ℕ × List ℕ, (∀ one', blockRow words 𝒖 one' c = some q) →
      (q.1.map fun x => if x = one then st.pin else Typed.col st.c tb g x) = q.1.map (Typed.col st.c tb g) ∧
      (q.2.map fun x => if x = one then st.pin else Typed.col st.c tb g x) = q.2.map (Typed.col st.c tb g))
    {i : ℕ} (hi : i < tb.parts.size) (hu : i < 𝒖.parts.size) (hge : 𝒖.size ≤ c) (hlo : 𝒖.parts[i].2 ≤ c)
    (hhi : c < 𝒖.parts[i].2 + (done.getD 𝒖.parts[i].1 default).size)
    (hat : AtSlot st.c tb.parts[i].1 (g * st.c.perVu.getD tb.parts[i].1 0 + tb.parts[i].2.1) (c - 𝒖.parts[i].2)
      (Typed.col st.c tb g c)) (j : ℕ) :
    rangesEntry st (slotEntryA st) (Typed.col st.c tb g c) j + deltaEntry st.da (Typed.col st.c tb g c) j =
        (((p.1.map fun x => if x = one then st.pin else Typed.col st.c tb g x).count j : ℕ) : ZMod 2) ∧
      rangesEntry st (slotEntryB st) (Typed.col st.c tb g c) j + deltaEntry st.db (Typed.col st.c tb g c) j =
        (((p.2.map fun x => if x = one then st.pin else Typed.col st.c tb g x).count j : ℕ) : ZMod 2) := by
  obtain ⟨-, hn, -, hns, hnh, -, hpn⟩ := T.part i hi hu
  have hnet' := hpn.1
  obtain ⟨nm, hnet⟩ := hnet'
  have hcst : (st.c.nets.getD tb.parts[i].1 default).2.constPos = (done.getD 𝒖.parts[i].1 default).const :=
    (Typed.netOfD_ok hnet).2.2.2.2.1
  obtain ⟨-, -, -, -, -, -, hca, hcb, -⟩ := Typed.netOfD_ok hnet
  have hwid : (done.getD 𝒖.parts[i].1 default).size ≤ 2 ^ (st.c.nets.getD tb.parts[i].1 default).2.unitLog := by
    have := size_le_slot hnet; exact this
  have hpm : 𝒖.parts[i] ∈ 𝒖.parts := Array.getElem_mem hu
  have hcpr := U.callee_prod _ hpm
  have hpr := U.part_reads _ hpm
  generalize hb : 𝒖.parts[i].2 = b at hlo hhi hat hpr
  generalize hcal : done.getD 𝒖.parts[i].1 default = d at hhi hcst hca hcb hnet hpn hwid hcpr hpr
  generalize hc' : c - b = c' at hat
  have hcd : c' < d.size := by omega
  have hceq : b + c' = c := by omega
  obtain ⟨hrd, hK1, hO⟩ := part_rows hd hpm (c := c') (by rw [hcal]; exact hcd)
  rw [hcal, hb, hceq] at hK1 hO
  rw [hcal] at hrd
  have hunc : ∀ q : List ℕ × List ℕ, d.rows.getD c' ([], []) = q → q.2 ≠ [] → d.reads.find? (·.covers c') = none := by
    intro q hq hne
    cases hr : d.reads.find? (·.covers c') with
    | none => rfl
    | some r => exact absurd (by rw [← hq]; exact (hcpr c' hcd r hr).1) hne
  have hK := constPairs_count (pin := st.pin) T.disj T.slotLog T.fit (T.range hn).1 (T.range hn).2.1 hat j
  have hc0 : ∀ side, crossCols (Typed.col st.c tb g) 𝒖 side c = [] := fun side => by
    unfold crossCols; rw [ite_eq_right (by omega)]
  have hju := (part_region hd hpm).1
  by_cases hcc : c' = d.const
  · -- the part's constant: a self row no read covers
    have hcr := (callee_rows hd (j := 𝒖.parts[i].1) (by omega) (by omega)).2
    rw [hcal] at hcr
    have hnc := hunc _ (getD_of_getElem? (hcc ▸ hcr)) (by simp)
    obtain ⟨hEA, hEB⟩ := T.entry_at hd hul U hg hcU hin' hn hns hnh hat ⟨nm, hnet⟩ ((Typed.rowsB d).getD c' #[]).toList
      (rowsA_lt hnet c').2 (fun h32 j' => hpn.entryB_unc h32 hnc j') j
    have hy := hat.2.2
    generalize hS : (st.c.rangeOf tb.parts[i].1).slot (g * st.c.perVu.getD tb.parts[i].1 0 + tb.parts[i].2.1) = S
      at hEA hEB hy
    have hf := hK1 hcc
    have hbp : p = ([one], [one]) := by
      unfold blockRow at hp; rw [hf] at hp; exact (Option.some.inj hp).symm
    subst hbp
    have hb0 : bindCols (Typed.col st.c tb g) 𝒖 c = [] := by unfold bindCols; rw [hf]
    have hP : c' = (st.c.nets.getD tb.parts[i].1 default).2.constPos := hcc.trans hcst.symm
    have hl : ∀ side : Bool, ((if side then Typed.rowsA d else Typed.rowsB d).getD c' #[]).toList.map (S + ·) =
        [Typed.col st.c tb g c] := by
      intro side; rw [hy, hcc]; cases side
      · rw [ite_eq_right Bool.false_ne_true, hcb]; rfl
      · rw [ite_eq_left rfl, hca]; rfl
    refine ⟨hEA.trans ?_, hEB.trans ?_⟩
    · rw [show (Typed.rowsA d) = (if true then Typed.rowsA d else Typed.rowsB d) from rfl, hl true, hK, hb0, hc0 true]
      refine (const_sum hP _ _ _).trans ?_
      simp
    · rw [show (Typed.rowsB d) = (if false then Typed.rowsA d else Typed.rowsB d) from rfl, hl false, hK, hb0, hc0 false]
      refine (const_sum hP _ _ _).trans ?_
      simp
  · have hK0 : (constPairs st.c st.pin).count (Typed.col st.c tb g c, j) = 0 := by
      rw [hK, ite_eq_right (fun h => hcc (h.1.trans hcst))]
    rcases hO hcc with ⟨s, hfs, hcin⟩ | ⟨hfn, hrow⟩
    · -- a bound input: its callee row is a self row no read covers, and Δ's binding reads its source
      have hcr := (callee_rows hd (j := 𝒖.parts[i].1) (by omega) (by omega)).1 c' (by rw [hcal]; exact hcin)
      rw [hcal] at hcr
      have hgd : d.rows.getD c' ([], []) = ([c'], [c']) := getD_of_getElem? hcr
      have hnc := hunc _ hgd (by simp)
      obtain ⟨hEA, hEB⟩ := T.entry_at hd hul U hg hcU hin' hn hns hnh hat ⟨nm, hnet⟩
        ((Typed.rowsB d).getD c' #[]).toList (rowsA_lt hnet c').2 (fun h32 j' => hpn.entryB_unc h32 hnc j') j
      have hy := hat.2.2
      generalize hS : (st.c.rangeOf tb.parts[i].1).slot (g * st.c.perVu.getD tb.parts[i].1 0 + tb.parts[i].2.1) = S
        at hEA hEB hy
      have hbp : p = (s, s) := by
        unfold blockRow at hp; rw [hfs] at hp; exact (Option.some.inj hp).symm
      subst hbp
      obtain ⟨hfa, hfb⟩ := hfree _ (fun one' => by unfold blockRow; rw [hfs])
      have hb1 : bindCols (Typed.col st.c tb g) 𝒖 c = Typed.col st.c tb g c :: s.map (Typed.col st.c tb g) := by
        unfold bindCols; rw [hfs]
      have hl : ([c'].toArray.toList.map (S + ·)) = [Typed.col st.c tb g c] := by rw [hy]; rfl
      refine ⟨hEA.trans ?_, hEB.trans ?_⟩
      · rw [rowsA_getD hcd, hgd, hl, hK0, hb1, hc0 true, hfa]
        exact bind_sum _ _ _
      · rw [rowsB_getD hcd, hgd, hl, hK0, hb1, hc0 false, hfb]
        exact bind_sum _ _ _
    · -- any other callee row: the callee's row as the verifier folds it, shifted into the part's slot
      have hbr := fun one' => blockRow_part (words := words) U hpm (c := c') (by rw [hcal]; exact hcd)
        (by rw [hb, hceq]; exact hfn) (by rw [hb, hceq, hcal]; exact hrow) one'
      simp only [hb, hcal, hceq] at hbr
      have hbp : p = shiftRow b ((d.rows.getD c' ([], [])).1, partB words d c') := by
        rw [hbr] at hp; exact (Option.some.inj hp).symm
      subst hbp
      obtain ⟨hfa, hfb⟩ := hfree _ hbr
      have hpb : ∀ y ∈ partB words d c', y < d.size := by
        intro y hy
        unfold partB at hy
        cases hr : d.reads.find? (·.covers c') with
        | none => rw [hr] at hy; exact hrd y (List.mem_append_right _ hy)
        | some r =>
          rw [hr] at hy
          unfold readSide at hy
          obtain ⟨l, hl, rfl⟩ := List.mem_map.1 hy
          exact (hcpr c' hcd r hr).2 l (List.mem_range.1 (List.mem_filter.1 hl).1)
      obtain ⟨hEA, hEB⟩ := T.entry_at hd hul U hg hcU hin' hn hns hnh hat ⟨nm, hnet⟩ (partB words d c')
        (fun y hy => lt_of_lt_of_le (hpb y hy) hwid)
        (fun h32 j' => hpn.entryB h32 hcd (fun r hr => (hcpr c' hcd r hr).1) j') j
      have hy := hat.2.2
      generalize hS : (st.c.rangeOf tb.parts[i].1).slot (g * st.c.perVu.getD tb.parts[i].1 0 + tb.parts[i].2.1) = S
        at hEA hEB hy
      have hb0 : bindCols (Typed.col st.c tb g) 𝒖 c = [] := by unfold bindCols; rw [hfn]
      have hshift : ∀ y < d.size, S + y = Typed.col st.c tb g (y + b) := by
        intro y hyl
        have hat' := T.at_part hd hg hi hu (x := y + b) (by omega) (by rw [hcal]; omega)
        rw [hb, Nat.add_sub_cancel] at hat'
        rw [hat'.2.2, hS]
      refine ⟨hEA.trans ?_, hEB.trans ?_⟩
      · rw [rowsA_getD hcd, hK0, hb0, hc0 true, hfa]
        simp only [shiftRow, List.map_map, List.count_nil, Nat.add_zero]
        rw [List.map_congr_left fun y hy => hshift y (hrd y (List.mem_append_left _ hy))]
        rfl
      · rw [hK0, hb0, hc0 false, hfb]
        simp only [shiftRow, List.map_map, List.count_nil, Nat.add_zero]
        rw [List.map_congr_left fun y hy => hshift y (hpb y hy)]
        rfl

/-- **The block rows at a unit row's position are its block row**, placed at `col g`: the four kinds of row the unit
has in its order past its inputs (its constant, an own row, a part's constant, a part's bound input, a part's other
row). -/
theorem rows_at (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) {g : ℕ} (hg : g < st.c.g) {one : ℕ}
    (hone : 𝒖.rows.size ≤ one) {c : ℕ} (hc : c ∈ order done 𝒖) (hin : c ∉ 𝒖.inCols.toList)
    {p : List ℕ × List ℕ} (hp : blockRow words 𝒖 one c = some p) (j : ℕ) :
    rangesEntry st (slotEntryA st) (Typed.col st.c tb g c) j + deltaEntry st.da (Typed.col st.c tb g c) j =
        (((p.1.map fun x => if x = one then st.pin else Typed.col st.c tb g x).count j : ℕ) : ZMod 2) ∧
      rangesEntry st (slotEntryB st) (Typed.col st.c tb g c) j + deltaEntry st.db (Typed.col st.c tb g c) j =
        (((p.2.map fun x => if x = one then st.pin else Typed.col st.c tb g x).count j : ℕ) : ZMod 2) := by
  have hin' : c ∉ 𝒖.inCols := fun h => hin (by simpa using h)
  have hcU := U.order_in c hc
  obtain ⟨-, hlt, -, -, -⟩ := order_sound hd one
  have hfree : ∀ q : List ℕ × List ℕ, (∀ one', blockRow words 𝒖 one' c = some q) →
      (q.1.map fun x => if x = one then st.pin else Typed.col st.c tb g x) = q.1.map (Typed.col st.c tb g) ∧
      (q.2.map fun x => if x = one then st.pin else Typed.col st.c tb g x) = q.2.map (Typed.col st.c tb g) := by
    intro q hq
    have hm := cols_in_order hd hc hin hq
    refine ⟨List.map_congr_left fun x hx => ?_, List.map_congr_left fun x hx => ?_⟩
    · have := hlt x (hm x (List.mem_append_left _ hx)); rw [ite_eq_right (by omega)]
    · have := hlt x (hm x (List.mem_append_right _ hx)); rw [ite_eq_right (by omega)]
  rcases T.at_inUnit hd hg hcU with ⟨hco, hat⟩ | ⟨i, hi, hu, hge, hlo, hhi, hat⟩
  · -- an own column: the root's slot
    obtain ⟨nm, hr⟩ := T.root
    obtain ⟨-, -, -, -, hcst, -, hca, hcb, -⟩ := Typed.netOfD_ok hr
    have hcst' : (st.c.nets.getD st.c.unitNet default).2.constPos = 𝒖.const := hcst
    have hlk : (st.c.nets.getD st.c.unitNet default).2.lookup = none := netOfD_lookup hr
    have hr' : Typed.netOfD (Typed.rootOf 𝒖) nm = .ok { (st.c.nets.getD st.c.unitNet default).2 with lookup := none } := by
      rw [with_lookup_none hlk]; exact hr
    obtain ⟨hEA, hEB⟩ := T.entry_at hd hul U hg hcU hin' T.unitNet (Ne.symm T.shaNet.2) (Ne.symm T.hmNet.2) hat
      ⟨nm, hr'⟩ ((Typed.rowsB (Typed.rootOf 𝒖)).getD c #[]).toList (rowsA_lt hr c).2
      (fun h32 j' => by rw [hlk]; simp only [lookupEntry, add_zero]; exact (netOfD_sparse hr h32 c j').2) j
    have hK := constPairs_count (pin := st.pin) T.disj T.slotLog T.fit (T.range T.unitNet).1 (T.range T.unitNet).2.1
      hat j
    have hy : Typed.col st.c tb g c = (st.c.rangeOf st.c.unitNet).slot g + c := hat.2.2
    by_cases hcc : c = 𝒖.const
    · -- the unit's constant
      subst hcc
      have hf := delta_const hd hul
      have hbp : p = ([one], [one]) := by
        unfold blockRow at hp; rw [hf] at hp; exact (Option.some.inj hp).symm
      subst hbp
      have hb0 : bindCols (Typed.col st.c tb g) 𝒖 𝒖.const = [] := by unfold bindCols; rw [hf]
      have hfil : [𝒖.const].filter (· ≥ 𝒖.size) = [] := by
        rw [List.filter_eq_nil_iff]
        intro x hx
        simp only [List.mem_singleton] at hx
        subst hx
        simp only [ge_iff_le, decide_eq_true_eq, not_le]
        exact hco
      have hc0 : ∀ side, crossCols (Typed.col st.c tb g) 𝒖 side 𝒖.const = [] := by
        intro side
        unfold crossCols
        rw [ite_eq_left hco, U.const_row]
        cases side <;> simp only [Bool.false_eq_true, ↓reduceIte, hfil, List.map_nil]
      have hca' : (Typed.rowsA (Typed.rootOf 𝒖)).getD 𝒖.const #[] = #[𝒖.const] := hca
      have hcb' : (Typed.rowsB (Typed.rootOf 𝒖)).getD 𝒖.const #[] = #[𝒖.const] := hcb
      have hP : 𝒖.const = (st.c.nets.getD st.c.unitNet default).2.constPos := hcst'.symm
      have hl : (#[𝒖.const].toList.map ((st.c.rangeOf st.c.unitNet).slot g + ·)) = [Typed.col st.c tb g 𝒖.const] := by
        rw [hy]; rfl
      refine ⟨hEA.trans ?_, hEB.trans ?_⟩
      · rw [hca', hl, hK, hb0, hc0 true]
        refine (const_sum hP _ _ _).trans ?_
        simp
      · rw [hcb', hl, hK, hb0, hc0 false]
        refine (const_sum hP _ _ _).trans ?_
        simp
    · -- an own row: the root's row there, and its cross entries
      have hf : 𝒖.delta.find? (·.1 == c) = none := by
        cases h : 𝒖.delta.find? (·.1 == c) with
        | none => rfl
        | some e =>
          have he1 : e.1 = c := by simpa using Array.find?_some h
          rcases delta_entry hd (Array.mem_of_find?_eq_some h) with ⟨h1, -⟩ | ⟨q, hq, ⟨h1, -⟩ | ⟨-, y, -, h1⟩⟩
          · exact absurd (he1 ▸ h1) hcc
          · have := (part_region hd hq).2.2.1; omega
          · have := (part_region hd hq).2.2.1; omega
      have hrow : 𝒖.rows[c]? = some (𝒖.rows.getD c ([], [])) := by
        obtain ⟨x, hx⟩ : ∃ x, 𝒖.rows[c]? = some x := ⟨_, Array.getElem?_eq_getElem (hlt c hc)⟩
        rw [hx, getD_of_getElem? hx]
      have hbp : p = 𝒖.rows.getD c ([], []) := by
        rw [T.blockRow_own hco hf, hrow] at hp; exact (Option.some.inj hp).symm
      subst hbp
      obtain ⟨hfa, hfb⟩ := hfree _ (fun one' => by rw [T.blockRow_own hco hf, hrow])
      have hb0 : bindCols (Typed.col st.c tb g) 𝒖 c = [] := by unfold bindCols; rw [hf]
      have hK0 : (constPairs st.c st.pin).count (Typed.col st.c tb g c, j) = 0 := by
        rw [hK, ite_eq_right (fun h => hcc (h.1.trans hcst'))]
      have hroot : ∀ x < 𝒖.size, Typed.col st.c tb g x = (st.c.rangeOf st.c.unitNet).slot g + x :=
        fun x hx => (T.at_own hg hx).2.2
      have hra : ((Typed.rowsA (Typed.rootOf 𝒖)).getD c #[]).toList =
          (𝒖.rows.getD c ([], [])).1.filter (· < 𝒖.size) := by
        rw [rowsA_getD (d := Typed.rootOf 𝒖) hco, rootOf_rows]
      have hrb : ((Typed.rowsB (Typed.rootOf 𝒖)).getD c #[]).toList =
          (𝒖.rows.getD c ([], [])).2.filter (· < 𝒖.size) := by
        rw [rowsB_getD (d := Typed.rootOf 𝒖) hco, rootOf_rows]
      have hmap : ∀ l : List ℕ, (l.filter (· < 𝒖.size)).map ((st.c.rangeOf st.c.unitNet).slot g + ·) =
          (l.filter (· < 𝒖.size)).map (Typed.col st.c tb g) := fun l =>
        List.map_congr_left fun x hx => (hroot x (by simpa using (List.mem_filter.1 hx).2)).symm
      refine ⟨hEA.trans ?_, hEB.trans ?_⟩
      · rw [hra, hK0, hb0, hfa, hmap,
          count_map_filter_split (𝒖.rows.getD c ([], [])).1 (Typed.col st.c tb g) 𝒖.size j]
        simp only [crossCols, hco, ↓reduceIte, List.count_nil, Nat.add_zero]
      · rw [hrb, hK0, hb0, hfb, hmap,
          count_map_filter_split (𝒖.rows.getD c ([], [])).2 (Typed.col st.c tb g) 𝒖.size j]
        simp only [crossCols, hco, ↓reduceIte, Bool.false_eq_true, List.count_nil, Nat.add_zero]
  · exact T.rows_part hd hul U hg hin' hcU hp hfree hi hu hge hlo hhi hat j

/-- **1e's facts at template instance `g` of a template statement** (T3): the block rows at `Typed.col c tb g c` are the
unit's block row there, for every column of its order past its inputs, and the pin's row reads the pin. -/
theorem blockFacts (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    {g : ℕ} (hg : g < st.c.g) {one : ℕ} (hone : 𝒖.rows.size ≤ one) :
    BlockFacts (T.model regs) words done 𝒖 one (Typed.col st.c tb g) := by
  have hpinv : ∀ j : Fin (2 ^ (T.model regs).kLog), (j = (T.model regs).pin) ↔ j.val = st.pin :=
    fun j => ⟨fun h => by rw [h]; rfl, fun h => Fin.ext h⟩
  refine ⟨fun c hc => T.col_lt hd hg (U.order_in c hc), fun c hc hin p hp r j hr => ?_,
    fun c hc hin p hp r j hr => ?_, fun j => ?_, fun j => ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
    rw [hr]
    exact (T.rows_at hd hul U hg hone hc hin hp j.val).1
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
    rw [hr]
    exact (T.rows_at hd hul U hg hone hc hin hp j.val).2
  · show rangesEntry st (slotEntryA st) st.pin j.val + deltaEntry st.da st.pin j.val = _
    rw [(T.placed_pin hd hul U j.val).1, if_congr (hpinv j) rfl rfl]
  · show rangesEntry st (slotEntryB st) st.pin j.val + deltaEntry st.db st.pin j.val = _
    rw [(T.placed_pin hd hul U j.val).2, if_congr (hpinv j) rfl rfl]

/-- **A template statement's placement at template instance `g`**: the unit's composed rows, placed
at `Typed.col c tb g` in the model of the accepted statement. This is a table class's `placed g`. -/
theorem placement (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    {g : ℕ} (hg : g < st.c.g) :
    Placement (T.model regs) (Rows.compose (ofBlock words done 𝒖) (ofBlock_ordered hd))
      (composePlace (T.model regs) _ (ofBlock_ordered hd) (Typed.col st.c tb g)
        ((T.blockFacts hd hul U regs hg le_rfl).realizes hd le_rfl).lt) :=
  (T.blockFacts hd hul U regs hg le_rfl).placement hd le_rfl

end TemplateLayout

end FlockSoundness
