import FlockSoundness.Protocol.ExecTyped
import FlockSoundness.Protocol.ComposePlace

/-!
# A typed template's unit, placed (1e for templates, T3)

A template statement (#277) places its unit `u = done[unit]` by parts: the root net (`Typed.rootOf u`, the unit's own
rows without their part columns) one slot per VU, each placed layout's net one slot per part, and Δ (`delta_typed`): the
flat entries, then per VU each input's copy of its message bit, each part's bound inputs, and the root rows' entries at
part columns (`crossOf`). Unit column `x` of VU `g` is at `Typed.col c tb g x`.

- **`TemplateLayout st tb words done u`**: the matrix-level facts of the accepted statement that this reads: the ranges, the
  root's and the parts' nets and slots, and Δ. A part's net (`PartNet`) is its callee's derived rows, and a generated
  read's lookup from the table the verifier holds.
- **`UnitShape done u`**: the facts about the unit's rows this reads: its constant row, its order within its own and its
  parts' columns, no Δ entry at an input, and a part's reads its callee's (`deriveChecked` gives each).
- **`TemplateLayout.blockFacts`**: with those, and #394's part lemmas from `deriveChecked … = .ok done`, the block rows
  at `col g c` are the unit's `blockRow` there, for every VU `g`: `BlockFacts` at `Typed.col c tb g`, which is a table
  class's `placed g` (`BlockFacts.placement`).
-/

open Flock FlockLevel3

namespace FlockSoundness

open Compose Flock.DeriveAll

/-! ## Counting pairs by row -/

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

end Typed

/-! ## The typed tail's rows -/

/-! ## A generated read's lookup reads what `readSide` reads -/

/-! ## A part's placed net -/

/-! ## The facts T3 reads -/

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

namespace TemplateLayout

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

/-! ## Where the typed tail writes -/

/-! ## Δ at a unit column's row -/

end TemplateLayout

/-! ## Counting a row's entries -/


/-! ## The block rows at each VU's unit -/

namespace TemplateLayout

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

end TemplateLayout

end FlockSoundness
