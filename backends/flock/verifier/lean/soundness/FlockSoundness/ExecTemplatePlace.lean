import FlockSoundness.ExecTemplate
import FlockSoundness.ComposePlace
import FlockSoundness.Types.Parts

/-!
# A typed template's unit, placed (1e for templates, T3)

A template statement (#277) places its unit `u = done[unit]` by parts: the root net (`Typed.rootOf u`, the unit's own
rows without their part columns) one slot per VU, each placed layout's net one slot per part, and Δ (`delta_typed`): the
flat entries, then per VU each input's copy of its message bit, each part's bound inputs, and the root rows' entries at
part columns (`crossOf`). Unit column `x` of VU `g` is at `Typed.col c tb g x`.

- **`TemplateLayout st tb done u`**: the matrix-level facts of the accepted statement that this reads: the ranges, the
  root's and the parts' nets and slots, and Δ.
- **`UnitShape done u`**: the facts about the unit's rows that `deriveChecked` doesn't state yet: its constant and its
  constant row are its own, its order is its own and its parts' columns, no Δ entry sits at an input, and it reads no
  table.
- **`TemplateLayout.blockFacts`**: with those, and #394's part lemmas from `deriveChecked … = .ok done`, the block rows
  at `col g c` are the unit's `blockRow` there, for every VU `g`: `BlockFacts` at `Typed.col c tb g`, which is a table
  class's `placed g` (`BlockFacts.placement`).
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

/-- Unit column `x` is its own or in one of its parts. -/
def InUnit (done : Array D) (u : D) (x : ℕ) : Prop :=
  x < u.size ∨ ∃ p ∈ u.parts, p.2 ≤ x ∧ x < p.2 + (done.getD p.1 default).size

/-! ## The facts T3 reads -/

/-- **The accepted template statement's layout, as T3 reads it**, for the class's unit `u`: the ranges (disjoint, in the
block, of their nets' slots, `G · per_vu` slots each, one per net); the root (its net `Typed.rootOf u`'s, one slot per
VU); each part `i` (its net that of its callee `done[u.parts[i].1]`, not the root's or the rows', at its slot `q_i` of
that net's, no two parts at one slot); the rows' circuits (not the root); Δ (the flat entries, whose other rows are input
bits of the rows' circuits, then the typed tail); the pin (a net slot's constant, whose row reads itself); and the
block's size. -/
structure TemplateLayout (st : Stmt) (tb : TypedBlock) (done : Array D) (u : D) : Prop where
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
  parts_size : tb.parts.size = u.parts.size
  part : ∀ i (hi : i < tb.parts.size) (hu : i < u.parts.size), tb.parts[i].2.2 = u.parts[i].2 ∧
    tb.parts[i].1 < st.c.nets.size ∧ tb.parts[i].1 ≠ st.c.unitNet ∧ tb.parts[i].1 ≠ st.c.shaNet ∧
    tb.parts[i].1 ≠ st.c.hmNet ∧ tb.parts[i].2.1 < st.c.perVu.getD tb.parts[i].1 0 ∧
    ∃ nm, Typed.netOfD (done.getD u.parts[i].1 default) nm = .ok (st.c.nets.getD tb.parts[i].1 default).2
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

/-- **The unit's rows, as T3 reads them**, beyond #394's part lemmas: its constant is an own column whose row is
`[const]·[const]`; its order lists own and part columns only; no Δ entry sits at one of its inputs; and it reads no
table (a template whose parts are derived layouts). -/
structure UnitShape (done : Array D) (u : D) : Prop where
  const_row : u.rows.getD u.const ([], []) = ([u.const], [u.const])
  order_in : ∀ c ∈ order done u, InUnit done u c
  in_delta : ∀ x ∈ u.inCols, u.delta.find? (·.1 == x) = none
  reads : u.reads = #[]

theorem netOfD_lookup {d : D} {nm : String} {n : Net} (h : Typed.netOfD d nm = .ok n) : n.lookup = none :=
  (Net.ofRows_ok h).2.2.2.2.2.2.2.2.2.2.2.2.1

theorem size_le_slot {d : D} {nm : String} {n : Net} (h : Typed.netOfD d nm = .ok n) : d.size ≤ 2 ^ n.unitLog := by
  rw [(Typed.netOfD_ok h).1]
  exact le_trans (le_two_pow_log2ceil _) (Nat.pow_le_pow_right (by norm_num) (le_max_right _ _))

namespace TemplateLayout

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

theorem range (T : TemplateLayout st tb done 𝒖) {n : ℕ} (hn : n < st.c.nets.size) :
    st.c.rangeOf n ∈ st.c.ranges.toList ∧ (st.c.rangeOf n).kind = .net n ∧
      (st.c.rangeOf n).count = st.c.g * st.c.perVu.getD n 0 ∧
      (st.c.rangeOf n).slotLog = (st.c.nets.getD n default).2.unitLog := by
  obtain ⟨hm, hk, -⟩ := rangeOf_spec (T.one_range n hn)
  exact ⟨hm, hk, T.count _ hm _ hk, T.slotLog _ hm _ hk⟩

/-- The root's slot in VU `g` holds its own columns. -/
theorem at_own (T : TemplateLayout st tb done 𝒖) {g x : ℕ} (hg : g < st.c.g) (hx : x < 𝒖.size) :
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

/-- Part `i`'s slot in VU `g` holds its callee's columns, from its base. -/
theorem at_part (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb done 𝒖) {g x i : ℕ}
    (hg : g < st.c.g) (hi : i < tb.parts.size) (hu : i < 𝒖.parts.size) (hlo : 𝒖.parts[i].2 ≤ x)
    (hhi : x < 𝒖.parts[i].2 + (done.getD 𝒖.parts[i].1 default).size) :
    AtSlot st.c tb.parts[i].1 (g * st.c.perVu.getD tb.parts[i].1 0 + tb.parts[i].2.1) (x - 𝒖.parts[i].2)
      (Typed.col st.c tb g x) := by
  obtain ⟨hb, hn, -, -, -, hq, nm, hnet⟩ := T.part i hi hu
  obtain ⟨-, hc, hsl⟩ := (T.range hn).2
  have hreg := part_region hd (Array.getElem_mem hu)
  refine ⟨by rw [hc]; exact Rows.mul_add_lt hg hq, ?_, ?_⟩
  · rw [hsl]
    have := size_le_slot hnet
    omega
  · rw [Typed.col_part hi (by rw [T.own]; omega) (by rw [hb]; exact hlo) (fun k hk hik => by
      have hk' : k < 𝒖.parts.size := by rw [← T.parts_size]; exact hk
      rw [(T.part k hk hk').1]
      have := parts_apart hd hik hk'
      omega), hb]
    rfl

/-- **Where a unit column is**: an own column in the root's slot, or a part's column in the part's slot. -/
theorem at_inUnit (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb done 𝒖)
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
theorem col_lt (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb done 𝒖)
    {g x : ℕ} (hg : g < st.c.g) (hx : InUnit done 𝒖 x) : Typed.col st.c tb g x < 2 ^ st.c.kLog := by
  rcases T.at_inUnit hd hg hx with ⟨-, h⟩ | ⟨i, hi, hu, -, -, -, h⟩
  · exact h.lt T.bound (T.range T.unitNet).1
  · exact h.lt T.bound (T.range (T.part i hi hu).2.1).1

/-- **Unit columns are at distinct positions**, across VUs. -/
theorem col_inj (hd : deriveChecked types ls info words unit = .ok done) (T : TemplateLayout st tb done 𝒖)
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

end TemplateLayout

end FlockSoundness
