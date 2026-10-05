import FlockSoundness.Protocol.ExecTemplate
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

/-! ## The typed tail's rows -/

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

/-! ## A generated read's lookup reads what `readSide` reads -/

/-! ## A part's placed net -/

/-- **A part's placed net**: its callee's derived rows (`netOfD`), with no lookup and no read, or a generated read's
lookup from the table the verifier holds for its read (`words`, its `u32` words). -/
def PartNet (words : String → Option (Array ℕ)) (d : D) (net : Net) : Prop :=
  (∃ nm, Typed.netOfD d nm = .ok { net with lookup := none }) ∧
    ((net.lookup = none ∧ d.reads = #[]) ∨
      ∃ r tb, d.reads.toList = [r] ∧ net.lookup = some ⟨tb, r.n, r.lo, r.k, r.prod0, r.loTop⟩ ∧
        (words r.table).getD #[] = (Array.range (tb.size / 4)).map (fun i => rd32 tb (4 * i)) ∧
        ((words r.table).getD #[]).size = 2 ^ r.n ∧ r.lo ≤ r.n)

/-! ## The facts T3 reads -/

/-- **The accepted template statement's layout, as T3 reads it**, for the class's unit `u`: the ranges (disjoint, in the
block, of their nets' slots, `G · per_vu` slots each, one per net); the root (its net `Typed.rootOf u`'s, one slot per
VU); each part `i` (its net that of its callee `done[u.parts[i].1]`, not the root's or the rows', at its slot `q_i` of
that net's, no two parts at one slot); the rows' circuits (not the root); Δ (the flat entries, whose other rows are input
bits of the rows' circuits, then the typed tail); the pin (a net slot's constant, whose row reads itself); and the
block's size. -/
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

set_option hygiene false in
local notation "𝒖" => done.getD unit default

theorem range (T : TemplateLayout st tb words done 𝒖) {n : ℕ} (hn : n < st.c.nets.size) :
    st.c.rangeOf n ∈ st.c.ranges.toList ∧ (st.c.rangeOf n).kind = .net n ∧
      (st.c.rangeOf n).count = st.c.g * st.c.perVu.getD n 0 ∧
      (st.c.rangeOf n).slotLog = (st.c.nets.getD n default).2.unitLog := by
  obtain ⟨hm, hk, -⟩ := rangeOf_spec (T.one_range n hn)
  exact ⟨hm, hk, T.count _ hm _ hk, T.slotLog _ hm _ hk⟩

/-! ## Where the typed tail writes -/

/-! ## Δ at a unit column's row -/

end TemplateLayout

/-! ## Counting a row's entries -/


/-! ## The block rows at each VU's unit -/

namespace TemplateLayout

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

end TemplateLayout

end FlockSoundness
