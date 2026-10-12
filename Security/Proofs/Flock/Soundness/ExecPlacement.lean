import Proofs.Flock.Soundness.ExecDelta
import Proofs.Flock.Level3.Placed

/-!
# An instance's rows, placed by the statement the verifier accepted

The verifier's level-0 matrices are `placedA st` and `placedB st` (flock-verifier's `FlockLevel3.Placed`, PR #156): column
`j`'s entries come from the last range covering `j`, rows in the same slot only, plus Δ's pairs. This file proves
`Placement` for an instance's stacked rows (`Rows.stack`, `placement_stack`) from the layout facts `Layout` names:
- **the ranges:** their spans are pairwise disjoint (`HmRow.check`'s pairwise test), a net range's slots are its net's
  `2^unitLog` rows, and every net's constant and input rows fit its slot;
- **the unit range:** `vus_per_block × units_per_vu` slots of the unit net, a net `Net.ofRows` built (a text net, or a
  typed flat class's `Typed.netOfD`) whose constant row reads itself and whose input rows end before its computed rows;
- **the pin:** the first slot's constant of a range with a slot, whose net's constant row reads itself;
- **Δ:** every net slot's constant other than the pin gets `(k, k)` and `(k, pin)`, and every other pair's row is an input
  bit of a net slot (`HmRow.delta`, `delta_split`).

Then at each unit copy's computed rows the matrices are the net's rows shifted to the slot, and Δ adds nothing there; at
each copy's constant row they read the pin; and so does the pin's row. `ExecSetup.lean` derives `Layout` from
`Stmt.setupH`.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## Spans and covers -/

/-- A range's span in the block, as `HmRow.check` measures it: its slots if packed, else the next power of two of them. -/
def spanH (r : Range) : ℕ := if r.packed then r.count * 2 ^ r.slotLog else nextPow2 r.count * 2 ^ r.slotLog

/-- Column `x` is in range `r`'s span. -/
def InSpan (r : Range) (x : ℕ) : Prop := r.start ≤ x ∧ x < r.start + spanH r

/-- Two ranges' spans do not overlap: `HmRow.check`'s pairwise test does not fire on them. -/
def Apart (r r' : Range) : Prop := ¬ (r.start < r'.start + spanH r' ∧ r'.start < r.start + spanH r)

instance instSymmRangeApart : Std.Symm Apart := ⟨fun _ _ h ⟨h1, h2⟩ => h ⟨h2, h1⟩⟩

theorem not_apart_of_inSpan {r r' : Range} {x : ℕ} (h : InSpan r x) (h' : InSpan r' x) : ¬ Apart r r' :=
  fun ha => ha ⟨by have := h.1; have := h'.2; omega, by have := h'.1; have := h.2; omega⟩

/-- Two ranges of a pairwise-apart list whose spans share a column are the same range. -/
theorem eq_of_inSpan {l : List Range} (hd : l.Pairwise Apart) {r r' : Range} (hr : r ∈ l) (hr' : r' ∈ l)
    {x : ℕ} (hx : InSpan r x) (hx' : InSpan r' x) : r = r' := by
  by_contra hne
  exact not_apart_of_inSpan hx hx' (hd.forall hr hr' hne)

theorem le_nextPow2 (n : ℕ) : n ≤ nextPow2 n := by
  unfold nextPow2 log2ceil
  split
  · simp only [pow_zero]; omega
  · have := Nat.lt_log2_self (n := n - 1)
    omega

theorem count_mul_le_spanH (r : Range) : r.count * 2 ^ r.slotLog ≤ spanH r := by
  unfold spanH
  split
  · exact le_rfl
  · exact Nat.mul_le_mul_right _ (le_nextPow2 _)

theorem covers_iff (r : Range) (x : ℕ) :
    covers r x ↔ r.start ≤ x ∧ x < r.start + r.count * 2 ^ r.slotLog := by
  have hW : 0 < 2 ^ r.slotLog := Nat.two_pow_pos _
  unfold covers Range.start
  rw [Nat.le_div_iff_mul_le hW, Nat.div_lt_iff_lt_mul hW, add_mul]

theorem inSpan_of_covers {r : Range} {x : ℕ} (h : covers r x) : InSpan r x := by
  obtain ⟨h1, h2⟩ := (covers_iff r x).1 h
  exact ⟨h1, lt_of_lt_of_le h2 (Nat.add_le_add_left (count_mul_le_spanH r) _)⟩

theorem slot_div {r : Range} (s : ℕ) {off : ℕ} (hoff : off < 2 ^ r.slotLog) :
    (r.slot s + off) / 2 ^ r.slotLog = r.pos0 + s ∧ (r.slot s + off) % 2 ^ r.slotLog = off := by
  unfold Range.slot
  refine ⟨?_, ?_⟩
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ (Nat.two_pow_pos _), Nat.div_eq_of_lt hoff, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hoff]

theorem covers_slot {r : Range} {s off : ℕ} (hs : s < r.count) (hoff : off < 2 ^ r.slotLog) :
    covers r (r.slot s + off) := by
  unfold covers
  rw [(slot_div s hoff).1]
  omega

/-- Two columns of a range's slots, each below the slot size, are equal only in the same slot at the same offset. -/
theorem slot_add_inj {r : Range} {s s' off off' : ℕ} (hoff : off < 2 ^ r.slotLog) (hoff' : off' < 2 ^ r.slotLog)
    (h : r.slot s + off = r.slot s' + off') : s = s' ∧ off = off' := by
  have h1 := slot_div s hoff
  have h2 := slot_div s' hoff'
  rw [h] at h1
  exact ⟨by omega, by omega⟩

/-! ## The last range covering a column -/

theorem lastCover_foldl (c : ℕ) : ∀ (l : List Range) (acc : Option Range) (r : Range),
    l.foldl (fun acc r => if covers r c then some r else acc) acc = some r → (r ∈ l ∧ covers r c) ∨ acc = some r
  | [], _, _, h => Or.inr h
  | a :: l, acc, r, h => by
    rw [List.foldl_cons] at h
    rcases lastCover_foldl c l _ r h with h | h
    · exact Or.inl ⟨List.mem_cons_of_mem _ h.1, h.2⟩
    · by_cases ha : covers a c
      · rw [ite_eq_left ha] at h
        cases h
        exact Or.inl ⟨List.mem_cons_self, ha⟩
      · rw [ite_eq_right ha] at h
        exact Or.inr h

theorem lastCover_mem {l : List Range} {c : ℕ} {r : Range} (h : lastCover l c = some r) : r ∈ l ∧ covers r c :=
  (lastCover_foldl c l none r h).resolve_right (by simp)

theorem lastCover_foldl_ne_none (c : ℕ) : ∀ (l : List Range) (acc : Option Range),
    (acc ≠ none ∨ ∃ r ∈ l, covers r c) → l.foldl (fun acc r => if covers r c then some r else acc) acc ≠ none
  | [], _, h => h.resolve_right (by simp)
  | a :: l, acc, h => by
    rw [List.foldl_cons]
    apply lastCover_foldl_ne_none c l
    by_cases ha : covers a c
    · left; rw [ite_eq_left ha]; simp
    · rw [ite_eq_right ha]
      rcases h with h | ⟨r, hr, hc⟩
      · exact Or.inl h
      · rcases List.mem_cons.1 hr with rfl | hr
        · exact absurd hc ha
        · exact Or.inr ⟨r, hr, hc⟩

theorem lastCover_ne_none {l : List Range} {c : ℕ} {r : Range} (hr : r ∈ l) (hc : covers r c) :
    lastCover l c ≠ none :=
  lastCover_foldl_ne_none c l none (Or.inr ⟨r, hr, hc⟩)

/-- **The ranges' entry at a row of a range**: with pairwise-apart spans, the entry at row `x` in a slot of `R` is `R`'s
local entry for columns in the same slot, and zero elsewhere. -/
theorem rangesEntry_of_covers {st : Stmt} (hd : st.c.ranges.toList.Pairwise Apart) {R : Range}
    (hR : R ∈ st.c.ranges.toList) {x : ℕ} (hx : covers R x) (e : Kind → ℕ → ℕ → ZMod 2) (j : ℕ) :
    rangesEntry st e x j =
      if j / 2 ^ R.slotLog = x / 2 ^ R.slotLog then e R.kind (x % 2 ^ R.slotLog) (j % 2 ^ R.slotLog) else 0 := by
  unfold rangesEntry
  split
  · rename_i rg hrg
    obtain ⟨hmem, hcov⟩ := lastCover_mem hrg
    by_cases hq : x / 2 ^ rg.slotLog = j / 2 ^ rg.slotLog
    · have hcx : covers rg x := by unfold covers; rw [hq]; exact hcov
      obtain rfl := eq_of_inSpan hd hmem hR (inSpan_of_covers hcx) (inSpan_of_covers hx)
      rw [ite_eq_left hq, ite_eq_left hq.symm]
    · rw [ite_eq_right hq]
      by_cases hq' : j / 2 ^ R.slotLog = x / 2 ^ R.slotLog
      · have hcj : covers R j := by unfold covers; rw [hq']; exact hx
        obtain rfl := eq_of_inSpan hd hmem hR (inSpan_of_covers hcov) (inSpan_of_covers hcj)
        exact absurd hq'.symm hq
      · rw [ite_eq_right hq']
  · rename_i hnone
    by_cases hq' : j / 2 ^ R.slotLog = x / 2 ^ R.slotLog
    · have hcj : covers R j := by unfold covers; rw [hq']; exact hx
      exact absurd hnone (lastCover_ne_none hR hcj)
    · rw [ite_eq_right hq']

/-! ## Counting -/

theorem count_flatMap_zero {α β : Type} [BEq β] {l : List α} {f : α → List β} {y : β}
    (h : ∀ a ∈ l, (f a).count y = 0) : (l.flatMap f).count y = 0 := by
  induction l with
  | nil => rfl
  | cons a t ih =>
    rw [List.flatMap_cons, List.count_append, h a List.mem_cons_self,
      ih (fun b hb => h b (List.mem_cons_of_mem _ hb))]

/-- Over pairwise-apart ranges, a count over their `flatMap` is the count at the one range whose span holds `x`, when
every range whose span misses `x` contributes nothing. -/
theorem count_flatMap_apart {l : List Range} (hd : l.Pairwise Apart) {R : Range} (hR : R ∈ l) {x : ℕ}
    (hx : InSpan R x) {β : Type} [BEq β] (f : Range → List β) (y : β)
    (h0 : ∀ a ∈ l, ¬ InSpan a x → (f a).count y = 0) : (l.flatMap f).count y = (f R).count y := by
  induction l with
  | nil => cases hR
  | cons a t ih =>
    have hd' := List.pairwise_cons.1 hd
    rw [List.flatMap_cons, List.count_append]
    by_cases ha : InSpan a x
    · obtain rfl := eq_of_inSpan hd List.mem_cons_self hR ha hx
      rw [count_flatMap_zero (fun b hb => h0 b (List.mem_cons_of_mem _ hb)
        (fun hb' => not_apart_of_inSpan ha hb' (hd'.1 b hb))), Nat.add_zero]
    · rw [h0 a List.mem_cons_self ha, Nat.zero_add]
      have hRt : R ∈ t := by
        rcases List.mem_cons.1 hR with rfl | h
        · exact absurd hx ha
        · exact h
      exact ih hd'.2 hRt (fun b hb => h0 b (List.mem_cons_of_mem _ hb))

theorem count_flatMap_range {β : Type} [BEq β] (G : ℕ → List β) (y : β) {n s₀ : ℕ} (hs : s₀ < n)
    (h0 : ∀ s < n, s ≠ s₀ → (G s).count y = 0) : ((List.range n).flatMap G).count y = (G s₀).count y := by
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append, List.count_append]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
    by_cases hsn : s₀ = n
    · subst hsn
      rw [count_flatMap_zero (fun s hs' => h0 s (by simp at hs'; omega) (by simp at hs'; omega)), Nat.zero_add]
    · rw [ih (by omega) (fun s hs' hne => h0 s (by omega) hne), h0 n (by omega) (Ne.symm hsn), Nat.add_zero]

theorem deltaEntry_eq_count (d : Array (ℕ × ℕ)) (r j : ℕ) :
    deltaEntry d r j = ((d.toList.count (r, j) : ℕ) : ZMod 2) := by
  unfold deltaEntry
  rw [List.count_eq_length_filter]
  congr 2
  apply List.filter_congr
  intro p _
  by_cases h : p = (r, j) <;> simp [h]

/-- The count mod 2 of a column among a list's columns shifted into slot `B` of size `W`: the count of its offset, if it
is in that slot. -/
theorem count_map_shift {N w W B : ℕ} (hW : 0 < W) (L : List (Fin w)) (hw : w ≤ W) (f : Fin w → Fin N)
    (hf : ∀ c, (f c).val = B * W + c.val) (j : Fin N) :
    (((L.map f).count j : ℕ) : ZMod 2) =
      if j.val / W = B then (((L.map Fin.val).count (j.val % W) : ℕ) : ZMod 2) else 0 := by
  have h1 : (L.map f).count j = ((L.map Fin.val).map (B * W + ·)).count j.val := by
    rw [← List.count_map_of_injective _ Fin.val Fin.val_injective, List.map_map, List.map_map]
    congr 1
    exact List.map_congr_left fun c _ => hf c
  rw [h1]
  by_cases hq : j.val / W = B
  · rw [ite_eq_left hq]
    have hj : j.val = B * W + j.val % W := by
      have := Nat.div_add_mod j.val W
      rw [hq, Nat.mul_comm] at this
      omega
    rw [hj, List.count_map_of_injective _ _ (fun a b h => by simpa using h), Nat.add_comm,
      Nat.add_mul_mod_self_right, Nat.mod_mod]
  · rw [ite_eq_right hq, List.count_eq_zero.2]
    · simp
    · intro hm
      obtain ⟨y, hy, hjy⟩ := List.mem_map.1 hm
      obtain ⟨c, -, rfl⟩ := List.mem_map.1 hy
      have hc : c.val < W := lt_of_lt_of_le c.isLt hw
      apply hq
      rw [← hjy, Nat.add_comm, Nat.add_mul_div_right _ _ hW, Nat.div_eq_of_lt hc, Nat.zero_add]

/-! ## `Sparse`'s entries are its rows' counts -/

theorem ofRows_rows {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m) :
    m.rows = rows.size := by
  unfold Sparse.ofRows at h
  simp only at h
  obtain ⟨st, -, h⟩ := except_bind_ok h
  cases h
  rfl

/-- **A built matrix's entry** (`sparseEntry`, the fold's terms) is the count, mod 2, of the column in its row. -/
theorem sparseEntry_ofRows {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m)
    (i i' : ℕ) : sparseEntry m i i' = (((m.row i).toList.count i' : ℕ) : ZMod 2) := by
  obtain ⟨-, hs, hl, -⟩ := ofRows_spec h
  have hr := ofRows_rows h
  have hls : m.len.size = rows.size := by rw [← Array.length_toList, hl]; simp
  have hss : m.start.size = rows.size := by rw [← Array.length_toList, hs, rowStarts_length]; simp
  unfold sparseEntry Sparse.row Sparse.col
  by_cases hi : i < m.rows
  · rw [ite_eq_left hi, Array.toList_map, Array.toList_range, List.count, List.countP_map,
      List.countP_eq_length_filter]
    have h1 : m.len[i]! = m.len.getD i 0 := by
      simp [Array.getD_eq_getD_getElem?, show i < m.len.size by omega]
    have h2 : m.start[i]! = m.start.getD i 0 := by
      simp [Array.getD_eq_getD_getElem?, show i < m.start.size by omega]
    rw [h1, h2]
    congr 2
  · rw [ite_eq_right hi]
    have h1 : m.len.getD i 0 = 0 := by
      simp [Array.getD_eq_getD_getElem?, show ¬ i < m.len.size by omega]
    simp [h1]

theorem sparseEntry_built_a {net : Net} (h : Built net) (i i' : ℕ) :
    sparseEntry net.a i i' = (((net.a.row i).toList.count i' : ℕ) : ZMod 2) :=
  let ⟨_, _, _, ha, _⟩ := built_rowOrder h
  sparseEntry_ofRows ha i i'

theorem sparseEntry_built_b {net : Net} (h : Built net) (i i' : ℕ) :
    sparseEntry net.b i i' = (((net.b.row i).toList.count i' : ℕ) : ZMod 2) :=
  let ⟨_, _, _, _, hb⟩ := built_rowOrder h
  sparseEntry_ofRows hb i i'

/-! ## Δ's first loop, and input bits -/

/-- A net's constant row reads itself, in `A` and in `B` (a lookup slot's table entries included), and its input rows
end before it. -/
def ConstRow (net : Net) : Prop :=
  net.inWords * WORD ≤ net.constPos ∧
    (∀ j, sparseEntry net.a net.constPos j = if j = net.constPos then 1 else 0) ∧
    ∀ j, sparseEntry net.b net.constPos j + lookupEntry net.lookup net.constPos j = if j = net.constPos then 1 else 0

/-- The pairs `HmRow.delta`'s first loop writes for range `r`: each slot's constant `k` of a net range, unless it is the
pin, gets `(k, k)` (cancelling the constant row's own entry) and `(k, pin)`. -/
def slotPairs (c : Circuit) (pin : ℕ) (r : Range) : List (ℕ × ℕ) :=
  match r.kind with
  | .net i => (List.range r.count).flatMap fun s =>
      if (r.slot s + (c.nets.getD i default).2.constPos != pin) = true then
        [(r.slot s + (c.nets.getD i default).2.constPos, r.slot s + (c.nets.getD i default).2.constPos),
          (r.slot s + (c.nets.getD i default).2.constPos, pin)]
      else []
  | _ => []

/-- The pairs of `HmRow.delta`'s first loop, range by range. -/
def constPairs (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) := c.ranges.toList.flatMap (slotPairs c pin)

theorem slotPairs_fst {c : Circuit} {pin : ℕ} {r : Range} {p : ℕ × ℕ} (hp : p ∈ slotPairs c pin r) :
    ∃ i, r.kind = .net i ∧ ∃ s < r.count, p.1 = r.slot s + (c.nets.getD i default).2.constPos ∧ p.1 ≠ pin := by
  unfold slotPairs at hp
  split at hp
  · rename_i i hk
    obtain ⟨s, hs, hp⟩ := List.mem_flatMap.1 hp
    split at hp
    · rename_i hne
      have hne' : r.slot s + (c.nets.getD i default).2.constPos ≠ pin := by simpa using hne
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
      rcases hp with rfl | rfl
      · exact ⟨i, hk, s, List.mem_range.1 hs, rfl, hne'⟩
      · exact ⟨i, hk, s, List.mem_range.1 hs, rfl, hne'⟩
    · cases hp
  · cases hp

theorem constPairs_fst {c : Circuit} {pin : ℕ} {p : ℕ × ℕ} (hp : p ∈ constPairs c pin) :
    ∃ r ∈ c.ranges.toList, ∃ i, r.kind = .net i ∧ ∃ s < r.count,
      p.1 = r.slot s + (c.nets.getD i default).2.constPos ∧ p.1 ≠ pin := by
  obtain ⟨r, hr, hp⟩ := List.mem_flatMap.1 hp
  exact ⟨r, hr, slotPairs_fst hp⟩

/-- Column `x` is an input bit of a net slot: below the net's input rows, in a slot of the net's range. -/
def InBit (c : Circuit) (x : ℕ) : Prop :=
  ∃ n s off, c.rangeOf n ∈ c.ranges.toList ∧ (c.rangeOf n).kind = .net n ∧ s < (c.rangeOf n).count ∧
    off < (c.nets.getD n default).2.inWords * WORD ∧ x = (c.rangeOf n).slot s + off

/-! ## The layout facts -/

/-- **The layout facts** the placement reads off an accepted statement (`ExecSetup.lean` derives them from
`Stmt.setupH`). -/
structure Layout (st : Stmt) : Prop where
  /-- `HmRow.check`'s pairwise test: no two ranges' spans overlap -/
  disj : st.c.ranges.toList.Pairwise Apart
  /-- every range inside the block -/
  bound : ∀ r ∈ st.c.ranges.toList, r.start + spanH r ≤ 2 ^ st.c.kLog
  /-- a net range's slots are its net's rows -/
  slotLog : ∀ r ∈ st.c.ranges.toList, ∀ i, r.kind = .net i → r.slotLog = (st.c.nets.getD i default).2.unitLog
  /-- every net's constant and input rows fit its slot -/
  fit : ∀ i, (st.c.nets.getD i default).2.constPos < 2 ^ (st.c.nets.getD i default).2.unitLog ∧
    (st.c.nets.getD i default).2.inWords * WORD ≤ 2 ^ (st.c.nets.getD i default).2.unitLog
  unit_mem : st.c.unitRange ∈ st.c.ranges.toList
  unit_kind : st.c.unitRange.kind = .net st.c.unitNet
  unit_count : st.c.unitRange.count = st.c.g * st.c.upv
  /-- the unit is a net `Net.ofRows` built (not a lookup slot): a text net, or a typed flat class's derived net -/
  unit_built : Built st.c.unit
  unit_lookup : st.c.unit.lookup = none
  unit_in : st.c.unit.inWords * WORD ≤ st.c.unit.constPos
  /-- the unit's constant row reads itself (its rows fit `rd32`'s words when its slot fits the block) -/
  unit_const : st.c.unit.unitLog ≤ 32 → ConstRow st.c.unit
  /-- the pin is the first slot's constant of a net range with a slot -/
  pin : ∃ rp ∈ st.c.ranges.toList, ∃ ip, rp.kind = .net ip ∧ 0 < rp.count ∧
    st.pin = rp.start + (st.c.nets.getD ip default).2.constPos ∧ ConstRow (st.c.nets.getD ip default).2
  delta_b : st.db = st.da
  /-- Δ: its first loop's pairs, then only input bits of net slots -/
  delta : ∃ rest, st.da.toList = constPairs st.c st.pin ++ rest ∧ ∀ p ∈ rest, InBit st.c p.1
  kLog : st.c.kLog ≤ 32
  kLog6 : 6 ≤ st.c.kLog
  m : st.m = st.c.kLog + st.nbl

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

theorem unit_slotLog : st.c.unitRange.slotLog = st.c.unit.unitLog := L.slotLog _ L.unit_mem _ L.unit_kind

theorem const_covers {r : Range} (hr : r ∈ st.c.ranges.toList) {i : ℕ} (hk : r.kind = .net i) {s : ℕ}
    (hs : s < r.count) : covers r (r.slot s + (st.c.nets.getD i default).2.constPos) :=
  covers_slot hs (by rw [L.slotLog r hr i hk]; exact (L.fit i).1)

/-- A column of a net range's slot at or past its net's input rows is no input bit. -/
theorem not_inBit {R : Range} (hR : R ∈ st.c.ranges.toList) {n₀ : ℕ} (hk : R.kind = .net n₀) {s₀ off : ℕ}
    (hs : s₀ < R.count) (hoff : off < 2 ^ R.slotLog) (hge : (st.c.nets.getD n₀ default).2.inWords * WORD ≤ off) :
    ¬ InBit st.c (R.slot s₀ + off) := by
  rintro ⟨n, s, off', hmem, hkind, hs', hoff', heq⟩
  have hoffW : off' < 2 ^ (st.c.rangeOf n).slotLog := by
    rw [L.slotLog _ hmem n hkind]; exact lt_of_lt_of_le hoff' (L.fit n).2
  have hc1 : covers (st.c.rangeOf n) (R.slot s₀ + off) := heq ▸ covers_slot hs' hoffW
  have hR' := eq_of_inSpan L.disj hmem hR (inSpan_of_covers hc1) (inSpan_of_covers (covers_slot hs hoff))
  rw [hR'] at heq hkind hoffW
  have hn : n = n₀ := Kind.net.inj (hkind.symm.trans hk)
  subst hn
  obtain ⟨-, rfl⟩ := slot_add_inj hoff hoffW heq
  omega

/-- No pair of Δ's first loop is at a column of a net range's slot below its net's constant. -/
theorem constPairs_not_below {R : Range} (hR : R ∈ st.c.ranges.toList) {n₀ : ℕ} (hk : R.kind = .net n₀) {s₀ off : ℕ}
    (hs : s₀ < R.count) (hoff : off < (st.c.nets.getD n₀ default).2.constPos) (j : ℕ) :
    (R.slot s₀ + off, j) ∉ constPairs st.c st.pin := by
  intro hm
  obtain ⟨r, hr, i, hki, s, hs', heq, -⟩ := constPairs_fst hm
  simp only at heq
  have hoffW : off < 2 ^ R.slotLog := by
    rw [L.slotLog R hR n₀ hk]; exact lt_trans hoff (L.fit n₀).1
  have hin := inSpan_of_covers (L.const_covers hr hki hs')
  rw [← heq] at hin
  obtain rfl := eq_of_inSpan L.disj hr hR hin (inSpan_of_covers (covers_slot hs hoffW))
  have hn : i = n₀ := Kind.net.inj (hki.symm.trans hk)
  subst hn
  obtain ⟨-, h⟩ := slot_add_inj hoffW (by rw [L.slotLog r hr i hki]; exact (L.fit i).1) heq
  omega

end Layout

theorem start_eq_slot (r : Range) : r.start = r.slot 0 := by simp [Range.start, Range.slot]

/-- Inside slot `B` of size `W`, a local row reading only its own offset `c` reads column `B·W + c`. -/
theorem ite_slot_eq {W B c j : ℕ} (hW : 0 < W) (hc : c < W) {v : ℕ → ZMod 2}
    (hv : ∀ j', v j' = if j' = c then 1 else 0) :
    (if j / W = B then v (j % W) else 0) = if j = B * W + c then 1 else 0 := by
  have hdm := Nat.div_add_mod j W
  by_cases h1 : j / W = B
  · rw [ite_eq_left h1, hv]
    rw [h1, Nat.mul_comm] at hdm
    by_cases h2 : j % W = c
    · rw [ite_eq_left h2, ite_eq_left (by omega)]
    · rw [ite_eq_right h2, ite_eq_right (by omega)]
  · rw [ite_eq_right h1, ite_eq_right]
    intro hj
    apply h1
    rw [hj, Nat.add_comm, Nat.add_mul_div_right _ _ hW, Nat.div_eq_of_lt hc, Nat.zero_add]

theorem ite_fin_val {N : ℕ} (j : Fin N) {p : ℕ} (hp : p < N) :
    (if j.val = p then (1 : ZMod 2) else 0) = if j = ⟨p, hp⟩ then 1 else 0 := by
  by_cases h : j.val = p <;> simp [h, Fin.ext_iff]

/-- A slot constant's row: its own entry, and Δ's `(k, k)` and `(k, pin)` unless it is the pin, read the pin. -/
theorem const_row_sum (a x p : ℕ) :
    ((if a = x then (1 : ZMod 2) else 0) +
        if x ≠ p then (if a = x then 1 else 0) + (if a = p then 1 else 0) else 0) = if a = p then 1 else 0 := by
  by_cases h3 : x = p
  · subst h3; simp
  · rw [ite_eq_left h3]
    by_cases h1 : a = x
    · subst h1
      rw [ite_eq_left rfl, ite_eq_right h3]
      decide
    · rw [ite_eq_right h1]
      simp

/-- The unit net's rows, as the audit's `Rows` (`Rows.ofNet`). -/
noncomputable def unitRows (st : Stmt) (h : Built st.c.unit) : Rows := Rows.ofNet h

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

theorem unit_const_lt : st.c.unit.constPos < 2 ^ st.c.unit.unitLog := (L.fit st.c.unitNet).1

theorem unitRows_nIn : (unitRows st L.unit_built).nIn = st.c.unit.inWords * WORD :=
  Nat.min_eq_left L.unit_in

theorem unitRows_nIn_nComp :
    (unitRows st L.unit_built).nIn + (unitRows st L.unit_built).nComp = st.c.unit.constPos := by
  show min _ _ + (_ - min _ _) = _
  have := Nat.min_le_right (st.c.unit.inWords * WORD) st.c.unit.constPos
  omega

theorem unitRows_w : (unitRows st L.unit_built).w ≤ 2 ^ st.c.unit.unitLog := by
  show _ + _ + 1 ≤ _
  rw [L.unitRows_nIn_nComp]
  exact L.unit_const_lt

theorem unitRows_one : (unitRows st L.unit_built).one.val = st.c.unit.constPos := Rows.ofNet_one _

theorem unitRows_a (hu : st.c.unit.unitLog ≤ 32) (i : Fin (unitRows st L.unit_built).nComp) :
    ((unitRows st L.unit_built).a i).map Fin.val =
      (st.c.unit.a.row ((unitRows st L.unit_built).comp i).val).toList :=
  Rows.ofNet_a _ hu i

theorem unitRows_b (hu : st.c.unit.unitLog ≤ 32) (i : Fin (unitRows st L.unit_built).nComp) :
    ((unitRows st L.unit_built).b i).map Fin.val =
      (st.c.unit.b.row ((unitRows st L.unit_built).comp i).val).toList :=
  Rows.ofNet_b _ hu i

theorem slot_lt {g u : ℕ} (hg : g < st.c.g) (hu : u < st.c.upv) : g * st.c.upv + u < st.c.unitRange.count := by
  rw [L.unit_count]
  exact Rows.mul_add_lt hg hu

theorem unit_slot_lt {s₀ off : ℕ} (hs : s₀ < st.c.unitRange.count) (hoff : off < 2 ^ st.c.unit.unitLog) :
    st.c.unitRange.slot s₀ + off < 2 ^ st.c.kLog := by
  rw [← L.unit_slotLog] at hoff
  have := (covers_iff _ _).1 (covers_slot hs hoff)
  have := count_mul_le_spanH st.c.unitRange
  have := L.bound _ L.unit_mem
  omega

theorem unitLog_le {s₀ : ℕ} (hs : s₀ < st.c.unitRange.count) : st.c.unit.unitLog ≤ 32 := by
  have hp := Nat.two_pow_pos st.c.unit.unitLog
  have h := L.unit_slot_lt hs (off := 2 ^ st.c.unit.unitLog - 1) (by omega)
  have h2 : 2 ^ st.c.unit.unitLog ≤ 2 ^ st.c.kLog := by omega
  have := (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).1 h2
  have := L.kLog
  omega

theorem pin_lt : st.pin < 2 ^ st.c.kLog := by
  obtain ⟨rp, hrp, ip, hk, hc, hpin, -⟩ := L.pin
  have := (covers_iff _ _).1 (L.const_covers hrp hk hc)
  have := count_mul_le_spanH rp
  have := L.bound rp hrp
  rw [start_eq_slot] at hpin
  omega

theorem hb {g : ℕ} (hg : g < st.c.g) :
    ∀ u < st.c.upv, st.c.unitRange.slot (g * st.c.upv + u) +
      ((unitRows st L.unit_built).nIn + (unitRows st L.unit_built).nComp) < 2 ^ st.c.kLog := by
  intro u hu
  rw [L.unitRows_nIn_nComp]
  exact L.unit_slot_lt (L.slot_lt hg hu) L.unit_const_lt

/-- The ranges' entry at a row of a unit slot: the unit net's local entry, in the same slot. -/
theorem rangesEntry_unit {s₀ off : ℕ} (hs : s₀ < st.c.unitRange.count) (hoff : off < 2 ^ st.c.unit.unitLog)
    (e : Kind → ℕ → ℕ → ZMod 2) (j : ℕ) :
    rangesEntry st e (st.c.unitRange.slot s₀ + off) j =
      if j / 2 ^ st.c.unit.unitLog = st.c.unitRange.pos0 + s₀ then
        e (.net st.c.unitNet) off (j % 2 ^ st.c.unit.unitLog) else 0 := by
  have hoff' : off < 2 ^ st.c.unitRange.slotLog := by rw [L.unit_slotLog]; exact hoff
  rw [rangesEntry_of_covers L.disj L.unit_mem (covers_slot hs hoff') e j, (slot_div s₀ hoff').1,
    (slot_div s₀ hoff').2, L.unit_kind, L.unit_slotLog]

/-- **Δ adds nothing at a unit copy's computed rows.** -/
theorem delta_unit_comp {s₀ off : ℕ} (hs : s₀ < st.c.unitRange.count) (hlo : st.c.unit.inWords * WORD ≤ off)
    (hhi : off < st.c.unit.constPos) (j : ℕ) : deltaEntry st.da (st.c.unitRange.slot s₀ + off) j = 0 := by
  obtain ⟨rest, hsplit, hrest⟩ := L.delta
  have hoffW : off < 2 ^ st.c.unitRange.slotLog := by rw [L.unit_slotLog]; exact lt_trans hhi L.unit_const_lt
  rw [deltaEntry_eq_count, hsplit, List.count_append,
    List.count_eq_zero.2 (L.constPairs_not_below L.unit_mem L.unit_kind hs hhi j),
    List.count_eq_zero.2 (fun hm => L.not_inBit L.unit_mem L.unit_kind hs hoffW hlo (hrest _ hm))]
  simp

/-- **Δ adds nothing at the pin's row.** -/
theorem delta_pin (j : ℕ) : deltaEntry st.da st.pin j = 0 := by
  obtain ⟨rest, hsplit, hrest⟩ := L.delta
  obtain ⟨rp, hrp, ip, hk, hc, hpin, hcr⟩ := L.pin
  have hoffW : (st.c.nets.getD ip default).2.constPos < 2 ^ rp.slotLog := by
    rw [L.slotLog rp hrp ip hk]; exact (L.fit ip).1
  rw [deltaEntry_eq_count, hsplit, List.count_append,
    List.count_eq_zero.2 (fun hm => (constPairs_fst hm).elim fun _ ⟨_, _, _, _, _, _, hne⟩ => hne rfl),
    List.count_eq_zero.2 (fun hm => L.not_inBit hrp hk hc hoffW hcr.1
      (by rw [← start_eq_slot, ← hpin]; exact hrest _ hm))]
  simp

/-- **Δ at a unit copy's constant row**: `(k, k)` and `(k, pin)`, unless the constant is the pin. -/
theorem delta_unit_const {s₀ : ℕ} (hs : s₀ < st.c.unitRange.count) (j : ℕ) :
    deltaEntry st.da (st.c.unitRange.slot s₀ + st.c.unit.constPos) j =
      if st.c.unitRange.slot s₀ + st.c.unit.constPos ≠ st.pin then
        (if j = st.c.unitRange.slot s₀ + st.c.unit.constPos then 1 else 0) + (if j = st.pin then 1 else 0)
      else 0 := by
  obtain ⟨rest, hsplit, hrest⟩ := L.delta
  have hoffW : st.c.unit.constPos < 2 ^ st.c.unitRange.slotLog := by rw [L.unit_slotLog]; exact L.unit_const_lt
  have hx := inSpan_of_covers (covers_slot hs hoffW)
  rw [deltaEntry_eq_count, hsplit, List.count_append,
    List.count_eq_zero.2 (fun hm => L.not_inBit L.unit_mem L.unit_kind hs hoffW L.unit_in (hrest _ hm)),
    Nat.add_zero, constPairs, count_flatMap_apart L.disj L.unit_mem hx _ _ (fun a ha hax => by
      refine List.count_eq_zero.2 fun hm => hax ?_
      obtain ⟨i, hki, s, hs', heq, -⟩ := slotPairs_fst hm
      simp only at heq
      rw [heq]
      exact inSpan_of_covers (L.const_covers ha hki hs'))]
  have hcu : (st.c.nets.getD st.c.unitNet default).2.constPos = st.c.unit.constPos := rfl
  simp only [slotPairs, L.unit_kind]
  rw [hcu, count_flatMap_range _ _ hs (fun s hs' hne => by
    refine List.count_eq_zero.2 fun hm => hne ?_
    split at hm
    · simp only [List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq] at hm
      exact (slot_add_inj (off := st.c.unit.constPos) hoffW hoffW
        (by rcases hm with ⟨h, -⟩ | ⟨h, -⟩ <;> exact h)).1.symm
    · cases hm)]
  by_cases hxp : st.c.unitRange.slot s₀ + st.c.unit.constPos = st.pin
  · have : (st.c.unitRange.slot s₀ + st.c.unit.constPos != st.pin) = false := by simp [hxp]
    rw [ite_eq_right (by simp [this]), ite_eq_right (by simpa using hxp)]
    simp
  · rw [ite_eq_left (bne_iff_ne.2 hxp), ite_eq_left hxp]
    by_cases h1 : j = st.c.unitRange.slot s₀ + st.c.unit.constPos
    · subst h1
      simp [hxp]
    · by_cases h2 : j = st.pin
      · subst h2
        simp [h1, Ne.symm h1]
      · simp [h1, h2, Ne.symm h1, Ne.symm h2]

end Layout

/-- **The statement the verifier accepted**, as the model's block R1CS: the verifier's `m`, `k_log` and pin, the level-0
matrices it folds (`placedA`, `placedB`), and public regions `regs`. -/
def Layout.model {st : Stmt} (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) :
    Model.Statement where
  m := st.m
  kLog := st.c.kLog
  hk := by rw [L.m]; omega
  hk6 := L.kLog6
  pin := ⟨st.pin, L.pin_lt⟩
  A₀ := placedA st
  B₀ := placedB st
  regions := regs

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

/-- **An instance's rows, placed by the accepted statement.** VU `g`'s `upv` unit slots, stacked (`Rows.stack`), sit at
`ur.slot (g·upv + u)` for copy `u`, with the shared constant at the pin (`stackPlace`), and the verifier's level-0
matrices place them there. -/
theorem placement (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {g : ℕ} (hg : g < st.c.g) :
    Placement (L.model regs) ((unitRows st L.unit_built).stack st.c.upv)
      (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg)) := by
  have hW : 0 < 2 ^ st.c.unit.unitLog := Nat.two_pow_pos _
  have hU := L.unit_built
  have hshift : ∀ u (hu : u < st.c.upv) c,
      (posIn (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) hu c).val =
        (st.c.unitRange.pos0 + (g * st.c.upv + u)) * 2 ^ st.c.unit.unitLog + c.val := by
    intro u hu c
    show st.c.unitRange.slot _ + c.val = _
    unfold Range.slot
    rw [L.unit_slotLog]
  have hcomp : ∀ i : Fin (unitRows st L.unit_built).nComp,
      st.c.unit.inWords * WORD ≤ ((unitRows st L.unit_built).comp i).val ∧
        ((unitRows st L.unit_built).comp i).val < st.c.unit.constPos := by
    intro i
    have := L.unitRows_nIn_nComp
    have := L.unitRows_nIn
    have := i.isLt
    show _ ≤ (unitRows st L.unit_built).nIn + i.val ∧ (unitRows st L.unit_built).nIn + i.val < _
    omega
  have hlk : (st.c.nets.getD st.c.unitNet default).2.lookup = none := L.unit_lookup
  have hS : (L.model regs).pin = (⟨st.pin, L.pin_lt⟩ : Fin (2 ^ (L.model regs).kLog)) := rfl
  refine placement_stack _ _ _ _ _ ?_ ?_ ?_ ?_ ?_ ?_
  · intro u hu i j
    have hs := L.slot_lt hg hu
    show rangesEntry st (slotEntryA st) (st.c.unitRange.slot (g * st.c.upv + u) +
        ((unitRows st L.unit_built).comp i).val) j.val +
      deltaEntry st.da (st.c.unitRange.slot (g * st.c.upv + u) + ((unitRows st L.unit_built).comp i).val) j.val = _
    rw [L.rangesEntry_unit hs (lt_trans (hcomp i).2 L.unit_const_lt),
      L.delta_unit_comp hs (hcomp i).1 (hcomp i).2, add_zero,
      count_map_shift hW _ L.unitRows_w _ (hshift u hu) j, L.unitRows_a (L.unitLog_le hs) i]
    exact if_congr Iff.rfl (sparseEntry_built_a hU _ _) rfl
  · intro u hu i j
    have hs := L.slot_lt hg hu
    show rangesEntry st (slotEntryB st) (st.c.unitRange.slot (g * st.c.upv + u) +
        ((unitRows st L.unit_built).comp i).val) j.val +
      deltaEntry st.db (st.c.unitRange.slot (g * st.c.upv + u) + ((unitRows st L.unit_built).comp i).val) j.val = _
    rw [L.delta_b, L.rangesEntry_unit hs (lt_trans (hcomp i).2 L.unit_const_lt),
      L.delta_unit_comp hs (hcomp i).1 (hcomp i).2, add_zero,
      count_map_shift hW _ L.unitRows_w _ (hshift u hu) j, L.unitRows_b (L.unitLog_le hs) i]
    refine if_congr Iff.rfl ?_ rfl
    show sparseEntry _ _ _ + lookupEntry (st.c.nets.getD st.c.unitNet default).2.lookup _ _ = _
    rw [hlk]
    show sparseEntry st.c.unit.b _ _ + 0 = _
    rw [add_zero, sparseEntry_built_b hU]
  · intro u hu j
    have hs := L.slot_lt hg hu
    show rangesEntry st (slotEntryA st) (st.c.unitRange.slot (g * st.c.upv + u) +
        (unitRows st L.unit_built).one.val) j.val +
      deltaEntry st.da (st.c.unitRange.slot (g * st.c.upv + u) + (unitRows st L.unit_built).one.val) j.val = _
    rw [L.unitRows_one, L.rangesEntry_unit hs L.unit_const_lt, L.delta_unit_const hs,
      ite_slot_eq (v := slotEntryA st (.net st.c.unitNet) st.c.unit.constPos) hW L.unit_const_lt
        (L.unit_const (L.unitLog_le hs)).2.1]
    have hx : (st.c.unitRange.pos0 + (g * st.c.upv + u)) * 2 ^ st.c.unit.unitLog + st.c.unit.constPos =
        st.c.unitRange.slot (g * st.c.upv + u) + st.c.unit.constPos := by
      unfold Range.slot; rw [L.unit_slotLog]
    rw [hx, hS, ← ite_fin_val j L.pin_lt]
    exact const_row_sum _ _ _
  · intro u hu j
    have hs := L.slot_lt hg hu
    show rangesEntry st (slotEntryB st) (st.c.unitRange.slot (g * st.c.upv + u) +
        (unitRows st L.unit_built).one.val) j.val +
      deltaEntry st.db (st.c.unitRange.slot (g * st.c.upv + u) + (unitRows st L.unit_built).one.val) j.val = _
    rw [L.delta_b, L.unitRows_one, L.rangesEntry_unit hs L.unit_const_lt, L.delta_unit_const hs,
      ite_slot_eq (v := slotEntryB st (.net st.c.unitNet) st.c.unit.constPos) hW L.unit_const_lt
        (L.unit_const (L.unitLog_le hs)).2.2]
    have hx : (st.c.unitRange.pos0 + (g * st.c.upv + u)) * 2 ^ st.c.unit.unitLog + st.c.unit.constPos =
        st.c.unitRange.slot (g * st.c.upv + u) + st.c.unit.constPos := by
      unfold Range.slot; rw [L.unit_slotLog]
    rw [hx, hS, ← ite_fin_val j L.pin_lt]
    exact const_row_sum _ _ _
  · intro j
    show rangesEntry st (slotEntryA st) st.pin j.val + deltaEntry st.da st.pin j.val = _
    rw [L.delta_pin, add_zero, hS, ← ite_fin_val j L.pin_lt]
    obtain ⟨rp, hrp, ip, hk, hc, hpin, hcr⟩ := L.pin
    have hoffW : (st.c.nets.getD ip default).2.constPos < 2 ^ rp.slotLog := by
      rw [L.slotLog rp hrp ip hk]; exact (L.fit ip).1
    have hp : st.pin = rp.slot 0 + (st.c.nets.getD ip default).2.constPos := by rw [hpin, start_eq_slot]
    rw [hp, rangesEntry_of_covers L.disj hrp (L.const_covers hrp hk hc) _ _, (slot_div 0 hoffW).1,
      (slot_div 0 hoffW).2, hk,
      ite_slot_eq (v := slotEntryA st (.net ip) (st.c.nets.getD ip default).2.constPos) (Nat.two_pow_pos _) hoffW hcr.2.1]
    rfl
  · intro j
    show rangesEntry st (slotEntryB st) st.pin j.val + deltaEntry st.db st.pin j.val = _
    rw [L.delta_b, L.delta_pin, add_zero, hS, ← ite_fin_val j L.pin_lt]
    obtain ⟨rp, hrp, ip, hk, hc, hpin, hcr⟩ := L.pin
    have hoffW : (st.c.nets.getD ip default).2.constPos < 2 ^ rp.slotLog := by
      rw [L.slotLog rp hrp ip hk]; exact (L.fit ip).1
    have hp : st.pin = rp.slot 0 + (st.c.nets.getD ip default).2.constPos := by rw [hpin, start_eq_slot]
    rw [hp, rangesEntry_of_covers L.disj hrp (L.const_covers hrp hk hc) _ _, (slot_div 0 hoffW).1,
      (slot_div 0 hoffW).2, hk,
      ite_slot_eq (v := slotEntryB st (.net ip) (st.c.nets.getD ip default).2.constPos) (Nat.two_pow_pos _) hoffW hcr.2.2]
    rfl

/-- **The unit's inputs on one gate carry one value** at VU `g`'s stacked unit slots in block `o`, in every witness
satisfying the accepted statement: the inputs that `inst` wires to one gate copy one source's position, or forced-zero
rows (`UnitPlace.aliased`). -/
def Aliased (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Audit.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n} (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv))
    (o : Fin (2 ^ (L.model regs).nbl)) {g : ℕ} (hg : g < st.c.g) : Prop :=
  ∀ c c', inst.wire c = inst.wire c' → c.val < ((unitRows st L.unit_built).stack st.c.upv).nIn →
    c'.val < ((unitRows st L.unit_built).stack st.c.upv).nIn → ∀ z : Model.Witness (L.model regs).m,
      (L.model regs).Satisfies z →
      (L.model regs).block z o (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) c) =
      (L.model regs).block z o (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) c')

/-- **A drawn unit's place in the accepted statement** (#145's `UnitPlace`, which `loweringSoundB_of_place` and #171's
`_placed` forms take): VU `g`'s stacked unit slots in block `o`, given the unit's instance in the audit circuit
(`inst`, the circuit side), whose inputs on one gate carry one value there (`hal`). -/
noncomputable def unitPlace (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Audit.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n} (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv))
    (o : Fin (2 ^ (L.model regs).nbl)) {g : ℕ} (hg : g < st.c.g) (hal : L.Aliased regs inst o hg) :
    UnitPlace P (L.model regs) u where
  R := (unitRows st L.unit_built).stack st.c.upv
  inst := inst
  o := o
  col := stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
    (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg)
  placed := L.placement regs hg
  aliased := hal

end Layout

end FlockSoundness
