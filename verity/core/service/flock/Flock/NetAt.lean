/-
A net committed at a position map: the logical rows, checked as `Net.ofRows` checks them (their order
included), placed at committed positions `pos`. Column `x` of the logical net is the committed net's column `pos[x]`, its
row `pos[x]` the logical row `x` with every reference `v` read as `pos[v]`; the committed slot's other rows are empty (a
forced zero, as past `useful`). So the committed system is the logical one permuted, and the order check stays over
logical rows. No verifier path reads it yet.

`pos` is one-to-one, keeps the input rows at their columns, each output group's columns in order from a multiple of its
width, and the constant last. The inverse map is built without a check of its own and then checked both ways (`posOk`),
so `ofRowsAt_ok` and the lemmas after it read every property off `List.all`s. `Tmpl.unrollAt` takes `pos` from
`Tmpl.posMap` under a layout.
-/
import Flock.TmplLayout

namespace Flock

namespace Net

/-- The inverse of `pos` on `[0, size)`: `some x` at `pos[x]` (the last such `x`), `none` elsewhere. Unchecked: `posOk`
checks it. -/
def invOf (pos : Array Nat) (size : Nat) : Array (Option Nat) :=
  (List.range pos.size).foldl (fun inv x => inv.setIfInBounds (pos.getD x 0) (some x)) (Array.replicate size none)

/-- `pos` has `useful` columns, each placed below `size` where `inv` has it, `inv` names only columns placed where it has
them, and the first `inCols` columns stay where they are. -/
def posOk (useful inCols size : Nat) (pos : Array Nat) (inv : Array (Option Nat)) : Bool :=
  pos.size == useful && inv.size == size &&
    (List.range useful).all (fun x => decide (pos.getD x 0 < size) && inv.getD (pos.getD x 0) none == some x) &&
    (List.range size).all (fun p => match inv.getD p none with
      | some x => decide (x < useful) && pos.getD x 0 == p
      | none => true) &&
    (List.range inCols).all fun i => pos.getD i 0 == i

/-- Each output group's columns placed in order from a multiple of its width. -/
def outsOk (pos : Array Nat) (og : Array Group) : Bool :=
  og.all fun g => pos.getD (g.col * WORD) 0 % (g.words * WORD) == 0 &&
    (List.range (g.words * WORD)).all fun i => pos.getD (g.col * WORD + i) 0 == pos.getD (g.col * WORD) 0 + i

/-- The committed rows: at `p`, the logical row `inv[p]` read through `pos`, else empty. -/
def placed (pos : Array Nat) (inv : Array (Option Nat)) (rows : Array (Array Nat)) : Array (Array Nat) :=
  inv.map fun
    | some x => (rows.getD x #[]).map (pos.getD · 0)
    | none => #[]

/-- An output group at its committed word. -/
def Group.at (pos : Array Nat) (g : Group) : Group := { g with col := pos.getD (g.col * WORD) 0 / WORD }

/-- **The net at `pos`**: `ofRows`' net on the logical rows, its A and B moved to the committed positions, its useful size
`pos[constPos] + 1` and its output groups at their committed words. Refuses a `pos` that is not one-to-one onto columns
below the constant's, moves an input row, or breaks an output group. -/
def ofRowsAt (name : String) (useful constPos : Nat) (inGroups outGroups : Array Group) (ra rb : Array (Array Nat))
    (pos : Array Nat) : V Net := do
  let n ← ofRows name useful constPos inGroups outGroups ra rb
  let size := pos.getD constPos 0 + 1
  let inv := invOf pos size
  if !posOk useful (n.inWords * WORD) size pos inv then
    throw s!"{name}: the position map is not one-to-one below the constant's {size - 1}, or moves an input row"
  if !outsOk pos outGroups then throw s!"{name}: an output group is not placed in order at a multiple of its width"
  let og := outGroups.map (Group.at pos)
  let unitLog := max 7 (log2ceil size)
  let a ← Sparse.ofRows (placed pos inv ra) (2 ^ unitLog)
  let b ← Sparse.ofRows (placed pos inv rb) (2 ^ unitLog)
  if og.any (fun g => g.col + g.words > 2 ^ unitLog / WORD) then
    throw s!"{name}: an output group overruns its unit's {2 ^ unitLog / WORD} words (D11)"
  return { n with useful := size, constPos := size - 1, outGroups := og, aT := .pure a, bT := .pure b, unitLog }

/-- A net `ofRowsAt` builds is the logical net `ofRows` built, moved: its A and B are the placed rows' at its slot size,
its size is one past the constant's position, its output groups are at their committed words (inside the slot, D11),
and `pos` passed `posOk` and `outsOk`. -/
theorem ofRowsAt_ok {name : String} {useful constPos : Nat} {ig og : Array Group} {ra rb : Array (Array Nat)}
    {pos : Array Nat} {n' : Net} (h : ofRowsAt name useful constPos ig og ra rb pos = .ok n') :
    ∃ n, ofRows name useful constPos ig og ra rb = .ok n ∧
      n'.useful = pos.getD constPos 0 + 1 ∧ n'.constPos = pos.getD constPos 0 ∧
      n'.unitLog = max 7 (log2ceil (pos.getD constPos 0 + 1)) ∧
      Sparse.ofRows (placed pos (invOf pos (pos.getD constPos 0 + 1)) ra) (2 ^ n'.unitLog) = .ok n'.a ∧
      Sparse.ofRows (placed pos (invOf pos (pos.getD constPos 0 + 1)) rb) (2 ^ n'.unitLog) = .ok n'.b ∧
      n'.name = n.name ∧ n'.inGroups = n.inGroups ∧ n'.inWords = n.inWords ∧ n'.lookup = n.lookup ∧
      n'.outGroups = og.map (Group.at pos) ∧
      posOk useful (n.inWords * WORD) (pos.getD constPos 0 + 1) pos (invOf pos (pos.getD constPos 0 + 1)) = true ∧
      outsOk pos og = true ∧ ∀ g ∈ n'.outGroups.toList, g.col + g.words ≤ 2 ^ n'.unitLog / WORD := by
  unfold ofRowsAt at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · exact absurd h (by simp)
  rename_i n hn
  refine ⟨n, hn, ?_⟩
  repeat' (first | split at h | (injection h with h; subst h))
  all_goals first
    | exact absurd h (by simp)
    | (rename_i hp ho _ _ ha _ _ hb hd
       refine ⟨rfl, by simp, rfl, ha, hb, rfl, rfl, rfl, rfl, rfl, by simpa using hp, by simpa using ho, ?_⟩
       intro g hg
       refine Nat.not_lt.mp fun hc => hd (Array.any_eq_true.mpr ?_)
       obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp (Array.mem_toList_iff.mp hg)
       exact ⟨i, hi, by simpa using hc⟩)

section
variable {useful inCols size : Nat} {pos : Array Nat} {inv : Array (Option Nat)}

theorem posOk_parts (h : posOk useful inCols size pos inv = true) :
    pos.size = useful ∧ inv.size = size ∧
      (∀ x < useful, pos.getD x 0 < size ∧ inv.getD (pos.getD x 0) none = some x) ∧
      (∀ p < size, ∀ x, inv.getD p none = some x → x < useful ∧ pos.getD x 0 = p) ∧
      ∀ i < inCols, pos.getD i 0 = i := by
  simp only [posOk, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨hs, hi⟩, h1⟩, h2⟩, h3⟩ := h
  refine ⟨hs, hi, fun x hx => ?_, fun p hp x hx => ?_, h3⟩
  · exact h1 x hx
  · have := h2 p hp
    rw [hx] at this
    simpa using this

/-- Each column is placed below the committed size. -/
theorem posOk_lt (h : posOk useful inCols size pos inv = true) {x : Nat} (hx : x < useful) : pos.getD x 0 < size :=
  ((posOk_parts h).2.2.1 x hx).1

/-- `pos` is one-to-one on the logical columns. -/
theorem posOk_inj (h : posOk useful inCols size pos inv = true) {x y : Nat} (hx : x < useful) (hy : y < useful)
    (he : pos.getD x 0 = pos.getD y 0) : x = y := by
  have a := ((posOk_parts h).2.2.1 x hx).2
  have b := ((posOk_parts h).2.2.1 y hy).2
  rw [he, b] at a
  exact (Option.some.inj a).symm

/-- The input rows stay at their columns. -/
theorem posOk_inputs (h : posOk useful inCols size pos inv = true) {i : Nat} (hi : i < inCols) : pos.getD i 0 = i :=
  (posOk_parts h).2.2.2.2 i hi

theorem placed_size (rows : Array (Array Nat)) : (placed pos inv rows).size = inv.size := by simp [placed]

/-- **Committed row `pos[x]` is logical row `x` read through `pos`.** -/
theorem placed_at (h : posOk useful inCols size pos inv = true) (rows : Array (Array Nat)) {x : Nat} (hx : x < useful) :
    (placed pos inv rows).getD (pos.getD x 0) #[] = (rows.getD x #[]).map (pos.getD · 0) := by
  obtain ⟨_, hs, h1, _, _⟩ := posOk_parts h
  obtain ⟨hlt, hinv⟩ := h1 x hx
  simp only [placed, Array.getD_eq_getD_getElem?, Array.getElem?_map] at hinv ⊢
  revert hinv
  cases inv[pos[x]?.getD 0]? with
  | none => simp
  | some o =>
    cases o with
    | none => simp
    | some y =>
      intro hy
      simp only [Option.getD_some, Option.some.injEq] at hy
      subst hy
      simp

/-- **Every other committed row is empty.** -/
theorem placed_gap (h : posOk useful inCols size pos inv = true) (rows : Array (Array Nat)) {p : Nat}
    (hn : ∀ x < useful, pos.getD x 0 ≠ p) : (placed pos inv rows).getD p #[] = #[] := by
  obtain ⟨_, hs, _, h2, _⟩ := posOk_parts h
  simp only [Array.getD_eq_getD_getElem?, placed, Array.getElem?_map]
  cases hq : inv[p]? with
  | none => rfl
  | some o =>
    cases o with
    | none => rfl
    | some x =>
      have hp : p < size := hs ▸ (Array.getElem?_eq_some_iff.mp hq).1
      have := h2 p hp x (by simp [Array.getD_eq_getD_getElem?, hq])
      exact absurd this.2 (hn x this.1)

end

end Net

/-- The templated unit unrolled and committed at layout `lays`' positions (`Tmpl.posMap`). -/
def Tmpl.unrollAt (t : Tmpl.Tmpl) (lays : Array Tmpl.Lay) : V Net := do
  let acc ← t.rows
  Net.ofRowsAt t.name (t.defs.getD t.root default).cols t.constPos t.inGroups t.outGroups acc.ra acc.rb
    (Tmpl.posMap t lays)

/-- `unrollAt`'s net, or the default where it refuses. -/
def Tmpl.unitFull (t : Tmpl.Tmpl) (lays : Array Tmpl.Lay) : Net :=
  match Tmpl.unrollAt t lays with
  | .ok n => n
  | .error _ => default

/-- **The templated unit's committed net, its rows built only when read**: its size and constant (the root's layout size),
port groups (the output groups at their committed words, `posOf`) and slot from the description, its A and B
`unrollAt`'s. Wherever `unrollOk` passes it is `unrollAt`'s net (`FlockSoundness.TmplFold.unitNet_eq`). -/
def Tmpl.unitNet (t : Tmpl.Tmpl) (lays : Array Tmpl.Lay) : Net :=
  let size := (lays.getD t.root default).size
  let full : Thunk Net := Thunk.mk fun _ => Tmpl.unitFull t lays
  { name := t.name, useful := size, constPos := size - 1, inGroups := t.inGroups,
    outGroups := t.outGroups.map fun g => { g with col := Tmpl.posOf t lays (g.col * WORD) / WORD },
    inWords := (t.inGroups.back?.map fun g => g.col + g.words).getD 0,
    aT := Thunk.mk fun _ => full.get.a, bT := Thunk.mk fun _ => full.get.b, unitLog := max 7 (log2ceil size) }

deriving instance DecidableEq for Group
deriving instance DecidableEq for Sparse
deriving instance DecidableEq for LookupTable
deriving instance DecidableEq for Net

/-- A layout mode by its name: `iv` (lenient), `iv/strict`, `iv/hybrid` (`T = 0`) or `iv/hybrid/T`. -/
def Tmpl.modeOf : String → Option Tmpl.Mode
  | "iv" => some .lenient
  | "iv/strict" => some .strict
  | "iv/hybrid" => some (.hybrid 0)
  | s => if s.startsWith "iv/hybrid/" then ((s.drop 10).toString.toNat?).map .hybrid else none
end Flock
