import FlockSoundness.ExecParse

/-!
# A typed template's Δ (#277), and the rows it makes copies

`HmRow.delta` writes a flat statement's entries (`delta_split`'s forms), then, for a typed template
(`c.typed = some tb`), per VU `g`:
1. each input bit `w`'s copy of its message bit: `(i, i)` and `(i, src)`, in `A` and `B`;
2. each part's bound input row `r`: `(i, i)` and `(i, col g x)` for its sources `x`, in `A` and `B`;
3. each root row's entries at its parts' columns: `A`'s in Δ_A, `B`'s in Δ_B;

with `i = col g r` and `col` the unit column's block position (`Typed.col`).

- `delta_typed`: Δ_A and Δ_B are the flat entries followed by `Typed.tail c tb true` and `Typed.tail c tb false`, the
  lists those loops write.
- `CopyRow S r src` and `ZeroRow S r`: the block rows at `r` read exactly `src`, or nothing, in `A` and `B`. A
  satisfying witness then carries at `r` what it carries at `src` (`CopyRow.block_eq`), or 0 (`ZeroRow.block_eq`).
-/

open Flock FlockLevel3

namespace FlockSoundness

/-! ## Copy and zero rows -/

section rows

variable {S : Model.Statement}

/-- **The block rows at `r` copy `src`**: in `A` and in `B` they read exactly `src`. -/
def CopyRow (S : Model.Statement) (r src : Fin (2 ^ S.kLog)) : Prop :=
  (∀ j, S.A₀ r j = if j = src then 1 else 0) ∧ ∀ j, S.B₀ r j = if j = src then 1 else 0

/-- **The block rows at `r` are zero**, in `A` and in `B`. -/
def ZeroRow (S : Model.Statement) (r : Fin (2 ^ S.kLog)) : Prop :=
  (∀ j, S.A₀ r j = 0) ∧ ∀ j, S.B₀ r j = 0

theorem mulVec_single (M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)) {r src : Fin (2 ^ S.kLog)}
    (h : ∀ j, M r j = if j = src then 1 else 0) (v : Fin (2 ^ S.kLog) → ZMod 2) : M.mulVec v r = v src := by
  simp only [Matrix.mulVec, dotProduct, h, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]

theorem mulVec_zero_row (M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)) {r : Fin (2 ^ S.kLog)}
    (h : ∀ j, M r j = 0) (v : Fin (2 ^ S.kLog) → ZMod 2) : M.mulVec v r = 0 := by
  simp [Matrix.mulVec, dotProduct, h]

/-- **A copy row carries its source**, in every block of a satisfying witness. -/
theorem CopyRow.block_eq {r src : Fin (2 ^ S.kLog)} (h : CopyRow S r src) {z : Model.Witness S.m}
    (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) : S.block z o r = S.block z o src := by
  have hr := hz.1 o r
  rw [mulVec_single S.A₀ h.1, mulVec_single S.B₀ h.2] at hr
  rw [← hr]
  generalize S.block z o src = x
  fin_cases x <;> rfl

/-- **A zero row carries 0**, in every block of a satisfying witness. -/
theorem ZeroRow.block_eq {r : Fin (2 ^ S.kLog)} (h : ZeroRow S r) {z : Model.Witness S.m} (hz : S.Satisfies z)
    (o : Fin (2 ^ S.nbl)) : S.block z o r = 0 := by
  have hr := hz.1 o r
  rw [mulVec_zero_row S.A₀ h.1, mulVec_zero_row S.B₀ h.2, zero_mul] at hr
  exact hr.symm

end rows

/-! ## The typed tail, as lists -/

namespace Typed

/-- Each row word's first message leaf (`HmRow.delta`'s `starts`). -/
def starts (c : Circuit) : Array ℕ :=
  (c.ports.foldl (fun (acc : Array ℕ × ℕ) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1

/-- Message bit `t` of byte `byte` of row `p`'s message in VU `g` (`HmRow.delta`'s `msgCol`): in its chain's
`sha512x3` slot, or past the chain in its hm96 slot's `pad`. -/
def msgCol (c : Circuit) (g p byte t : ℕ) : ℕ :=
  let (_, where_) := HmRow.rowSlots c.ports
  let sin := portCols (c.nets.getD c.shaNet default).2.inGroups
  let hin := portCols (c.nets.getD c.hmNet default).2.inGroups
  let (k, o) := HmRow.byteAt c.ports[p]! byte
  if k < (where_[p]!).size then
    let (slot, sub) := (where_[p]!)[k]!
    c.slotCol c.shaNet g slot + sin[2 * sub + 1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
  else c.slotCol c.hmNet g p + hin[1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t

/-- A row byte's offset in its block (`HmRow.byteAt`), segmented or not. -/
theorem byteAt_snd (p : Port) (byte : ℕ) : (HmRow.byteAt p byte).2 = byte % 128 := by
  unfold HmRow.byteAt
  split <;> rfl

/-- An unsegmented row's byte `byte` is in its compression `byte / 128`. -/
theorem byteAt_none {p : Port} (hs : p.segs = none) (byte : ℕ) : HmRow.byteAt p byte = (byte / 128, byte % 128) := by
  unfold HmRow.byteAt
  rw [hs]

/-- **A unit column's block position** in VU `g` (`HmRow.delta`'s `col`): the root's own columns in its slot, a part's
in its layout's slot. -/
def col (c : Circuit) (tb : TypedBlock) (g u : ℕ) : ℕ :=
  if u < tb.own then c.slotCol c.unitNet g 0 + u
  else match tb.parts.toList.reverse.find? (·.2.2 ≤ u) with
    | some (n, q, b) => c.slotCol n g q + (u - b)
    | none => 0

/-- **The message bit input `w` of VU `g` copies**: bit `w % 16` of row word `w / 16`. -/
def copySrc (c : Circuit) (g w : ℕ) : ℕ :=
  let (leaf, bit) := (w / 16, w % 16)
  let p := ((List.range (starts c).size).reverse.find? (fun q => (starts c)[q]! ≤ leaf)).getD 0
  msgCol c g p (2 * (leaf - (starts c)[p]!) + bit / 8) (bit % 8)

/-- VU `g`'s typed entries, in `A` (`side = true`) or `B`. -/
def tailVU (c : Circuit) (tb : TypedBlock) (side : Bool) (g : ℕ) : List (ℕ × ℕ) :=
  ((List.range tb.inCols.size).flatMap fun w =>
      [(col c tb g tb.inCols[w]!, col c tb g tb.inCols[w]!), (col c tb g tb.inCols[w]!, copySrc c g w)]) ++
    (tb.bind.toList.flatMap fun rs =>
      (col c tb g rs.1, col c tb g rs.1) :: rs.2.toList.map fun x => (col c tb g rs.1, col c tb g x)) ++
    tb.cross.toList.flatMap fun e => (if side then e.2.1 else e.2.2).toList.map fun x => (col c tb g e.1, col c tb g x)

/-- **The typed tail**: every VU's typed entries, in `A` (`side = true`) or `B`. -/
def tail (c : Circuit) (tb : TypedBlock) (side : Bool) : List (ℕ × ℕ) := (List.range c.g).flatMap (tailVU c tb side)

end Typed

/-! ## `HmRow.delta` for a typed template -/

theorem run_bind_pair {P : Array (ℕ × ℕ) → Prop} {Q : Array (ℕ × ℕ) × Array (ℕ × ℕ) → Prop} {m : Id (Array (ℕ × ℕ))}
    {k : Array (ℕ × ℕ) → Id (Array (ℕ × ℕ) × Array (ℕ × ℕ))} (hm : P (Id.run m))
    (hk : ∀ d, P d → Q (Id.run (k d))) : Q (Id.run (m >>= k)) := hk _ hm

/-- Δ's pair is the flat entries `d` (`delta_split`'s forms), then the typed tail on each side. -/
def TailQ (c : Circuit) (pin : ℕ) (tb : TypedBlock) (r : Array (ℕ × ℕ) × Array (ℕ × ℕ)) : Prop :=
  ∃ d : Array (ℕ × ℕ), ExtL (constPairs c pin) (DeltaIn c) d ∧
    r.1.toList = d.toList ++ Typed.tail c tb true ∧ r.2.toList = d.toList ++ Typed.tail c tb false

end FlockSoundness
