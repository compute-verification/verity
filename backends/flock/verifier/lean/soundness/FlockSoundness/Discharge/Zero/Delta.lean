import FlockSoundness.ExecFlatCopies

/-!
# The zero as Δ writes it (`hZero`, e2e)

A statement without a typed template (an untyped one, or a typed flat class) has Δ copy a forced-zero row into a unit
copy's input bit in two cases only (`HmRow.delta`):
- a leaf port whose leaf is the zero (`-1`), or whose bit is past its row's length (`PastBits`), copies the copy's own
  forced-zero row, `ur.slot k + unit.useful`;
- a leaf cut's bits past the row word's 16, or past its row's length, copy the unit range's first slot's forced-zero row,
  `ur.slot 0 + unit.useful`.

`ZeroPort` and `ZeroAt` name those inputs, by port and bit and by slot offset. `flatSrc_zero` and `flatSrcAt_zero`: Δ's
source there (`Flat.flatSrc`, `flatSrcAt`, the column `HmRow.delta` copies into the input) is one of those forced-zero
rows. The rows' constant bits that are 0 copy `ur.slot 0 + unit.useful` too, but they are no unit's inputs.
-/

open Flock

namespace FlockSoundness.Discharge.Zero

/-- **Bit `t` of row word `x` is past its row's length**: the port holding the word (the last whose first word is at or
before it, `Typed.starts`) has at most `16 (x − start) + t` bits (`Port.bits`). -/
def PastBits (c : Circuit) (x t : ℕ) : Prop :=
  c.ports[((List.range (Typed.starts c).size).reverse.find? (fun q => (Typed.starts c)[q]! ≤ x)).getD 0]!.bits ≤
    16 * (x - (Typed.starts c)[((List.range (Typed.starts c).size).reverse.find?
      (fun q => (Typed.starts c)[q]! ≤ x)).getD 0]!) + t

instance (c : Circuit) (x t : ℕ) : Decidable (PastBits c x t) := by
  unfold PastBits
  infer_instance

/-- **Unit copy `s₀`'s input port `p`, bit `t`, reads the zero**: a leaf port whose leaf is the zero (`-1`) or whose bit
is past its row's length, or a cut port whose leaf cut's bit is past the 16 bits of the row word it wires in or past
that row's length. -/
def ZeroPort (c : Circuit) (s₀ p t : ℕ) : Prop :=
  (p < (c.unit.inGroups.getD 0 default).bits.size ∧
      ((c.leavesIn.getD (s₀ % c.upv) #[])[p]! < 0 ∨ PastBits c ((c.leavesIn.getD (s₀ % c.upv) #[])[p]!).toNat t)) ∨
    ((c.unit.inGroups.getD 0 default).bits.size ≤ p ∧
      ∃ e ∈ (c.leafCuts.toList.filter (fun e => decide (e.1 = s₀ % c.upv ∧ e.2.1 = p))).head?,
        16 ≤ t ∨ PastBits c e.2.2 t)

/-- **Unit copy `s₀`'s input at slot offset `off` reads the zero**: `ZeroPort` at the port `off` is a bit of
(`portOf`) and its bit there. -/
def ZeroAt (c : Circuit) (s₀ off : ℕ) : Prop :=
  ZeroPort c s₀ (portOf (portCols c.unit.inGroups) off)
    (off - (portCols c.unit.inGroups)[portOf (portCols c.unit.inGroups) off]!.1)

/-- **Δ copies a forced-zero row into an input that reads the zero**: the copy's own (`ur.slot s₀ + unit.useful`, a
leaf port) or the unit range's first slot's (`ur.slot 0 + unit.useful`, a cut port). -/
theorem flatSrc_zero {c : Circuit} {s₀ p t : ℕ} (h : ZeroPort c s₀ p t) :
    Flat.flatSrc c s₀ p t = c.unitRange.slot s₀ + c.unit.useful ∨
      Flat.flatSrc c s₀ p t = c.unitRange.slot 0 + c.unit.useful := by
  unfold Flat.flatSrc
  rcases h with ⟨hp, hl⟩ | ⟨hp, e, he, ht⟩
  · rw [ite_eq_left hp]
    left
    unfold Flat.leafSrc
    split
    · rfl
    · rename_i hn
      rcases hl with hl | hl
      · exact absurd hl hn
      · unfold PastBits at hl
        dsimp only
        rw [ite_eq_right (by omega)]
  · rw [ite_eq_right (by omega)]
    right
    generalize c.leafCuts.toList.filter (fun e => decide (e.1 = s₀ % c.upv ∧ e.2.1 = p)) = l at he ⊢
    cases l with
    | nil => cases he
    | cons a r =>
      obtain rfl : a = e := Option.some.inj he
      unfold Flat.cutSrc
      unfold PastBits at ht
      dsimp only
      rw [ite_eq_right (by simp only [Bool.and_eq_true, decide_eq_true_eq]; omega)]

/-- **`flatSrc_zero` at slot offset `off`.** -/
theorem flatSrcAt_zero {c : Circuit} {s₀ off : ℕ} (h : ZeroAt c s₀ off) :
    flatSrcAt c s₀ off = c.unitRange.slot s₀ + c.unit.useful ∨
      flatSrcAt c s₀ off = c.unitRange.slot 0 + c.unit.useful :=
  flatSrc_zero h

end FlockSoundness.Discharge.Zero
