import Proofs.Flock.Soundness.ExecFlatCopies

/-!
# The zero as Δ writes it (`hZero`, e2e)

A statement without a typed template (an untyped one, or a typed flat class) has Δ copy a forced-zero row into a unit
copy's input bit in two cases only (`HmRow.delta`):
- a leaf port whose leaf is the zero (`-1`), or whose bit is past its row's length, copies the copy's own forced-zero
  row, `ur.slot k + unit.useful`;
- a leaf cut's bits past the row word's 16, or past its row's length, copy the unit range's first slot's forced-zero
  row, `ur.slot 0 + unit.useful`.

`ZeroPort` and `ZeroAt` name those inputs, by port and bit and by slot offset. `flatSrc_zero` and `flatSrcAt_zero`: Δ's
source there (`Flat.flatSrc`, `flatSrcAt`, the column `HmRow.delta` copies into the input) is one of those forced-zero
rows. The rows' constant bits that are 0 copy `ur.slot 0 + unit.useful` too, but they are no unit's inputs.
-/

open Flock

namespace FlockSoundness.Discharge.Zero

/-- **Unit copy `s₀`'s input port `p`, bit `t`, reads the zero**: a leaf port whose leaf is the zero (`-1`) or whose bit
is past its row's length, or a cut port's bit past the 16 bits of the row word the first leaf cut into it wires, or past
that row's length. -/
def ZeroPort (c : Circuit) (s₀ p t : ℕ) : Prop :=
  (p < (c.unit.inGroups.getD 0 default).bits.size ∧ ((c.leavesIn.getD (s₀ % c.upv) #[])[p]! < 0 ∨
    ¬ 16 * (((c.leavesIn.getD (s₀ % c.upv) #[])[p]!).toNat -
          (Typed.starts c)[((List.range (Typed.starts c).size).reverse.find?
            (fun q => (Typed.starts c)[q]! ≤ ((c.leavesIn.getD (s₀ % c.upv) #[])[p]!).toNat)).getD 0]!) + t <
      c.ports[((List.range (Typed.starts c).size).reverse.find?
        (fun q => (Typed.starts c)[q]! ≤ ((c.leavesIn.getD (s₀ % c.upv) #[])[p]!).toNat)).getD 0]!.bits)) ∨
    ((c.unit.inGroups.getD 0 default).bits.size ≤ p ∧
      ∃ e ∈ (c.leafCuts.toList.filter fun e => decide (e.1 = s₀ % c.upv ∧ e.2.1 = p)).head?,
        ¬ (t < 16 ∧ 16 * (e.2.2 - (Typed.starts c)[((List.range (Typed.starts c).size).reverse.find?
            (fun q => (Typed.starts c)[q]! ≤ e.2.2)).getD 0]!) + t <
          c.ports[((List.range (Typed.starts c).size).reverse.find?
            (fun q => (Typed.starts c)[q]! ≤ e.2.2)).getD 0]!.bits))

/-- **Unit copy `s₀`'s input at slot offset `off` reads the zero**: `ZeroPort` at the port `off` is a bit of
(`portOf`) and its bit there. -/
def ZeroAt (c : Circuit) (s₀ off : ℕ) : Prop :=
  ZeroPort c s₀ (portOf (portCols c.unit.inGroups) off)
    (off - (portCols c.unit.inGroups)[portOf (portCols c.unit.inGroups) off]!.1)

/-- **Δ copies a forced-zero row into an input that reads the zero**: the copy's own (`ur.slot s₀ + unit.useful`, a
zero leaf) or the unit range's first slot's (`ur.slot 0 + unit.useful`, a cut bit past 16). -/
theorem flatSrc_zero {c : Circuit} {s₀ p t : ℕ} (h : ZeroPort c s₀ p t) :
    Flat.flatSrc c s₀ p t = c.unitRange.slot s₀ + c.unit.useful ∨
      Flat.flatSrc c s₀ p t = c.unitRange.slot 0 + c.unit.useful := by
  unfold Flat.flatSrc
  rcases h with ⟨hp, hl | hb⟩ | ⟨hp, e, he, hz⟩
  · rw [ite_eq_left hp]
    left
    unfold Flat.leafSrc
    rw [ite_eq_left hl]
  · rw [ite_eq_left hp]
    left
    unfold Flat.leafSrc
    by_cases hl : (c.leavesIn.getD (s₀ % c.upv) #[])[p]! < 0
    · rw [ite_eq_left hl]
    · simp only [hl, hb, ↓reduceIte]
  · rw [ite_eq_right (by omega)]
    rcases hc : c.leafCuts.toList.filter (fun e => decide (e.1 = s₀ % c.upv ∧ e.2.1 = p)) with _ | ⟨e', rest⟩
    · rw [hc] at he
      cases he
    · rw [hc] at he
      cases he
      right
      simp only [hc, Flat.cutSrc]
      rw [ite_eq_right (by simpa only [Bool.and_eq_true, decide_eq_true_eq] using hz)]

/-- **`flatSrc_zero` at slot offset `off`.** -/
theorem flatSrcAt_zero {c : Circuit} {s₀ off : ℕ} (h : ZeroAt c s₀ off) :
    flatSrcAt c s₀ off = c.unitRange.slot s₀ + c.unit.useful ∨
      flatSrcAt c s₀ off = c.unitRange.slot 0 + c.unit.useful :=
  flatSrc_zero h

end FlockSoundness.Discharge.Zero
