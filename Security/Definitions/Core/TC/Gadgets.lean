import Definitions.Core.TC.Relation

/-!
# The range and is-zero gadgets, chunk by chunk

`Relation.RANGE` and `Relation.ISZERO` are the predicates the gadgets of `relation.Census` enforce. Here the gadgets
themselves, with their chunk columns as a circuit holds them:

* `Census.RANGE x bits`: `x` split into little-endian chunks of 16 bits (the last one `bits % 16` wide), one linear
  split and one range-table lookup per chunk (`rangeWidths`, `Chunks`).
* `Census.ISZERO x bits z`: `x` split into chunks of 12 bits, one is-zero lookup per chunk, the flags multiplied into
  `z` (`isZeroWidths`, `ZeroChunks`).

`Verity.SecurityProofs.TC.Gadgets` proves that each holds exactly when its predicate does.
-/

namespace Verity.TC.Gadgets

open Verity.TC.Relation

/-- `x` is the little-endian sum of chunks of the given widths, each in its range table. -/
def Chunks : List Nat → Int → Prop
  | [], x => x = 0
  | w :: ws, x => ∃ c y, 0 ≤ c ∧ c < 2 ^ w ∧ Chunks ws y ∧ x = c + 2 ^ w * y

/-- The chunk widths `Census.RANGE` uses: 16-bit chunks, the last `bits % 16` wide. -/
def rangeWidths (bits : Nat) : List Nat :=
  List.replicate (bits / 16) 16 ++ (if bits % 16 ≠ 0 then [bits % 16] else [])

/-- The chunk widths `Census.ISZERO` uses: 12-bit chunks, the last `bits % 12` wide. -/
def isZeroWidths (bits : Nat) : List Nat :=
  List.replicate (bits / 12) 12 ++ (if bits % 12 ≠ 0 then [bits % 12] else [])

/-- `x` is the little-endian sum of in-range chunks, and `z` is the product of their is-zero flags. -/
def ZeroChunks : List Nat → Int → Int → Prop
  | [], x, z => x = 0 ∧ z = 1
  | w :: ws, x, z => ∃ c y z', 0 ≤ c ∧ c < 2 ^ w ∧ ZeroChunks ws y z' ∧ x = c + 2 ^ w * y ∧
      z = (if c = 0 then 1 else 0) * z'

end Verity.TC.Gadgets
