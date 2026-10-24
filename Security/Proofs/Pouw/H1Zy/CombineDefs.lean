import Definitions.Pouw.H1Zy.Combine

/-! The proof-only part of `Definitions.Pouw.H1Zy.Combine`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.H1Zy

/-- `PouwXor32_v1`, `PouwXor32_v2`'s word view: `a xor b`. -/
def xor32 (a b : BitVec 32) : BitVec 32 := a ^^^ b

/-- `PouwEq32_v1`, `PouwEq32_v2`'s word view (`segment.PouwEq32Word`): `1` exactly when `x = y`. -/
def eq32 (x y : BitVec 32) : Bool := decide (x = y)

/-- **The combine's `ok`** (`h1zy._combine_body`) on its `L` lanes' partials and the receipt, `nw` u32 words each:
for every word `w`, the `PouwXor32` tree over the lanes' words `w` in lane order is the receipt's word `w`
(`PouwEq32`), and the comparisons' `PouwAndBit` tree. -/
def combineOk {L nw : ℕ} (partials : Fin L → Fin nw → BitVec 32) (receipt : Fin nw → BitVec 32) : Bool :=
  tree andBit true (List.ofFn fun w => eq32 (tree xor32 0 (List.ofFn fun l => partials l w)) (receipt w))

end Pouw.H1Zy
