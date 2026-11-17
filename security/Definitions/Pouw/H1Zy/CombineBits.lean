module

public import Definitions.Pouw.H1Zy.Combine
import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike

/-! `PouwH1Combine_v1{TILE}` on bits, its three pieces at their bit lowerings. -/

@[expose] public section

namespace Pouw.H1Zy

/-- A word's bits, LSB first. -/
def bitsOf (x : BitVec 32) : Fin 32 → Bool := fun i => x.getLsbD i.val

/-- `PouwXor32_v2`'s lowering: bit `i` is `a i xor b i`. -/
def xor32Bits (a b : Fin 32 → Bool) : Fin 32 → Bool := fun i => xor (a i) (b i)

/-- `PouwEq32_v2`'s lowering (`segment._eq`): the AND tree of `not (x i xor y i)` over the 32 bits in order. -/
def eq32Bits (x y : Fin 32 → Bool) : Bool := tree andBit true (List.ofFn fun i => !(xor (x i) (y i)))

/-- `PouwH1Combine_v1`'s body on bits: per word, the `xor32Bits` tree over the lanes' words in lane order, held to the
receipt's word by `eq32Bits`, and the `andBit` tree over the words. -/
def combineBits {L nw : ℕ} (partials : Fin L → Fin nw → Fin 32 → Bool) (receipt : Fin nw → Fin 32 → Bool) : Bool :=
  tree andBit true
    (List.ofFn fun w => eq32Bits (tree xor32Bits (fun _ => false) (List.ofFn fun l => partials l w)) (receipt w))

end Pouw.H1Zy
