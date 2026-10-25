import Definitions.Pouw.H1Zy.Combine

/-!
# `PouwH1Combine_v1{TILE}` on bits (trusted)

The combine's body as circuit-check lowers it (`h1zy._combine_body`), each word a `Fin 32 → Bool`, bit `i` the word's
bit `i` LSB first (`bitsOf`), with its three pieces at their bit lowerings rather than their word views:

* `PouwXor32_v2` (`boolean.PouwXor32`, pinned `{"and": 0}` in `boolean.circuit-check.json`): the 32 bits' XORs
  (`xor32Bits`);
* `PouwEq32_v2` (`segment.PouwEq32`, `segment._eq`, not pinned): `and_reduce` of the 32 bits' XNORs, `and_reduce`
  (`verity/ml/circuits/boolean/forms.py`) being the pairwise AND tree with `1` for the empty list (`eq32Bits`);
* `PouwAndBit_v2` (`h1zy.PouwAndBit`, pinned `{"and": 1}` in `h1zy.circuit-check.json`): one AND, `andBit` itself.

They compose as on words (`Definitions.Pouw.H1Zy.Combine`): per word a `tree` of `xor32Bits` over the lanes, its
`eq32Bits` against the receipt's word, and a `tree` of `andBit` over the words (`combineBits`). At a leaf of 32 lanes and
32 words that is `32 · 31 + 31 = 1,023` ANDs. `Proofs.Pouw.H1Zy.CombineBits` proves each piece's lowering computes its
word view and `combineBits` computes `combineOk`.
-/

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
