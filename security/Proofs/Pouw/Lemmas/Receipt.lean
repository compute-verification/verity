module

public import Definitions.Pouw.Guarantees.Receipt

/-!
# The receipt folds duo's, fx1's and li's binding: lemmas

The statements of `Definitions.Pouw.Guarantees.Receipt` that are steps of its guarantees' proofs rather than guarantees.
Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same name, and
audited with the proofs: no record pins it.

* `DuoReceiptNotInjective`: at width 32 with the served word 0's keys, one start state and two words give one receipt.
* `DuoPairAtStateNotInjective`: at width 32 with the keys of the served chain 0's first two words, one start state and
  two word pairs give one state.
-/

@[expose] public section

namespace Pouw.Guarantees.Receipt

open Pouw.Receipt

/-- **The served receipt is not injective in a word at a state.** At width 32, with the XOR key `0x6d403793`, multiplier
`0x9e43d96b` and rotation 6 of the served word 0 (`fgemm.cu`'s `key_xor(0, 0)`, `key_mul(0, 0)`, `rot_of(DUO, 0, 0)`) and
receipt rotation 16, from the state `(0x247a23f6, 0xc48c9971)` the words `0x1632c` and `0x2237a` give the receipt
`0x26d90889`. -/
def DuoReceiptNotInjective : Prop :=
  ¬ Function.Injective (fun d : BitVec 32 =>
    duoReceipt 16 (duoWord (0x6d403793#32) (0x9e43d96b#32) 6 (0x247a23f6#32, 0xc48c9971#32) d))

/-- **Two duo words are not bound at a state.** At width 32, with the keys of the served chain 0's first two words
(`fgemm.cu`'s `key_xor(0, t)`, `key_mul(0, t)`, `rot_of(DUO, 0, t)` at `t = 0` and `t = 2`: `(0x6d403793, 0x9e43d96b, 6)`
and `(0x5cc85cac, 0x5e95a543, 20)`), from the state `(0x247a23f6, 0xc48c9971)` the word pairs `(0x40156b95, 0x0)` and
`(0xb4d85e5c, 0x375f7285)` reach one state. -/
def DuoPairAtStateNotInjective : Prop :=
  ¬ Function.Injective (fun d : BitVec 32 × BitVec 32 =>
    duoPair (0x6d403793#32) (0x9e43d96b#32) 6 (0x5cc85cac#32) (0x5e95a543#32) 20 d (0x247a23f6#32, 0xc48c9971#32))

end Pouw.Guarantees.Receipt
