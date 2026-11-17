module

public import Definitions.Pouw.Guarantees.Receipt

/-! Steps of the receipt folds' binding proofs, stated as Props that no record pins. -/

@[expose] public section

namespace Pouw.Guarantees.Receipt

open Pouw.Receipt

/-- The served receipt is not injective in a word at a state. -/
def DuoReceiptNotInjective : Prop :=
  ¬ Function.Injective (fun d : BitVec 32 =>
    duoReceipt 16 (duoWord (0x6d403793#32) (0x9e43d96b#32) 6 (0x247a23f6#32, 0xc48c9971#32) d))

/-- Two duo words are not bound at a state. -/
def DuoPairAtStateNotInjective : Prop :=
  ¬ Function.Injective (fun d : BitVec 32 × BitVec 32 =>
    duoPair (0x6d403793#32) (0x9e43d96b#32) 6 (0x5cc85cac#32) (0x5e95a543#32) 20 d (0x247a23f6#32, 0xc48c9971#32))

end Pouw.Guarantees.Receipt
