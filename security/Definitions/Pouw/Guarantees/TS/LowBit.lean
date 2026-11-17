module

public import Definitions.Pouw.TS.Fold
public import Definitions.Pouw.Receipt.Li

/-! A one-register fold's low bit is an XOR of its words' low bits, so a lower bound read off it gives no more than `FoldFanIn`. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS

/-- **li's low bit is a parity.** For words of `n + 1` bits, any half steps with odd multipliers and any start `y`:
bit 0 of li's chain is the XOR of `y`'s bit 0 and, half step by half step, the bits 0 of `d₀`, the key `k` and
`d₁`. -/
def LiLowBit : Prop :=
  ∀ (n : ℕ) (hs : List (Pouw.Receipt.LiOddHalfStep (n + 1))) (y : BitVec (n + 1)),
    (Pouw.Receipt.liChain (hs.map Pouw.Receipt.LiOddHalfStep.val) y).getLsbD 0 =
      hs.foldl (fun x h => x ^^ h.2.2.1.getLsbD 0 ^^ h.1.getLsbD 0 ^^ h.2.2.2.getLsbD 0) (y.getLsbD 0)

end ResourceAccounting.Guarantees.TS
