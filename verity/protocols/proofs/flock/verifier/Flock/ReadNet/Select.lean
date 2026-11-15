/-
The table read's comparator and select as builder programs (`gates.embedding._below`, `_select`; the traced Definition
`EmbeddingReadSelect_v1{V, H}`).
-/
module

public import Flock.HmNets

@[expose] public section

namespace Flock
namespace ReadNet

open HmNets

/-- `embedding._below` from bit `k` on: `lt` is whether the bits below `k` are below `bound`'s, and each bit updates it
with one AND. -/
def belowGo (bound : Nat) : List Lin → Nat → Lin → M Lin
  | [], _, lt => pure lt
  | b :: bs, k, lt => do
    let lt ← if bound.testBit k then do
        let r ← HmNets.and b (lt ^^^ Lin.one)
        pure (r ^^^ Lin.one)
      else HmNets.and (b ^^^ Lin.one) lt
    belowGo bound bs (k + 1) lt

/-- `embedding._below`: 1 iff the unsigned `bits` (least significant first) are below the constant `bound`. -/
def below (bits : List Lin) (bound : Nat) : M Lin := belowGo bound bits 0 Lin.zero

/-- `embedding.FILL`, the word a token outside the vocabulary reads. -/
def FILL : Nat := 0x7FFF

/-- `FILL`'s 16 bits, least significant first, as constants. -/
def fillBits : List Lin := (List.range 16).map fun j => Lin.const (FILL.testBit j)

/-- `embedding._select` (`EmbeddingReadSelect_v1{V, H}`) on the row's forms, its 16-bit words one after another:
`inside = tok < V` (unsigned); `h = row` where inside, else `FILL` in every word, one AND a bit; `ok = opened`, or 1
outside, one AND. -/
def select (V : Nat) (tok : List Lin) (opened : Lin) (row : List Lin) : M (List Lin × Lin) := do
  let inside ← below tok V
  let h ← (row.zip (List.replicate (row.length / 16) fillBits).flatten).mapM fun rf => do
    let r ← HmNets.and inside (rf.1 ^^^ rf.2)
    pure (rf.2 ^^^ r)
  let r ← HmNets.and inside (opened ^^^ Lin.one)
  pure (h, Lin.one ^^^ r)

end ReadNet
end Flock
