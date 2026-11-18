module

public import Flock.Program

/-! A type's bits and its u16 words as a period commits a field (`interface.layout`'s `o<j>`): the link check's offsets
(`HmLinks`) and SPOT's edges and rows (`Spot.Bits`, `Spot.Edges`) read this one count. -/

@[expose] public section

namespace Flock.Program

/-- **A type's bits**: its leaves' widths summed. -/
def bitsOf (t : Ty) : Nat := t.widths.foldl (· + ·) 0

/-- **A value's 16-bit words**: its bits packed sixteen to a word, least significant first, the last word's high bits
padding: two 8-bit leaves share a word. A parameter is one value, and so is a return field. -/
def wordsOf (t : Ty) : Nat := (bitsOf t + 15) / 16

end Flock.Program
