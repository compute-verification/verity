import Definitions.Nci.Products.Instances

/-!
# Products of two fresh operands without the word level: the partition their nontrivial lemmas are stated at (trusted)

`Pair.P1`, `Pair`'s two MACs, gates 4 and 5, in one block: the block holds the entry's whole chain, so it reads no
partial sum, and with the operands registered it reads nothing beyond `θ`.
-/

namespace Nci.Products.Pair

/-- One block, gates 4 and 5. -/
abbrev P1 : Verity.Partition C 1 := ⟨![none, none, none, none, some 0, some 0], by decide⟩

end Nci.Products.Pair
