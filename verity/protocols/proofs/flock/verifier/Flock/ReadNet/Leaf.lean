/-
hm96-sha512's leaf under the pinned key as a builder program (`gates.hm96.leaf_gadget`, the traced Definition
`Hm96Sha512Leaf_v1`).
-/
module

public import Flock.ReadNet.Sha

@[expose] public section

namespace Flock
namespace ReadNet

open HmNets

/-- `hm96.leaf_gadget` (`Hm96Sha512Leaf_v1`) under the pinned key, of the digest `x` (8 words) and the salt `y`
(24 words), each in a digest's layout: `b = x ⊕ M·y` (byte-string bit `i` is `x`'s plus `⊕_j key_{i+j} y_j`), each word
committed (`sha512.commit`); `c = SHA-512(salt_prefix ‖ y)` and the leaf `SHA-512(leaf_prefix(key) ‖ b ‖ c)`, each from
its prefix's chaining value. The output is `b ‖ c ‖ leaf`, 24 words. -/
def leaf (x y : List Word) : M (List Word) := do
  let xf := x.flatten
  let ys := (List.range 1536).map fun j => y.flatten.getD (wordBit j) Lin.zero
  let bi := fun i => maskBit Hm96.Default512.key ys i (xf.getD (wordBit i) Lin.zero)
  let b ← (List.range 8).mapM fun w => commitW ((List.range 64).map fun p => bi (wordBit (64 * w + p)))
  let c ← shaW 2 (midstate Hm96.Default512.saltPrefix) (y ++ constWords (padTail 320))
  let leaf ← shaW 2 (midstate Hm96.Default512.leafPrefix) (b ++ c ++ constWords (padTail 256))
  pure (b ++ c ++ leaf)

end ReadNet
end Flock
