module

public import Definitions.Core.NaiveStrassen

/-!
# Naive tiled against Strassen: D4's guarantees

Theorem 1 at the served depth, and its size form, on every atom table that the RTX PRO 6000's data sheet dominates
(`Verity.Accel.NaiveStrassen.Packed.Dom`): one each of whose atoms, wherever it holds, weighs at least the lightest
atom of that sheet holding there. The H200 SXM's data sheet is such a table. Served dimensions are at most 65536, and
a leaf is at least 16 (sm_120 has no `wgmma`, and the H200's 16-wide `mma.sync` tiles are priced at its `wgmma`'s
weight, the most permissive price), so a recursion on a served product is at most `log₂(65536 / 16) = 12` deep, and
the size form's `c` is `65536 / 16 = 4096`. Both are about int8 operands at `(256, 256)`.
-/

@[expose] public section

namespace Verity.Guarantees

open Verity.Accel.NaiveStrassen

/-- **Theorem 1 at the served depth**: on every atom table the RTX PRO 6000's data sheet dominates, at every depth
`l ≤ 12`, every schedule for int8 operands that applies Strassen or Winograd somewhere costs at least naive tiled's
`8^(l+1)` plus 32. -/
def NaiveBeatsStrassenAny : Prop :=
  ∀ fs : List Atom, Packed.Dom fs → NaiveBeatsStrassen fs 12

/-- **Theorem 1 in size**: on every atom table the RTX PRO 6000's data sheet dominates, an `n × n` int8 product
padded to `2^l` leaves of size `b`, at any depth `l`, with Strassen or Winograd somewhere, costs more than naive
tiled's `8 n³` whenever `n ≤ 4096 b`. -/
def NaiveBeatsStrassenAnySize : Prop :=
  ∀ fs : List Atom, Packed.Dom fs → NaiveBeatsStrassenUpTo fs 4096

end Verity.Guarantees
