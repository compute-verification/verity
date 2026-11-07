module

public import Definitions.Core.Guarantees.Formalism
public import Definitions.Core.Formalism.Instances

/-!
# The size of a `(c, k)`-partitioned circuit: honest and nontrivial lemmas

`CKSize` has two statements here, at the instances of `Security/Definitions/Core/Formalism/Instances.lean`.
`CKSize_honest`: its premise holds at `Honest`, a `(2, 1)`-partition with a proof block of fewer than `c` gates.
`CKSize_nontrivial`: at `Counter`, one proof block of three gates, the premise fails at `(c, k) = (2, 1)`, and so does
the bound, `|C| = 3 > 2 · 1`.

The proofs are in `Security/Proofs/Core/Formalism/Witnesses.lean`.
-/

@[expose] public section

namespace Verity.Guarantees

/-- **`CKSize`'s premise holds** at `Honest`, a `(2, 1)`-partition whose second proof block has one gate. -/
def CKSize_honest : Prop := Formalism.Honest.IsCK Formalism.d 2 1

/-- **`CKSize` is false without its premise**: at `Counter`, one proof block of three gates, the partition is no
`(2, 1)`-partition, and the circuit's size is above `c · B = 2 · 1`. -/
def CKSize_nontrivial : Prop :=
  ¬ Formalism.Counter.IsCK Formalism.d 2 1 ∧ ¬ Formalism.d.D.size Formalism.d.D.computed ≤ 2 * 1

end Verity.Guarantees
