module

public import Proofs.Core.Fp32.Vectors
public import Proofs.Core.Fp32.Round
import Mathlib.Data.Rat.Floor
import Mathlib.Tactic.Positivity.Finset

/-!
# The FP32 arithmetic, and the Python that mirrors it

`Vectors` checks `verity.ml.fp32`'s evaluators against `Verity.Fp32`'s definitions, word for word, and `Round` proves
`roundRn` the nearest FP32 word, ties to even.
-/

@[expose] public section
