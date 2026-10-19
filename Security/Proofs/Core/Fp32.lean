import Proofs.Core.Fp32.Vectors
import Proofs.Core.Fp32.Round

/-!
# The FP32 arithmetic, and the Python that mirrors it

`Vectors` checks `verity.ml.fp32`'s evaluators against `Verity.Fp32`'s definitions, word for word, and `Round` proves
`roundRn` the nearest FP32 word, ties to even.
-/
