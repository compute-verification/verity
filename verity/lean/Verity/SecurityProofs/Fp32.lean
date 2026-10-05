import Verity.SecurityProofs.Fp32.Vectors
import Verity.SecurityProofs.Fp32.Round

/-!
# The FP32 arithmetic held to the Python

`Vectors` checks `Verity.Fp32`'s definitions against `verity.ml.fp32`'s evaluators, word for word, and `Round` proves
`roundRn` the nearest FP32 word, ties to even.
-/
