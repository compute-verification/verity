import FlockSoundness.Accounting.Schedule
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import FlockSoundness.Protocol.Accounting.Bound

/-!
# The soundness bound as a function of the parameters

`repError s st` is the probability that one rep accepts when the committed values do not satisfy
the statement, and `tableError s st` the probability that a table's two reps both accept. They
are functions of the `fast100` schedule `s` (PROTOCOL.md §15) and of the statement's shape `st`
(§16), with one term per verifier coin of PROTOCOL.md §9–§13 (the coin's step id is in each
docstring). The soundness theorem proves that these are bounds; `Accounting.Fast100` evaluates them.

**Where each term comes from.** A coin drawn after the prover has fixed a nonzero polynomial of
total degree `D` hits a zero with probability at most `D / |field|` (Schwartz–Zippel). In the
list-decoding regime the level-0 commitment is close to up to `L₀` codewords, and a claim must
fail for every one of them, so every term before Ligerito's binding carries a factor `L₀` (paper
Appendix A, "Security"). Ligerito's per-level terms follow paper Appendix C.3 with the spec's
changes (§19, A11): folding challenges in GF(2^256), two out-of-domain samples per committed
level, stratified queries.

**List sizes.** `Level.listBound` is ArkLib's proved Johnson bound for interleaved Reed–Solomon
codes (`irs_lambda_le_johnson_mds`): at radius `1 − √ρ − η` at most `1 / (2ηρ) = 100·2^r`
codewords. Flock's own accounting uses the sharper `1 / (2η√ρ)`; the difference moves the total
by less than 0.001 bit.
-/

namespace FlockSoundness.Accounting

example : [218, 106, 71, 53, 43, 36, 32].map ceilLog2 = [8, 7, 7, 6, 6, 6, 5] := by decide

namespace Level

end Level

end FlockSoundness.Accounting
