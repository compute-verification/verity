module

public import Mathlib.Data.Nat.Notation

/-!
# The streamed wipe's residual, in integers

The labels the sampling half of the streamed wipe leaves unanswered (`wipeMissing`) and the wipe residual in bits
(`wipeResidual`), arithmetic on naturals that `Pous.Guarantees.StreamedWipeResidual`
(`security/Definitions/Pous/Guarantees/StreamedWipeResidual.lean`) states its bound at.
-/

@[expose] public section

namespace Pous.StreamedWipe

/-- **The labels the sampling half leaves unanswered**, `D = ⌈(s + 1) · 0.6931471808 · N / k⌉`: the least `D` with
`k · D / N ≥ (s + 1) · 0.6931471808`, so that `(1 − D/N)^k ≤ e^(−k·D/N) ≤ 2^-(s+1)`, as `0.6931471808` is above
`ln 2`. In integers: `⌈(s + 1) · 6931471808 · N / (k · 10^10)⌉`. At `k = 0` no label is shown held, and it is
`N + 1`, so that `D − 1 = N` and the residual counts every label's bits: at least `N · b`, where the guarantee's side
condition fails. -/
def wipeMissing (N k s : ℕ) : ℕ :=
  if k = 0 then N + 1 else ((s + 1) * 6931471808 * N + k * 10 ^ 10 - 1) / (k * 10 ^ 10)

/-- **The wipe residual, in bits**, at soundness error `2^-s` plus the PRF's advantage, for `N` labels of `b` bits
filling the whole memory and `k` challenges: with `D = wipeMissing N k s` the labels the sampling half leaves
unanswered (sampling),
`(⌊log₂ k⌋ + 1) + (⌊log₂ N⌋ + 1 + b) · (D − 1) + (s + 1)`:
`b · (D − 1)` bits for the labels the prover need not hold, `(⌊log₂ N⌋ + 1) · (D − 1)` for which ones they are
(counting: `C(N, D − 1) ≤ N^(D−1) ≤ 2^((⌊log₂ N⌋ + 1)·(D − 1))`), `⌊log₂ k⌋ + 1` for the round that finds them
(`k ≤ 2^(⌊log₂ k⌋ + 1)`), and `s + 1` for the compression half of the error (slack). It over-estimates the residual
the bound gives at `g = N − D`, and is computed by `decide` on naturals. At `k = 0` it is at least `N · b`, and at
`N = 0` at least 1 bit against `N · b = 0`: either way `wipeResidual N k b s < N · b` fails and the bound says
nothing. -/
def wipeResidual (N k b s : ℕ) : ℕ :=
  (k.log2 + 1) + (N.log2 + 1 + b) * (wipeMissing N k s - 1) + (s + 1)

end Pous.StreamedWipe
