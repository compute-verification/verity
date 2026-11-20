module

public import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed wipe's residual as a function of the fill and the challenge count

`Pous.Guarantees.StreamedWipe` at `g = N − D`, with the error split in two halves of `2^-(s+1)`, as
`StreamedWipeMeets2600GB` splits it, and every number left a variable: the wipe residual `wipeResidual N k b s`, in
bits, is the prior content an accepted prover can keep beyond the egress, at soundness error `2^-s` plus the PRF's
advantage, for a wipe of `N` labels of `b` bits whose fill is the whole memory (`M = N · b`) and `k` sequential
challenges. `N` and `b` come from the hardware description, and `k` is the wipe's challenge count, which the wipe's
check outputs. The game is `Security/Definitions/Pous/Model/StreamedWipe.lean`, the proof
`Security/Proofs/Pous/StreamedWipe/Residual.lean`.

**What it rests on.** No named assumption, as `StreamedWipe`.
-/

@[expose] public section

namespace Pous.StreamedWipe

/-- **The labels the sampling half leaves unanswered**, `D = ⌈(s + 1) · 0.6931471808 · N / k⌉`: the least `D` with
`k · D / N ≥ (s + 1) · 0.6931471808`, so that `(1 − D/N)^k ≤ e^(−k·D/N) ≤ 2^-(s+1)`, as `0.6931471808` is above
`ln 2`. In integers: `⌈(s + 1) · 6931471808 · N / (k · 10^10)⌉`. -/
def wipeMissing (N k s : ℕ) : ℕ := ((s + 1) * 6931471808 * N + k * 10 ^ 10 - 1) / (k * 10 ^ 10)

/-- **The wipe residual, in bits**, at soundness error `2^-s` plus the PRF's advantage, for `N` labels of `b` bits
filling the whole memory and `k` challenges: with `D = wipeMissing N k s` the labels the sampling half leaves
unanswered (sampling),
`(⌊log₂ k⌋ + 1) + (⌊log₂ N⌋ + 1 + b) · (D − 1) + (s + 1)`:
`b · (D − 1)` bits for the labels the prover need not hold, `(⌊log₂ N⌋ + 1) · (D − 1)` for which ones they are
(counting: `C(N, D − 1) ≤ N^(D−1) ≤ 2^((⌊log₂ N⌋ + 1)·(D − 1))`), `⌊log₂ k⌋ + 1` for the round that finds them
(`k ≤ 2^(⌊log₂ k⌋ + 1)`), and `s + 1` for the compression half of the error (slack). It over-estimates the residual
the bound gives at `g = N − D`, and is computed by `decide` on naturals. -/
def wipeResidual (N k b s : ℕ) : ℕ :=
  (k.log2 + 1) + (N.log2 + 1 + b) * (wipeMissing N k s - 1) + (s + 1)

end Pous.StreamedWipe

namespace Pous.Guarantees

open scoped ENNReal

/-- **The streamed wipe at any fill and challenge count: `E + wipeResidual N k b s` bits at `2^-s`.** For a wipe of
`N ≥ 1` labels of `b` bits whose fill is the whole memory (`M = N · b`), `k ≥ 1` sequential challenges and a soundness
exponent `s`, with the wipe residual below the memory (`wipeResidual N k b s < N · b`; at or above it the bound says
nothing): a prover with `E` bits of egress that keeps at least `E + wipeResidual N k b s` bits of uniform prior content
through the wipe is accepted with probability at most `2^-s` plus the PRF's advantage.

Same footing as `StreamedWipe`: no named assumption. Proof: `PousStreamedWipe.streamed_wipe_residual`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeResidual : Prop :=
  ∀ (N k b s q E κ : ℕ) (prf : Bits κ → Fin N → Bits b) (P : StreamedWipe.Prover N k b q (N * b) E),
    0 < N → 0 < k → StreamedWipe.wipeResidual N k b s < N * b → E + StreamedWipe.wipeResidual N k b s ≤ q →
    StreamedWipe.winReal prf P ≤ (2⁻¹ : ℝ≥0∞) ^ s + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

end Pous.Guarantees
