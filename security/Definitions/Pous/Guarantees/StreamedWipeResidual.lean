module

public import Definitions.Pous.Model.StreamedWipe
public import Definitions.Pous.Model.StreamedWipeResidual

/-!
# The streamed wipe's residual as a function of the fill and the challenge count

`Pous.Guarantees.StreamedWipe` at `g = N − D`, with the error split in two halves of `2^-(s+1)`, as
`StreamedWipeMeets2600GB` splits it, and every number left a variable: the wipe residual `wipeResidual N k b s`
(`Definitions.Pous.Model.StreamedWipeResidual`), in
bits, is the prior content an accepted prover can keep beyond the egress, at soundness error `2^-s` plus the PRF's
advantage, for a wipe of `N` labels of `b` bits whose fill is the whole memory (`M = N · b`) and `k` sequential
challenges. `N` and `b` come from the hardware description, and `k` is the wipe's challenge count, which the wipe's
check outputs. The game is `security/Definitions/Pous/Model/StreamedWipe.lean`, the proof
`security/Proofs/Pous/StreamedWipe/Residual.lean`.

**What it rests on.** No named assumption, as `StreamedWipe`.
-/

@[expose] public section

namespace Pous.Guarantees

open scoped ENNReal

/-- **The streamed wipe at any fill and challenge count: `E + wipeResidual N k b s` bits at `2^-s`.** For a wipe of
`N ≥ 1` labels of `b` bits whose fill is the whole memory (`M = N · b`), `k ≥ 1` sequential challenges and a soundness
exponent `s`, where the wipe residual is covered (`StreamedWipe.Covers N k b s q E`: the residual below the memory,
`wipeResidual N k b s < N · b`, at or above which the bound says nothing, and the `q` bits of uniform prior content
kept at least `E + wipeResidual N k b s`): a prover with `E` bits of egress that keeps those `q` bits through the wipe
is accepted with probability at most `2^-s` plus the PRF's advantage.

Same footing as `StreamedWipe`: no named assumption. Proof: `PousStreamedWipe.streamed_wipe_residual`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeResidual : Prop :=
  ∀ (N k b s q E κ : ℕ) (prf : Bits κ → Fin N → Bits b) (P : StreamedWipe.Prover N k b q (N * b) E),
    0 < N → 0 < k → StreamedWipe.Covers N k b s q E →
    StreamedWipe.winReal prf P ≤ (2⁻¹ : ℝ≥0∞) ^ s + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

end Pous.Guarantees
