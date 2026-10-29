import Definitions.Pouw.Guarantees.Lead

/-!
# The gap at the three points, and the nontrivial lemma (STATEMENT; owner N-PREM)

`gapAt` and `gapAt320` evaluate `Pouw.Lead.gapShare` at T 512 × 64, T 512 × 512 and T 8,192 for every call of
`n, K ≥ 4,096`, at caps 32 and 320; the worst call is at `K = 4,113`, the first `K` past 4,096 that adds a step.
`nontrivial` is the guarantee's nontrivial lemma.
-/

namespace Pouw.SecurityProofs.Lead.Statement

open Pouw.Guarantees.Lead

/-- Owner: N-PREM. The gap's share of stock at the three points, cap 32: rational arithmetic over `n, K ≥ 4,096`, with
`stepsFor K ≤ (K + 32) / 28` and `wordChunks ≤ m n S / 65,536 + 1`. -/
theorem gapAt : GapAt := by
  sorry

/-- Owner: N-PREM. The gap's share of stock at the three points, cap 320. -/
theorem gapAt320 : GapAt320 := by
  sorry

/-- Owner: N-PREM. The nontrivial lemma: the honest work exceeds the gap at the three points, and the empty strategy
does no work and hashes nothing. -/
theorem nontrivial : Nontrivial := by
  sorry

end Pouw.SecurityProofs.Lead.Statement
