import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Definitions.Pous.Model.BandChain

/-!
# The background-rebuild budget

The counts `Pous.Guarantees.BackgroundRebuildExcluded` reads, all in sequential `Π₂` calls of at least `call_ns` each:
what re-encoding one band segment from its data costs, how long a row of the graded audit's rounds may last, and the cap
on a graded audit's whole duration that the guarantee requires the memory challenger to enforce (`PROTOCOL.md`,
"Isolation over the whole graded window"). A call of at least `call_ns` is the ideal-permutation model's unit; at the
reference Feistel-SHAKE256 `Π₂` a prover answers from 1.53% of `C` in 481 Keccak-f without completing a call.
-/

namespace Pous.Rebuild

open Pous.Sponge Pous.BandChain

/-- **The cost of re-encoding one band segment from `W`**: `R(d) = Σ_{v < B} (min(v, d) + 1)`. Label `v` of
`chainBand B d` absorbs its parents newest first and then the pad, one `Π₂` call per absorbed block
(`Pous.BandChain.absorbedG`), so it costs `(parentsDesc (bandDAG B d) v).length + 1` calls. -/
def rebuildCalls (B d : ℕ) : ℕ := ∑ v : Fin B, ((parentsDesc (bandDAG B d) v).length + 1)

/-- **How long a graded row's rounds may last, in call-times**: `k` rounds of which at most `t` are late. A round runs
from its challenge's send to its answer's receipt, and `D = ⌊(Δ + RTT) / call_ns⌋` (`effective_rounds`) admits any
span under `(D + 1)·call_ns`; a late round's is under `(D_late + 1)·call_ns`. The memory challenger's gaps between
rounds are not counted. -/
def gradedRoundSpan (D D_late k t : ℕ) : ℕ := (k - t) * (D + 1) + t * (D_late + 1)

/-- **The cap on a graded audit's duration, in call-times**, which the guarantee requires the memory challenger to
enforce: it aborts, without accepting, an audit that lasts more than `auditCap · call_ns` on its own clock, from its
first challenge's send to its last answer's receipt. A rebuild running beside the audit, through its rounds and its
gaps, then completes at most `auditCap` sequential calls. -/
def auditCap : ℕ := 1505

end Pous.Rebuild
