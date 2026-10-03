import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Pous.Protocol.Model.BandChain

/-!
# The background-rebuild budget

The two counts `Pous.Guarantees.BackgroundRebuildExcluded` compares, both in sequential `Π₂` calls: what re-encoding one
band segment from its data costs, and how long a row of the graded audit lasts (`PROTOCOL.md`, "Isolation over the
whole graded window"). Added on 3 Oct 2026, with that guarantee.
-/

namespace Pous.Rebuild

open Pous.Sponge Pous.BandChain

/-- **The cost of re-encoding one band segment from `W`**: `R(d) = Σ_{v < B} (min(v, d) + 1)`. Label `v` of
`chainBand B d` absorbs its parents newest first and then the pad, one `Π₂` call per absorbed block
(`Pous.BandChain.absorbedG`), so it costs `(parentsDesc (bandDAG B d) v).length + 1` calls. -/
def rebuildCalls (B d : ℕ) : ℕ := ∑ v : Fin B, ((parentsDesc (bandDAG B d) v).length + 1)

/-- **A graded row's window**: `k` rounds of which at most `t` are late, `k − t` within `D` sequential calls each and `t`
within `D_late`. -/
def gradedWindow (D D_late k t : ℕ) : ℕ := (k - t) * D + t * D_late

end Pous.Rebuild
