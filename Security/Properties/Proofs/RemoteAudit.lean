import Properties.Proofs.RemoteAudit.Window
import Properties.Proofs.RemoteAudit.Draw
import Properties.Proofs.RemoteAudit.Plan
import Properties.Proofs.RemoteAudit.Session
import Properties.Proofs.RemoteAudit.Released

/-!
# The remote bandwidth audit's verifier: its properties, proved (`Properties.RemoteAudit`)

`Window`: an accepted window checks every drawn opening (`acceptChecksOpenings`, `acceptOpensDraws`). `Draw`: the draws
are the sampler's subset mapped to distinct receipts (`drawIsSubset`), on `Bisect` (`bisectRight`, running ends) and
`Sort` (`Array.qsort` permutes). `Plan`: an accepted plan reads distinct blocks (`planReadsDistinct`,
`changedReadsDistinct`). `Session`: one invariant of every state a run reaches, and from it the session's statements
(`acceptedMinPasses`, `passesStepped`, `passesReadDistinct`, `appendsInOrder`, `passesChained`). `Released`: a
second invariant, from an accepted release on, and from it `releasedStayOut`.
-/
