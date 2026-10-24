import Proofs.Properties.Accounting.Remote.Window
import Proofs.Properties.Accounting.Remote.Draw
import Proofs.Properties.Accounting.Remote.Plan
import Proofs.Properties.Accounting.Remote.Session
import Proofs.Properties.Accounting.Remote.Released
import Proofs.Properties.Accounting.Remote.Lifetime

/-!
# The remote bandwidth audit's verifier: its properties, proved (`Properties.Accounting.Remote`)

`Window`: an accepted window checks every drawn opening (`acceptChecksOpenings`, `acceptOpensDraws`). `Draw`: the draws
are the sampler's subset mapped to distinct receipts (`drawIsSubset`), on `Bisect` (`bisectRight`, running ends) and
`Sort` (`Array.qsort` permutes). `Plan`: an accepted plan reads distinct blocks (`planReadsDistinct`,
`changedReadsDistinct`). `Session`: one invariant of every state a run reaches, and from it the session's statements
(`acceptedMinPasses`, `passesStepped`, `passesReadDistinct`, `appendsInOrder`, `passesChained`). `Released`: a
second invariant, from an accepted release on, and from it `releasedStayOut`. `Lifetime`: a third, on the windows
appends joined, and from it `lifetimeCapped`.
-/
