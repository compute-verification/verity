import Proofs.Properties.PoubCheck.Window
import Proofs.Properties.PoubCheck.Draw
import Proofs.Properties.PoubCheck.Plan
import Proofs.Properties.PoubCheck.Session
import Proofs.Properties.PoubCheck.Released
import Proofs.Properties.PoubCheck.Lifetime
import Proofs.Properties.PoubCheck.Pooled

/-!
# PoUB's verifier: its properties, proved (`Properties.PoubCheck`)

`Window`: an accepted window checks every drawn opening (`acceptChecksOpenings`, `acceptOpensDraws`), and the
verifier's process's chunked check is the whole window's (`chunkedIsWhole`). `Draw`: the draws
are the sampler's subset mapped to distinct receipts (`drawIsSubset`), on `Bisect` (`bisectRight`, running ends) and
`Sort` (`Array.qsort` permutes). `Plan`: an accepted plan reads distinct blocks (`planReadsDistinct`,
`changedReadsDistinct`). `Session`: one invariant of every state a run reaches, and from it the session's statements
(`acceptedMinPasses`, `passesStepped`, `passesReadDistinct`, `appendsInOrder`, `passesChained`). `Released`: a
second invariant, from an accepted release on, and from it `releasedStayOut`. `Lifetime`: a third, on the windows
appends joined, and from it `lifetimeCapped`. `Pooled`: the pooled bound the code computes, and the W the memory line
reads, cover the draw's (`pooledCovers`, `pooledCoversSession`, `lineCoversSession`, stated in
`Properties.PoubCheck.Pooled`).
-/
