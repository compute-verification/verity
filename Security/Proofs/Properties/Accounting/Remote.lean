module

public import Proofs.Properties.Accounting.Remote.Window
public import Proofs.Properties.Accounting.Remote.Draw
public import Proofs.Properties.Accounting.Remote.Plan
public import Proofs.Properties.Accounting.Remote.Session
public import Proofs.Properties.Accounting.Remote.Released
public import Proofs.Properties.Accounting.Remote.Lifetime
public import Proofs.Properties.Accounting.Remote.Pooled

/-!
# The remote bandwidth audit's verifier: its properties, proved (`Properties.Accounting.Remote`)

`Window`: an accepted window checks every sampled opening (`acceptChecksOpenings`, `acceptOpensSamples`), and the
verifier's process's chunked check is the whole window's (`chunkedIsWhole`). `Draw`: the draws
are the sampler's subset mapped to distinct receipts (`sampleIsSubset`), on `Bisect` (`bisectRight`, running ends) and
`Sort` (`Array.qsort` permutes). `Plan`: an accepted plan reads distinct blocks (`planReadsDistinct`,
`changedReadsDistinct`). `Session`: one invariant of every state a run reaches, and from it the session's statements
(`acceptedMinPasses`, `passesStepped`, `passesReadDistinct`, `appendsInOrder`, `passesChained`). `Released`: a
second invariant, from an accepted release on, and from it `releasedStayOut`. `Lifetime`: a third, on the windows
appends joined, and from it `lifetimeCapped`. `Pooled`: the pooled bound the code computes, and the W the memory line
reads, cover the draw's (`pooledCovers`, `pooledCoversSession`, `lineCoversSession`, stated in
`Properties.RemoteAudit.Pooled`).
-/

@[expose] public section
