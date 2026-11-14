module

public import Proofs.Properties.Compose.Remote.Window
public import Proofs.Properties.Compose.Remote.Sample
public import Proofs.Properties.Compose.Remote.Plan
public import Proofs.Properties.Compose.Remote.Run
public import Proofs.Properties.Compose.Remote.Released
public import Proofs.Properties.Compose.Remote.Lifetime
public import Proofs.Properties.Compose.Remote.Pooled

/-!
# PoUB's verifier: its properties, proved (`Properties.Compose.Remote`)

`Window`: an accepted window checks every sampled opening (`acceptChecksOpenings`, `acceptOpensSamples`), and the
verifier's process's chunked check is the whole window's (`chunkedIsWhole`). `Draw`: the samples
are the sampler's subset mapped to distinct receipts (`sampleIsSubset`), on `Bisect` (`bisectRight`, running ends) and
`Sort` (`Array.qsort` permutes). `Plan`: an accepted plan reads distinct blocks (`planReadsDistinct`,
`changedReadsDistinct`). `Session`: one invariant of every state a run reaches, and from it the session's statements
(`acceptedMinPasses`, `passesStepped`, `passesReadDistinct`, `appendsInOrder`, `passesChained`). `Released`: a
second invariant, from an accepted release on, and from it `releasedStayOut`. `Lifetime`: a third, on the windows
appends joined, and from it `lifetimeCapped`. `Pooled`: the pooled bound the code computes, and the W the memory line
reads, cover the sample's (`pooledCovers`, `pooledCoversRun`, `lineCoversRun`, stated in
`Properties.Poub.Pooled`).
-/

@[expose] public section
