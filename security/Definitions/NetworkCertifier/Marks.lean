module

public import Definitions.NetworkCertifier.Warden

/-!
# Marks: the execution phase of a mark, and a span's tail

Definitions only. The network certifier's run is read in execution phases of `T` buckets. A mark is a request the
network certifier certifies at the execution phase it handles it in, and a span runs from one mark to a later one. What
the network certifier has read from the server by the later mark it sends in cells of connections it had admitted by
then, the last of them within the span's tail.

`markPhase` is the network certifier's code on a mark. `spanTail` is the tail, in execution phases, on a schedule whose
lanes each hold at most `N` buckets of egress. `Definitions.NetworkCertifier.Guarantees.Marks` states what holds of
them.
-/

@[expose] public section

namespace NetTiming

/-- **The execution phase of a mark.** The network certifier's code stamps a mark request it handles with `ticks`
buckets run, its buckets `0 … ticks − 1`, with `ticks / T`: the execution phase, of `T` buckets, of the next bucket it
runs. It reads no clock and nothing of the request. -/
def markPhase (T ticks : ℕ) : ℕ := ticks / T

/-- **A span's tail**, in execution phases of `T` buckets: `⌈(D + N − 1) / T⌉`, `N` the most buckets of egress a lane
holds. A connection's cell `j` is sent `D + ⌊j / r⌋` buckets after its admission, so its last cell is sent at most
`D + N − 1` buckets after it. On a schedule whose lanes hold at most `egress_cells` cells, `N` is
`⌊(egress_cells − 1) / r⌋ + 1` and the tail is `⌈(D + ⌊(egress_cells − 1) / r⌋) / T⌉`. -/
def spanTail (D N T : ℕ) : ℕ := (D + N - 1 + T - 1) / T

end NetTiming
