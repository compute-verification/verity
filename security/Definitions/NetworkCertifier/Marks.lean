module

public import Definitions.NetworkCertifier.Warden

/-!
# Marks: a mark's phase and a span's tail

Definitions only. The schedule's buckets (`Definitions.NetworkCertifier.Warden`) run in execution phases of `T` buckets
each. The network certifier signs a **mark** over a message it is handed and stamps it with the phase of the bucket it
handles it in (`markPhase`). A **span** runs from one mark, its start mark, to a later one, its end mark. At the end
mark the device may still hold egress cells of connections it admitted before it: `spanTail` is how many phases past
the end mark's the last of them is sent in, on a schedule whose connections each have at most `E` egress cells. A count
over a span is over the phases from the start mark's to the end mark's plus the tail.
-/

@[expose] public section

namespace NetTiming

/-- **A mark's phase**: the execution phase, of `T` buckets each, of the bucket `t` the network certifier handles the
mark in, `⌊t/T⌋`. It is the device's code on a mark: the phase it stamps the mark with. -/
def markPhase (T t : ℕ) : ℕ := t / T

/-- **A span's tail**, in execution phases of `T` buckets: `⌈(D + ⌊(E − 1)/r⌋) / T⌉`, on a schedule of `r` cells per
bucket and a latency of `D` buckets whose connections each have at most `E` egress cells. A connection admitted at
bucket `s` sends its last cell by bucket `s + D + ⌊(E − 1)/r⌋` (`CertifierSchedule.bucket`), so the tail is that many
buckets past the admission, in phases, rounded up. -/
def spanTail (D E r T : ℕ) : ℕ := (D + (E - 1) / r + T - 1) / T

end NetTiming
