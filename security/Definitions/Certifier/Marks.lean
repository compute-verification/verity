module

public import Definitions.Certifier.Cells

/-! Marks: a mark's phase and a span's tail. -/

@[expose] public section

namespace NetTiming

/-- **A mark's phase**: the execution phase, of `T` buckets each, of the bucket `t` the network attester handles the
mark in (the ticks it has done), `⌊t/T⌋`. It is the device's code on a mark: the phase it stamps the mark with. -/
def markPhase (T t : ℕ) : ℕ := t / T

/-- **A span's tail**, in execution phases of `T` buckets: `⌈(D + ⌊(E − 1)/r⌋) / T⌉`, on a schedule of `r` cells per
bucket and a latency of `D` buckets whose connections each have at most `E` egress cells. A connection admitted at
bucket `s` sends its last cell by bucket `s + D + ⌊(E − 1)/r⌋` (`CertifierSchedule.bucket`), so the tail is that many
buckets past the admission, in phases, rounded up. -/
def spanTail (D E r T : ℕ) : ℕ := (D + (E - 1) / r + T - 1) / T

end NetTiming
