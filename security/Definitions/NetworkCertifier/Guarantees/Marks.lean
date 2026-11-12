module

public import Definitions.NetworkCertifier.Marks

/-!
# The marks' lines of the device model

`Definitions.NetworkCertifier.Marks`. The statements only: two lines of the network certifier's model, and what the two
give for a span between two marks. Each is about the schedule's arithmetic and takes no premise. Under `trusted-host`
the network certifier's run is its code, so each holds of the cells it sends and of the phases it stamps its marks with.
-/

@[expose] public section

namespace NetTiming.Guarantees

open NetTiming

/-- **What the network certifier holds uncut at a bucket is cut within the tail.** On a schedule `s` whose connections
each have at most `E` egress cells, in execution phases of `T` buckets: every egress cell `j` of a connection `c`
admitted before bucket `t` is sent in a phase at most `⌊t/T⌋ + spanTail`. The cells of the connections admitted before
`t` are all of the server's bytes the device holds at `t`: its code reads the server's bytes only on a connection it
opened when it admitted the client's (`warden.py`). -/
def HeldCutInTail : Prop :=
  ∀ {κ : Type} (s : CertifierSchedule κ) (E T t : ℕ) (c : κ) (j : ℕ),
    s.cells c ≤ E → s.adm c < t → j < s.cells c →
      s.bucket c j / T ≤ t / T + spanTail s.D E s.r T

/-- **A mark's phase orders it against every bucket.** For a mark handled in bucket `t` and a bucket `b`, in execution
phases of `T` buckets: a bucket at or after the mark's is in the mark's phase or a later one, and a bucket before the
mark's is in the mark's phase or an earlier one. -/
def MarkOrders : Prop :=
  ∀ (T t b : ℕ), (t ≤ b → markPhase T t ≤ b / T) ∧ (b < t → b / T ≤ markPhase T t)

/-- **Every cell sent between two marks is in a phase from the start mark's to the end mark's plus the tail**
(`HeldCutInTail` and `MarkOrders` together). On a schedule `s` whose connections each have at most `E` egress cells, in
execution phases of `T` buckets, for a start mark handled in bucket `t₀` and an end mark handled in bucket `t₁`: every
egress cell `j` of a connection `c` admitted before `t₁` that is sent at or after `t₀` is sent in a phase from
`markPhase T t₀` to `markPhase T t₁ + spanTail`. So a count over those phases is over every cell the device sends from
the start mark on out of what it held by the end mark. -/
def SpanCovers : Prop :=
  ∀ {κ : Type} (s : CertifierSchedule κ) (E T t₀ t₁ : ℕ) (c : κ) (j : ℕ),
    s.cells c ≤ E → j < s.cells c → s.adm c < t₁ → t₀ ≤ s.bucket c j →
      markPhase T t₀ ≤ s.bucket c j / T ∧ s.bucket c j / T ≤ markPhase T t₁ + spanTail s.D E s.r T

end NetTiming.Guarantees
