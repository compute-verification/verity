module

public import Definitions.NetworkCertifier.Marks

/-!
# The marks' guarantees

`Definitions.NetworkCertifier.Marks`. The statements only; `Proofs.NetworkCertifier.Marks` proves them.
-/

@[expose] public section

namespace NetTiming.Guarantees

open NetTiming

/-- **What the network certifier has read by a bucket is sent within the tail.** On a schedule `s` whose lanes each
hold at most `N` buckets of egress, every cell of a connection admitted before bucket `t` is sent in an execution
phase at most `⌊t / T⌋ + spanTail s.D N T`. The network certifier reads the server's bytes only on a connection it
opened when it admitted that client's connection, so the cells of connections admitted before `t` are cut from all the
server's bytes it has read by then. -/
def HeldCutInTail : Prop :=
  ∀ {κ : Type} (s : CertifierSchedule κ) (N T t : ℕ) (c : κ) (j : ℕ),
    s.cells c ≤ s.r * N → s.adm c < t → j < s.cells c →
      s.bucket c j / T ≤ t / T + spanTail s.D N T

/-- **A mark's execution phase orders it against every bucket.** A bucket the network certifier runs after it handled
a mark is in the mark's execution phase or a later one, and a bucket it ran before is in the mark's or an earlier one.
Under `trusted-host` the network certifier's run stamps each mark with `markPhase`, so the execution phase a mark is
certified at is the one it arrived in. -/
def MarkOrders : Prop :=
  ∀ (T ticks b : ℕ), (ticks ≤ b → markPhase T ticks ≤ b / T) ∧ (b < ticks → b / T ≤ markPhase T ticks)

/-- **The span holds every cell between two marks.** With the start mark handled at `ts` buckets run and the end mark
at `te`, every cell of a connection admitted before bucket `te` that is sent at bucket `ts` or later is in an
execution phase from the start mark's to the end mark's plus the tail. The start needs no tail. -/
def SpanCovers : Prop :=
  ∀ {κ : Type} (s : CertifierSchedule κ) (N T ts te : ℕ) (c : κ) (j : ℕ),
    s.cells c ≤ s.r * N → j < s.cells c → s.adm c < te → ts ≤ s.bucket c j →
      markPhase T ts ≤ s.bucket c j / T ∧ s.bucket c j / T ≤ markPhase T te + spanTail s.D N T

end NetTiming.Guarantees
