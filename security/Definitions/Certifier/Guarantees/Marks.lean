module

public import Definitions.Certifier.Marks

/-! The marks' lines of the network attester's model, about the schedule's arithmetic, with no premise. -/

@[expose] public section

namespace NetTiming.Guarantees

open NetTiming

/-- The cells of the connections admitted before `t` are all of the server's bytes the device reads by `t`: its code reads
them only on a connection it opened when it admitted the client's. -/
def HeldCutInTail : Prop :=
  ∀ {κ : Type} (s : CertifierSchedule κ) (E T t : ℕ) (c : κ) (j : ℕ),
    s.cells c ≤ E → s.adm c < t → j < s.cells c →
      s.bucket c j / T ≤ t / T + spanTail s.D E s.r T

def MarkOrders : Prop :=
  ∀ (T t b : ℕ), (t ≤ b → markPhase T t ≤ b / T) ∧ (b < t → b / T ≤ markPhase T t)

/-- **Every cell sent between two marks is in a phase from the start mark's to the end mark's plus the tail**
(`HeldCutInTail` and `MarkOrders` together). On a schedule `s` whose connections each have at most `E` egress cells, in
execution phases of `T` buckets, for a start mark handled in bucket `t₀` and an end mark handled in bucket `t₁`: every
egress cell `j` of a connection `c` admitted before `t₁` that is sent at or after `t₀` is sent in a phase from
`markPhase T t₀` to `markPhase T t₁ + spanTail`. So a count over those phases is over every cell the device sends from
the start mark on out of what it read by the end mark. -/
def SpanCovers : Prop :=
  ∀ {κ : Type} (s : CertifierSchedule κ) (E T t₀ t₁ : ℕ) (c : κ) (j : ℕ),
    s.cells c ≤ E → j < s.cells c → s.adm c < t₁ → t₀ ≤ s.bucket c j →
      markPhase T t₀ ≤ s.bucket c j / T ∧ s.bucket c j / T ≤ markPhase T t₁ + spanTail s.D E s.r T

end NetTiming.Guarantees
