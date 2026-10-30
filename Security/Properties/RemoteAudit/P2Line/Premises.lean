import Properties.RemoteAudit.P2Line

/-!
# What ties PoUB-P2's line of record to the period's bandwidth claim (`Properties.RemoteAudit.P2Line`)

The line at the subprotocol (`Properties.RemoteAudit.P2Subprotocol.LineHeldAtSubprotocol`) bounds the probability that
the P2 subprotocol accepts while the record's credit claims more bits than its passes read on the node
(`P2Subprotocol.ReadBy`). The period's bandwidth claim (`Accounting.PeriodRun.Holds` at `.poub`) is `Q ≤ read`: the bits
the certificate claims at most the bits the period's schedule has its bandwidth account read from device memory over
the window (`Accounting.PeriodRun.read`). Two named premises join them, each at a record the verifier keeps
(`Replay.summary`):

* `ClaimWithinCredit`: the certificate claims no more than the credit, `Q ≤ 8·q_bytes` of the record's credit
  (`P2Session.credit`);
* `PassesWithinSchedule`: the schedule reads at least what the record's passes read on the node, `8·UNIT_BYTES` bits
  for each unit its `P` passes read (`Trace.reads`, `passes`).

Where the subprotocol accepts and both hold, the credit's bits sit between the claim and the schedule's, so the line's
event is the claim's failure.
-/

namespace Properties.RemoteAudit.P2Line

open Accounting.Remote.Held

/-- **The certificate claims no more than the credit**: where the verifier records the session as `sm`, the bits the
certificate claims, `Q`, are at most `8·q_bytes` of `sm`'s credit. -/
def ClaimWithinCredit (Q : ℝ) (node : Audited) (r : Replay) : Prop :=
  ∀ sm, r.summary node = some sm → Q ≤ 8 * (RemoteAudit.P2Session.credit sm).qBytes

/-- **The schedule reads what the passes read**: where the verifier records the session as `sm`, the bits of the blocks
its `P` passes read on the trace at `ω`, `UNIT_BYTES` bytes a block, are at most `read`, the bits the period's schedule
has the bandwidth account read from device memory over its window. -/
def PassesWithinSchedule {Ω : Type} (read : ℕ) (T : Trace Ω) (ω : Ω) (node : Audited) (r : Replay) : Prop :=
  ∀ sm, r.summary node = some sm → (8 * RemoteAudit.P2.UNIT_BYTES : ℝ) * T.reads ω (passes sm) ≤ read

end Properties.RemoteAudit.P2Line
