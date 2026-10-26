import Properties.RemoteAudit.Line.Record

/-!
# What the line of record reads of its record (`Properties.RemoteAudit.Line`)

What ties the record a gate run's line reads (`Record`, through the Lean verifier's record of the session,
`Record.summary`) to the trace (`Accounting.Remote.Held.Trace`) the memory model states its guarantees over. Nothing of
it is a premise: `LineHeld` reads the model's premises at the trace of record (`ofRecord`), whose verdict is the Lean
verifier's on the record.

* `Registered`: the trace's registered blocks are those of the registration the session ran on (`Node.blocks`). It
  holds of the trace of record, which is over the registration's blocks (`ofRecord`, `overBlocks`;
  `registered_ofRecord`): a trace's registered blocks are the model's name for the store, which the verifier can't see,
  and the model's store is the registration's. What the verifier sees, it checks: the line reads N off the Lean
  verifier's record (`Summary.blocks`, the registration's on an accepted run, `record_summary`), and the verifier's
  `line_record` refuses a record given bare that is not of the registration it is given, its digest and its registered
  blocks (`RemoteAudit.Line.lineOfRegistered`, `Properties.RemoteAudit.Line.LineRegistered`).

The rest of what the line reads of its record is code, proved, at any windows a draw g (`Node.groupsPerCoin`). The
draw's sizes are the record's: `LineHeld` reads the draw at `recordDraw`, one draw for each of the session's `⌈W/g⌉`
(`Node.draws`), draw `c`'s size the receipts its windows have in the state the record's session reaches, and on an
accepted record those are the Lean verifier's record's (`Record.receipts`, `drawReceipts`), which the line reads. Each
is below `receiptMax + 1 = 2^64 < 2^448`, so each draw's sampler word is the top bits of its stream's first SHA-512
block (`Accounting.Remote.Draw.width`): `receive` rejects a pass that would take its window's receipts to
`RECEIPT_LIMIT` (`Properties.RemoteAudit.ReceiptsBelowLimit`), which bounds a draw at g = 1, and above it the line
refuses a draw whose windows' receipts sum to `RECEIPT_LIMIT` or more (`RemoteAudit.Line.ofSummaryAt`). The line has
one group for each of the session's windows: an accepted session has its parameters' `windows` windows
(`Session.State.outcome`, its parameters kept by `Shared.run` at every g), and the line's groups are its record's
windows (`RemoteAudit.Line.ofSummaryAt`). The record's clock is the trace's: that each credited pass ran inside its
group's window on the record's clock is the model's own `Paced` premise read at `recordPace`.
-/

namespace Properties.RemoteAudit.Line

open Accounting.Remote.Held

/-- **The registered store**: the trace's registered blocks are the node's registration's. -/
def Registered {Ω : Type} (T : Trace Ω) (node : Node) : Prop := T.blocks = node.blocks

end Properties.RemoteAudit.Line
