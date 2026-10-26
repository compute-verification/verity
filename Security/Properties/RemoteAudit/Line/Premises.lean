import Properties.RemoteAudit.Line.Record

/-!
# What the line of record assumes of its record (`Properties.RemoteAudit.Line`)

The code-to-model premise of `Properties.RemoteAudit.Line.LineHeld`, a named `Prop` taken as a hypothesis: what ties
the record a gate run's line reads (`Record`, through the Lean verifier's record of the session, `Record.summary`) to
the trace (`Accounting.Remote.Held.Trace`) the memory model states its guarantees over. `LineHeld` reads it, like the
model's premises, at the trace of record (`ofRecord`), whose verdict is the Lean verifier's on the record.

* `Registered`: the trace's registered blocks are those of the registration the session ran on (`Node.blocks`). The
  line reads N off the Lean verifier's record (`Summary.blocks`, which a run gives as the node's registration's), and
  the line of record credits a session only when the verifier's process held that registration's hello to the pinned
  model's store (`RemoteAudit.Store.hold`, `auditor_line.py`'s *bound*): what is left is the model's naming of its
  store, that the trace is over that registration's blocks. It closes once the trace is indexed by its registered
  blocks (`Trace N`, `Trace.blocks` gone), a change to `Definitions.Accounting.Remote.Held` that retypes every remote
  guarantee: `Definitions.Accounting.Remote.Held`, `Definitions.Accounting.Assumptions.RemoteHeld`,
  `Definitions.Accounting.Assumptions.RemoteDraw`, `Definitions.Accounting.Assumptions.RemoteSketch`,
  `Definitions.Accounting.Guarantees.RemoteHeld`, `Definitions.Accounting.Guarantees.RemoteDraw`,
  `Definitions.Accounting.Guarantees.RemoteSketch`, `Proofs.Accounting.RemoteHeld` and its `Draw`, `Line`, `Meets`,
  `OverTime`, `Sketch` and `Window`, and `Properties.RemoteAudit.Line` with its `Premises` and `Record`; with it every
  lock record that reads `Trace` changes.

The rest of what the line reads of its record is code, proved. The draw's sizes are the record's: `LineHeld` reads the
draw at `recordDraw`, window `t`'s size the receipts it has in the state the record's session reaches, and on an
accepted record those are the Lean verifier's record's (`Record.receipts`), which the line reads. Each is below
`receiptMax + 1 = 2^64 < 2^448`, so each draw's sampler word is the top bits of its stream's first SHA-512 block
(`Accounting.Remote.Draw.width`): `receive` rejects a pass that would take its window's receipts to `RECEIPT_LIMIT`
(`Properties.RemoteAudit.ReceiptsBelowLimit`). The line has one group for each of the session's windows: an accepted
session has its parameters' `windows` windows (`Properties.RemoteAudit.AcceptedMinPasses`), and the line's groups are
its record's windows (`RemoteAudit.Line.ofSummary`). The record's clock is the trace's: that each credited pass ran
inside its group's window on the record's clock is the model's own `Paced` premise read at `recordPace`.
-/

namespace Properties.RemoteAudit.Line

open Accounting.Remote.Held

/-- **The registered store**: the trace's registered blocks are the node's registration's. -/
def Registered {Ω : Type} (T : Trace Ω) (node : Node) : Prop := T.blocks = node.blocks

end Properties.RemoteAudit.Line
