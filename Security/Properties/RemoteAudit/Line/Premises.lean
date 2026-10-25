import Definitions.Accounting.Remote.Draw
import Properties.RemoteAudit.Line.Record

/-!
# What the line of record assumes of its record (`Properties.RemoteAudit.Line`)

The code-to-model premises of `Properties.RemoteAudit.Line.LineHeld`, each a named `Prop` taken as a hypothesis: what
ties the record a gate run's line reads (`Record`) to the trace (`Accounting.Remote.Held.Trace`) and the opening draw
(`Accounting.Remote.Draw.Session`) the memory model states its guarantees over. Each is a gap the code does not close
yet, and says which code would close it. `LineHeld` reads them, like the model's premises, at the trace of record
(`ofRecord`), whose verdict is the Lean verifier's on the record, so on every outcome where `Verified` holds.

* `Registered`: the trace's registered blocks are the node's.
* `WindowsOfRecord`: on an accepted record the line has one group for each of the session's windows. The groups are
  built by the line's Python glue (`benchmarks/remote_session/auditor_line.py`'s `auditor_seen` and `verifier.py`'s
  `paced_groups`: one group a window of `verifier.json`'s `windows_seen`), not by Lean. The Lean verifier's half is proved: an accepted session has its
  parameters' `windows` windows (`Properties.RemoteAudit.AcceptedMinPasses`). It closes once the line's groups are
  indexed by the windows of the record the Lean verifier returns (its `outcome` request's `record`,
  `Session.State.summary`), with the auditor's times beside each, and `Node.windows` is the session's `params.windows`.
* `DrawsOfRecord`: on an accepted record the windows the line computes its pooled bound from
  (`RemoteAudit.Pooled.pooled`) are the session's, each window's receipts and its draws (`Draw.Session.size`,
  `Draw.Session.draws`). The line reads them through Python: `verifier.py`'s `recorded_windows` takes window `t` as
  `(Σ lengths_t, profile.drawn.receipt)` of `verifier.json`'s record, whose profile `audit.Record.profile`
  (`window_profile`) computes from the record. It closes once the line reads window `t` as the Lean record's
  `(Σ lengths_t, drawn_t)` (`Session.WindowSummary`) and the draw's `size t` is that window's receipts:
  `Properties.RemoteAudit.DrawIsSubset` gives the code's draws as `min(s, U)` of them, the model's
  `Draw.Session.draws`. With it, `Properties.RemoteAudit.LineCoversSession` gives the line's W at least the session's
  pooled bound.

The record's clock is the trace's: that each credited pass ran inside its group's window on the record's clock is the
model's own `Paced` premise read at `recordPace`.
-/

namespace Properties.RemoteAudit.Line

open Accounting.Remote.Held Accounting.Remote.Draw

/-- **The registered store**: the trace's registered blocks are the node's. -/
def Registered {Ω : Type} (T : Trace Ω) (node : Node) : Prop := T.blocks = node.blocks

/-- **One group a window**: on every accepting outcome the record has the node's `windows` groups. -/
def WindowsOfRecord {Ω : Type} (T : Trace Ω) (node : Node) (rec : Ω → Record) : Prop :=
  ∀ ω, T.accepts ω → (rec ω).groups.size = node.windows

/-- **The record's windows are the session's**: on every accepting outcome the windows the line computes its pooled
bound from are the session's, each window's receipts and its draws. -/
def DrawsOfRecord {n m : ℕ} {k : Fin n → ℕ} {X : Fin n → Type} (D : Session n m k X) (T : Trace (Coins n m k X))
    (rec : Coins n m k X → Record) : Prop :=
  ∀ ω, T.accepts ω → (rec ω).windows = Array.ofFn fun t => ((D.size t ω : ℕ), D.draws t ω)

end Properties.RemoteAudit.Line
