import Definitions.Accounting.Remote.Draw
import Properties.RemoteAudit.Line.Record

/-!
# What the line of record assumes of its record (`Properties.RemoteAudit.Line`)

The code-to-model premises of `Properties.RemoteAudit.Line.LineHeld`, each a named `Prop` taken as a hypothesis: what
ties the record a gate run's line reads (`Record`) to the trace (`Accounting.Remote.Held.Trace`) and the opening draw
(`Accounting.Remote.Draw.Session`) the memory model states its guarantees over. Each is a gap the code does not close
yet, and says which code would close it.

* `VerdictOfRecord`: the trace accepts wherever the Lean verifier accepts the record's session. The model's verdict is
  the verifier's.
* `Registered`: the trace's registered blocks are the node's.
* `WindowsOfRecord`: on an accepting outcome the record has one group for each of the session's windows. The groups
  are built from the verifier's summary by the line's Python glue (`benchmarks/remote_session/auditor_line.py`), not by
  Lean.
* `DrawsOfRecord`: on an accepting outcome the windows the line computes its pooled bound from
  (`RemoteAudit.Pooled.pooled`) are the session's, each window's receipts and its draws (`Draw.Session.size`,
  `Draw.Session.draws`). The record's windows are the verifier's record's (`audit.Record`), read by Python. With it,
  `Properties.RemoteAudit.LineCoversSession` gives the line's W at least the session's pooled bound.
* `DeltaOfNode`: the δ the line reads its pooled bound at is nonnegative, as a probability's bound is.

The record's clock is the trace's: that each credited pass ran inside its group's window on the record's clock is the
model's own `Paced` premise read at `recordPace`.
-/

namespace Properties.RemoteAudit.Line

open Accounting.Remote.Held Accounting.Remote.Draw

/-- **The verdict of record**: the trace accepts on every outcome whose record's session the Lean verifier accepts. -/
def VerdictOfRecord {Ω : Type} (T : Trace Ω) (rec : Ω → Record) : Prop := ∀ ω, Verified (rec ω) → T.accepts ω

/-- **The registered store**: the trace's registered blocks are the node's. -/
def Registered {Ω : Type} (T : Trace Ω) (node : Node) : Prop := T.blocks = node.blocks

/-- **One group a window**: on every accepting outcome the record has the node's `windows` groups. -/
def WindowsOfRecord {Ω : Type} (T : Trace Ω) (node : Node) (rec : Ω → Record) : Prop :=
  ∀ ω, T.accepts ω → (rec ω).groups.size = node.windows

/-- **The record's windows are the session's**: on every accepting outcome the windows the line computes its pooled
bound from are the session's, each window's receipts and its draws. -/
def DrawsOfRecord {n m : ℕ} {k : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Session n m k κ X) (T : Trace (Coins n κ X))
    (rec : Coins n κ X → Record) : Prop :=
  ∀ ω, T.accepts ω → (rec ω).windows = Array.ofFn fun t => ((D.size t ω : ℕ), D.draws t ω)

/-- **The line's δ**: the δ the line reads its pooled bound at is nonnegative. -/
def DeltaOfNode (node : Node) : Prop := 0 ≤ node.delta

end Properties.RemoteAudit.Line
