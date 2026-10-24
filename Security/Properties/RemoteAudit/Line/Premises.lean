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
* `PooledOfRecord`: on an accepting outcome the record's wrong receipts' bound is at least the session's pooled bound
  (`Draw.Session.pooled`). The bound is `audit.pooled`'s, computed in Python; the line computing it in Lean from the
  windows' sizes and draws, with a proof that it covers `Session.pooled`, closes it.

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

/-- **The record's pooled bound**: on every accepting outcome the record's wrong receipts' bound is at least the
session's pooled bound at `δ`. -/
def PooledOfRecord {n m : ℕ} {k : Fin n → ℕ} {X : Fin n → Type} (D : Session n m k X) (T : Trace (Coins n m k X))
    (rec : Coins n m k X → Record) (δ : ℝ) : Prop :=
  ∀ ω, T.accepts ω → D.pooled δ ω ≤ (rec ω).wrong

end Properties.RemoteAudit.Line
