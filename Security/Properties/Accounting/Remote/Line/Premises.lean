import Definitions.Accounting.Remote.Draw
import Properties.Accounting.Remote.Line.Record

/-!
# What the line of record assumes of its record (`Properties.RemoteAudit.Line`)

The code-to-model premises of `Properties.RemoteAudit.Line.LineHeld`, each a named `Prop` taken as a hypothesis: what
ties the record a gate run's line reads (`Record`, through the Lean verifier's record of the session,
`Record.summary`) to the trace (`Accounting.Remote.Held.Trace`) and the opening draw (`Accounting.Remote.Draw.Session`)
the memory model states its guarantees over. `LineHeld` reads them, like the model's premises, at the trace of record
(`ofRecord`), whose verdict is the Lean verifier's on the record, so on every outcome where `Verified` holds. Each is an
equality between a model object and the record, and says what would close it.

* `Registered`: the trace's registered blocks are those of the registration the session ran on (`Node.blocks`). The
  line reads N off the Lean verifier's record (`Summary.blocks`, which a run gives as the node's registration's), and
  the line of record credits a session only when the verifier's process held that registration's hello to the pinned
  model's store (`RemoteAudit.Store.hold`, `auditor_line.py`'s *bound*): what is left is the model's naming of its
  store, that the trace is over that registration's blocks. It closes once the trace is indexed by the registration
  (`Trace.blocks` read off it), a change to `Definitions`.
* `SizesOfRecord`: on an accepted record each window's receipts in the Lean verifier's record (`Σ lengths_t`,
  `Session.WindowSummary`) are the draw's `size t`. The draws need no premise: the line reads window `t`'s from the
  record and refuses one that isn't `min(s, U_t)` at the session's `samples` (`RemoteAudit.Line.ofSummary`), the
  model's `Draw.Session.draws` at `s = Node.params.samples`, so `Properties.RemoteAudit.LineCoversSession` gives the
  line's W at least the session's pooled bound. It closes once the draw's sizes are the record's (`size t ω` the record
  at `ω`'s, which needs a bound `m` on an accepted window's receipts and `Live` read at those sizes).
* `DrawFits`: the session's windows have fewer than `2^448` receipts (`m`, the draw's bound on `U_t`), so each draw's
  sampler word is the top bits of its stream's first SHA-512 block (`Accounting.Remote.Draw.width`). No code bounds a
  window's receipts yet; gate run 7's windows have 130,023,424.

That the line has one group for each of the session's windows is no premise: an accepted session has its parameters'
`windows` windows (`Properties.RemoteAudit.AcceptedMinPasses`), and the line's groups are its record's windows
(`RemoteAudit.Line.ofSummary`). The record's clock is the trace's: that each credited pass ran inside its group's window
on the record's clock is the model's own `Paced` premise read at `recordPace`.
-/

namespace Properties.RemoteAudit.Line

open Accounting.Remote.Held Accounting.Remote.Draw

/-- **The registered store**: the trace's registered blocks are the node's registration's. -/
def Registered {Ω : Type} (T : Trace Ω) (node : Node) : Prop := T.blocks = node.blocks

/-- **The record's windows are the draw's**: on every accepting outcome each window's receipts in the Lean verifier's
record are the draw's size of that window. -/
def SizesOfRecord {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Session n m s κ X) (T : Trace (Coins n κ X))
    (node : Node) (rec : Coins n κ X → Record) : Prop :=
  ∀ ω, T.accepts ω → (rec ω).receipts node = Array.ofFn fun t => (D.size t ω : ℕ)

/-- **The draw's word fits a block**: a window has fewer than `2^448` receipts, so each draw's sampler word lies in its
stream's first SHA-512 block. -/
def DrawFits (m : ℕ) : Prop := m < 2 ^ 448

end Properties.RemoteAudit.Line
