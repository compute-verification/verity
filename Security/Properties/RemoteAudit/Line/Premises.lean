import Definitions.Accounting.Remote.Draw
import Properties.RemoteAudit.Line.Record

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
* `SizesOfRecord`: on an accepted record each draw's receipts in the Lean verifier's record (`Record.receipts`: a
  window's `Σ lengths_t`, `Session.WindowSummary`, at one window a coin, and its `g` windows' summed at
  `Node.groupsPerCoin` g, `drawReceipts`) are the draw's `size c`, so the model's draws are the session's ⌈W/g⌉ draws.
  The draws need no premise: the line reads draw `c`'s from the record and refuses one that isn't `min(s, U_c)` at the
  session's `samples` (`RemoteAudit.Line.ofSummaryAt`), the model's `Draw.Session.draws` at `s = Node.params.samples`,
  so `Properties.RemoteAudit.LineCoversSession` gives the line's W at least the session's pooled bound. It closes once
  the draw's sizes are the record's (`size c ω` the record at `ω`'s, which needs a bound `m` on an accepted draw's
  receipts and `Live` read at those sizes).
* `DrawFits`: the session's draws have fewer than `2^448` receipts (`m`, the draw's bound on `U_c`), so each draw's
  sampler word is the top bits of its stream's first SHA-512 block (`Accounting.Remote.Draw.width`). No code bounds a
  draw's receipts yet; gate run 7's windows have 130,023,424, and a draw of g windows has up to g times a window's.

That the line has one group for each of the session's windows is no premise: an accepted session has its parameters'
`windows` windows (`Properties.RemoteAudit.AcceptedMinPasses` at one window a coin, and at every g the invariant
`Shared.step` keeps, `Properties.RemoteAudit.Shared.sstep`), and the line's groups are its record's windows
(`RemoteAudit.Line.ofSummaryAt`). The record's clock is the trace's: that each credited pass ran inside its group's
window on the record's clock is the model's own `Paced` premise read at `recordPace`.
-/

namespace Properties.RemoteAudit.Line

open Accounting.Remote.Held Accounting.Remote.Draw

/-- **The registered store**: the trace's registered blocks are the node's registration's. -/
def Registered {Ω : Type} (T : Trace Ω) (node : Node) : Prop := T.blocks = node.blocks

/-- **The record's draws are the draw's**: on every accepting outcome each draw's receipts in the Lean verifier's
record (a window's at one window a coin) are the draw's size of that draw. -/
def SizesOfRecord {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} (D : Session n m s κ X) (T : Trace (Coins n κ X))
    (node : Node) (rec : Coins n κ X → Record) : Prop :=
  ∀ ω, T.accepts ω → (rec ω).receipts node = Array.ofFn fun t => (D.size t ω : ℕ)

/-- **The draw's word fits a block**: a draw has fewer than `2^448` receipts, so each draw's sampler word lies in its
stream's first SHA-512 block. -/
def DrawFits (m : ℕ) : Prop := m < 2 ^ 448

end Properties.RemoteAudit.Line
