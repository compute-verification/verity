import Properties.RemoteAudit.Line.Record

/-!
# What the line of record assumes of its record and its draw (`Properties.RemoteAudit.Line`)

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

Two more premises are not `LineHeld`'s, which keeps their terms as numbers in its bound, but
`Properties.RemoteAudit.Component.LineHeldAtComponent`'s, which takes them so that its bound's only numbers past the
sketch's and the binding's are the live sample's (`δ + Σ_c s/2^64`):

* `OpeningsPrf` (`prf/sha-512`): each draw's PRF advantage at its window test is nothing, SHA-512's stream taken as
  an ideal PRF at this use.
* `CoinsHidden` (`statistical-hiding`, hm96): each draw's hiding term is nothing, the coin commitment's leaves taken
  as hiding their coins outright.
-/

namespace Properties.RemoteAudit.Line

open Accounting.Remote.Held

/-- **The registered store**: the trace's registered blocks are the node's registration's. -/
def Registered {Ω : Type} (T : Trace Ω) (node : Node) : Prop := T.blocks = node.blocks

/-- **The openings stream is a PRF at the window test** (`prf/sha-512`): for every draw `t`, the first blocks SHA-512
expands draw `t`'s openings coin to (`prf t`, `verity.ml.randomness`'s keyed stream) make window `t`'s distinguisher
(`Session.distinguisher` of the session played at the commitment with coin `t`'s leaf committing zeros, at `δ`) true no
more often than uniform first blocks do: its PRF advantage is nothing. It is the random-oracle step (`random-oracle`)
at this one use, a fixed predicate of the coins and first blocks with no oracle access, the ideal PRF taken for
SHA-512's stream; `LineHeld` keeps the advantage as its term `Σ_t Adv(prf t)`, which `ledger.PRF_CLAIM` names with no
value. It is false of a prover whose correct receipts follow the stream's structure: one that reads, from the openings
coin, which receipts SHA-512 will draw. -/
def OpeningsPrf {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X Cm : Type} [Fintype X]
    (F : Accounting.Remote.Draw.Committed n m s κ X Cm) (com : Accounting.Remote.Draw.Coins n κ X → Cm)
    (prf : Fin n → Pous.Bits κ → Accounting.Remote.Draw.Blocks m) (δ : ℝ) : Prop :=
  ∀ t, Pous.StreamedWipe.prfAdvantage (prf t)
    ((F.play (Accounting.Remote.Draw.Committed.junk com t)).distinguisher prf δ t) = 0

/-- **The coin commitment hides the openings coins** (`statistical-hiding`, hm96): for every draw `t`, the session the
prover plays at the deployed commitment and at the commitment with coin `t`'s leaf committing zeros reach their
potentials after `t` and `t + 1` windows equally often (`Committed.hide`, at `δ`): window `t`'s hiding term is
nothing. hm96 §3 puts each of its two distances within 2^-192 under the pinned key (`Claims.Hm96Hiding`, at a key that
`common-reference-string` and `hash-derived-key` take to be a good one), `n·2^-191` in all, which the ledger still
charges in δ (`ledger.hiding`); this premise takes them as nothing, so that the bound's numbers are the live
sample's, and `LineHeld` keeps them as its term `Σ_t hide_t`. It is false of a prover that reads coin `t` off its
commitment: a leaf whose salt it learns, or one whose commitment bits tell its coin. -/
def CoinsHidden {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X Cm : Type} [Fintype X]
    (F : Accounting.Remote.Draw.Committed n m s κ X Cm) (com : Accounting.Remote.Draw.Coins n κ X → Cm)
    (prf : Fin n → Pous.Bits κ → Accounting.Remote.Draw.Blocks m) (δ : ℝ) : Prop :=
  ∀ t, F.hide com prf δ t = 0

end Properties.RemoteAudit.Line
