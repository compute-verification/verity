import Definitions.Accounting.Remote.SketchLine
import Models.Accounting.Assumptions.RateAuditDeployed
import Models.Accounting.Assumptions.RemoteOutside
import Models.Accounting.Assumptions.RemoteSketch

/-!
# The receipts' sketch bound on the memory line's coins (trusted)

The memory line (`Properties.RemoteAudit.Line.LineHeld`, `LineHeldIdentity`) takes the receipts' sketch bound
wherever computed, `SketchBoundAt T O δs`, as a premise on its trace over the session's coins
(`Accounting.Remote.Draw.Coins n κ X`). `SketchBoundDeployed` derives the sketch bound on a trace over a session's
sketch outcomes (`Accounting.Remote.Sketch.Outcome`: the prover's own coins, the registered store, SHA-512's outputs).
`SketchBoundLine` carries the one to the other: on the line's coins whose other coins are a sketch outcome
(`X = Outcome β Hs N 128`), the sketch model's premises on the line's trace read on the sketch outcomes
(`Accounting.Remote.Held.Trace.sketch`) give `SketchBoundAt` on the line's trace at `SketchBoundDeployed`'s

  `δs = Q/2^96 + (Q + P)·(ε + N/2^512)`,

`timing.sketch_failure(C, P, 16, blocks=N, outside=F)` at `Q = F·C`, the δs `ledger.sketch` puts in the memory line's δ.
So `LineHeld` at that `X` holds with its `SketchBoundAt` premise replaced by the sketch model's premises.

The sketch session `A` reads the openings coins among its prover's coins (`β × (Fin n → Bits κ)`): its devices may read
them in any way, so the bound covers a prover that sees each openings coin when it is released, or earlier. What stays
assumed is `SketchBoundDeployed`'s: SHA-512's keys at the candidate seeds are uniform (`SketchHashRandom`, claim
`random-oracle`), the store is uniform (the coins'), the devices obtain sectors whole and verbatim (`RateSketch.Device`,
claim `rate-sketch-verbatim`), the signs' rank tail (`SketchRank`), the hash budget (`KeyBudget`), and the model's
receipts and obtained blocks are the trace's (`Receipted`, `Present`, on `T.sketch O`, where `Present` allows a block
computed on outside the node). That no device outside the node computes on the store is `NoOutsideCompute`, the line's
own premise.
-/

namespace Accounting.Guarantees.Remote

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Sketch
  Accounting.Remote.Sketch.Assumptions

/-- **The sketch bound on the memory line's coins.** On a trace over the line's coins whose other coins are a sketch
outcome and any record `O` of computation outside the node, with the sketch model's premises on the trace read on the
sketch outcomes (`T.sketch O`), at the deployed receipts (`m = 128`, 16 sign rows, `h = 512`, `r = 12`) for a prover
of `Q` SHA-512 compressions, at most `Q + P` candidate seeds and `Q` receipt-hash queries: `SketchBoundAt T O` at
`Q/2^96 + (Q + P)·(ε + N/2^512)`. -/
def SketchBoundLine : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (n κ P N S L q Q : ℕ) (G qtot : Fin P → ℕ) (ε : ℝ)
    (A : Session (β × (Fin n → Pous.Bits κ)) Hs P N 128 512 S L q 16 G)
    (T : Trace (Accounting.Remote.Draw.Coins n κ (Outcome β Hs N 128))) (O : Outside T),
    0 ≤ ε → T.passes ≤ P →
    Accounting.Assumptions.SketchRank 16 128 12 (ENNReal.ofReal ε) →
    ∑ j, G j ≤ Q + P → ∑ j, qtot j ≤ Q →
    Accounting.Assumptions.SketchHashRandom A → KeyBudget A qtot → Receipted A (T.sketch O) →
    Present A (T.sketch O) →
    SketchBoundAt T O ((Q : ℝ) / 2 ^ 96 + ((Q : ℝ) + P) * (ε + N / 2 ^ 512))

end Accounting.Guarantees.Remote
