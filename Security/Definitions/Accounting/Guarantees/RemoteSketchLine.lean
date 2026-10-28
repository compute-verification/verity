import Definitions.Accounting.Remote.SketchLine
import Definitions.Accounting.Assumptions.RemoteHeld
import Definitions.Accounting.Assumptions.RemoteOutside

/-!
# The receipts' sketch bound on the memory line's coins (trusted)

The memory line (`Properties.RemoteAudit.Line.LineHeld`) takes the receipts' sketch bound
wherever computed, `SketchBoundAt T O δs`, as a premise on its trace over the session's coins
(`Accounting.Remote.Draw.DrawCoins n κ X`). `SketchBoundDeployed` derives the sketch bound on any trace over a session's
sketch outcomes (`Accounting.Remote.Sketch.Outcome`: the prover's own coins, the registered store, SHA-512's outputs),
for every type of prover's coins. `SketchBoundLine` carries the one to the other: on the line's coins whose other coins
are a sketch outcome (`X = Outcome β Hs N m`), the sketch bound on the line's trace read on the sketch outcomes
(`Accounting.Remote.Held.Trace.sketch`) is `SketchBoundAt` on the line's trace, at the same δ.

So `SketchBoundDeployed` at prover's coins `β × (Fin n → Bits κ)` and the trace `T.sketch O`, under its premises
there, gives `LineHeld`'s `SketchBoundAt` premise at `X = Outcome β Hs N 128` at `SketchBoundDeployed`'s δs,
`timing.sketch_failure(C, P, 16, blocks=N, outside=F)` at `Q = F·C`, the δs `ledger.sketch` puts in the memory line's
δ. The sketch session reads the openings coins among its prover's coins: its devices may read them in any way, so the
bound covers a prover that sees each openings coin when it is released, or earlier. `Present` on `T.sketch O` allows a
block computed on outside the node, which the relay prover meets; that no device outside the node computes for it is
what the line reads from the certifiers' record (`Covered`, `IngressCounted`: no counted ingress during the credited
passes, at δ_net). What stays assumed is `SketchBoundDeployed`'s premises, read on `T.sketch O`.
-/

namespace Accounting.Guarantees.Remote

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Sketch

/-- **The sketch bound on the memory line's coins.** On a trace over the line's coins whose other coins are a sketch
outcome and any record `O` of computation outside the node, the receipts' sketch bound on the trace read on the sketch
outcomes (`T.sketch O`) at `δ` gives `SketchBoundAt T O δ`. -/
def SketchBoundLine : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (n κ N m : ℕ)
    (T : Trace (Accounting.Remote.Draw.DrawCoins n κ (Outcome β Hs N m))) (O : Outside T) (δ : ℝ),
    SketchBound (T.sketch O) δ → SketchBoundAt T O δ

end Accounting.Guarantees.Remote
