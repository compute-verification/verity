import Definitions.Accounting.Remote.Draw
import Definitions.Accounting.Remote.Outside
import Definitions.Accounting.Remote.Sketch

/-!
# The memory line's coins as a sketch outcome (trusted)

The memory line (`Properties.RemoteAudit.Line.LineHeld`) is a statement over a session's coins
`Accounting.Remote.Draw.DrawCoins n κ X`: the other coins `X` (the prover's, the world's, every window's challenge) and each
window's openings coin, all uniform. The receipts' sketch bound (`Accounting.Guarantees.Remote.SketchBoundDeployed`) is
a statement over a session's outcomes `Accounting.Remote.Sketch.Outcome β Hs N m`: the prover's own coins, the
registered store and SHA-512's outputs, all uniform. One space carries both: the line's coins whose other coins are a
sketch outcome, `Draw.DrawCoins n κ (Outcome β Hs N m)`. Read the other way round it is a sketch outcome whose prover's
coins are `β` and the windows' openings coins, `Outcome (β × (Fin n → Bits κ)) Hs N m` (`lineCoins`, a bijection). The
prover is sent every openings coin once it is released, and the sketch model's devices may read its prover's coins in
any way, so a device that reads an openings coin early is one of the model's.

* `lineCoins ω`: the sketch outcome `ω` read as the line's coins.
* `Trace.sketch T O`: the line's trace `T` on that space read on the sketch model's outcomes, with the registered blocks
  a device outside the node computes on at a tick (`Outside.blocks`) counted with the chips' then. Where a block is,
  the sketch model reads only `present`, so `Present A (T.sketch O)` is the sketch model's `Present` with outside
  computation allowed (a block the kept candidate obtained whole was on the node or computed on outside it at a tick
  of its pass), which the relay prover meets, and `SketchBound (T.sketch O) δ` is `SketchBoundAt T O δ` over the line's
  coins.
-/

namespace Accounting.Remote.Sketch

variable {β Hs : Type} {n κ N m : ℕ}

/-- A sketch outcome whose prover's coins carry the windows' openings coins, read as the memory line's coins: the other
coins the prover's own, the store and SHA-512's outputs, and the openings coins. -/
def lineCoins (ω : Outcome (β × (Fin n → Pous.Bits κ)) Hs N m) :
    Accounting.Remote.Draw.DrawCoins n κ (Outcome β Hs N m) :=
  ((ω.1.1, ω.2.1, ω.2.2), ω.1.2)

end Accounting.Remote.Sketch

namespace Accounting.Remote.Held.Trace

open Accounting.Remote.Sketch

variable {β Hs : Type} {n κ N m : ℕ}

/-- The memory line's trace over coins whose other coins are a sketch outcome, read on the sketch model's outcomes
(`lineCoins`), with the registered blocks a device outside the node computes on at a tick counted with the chips'. -/
def sketch (T : Trace (Accounting.Remote.Draw.DrawCoins n κ (Outcome β Hs N m))) (O : Outside T) :
    Trace (Outcome (β × (Fin n → Pous.Bits κ)) Hs N m) where
  blocks := T.blocks
  passes := T.passes
  accepts ω := T.accepts (lineCoins ω)
  tick ω := T.tick (lineCoins ω)
  plan ω := T.plan (lineCoins ω)
  correct ω := T.correct (lineCoins ω)
  mem ω := T.mem (lineCoins ω)
  chip ω t := T.chip (lineCoins ω) t ∪ O.blocks (lineCoins ω) t
  rebuilt ω := T.rebuilt (lineCoins ω)
  disk ω := T.disk (lineCoins ω)
  link ω := T.link (lineCoins ω)

end Accounting.Remote.Held.Trace
