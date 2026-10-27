import Definitions.Accounting.Guarantees.PoubSketchLine
import Proofs.Pouw.Prob

/-!
# The receipts' sketch bound on the memory line's coins: the proof

The line's coins whose other coins are a sketch outcome and the sketch outcomes whose prover's coins carry the openings
coins are one uniform space read two ways (`lineCoinsEquiv`, whose forward map is `lineCoins`), so an event has one
probability on both (`Pouw.pr_equiv`). On the sketch outcomes the line's trace `T.sketch O` counts a block computed on
outside the node with the chips', so its sketch bound's event is `SketchBoundAt T O`'s read through `lineCoins`.
-/

namespace Accounting.SecurityProofs.Poub

open Accounting.Poub.Held Accounting.Poub.Held.Assumptions Accounting.Poub.Sketch

/-- The sketch outcomes whose prover's coins carry the openings coins are the line's coins whose other coins are a
sketch outcome, reordered. -/
def lineCoinsEquiv {β Hs : Type} {n κ N m : ℕ} :
    Outcome (β × (Fin n → Pous.Bits κ)) Hs N m ≃ Accounting.Poub.Draw.Coins n κ (Outcome β Hs N m) where
  toFun := lineCoins
  invFun ω := ((ω.1.1, ω.2), ω.1.2.1, ω.1.2.2)
  left_inv _ := rfl
  right_inv _ := rfl

/-- **The sketch bound on the memory line's coins** (`Accounting.Guarantees.Poub.SketchBoundLine`): the sketch bound
on the line's trace read on the sketch outcomes is `SketchBoundAt` on the line's trace. -/
theorem SketchBoundLine : Accounting.Guarantees.Poub.SketchBoundLine := by
  intro β Hs _ _ n κ N m T O δ h
  unfold SketchBound SketchBoundAt Trace.Holds at *
  rw [← Pouw.pr_equiv lineCoinsEquiv]
  refine le_of_eq_of_le ?_ h
  congr 1
  funext ω
  simp only [Trace.present, Trace.sketch, Finset.union_assoc]
  rfl

end Accounting.SecurityProofs.Poub
