import Models.Accounting.Guarantees.RemoteSketchLine
import Proofs.Accounting.RemoteHeld.Sketch
import Proofs.Pouw.Prob

/-!
# The receipts' sketch bound on the memory line's coins: the proof

The line's coins whose other coins are a sketch outcome and the sketch outcomes whose prover's coins carry the openings
coins are one uniform space read two ways (`lineCoinsEquiv`, whose forward map is `lineCoins`), so an event has one
probability on both (`Pouw.pr_equiv`). On the sketch outcomes the line's trace `T.sketch O` counts a block computed on
outside the node with the chips', so its sketch bound's event is `SketchBoundAt T O`'s read through `lineCoins`
(`sketchBoundAt_of_sketch`). `SketchBoundDeployed` on `T.sketch O` gives that sketch bound, hence `SketchBoundLine`.
-/

namespace Accounting.SecurityProofs.Remote

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Sketch
  Accounting.Remote.Sketch.Assumptions

section Line

variable {β Hs : Type} {n κ N m : ℕ}

/-- The sketch outcomes whose prover's coins carry the openings coins are the line's coins whose other coins are a
sketch outcome, reordered. -/
def lineCoinsEquiv : Outcome (β × (Fin n → Pous.Bits κ)) Hs N m ≃ Accounting.Remote.Draw.Coins n κ (Outcome β Hs N m)
    where
  toFun := lineCoins
  invFun ω := ((ω.1.1, ω.2), ω.1.2.1, ω.1.2.2)
  left_inv _ := rfl
  right_inv _ := rfl

variable [Fintype β] [Fintype Hs]

/-- The sketch bound on the line's trace read on the sketch outcomes is `SketchBoundAt` on the line's trace. -/
theorem sketchBoundAt_of_sketch {T : Trace (Accounting.Remote.Draw.Coins n κ (Outcome β Hs N m))} {O : Outside T}
    {δ : ℝ} (h : SketchBound (T.sketch O) δ) : SketchBoundAt T O δ := by
  unfold SketchBound SketchBoundAt Trace.Holds at *
  rw [← Pouw.pr_equiv lineCoinsEquiv]
  refine le_of_eq_of_le ?_ h
  congr 1
  funext ω
  simp only [Trace.present, Trace.sketch, Finset.union_assoc]
  rfl

end Line

/-- **The sketch bound on the memory line's coins** (`Accounting.Guarantees.Remote.SketchBoundLine`):
`SketchBoundDeployed` on the line's trace read on the sketch outcomes, carried back to the line's coins. -/
theorem SketchBoundLine : Accounting.Guarantees.Remote.SketchBoundLine := by
  intro β Hs _ _ n κ P N S L q Q G qtot ε A T O hε hP hRank hG hqtot hH hq hrec hpres
  exact sketchBoundAt_of_sketch
    (SketchBoundDeployed (β × (Fin n → Pous.Bits κ)) Hs P N S L q Q G qtot ε A (T.sketch O) hε hP hRank hG hqtot hH
      hq hrec hpres)

end Accounting.SecurityProofs.Remote
