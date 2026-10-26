import Definitions.Pouw.Guarantees.DitherInstance
import Proofs.Pouw.RelativeSecurity

/-!
# The headline on the cards from the headline on the model

* `workSoundAt_of_workSound`: `WorkSound` gives `WorkSoundAt c` on every card table: an instance of its quantifiers,
  with the timed verdict accepting no more than the verdict.
* `workSoundH200_of_workSound`, `workSoundSm120_of_workSound`: the two instances.
* `h200_floors`, `sm120_floors`: θ's floors with the product form are met by some open parameter on each card, so
  `WorkSoundAt`'s hypotheses on `x` are not empty.
-/

namespace Pouw.SecurityProofs.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.Guarantees.DitherInstance

open scoped NNRat

/-- `WorkSound` gives the headline on every card table. -/
theorem workSoundAt_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) (c : Verity.Accel.Card) :
    WorkSoundAt c := by
  intro x _ _ D P L G _ k verdict Pd μ pubmm pubsha Hmm Hsha Ho η δ nmax macs Ein Eout κS κx Δ t
    hdual hmm hsha ho hH hpm hps hη hδ hδ1 hn hm hEi hEo g1 g2 p1 p2 sak sha svc imp ext A
  have hβ : 0 ≤ nmax * Hmm / macs := div_nonneg (mul_nonneg hn hmm.le) hm.le
  have hK : 0 ≤ Ein + Eout := add_nonneg hEi hEo
  refine le_trans (prob_mono fun ox hox => ⟨hox.1.1, hox.2⟩)
    (h G P k verdict Pd μ pubmm pubsha Hmm Hsha Ho η δ (nmax * Hmm / macs) (Ein + Eout) κS κx hdual hmm hsha ho hH
      hpm hps hη hδ hδ1 hβ hK g1 g2 p1 p2 sak sha svc imp ext A)

theorem workSoundH200_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) : WorkSoundH200 :=
  workSoundAt_of_workSound h _

theorem workSoundSm120_of_workSound (h : Pouw.Guarantees.Dither.WorkSound) : WorkSoundSm120 :=
  workSoundAt_of_workSound h _

/-- An open parameter at θ's floors: every dot the constant zero (in product form), `sync = 1`, `b1 = 1/8`,
`madHi = 2`, no backlog. -/
def floorOpen : Verity.Accel.Open where
  slack := 0
  b1 := 1 / 8
  sync := 1
  madHi := 2
  dot _ _ _ _ := fun _ _ _ => 0

theorem floorOpen_productForm : Verity.Assumptions.AccelProductForm floorOpen :=
  fun _ _ _ _ => ⟨fun _ _ _ _ => 0, fun _ _ _ _ _ => rfl⟩

/-- The H200's floors with the product form are met. -/
theorem h200_floors :
    ∃ x, Verity.Accel.H200.card.floors x ∧ Verity.Assumptions.AccelProductForm x :=
  ⟨floorOpen, ⟨le_rfl, le_rfl, le_rfl⟩, floorOpen_productForm⟩

/-- The RTX PRO 6000's floors with the product form are met. -/
theorem sm120_floors :
    ∃ x, Verity.Accel.Sm120.card.floors x ∧ Verity.Assumptions.AccelProductForm x :=
  ⟨floorOpen, le_rfl, floorOpen_productForm⟩

end Pouw.SecurityProofs.DitherInstance
