module

public import Definitions.Spot.Sampling.Scheme
public import Proofs.Core.Game.Basic

/-!
# From the sampling layer to a scheme

`Spot.Sampling.givesAuditBound`: `Sound` gives `GivesAuditBound`, at the event "the sample misses the committed
transcript's incorrect set with probability at most `δ`".
-/

@[expose] public section

namespace Spot.Sampling

open Verity Verity.Game
open scoped ENNReal

/-- **The sampling layer gives every scheme its audit bound.** -/
theorem givesAuditBound (h : Sound) : GivesAuditBound := by
  intro U _ L Reg session A εks εbind hks hlink σ δ
  have hs := h L Reg session A εks εbind hks hlink (fun Y _ => L.missProb (A.incorrect Y) ≤ δ) σ
  refine hs.trans ?_
  rw [add_assoc]
  refine add_le_add ?_ le_rfl
  by_cases hE : L.missProb (A.incorrect (A.committedOf σ)) ≤ δ
  · exact (prCoin_mono fun ω hω => hω.2).trans hE
  · exact (prCoin_mono (q := fun _ => False) fun ω hω => hE hω.1).trans (by simp [prCoin, avg])

end Spot.Sampling
