import Mathlib.Data.Fintype.Pi
import Definitions.Accounting.Guarantees.CheckedCarrier

/-!
# A value check bounds what a carrier stacks, proved

`CheckedCarrierBits` by counting: every assignment of the carried sectors extends to a store, so the decoder's reading
of the contents takes all `2^(256·|carried|)` values as the store ranges, and it takes at most as many as the contents
do, which lie in `Acc`.
-/

namespace Accounting.SecurityProofs

open Accounting.RateSketch

theorem CheckedCarrierBits : Accounting.Guarantees.CheckedCarrierBits := by
  intro N m α v carried dec Acc hAcc hdec
  classical
  have hsurj : Function.Surjective fun E : SketchStore N m => dec (v E) := by
    intro w
    refine ⟨fun p => if hp : p ∈ carried then w ⟨p, hp⟩ else fun _ => 0, ?_⟩
    dsimp only
    rw [hdec]
    funext p
    simp [p.2]
  have h1 : Fintype.card (carried → Sector) ≤ (Finset.univ.image v).card := by
    calc Fintype.card (carried → Sector) = (Finset.univ.image fun E : SketchStore N m => dec (v E)).card := by
          rw [Finset.image_univ_of_surjective hsurj, Finset.card_univ]
      _ ≤ (Finset.univ.image v).card := by
          have h := Finset.card_image_le (s := Finset.univ.image v) (f := dec)
          rwa [Finset.image_image] at h
  have h2 : (Finset.univ.image v).card ≤ Acc.card := Finset.card_le_card fun a ha => by
    obtain ⟨E, -, rfl⟩ := Finset.mem_image.mp ha
    exact hAcc E
  rw [Fintype.card_fun, Fintype.card_coe] at h1
  have hs : Fintype.card Sector = 2 ^ 256 := by simp [Sector]
  rw [hs, ← pow_mul, mul_comm] at h1
  exact h1.trans h2

end Accounting.SecurityProofs
