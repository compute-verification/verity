import Models.Pouw.Assumptions.Dimension.H100Cite
import Proofs.Pouw.Dimension.H100FragProofs
import Proofs.Pouw.Dimension.H100MeasuredProofs

/-!
# How the H100 pins are cited: proofs

* **Packed codes** (`distinctWritesPacked`): `distinctWritesMH100Floor` on the slots, and the pigeonhole bound
  `|codes| ≤ κ·|T|`.
* **`FragLanes`** (`denseWritesFragLanes`, `fragLanesSat`): `FragLanes` unfolds to `denseWritesFrag`'s lane bound. The
  sum of the coordinates is dense for `LaneFns m 0` (`denseLaneWitness`), so it depends on no `m` lanes.
* **Round 7's prices** (`distinctWritesMeasured`): `distinctWritesMH100On` with `h32Measured`.
-/

namespace Pouw.Dimension.H100

open Finset

theorem distinctWritesPacked : DistinctWritesPacked := by
  intro c pr hc G F hF S fpSem L P T ι codes slot κ hL hP hT hO hslot hfib
  classical
  have h1 := distinctWritesMH100Floor c pr hc G F hF S fpSem L P T hL hP hT hO
  have h2 : codes.card ≤ κ * T.card :=
    card_le_mul_card_image_of_maps_to hslot κ fun t ht =>
      hfib t ht _ (filter_subset _ _) fun i hi => (mem_filter.1 hi).2
  calc c * codes.card ≤ c * (κ * T.card) := Nat.mul_le_mul_left _ h2
    _ = κ * (c * T.card) := by ring
    _ ≤ κ * cost100 pr P := Nat.mul_le_mul_left _ h1

theorem denseWritesFragLanes : DenseWritesFragLanes :=
  fun a c pr hpr hca hac G Λ S p F W K q β hF hβ hW =>
    denseWritesFrag a c pr hpr hca hac G Λ S p F W K q β hF hβ hW

theorem fragLanesSat : FragLanesSat := by
  refine ⟨fun G Λ S p => ?_, fun m h => ?_⟩
  · rintro w ⟨j, rfl⟩
    exact ⟨{j}, by simp, fun g _ g' _ hg => hg j (mem_singleton_self j)⟩
  · obtain ⟨J, hJ, hdep⟩ := h (sumCoords m) (Set.mem_singleton _)
    exact (denseLaneWitness m).1 (sumCoords m) (mem_singleton_self _) (fun _ => 0) ⟨0, rfl⟩ (sumCoords m)
      ⟨J, hJ, sumCoords m, hdep, fun g _ => by simp⟩ fun g _ => by simp

theorem distinctWritesMeasured : DistinctWritesMeasured :=
  fun G F hF S fpSem L P T => distinctWritesMH100On h100Measured h32Measured G F hF S fpSem L P T

end Pouw.Dimension.H100
