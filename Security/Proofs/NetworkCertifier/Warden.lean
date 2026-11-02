import Definitions.NetworkCertifier.Warden
import Mathlib.Data.Finset.Option
import Mathlib.Data.Fintype.BigOperators

/-!
# The warden's code is the client view

The warden's code on a lane emits, cell by cell, what the client view gives for the lane's first late cell
(`wardenLane_eq`): its data cells and padding before it, the fail cell, padding after. So the code's output is the
client view of the schedule, the data and the fail cells (`wardenCode_eq`), which `Tap` reads the network certifier's
cells from.
-/

namespace NetTiming

/-- A cell before the fail cell is the stream's. -/
theorem cellView_of_lt {Cell : Type} (stream : List Cell) {o : Option ℕ} {j : ℕ} (h : ∀ f, o = some f → j < f) :
    cellView stream o j = streamCell stream j := by
  cases o with
  | none => rfl
  | some f => simp [cellView, h f rfl]

/-- **The warden's lane is the client view of its first late cell.** -/
theorem wardenLane_eq {Cell : Type} (a : Arrivals Cell) (t : ℕ → ℕ) (j k : ℕ) :
    wardenLane a t j k = (List.range' j k).map fun i =>
      (t i, cellView a.stream ((List.range' j k).find? fun i => decide (i ≤ a.stream.length ∧ t i < a.ready i)) i) := by
  induction k generalizing j with
  | zero => simp [wardenLane]
  | succ k ih =>
    have hafter : ∀ f, (List.range' (j + 1) k).find? (fun i => decide (i ≤ a.stream.length ∧ t i < a.ready i)) =
        some f → j < f := by
      intro f hf
      have := List.mem_of_find?_eq_some hf
      rw [List.mem_range'_1] at this
      omega
    rw [show List.range' j (k + 1) = j :: List.range' (j + 1) k from rfl, List.map_cons]
    by_cases hend : a.stream.length < j
    · have hp : ¬ (j ≤ a.stream.length ∧ t j < a.ready j) := fun h => by omega
      rw [List.find?_cons_of_neg (by simpa using hp)]
      simp only [wardenLane, hend, ↓reduceIte, ih]
      congr 1
      rw [cellView_of_lt _ hafter]
      simp [streamCell, show ¬ j < a.stream.length by omega]
    · by_cases hlate : t j < a.ready j
      · have hp : j ≤ a.stream.length ∧ t j < a.ready j := ⟨by omega, hlate⟩
        rw [List.find?_cons_of_pos (by simpa using hp)]
        simp only [wardenLane, hend, hlate, ↓reduceIte]
        refine List.cons_eq_cons.mpr ⟨by simp [cellView], ?_⟩
        apply List.map_congr_left
        intro i hi
        rw [List.mem_range'_1] at hi
        simp [cellView, show ¬ i < j by omega, show i ≠ j by omega]
      · have hp : ¬ (j ≤ a.stream.length ∧ t j < a.ready j) := fun h => hlate h.2
        rw [List.find?_cons_of_neg (by simpa using hp)]
        simp only [wardenLane, hend, hlate, ↓reduceIte, ih]
        rw [cellView_of_lt _ hafter]

theorem wardenCode_eq {κ Cell : Type} (s : WardenSchedule κ) (arr : κ → Arrivals Cell) :
    wardenCode s arr = clientView s (fun c => (arr c).stream) (failCell s arr) := by
  funext c
  simp only [wardenCode, clientView, failCell, wardenLane_eq, List.range_eq_range']

end NetTiming
