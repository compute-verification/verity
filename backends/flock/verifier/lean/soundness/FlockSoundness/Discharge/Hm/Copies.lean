/-
Δ's copies into the rows' circuits' inputs. Each of Δ's later writes is a copy `(dst, dst), (dst, src)` (`pair2`), and at
each `cv`, `pad` and chain bit of a row there is exactly one: the rows' wires (`rowWires_eq`) go into distinct inputs,
since port `p`'s compressions sit at the distinct places `at_p + k` (`rowSlots_eq`).
-/
import FlockSoundness.Discharge.Hm.Delta
import FlockSoundness.Protocol.Discharge.Hm.Copies

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

/-! ## Copies, counted -/

theorem filter_map_none {β : Type} (l : List ℕ) (D : ℕ) (S : ℕ → β) (x : ℕ) (h : ∀ t ∈ l, D + t ≠ x) :
    (l.map fun t => (D + t, S t)).filter (fun e => e.1 = x) = [] := by
  rw [List.map_eq_flatMap]
  exact filter_flatMap_none _ _ _ fun t ht => by simp [h t ht]

theorem leafPairs_nil {c : Circuit} (h1 : c.leavesIn = #[#[]]) (h2 : c.upv = 1) : Flat.leafPairs c = [] := by
  simp [Flat.leafPairs, h1, h2]

theorem cutPairs_nil {c : Circuit} (h : c.leafCuts = #[]) : Flat.cutPairs c = [] := by
  simp [Flat.cutPairs, h]

end FlockSoundness.Discharge.Hm
