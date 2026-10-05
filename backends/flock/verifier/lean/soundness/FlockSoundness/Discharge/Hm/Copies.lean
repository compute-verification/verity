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

theorem leafPairs_nil {c : Circuit} (h1 : c.leavesIn = #[#[]]) (h2 : c.upv = 1) : Flat.leafPairs c = [] := by
  simp [Flat.leafPairs, h1, h2]

theorem cutPairs_nil {c : Circuit} (h : c.leafCuts = #[]) : Flat.cutPairs c = [] := by
  simp [Flat.cutPairs, h]

end FlockSoundness.Discharge.Hm
