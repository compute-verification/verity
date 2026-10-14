import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-!
# Piece (h): V*'s files parse

What `vstarParsesScoped_of_list` rests on: `built_of_parse`, a parsed circuit's unit is a net `Net.ofRows` built.
-/

namespace FlockC3.VStarList

/-- **A parsed circuit's unit is built**: `HmRow.parse`'s nets are `Net.ofRows`'s or lookup slots, and its check
refuses a lookup slot as the unit. -/
theorem built_of_parse {tags : Flock.Tags} {cf : ByteArray} {c : Flock.Circuit}
    (h : Flock.HmRow.parse tags cf [] = .ok c) : FlockSoundness.Built c.unit := by
  have P := FlockSoundness.parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  exact FlockSoundness.unit_built P (FlockSoundness.check_facts P.typed hls)

end FlockC3.VStarList
