import FlockSoundness.Protocol.Discharge.DecodeZL.Statements

/-!
# The opening's rounds carry its messages (R11b)

`op_carries`: the ring switches' rounds of `rsFrames`, read by `wOP`, are `opMsgs`, cut at `opShape`'s coins. Each ring
switch's round closes at its `sq 2 7` and carries the switch's 128 values (`carries_switch`; the batch label rides in
front of the first), the rest by induction (`carries_switches`), and the batching coin's round carries nothing.
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model

/-- **The opening's wires**: each ring switch's round its 128 values, the batching coin's round nothing. -/
def wOP (n : ℕ) : List (List Wire) := List.replicate n [.v 128] ++ [[]]

end FlockSoundness.Discharge.DecodeZL
