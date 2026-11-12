module

/-!
# A tree's depth and padded width

`log2ceil n` is the least `d` with `n ≤ 2 ^ d`, and `nextPow2 n` that power: a tree over `n` leaves pads them to
`nextPow2 n` and is `log2ceil n` levels deep. Frame-v3 (`Commitments.FV3`) and Flock's circuits (`Flock.Net`) read them.
-/

@[expose] public section

namespace Flock

def log2ceil (n : Nat) : Nat := if n ≤ 1 then 0 else Nat.log2 (n - 1) + 1
def nextPow2 (n : Nat) : Nat := 2 ^ log2ceil n

end Flock
