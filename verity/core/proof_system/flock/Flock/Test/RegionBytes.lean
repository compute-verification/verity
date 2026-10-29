/-
`checkRegionBytes`, the setup's region-size check: a region of four free bits carries two bytes in each of the `2^nbl`
blocks. A region one byte short, or one block's bytes short, is refused; so is a region one byte long.
-/
import Flock.Statement

namespace Flock.Test.RegionBytes

def region (bytes : Nat → Nat) : Region :=
  { name := "r", free := #[0, 1, 2, 3], mask := 0, fixed := 0, bytes := fun b => ByteArray.mk (Array.replicate (bytes b) 0) }

def ok (nbl : Nat) (regions : List Region) : Bool := (checkRegionBytes nbl regions.toArray).isOk

#guard ok 3 [region fun _ => 2]
#guard ok 3 [region fun _ => 2, region fun _ => 2]
#guard ok 0 [region fun b => if b = 0 then 2 else 1]
#guard ok 3 []
#guard !ok 3 [region fun _ => 1]
#guard !ok 3 [region fun _ => 3]
#guard !ok 3 [region fun b => if b = 7 then 1 else 2]
#guard ok 2 [region fun b => if b = 7 then 1 else 2]
#guard !ok 3 [region fun _ => 2, region fun _ => 0]

end Flock.Test.RegionBytes
