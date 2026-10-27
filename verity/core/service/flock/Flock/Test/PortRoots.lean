/-
`PortRoots`: a circuit with a `row/v1` port or a segmented `row-seg/v1` port is refused before its public file is read;
one of unsegmented `row/v2` ports passes `rowsV2`, and one that names a port twice is refused too. Two ports of different names have different registered roots at
SHA-512, and two of one name one root under one domain, neither a collision; under a hash with one value every pair of
differently named ports is one, and `checkWith` refuses it. A refusal at SHA-512 is a file holding a SHA-512 collision.
-/
import Flock.PortRoots

namespace Flock.Test.PortRoots

open Flock Flock.PortRoots

def v1 (name : String) : Port := { name, words := 64, chunks := 2, nb := 0 }
def v2 (name : String) : Port := { name, words := 64, chunks := 2, nb := 0, bits := 1024, v2 := true }
def seg (name : String) : Port := { v2 name with segs := some #[(1024, 9)] }

def circ (ports : Array Port) : Circuit := { (default : Circuit) with ports }

/-- Two instances' `b ‖ c` per port, distinct bytes each. -/
def pub (np : Nat) : Public :=
  { (default : Public) with n := 2, digests := (Array.range (2 * np)).map fun i => ByteArray.mk #[i.toUInt8] }

def why (H : ByteArray → ByteArray) (c : Circuit) : Option String :=
  match checkWith H (.ok c) Tags.circuit ByteArray.empty with
  | .ok () => none
  | .error e => some e

#guard rowsV2 (circ #[v2 "x", v2 "y"])
#guard !rowsV2 (circ #[v2 "x", v1 "y"])
#guard !rowsV2 (circ #[seg "x", v2 "y"])
#guard why Sha512.hash (circ #[v1 "x", v2 "y"]) == some (ProvedScope.refusal ROWS_WHY)
#guard why Sha512.hash (circ #[v2 "x", seg "y"]) == some (ProvedScope.refusal ROWS_WHY)
#guard why Sha512.hash (circ #[v2 "x", v2 "x"]) == some (ProvedScope.refusal NAMES_WHY)

#guard !collides (circ #[v2 "x", v2 "y"]) (pub 2) 2
#guard !collides (circ #[v2 "x", v2 "x"]) (pub 2) 2
#guard rootOf (circ #[v2 "x", v2 "y"]) (pub 2) 2 0 != rootOf (circ #[v2 "x", v2 "y"]) (pub 2) 2 1
#guard collidesWith (fun _ => ByteArray.empty) (circ #[v2 "x", v2 "y"]) (pub 2) 2
#guard !collidesWith (fun _ => ByteArray.empty) (circ #[v2 "x", v2 "x"]) (pub 2) 2

/-- `checkWith`'s roots step on a loaded public file: the refusal `collidesWith` gives. -/
def rootsWhy (H : ByteArray → ByteArray) (c : Circuit) (p : Public) : Option String :=
  if collidesWith H c p p.n then some (ProvedScope.refusal ROOTS_WHY) else none

#guard rootsWhy (fun _ => ByteArray.empty) (circ #[v2 "x", v2 "y"]) (pub 2) == some (ProvedScope.refusal ROOTS_WHY)
#guard rootsWhy Sha512.hash (circ #[v2 "x", v2 "y"]) (pub 2) == none

end Flock.Test.PortRoots
