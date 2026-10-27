/-
`PortRoots`: circuits of `row/v1` ports, of `row/v2` ports, and of `row/v2` and `row-seg/v1` ports pass `rowsOk`; a mix
of `row/v1` and bit rows, or a bit row of the wrong shape, is refused before its public file is read, and so is one that
names a port twice. Each port's leaves are framed at its own schema, so two ports of one name at different schemas have
different roots. Two ports of different names have different registered roots at SHA-512, and two of one name one root
under one domain, neither a collision; under a hash with one value every pair of differently named ports is one, and
`checkWith` refuses it. A refusal at SHA-512 is a file holding a SHA-512 collision.
-/
import Flock.PortRoots

namespace Flock.Test.PortRoots

open Flock Flock.PortRoots

def v1 (name : String) : Port := { name, words := 64, chunks := 2, nb := 0, bits := 1024 }
def v2 (name : String) : Port := { name, words := 64, chunks := 1, nb := 0, bits := 1024, v2 := true }
def seg (name : String) : Port :=
  let sh := (HmRow.segShape 1024).1
  let s := (1024 + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS
  { v2 name with segs := some sh, chunks := ((sh.toList.take s).map Prod.snd).sum + ((64 * s + 17 + 127) / 128 - 1) }

def circ (ports : Array Port) : Circuit := { (default : Circuit) with ports }

/-- Two instances' `b ‖ c` per port, distinct bytes each. -/
def pub (np : Nat) : Public :=
  { (default : Public) with n := 2, digests := (Array.range (2 * np)).map fun i => ByteArray.mk #[i.toUInt8] }

def why (H : ByteArray → ByteArray) (c : Circuit) : Option String :=
  match checkWith H (.ok c) Tags.circuit ByteArray.empty with
  | .ok () => none
  | .error e => some e

#guard ProvedScope.v2Row (v2 "x") && ProvedScope.segRow (seg "x")
#guard rowsOk (circ #[v1 "x", v1 "y"])
#guard rowsOk (circ #[v2 "x", v2 "y"])
#guard rowsOk (circ #[v2 "x", seg "y"])
#guard rowsOk (circ #[seg "x", seg "y"])
#guard !rowsOk (circ #[v1 "x", v2 "y"])
#guard !rowsOk (circ #[seg "x", v1 "y"])
#guard !rowsOk (circ #[{ v2 "x" with chunks := 2 }])
#guard why Sha512.hash (circ #[v1 "x", v2 "y"]) == some (ProvedScope.refusal ROWS_WHY)
#guard why Sha512.hash (circ #[{ v2 "x" with chunks := 2 }]) == some (ProvedScope.refusal ROWS_WHY)
#guard why Sha512.hash (circ #[v1 "x", v1 "x"]) == some (ProvedScope.refusal NAMES_WHY)
#guard why Sha512.hash (circ #[v2 "x", seg "x"]) == some (ProvedScope.refusal NAMES_WHY)

#guard (List.range 3).map (schemaOf (circ #[v1 "x", v2 "y", seg "z"])) ==
  [HmRow.ROW_SCHEMA, HmRow.ROW_SCHEMA_V2, HmRow.ROW_SCHEMA_SEG]
#guard rootOf (circ #[v2 "x", seg "x"]) (pub 2) 2 0 != rootOf (circ #[v2 "x", seg "x"]) (pub 2) 2 1

#guard !collides (circ #[v1 "x", v1 "y"]) (pub 2) 2
#guard !collides (circ #[v2 "x", v2 "y"]) (pub 2) 2
#guard !collides (circ #[v2 "x", seg "y"]) (pub 2) 2
#guard !collides (circ #[v2 "x", v2 "x"]) (pub 2) 2
#guard rootOf (circ #[v2 "x", v2 "y"]) (pub 2) 2 0 != rootOf (circ #[v2 "x", v2 "y"]) (pub 2) 2 1
#guard collidesWith (fun _ => ByteArray.empty) (circ #[v1 "x", v1 "y"]) (pub 2) 2
#guard collidesWith (fun _ => ByteArray.empty) (circ #[v2 "x", seg "y"]) (pub 2) 2
#guard !collidesWith (fun _ => ByteArray.empty) (circ #[v2 "x", v2 "x"]) (pub 2) 2

/-- `checkWith`'s roots step on a loaded public file: the refusal `collidesWith` gives. -/
def rootsWhy (H : ByteArray → ByteArray) (c : Circuit) (p : Public) : Option String :=
  if collidesWith H c p p.n then some (ProvedScope.refusal ROOTS_WHY) else none

#guard rootsWhy (fun _ => ByteArray.empty) (circ #[v2 "x", v2 "y"]) (pub 2) == some (ProvedScope.refusal ROOTS_WHY)
#guard rootsWhy (fun _ => ByteArray.empty) (circ #[v1 "x", v1 "y"]) (pub 2) == some (ProvedScope.refusal ROOTS_WHY)
#guard rootsWhy Sha512.hash (circ #[v2 "x", seg "y"]) (pub 2) == none

end Flock.Test.PortRoots
