/-
`PortRoots`: circuits of `row/v1` ports, of `row/v2` ports, and of `row/v2` and `row-seg/v1` ports pass `rowsOk`; a mix
of `row/v1` and bit rows, or a bit row of the wrong shape, is refused before its public file is read, and so is one that
names a port twice. Each port's leaves are framed at its own schema, so two ports of one name at different schemas have
different roots. Two ports of different names have different registered roots at SHA-512, and two of one name one root
under one domain, neither a collision; under a hash with one value every pair of differently named ports is one, and
`loadedWhy` refuses it. A refusal at SHA-512 is a file holding a SHA-512 collision.

On the loaded public file: three ports at distinct 64-byte published domains pass, and their published roots are
distinct and not their registered ones; a short domain, a missing one, one that is not hex and a header without
`frame_v3` are refused (`IDS_WHY`), and so are two ports of one domain (`DUP_IDS_WHY`) and a row of `2^64` bits
(`WORDS_WHY`). At shared rows each port's published tree is over its table's rows.

On the header (`headerWhy`): a header whose `frame_v3` names the frame, the row key, each port's schema and each port's
published root over the file's rows passes at a circuit whose pinned dummy rows are the padding instance's. A header
naming `public_inputs` is refused where the circuit's META names none (`INPUTS_WHY`) and passed where it names them;
another row, another frame, row key or schema, and a pinned dummy row that is not the padding instance's are refused
(`HEADER_WHY`).
-/
module

public import Flock.PortRoots
meta import Flock.PortRoots

@[expose] public section

namespace Flock.Test.PortRoots

open Flock Flock.PortRoots
open Lean (Json)

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

#guard loadedWhy (fun _ => ByteArray.empty) (circ #[v2 "x", v2 "y"]) (pub 2) == some ROOTS_WHY
#guard loadedWhy (fun _ => ByteArray.empty) (circ #[v1 "x", v1 "y"]) (pub 2) == some ROOTS_WHY

/-- A 64-byte published domain, one byte repeated. -/
def id64 (b : UInt8) : ByteArray := ByteArray.mk (Array.replicate 64 b)

/-- `pub np` at a header whose `frame_v3.domain_ids` maps each name to its bytes' hex. -/
def pubIds (np : Nat) (ids : List (String × ByteArray)) : Public :=
  { pub np with header := Json.mkObj [("frame_v3", Json.mkObj [("domain_ids",
      Json.mkObj (ids.map fun (k, d) => (k, Json.str (hexOf d))))])] }

def three : Circuit := circ #[v2 "x", v2 "y", seg "z"]
def ids3 : List (String × ByteArray) := [("x", id64 1), ("y", id64 2), ("z", id64 3)]

-- three ports at distinct 64-byte domains pass, their published roots distinct
#guard (List.range 3).map (idOf three (pubIds 3 ids3)) == [id64 1, id64 2, id64 3]
#guard idsSized three (pubIds 3 ids3) && !dupId three (pubIds 3 ids3) && !collidesPub three (pubIds 3 ids3)
#guard loadedWhy Sha512.hash three (pubIds 3 ids3) == none
#guard pubRoot three (pubIds 3 ids3) 0 != pubRoot three (pubIds 3 ids3) 1
#guard pubRoot three (pubIds 3 ids3) 0 != rootOf three (pubIds 3 ids3) 2 0
-- a short domain, a missing one, one that is not hex, and none at all
#guard loadedWhy Sha512.hash three (pubIds 3 [("x", id64 1), ("y", (id64 2).extract 0 63), ("z", id64 3)]) ==
  some IDS_WHY
#guard loadedWhy Sha512.hash three (pubIds 3 [("x", id64 1), ("z", id64 3)]) == some IDS_WHY
#guard loadedWhy Sha512.hash three { pubIds 3 ids3 with header := Json.mkObj [("frame_v3", Json.mkObj [("domain_ids",
  Json.mkObj [("x", Json.str (hexOf (id64 1))), ("y", Json.str "zz"), ("z", Json.str (hexOf (id64 3)))])])] } ==
  some IDS_WHY
#guard loadedWhy Sha512.hash three (pub 3) == some IDS_WHY
-- two ports of one domain
#guard loadedWhy Sha512.hash three (pubIds 3 [("x", id64 1), ("y", id64 2), ("z", id64 1)]) == some DUP_IDS_WHY
-- under a hash with one value two ports' published roots are one, under different domains
#guard collidesPubWith (fun _ => ByteArray.empty) three (pubIds 3 ids3)
-- a row of 2^64 bits
#guard wordsOk three && !wordsOk (circ #[{ v1 "x" with words := 2 ^ 60 }]) &&
  wordsOk (circ #[{ v1 "x" with words := 2 ^ 60 - 1 }])
#guard loadedWhy Sha512.hash (circ #[v1 "x", { v1 "y" with words := 2 ^ 60 }])
  (pubIds 2 [("x", id64 1), ("y", id64 2)]) == some WORDS_WHY

/-- `pubIds 2` at shared rows: port `x`'s table three rows from digest 0, port `y`'s one row from digest 3. -/
def tabled : Public :=
  { pubIds 2 [("x", id64 1), ("y", id64 2)] with
    digests := (Array.range 4).map fun i => ByteArray.mk #[(i + 10).toUInt8],
    tables := some { counts := #[3, 1], starts := #[0, 3], refs := #[0, 0, 2, 0] } }

#guard countOf tabled 0 == 3 && countOf tabled 1 == 1 && countOf (pub 2) 1 == 2
#guard digOf (circ #[v2 "x", v2 "y"]) tabled 1 0 == ByteArray.mk #[13]
#guard digOf (circ #[v2 "x", v2 "y"]) (pub 2) 1 1 == ByteArray.mk #[3]
#guard loadedWhy Sha512.hash (circ #[v2 "x", v2 "y"]) tabled == none

/-- `three` with each port's pinned dummy `b ‖ c` its padding row's under a zero salt, as a circuit file's are. -/
def pinned : Circuit :=
  { three with dummyDigests := (Array.range 3).map fun p =>
      HmOut.rowBcZeroSalt (three.ports.getD p default) (HmOut.dummyRow three p) }

/-- A staged header's `frame_v3` for `three` over `pub 3`'s rows: the frame, the row key, the published domains, and
each port's schema and published root. A later entry of one key replaces it. -/
def fv3 : List (String × Json) :=
  let byPort := fun (f : Nat → String) =>
    Json.mkObj ((List.range 3).map fun q => ((three.ports.getD q default).name, Json.str (f q)))
  [("frame", Json.str "frame-v3-sha512"), ("row_key_sha512", Json.str (hexOf (Sha512.hash Hm96.Default512.key))),
   ("domain_ids", Json.mkObj (ids3.map fun (k, d) => (k, Json.str (hexOf d)))),
   ("schemas", byPort (schemaOf three)), ("roots", byPort fun q => hexOf (pubRoot three (pubIds 3 ids3) q))]

/-- `pub 3` at a header of the `frame_v3` entries `fv` and the further keys `more`. -/
def headed (fv : List (String × Json)) (more : List (String × Json) := []) : Public :=
  { pub 3 with header := Json.mkObj (("frame_v3", Json.mkObj fv) :: more) }

/-- `headerWhy` refuses the public file for `HmOut.checkPublic`'s reason. -/
def rowsWhy (c : Circuit) (p : Public) : Bool := (headerWhy c p).any (·.startsWith HEADER_WHY)

-- a staged header over its own rows passes
#guard headerWhy pinned (headed fv3) == none
-- a header naming `public_inputs`: refused where META names none, the setup's to hold where META names them
#guard headerWhy pinned (headed fv3 [("public_inputs", Json.arr #[])]) == some INPUTS_WHY
#guard headerWhy { pinned with metaJ := Json.mkObj [("public_inputs", Json.arr #[])] }
  (headed fv3 [("public_inputs", Json.arr #[])]) == none
-- another row than the header's root is over
#guard rowsWhy pinned { headed fv3 with digests := (pub 3).digests.setIfInBounds 0 (ByteArray.mk #[9]) }
-- another frame, another row key, another schema
#guard rowsWhy pinned (headed (fv3 ++ [("frame", Json.str "frame-v3")]))
#guard rowsWhy pinned (headed (fv3 ++ [("row_key_sha512", Json.str (hexOf (id64 0)))]))
#guard rowsWhy pinned (headed (fv3 ++ [("schemas", Json.mkObj [("x", Json.str HmRow.ROW_SCHEMA_V2),
  ("y", Json.str HmRow.ROW_SCHEMA), ("z", Json.str HmRow.ROW_SCHEMA_SEG)])]))
-- a pinned dummy row that is not the padding instance's
#guard rowsWhy three (headed fv3)

end Flock.Test.PortRoots
