module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Circuit

/-!
# The honest session's public file

The public file the circuit file is staged with: one instance, unit 0, of the circuit's class, its one row's leaf string
`rowBc` committed under a frame-v3 tree of one leaf, and no output words (the outputs are hidden). The header's
digests are spliced terms: the circuit file's SHA-512, the class's, the row key's and the tree's root.

Piece P2 owns this file's values: it fixes the header's members to the ones `HmOut.loadPublic` and `HmIn.loadPublic`
accept.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)
open Flock

/-- The circuit file's SHA-512. -/
def circuitSha : ByteArray := Sha512.hash circuitFile

/-- The class of the circuit file's instances (`HmRow.classOf`), as a term: the file with META's bindings removed. -/
def classSha : ByteArray :=
  Sha512.hash (headLine.toUTF8 ++ "\nMETA ".toUTF8 ++
    ((canonUtf8 (Json.mkObj (metaPairs.filter fun kv => !HmRow.BINDING_KEYS.contains kv.1))).toOption.getD "").toUTF8 ++
    "\n".toUTF8 ++ (RecDeploy.VStarNets.joinNl ((circuitLines (HmNets.shaLaid ()) (HmNets.hmLaid ())).drop 2)).toUTF8)

/-- The instance's committed row, its leaf string. -/
def rowBc : ByteArray := ByteArray.mk (Array.replicate 128 0)

/-- The row's frame-v3 domain, 64 bytes. -/
def domY : ByteArray := ByteArray.mk (Array.replicate 64 0)

/-- The frame-v3 root over the one row. -/
def rootY : ByteArray := FV3.root domY #[FV3.leaf domY 0 HmRow.ROW_SCHEMA_V2 (Hm96.Default512.treeLeaf rowBc)]

/-- The header's `units`: instance 0 is unit 0, of the circuit's class. -/
def unitsJ : Json := Json.mkObj [("blocks", .arr #[.arr #[jn 0]]), ("classes", .arr #[.str (hexOf classSha)]),
  ("indices", .arr #[jn 0]), ("units_per_vu", jn 1), ("vus_per_block", jn 1)]

/-- The header's `frame_v3`. -/
def fv3J : Json := Json.mkObj [("domain_ids", Json.mkObj [("y", .str (hexOf domY))]),
  ("frame", .str "frame-v3-sha512"), ("roots", Json.mkObj [("y", .str (hexOf rootY))]),
  ("row_key_sha512", .str (hexOf (Sha512.hash Hm96.Default512.key))),
  ("schemas", Json.mkObj [("y", .str HmRow.ROW_SCHEMA_V2)])]

/-- The public file's header. -/
def hdrJ : Json := Json.mkObj [("circuit_sha512", .str (hexOf circuitSha)), ("format", .str Flock.Tags.circuit.publicFormat),
  ("frame_v3", fv3J), ("instances", jn nUnits), ("rows", .bool false), ("statement", .str Flock.Tags.circuit.statement),
  ("units", unitsJ)]

/-- The registered public file: the header's line, then the row's leaf string. -/
def publicFile : ByteArray := (Firewall.printJson hdrJ).toUTF8 ++ "\n".toUTF8 ++ rowBc

end Verity.Sampled.Instance.HonestSession
