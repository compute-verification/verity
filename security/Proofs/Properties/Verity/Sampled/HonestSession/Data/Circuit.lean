module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Program
public import Proofs.Flock.SessionLoop.Concrete.File
public import Flock.Firewall

/-!
# The honest session's circuit file

A hidden-output circuit file (`Tags.circuit`): one unit, `Concrete.unitLaid`'s sixteen-word copy, reading no input row
(every input leaf `-1`, the zero word) and writing one `sha512/row/v2` output row of 64 words, `yPort`; the verifier's
two fixed nets (`sha512x3`, `hm96`) as parameters, as in `Proofs.Flock.SessionLoop.Concrete`. Its META is a JSON value
whose every digest is a spliced term: the program's SHA-512, the partition digest, the row key's SHA-512, the output
row's prefix, midstate and pad, and the dummy row's leaf string. `circuitFile` is the file at the verifier's own nets.

Piece P1 owns this file's values: it fixes the META's members to the ones `HmOut.parse` accepts.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)
open Flock

/-- The circuit's one row: its hidden output, a `sha512/row/v2` row of 64 words. -/
def yPort : Flock.Port := { name := "y", words := 64, chunks := 1, nb := 0, bits := 1024, v2 := true }

/-- The output row's 128 bytes at the zero value. -/
def zeroRow : ByteArray := ByteArray.mk (Array.replicate 128 0)

/-- The dummy row's leaf string, the row's zero value under a zero salt. -/
def dummyBc : ByteArray := HmOut.rowBcZeroSalt yPort zeroRow

/-- The unit's name in the circuit file's first line. -/
def unitName : String := "honest-unit"

/-- The circuit file's META members, in key order. -/
def metaPairs : List (String × Json) := [
  ("dummy", Json.mkObj [("bc", .arr #[.str (hexOf dummyBc)]), ("outputs", .arr #[jn 0])]),
  ("format", .str "flock-circuit"),
  ("input_ports", jn 0),
  ("k_log", jn 22),
  ("leaf_scheme", Flock.Tags.leafSchemeHm96Os),
  ("leaves_in", .arr #[.arr #[.num ⟨-1, 0⟩]]),
  ("leaves_out", .arr #[.arr #[jn 0]]),
  ("mask_log", jn 14),
  ("out_ports", .arr #[.arr #[.str "y", jn 16]]),
  ("partition", Json.mkObj [("digest", .str (hexOf partitionDigestB)), ("rule", .str HmRow.PARTITION_V1)]),
  ("per_vu", Json.mkObj [("hm96", jn 1), ("sha512x3", jn 1), ("unit", jn 1)]),
  ("ports", .arr #[.arr #[.str "y", jn 64, jn 1]]),
  ("program_digests", Json.mkObj [(templateId, .str (hexOf templateDigest))]),
  ("program_sha512", .str (hexOf programSha)),
  ("ranges", .arr #[
    Json.mkObj [("count", jn 1), ("net", .str "hm96"), ("pos0", jn 0), ("slot_log", jn 18), ("type", .str "unit")],
    Json.mkObj [("count", jn 1), ("net", .str "sha512x3"), ("pos0", jn 1), ("slot_log", jn 18), ("type", .str "unit")],
    Json.mkObj [("count", jn 1), ("net", .str "unit"), ("pos0", jn 1024), ("slot_log", jn 9), ("type", .str "unit")],
    Json.mkObj [("count", jn 1), ("pos0", jn 64), ("slot_log", jn 14), ("type", .str "mask")]]),
  ("row_key_sha512", .str (hexOf (Sha512.hash Hm96.Default512.key))),
  ("row_schemas", Json.mkObj [("y", .str HmRow.ROW_SCHEMA_V2)]),
  ("rows", .arr #[Json.mkObj [("midstate", .arr ((HmRow.midstateOf yPort).map fun w => Json.str (HmRow.hex16 w))),
    ("pad", .str (hexOf (HmRow.tailOf yPort))), ("prefix", .str (hexOf (HmRow.prefixOf yPort)))]]),
  ("scheme", .str "frame-v3-sha512/hm96-sha512"),
  ("statement", .str Flock.Tags.circuit.name),
  ("template", .str templateId),
  ("units_per_vu", jn 1),
  ("vus_per_block", jn 1),
  ("wires", .arr #[.arr #[.str "hm96", jn 0, jn 0, .str "sha512x3", jn 0, jn 0]])]

/-- The circuit file's META. -/
def metaJ : Json := Json.mkObj metaPairs

/-- The META line's text. -/
def metaText : String := Firewall.printJson metaJ

/-- The circuit file's first line. -/
def headLine : String := "flock-circuit " ++ unitName

/-- The circuit file's lines at the fixed nets `ls` (`sha512x3`) and `lh` (`hm96`). -/
def circuitLines (ls lh : HmNets.Laid) : List String :=
  RecDeploy.VStarNets.fileLs unitName metaText FlockDeploy.Concrete.unitLaid ls lh

/-- The circuit file at the fixed nets `ls` and `lh`. -/
def circuitFileOf (ls lh : HmNets.Laid) : ByteArray := (RecDeploy.VStarNets.joinNl (circuitLines ls lh)).toUTF8

/-- The registered circuit file, at the verifier's own nets. -/
def circuitFile : ByteArray := circuitFileOf (HmNets.shaLaid ()) (HmNets.hmLaid ())

end Verity.Sampled.Instance.HonestSession
