import Flock.Canon
import Flock.Hash

/-!
# A one-stage window record's digest

A one-stage window record (format `verity/registration/v1` or `verity/registration/v2`) is named by
`SHA-512(format ‖ 0 ‖ canon(record))`, its own format the tag. This is that digest's one definition, in C-Flock's
verifier so that every reader reaches it without a new requirement: PoUW's check (`PouwCheck.Registration.digest`),
the network check (`NetworkCheck.Period.answered`), sampled proofs' registration and audit record
(`SampledProofs.Registration`, `SampledProofs.Record`) and C-Flock's recursive setup (`RecVStar.registration`) read it
here. Python's `registration.digest` is its mirror on records whose numbers are integers and whose characters are all
below 0x7f: from DEL on, Python's `canonical_json` escapes a character that `canon` writes as it is.
-/

namespace Flock.WindowRecord

open Lean (Json)

def FORMAT : String := "verity/registration/v1"
def FORMAT_V2 : String := "verity/registration/v2"
def FORMATS : List String := [FORMAT, FORMAT_V2]

/-- `SHA-512(format ‖ 0 ‖ canon)`: the digest of a record of format `format` whose canonical JSON is the bytes `canon`. -/
def digestOf (format : String) (canon : ByteArray) : ByteArray :=
  Sha512.hash (format.toUTF8 ++ ByteArray.mk #[0] ++ canon)

/-- The record's own tag: `FORMAT_V2` when its `format` is that string, else `FORMAT`. -/
def tagOf (rec : Json) : String :=
  if (rec.getObjValAs? String "format").toOption == some FORMAT_V2 then FORMAT_V2 else FORMAT

/-- The record's digest: `digestOf` its own tag (`tagOf`) and its canonical JSON (`Flock.canon`). -/
def digest (rec : Json) : V ByteArray := do
  return digestOf (tagOf rec) (← canon rec).toUTF8

/-- `digest`, as lowercase hex. -/
def digestHex (rec : Json) : V String := return hexOf (← digest rec)

end Flock.WindowRecord
