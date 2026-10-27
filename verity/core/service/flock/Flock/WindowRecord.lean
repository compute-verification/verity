/-
One-stage's window record (`verity/registration/v1|v2`, `one_stage/registration.py`): the record a window commits to,
which sampled proofs, C-Flock's META, PoUW, NCI and the network check all read, and its one digest,
`SHA-512(format ‖ 0 ‖ canon(record))` with the record's own format as the tag: v2 when its `format` is the string
`verity/registration/v2`, else v1, as `registration.digest` tests it.

`canon` is `Flock.canon`, which writes non-ASCII as UTF-8 and refuses a non-integer number, while Python's
`canonical_json` escapes every character outside `' '..'~'`. The two agree on a record whose strings hold only
characters below 0x7f and whose numbers are integers, nested less than a thousand deep.
-/
import Flock.Canon
import Flock.Hash

namespace Flock.WindowRecord

open Lean (Json)

def FORMAT : String := "verity/registration/v1"
def FORMAT_V2 : String := "verity/registration/v2"
def FORMATS : List String := [FORMAT, FORMAT_V2]

/-- `SHA-512(format ‖ 0 ‖ canon)`, over a record's canonical bytes. -/
def digestOf (format : String) (canon : ByteArray) : ByteArray :=
  Sha512.hash (format.toUTF8 ++ ByteArray.mk #[0] ++ canon)

/-- The record's own format as its digest's tag: v2 when its `format` is the string v2, else v1. -/
def tagOf (rec : Json) : String :=
  if (rec.getObjValAs? String "format").toOption == some FORMAT_V2 then FORMAT_V2 else FORMAT

def digest (rec : Json) : V ByteArray := do
  return digestOf (tagOf rec) (← canon rec).toUTF8

def digestHex (rec : Json) : V String := return hexOf (← digest rec)

end Flock.WindowRecord
