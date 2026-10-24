/-
`HmRow.checkPartition` refuses a statement bound to `Q_call` v1, retired from the verifier of record, while the form check
still takes the object (sampled proofs' `Partition.valid` reads it), and keeps deriving nothing for a `Q_word` v1 binding
it is not given the program for.
-/
import Flock.HmRow

namespace Flock.Test.QcallRetired

open Lean (Json)
open Flock

def program : String := "".pushn 'a' 128

def obj (name : String) (params : List (String × Json)) : Json :=
  Json.mkObj [("format", Json.str HmRow.PARTITION_V1), ("program", Json.str program),
    ("query", Json.mkObj [("name", Json.str name), ("version", Lean.toJson (1 : Nat)), ("params", Json.mkObj params)])]

def qcall : Json := obj "Q_call" [("X", Lean.toJson (16 : Nat))]
def qword : Json := obj "Q_word" [("X", Lean.toJson (16 : Nat)), ("W", Lean.toJson (32 : Nat))]

/-- A circuit whose META binds `o`'s digest over `program`. -/
def bound (o : Json) : Circuit :=
  let digest := match HmRow.partitionDigest o with
    | .ok d => hexOf d
    | .error _ => ""
  { (default : Circuit) with metaJ := Json.mkObj [("program_sha512", Json.str program),
      ("partition", Json.mkObj [("rule", Json.str HmRow.PARTITION_V1), ("digest", Json.str digest)])] }

#guard (HmRow.checkPartitionObject qcall).isOk
#guard !(HmRow.checkPartitionObject (obj "Q_call" [("X", Lean.toJson (70000 : Nat))])).isOk
#guard match HmRow.checkPartition (bound qcall) (some qcall) with
  | .error e => e == "partition: Q_call v1 is retired from the verifier of record; bind the circuit to a template query"
  | .ok _ => false
#guard match HmRow.checkPartition (bound qword) (some qword) with
  | .ok f => f.query == "Q_word v1" && !f.derived
  | .error _ => false

end Flock.Test.QcallRetired
