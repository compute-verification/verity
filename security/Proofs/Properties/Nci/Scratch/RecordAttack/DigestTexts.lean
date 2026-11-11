import Proofs.Properties.Nci.Record

/-!
# Two Program texts whose `callees` differ in order

A scratch file, never merged, and evaluated, not proved: each `#eval` runs the compiled decoder and digest. The two
texts have the same `definitions`: `Root_v1` calls `K_v1`, `K_v1` calls callee 0, `B_v1` calls `C_v1`, and `A_v1`
and `C_v1` have no node. Callee 0 is `A_v1` in one text and `B_v1` in the other.

* `calleesOf`: the Definitions the decoder gives `K_v1`'s nodes.
* `digestOf`: `K_v1`'s closure digest, as the compiled digest computes it.
* `rawOf`: the same digest where the form's comparison answers no: the SHA-512 of the one-entry object that maps
  `K_v1` to the SHA-512 of its own descriptor entry.
-/

namespace NciCheck.Properties.Record.Attack.Texts

open Lean (Json)
open NciCheck.DefinitionDigest (closureDigest)

/-- A composite of no parameter and no node, returning the empty tuple. -/
def bare (id : String) : String :=
  "\"" ++ id ++ "\":{\"bindings\":{},\"body\":{\"nodes\":[],\"params\":[],\"ret\":{\"concat\":[]}," ++
  "\"ret_type\":[\"Tuple\",[]]},\"form\":\"composite\",\"id\":\"" ++ id ++ "\"}"

/-- A composite of no parameter whose one node calls callee `fn`, returning the empty tuple. -/
def calling (id : String) (fn : Nat) : String :=
  "\"" ++ id ++ "\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":" ++ toString fn ++
  ",\"form\":\"call\",\"out\":0}],\"params\":[],\"ret\":{\"concat\":[]}," ++
  "\"ret_type\":[\"Tuple\",[]]},\"form\":\"composite\",\"id\":\"" ++ id ++ "\"}"

/-- The Program's text with the interning table `callees`. -/
def text (callees : String) : String :=
  "{\"annotations\":{},\"callees\":" ++ callees ++ ",\"definitions\":{" ++ bare "A_v1" ++ "," ++
  calling "B_v1" 3 ++ "," ++ bare "C_v1" ++ "," ++ calling "K_v1" 0 ++ "," ++ calling "Root_v1" 2 ++
  "},\"program_fn\":\"Root_v1\",\"root\":\"Root_v1\",\"schema\":\"verity-ir/descriptor/v1\"," ++
  "\"types\":[[\"Tuple\",[]]]}"

def textA : String := text "[\"A_v1\",\"B_v1\",\"K_v1\",\"C_v1\"]"

def textB : String := text "[\"B_v1\",\"A_v1\",\"K_v1\",\"C_v1\"]"

/-- The Definitions the decoder gives `K_v1`'s nodes, with each one's gates. -/
def calleesOf (t : String) : Except String (List (String × Nat)) := do
  let P ← Flock.Program.decode t.toUTF8
  let some k := P.index["K_v1"]? | throw "the decoder lists no K_v1"
  let some d := P.defs[k]? | throw "the decoder holds no K_v1"
  pure (d.body.nodes.toList.map fun n => ((P.defs[n.fn]!).key, (P.defs[n.fn]!).body.nodes.size))

/-- `K_v1`'s closure digest, as the compiled digest computes it. -/
def digestOf (t : String) : Except String String :=
  match Flock.JsonParse.parse t with
  | .ok j => closureDigest j "K_v1"
  | .error e => .error e

/-- `K_v1`'s closure digest where the form's comparison answers no. -/
def rawOf (t : String) : Except String String :=
  match Flock.JsonParse.parse t with
  | .ok j => do
    let e := (j.getObjValD "definitions").getObjValD "K_v1"
    let d := Flock.hexOf (Flock.Sha512.hash (← Flock.canon e).toUTF8)
    pure (Flock.hexOf (Flock.Sha512.hash (← Flock.canon (Json.mkObj [("K_v1", .str d)])).toUTF8))
  | .error e => .error e

def same (a b : Except String String) : String :=
  match a, b with
  | .ok x, .ok y => if x == y then "one digest" else "two digests"
  | _, _ => "no digest"

#eval calleesOf textA
#eval calleesOf textB
#eval digestOf textA
#eval digestOf textB
#eval same (digestOf textA) (digestOf textB)
#eval rawOf textA
#eval rawOf textB
#eval same (rawOf textA) (rawOf textB)

end NciCheck.Properties.Record.Attack.Texts
