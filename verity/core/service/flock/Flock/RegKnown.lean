/-
`Flock.HmRow` (registered reads): a value the verifier registers itself at public content. Its `--registered` port
entry names the content under `known` (each row's u16 words and its salt), and the verifier computes the value's
registered root from it and refuses a file whose stated root is another (`checkKnownAll`, which `flock-verify` runs on
`--registered F` before any statement: `FlockVerify.registeredInput`). A registrant's row that opens the root is then
one of those rows, or SHA-512 collides. h1_prove's `ok` rows are such a value: row `u` the one word `1` under the public
`ok_salt(window, u)` (`h1unit.register_ok`).
-/
import Flock.Registered

namespace Flock
namespace Registered

open Lean (Json)

/-- The key of a port entry the verifier registers at known content: `{"rows": [[w, …], …], "salts": [hex, …]}`, row
`i`'s u16 words and its salt. -/
def KNOWN_KEY : String := "known"

/-- `sha512/row/v2`'s message for a bit row of u16 words: `P(16 · words)`, then each word low byte first
(`registered.row_digest` under `hm96-sha512/row/v2` at 16-bit leaves). -/
def rowBytesV2 (row : Array Nat) : ByteArray :=
  row.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8) (HmRow.rowPrefixV2 (16 * row.size))

/-- The hm96 leaf of a known row: of `row`'s `sha512/row/v2` digest under `salt` (`registered.Registered.commit`). -/
def knownPayload (row : Array Nat) (salt : ByteArray) : ByteArray :=
  Hm96.Default512.treeLeaf (Hm96.Default512.commitString (Sha512.hash (rowBytesV2 row)) salt)

/-- The frame-v3-sha512 leaves of a value registered under `reg` at known content `ks`: row `i`'s hm96 leaf, framed at
`i` under `reg`'s domain and schema. -/
def knownLeaves (reg : Port) (ks : Array (Array Nat × ByteArray)) : Array ByteArray :=
  (Array.range ks.size).map fun i => leafAs reg.schema reg.domain i (knownPayload ks[i]!.1 ks[i]!.2)

/-- **A port the verifier registers at known content `ks` commits it**: its rows are `hm96-sha512/row/v2` rows of 16-bit
leaves, as many as its leaves, each of the port's `words` u16 words under a 192-byte salt, and its registered root is
the frame-v3-sha512 root of their leaves (`knownLeaves`). -/
def checkKnown (reg : Port) (ks : Array (Array Nat × ByteArray)) : V Unit :=
  if reg.schema != HmRow.ROW_SCHEMA_V2 || reg.wordBits != 16 then
    throw s!"registered reads: port {reg.name}: known content is of {HmRow.ROW_SCHEMA_V2} rows of 16-bit leaves, and {reg.value}'s are {reg.schema} rows of {reg.wordBits}-bit leaves"
  else if ks.size != reg.leaves then
    throw s!"registered reads: port {reg.name}: {ks.size} known rows of {reg.value}'s {reg.leaves}"
  else if ks.any fun (r, s) => some r.size != reg.words || r.any (· ≥ 2 ^ 16) || s.size != HmRow.SALT_BYTES then
    throw s!"registered reads: port {reg.name}: a known row is not {reg.words.getD 0} u16 words under a {HmRow.SALT_BYTES}-byte salt"
  else if FV3.root reg.domain (knownLeaves reg ks) Sha512.hash != reg.root then
    throw s!"registered reads: port {reg.name}: {reg.value}'s registered root is not the root of its {ks.size} known rows"
  else pure ()

/-- One port entry's known content (`KNOWN_KEY`): one salt a row. -/
def knownOfEntry (e : Json) (ctx : String) : V (Array (Array Nat × ByteArray)) := do
  let rows ← (← J.arr e "rows" ctx).mapM fun r => do
    (← J.asArr r s!"{ctx}: a row").mapM (J.asNat · s!"{ctx}: a row's word")
  let salts ← (← J.arr e "salts" ctx).mapM fun s => do unhex (← J.asStr s s!"{ctx}: a salt")
  if rows.size != salts.size then throw s!"{ctx}: {rows.size} rows and {salts.size} salts"
  return rows.zip salts

/-- **The known content a `--registered` file names**: each port entry with a `known` key, by port name. -/
def knownOf (j : Json) : V (Array (String × Array (Array Nat × ByteArray))) := do
  let .ok (.obj kvs) := j.getObjVal? "ports" | throw "registered reads: ports is not an object"
  kvs.toArray.filterMapM fun (name, e) => do
    if J.has e KNOWN_KEY then
      return some (name, ← knownOfEntry (← J.get e KNOWN_KEY "registered reads") s!"registered reads: port {name}: known")
    else return none

/-- **Every port named at known content commits it**: each `(name, ks)` of `known` is `own`'s first port of that name
(the one the registered check reads), and `checkKnown` accepts it at `ks`. -/
def checkKnownAll (own : Array Port) (known : Array (String × Array (Array Nat × ByteArray))) : V Unit :=
  known.toList.forM fun (name, ks) => do
    let some reg := own.toList.find? (·.name == name)
      | throw s!"registered reads: known content for port {name}, which the file does not register"
    checkKnown reg ks

end Registered
end Flock
