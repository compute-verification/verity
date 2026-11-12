/-
`Flock.HmRow` (registered reads): a value the verifier registers itself at public content. Its `--registered` port
entry names the content under `known` (each row's u16 words and its salt), and the verifier computes the value's
registered root from it and refuses a file whose stated root is another (`checkKnownAll`, which `flock-verify` runs on
`--registered F` before any statement: `FlockVerify.registeredInput`). A registrant's row that opens the root is then
one of those rows, or SHA-512 collides. h1_prove's `ok` rows are such a value: row `u` the one word `1` under the public
`ok_salt(window, u)` (`h1unit.register_ok`).

A statement publishes a tree of its own for such a port, over its instances' rows at its own domain. The verifier
computes that tree too (`publishedRoot`), and lists a statement's known trees for the ZK server's release side
(`knownTrees`, `knownList`; `flock-verify known-trees`), which holds the statement's trees to the list
(`Firewall.Salts.readsKnown`).
-/
module

public import Flock.Registered

@[expose] public section

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

/-! ### The trees a statement publishes at known content -/

/-- The value's rows the table of port `reg` holds in a statement whose instances are the units `indices`: instance
`i`'s row is the row the verifier's program reads in unit `indices[i]` (`Port.readAt`), in instance order. Refused
where the program reads no row of the value in one of the units, and where two instances read one row: an input
port's table then holds that row once and an output row's table once an instance (`registered.table`), and this reads
no port's direction. -/
def tableRows (reg : Port) (indices : Array Nat) : V (Array Nat) := do
  let rows ← indices.mapM fun u => match reg.readAt u with
    | some (some p) => pure p
    | _ => throw s!"known trees: port {reg.name}: the verifier's program reads no row of {reg.value} in unit {u}"
  let sorted := rows.qsort (· < ·)
  match (List.range (sorted.size - 1)).find? (fun i => sorted[i]! == sorted[i + 1]!) with
  | some i =>
    throw s!"known trees: port {reg.name}: two instances read row {sorted[i]!} of {reg.value}, and its table's layout is then its port's direction's"
  | none => return rows

/-- **The tree a statement publishes for a port at known content**: the frame-v3-sha512 root, at the statement's
published domain `dom` for the port, of the table whose row `r` is the value's known row `rows[r]` (`ks`: its words
and its salt), each leaf framed at `r` under `reg`'s schema, as `PortRoots.pubRoot` roots the public file's `b ‖ c`.
The verifier computes it from its own registered file and the statement's layout, and reads no row of the stack's. -/
def publishedRoot (reg : Port) (dom : ByteArray) (ks : Array (Array Nat × ByteArray)) (rows : Array Nat) : ByteArray :=
  FV3.root dom ((Array.range rows.size).map fun r =>
    let (row, salt) := ks.getD rows[r]! (#[], ByteArray.empty)
    leafAs reg.schema dom r (knownPayload row salt)) Sha512.hash

/-- A known tree's entry in the verifier's list of a statement's known trees: `D ‖ be64 N ‖ u8 schema ‖ ROOT`, the
published tree's domain, row count, schema (its index in `SCHEMAS`) and root. -/
def knownEntry (dom : ByteArray) (n : Nat) (schema : String) (root : ByteArray) : ByteArray :=
  dom ++ Flock.be64 n.toUInt64 ++ ByteArray.mk #[((SCHEMAS.findIdx? (· == schema)).getD SCHEMAS.length).toUInt8] ++ root

/-- **The verifier's list of a statement's known trees**, which follows the draw in its answer to the session's
`Register` (`Firewall.Salts.parseKnown` reads it): `be32 K`, then the `K` entries. -/
def knownList (entries : Array ByteArray) : ByteArray :=
  entries.foldl (· ++ ·) (Flock.be32 entries.size.toUInt32)

/-- **The known trees of the statement whose public file's header is `header`**, by port, each its entry
(`knownEntry`): for each port of `own` named at known content in `known` that the header publishes a tree for (a
domain under `frame_v3.domain_ids`), the tree over its instances' rows (`tableRows` at the header's `units.indices`,
`publishedRoot`) at that domain. A port at known content that the header publishes no tree for has no entry. -/
def knownTrees (own : Array Port) (known : Array (String × Array (Array Nat × ByteArray))) (header : Json) :
    V (Array (String × ByteArray)) := do
  let ctx := "public file"
  let indices ← (← J.arr (← J.get header "units" ctx) "indices" ctx).mapM (J.asNat · "a unit index")
  let doms ← J.get (← J.get header "frame_v3" ctx) "domain_ids" s!"{ctx}.frame_v3"
  known.filterMapM fun (name, ks) => do
    if !J.has doms name then return none
    let some reg := own.toList.find? (·.name == name)
      | throw s!"known trees: known content for port {name}, which the file does not register"
    let dom ← unhex (← J.str doms name s!"{ctx}.frame_v3.domain_ids")
    if dom.size != 64 then throw s!"known trees: port {name}: a published domain of {dom.size} bytes, not 64"
    let rows ← tableRows reg indices
    if let some p := rows.find? (· ≥ ks.size) then
      throw s!"known trees: port {name}: the verifier's program reads row {p} of {reg.value}, and its known rows are {ks.size}"
    return some (name, knownEntry dom rows.size reg.schema (publishedRoot reg dom ks rows))

/-- `knownTrees` from the verifier's registered file `registered`, each of its ports at known content held to its
content's root first (`checkKnownAll`). -/
def knownTreesOf (registered header : Json) : V (Array (String × ByteArray)) := do
  let own ← ofJson registered
  let known ← knownOf registered
  checkKnownAll own known
  knownTrees own known header

end Registered
end Flock
