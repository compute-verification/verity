/-
Produced values (`one_stage/produced.py`): rows the prover computes after the draw from what it committed
before it, committed under hiding leaves and read as inputs by several statements of a session, each read opened against
the one root the record names. They are never outputs. H1's drawn leaf is proved as its 32 lanes and their combine
(`pouw.leaf_lanes`), which share the leaf's lane partials (`partials`, row l lane l).

* `domain`: a registered value's (`Registered.domainBound`) with `"produced": D` in its binding, D the digest of the
  registration record whose draw the values follow, so a produced tree opens under no other registration's domain and a
  registered value's root (whose binding never has `"produced"`) opens under none;
* `check`: the post-draw record `{"format", "registration", "draw", "produced": [{"value", "root", "leaves", "words",
  "domain"} ...], "openings": {value: [[position, b ‖ c] ...]}}` against the verifier's registration digest, its draw's
  digest and its own entries (`Mine`), Python's P- codes in its order, a value's codes in the order of the verifier's
  value names; `openings` is each row's commit-time opening by its digest `b ‖ c`, every row of each produced value and
  the read rows of the registered values the sessions read (`P-openings`);
* `opened`: the roots the verifiers of record report opening the produced values against (`Record.sessionRegistered`)
  must each be the accepted record's (`P-opened`);
* `wanted`, `shared`: every row the verifiers of record report opening (their verdicts' `openings`, `Flock.registeredOpenings`) of
  a value the verifier holds openings of must be that opening (`P-shared`), so the sessions that read a row read the one
  opening the record fixed, with no appeal to SHA-512's collision resistance;
* `known`: a port the verifier registers at known content (`produced.known_port`): the root and `b ‖ c` of its row under
  a known salt, the one row of a produced value's tree, so a session's row of it opens the root only as that content.

The verifier's own entries and every read are the protocol's, in Python (`produced.entry`, `pouw.leaf_lanes.reads`);
Flock's `--registered` opens them (`Flock.Registered`).
-/
import Flock.Registered
import SampledProofs.Registered

namespace SampledProofs.Produced

open Lean (Json)
open Flock (V)

def FORMAT : String := "verity/registration/produced/v1"
def KEYS : Array String := sortStrings #["draw", "format", "openings", "produced", "registration"]
/-- A row's hm96-sha512 commitment string `b ‖ c`, in bytes. -/
def BC_BYTES : Nat := 128
/-- A known row's salt, in bytes (`hm96`'s). -/
def SALT_BYTES : Nat := 192
/-- The domain binding's key a produced value adds: the registration's digest. -/
def PRODUCED_KEY : String := "produced"

/-- Lowercase hex of 64 bytes (Python's `_hex64`). -/
def hex64 (s : String) : Bool := s.length == 128 && isHex s

/-- Lowercase hex of a `b ‖ c` (Python's `_hex_bc`). -/
def hexBc (s : String) : Bool := s.length == 2 * BC_BYTES && isHex s

/-- A `[position, b ‖ c]` pair, the position an integer (Python's `type(r[0]) is int`). -/
def openingRow? : Json → Option (Int × String)
  | .arr #[p, .str bc] => (int? p).map (·, bc)
  | _ => none

/-- `produced.domain(program, value, words, leaves, word_bits, registration).domain_id`. -/
def domain (program value : String) (words leaves wordBits : Nat) (registration : String) : V ByteArray := do
  if !hex64 registration then throw "a registration digest is lowercase hex of 64 bytes"
  Registered.domainBound program value words leaves wordBits PRODUCED_KEY registration

/-- The verifier's own entry for one produced value (`produced.entries`). -/
structure Mine where
  value : String
  leaves : Nat
  words : Nat
  wordBits : Nat
  domain : String

/-- `produced._openings_ok`: `{value: [[position, b ‖ c] ...]}`, positions natural and distinct within a value and every
`b ‖ c` hex, and each of `mine`'s values with one opening at each of its rows, in order. -/
def openingsOk (ops : Json) (mine : Array Mine) : Bool :=
  match ops with
  | .obj _ =>
    let vs := entries ops
    vs.all (fun (_, rows) => match rows with
        | .arr rs =>
          let ps := rs.filterMap fun r => (openingRow? r).bind fun (p, bc) => if 0 ≤ p && hexBc bc then some p else none
          ps.size == rs.size && ps.toList.eraseDups.length == ps.size
        | _ => false) &&
      mine.all fun m => match vs.find? (·.1 == m.value) with
        | some (_, .arr rs) =>
          rs.size == m.leaves &&
            rs.map (fun r => (openingRow? r).map (·.1)) == (Array.range m.leaves).map fun (i : Nat) => some (i : Int)
        | _ => false
  | _ => false

/-- `produced.check`'s codes, empty when the record holds; `mine` sorted by value. -/
def check (rec : Json) (registration draw : String) (mine : Array Mine) : Array String := Id.run do
  if keys? rec != some KEYS then return #["P-keys"]
  let mut c : Array String := #[]
  if !jeq (getD rec "format") (Json.str FORMAT) then c := addCode c "P-format"
  if !jeq (getD rec "registration") (Json.str registration) then c := addCode c "P-registration"
  if !jeq (getD rec "draw") (Json.str draw) then c := addCode c "P-draw"
  let widthKeys := sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY)
  let .arr got := getD rec "produced" | return addCode c "P-produced"
  if !got.all (fun e => keys? e == some Registered.ENTRY_KEYS || keys? e == some widthKeys) then
    return addCode c "P-produced"
  let values := got.map (getD · "value")
  let names := values.filterMap fun v => match v with | .str s => some s | _ => none
  let distinct := names.toList.eraseDups
  if names.size != values.size || distinct.length != names.size || distinct.length != mine.size
      || !distinct.all (fun v => mine.any (·.value == v)) then
    c := addCode c "P-values"
  if !openingsOk (getD rec "openings") mine then c := addCode c "P-openings"
  for want in mine do
    -- Python's `{e["value"]: e}`: a value given twice reads as its last entry
    let some e := got.reverse.find? (fun e => jeq (getD e "value") (Json.str want.value)) | continue
    if !(match getD e "root" with | .str s => hex64 s | _ => false) then c := addCode c "P-root"
    if int? (getD e "leaves") != some (want.leaves : Int) then c := addCode c "P-leaves"
    if int? (getD e "words") != some (want.words : Int) then c := addCode c "P-words"
    if !jeq (getD e "domain") (Json.str want.domain) then c := addCode c "P-domain"
    let badWidth := match field? e Registered.WIDTH_KEY with
      | none => want.wordBits != Registered.WORD_BITS
      | some j => match int? j with
        | some w => w != want.wordBits || w == Registered.WORD_BITS
        | none => true
    if badWidth then c := addCode c "P-word_bits"
  return c

/-- `produced.opened`: `#["P-opened"]` unless every produced value of the accepted record `rec` was opened against its
root, `opened` the `{value: root}` the verifiers of record report (none: they report none). -/
def opened (rec : Json) (opened : Option (Array (String × Json))) : Array String :=
  let got := opened.getD #[]
  let want := match getD rec "produced" with | .arr es => es | _ => #[]
  let ok := want.all fun e => match got.find? (fun (v, _) => jeq (Json.str v) (getD e "value")) with
    | some (_, x) => jeq x (getD e "root")
    | none => jeq Json.null (getD e "root")
  if ok then #[] else #["P-opened"]

/-- `produced.wanted`: the openings the verifier holds, the accepted record's and its own ports' (`known`), a value in both
read as `known`'s. -/
def wanted (rec : Json) (known : Array (String × Array (Nat × String))) : Array (String × Array (Nat × String)) :=
  let ofRec := (entries (getD rec "openings")).filterMap fun (v, rows) =>
    match rows with
    | .arr rs => if known.any (·.1 == v) then none else
        some (v, rs.filterMap fun r => (openingRow? r).bind fun (p, bc) => if 0 ≤ p then some (p.toNat, bc) else none)
    | _ => none
  known ++ ofRec

/-- `produced.shared`: `#["P-shared"]` unless every row each accepted verdict of record reports opening (its `openings`,
`{port: [[position, b ‖ c] ...]}`, the port's value its `registered`'s) of a value `want` holds openings of is `want`'s
opening at that position: a port of such a value with no reported rows, a row at a position `want` has no opening for,
or another opening, fails. `want`: the record's openings and the verifier's own ports' (`produced.wanted`). -/
def shared (want : Array (String × Array (Nat × String))) (verdicts : Array (String × Json)) : Array String := Id.run do
  for (_, v) in verdicts do
    if !(truthy (getD v "of_record") && jeq (getD v "accepted") (.bool true)) then continue
    let reg := getD v "registered"
    let .obj _ := reg | continue
    let ops := getD v "openings"
    for (port, e) in entries reg do
      let .str value := getD e "value" | continue
      let some (_, rows) := want.find? (·.1 == value) | continue
      let got := match ops with | .obj _ => getD ops port | _ => Json.null
      let .arr rs := got | return #["P-shared"]
      if rs.isEmpty then return #["P-shared"]
      for r in rs do
        let ok := match openingRow? r with
          | some (p, bc) => rows.any fun (q, b) => (q : Int) == p && b == bc
          | none => false
        if !ok then return #["P-shared"]
  return #[]

/-- A row's `sha512/row/v2` digest (`registered.row_digest` at `SCHEMA_V2`): SHA-512 of `P(16 · words)` and its u16 words,
each little-endian. -/
def v2Digest (u16s : Array Nat) : ByteArray :=
  let bytes := u16s.foldl (fun b w => (b.push (w % 256).toUInt8).push (w / 256 % 256).toUInt8) ByteArray.empty
  Flock.Sha512.hash (Flock.HmRow.rowPrefixV2 (16 * u16s.size) ++ bytes)

/-- `produced.known_port`'s root and opening `b ‖ c`: `row`, leaves of `wordBits` bits, committed under the known `salt`
as the one `hm96-sha512/row/v2` row of a produced value's tree under `registration`, whose root is its one leaf. -/
def known (program value : String) (row : Array Nat) (salt : ByteArray) (registration : String) (wordBits : Nat) :
    V (ByteArray × ByteArray) := do
  if row.isEmpty then throw "a registered value is at least one row, every row the same width"
  if salt.size != SALT_BYTES then throw s!"one {SALT_BYTES}-byte salt per row"
  if !Registered.WIDTHS.contains wordBits then throw s!"a registered value's leaves are 16 or 32 bits, not {wordBits}"
  if row.any (· ≥ 2 ^ wordBits) then throw s!"a registered leaf does not fit {wordBits} bits"
  let dom ← domain program value row.size 1 wordBits registration
  let u16s := row.foldl (fun acc v => acc ++ (Array.range (wordBits / 16)).map fun i => v / 2 ^ (16 * i) % 65536) #[]
  let bc := Flock.Hm96.Default512.commitString (v2Digest u16s) salt
  return (Flock.Registered.leafAs Flock.HmRow.ROW_SCHEMA_V2 dom 0 (Flock.Hm96.Default512.treeLeaf bc), bc)

end SampledProofs.Produced
