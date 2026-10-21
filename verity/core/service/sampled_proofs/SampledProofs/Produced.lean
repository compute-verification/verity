/-
Produced values (`one_stage/produced.py`, PRF-134): a value one statement of a session outputs and another reads,
committed after the draw, each read opened against the one root it was output under. H1's leaf proves its seeds, lanes
and combine as one statement each (`pouw.circuit.h1lane`): the seed pair and the lanes' partials are produced values.

* `domain`: a registered value's (`Registered.domainBound`) with `"produced": D` in its binding, D the digest of the
  registration record whose draw the values follow, so a produced tree opens under no other registration's domain and a
  registered value's root (whose binding never has `"produced"`) opens under none;
* `check`: the post-draw record `{"format", "registration", "draw", "produced": [{"value", "root", "leaves", "words",
  "domain"} ...]}` against the verifier's registration digest, its draw's digest and its own entries (`Mine`), Python's
  P- codes in its order, a value's codes in the order of the verifier's value names;
* `opened`: the roots the verifiers of record report opening the produced values against (`Record.sessionRegistered`)
  must each be the accepted record's (`P-opened`).

The verifier's own entries and every read are derived from its Program in Python (`produced.entries`, `produced.reads`),
as the registered values' reads are (`registered.reads_file`); Flock's `--registered` opens them (`Flock.Registered`).
-/
import SampledProofs.Registered

namespace SampledProofs.Produced

open Lean (Json)
open Flock (V)

def FORMAT : String := "verity/registration/produced/v0"
def KEYS : Array String := sortStrings #["draw", "format", "produced", "registration"]
/-- The domain binding's key a produced value adds: the registration's digest. -/
def PRODUCED_KEY : String := "produced"

/-- Lowercase hex of 64 bytes (Python's `_hex64`). -/
def hex64 (s : String) : Bool := s.length == 128 && isHex s

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

end SampledProofs.Produced
