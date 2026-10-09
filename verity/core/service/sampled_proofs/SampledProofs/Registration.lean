/-
The registration record (`one_stage/registration.py`): what serving is committed to, fixed before the verifier draws, its
digest, and R1–R7 against the verifier's own copies (`check`).

    {"format": "verity/registration/v1", "protocol": "verity/sampled-proofs/one-stage/v1",
     "program": "<SHA-512 of the program>", "query": "<SHA-512 of the query>", "partition": "<partition digest>",
     "population": <units the query derives>, "law": <the law, expanded>, "scheme": "<leaf and tree scheme>",
     "roots": [{"port", "root", "leaves", "domain"}, ...], "leaf_layer": "<SHA-512 of the verifier's public file>" | null,
     "anchors": [], "window": {"kind": "served" | "stand-in", ...provenance}, "previous": null | "<digest>"}

A v2 record (`verity/registration/v2`) adds `"registered": [{"value", "root", "leaves", "words", "domain"}, ...]` (with
`"word_bits"` for a value whose leaves are not 16 bits), and R7 holds it to the verifier's own entries
(`Registered.entries`) and each value an earlier record of the log registered to that record's root (`heldRoots`). The
verifier expects a v2 record exactly when it expects values registered.
-/
import SampledProofs.Draw
import SampledProofs.Registered

namespace SampledProofs.Registration

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc)

def FORMAT : String := "verity/registration/v1"
def FORMAT_V2 : String := "verity/registration/v2"
def PROTOCOL : String := "verity/sampled-proofs/one-stage/v1"
def KINDS : List String := ["served", "stand-in"]

def KEYS : Array String := sortStrings #["format", "protocol", "program", "query", "partition", "population", "law",
  "scheme", "roots", "leaf_layer", "anchors", "window", "previous"]
def KEYS_V2 : Array String := sortStrings (KEYS.push "registered")
def ROOT_KEYS : Array String := sortStrings #["port", "root", "leaves", "domain"]

/-- `SHA-512(format || 0 || canon(record))`, the record's own format (v1 or v2) as the tag. -/
def digest (rec : Json) : V ByteArray := do
  let tag := if jeq (getD rec "format") FORMAT_V2 then FORMAT_V2 else FORMAT
  return Flock.Sha512.hash (tag.toUTF8.push 0 ++ (← Flock.canon rec).toUTF8)

def digestHex (rec : Json) : V String := return Flock.hexOf (← digest rec)

/-- `registration.query_digest`: SHA-512 of the canonical query. -/
def queryDigest (q : Json) : V String := canonSha512 q

/-- The verifier's own copies that R1–R7 hold the record to. -/
structure Verifier where
  program : Desc
  query : Json
  /-- the agreed law: a rule, which `Draw.derive` expands -/
  law : Json
  workTable : Option Json
  scheme : String
  /-- each port its layout derives, with its `{"leaves", "domain"}` -/
  ports : Array (String × Json)
  leafLayer : Option String
  anchors : Array Json
  /-- this prover's earlier records, oldest first -/
  log : Array Json
  /-- its own partition object, when it isn't the canonical one (a record made before the canonical envelope) -/
  partition : Option Json
  /-- the parameters it expects registered, `(parameter, words)` each; none: a v1 record -/
  registered : Array (String × Nat) := #[]
  /-- its program function's parameters, which `registered` names -/
  params : Array Registered.Param := #[]

private def optStr : Option String → Json
  | some s => Json.str s
  | none => Json.null

/-- `registration.held_roots`: `(value, root)` for every value an earlier record of the log registered, the root of its
first registration. -/
def heldRoots (log : Array Json) : Array (String × String) :=
  log.foldl (fun held rec => match getD rec "registered" with
    | .arr es => es.foldl (fun held e => match getD e "value", getD e "root" with
      | .str v, .str r => if held.any (·.1 == v) then held else held.push (v, r)
      | _, _ => held) held
    | _ => held) #[]

/-- `registration._check_registered`, R7: the record's `registered` against the verifier's own entries (`mine`, an error
where `registered_entries` raises) and the roots the log holds. -/
def checkRegistered (c : Array String) (got : Json) (mine : V (Array Registered.Entry))
    (held : Array (String × String)) : Array String := Id.run do
  let .ok mine := mine | return addCode c "R7-registered"
  let widthKeys := sortStrings (Registered.ENTRY_KEYS.push Registered.WIDTH_KEY)
  let .arr got := got | return addCode c "R7-registered"
  if !got.all (fun e => keys? e == some Registered.ENTRY_KEYS || keys? e == some widthKeys) then
    return addCode c "R7-registered"
  let mut c := c
  -- Python's `{e["value"]: e}`: a value that isn't a string is none of the verifier's, which are parameter names
  let values := got.map (getD · "value")
  let names := values.filterMap fun v => match v with | .str s => some s | _ => none
  let distinct := names.toList.eraseDups
  if names.size != values.size || distinct.length != values.size || distinct.length != mine.size
      || !distinct.all (fun v => mine.any (·.value == v)) then
    c := addCode c "R7-values"
  for want in mine do
    -- a value registered twice reads as its last entry, as Python's dict does
    let some e := got.reverse.find? (fun e => jeq (getD e "value") (Json.str want.value)) | continue
    if !(match getD e "root" with | .str s => s.length == 128 && isHex s | _ => false) then c := addCode c "R7-root"
    -- a count is an integer, never a boolean equal to it
    if int? (getD e "leaves") != some (want.leaves : Int) then c := addCode c "R7-leaves"
    if int? (getD e "words") != some (want.words : Int) then c := addCode c "R7-words"
    if !jeq (getD e "domain") (Json.str want.domain) then c := addCode c "R7-domain"
    -- 16 is named by leaving word_bits out: one record digest per registration
    let badWidth := match field? e Registered.WIDTH_KEY with
      | none => want.wordBits != Registered.WORD_BITS
      | some j => match int? j with
        | some w => w != want.wordBits || w == Registered.WORD_BITS
        | none => true
    if badWidth then c := addCode c "R7-word_bits"
    if let some (_, r) := held.find? (·.1 == want.value) then
      if !jeq (getD e "root") (Json.str r) then c := addCode c "R7-held"
  return c

/-- `registration.check`'s failure codes, R1–R7 in order: empty when the record holds. -/
def check (rec : Json) (v : Verifier) : Array String := Id.run do
  let v2 := !v.registered.isEmpty
  if keys? rec != some (if v2 then KEYS_V2 else KEYS) then return #["keys"]
  let get := getD rec
  let mut c : Array String := #[]
  -- R1
  if !(jeq (get "format") (if v2 then FORMAT_V2 else FORMAT) && jeq (get "protocol") PROTOCOL) then
    c := addCode c "R1-protocol"
  match Draw.derive v.program v.query v.law v.workTable with
  | .error _ => c := addCode c "R1-law"
  | .ok mine => if !Draw.sameLaw (get "law") mine then c := addCode c "R1-law"
  let w := get "window"
  if !(w matches .obj _) || !KINDS.any (jeq (getD w "kind") ·) then c := addCode c "window"
  -- R2
  let program := Flock.hexOf v.program.sha512
  let part := v.partition.getD (Partition.build program v.query)
  if !jeq (getD part "program") program then c := addCode c "R2-program"
  let partDigest := Partition.digest part
  if partDigest.isNone then c := addCode c "R2-partition"
  let mine : List (String × Option Json) :=
    [("program", some (getD part "program")), ("query", (queryDigest (getD part "query")).toOption.map Json.str),
     ("partition", partDigest.map Json.str)]
  for (name, m) in mine do
    if let some m := m then
      if !jeq (get name) m then c := addCode c s!"R2-{name}"
  match Partition.units v.program v.query with
  | .error _ => c := addCode c "R2-query"
  | .ok population => if !jeq (get "population") (Lean.toJson population) then c := addCode c "R2-population"
  -- R3
  if !jeq (get "scheme") v.scheme then c := addCode c "R3-scheme"
  let roots := match get "roots" with | .arr rs => rs | _ => #[]
  if roots.isEmpty || !roots.all (keys? · == some ROOT_KEYS) then c := addCode c "R3-roots"
  else
    let distinct := roots.foldl (fun (acc : Array Json) r =>
      if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")) #[]
    if distinct.size != roots.size || distinct.size != v.ports.size
        || !v.ports.all (fun (p, _) => distinct.any (jeq · p)) then
      c := addCode c "R3-ports"
    for (p, want) in v.ports do
      -- a port registered twice reads as its last root, as Python's dict does
      let some r := roots.reverse.find? (fun r => jeq (getD r "port") p) | continue
      if !(match getD r "root" with | .str s => isHex s | _ => false) then c := addCode c "R3-root"
      if !jeq (getD r "leaves") (getD want "leaves") then c := addCode c "R3-leaves"
      if !jeq (getD r "domain") (getD want "domain") then c := addCode c "R3-domain"
  -- R4: when the verifier holds the leaf layer, the record must name it (a null leaf layer is refused)
  if !jeq (get "leaf_layer") (optStr v.leafLayer) then c := addCode c "R4-leaf-layer"
  -- R5
  if !jeq (get "anchors") (Json.arr v.anchors) then c := addCode c "R5-anchors"
  -- R6
  let previous := match v.log.back? with
    | some e => (digestHex e).toOption.map Json.str |>.getD Json.null
    | none => Json.null
  if !jeq (get "previous") previous then c := addCode c "R6-previous"
  if v.log.any (fun e => jeq (getD e "window") (get "window")) then c := addCode c "R6-window"
  -- R7
  if v2 then
    c := checkRegistered c (get "registered") (Registered.entries v.program v.params v.registered) (heldRoots v.log)
  return c

end SampledProofs.Registration
