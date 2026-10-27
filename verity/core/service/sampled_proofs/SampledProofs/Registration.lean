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
(`Registered.entries`), a value the verifier pinned before the record to its root (`pinned`: the weights' root recorded
before the salt), and each value an earlier record of the log registered under the domain it takes now to that record's
root (`heldRoots`). The verifier expects a v2 record exactly when it expects values registered. Its entry for a value
counts the value's rows in the row format the verifier holds for it (`schemas`; a bit row's `words` hold as many of its
leaves as their bits do), and in `row/v1` rows when it names none.

A verifier that names its window's open root R (`windowRoot`, which it reads from the challenger's signed open of the
window, never from the record) takes every registered value's domain under R (`Registered.domainAt`) but the ones it
holds across windows: `held` (the weights), and its pinned, supplied and output values (`fixedOf`). A window-independent
program then registers each window's values under the same names: a per-window value is held to no earlier window's
root, and a root registered in another window opens under no domain R7 takes. One that names no R is refused before the
record is read when a value it doesn't hold across windows was registered by an earlier record under the domain it takes
now (`unwindowed`, `OneStage.paramsOfJson`). A value it fills itself (`supplied`,
`Supplied.root`: rows of the record's roots, positions after the log's, its coin bound to its binding and the record's
window and its own bytes, under public salts) is held to the root it recomputes from this record and the log instead
(`R7-supplied`), never to the log's.

A verifier bound to a stream and a prover (`binding`: its stream's digest and the identity it registered the prover under)
holds the record's window to name both (`R6-stream`, `R6-prover`), and a window salt is bound to them, so a tile seeded
for one prover's stream is refused in any other's.

A value that is an output of an earlier run (`outputs`; never NCI's baseline θ, which holds registered public data only)
reuses that run's commitment: its entry has the domain and root of the port in the `roots` of the log's record with that
digest, whose leaf count must be the value's rows (`R7-output` otherwise). One may instead reuse an earlier record's
registered value (`Earlier.value`): its entry has the domain and root of that value's entry in the `registered` of the
log's record with that digest, whose leaves, words and word bits must be the value's own (`R7-output` otherwise), so
nothing is registered afresh.
-/
import Flock.WindowRecord
import SampledProofs.Draw
import SampledProofs.Registered
import SampledProofs.Supplied
import Std.Data.HashMap

namespace SampledProofs.Registration

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc)

/- The record's formats and its digest, `SHA-512(format || 0 || canon(record))` with the record's own format as the tag:
the window record's, which every reader of the record shares. -/
export Flock.WindowRecord (FORMAT FORMAT_V2 digest digestHex)

def PROTOCOL : String := "verity/sampled-proofs/one-stage/v1"
def KINDS : List String := ["served", "stand-in"]

def KEYS : Array String := sortStrings #["format", "protocol", "program", "query", "partition", "population", "law",
  "scheme", "roots", "leaf_layer", "anchors", "window", "previous"]
def KEYS_V2 : Array String := sortStrings (KEYS.push "registered")
def ROOT_KEYS : Array String := sortStrings #["port", "root", "leaves", "domain"]

/-- `registration.query_digest`: SHA-512 of the canonical query. -/
def queryDigest (q : Json) : V String := canonSha512 q

/-- What an output value reuses of the log's earlier record: one of its ports (`roots`), or one of its registered values
(`registered`). -/
inductive Earlier where
  | port (name : String)
  | value (name : String)
  deriving Inhabited

/-- The verifier's own copies that R1–R7 hold the record to. -/
structure Verifier where
  program : Program
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
  /-- of `registered`, the values it fills itself, each one's rows of sources (`Supplied.sources`) -/
  supplied : Array (String × Supplied.Rows) := #[]
  /-- of `registered`, values it doesn't fill whose roots it held before the record, `(value, root)` each -/
  pinned : Array (String × String) := #[]
  /-- of `registered`, values it neither fills nor pinned that are outputs of an earlier run or reuse an earlier record's
  registered value, `(value, record digest, its port or value)` each, sorted by value -/
  outputs : Array (String × String × Earlier) := #[]
  /-- the stream and prover it registers the record for, `(stream, prover)`, each lowercase hex -/
  binding : Option (String × String) := none
  /-- of `registered`, values it doesn't fill whose rows are not `row/v1` rows, `(value, schema)` each
  (`Registered.SCHEMAS`): a bit row's leaves are counted in its format (`Registered.entries`) -/
  schemas : Array (String × String) := #[]
  /-- the window's open root, lowercase hex of 64 bytes, from the challenger's signed open of the window -/
  windowRoot : Option String := none
  /-- of `registered`, values it holds across windows, whose domain leaves `windowRoot` out -/
  held : Array String := #[]

private def optStr : Option String → Json
  | some s => Json.str s
  | none => Json.null

/-- The values whose domain leaves the window's open root out: `held`, then the pinned, supplied and output values. -/
def fixedOf (v : Verifier) : Array String :=
  v.held ++ v.pinned.map (·.1) ++ v.supplied.map (·.1) ++ v.outputs.map (·.1)

/-- The verifier's own entries (`registration._check_registered`'s `registered_entries`): each value's domain under the
window's open root, but `fixedOf`'s. -/
def ownEntries (v : Verifier) : V (Array Registered.Entry) :=
  Registered.entries v.program.desc v.params v.registered v.schemas v.windowRoot (fixedOf v)

/-- `registration.held_roots`, read at the verifier's own entries (`mine`, none where they don't compute): `(value,
root)` for each of its values an earlier record of the log registered under the domain `mine` gives it, the root of
its first registration under that domain. -/
def heldRoots (log : Array Json) (mine : V (Array Registered.Entry)) : Array (String × String) :=
  let mine := match mine with | .ok m => m | .error _ => #[]
  log.foldl (fun held rec => match getD rec "registered" with
    | .arr es => es.foldl (fun held e => match getD e "value", getD e "root", getD e "domain" with
      | .str v, .str r, .str d =>
        if held.any (·.1 == v) || !mine.any (fun w => w.value == v && w.domain == d) then held else held.push (v, r)
      | _, _, _ => held) held
    | _ => held) #[]

/-- `registration.unwindowed`: for a verifier that names no window root, its values it doesn't hold across windows
(`fixedOf`) that an earlier record of the log registered under the domain it takes now, a window-independent program's
fixed-name per-window values from its second window on; none with a window root, or where its entries don't compute. -/
def unwindowed (v : Verifier) : Array String :=
  if v.windowRoot.isSome then #[] else
    let fixed := fixedOf v
    ((heldRoots v.log (ownEntries v)).map (·.1)).filter (!fixed.contains ·)

/-- `registration._output_port`: the `roots` entry for `port` of the log's first record with digest `record` (its last,
as a dict reads it). -/
def outputPort (log : Array Json) (record port : String) : Option Json := do
  let r ← log.find? fun r => match digestHex r with | .ok h => h == record | .error _ => false
  let .arr roots := getD r "roots" | none
  roots.reverse.find? fun e => (e matches .obj _) && jeq (getD e "port") (Json.str port)

/-- The domain and root of `port` in the log's record with digest `record` (`outputPort`), when its leaf count is
`leaves` and both are strings. -/
def outputRoot (log : Array Json) (leaves : Nat) (record port : String) : Option (String × String) :=
  (outputPort log record port).bind fun r =>
    match int? (getD r "leaves") == some (leaves : Int), getD r "domain", getD r "root" with
    | true, .str dom, .str root => some (dom, root)
    | _, _, _ => none

/-- `registration._output_port` on `registered`: the entry for `value` in the `registered` of the log's first record with
digest `record` (its last naming it, as a dict reads it). -/
def outputValue (log : Array Json) (record value : String) : Option Json := do
  let r ← log.find? fun r => match digestHex r with | .ok h => h == record | .error _ => false
  let .arr es := getD r "registered" | none
  es.reverse.find? fun e => (e matches .obj _) && jeq (getD e "value") (Json.str value)

/-- An entry's word bits: its `word_bits`, an integer, or 16 when it leaves it out. -/
def widthOf (e : Json) : Option Int :=
  match field? e Registered.WIDTH_KEY with
  | none => some (Registered.WORD_BITS : Int)
  | some j => int? j

/-- The domain and root of `value`'s entry in the log's record with digest `record` (`outputValue`), when its leaves,
words and word bits are `leaves`, `words` and `bits` and both are strings. -/
def valueRoot (log : Array Json) (leaves words bits : Nat) (record value : String) : Option (String × String) :=
  (outputValue log record value).bind fun e =>
    match int? (getD e "leaves") == some (leaves : Int) && int? (getD e "words") == some (words : Int)
        && widthOf e == some (bits : Int), getD e "domain", getD e "root" with
    | true, .str dom, .str root => some (dom, root)
    | _, _, _ => none

/-- The domain and root an output value with entry `want` takes from the log's record with digest `record`: its port's
(`outputRoot`, at `want`'s leaf count) or its registered value's (`valueRoot`, at `want`'s leaves, words and bits). -/
def earlierRoot (log : Array Json) (want : Registered.Entry) (record : String) : Earlier → Option (String × String)
  | .port p => outputRoot log want.leaves record p
  | .value u => valueRoot log want.leaves want.words want.wordBits record u

/-- One output value `(v, record, earlier)` with an entry in `mine` takes its earlier record's port or registered value
(`earlierRoot`): its domain in `mine` and its root ahead of what `held` holds, or `R7-output` in `c`. -/
def applyOutput (log : Array Json) :
    Array String × Array Registered.Entry × Array (String × String) → String × String × Earlier →
      Array String × Array Registered.Entry × Array (String × String)
  | (c, mine, held), (v, d, p) =>
    match mine.find? (·.value == v) with
    | none => (c, mine, held)
    | some want => match earlierRoot log want d p with
      | some (dom, root) =>
        (c, mine.map fun e => if e.value == v then { e with domain := dom } else e, #[(v, root)] ++ held)
      | none => (addCode c "R7-output", mine, held)

/-- `registration._check_registered`'s output values, before the rest of R7: `applyOutput` for each, in order. -/
def applyOutputs (c : Array String) (mine : Array Registered.Entry) (held : Array (String × String))
    (outputs : Array (String × String × Earlier)) (log : Array Json) :
    Array String × Array Registered.Entry × Array (String × String) :=
  outputs.foldl (applyOutput log) (c, mine, held)

/-- `registration._check_registered`, R7: the record's `registered` against the verifier's own entries (`mine`, an error
where `registered_entries` raises), the roots the log holds, and each supplied value's root to the one `rootOf` recomputes
(an error where `supplied_root` raises). -/
def checkRegistered (c : Array String) (got : Json) (mine : V (Array Registered.Entry))
    (held : Array (String × String)) (supplied : Array (String × Supplied.Rows))
    (rootOf : String → Supplied.Rows → Registered.Entry → V String) : Array String := Id.run do
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
    if !supplied.any (·.1 == want.value) then
      if let some (_, r) := held.find? (·.1 == want.value) then
        if !jeq (getD e "root") (Json.str r) then c := addCode c "R7-held"
  for (v, rs) in supplied do
    match mine.find? (·.value == v) with
    | none => c := addCode c "R7-supplied"
    | some want =>
      if let some e := got.reverse.find? (fun e => jeq (getD e "value") (Json.str v)) then
        match rootOf v rs want with
        | .error _ => c := addCode c "R7-supplied"
        | .ok r => if !jeq (getD e "root") (Json.str r) then c := addCode c "R7-supplied"
  return c

/-- R7 with output values: `checkRegistered` on the entries and held roots `applyOutputs` leaves, after its codes. -/
def checkOutputs (c : Array String) (got : Json) (mine : V (Array Registered.Entry))
    (held : Array (String × String)) (supplied : Array (String × Supplied.Rows))
    (rootOf : String → Supplied.Rows → Registered.Entry → V String)
    (outputs : Array (String × String × Earlier)) (log : Array Json) : Array String :=
  match mine with
  | .error _ => addCode c "R7-registered"
  | .ok mine =>
    let (c, mine, held) := applyOutputs c mine held outputs log
    checkRegistered c got (.ok mine) held supplied rootOf

/-- R3's index of the record's roots, each root's port canonicalized once: the last root naming each canonical port, and
how many distinct ports the roots name, a port with no canonical form counting as one of its own (`jeq` equals it to
nothing). -/
structure PortIndex where
  last : Std.HashMap String Json := {}
  count : Nat := 0

/-- One root into the index. The index is taken apart first and the map read and written in one call, so a map held
once is updated in place: reading it after the insert would copy it, root by root. -/
def PortIndex.add : PortIndex → Json → PortIndex
  | ⟨last, count⟩, r =>
    match (Flock.canon (getD r "port")).toOption with
    | none => ⟨last, count + 1⟩
    | some k =>
      let p := last.containsThenInsert k r
      ⟨p.2, if p.1 then count else count + 1⟩

def portIndex (roots : Array Json) : PortIndex := roots.foldl PortIndex.add {}

/-- The last root naming the verifier's port `p`. -/
def PortIndex.find (ix : PortIndex) (p : String) : Option Json :=
  match Flock.canon (Json.str p) with
  | .ok k => ix.last[k]?
  | .error _ => none

def PortIndex.has (ix : PortIndex) (p : String) : Bool := (ix.find p).isSome

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
    let ix := portIndex roots
    if ix.count != roots.size || ix.count != v.ports.size || !v.ports.all (fun (p, _) => ix.has p) then
      c := addCode c "R3-ports"
    for (p, want) in v.ports do
      -- a port registered twice reads as its last root, as Python's dict does
      let some r := ix.find p | continue
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
  if let some (stream, prover) := v.binding then
    if !jeq (getD (get "window") "stream") (Json.str stream) then c := addCode c "R6-stream"
    if !jeq (getD (get "window") "prover") (Json.str prover) then c := addCode c "R6-prover"
  -- R7
  if v2 then
    c := checkOutputs c (get "registered") (ownEntries v) (v.pinned ++ heldRoots v.log (ownEntries v)) v.supplied
      (Supplied.root rec v.log v.binding v.params (Flock.hexOf v.program.desc.sha512) (v.supplied.map (·.1)))
      v.outputs v.log
  return c

/-- `check` as first written, R3 quadratic in the ports: each root's port compared, re-canonicalized, with every port
before it and every verifier port with every root. The reference `check` is proved equal to (`check_eq_checkRef`). -/
def checkRef (rec : Json) (v : Verifier) : Array String := Id.run do
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
  if let some (stream, prover) := v.binding then
    if !jeq (getD (get "window") "stream") (Json.str stream) then c := addCode c "R6-stream"
    if !jeq (getD (get "window") "prover") (Json.str prover) then c := addCode c "R6-prover"
  -- R7
  if v2 then
    c := checkOutputs c (get "registered") (ownEntries v) (v.pinned ++ heldRoots v.log (ownEntries v)) v.supplied
      (Supplied.root rec v.log v.binding v.params (Flock.hexOf v.program.desc.sha512) (v.supplied.map (·.1)))
      v.outputs v.log
  return c

/-! ## `check` is `checkRef` -/

theorem jeq_of_canon {x y : Json} {k : String} (h : Flock.canon y = .ok k) :
    jeq x y = ((Flock.canon x).toOption == some k) := by
  unfold jeq
  rw [h]
  cases Flock.canon x <;> simp [Except.toOption]

theorem jeq_of_canon_error {x y : Json} {e : String} (h : Flock.canon y = .error e) : jeq x y = false := by
  unfold jeq
  rw [h]
  cases Flock.canon x <;> rfl

private def distinctStep (acc : Array Json) (r : Json) : Array Json :=
  if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")

private def keyIs (k : String) (x : Json) : Bool := (Flock.canon x).toOption == some k

private def Inv (acc : Array Json) (ix : PortIndex) (seen : List Json) : Prop :=
  ix.count = acc.size ∧ (∀ k, ix.last.contains k = acc.any (keyIs k)) ∧
    ∀ k, ix.last[k]? = seen.find? fun r => keyIs k (getD r "port")

private theorem inv_step {acc : Array Json} {ix : PortIndex} {seen : List Json} (h : Inv acc ix seen) (r : Json) :
    Inv (distinctStep acc r) (ix.add r) (r :: seen) := by
  obtain ⟨hc, hk, hl⟩ := h
  obtain ⟨last, count⟩ := ix
  unfold distinctStep PortIndex.add
  simp only [Std.HashMap.containsThenInsert_fst, Std.HashMap.containsThenInsert_snd] at *
  rcases hx : Flock.canon (getD r "port") with e | k₀
  · have hn : acc.any (jeq · (getD r "port")) = false := by simp [jeq_of_canon_error hx]
    have hkey : ∀ k, keyIs k (getD r "port") = false := fun k => by simp [keyIs, hx, Except.toOption]
    simp only [hn, Except.toOption, Bool.false_eq_true, ↓reduceIte]
    refine ⟨by simp [hc], fun k => by simp [hk, hkey], fun k => by simp [hkey, hl]⟩
  · have hany : acc.any (jeq · (getD r "port")) = last.contains k₀ := by
      rw [hk]; congr 1; funext y; rw [jeq_of_canon hx]; rfl
    have hkey : ∀ k, keyIs k (getD r "port") = (k₀ == k) := fun k => by
      simp [keyIs, hx, Except.toOption]
    simp only [Except.toOption, hany]
    refine ⟨?_, fun k => ?_, fun k => ?_⟩
    · cases h0 : last.contains k₀ <;> simp [hc]
    · rw [Std.HashMap.contains_insert]
      cases h0 : last.contains k₀
      · simp only [Bool.false_eq_true, ↓reduceIte, Array.any_push, ← hk, hkey, Bool.or_comm]
      · simp only [↓reduceIte, ← hk]
        by_cases he : k₀ = k
        · subst he; simp [h0]
        · simp [he]
    · rw [Std.HashMap.getElem?_insert, List.find?_cons, hkey, hl]
      by_cases he : k₀ = k
      · simp [he]
      · rw [show (k₀ == k) = false from by simp [he]]; rfl

private theorem inv_foldl (l : List Json) :
    ∀ acc ix seen, Inv acc ix seen → Inv (l.foldl distinctStep acc) (l.foldl PortIndex.add ix) (l.reverse ++ seen) := by
  induction l with
  | nil => intro acc ix seen h; simpa using h
  | cons r l ih =>
    intro acc ix seen h
    simpa [List.foldl_cons, List.reverse_cons, List.append_assoc] using ih _ _ _ (inv_step h r)

private theorem inv_portIndex (roots : Array Json) :
    Inv (roots.foldl distinctStep #[]) (portIndex roots) roots.toList.reverse := by
  unfold portIndex
  rw [← Array.foldl_toList, ← Array.foldl_toList]
  simpa using inv_foldl roots.toList #[] {} [] ⟨rfl, fun k => by simp, fun k => by simp⟩

theorem portIndex_count (roots : Array Json) :
    (portIndex roots).count = (roots.foldl (fun (acc : Array Json) r =>
      if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")) #[]).size :=
  (inv_portIndex roots).1

theorem portIndex_has (roots : Array Json) (p : String) :
    (portIndex roots).has p = (roots.foldl (fun (acc : Array Json) r =>
      if acc.any (jeq · (getD r "port")) then acc else acc.push (getD r "port")) #[]).any (jeq · (Json.str p)) := by
  unfold PortIndex.has PortIndex.find
  rcases hp : Flock.canon (Json.str p) with e | k
  · simp [jeq_of_canon_error hp]
  · show (portIndex roots).last[k]?.isSome = _
    rw [Std.HashMap.isSome_getElem?_eq_contains, (inv_portIndex roots).2.1 k]
    congr 1; funext y; rw [jeq_of_canon hp]; rfl

theorem portIndex_find (roots : Array Json) (p : String) :
    (portIndex roots).find p = roots.reverse.find? (fun r => jeq (getD r "port") (Json.str p)) := by
  unfold PortIndex.find
  rcases hp : Flock.canon (Json.str p) with e | k
  · simp only [jeq_of_canon_error hp]
    exact (Array.find?_eq_none.2 (by simp)).symm
  · show (portIndex roots).last[k]? = _
    rw [(inv_portIndex roots).2.2 k, ← Array.find?_toList, Array.toList_reverse]
    congr 1; funext r; rw [jeq_of_canon hp]; rfl

/-- **R3 at `O(n)` hash-map operations is R3 as first written**: everything a guarantee reads of `check` is `checkRef`'s. -/
theorem check_eq_checkRef (rec : Json) (v : Verifier) : check rec v = checkRef rec v := by
  simp only [check, checkRef, portIndex_count, portIndex_has, portIndex_find]

end SampledProofs.Registration
