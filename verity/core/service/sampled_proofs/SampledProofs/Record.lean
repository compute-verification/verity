/-
The one-stage audit record (`audit.audit_record`), whole, and what the drivers check beside it:

* `matchesPublic`: `registration.matches_public`, the verifier's own public file (its scheme, roots, their domains and its
  instance count) against the record, with Python's codes in its order (`scheme`, `roots`, `domains`, `population`) and
  details;
* `sessionRegistered`: `audit.session_registered`, the `{value: root}` the verifiers of record that accepted report opening
  the sessions' registered reads against (a verdict's `registered`, `{port: {"value", "root"}}`), a value reported under
  two roots mapped to null; none when no such verdict reports any; `ownRegistered` keeps a registration's own values (its
  roots' ports and registered values);
* `checks`: `audit_record`'s checks, each with Python's detail: the draw's and the proved units' always, and the receipt's,
  the served draw's, the sessions' roots' and the registered roots' when they fail;
* `record`: the record, every field as `audit_record` writes it, the registered roots its checks read derived here from
  the verdicts (`sessionRegistered`, `ownRegistered`), never taken as input. Python writes `delta` as a float: here it is the double nearest
  the exact δ given, which `PyJson` writes as Python's `repr` does. `wrong_units.bound` is exact under every law
  (`Bound.boundOf`).
-/
import SampledProofs.Bound
import SampledProofs.Draw
import SampledProofs.PyJson
import Flock.WindowRecord
import CoreProtocol.Profile

namespace SampledProofs.Record

open Lean (Json)
open Flock (V)

def FORMAT : String := "verity/one-stage/audit/v0"
def PROFILE : String := "verity/integrity-profile/v1"
def LEVEL : String := "unit"

private def obj (kvs : List (String × Json)) : Json := Json.mkObj kvs

/-- Python's `==` on two dicts of JSON values with distinct keys. -/
def sameDict (a b : Array (String × Json)) : Bool :=
  a.size == b.size && a.all fun (k, v) => b.any fun (k', v') => k == k' && jeq v v'

/-- `{r["port"]: r for r in rec["roots"]}`: the last root of each port. -/
def byPort (rec : Json) : Array (String × Json) :=
  (match getD rec "roots" with | .arr rs => rs | _ => #[]).foldl (fun (acc : Array (String × Json)) r => match getD r "port" with
    | .str p => (acc.filter fun (q : String × Json) => q.1 != p).push (p, r)
    | _ => acc) #[]

/-- `{e["value"]: e["root"] for e in rec.get("registered", [])}`: the last root of each value. -/
def registeredRoots (rec : Json) : Array (String × Json) :=
  (match getD rec "registered" with | .arr es => es | _ => #[]).foldl (fun (acc : Array (String × Json)) e => match getD e "value" with
    | .str v => (acc.filter fun (q : String × Json) => q.1 != v).push (v, getD e "root")
    | _ => acc) #[]

def matchesPublic (rec scheme roots domains population : Json) : Check := Id.run do
  let reg := byPort rec
  let regRoots := reg.map fun (p, r) => (p, getD r "root")
  let regDomains := reg.map fun (p, r) => (p, getD r "domain")
  let mut c : Check := {}
  if !jeq scheme (getD rec "scheme") then
    c := c.fail "scheme" (obj [("registered", getD rec "scheme"), ("public", scheme)])
  if !sameDict regRoots (entries roots) then
    c := c.fail "roots" (obj [("registered", obj regRoots.toList), ("public", roots)])
  if !sameDict (reg.map fun (p, _) => (p, getD domains p)) regDomains then
    c := c.fail "domains" (obj [("registered", obj regDomains.toList), ("public", domains)])
  if !jeq population (getD rec "population") then
    c := c.fail "population" (obj [("registered", getD rec "population"), ("public", population)])
  return c

def sessionRegistered (verdicts : Array (String × Json)) : Option (Array (String × Json)) := Id.run do
  let mut out : Option (Array (String × Json)) := none
  for (_, v) in verdicts do
    if truthy (getD v "of_record") && jeq (getD v "accepted") (.bool true) && (getD v "registered" matches .obj _) then
      let mut acc := out.getD #[]
      for (_, e) in entries (getD v "registered") do
        let .str value := getD e "value" | continue
        let root := getD e "root"
        acc := match acc.find? (·.1 == value) with
          | some (_, r) => acc.map fun (k, x) => if k == value then (k, if jeq r root then r else .null) else (k, x)
          | none => acc.push (value, root)
      out := some acc
  return out

def ownRegistered (rec : Json) (opened : Option (Array (String × Json))) : Option (Array (String × Json)) :=
  let named := (byPort rec).map (·.1) ++ (registeredRoots rec).map (·.1)
  opened.map (·.filter fun (v, _) => named.contains v)

private def sortedInts (xs : Array Int) : Array Int := xs.qsort (· < ·)

/-- A check `audit_record` writes itself: its detail is the object given, not keyed by its code. -/
private def failed (code : String) (detail : Json) : Check := { codes := #[code], detail := entries detail }

/-- `audit_record`'s checks, in its order: `draw` and `proved` always, then `receipt`, `served`, `roots` and `registered`
only when they fail. `digest` is the registration's, `units` the draw's. -/
def checks (reg receipt draw law : Json) (units : Array Int) (provedUnits : Array Json) (instances : Json) (digest : String)
    (served : Option (Array Int)) (sessionRoots : Option (Array (String × Json)))
    (sessionRegistered : Option (Array (String × Json))) : Array (String × Check) := Id.run do
  let mut out := #[("draw", Draw.checkDrawFull draw (getD reg "population") law),
    ("proved", Draw.checkProvedFull draw provedUnits instances)]
  if !jeq (getD receipt "registration") (Json.str digest) then
    out := out.push ("receipt", failed "receipt" (obj [("receipt", getD receipt "registration")]))
  if let some s := served then
    if sortedInts s != sortedInts units then
      out := out.push ("served", failed "served" (obj [("served", intsJson ((sortedInts s).extract 0 16)),
        ("drawn", intsJson (units.extract 0 16))]))
  if let some roots := sessionRoots then
    let registered := byPort reg
    let wrong := sortStrings (roots.filterMap fun (p, r) =>
      match registered.find? (·.1 == p) with
      | some (_, x) => if jeq (getD x "root") r then none else some p
      | none => some p)
    if roots.isEmpty || !wrong.isEmpty then
      out := out.push ("roots", failed "roots" (obj [("ports", Json.arr (wrong.map Json.str))]))
  if (field? reg "registered").isSome || sessionRegistered.isSome then
    let want := registeredRoots reg
    let opened := sessionRegistered.getD #[]
    if !sameDict opened want then
      let wrong := want.filterMap fun (v, r) => match opened.find? (·.1 == v) with
        | some (_, o) => if jeq o r then none else some v
        | none => some v
      out := out.push ("registered", failed "registered" (obj [
        ("registered", Json.arr ((sortStrings (want.map (·.1))).map Json.str)),
        ("opened", Json.arr ((sortStrings (opened.map (·.1))).map Json.str)),
        ("wrong", Json.arr ((sortStrings wrong).map Json.str))]))
  return out

/-- The profile's draw of the law a draw names (`audit.profile`): a work law is the stratified law at its sizes. -/
def profileDraw (law : Json) : Option Core.Draw :=
  match getD law "law" with
  | .str "subset" => (nat? (getD law "k")).map .subset
  | .str "bernoulli" => do
    let .str p := getD law "p" | none
    let (a, b) ← Draw.probability? p
    return .bernoulli (← Core.Ratio.mk' a b)
  | .str "stratified" | .str "work" => do
    let .arr ss := getD law "strata" | none
    let strata ← ss.mapM fun s => do
      let .str t := getD s "template" | none
      let rs ← Draw.ranges? (getD s "units")
      return ({ name := t, size := (rs.foldl (fun (a : Int) ((lo, hi) : Int × Int) => a + (hi - lo)) 0).toNat,
                k := ← nat? (getD s "k") } : Core.Stratum)
    return .stratified strata
  | _ => none

/-- δ as Python's float: the nearest double, as a number `PyJson` writes as its `repr`. -/
def floatJson (r : Core.Ratio) : V Json := do
  let some (m, e) := PyJson.nearest r.num r.den | throw "delta is past the largest double"
  let (digits, point) := PyJson.shortest m e
  let ex : Int := (digits.length : Int) - point
  if m == 0 || ex < 1 then throw "delta is not a double in (0, 1)"
  return Json.num ⟨digits.toNat!, ex.toNat⟩

/-- `SHA-512` of Python's `canonical_json`, hex. -/
def pySha512 (j : Json) : V String := do return sha512Hex (← PyJson.canonical j).toUTF8

/-- A record's canonical JSON is Python's `canonical_json`: `Flock.canon` writes it, and it holds only characters below
0x7f, where the two write every string alike. -/
def pyCanonical (rec : Json) : Bool :=
  match Flock.canon rec with
  | .ok s => s.toList.all fun c => c.toNat < 0x7f
  | .error _ => false

structure Input where
  registration : Json
  receipt : Json
  draw : Json
  units : Array Int
  law : Json
  provedUnits : Array Json
  instances : Json
  statement : Json
  verdicts : Array (String × Json)
  delta : Core.Ratio
  sessions : Array Json
  served : Option (Array Int)
  sessionRoots : Option (Array (String × Json))

private def need (ok : Bool) (what : String) : V Unit := if ok then pure () else throw what

private def ints? (j : Json) : Option (Array Int) := match j with
  | .arr xs => xs.mapM int?
  | _ => none

private def INPUT_KEYS : Array String := sortStrings #["registration", "receipt", "draw", "law", "proved_units", "instances",
  "statement", "verdicts", "delta", "sessions", "served", "session_roots"]

/-- `audit_record`'s arguments, on the inputs it reads without raising and the drivers give it: a registration whose
canonical JSON is Python's (`pyCanonical`, so its digest is `registration.digest`'s) and that has every field the record reads (its roots and registered entries objects with a string port or value), a draw object whose
units (and closure) are integers, its population an integer and a subset's `k` present, integer proved units and
instances, verdicts objects (a verifier of record that accepted reporting its registered reads as objects with a string
`value` and a `root`), session digests as strings, served units integers, and δ an exact ratio in (0, 1). -/
def inputOf (j : Json) : V Input := do
  need (keys? j == some INPUT_KEYS) s!"the record's input is exactly {", ".intercalate INPUT_KEYS.toList}"
  let reg := getD j "registration"
  need (reg matches .obj _) "the registration is not an object"
  need (pyCanonical reg) "the registration holds a number that is not an integer, or a character from 0x7f on"
  for k in ["anchors", "law", "partition", "population", "program", "protocol", "query", "roots", "scheme"] do
    need (field? reg k).isSome s!"the registration has no {k}"
  need (int? (getD reg "population")).isSome "the registration's population is not an integer"
  need (getD reg "anchors" matches .arr _) "the registration's anchors are not a list"
  let .arr roots := getD reg "roots" | throw "the registration's roots are not a list"
  need (roots.all fun r => (getD r "port" matches .str _) && (field? r "root").isSome && (field? r "leaves").isSome)
    "a registered root is not an object with a string port, a root and leaves"
  if let some es := field? reg "registered" then
    let .arr es := es | throw "the registration's registered values are not a list"
    need (es.all fun e => (getD e "value" matches .str _) && ["root", "leaves", "domain"].all (field? e · |>.isSome))
      "a registered value is not an object with a string value, a root, leaves and a domain"
  let receipt := getD j "receipt"
  need (receipt matches .obj _) "the receipt is not an object"
  let draw := getD j "draw"
  need (draw matches .obj _) "the draw is not an object"
  let some units := ints? (getD draw "units") | throw "the draw's units are not integers"
  if let some c := field? draw "closure" then need (ints? c).isSome "the draw's closure is not integers"
  if let some n := field? draw "population" then need (int? n).isSome "the draw's population is not an integer"
  if jeq (getD draw "law") "subset" then need (field? draw "k").isSome "a subset draw has no k"
  let law := getD j "law"
  need (law matches .obj _) "the law is not an object"
  let .arr provedUnits := getD j "proved_units" | throw "the proved units are not integers"
  need (provedUnits.all (int? · |>.isSome)) "the proved units are not integers"
  need (int? (getD j "instances")).isSome "instances is not an integer"
  need (getD j "statement" matches .obj _) "the statement is not an object"
  let .obj _ := getD j "verdicts" | throw "verdicts is not an object"
  let verdicts := entries (getD j "verdicts")
  need (verdicts.all (·.2 matches .obj _)) "a verdict is not an object"
  need (verdicts.all fun (_, v) =>
      !(truthy (getD v "of_record") && jeq (getD v "accepted") (.bool true) && (getD v "registered" matches .obj _))
      || (entries (getD v "registered")).all fun (_, e) => (getD e "value" matches .str _) && (field? e "root").isSome)
    "a verifier of record reports registered reads that are not objects with a string value and a root"
  let .arr sessions := getD j "sessions" | throw "sessions is not a list of strings"
  need (sessions.all (· matches .str _)) "sessions is not a list of strings"
  let served ← match getD j "served" with
    | .null => pure none
    | s => match ints? s with
      | some s => pure (some s)
      | none => throw "served is not a list of integers"
  let sessionRoots ← match getD j "session_roots" with
    | .null => pure none
    | .obj kvs => pure (some (Json.obj kvs |> entries))
    | _ => throw "session_roots is not an object"
  let .str d := getD j "delta" | throw "delta is not a ratio"
  let delta ← Core.Ratio.ofString d "delta"
  need (delta.num != 0 && delta.num < delta.den) "delta is not in (0, 1)"
  return { registration := reg, receipt, draw, units, law, provedUnits, instances := getD j "instances",
           statement := getD j "statement", verdicts, delta, sessions, served, sessionRoots }

private def strataDrawn (draw : Json) (units : Array Int) : Json :=
  obj ((match getD draw "strata" with | .arr ss => ss | _ => #[]).toList.filterMap fun s =>
    match getD s "template" with
    | .str t =>
      let rs := (Draw.ranges? (getD s "units")).getD #[]
      some (t, Lean.toJson (units.filter fun u => rs.any fun (lo, hi) => lo ≤ u && u < hi).size)
    | _ => none)

def record (i : Input) : V Json := do
  let reg := i.registration
  let get := getD reg
  let digest ← Flock.WindowRecord.digestHex reg
  let cs := checks reg i.receipt i.draw i.law i.units i.provedUnits i.instances digest i.served i.sessionRoots
    (ownRegistered reg (sessionRegistered i.verdicts))
  let accepted' := fun (v : Json) => jeq (getD v "accepted") (.bool true)
  let ran := i.verdicts.filter fun (_, v) => !(getD v "accepted" matches .null)
  let ofRecord := i.verdicts.filter fun (_, v) => truthy (getD v "of_record")
  let accepted := !ofRecord.isEmpty && ofRecord.all (accepted' ·.2) && ran.all (accepted' ·.2)
    && cs.all (·.2.codes.isEmpty)
  let kind := getD i.draw "law"
  let stratified := jeq kind "stratified" || jeq kind "work"
  let mut profile := Json.null
  let mut integrity := Json.null
  let mut wrongUnits := Json.null
  if accepted then
    let delta ← floatJson i.delta
    let some draw := profileDraw i.draw | throw "the draw's law has no profile"
    let population ← match nat? (get "population") with
      | some n => pure n
      | none => throw "the population is negative"
    profile := if stratified then
        match draw.toJson with
        | .obj kvs => (Json.obj kvs).setObjVal! "delta" delta |>.setObjVal! "drawn" (strataDrawn i.draw i.units)
        | j => j
      else obj [("programs", Json.arr #[get "program"]), ("prescribed", Json.arr #[]), ("linked", Json.bool false),
        ("covered", Json.arr #[]), ("levels", Json.arr #[obj [("name", Json.str LEVEL), ("query", get "partition"),
          ("size", Lean.toJson population), ("draw", draw.toJson)]]), ("delta", delta),
        ("drawn", obj [(LEVEL, Lean.toJson i.units.size)])]
    let digests := ofRecord.map (getD ·.2 "digest")
    let roots := match get "roots" with
      | .arr rs => rs.map fun r => obj [("port", getD r "port"), ("root", getD r "root"), ("leaves", getD r "leaves")]
      | _ => #[]
    let inputs := [("anchored", get "anchors"), ("linked", Json.arr #[])] ++
      match field? reg "registered" with
      | some (.arr es) => [("registered", Json.arr (es.map fun e =>
          obj (["value", "root", "leaves", "domain"].map fun k => (k, getD e k))))]
      | _ => []
    integrity := obj [("format", Json.str PROFILE), ("protocol", get "protocol"), ("program", get "program"),
      ("query", get "query"), ("partition", get "partition"), ("population", get "population"),
      ("registration", Json.str digest), ("scheme", get "scheme"), ("roots", Json.arr roots), ("law", get "law"),
      ("drawn", intsJson i.units), ("inputs", obj inputs), ("budget_log2_t", Json.null), ("terms", Json.null),
      ("epsilon_log2", Json.null),
      ("evidence", obj [("receipt", Json.str (← pySha512 i.receipt)), ("draw", Json.str (← pySha512 i.draw)),
        ("sessions", Json.arr i.sessions),
        ("verdict", if digests.size == 1 then digests[0]! else Json.arr digests)])]
    let strata := Bound.strataOf population draw
    let rule := "except with probability delta, an accepted transcript has at most `bound` wrong units"
    wrongUnits := obj ([("delta", delta), ("of", get "population"),
        ("bound", Lean.toJson (Bound.boundOf population draw i.delta)),
        ("rule", Json.str (if stratified then rule ++ ", and at most `per_template[t]` of them in template t" else rule))] ++
      if stratified then
        [("per_template", obj ((strata.getD #[]).toList.map fun s => (s.name, Lean.toJson (Bound.stratumBound s i.delta))))]
      else [])
  return obj [("format", Json.str FORMAT), ("registration", reg), ("registration_digest", Json.str digest),
    ("receipt", i.receipt), ("draw", i.draw), ("law", Draw.lawOf i.draw), ("proved_units", Json.arr i.provedUnits),
    ("statement", i.statement), ("checks", obj (cs.toList.map fun (n, c) => (n, c.toJson))),
    ("verdicts", obj i.verdicts.toList), ("accepted", Json.bool accepted), ("complete", Json.bool (ran.size == i.verdicts.size)),
    ("profile", profile), ("integrity_profile", integrity), ("wrong_units", wrongUnits)]

/-! ## `Flock.canon` is `canonical_json` on records `pyCanonical` admits

On such a record the two encoders write the same bytes, so the window record's digest is `SHA-512(format || 0 ||
canonical_json(record))`; a character from 0x7f on, or a fraction, is where they part. -/

private def sameCanon (j : Json) : Bool :=
  let py := (PyJson.canonical j).toOption
  pyCanonical j && (Flock.canon j).toOption == py && (Flock.WindowRecord.digestHex j).toOption ==
    py.map fun t => sha512Hex ((Flock.WindowRecord.tagOf j).toUTF8 ++ ByteArray.mk #[0] ++ t.toUTF8)

private def parsed (s : String) : Json := (Json.parse s).toOption.getD .null

private def PRINTABLE : String := String.ofList ((List.range 0x7f).map Char.ofNat)

#guard sameCanon (parsed "{\"format\":\"verity/registration/v2\",\"protocol\":\"verity/sampled-proofs/one-stage/v1\",\"program\":\"ab01\",\"roots\":[{\"port\":\"x\",\"root\":\"00ff\",\"leaves\":4,\"domain\":\"11\"}],\"window\":{\"kind\":\"served\",\"stream\":\"aa\",\"prover\":\"bb\"},\"previous\":null,\"law\":{\"law\":\"bernoulli\",\"p\":\"1/3\"},\"population\":12,\"anchors\":[],\"registered\":[{\"value\":\"W\",\"root\":\"cc\",\"leaves\":2,\"words\":4,\"domain\":\"dd\",\"word_bits\":32}],\"leaf_layer\":null,\"neg\":-5,\"t\":true,\"e\":{}}")
#guard sameCanon (parsed "{\"format\":\"verity/registration/v1\",\"b\":1,\"B\":2,\"ab\":3,\"a\":4,\"_\":[[],[{}],0]}")
#guard sameCanon (Json.mkObj [("format", Json.str "verity/registration/v1"), (PRINTABLE, Json.str PRINTABLE)])
#guard !pyCanonical (Json.str "\x7f") && (Flock.canon (Json.str "\x7f")).toOption != (PyJson.canonical (Json.str "\x7f")).toOption
#guard !pyCanonical (Json.str "dé")
#guard !pyCanonical (parsed "{\"p\":0.5}")

end SampledProofs.Record
