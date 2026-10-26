/-
Sampled proofs' one-stage audit as a `Core.Protocol` (`one_stage.audit.audit_record`, with `registration.check` before it):

* **params**: the verifier's own copies, canonical JSON with exactly the keys `program` (its descriptor), `query`, `law`
  (the agreed rule: `{"law": "subset", "k"}`, `{"law": "bernoulli", "p": "NUM/DEN"}`, `{"law": "stratified" | "work",
  "k"}`), `work_table` (or null), `scheme`, `ports` (`{port: {"leaves", "domain"}}`, what its layout derives),
  `leaf_layer` (or null), `anchors`, `log` (this prover's earlier records; a record's `registered`, when truthy, a list of
  objects with a string `value` and `root`), `partition` (its own partition object, or null for the canonical one),
  `registered` (null, or `{parameter: words}`, the values it expects registered: empty or null expects a v1 record) and
  `delta` (an exact ratio in (0, 1)); with `registered` not null, also `parameters`, its program function's `[[name,
  type], ...]`, which the descriptor does not carry (`Registered.params`); and, when it expects values registered, it may
  add `supplied`, the ones it fills itself (`Supplied.sources`: `{value: [[{"port" | "value" | "bytes" | "position" |
  "window": string}, ...], ...]}`), and `pinned`, the roots it held before the record for others (`{value: root}`, each
  of a value it expects registered and doesn't fill, lowercase hex of 64 bytes), and `outputs`, the ones that are
  outputs of an earlier run or reuse an earlier record's registered value (`{value: {"record", "port" | "value"}}`, each
  of a value it expects registered and neither fills nor pinned, `record` lowercase hex of 64 bytes and `port` or `value`
  a string), each refused with any other form or when it expects
  none registered; and `row_schemas`, the ones whose rows are not `row/v1` rows (`{value: schema}`, each of a value it
  expects registered and doesn't fill, one of `Registered.SCHEMAS`), which R7 counts a row's leaves in, refused in any
  other form; and it may add `binding`, `{"stream", "prover"}`: its stream's digest (lowercase hex of 64 bytes) and the
  identity it registered the prover under (lowercase hex of whole bytes), refused in any other form;
* **commitment**: the registration record, canonical JSON, fixed before the draw; its digest is
  `SHA-512(format || 0 || canon(record))` (`Registration.digest`), which the receipt names;
* **evidence**: canonical JSON with exactly `receipt`, `draw` (the verifier's draw), `proved_units`, `instances`,
  `statement`, `verdicts` (`{name: {"accepted": true | false | null, "of_record", ...}}`, each verifier's verdict on the
  drawn units' sessions; one of record that accepted may report the roots it opened the sessions' registered reads
  against, `"registered": {port: {"value": string, "root"}}`), `sessions`, `served` and `session_roots` (each null when
  absent);
* **check**: R1–R7 (`Registration.check`), then the record's checks (`Record.checks`: the draw, the proved units, the
  receipt, the served draw, the sessions' roots, and the roots the verifiers of record report opening the sessions'
  registered reads against, `Record.sessionRegistered`, each the registered one). It rejects when one fails or
  a verdict that ran did not accept; it is undecided when they all hold but no verifier of record accepted because none
  is named or one has not run; and it accepts exactly when `audit_record` does;
* on acceptance, **public** is the drawn and proved units with the exact bounds on wrong units at δ (`Bound.boundOf`,
  and each template's own under a stratified or work law), and the **profile** is `audit.profile`'s `IntegrityProfile`:
  one level `unit`, the partition's units, drawn by the verifier's law.
-/
import CoreProtocol.Cli
import CoreProtocol.Profile
import SampledProofs.Registration
import SampledProofs.Record

namespace SampledProofs.OneStage

open Lean (Json)
open Flock (V)

structure Params where
  verifier : Registration.Verifier
  delta : Core.Ratio

structure Commitment where
  record : Json
  digest : ByteArray

structure Evidence where
  receipt : Json
  draw : Json
  provedUnits : Array Nat
  instances : Nat
  statement : Json
  verdicts : Array (String × Json)
  sessions : Array String
  served : Option (Array Nat)
  sessionRoots : Option Json

structure Public where
  drawn : Array Nat
  proved : Array Nat
  bound : Nat
  perTemplate : Option (Array (String × Nat))

private def exactly (j : Json) (ks : Array String) (what : String) : V Unit :=
  if keys? j != some (sortStrings ks) then throw s!"the keys of {what} are not {", ".intercalate (sortStrings ks).toList}"
  else pure ()

private def obj (j : Json) (what : String) : V Json :=
  if j matches .obj _ then pure j else throw s!"{what} is not an object"

private def nats (j : Json) (what : String) : V (Array Nat) := do
  let .arr xs := j | throw s!"{what} is not a list of natural numbers"
  xs.mapM fun x => match nat? x with
    | some n => pure n
    | none => throw s!"{what} is not a list of natural numbers"

/-- The agreed law, a rule `Draw.derive` expands. -/
def rule (law : Json) : V Json := do
  match getD law "law" with
  | .str "bernoulli" =>
    exactly law #["law", "p"] "the law"
    let .str p := getD law "p" | throw "the law's p is not NUM/DEN"
    if (Draw.probability? p).isNone then throw "the law's p is not NUM/DEN, 0 <= NUM <= DEN, 0 < DEN"
  | .str "subset" | .str "stratified" | .str "work" =>
    exactly law #["k", "law"] "the law"
    if (nat? (getD law "k")).isNone then throw "the law's k is not a natural number"
  | _ => throw "the law is not subset, bernoulli, stratified or work"
  return law

def PARAM_KEYS : Array String := #["anchors", "delta", "law", "leaf_layer", "log", "partition", "ports", "program", "query",
  "registered", "scheme", "work_table"]

/-- A record of the log: an object whose `registered`, when truthy, lists objects with a string `value`, `root` and
`domain`, which `Registration.heldRoots` reads (Python's `held_roots` raises on anything else). -/
private def logRecord (j : Json) : V Json := do
  let j ← obj j "a record of the log"
  let r := getD j "registered"
  if truthy r && !(match r with
      | .arr es => es.all fun e =>
        (getD e "value" matches .str _) && (getD e "root" matches .str _) && (getD e "domain" matches .str _)
      | _ => false) then
    throw "a record of the log registers values that are not objects with a string value, root and domain"
  return j

def paramsOfJson (j : Json) : V Params := do
  let v2 := !(getD j "registered" matches .null)
  let keys := if v2 then PARAM_KEYS.push "parameters" else PARAM_KEYS
  let keys := if (field? j "supplied").isSome then keys.push "supplied" else keys
  let keys := if (field? j "pinned").isSome then keys.push "pinned" else keys
  let keys := if (field? j "outputs").isSome then keys.push "outputs" else keys
  let keys := if (field? j "row_schemas").isSome then keys.push "row_schemas" else keys
  let keys := if (field? j "window_root").isSome then keys.push "window_root" else keys
  let keys := if (field? j "held").isSome then keys.push "held" else keys
  exactly j (if (field? j "binding").isSome then keys.push "binding" else keys) "the parameters"
  let program ← Program.parse (← Flock.canon (← obj (getD j "program") "program")).toUTF8
  let query ← obj (getD j "query") "query"
  let law ← rule (getD j "law")
  let workTable ← match getD j "work_table" with
    | .null => pure none
    | t => some <$> obj t "work_table"
  let .str scheme := getD j "scheme" | throw "scheme is not a string"
  let ports ← (entries (← obj (getD j "ports") "ports")).mapM fun (p, v) => do
    exactly v #["domain", "leaves"] s!"port {p}"
    if (nat? (getD v "leaves")).isNone || !(getD v "domain" matches .str _) then
      throw s!"port {p}: leaves is not a natural number or domain not a string"
    pure (p, v)
  let leafLayer ← match getD j "leaf_layer" with
    | .null => pure none
    | .str s => pure (some s)
    | _ => throw "leaf_layer is not a string or null"
  let .arr anchors := getD j "anchors" | throw "anchors is not a list"
  let .arr log := getD j "log" | throw "log is not a list"
  let log ← log.mapM logRecord
  let partition ← match getD j "partition" with
    | .null => pure none
    | p => do
      let p ← obj p "partition"
      if !(getD p "program" matches .str _) || !(getD p "query" matches .obj _) then
        throw "partition names no program digest or query"
      pure (some p)
  let (registered, params) ← if !v2 then pure (#[], #[]) else do
    let registered ← (entries (← obj (getD j "registered") "registered")).mapM fun (v, w) => match nat? w with
      | some w => pure (v, w)
      | none => throw s!"registered: the words of {v} are not a natural number"
    pure (registered, ← Registered.params program.desc (getD j "parameters"))
  let supplied ← match field? j "supplied" with
    | none => pure #[]
    | some s => do
      if registered.isEmpty then throw "supplied values are registered values: the verifier expects none registered"
      Supplied.sources s
  let pinned ← match field? j "pinned" with
    | none => pure #[]
    | some (.obj kvs) => do
      if registered.isEmpty then throw "pinned roots are registered values' roots: the verifier expects none registered"
      kvs.toArray.mapM fun (v, r) => match r with
        | .str r => if registered.any (·.1 == v) && !supplied.any (·.1 == v) && r.length == 128 && isHex r then
            pure (v, r) else throw s!"pinned value {v}: not a registered value it doesn't fill, pinned to 64 bytes"
        | _ => throw s!"pinned value {v}: its root is not a string"
    | some _ => throw "pinned roots are not an object"
  let outputs ← match field? j "outputs" with
    | none => pure #[]
    | some (.obj kvs) => do
      if registered.isEmpty then
        throw "output values are registered values' earlier ports: the verifier expects none registered"
      kvs.toArray.mapM fun (v, o) => do
        let fail := s!"output value {v}: not a registered value it neither fills nor pinned, with an earlier record's " ++
          "digest and a port or a value"
        if !(registered.any (·.1 == v) && !supplied.any (·.1 == v) && !pinned.any (·.1 == v)) then throw fail
        let (key, mk) ← if keys? o == some #["port", "record"] then pure ("port", Registration.Earlier.port)
          else if keys? o == some #["record", "value"] then pure ("value", Registration.Earlier.value) else throw fail
        match getD o "record", getD o key with
        | .str d, .str p => if d.length == 128 && isHex d then pure (v, d, mk p) else throw fail
        | _, _ => throw fail
    | some _ => throw "output values are not an object"
  let schemas ← match field? j "row_schemas" with
    | none => pure #[]
    | some (.obj kvs) => do
      if registered.isEmpty then throw "row formats are registered values' formats: the verifier expects none registered"
      kvs.toArray.mapM fun (v, s) => match s with
        | .str s => if registered.any (·.1 == v) && !supplied.any (·.1 == v) && Registered.SCHEMAS.contains s then
            pure (v, s) else throw s!"row format of {v}: not a registered value it doesn't fill, in one of {Registered.SCHEMAS}"
        | _ => throw s!"row format of {v}: not a string"
    | some _ => throw "row formats are not an object"
  let binding ← match field? j "binding" with
    | none => pure none
    | some b => do
      exactly b #["prover", "stream"] "the binding"
      let (.str stream, .str prover) := (getD b "stream", getD b "prover")
        | throw "the binding's stream and prover are not strings"
      if !(stream.length == 128 && isHex stream) then throw "the binding's stream is not lowercase hex of 64 bytes"
      if !(isHex prover && prover.length % 2 == 0) then throw "the binding's prover is not lowercase hex bytes"
      pure (some (stream, prover))
  let windowRoot ← match getD j "window_root" with
    | .null => pure none
    | .str r => if r.length == 128 && isHex r then pure (some r) else throw "a window root is lowercase hex of 64 bytes"
    | _ => throw "a window root is lowercase hex of 64 bytes"
  let held ← match getD j "held" with
    | .null => pure #[]
    | .arr vs => vs.mapM fun v => match v with
      | .str v => if registered.any (·.1 == v) then pure v else throw s!"held value {v}: not a registered value"
      | _ => throw "held values are a list of registered values"
    | _ => throw "held values are a list of registered values"
  let .str d := getD j "delta" | throw "delta is not a ratio"
  let delta ← Core.Ratio.ofString d "delta"
  if delta.num == 0 || delta.den ≤ delta.num then throw "delta is not in (0, 1)"
  let p : Params := { verifier := { program, query, law, workTable, scheme, ports, leafLayer, anchors, log, partition,
                                    registered, params, supplied, pinned, outputs, binding, schemas, windowRoot, held },
                      delta }
  let again := Registration.unwindowed p.verifier
  if !again.isEmpty then
    throw (s!"registered values {again.toList} were registered by an earlier record of the log under the domains they " ++
      "take now, and the verifier names no window root: a value registered afresh each window under a fixed name takes " ++
      "the window's open root (window_root, from the challenger's signed open of the window), and a value held across " ++
      "windows is named in held (or pinned)")
  pure p

def commitmentOf (b : ByteArray) : V Commitment := do
  let j ← Core.canonicalJson b
  if !(j matches .obj _) then throw "the registration record is not an object"
  return { record := j, digest := ← Registration.digest j }

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  exactly j #["draw", "instances", "proved_units", "receipt", "served", "session_roots", "sessions", "statement",
    "verdicts"] "the evidence"
  let some instances := nat? (getD j "instances") | throw "instances is not a natural number"
  let verdicts ← (entries (← obj (getD j "verdicts") "verdicts")).mapM fun (n, v) => do pure (n, ← obj v s!"verdict {n}")
  if !verdicts.all fun (_, v) =>
      !(truthy (getD v "of_record") && jeq (getD v "accepted") (.bool true) && (getD v "registered" matches .obj _))
      || (entries (getD v "registered")).all fun (_, e) => (getD e "value" matches .str _) && (field? e "root").isSome then
    throw "a verifier of record reports registered reads that are not objects with a string value and a root"
  let .arr ss := getD j "sessions" | throw "sessions is not a list of strings"
  let sessions ← ss.mapM fun s => match s with
    | .str s => pure s
    | _ => throw "sessions is not a list of strings"
  let opt := fun (k : String) (f : Json → V Json) => match getD j k with
    | .null => pure none
    | x => some <$> f x
  return { receipt := ← obj (getD j "receipt") "receipt", draw := getD j "draw",
           provedUnits := ← nats (getD j "proved_units") "proved_units", instances,
           statement := ← obj (getD j "statement") "statement", verdicts, sessions,
           served := ← match getD j "served" with
             | .null => pure none
             | x => some <$> nats x "served",
           sessionRoots := ← opt "session_roots" (obj · "session_roots") }

/-- `audit_record`'s checks after the draw's (`Record.checks`): the names of those that fail. The registered roots the
sessions' reads were opened against are the verdicts' own (`Record.sessionRegistered`), for the registration's own
values (`Record.ownRegistered`). -/
def recordChecks (rec : Json) (digest : String) (e : Evidence) : Array String :=
  let units := match getD e.draw "units" with | .arr us => us.filterMap int? | _ => #[]
  let roots := e.sessionRoots.map entries
  let cs := Record.checks rec e.receipt e.draw Json.null units (e.provedUnits.map Lean.toJson) (Lean.toJson e.instances)
    digest (e.served.map (·.map fun (n : Nat) => (n : Int))) roots
    (Record.ownRegistered rec (Record.sessionRegistered e.verdicts))
  (cs.filter fun (n, c) => n != "draw" && !c.codes.isEmpty).map (·.1)


/-- The verdicts' share of `audit_record`'s acceptance, once the record's checks hold: a verdict that ran (`accepted`
not null) and did not accept rejects; with none of record (`of_record` truthy), or one of record that has not run, the
audit is undecided. -/
def verdictOf (verdicts : Array (String × Json)) : Core.Outcome Unit Unit :=
  match verdicts.find? fun (_, x) => !(getD x "accepted" matches .null) && !jeq (getD x "accepted") (.bool true) with
  | some (name, _) => .reject s!"verdict {name} did not accept"
  | none =>
  let ofRecord := verdicts.filter fun (_, x) => truthy (getD x "of_record")
  if ofRecord.isEmpty then .undecided "no verifier of record" else
  match ofRecord.find? fun (_, x) => getD x "accepted" matches .null with
  | some (name, _) => .undecided s!"the verifier of record {name} has not run"
  | none => .accept () ()

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Public Core.Integrity :=
  let v := p.verifier
  let rec_ := c.record
  let codes := Registration.check rec_ v
  if !codes.isEmpty then .reject s!"registration: {", ".intercalate codes.toList}" else
  match Draw.derive v.program v.query v.law v.workTable with
  | .error err => .reject s!"registration: R1-law: {err}"
  | .ok law =>
  let drawCodes := Draw.checkDraw e.draw (getD rec_ "population") law
  if !drawCodes.isEmpty then .reject s!"draw: {", ".intercalate drawCodes.toList}" else
  let failed := recordChecks rec_ (Flock.hexOf c.digest) e
  if !failed.isEmpty then .reject s!"the audit's checks failed: {", ".intercalate failed.toList}" else
  match verdictOf e.verdicts with
  | .reject r => .reject r
  | .undecided r => .undecided r
  | .accept () () =>
  match Record.profileDraw law, nat? (getD rec_ "population") with
  | some draw, some population =>
    let units := match getD e.draw "units" with | .arr us => us.filterMap nat? | _ => #[]
    let str := fun (k : String) => match getD rec_ k with | .str s => s | _ => ""
    let strata := Bound.strataOf population draw
    let perTemplate := if jeq (getD law "law") "stratified" || jeq (getD law "law") "work" then
      strata.map (·.map fun s => (s.name, Bound.stratumBound s p.delta)) else none
    .accept
      { drawn := units, proved := (Draw.proved e.draw).filterMap fun (i : Int) => if 0 ≤ i then some i.toNat else none,
        bound := Bound.boundOf population draw p.delta, perTemplate }
      { programs := #[str "program"], prescribed := #[], linked := false, covered := #[],
        levels := #[{ name := Record.LEVEL, query := some (str "partition"), size := population, draw := some draw }],
        delta := p.delta, drawn := #[(Record.LEVEL, units.size)] }
  | _, _ => .reject "the law the draw passed under has no profile"

def publicJson (x : Public) : Json :=
  Json.mkObj [("drawn", natsJson x.drawn), ("proved", natsJson x.proved),
    ("bound", Lean.toJson x.bound),
    ("per_template", match x.perTemplate with
      | some ts => Json.mkObj (ts.toList.map fun (t, m) => (t, Lean.toJson m))
      | none => Json.null)]

def protocol : Core.Protocol where
  id := Registration.PROTOCOL
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Public
  Profile := Core.Integrity
  params := paramsOfJson
  commitment := commitmentOf
  evidence := evidenceOf
  digest c := c.digest
  check := check
  publicJson := publicJson
  profileJson := Core.Integrity.toJson

end SampledProofs.OneStage
