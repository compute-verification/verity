module

public import PouwCheck.Lanes

/-!
# A served window's statements verified (`credit._verified`)

The verifier of a window's statements, which `credit.window` and `credit.period_credit` take each sampled
proof block's proof outcome from (`credit._lean_verified`, through `pouw-check-difftest`'s `credit-verified`), with
`credit._verified` the Python reference the difftest (`tests/test_verified_difftest.py`) holds to it, string for string.

A window's evidence carries its statements, one a call with a sampled proof block (`STATEMENT_KEYS`: the call, the paths of its
circuit file and its public file, and its sessions' directories), never a verdict. The Flock verifier
(`flock-verify verify --zk`) runs outside this process, so `verified` is asked twice:

* with no answers (`runs` null), it names each verifier run (`asks`): in statement order, up to the first statement a
  check before the verifier refuses, the statement's index, the circuit file's SHA-512 the period's open committed for
  its call's K (`CIRCUIT_FILES`, `{k<K>: SHA-512, live: SHA-512}`), the call's `--registered` file
  (`Credit.registeredFile`, at the roots the window's entries register, its open root, its rounds and its Program salt)
  and the partition object (`Credit.partitionOf`). The caller runs the verifier on exactly those inputs, against the window's descriptor;
* with the answers (`runs`, one a run in that order, as far as the caller ran them), it judges them (`judge`): a
  verifier that could not run or refused the statement refuses the window (`proof`, or the verifier's code), a session it
  rejected refuses the window whole (`failed`: no partial credit, and no retry), a session that opened its call's public
  table or the window's ok rows (`OK_PORT`, o1) anywhere but at the `--registered` file's value and root refuses it
  (`registration`), and so does one that proved a unit of another call (`proof`). Each unit an accepted session proved
  is one outcome (`credit.PROOF_KEYS`), which `Credit.window` then holds to the sample and the window's registration.

Before any run: the circuit files' form (`FILES_FORM`, `filesRefusal`), the statements' form (`STATEMENTS_FORM`, each
call one of the window's), one statement a call (a retry never reproves), a statement only for a call with a sampled proof block,
a committed circuit file for its K (`noStatementFiles`), and the call's `--registered` file (its refusals,
`registration`). Every refusal is (code, why), the first rule the input breaks, with Python's words.

**The lanes form** (`PouwCheck.Lanes`): the window's statements may be its lanes evidence (`Lanes.LANES_KEYS`), whose
statements are call statements and sampled leaves' (`Lanes.leafStatement?`), at the Ks the circuit files commit as
`lane<K>` and `combine<K>`. Each sampled lanes-form leaf's runs follow the call statements' (`Lanes.leafAsks`, under the
window's sample), each against its own leaf Program; their sessions are judged by `Lanes.judged`, and once all have
accepted, the produced record's checks (`Lanes.outcomes`) give each sampled leaf its one outcome.
-/

@[expose] public section

namespace PouwCheck.Verified

open Lean (Json toJson)
open PouwCheck.Credit

def STATEMENT_KEYS : List String := ["call", "circuit", "public", "sessions"]

def STATEMENTS_FORM : String :=
  "the window's statements are not a list of {call, circuit, public, sessions}, each a call of the " ++
  "window, the paths of its circuit file and its public file, and its sessions' directories (at least " ++
  "one): the window carries each call's proof files, and credit verifies them itself, never a verdict"

/-- The period-open's field holding each K's template's circuit file's SHA-512 and the live Program's template's,
`{k<K>: SHA-512, live: SHA-512}`. -/
def CIRCUIT_FILES : String := "circuit_files"

def FILES_FORM : String :=
  s!"the committed circuit files are not \{k<K>: SHA-512, live: SHA-512} (the period-open's " ++
  s!"{CIRCUIT_FILES}, each digest 128 lowercase hex)"

/-- `credit._files_refusal`: the refusal (`circuit`) of committed circuit files `files` that are not an object of
SHA-512 digests, but for `Lanes.LEAF_PROGRAMS`, an object of them, in the lanes form (`Lanes.lanesFiles`); none when
they are. -/
def filesRefusal (files : Json) : Option (String × String) :=
  let form := isObj files && (fields files).all fun (x, v) =>
    if x == Lanes.LEAF_PROGRAMS then isObj v && (fields v).all fun (_, w) => (digest? w).isSome
    else (digest? v).isSome
  if form && (Lanes.lanesFiles files).isNone then none
  else some ("circuit", if isObj files then (Lanes.lanesFiles files).getD FILES_FORM else FILES_FORM)

def noStatementFiles (k : Nat) : String :=
  s!"the period's open commits no circuit file for K = {k}'s template ({CIRCUIT_FILES} k{k}, the SHA-512 of the " ++
  "circuit file the auditor's lowering built for it), so no statement of its calls can be verified: commit " ++
  "each K's circuit file in the period-open"

/-- A statement of the form `STATEMENT_KEYS` asks: its call (the paths are the caller's to read). -/
def call? (s : Json) : Option Nat :=
  match keys s, nat? (get s "call"), get s "circuit", get s "public", get s "sessions" with
  | some ks, some c, .str _, .str _, .arr xs =>
    if ks == STATEMENT_KEYS && !xs.isEmpty && xs.all (str? · |>.isSome) then some c else none
  | _, _, _, _, _ => none

/-- A list's items, none for anything else. -/
def arrOf : Json → List Json
  | .arr a => a.toList
  | _ => []

/-- `{value: [root, ...]}` over the window's entries, each in order (`credit._verified`'s `roots`). -/
def rootsOf (entries : Json) : Json := Id.run do
  let es := arrOf entries
  let mut acc : Array (String × Array Json) := #[]
  for e in es do
    if let .str v := get e "value" then
      match acc.findIdx? (·.1 == v) with
      | some i => acc := acc.modify i fun (k, rs) => (k, rs.push (get e "root"))
      | none => acc := acc.push (v, #[get e "root"])
  return Json.mkObj (acc.toList.map fun (k, rs) => (k, Json.arr rs))

/-- Each call's first unit, in call order, of a layout `layoutOf` accepted. -/
def startsOf (sizes : List Nat) : List Nat :=
  (sizes.foldl (fun (xs, at_) n => (xs.push at_, at_ + n)) ((#[] : Array Nat), 0)).1.toList

/-- One verifier run `asks` names: the statement's index and call, and the verifier's inputs for it. -/
structure Ask where
  statement : Nat
  call : Nat
  circuit : String
  registered : Json

/-- `credit._asks`: the verifier runs in statement order up to the first statement a check before the verifier refuses,
and that refusal (none when every statement passes them), each run's partition object `partitionOf` the descriptor's.
Over the window's circuit (its descriptor's SHA-512 `program`, and `shape` and `layout`, `shapeOf` and `layoutOf` of
its Program), its registered `entries`, open root `window`, sampled `units`, `steps` rounds, Program salt `salt` and
sample `draw`. `statements` is a list of call statements, or the lanes evidence (`Lanes.LANES_KEYS`) whose `statements`
hold call statements and sampled leaves'; each sampled leaf's runs (`Lanes.leafAsks`) follow the call statements'. -/
def asks (statements files : Json) (program : String) (shape : Shape) (layout entries : Json) (window : String)
    (units : List Nat) (steps : Nat) (salt : String) (draw : Json) :
    R (Array Ask × Array Lanes.LeafAsk × Option (String × String)) := do
  if let some why := filesRefusal files then throw why
  let lanes : Option Json := if keys statements == some Lanes.LANES_KEYS then some statements else none
  if let some ev := lanes then
    let progs := get ev Lanes.LEAF_PROGRAMS
    unless isObj progs && (fields progs).all (fun (_, v) => (str? v).isSome) do throw ("proof", Lanes.LANES_FORM)
  let form := if lanes.isSome then Lanes.LANES_FORM else STATEMENTS_FORM
  let sizes := (callWork layout).map (·.1)
  let .arr xs := (match lanes with | some ev => get ev "statements" | none => statements) | throw ("proof", form)
  let mut ss : Array (Nat × Nat) := #[]
  let mut leafs : Array (Nat × Nat × Lanes.St) := #[]
  for (s, i) in xs.toList.zip (List.range xs.size) do
    if let some c := (call? s).filter (· < sizes.length) then ss := ss.push (i, c)
    else if let some (u, st) := lanes.bind fun _ => Lanes.leafStatement? s sizes.sum then leafs := leafs.push (i, u, st)
    else throw ("proof", form)
  let starts := startsOf sizes
  let drawn := units.map fun u => (starts.filter (· ≤ u)).length - 1
  let roots := rootsOf entries
  let lins : List (String × Nat) := (arrOf (get layout "linears")).filterMap fun x =>
    match arrOf x with
    | [.str s, k, _] => (s, ·) <$> nat? k
    | _ => none
  let callLin : List String := (arrOf (get layout "calls")).map fun x => ((arrOf x).head?.bind str?).getD ""
  let mut seen : List Nat := []
  let mut out : Array Ask := #[]
  for (i, c) in ss do
    if seen.contains c then
      return (out, #[], some ("outcome", s!"call {c} has two statements: a call's drawn units are proved in one " ++
        "statement and verified once, and a retry never reproves them"))
    seen := c :: seen
    unless drawn.contains c do
      return (out, #[], some ("outcome", s!"call {c} has a statement and no drawn unit: only the draw's units are proved"))
    let lin := callLin.getD c ""
    let k := ((lins.reverse.find? (·.1 == lin)).map (·.2)).getD 0
    let some (.str sha) := PouwWindow.field? files s!"k{k}" | return (out, #[], some ("circuit", noStatementFiles k))
    match registeredFile program layout shape c roots window steps salt with
    | .ok reg => out := out.push ⟨i, c, sha, reg⟩
    | .error why => return (out, #[], some why)
  let (got, why) ← Lanes.leafAsks lanes leafs files program layout shape roots window units steps salt draw
  return (out, got, why)

/-- A run's input as the caller reads it: `{statement, circuit_sha512, registered, partition}`. -/
def Ask.toJson (a : Ask) (partition : Json) : Json :=
  Json.mkObj [("statement", Lean.toJson a.statement), ("circuit_sha512", Json.str a.circuit),
    ("registered", a.registered), ("partition", partition)]

/-- One session's finding from the verifier's run (`proofs.Finding`). -/
structure Finding where
  session : String
  accepted : Bool
  why : String
  program : Json
  registered : Json
  units : List Nat
  openings : Json

def finding (f : Json) : R Finding := do
  match get f "session", get f "accepted", get f "why", get f "program",
      (·.map nat?) <$> (Json.getArr? (get f "units")).toOption with
  | .str s, .bool a, .str w, p, some us =>
    let some us := us.toList.mapM id | throw ("error", "credit-verified: a finding's units are not natural numbers")
    pure ⟨s, a, w, p, get f "registered", us, get f "openings"⟩
  | _, _, _, _, _ =>
    throw ("error", "credit-verified: a finding is not {session, accepted, why, program, registered, units}")

/-- `credit._verified`'s judgment of the verifier's answers `runs` (one an `asks` run, in its order, as far as the
caller ran them): each sampled proof block's outcome `{unit, verdict, program, registered}`, else the refusal of the first answer
that credits nothing, then `asks`' own refusal, then (with sampled leaves' runs) `Lanes.outcomes`. An answer is
`{"error": why}` (the verifier could not run), `{"refused": [code, why]}` (it refused the statement) or
`{"findings": [...]}` (`finding`, one a session). -/
def judge (statements files : Json) (program : String) (shape : Shape) (layout entries : Json) (window : String)
    (units : List Nat) (steps : Nat) (salt : String) (draw : Json) (runs : Array Json) : R (Array Json) := do
  let (as, leaves, refused) ← asks statements files program shape layout entries window units steps salt draw
  let sizes := (callWork layout).map (·.1)
  let starts := startsOf sizes
  let mut out : Array Json := #[]
  let mut j := 0
  for a in as do
    let c := a.call
    let some ans := runs[j]? | throw ("error", s!"credit-verified: no verifier answer for statement {a.statement}")
    j := j + 1
    match get ans "error", get ans "refused", get ans "findings" with
    | .str e, _, _ =>
      throw ("proof", s!"call {c}'s statement could not be verified, so the window stays refused: {e}")
    | _, .arr #[.str code, .str why], _ => throw (code, s!"call {c}'s statement: {why}")
    | _, _, .arr fs =>
      let portOf := fun (p : String) => get (get a.registered "ports") p
      let pv := (str? (get (portOf PUB_PORT) "value")).getD ""
      let pr := (str? (get (portOf PUB_PORT) "root")).getD ""
      let ov := (str? (get (portOf OK_PORT) "value")).getD ""
      let orr := (str? (get (portOf OK_PORT) "root")).getD ""
      let lo := starts.getD c 0
      let hi := lo + sizes.getD c 0
      for fj in fs do
        let f ← finding fj
        unless f.accepted do
          throw ("failed", s!"call {c}'s session {f.session} was rejected by the verifier of record ({f.why}): a " ++
            "failed drawn unit refuses the whole window, not only its call (no partial credit, and no retry)")
        let opened := (finding? f.registered).getD []
        if opened.any fun (p, v, root) => (pubCall? (.str v)).isSome && (p, v, root) != (PUB_PORT, pv, pr) then
          throw ("registration", s!"call {c}'s session {f.session} opened a public table at another port, value or " ++
            s!"root than the verifier's own {pv} at port {PUB_PORT}: a unit's public row is the window's salt, its " ++
            "position and its call's roots, as credit builds them")
        unless opened.filter (fun (p, v, _) => p == OK_PORT || v == ov) == [(OK_PORT, ov, orr)] do
          throw ("registration", s!"call {c}'s session {f.session} opened the ok rows at another port, value or root " ++
            s!"than the verifier's own {ov} at port {OK_PORT}, or not at all: a unit's ok row is the one word 1 " ++
            "under its public ok salt, as credit registers it, so a unit that computed 0 opens none of it")
        for u in f.units do
          unless lo ≤ u && u < hi do
            throw ("proof", s!"call {c}'s session {f.session} proved unit {u}, which is no unit of call {c}: a " ++
              "call's statement proves its own call's units")
          out := out.push (Json.mkObj [("unit", toJson u), ("verdict", Json.str "accept"), ("program", f.program),
            ("registered", f.registered)])
    | _, _, _ => throw ("error", "credit-verified: an answer is not {error}, {refused: [code, why]} or {findings}")
  let mut held : Lanes.Held := {}
  for a in leaves do
    let what := s!"drawn leaf {a.unit}'s {a.leaf} statement"
    let some ans := runs[j]? | throw ("error", s!"credit-verified: no verifier answer for statement {a.statement}")
    j := j + 1
    match get ans "error", get ans "refused", get ans "findings" with
    | .str e, _, _ => throw ("proof", s!"{what} could not be verified, so the window stays refused: {e}")
    | _, .arr #[.str code, .str why], _ => throw (code, s!"{what}: {why}")
    | _, _, .arr fs =>
      for fj in fs do
        let f ← finding fj
        held ← Lanes.judged a f.session f.accepted f.why f.program f.registered f.units f.openings held
    | _, _, _ => throw ("error", "credit-verified: an answer is not {error}, {refused: [code, why]} or {findings}")
  if let some why := refused then throw why
  if !leaves.isEmpty then out := out ++ (← Lanes.outcomes (get statements "produced") held program)
  pure out

/-- `credit-verified`: on `{statements, files, descriptor, entries, window, units, steps, salt, draw, runs}` (the
window's statements, its period's committed circuit files, its descriptor in hex, its registered entries, its open root,
its sampled proof blocks, its rounds, its Program salt and its sample), with `runs` null, `{"run": [...]}`, the verifier runs
`asks` names (`Ask.toJson`, then `Lanes.LeafAsk.toJson`), or the verdict when it names none; with `runs` the answers, the verdict: `{"outcomes": [...]}` (`judge`)
or `{"refused": [code, why]}` (as `refusal` writes it). -/
def verified (j : Json) : R Json := do
  let some descriptor := (match get j "descriptor" with
    | .str s => if isHexBytes s then hex? (.str s) else none
    | _ => none) | layoutRefuse DESCRIPTOR_HEX
  let P ← programOf descriptor
  let layout ← layoutOf P
  let program := Flock.hexOf (Flock.Sha512.hash descriptor)
  let shape := shapeOf P
  let some units := (Json.getArr? (get j "units")).toOption.bind (·.toList.mapM nat?)
    | throw ("error", "credit-verified: the drawn units are not a list of natural numbers")
  let (some steps, some window, some salt) := (nat? (get j "steps"), str? (get j "window"), str? (get j "salt"))
    | throw ("error", "credit-verified: the window's steps, open root or salt are malformed")
  let (statements, files, entries, draw) := (get j "statements", get j "files", get j "entries", get j "draw")
  match get j "runs" with
  | .null =>
    let (as, leaves, refused) ← asks statements files program shape layout entries window units steps salt draw
    if as.isEmpty && leaves.isEmpty then
      if let some why := refused then throw why
      return Json.mkObj [("outcomes", Json.arr #[])]
    let partition := partitionOf program shape
    return Json.mkObj [("run", Json.arr (as.map (·.toJson partition) ++ leaves.map (·.toJson)))]
  | .arr runs =>
    let out ← judge statements files program shape layout entries window units steps salt draw runs
    return Json.mkObj [("outcomes", Json.arr out)]
  | _ => throw ("error", "credit-verified: the verifier's answers are neither null nor a list")

end PouwCheck.Verified
