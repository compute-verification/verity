import PouwCheck.Credit

/-!
# A served window's statements verified (`credit._verified`)

The verifier of record of a window's statements, which `credit.window` and `credit.period_credit` take each drawn
unit's proof outcome from (`credit._lean_verified`, through `pouw-check-difftest`'s `credit-verified`), with
`credit._verified` the Python reference the difftest (`tests/test_verified_difftest.py`) holds to it, string for string.

A window's evidence carries its statements, one a call with a drawn unit (`STATEMENT_KEYS`: the call, the paths of its
circuit file and its public file, and its sessions' directories), never a verdict. The C-Flock verifier of record
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
  is one outcome (`credit.PROOF_KEYS`), which `Credit.window` then holds to the draw and the window's registration.

Before any run: the circuit files' form (`FILES_FORM`), the statements' form (`STATEMENTS_FORM`, each call one of the
window's), one statement a call (a retry never reproves), a statement only for a call with a drawn unit, a committed
circuit file for its K (`noStatementFiles`), and the call's `--registered` file (its refusals, `registration`).
Every refusal is (code, why), the first rule the input breaks, with Python's words.
-/

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
Over the window's descriptor `descriptor`, its layout `layout` (`layoutOf` of it), its registered `entries`, open root
`window`, drawn `units`, `steps` rounds and Program salt `salt`. -/
def asks (statements files : Json) (descriptor : ByteArray) (layout entries : Json) (window : String)
    (units : List Nat) (steps : Nat) (salt : String) : R (Array Ask × Option (String × String)) := do
  unless isObj files && (fields files).all (fun (_, v) => (digest? v).isSome) do throw ("circuit", FILES_FORM)
  let some ss := (match statements with
    | .arr xs => xs.toList.mapM fun s => (s, ·) <$> call? s
    | _ => none) | throw ("proof", STATEMENTS_FORM)
  let sizes := (callWork layout).map (·.1)
  if ss.any (·.2 ≥ sizes.length) then throw ("proof", STATEMENTS_FORM)
  let starts := startsOf sizes
  let drawn := units.map fun u => (starts.filter (· ≤ u)).length - 1
  let program := Flock.hexOf (Flock.Sha512.hash descriptor)
  let shape := shapeOf descriptor
  let roots := rootsOf entries
  let lins : List (String × Nat) := (arrOf (get layout "linears")).filterMap fun x =>
    match arrOf x with
    | [.str s, k, _] => (s, ·) <$> nat? k
    | _ => none
  let callLin : List String := (arrOf (get layout "calls")).map fun x => ((arrOf x).head?.bind str?).getD ""
  let mut seen : List Nat := []
  let mut out : Array Ask := #[]
  let mut i := 0
  for (_, c) in ss do
    if seen.contains c then
      return (out, some ("outcome", s!"call {c} has two statements: a call's drawn units are proved in one statement " ++
        "and verified once, and a retry never reproves them"))
    seen := c :: seen
    unless drawn.contains c do
      return (out, some ("outcome", s!"call {c} has a statement and no drawn unit: only the draw's units are proved"))
    let lin := callLin.getD c ""
    let k := ((lins.reverse.find? (·.1 == lin)).map (·.2)).getD 0
    let some (.str sha) := PouwWindow.field? files s!"k{k}" | return (out, some ("circuit", noStatementFiles k))
    match registeredFile program layout shape c roots window steps salt with
    | .ok reg => out := out.push ⟨i, c, sha, reg⟩
    | .error why => return (out, some why)
    i := i + 1
  return (out, none)

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

def finding (f : Json) : R Finding := do
  match get f "session", get f "accepted", get f "why", get f "program",
      (·.map nat?) <$> (Json.getArr? (get f "units")).toOption with
  | .str s, .bool a, .str w, p, some us =>
    let some us := us.toList.mapM id | throw ("error", "credit-verified: a finding's units are not natural numbers")
    pure ⟨s, a, w, p, get f "registered", us⟩
  | _, _, _, _, _ =>
    throw ("error", "credit-verified: a finding is not {session, accepted, why, program, registered, units}")

/-- `credit._verified`'s judgment of the verifier's answers `runs` (one an `asks` run, in its order, as far as the
caller ran them): each drawn unit's outcome `{unit, verdict, program, registered}`, else the refusal of the first answer
that credits nothing, then `asks`' own refusal. An answer is `{"error": why}` (the verifier could not run),
`{"refused": [code, why]}` (it refused the statement) or `{"findings": [...]}` (`finding`, one a session). -/
def judge (statements files : Json) (descriptor : ByteArray) (layout entries : Json) (window : String)
    (units : List Nat) (steps : Nat) (salt : String) (runs : Array Json) : R (Array Json) := do
  let (as, refused) ← asks statements files descriptor layout entries window units steps salt
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
  if let some why := refused then throw why
  pure out

/-- `credit-verified`: on `{statements, files, descriptor, entries, window, units, steps, salt, runs}` (the window's
statements, its period's committed circuit files, its descriptor in hex, its registered entries, its open root, its
drawn units, its rounds and its Program salt), with `runs` null, `{"run": [...]}`, the verifier runs `asks` names
(`Ask.toJson`), or the verdict when it names none; with `runs` the answers, the verdict: `{"outcomes": [...]}` (`judge`)
or `{"refused": [code, why]}` (as `refusal` writes it). -/
def verified (j : Json) : R Json := do
  let some descriptor := (match get j "descriptor" with
    | .str s => if isHexBytes s then hex? (.str s) else none
    | _ => none) | layoutRefuse DESCRIPTOR_HEX
  let layout ← layoutOf descriptor
  let some units := (Json.getArr? (get j "units")).toOption.bind (·.toList.mapM nat?)
    | throw ("error", "credit-verified: the drawn units are not a list of natural numbers")
  let (some steps, some window, some salt) := (nat? (get j "steps"), str? (get j "window"), str? (get j "salt"))
    | throw ("error", "credit-verified: the window's steps, open root or salt are malformed")
  let (statements, files, entries) := (get j "statements", get j "files", get j "entries")
  match get j "runs" with
  | .null =>
    let (as, refused) ← asks statements files descriptor layout entries window units steps salt
    if as.isEmpty then
      if let some why := refused then throw why
      return Json.mkObj [("outcomes", Json.arr #[])]
    let partition := partitionOf (Flock.hexOf (Flock.Sha512.hash descriptor)) (shapeOf descriptor)
    return Json.mkObj [("run", Json.arr (as.map (·.toJson partition)))]
  | .arr runs =>
    let out ← judge statements files descriptor layout entries window units steps salt runs
    return Json.mkObj [("outcomes", Json.arr out)]
  | _ => throw ("error", "credit-verified: the verifier's answers are neither null nor a list")

end PouwCheck.Verified
