import PouwCheck.Verified

/-!
# A served window's live statements verified (`credit._live_verified`)

The verifier of record of a window's live@<s> statements, which `credit.window` and `credit.period_credit` take each
round's live proof outcome from (`credit._lean_live_verified`, through `pouw-check-difftest`'s `credit-live-verified`),
with `credit._live_verified` the Python reference the difftest (`tests/test_live_verified_difftest.py`) holds to it,
string for string. `Credit.Internal.live` then holds those outcomes to the window, as `credit._live` does.

A window's live evidence carries its live Program's descriptor and its statements, one a round (`LIVE_STATEMENT_KEYS`:
the round, the paths of its circuit file and its public file, and its sessions' directories), never a verdict. The
C-Flock verifier of record runs outside this process, so `liveVerified` is asked twice, as `Verified.verified` is:

* with no answers (`runs` null), it names each verifier run (`asks`): in statement order, up to the first statement a
  check before the verifier refuses, the statement's index, the live template's circuit file's SHA-512 the period's open
  committed (`CIRCUIT_FILES`' `live`), round s's `--registered` file (`readsFile`, `live_reads.reads_file`: its slot
  ports at round s's slot table, its header ports at round 0's request table, at the roots the window's entries
  register, `m` at the known `m@<s>` of the step's m_s, `s` at the known `round@<s>`, each `cap_<i>` at the known
  `caps` of the period's committed max_tokens and `o0` at the known `live-ok` rows) and the partition object (every
  instance of the live template one unit). The caller runs the verifier on exactly those inputs, against the live
  descriptor;
* with the answers (`runs`, one a run in that order, as far as the caller ran them), it judges them (`judge`): a
  verifier that could not run, or a session it rejected, refuses the window (`live`), a refusal of the statement
  refuses it with the verifier's code, and so does a session that proved another unit than its round's (`live`). Each
  accepted session is one outcome (`credit.PROOF_KEYS`, unit s), which `Credit.Internal.live` then holds to the window.

Before any run: the circuit files' form (`FILES_FORM`), a committed live circuit file (`noLiveFiles`), the statements'
form (`LIVE_STATEMENTS_FORM`), and one statement a round of the window's (a retry never reproves). Every refusal is
(code, why), the first rule the input breaks, with Python's words.
-/

namespace PouwCheck.LiveVerified

open Lean (Json toJson)
open PouwCheck.Credit
open PouwCheck.Verified (CIRCUIT_FILES finding)

def LIVE_STATEMENT_KEYS : List String := ["circuit", "public", "round", "sessions"]

def LIVE_STATEMENTS_FORM : String :=
  "the window's live evidence is not {descriptor, statements}: its live Program's descriptor in lowercase hex and " ++
  "a list of {circuit, public, round, sessions}, each round's live@<s> proof files: the paths of its circuit file " ++
  "and its public file, and its sessions' directories (at least one)"

def noLiveFiles : String :=
  s!"the period's open commits no circuit file for the live Program's template ({CIRCUIT_FILES} {LIVE}, the " ++
  "SHA-512 of the circuit file the auditor's lowering built for it), so no round's live@<s> can be verified: " ++
  "commit it in the period-open beside each K's"

/-- A live statement of the form `LIVE_STATEMENT_KEYS` asks: its round (the paths are the caller's to read). -/
def round? (s : Json) : Option Nat :=
  match keys s, nat? (get s "round"), get s "circuit", get s "public", get s "sessions" with
  | some ks, some r, .str _, .str _, .arr xs =>
    if ks == LIVE_STATEMENT_KEYS && !xs.isEmpty && xs.all (str? · |>.isSome) then some r else none
  | _, _, _, _, _ => none

/-- A known value's port of a `--registered` file (`live_reads.reads_file`'s `known`): the value `v`'s rows `rows` (each
with its salt), a row/v2 value of 16-bit leaves of `words` words under `SampledProofs.Registered.domainAt` of `program`
binding the window's open root `window`, read at each row of `rowsRead` by unit `s` alone, one port a row. -/
def knownPorts (program window v : String) (rows : Array (Array Nat × ByteArray)) (words s : Nat)
    (rowsRead : List Nat) : R (List Json) := do
  let d ← (SampledProofs.Registered.domainAt program v words rows.size 16 (some window)).mapError fun why =>
    (LIVE, why)
  let root ← knownRoot program window v rows words
  pure (rowsRead.map fun row => Json.mkObj [("value", Json.str v), ("root", Json.str root),
    ("leaves", toJson rows.size), ("words", toJson words), ("word_bits", toJson (16 : Nat)),
    ("domain", Json.str (Flock.hexOf d)), (SCHEMA_KEY, Json.str Flock.HmRow.ROW_SCHEMA_V2),
    ("reads", Json.arr ((List.replicate s Json.null ++ [toJson row]).toArray)),
    ("known", Json.mkObj [("rows", toJson (rows.map (·.1))),
      ("salts", Json.arr (rows.map fun k => Json.str (Flock.hexOf k.2)))])])

/-- `knownPorts` of the one row `row`. -/
def knownPort (program window v : String) (rows : Array (Array Nat × ByteArray)) (words s row : Nat) : R Json := do
  let some p := (← knownPorts program window v rows words s [row]).head? | throw ("error", "no port")
  pure p

/-- `live_reads.reads_file`: the verifier's `--registered` file (`Flock.Registered.FORMAT_V0`) of round s's statement
`live@<s>`, unit s of the live Program of `steps` rounds at T = `t`: its slot ports `slot_<i>` at round s's
`<pre>slots@<s>` and its header ports `req_<i>` at `<pre>requests@0` (a `row-seg/v1` value), at the roots `roots` gives
them, each under the registered domain of the window Program's digest `program`, the slot table bound to the window's
open root `window` and the request table to its session `session`; `m` and `s` at their known content
(`live_reads.m_table`, `round_table`: one row of two 16-bit words under `PouwCheck.publicRowSalt` of `window` and s),
each `cap_<i>` at row i of the known `<pre>caps` of the period's committed `maxTokens` (`live_reads.caps_table`, as
`Credit.capsRoot` reads it: row q request q's max_tokens, 0 past them, under `publicRowSalt` of `window` and q) and the
output row `o0` at the known `live-ok` rows (`ok_table`: row s' the one word 1 under `okSalt` of `window` and s').
Units other than s read nothing. -/
def readsFile (program pre window session : String) (t steps s m : Nat) (roots : List (String × String))
    (maxTokens : List Nat) : R Json := do
  let (sv, rv) := (s!"{pre}{SLOTS}@{s}", s!"{pre}{REQUESTS}@0")
  let reads := fun (row : Nat) => Json.arr ((List.replicate s Json.null ++ [toJson row]).toArray)
  let sd ← (SampledProofs.Registered.domainAt program sv SLOT_WORDS t 16 (some window)).mapError fun why =>
    (LIVE, why)
  let rd ← (SampledProofs.Registered.domainBound program rv HEADER_WORDS t 16 "session" session).mapError fun why =>
    (LIVE, why)
  let port := fun (v : String) (words : Nat) (d : ByteArray) (schema : String) (row : Nat) =>
    Json.mkObj [("value", Json.str v), ("root", Json.str ((roots.lookup v).getD "")), ("leaves", toJson t),
      ("words", toJson words), ("word_bits", toJson (16 : Nat)), ("domain", Json.str (Flock.hexOf d)),
      (SCHEMA_KEY, Json.str schema), ("reads", reads row)]
  let slots := (List.range t).map fun i => (s!"slot_{i}", port sv SLOT_WORDS sd Flock.HmRow.ROW_SCHEMA_V2 i)
  let reqs := (List.range t).map fun i => (s!"req_{i}", port rv HEADER_WORDS rd Flock.HmRow.ROW_SCHEMA_SEG i)
  let mp ← knownPort program window s!"{pre}{M_ROWS}@{s}"
    #[(#[m % 65536, m / 65536], PouwCheck.publicRowSalt window s)] SLOT_WORDS s 0
  let sp ← knownPort program window s!"{pre}{ROUND_ROWS}@{s}"
    #[(#[s % 65536, s / 65536], PouwCheck.publicRowSalt window s)] SLOT_WORDS s 0
  let capRows := (Array.range t).map fun q =>
    let c := maxTokens.getD q 0
    (#[c % 65536, c / 65536], PouwCheck.publicRowSalt window q)
  let caps := (List.range t).zip (← knownPorts program window s!"{pre}{CAPS}" capRows SLOT_WORDS s (List.range t))
    |>.map fun (i, p) => (s!"cap_{i}", p)
  let op ← knownPort program window s!"{pre}{LIVE_OK_VALUE}"
    ((Array.range steps).map fun x => (OK_ROW, okSalt window x)) OK_ROW.size s s
  pure (Json.mkObj [("format", Json.str Flock.Registered.FORMAT_V0),
    ("ports", Json.mkObj (slots ++ reqs ++ [(M_ROWS, mp), (ROUND_PORT, sp)] ++ caps ++ [(LIVE_OK_PORT, op)]))])

/-- `credit._live_asks`: the verifier runs in statement order up to the first statement a check before the verifier
refuses, and that refusal (none when every statement passes them), each `Verified.Ask` (its `call` the statement's
round). Over the period's committed circuit files `files`, the live descriptor `raw` (`liveOf` of it `got`), the live
tables' roots `roots` (`liveRoots`), each step's m_s `ms`, the window Program's digest `program`, its open root `window`
its session `session` and the period's committed max_tokens `maxTokens`. -/
def asks (files statements : Json) (got : Live) (roots : List (String × String)) (ms : List Nat)
    (program window session : String) (maxTokens : List Nat) : R (Array Verified.Ask × Option (String × String)) := do
  unless isObj files && (fields files).all (fun (_, v) => (digest? v).isSome) do
    throw ("circuit", Verified.FILES_FORM)
  let some (.str sha) := PouwWindow.field? files LIVE | throw ("circuit", noLiveFiles)
  let some ss := (match statements with
    | .arr xs => xs.toList.mapM round?
    | _ => none) | throw (LIVE, LIVE_STATEMENTS_FORM)
  let mut seen : List Nat := []
  let mut out : Array Verified.Ask := #[]
  let mut i := 0
  for s in ss do
    if got.s ≤ s || seen.contains s then
      return (out, some (LIVE, s!"round {s} has " ++
        s!"{if seen.contains s then "two live statements" else "a live statement"} and the window has {got.s} " ++
        "rounds: each round's live@<s> is one statement, verified once"))
    seen := s :: seen
    let reg ← readsFile program got.pre window session got.t got.s s (ms.getD s 0) roots maxTokens
    out := out.push ⟨i, s, sha, reg⟩
    i := i + 1
  return (out, none)

/-- The live descriptor's partition object: every instance of its template `unit` one unit. -/
def livePartition (digest unit : String) : Json :=
  Json.mkObj [("format", Json.str PARTITION_FORMAT), ("program", Json.str digest),
    ("query", Json.mkObj [("name", Json.str QUERY_NAME), ("version", toJson QUERY_VERSION),
      ("params", Json.mkObj [("templates", Json.arr #[Json.str unit])])])]

/-- `credit._live_verified`'s judgment of the verifier's answers `runs` (one an `asks` run, in its order, as far as the
caller ran them): each round's outcome `{unit, verdict, program, registered}`, else the refusal of the first answer
that credits nothing, then `asks`' own refusal. An answer is `Verified.judge`'s form. -/
def judge (files statements : Json) (got : Live) (roots : List (String × String)) (ms : List Nat)
    (program window session : String) (maxTokens : List Nat) (runs : Array Json) : R (Array Json) := do
  let (as, refused) ← asks files statements got roots ms program window session maxTokens
  let mut out : Array Json := #[]
  let mut j := 0
  for a in as do
    let s := a.call
    let some ans := runs[j]? | throw ("error", s!"credit-live-verified: no verifier answer for statement {a.statement}")
    j := j + 1
    match get ans "error", get ans "refused", get ans "findings" with
    | .str e, _, _ => throw (LIVE, s!"round {s}'s live@{s} could not be verified, so the window stays refused: {e}")
    | _, .arr #[.str code, .str why], _ => throw (code, s!"round {s}'s live@{s}: {why}")
    | _, _, .arr fs =>
      for fj in fs do
        let f ← finding fj
        unless f.accepted do
          throw (LIVE, s!"round {s}'s live@{s} session {f.session} was rejected by the verifier of record " ++
            s!"({f.why}): its served rows m_s = {ms.getD s 0} are not proved (no retry)")
        unless f.units == [s] do
          throw (LIVE, s!"round {s}'s live@{s} session {f.session} proved units {f.units}, not unit {s}: round s's " ++
            "statement proves its own round's unit")
        out := out.push (Json.mkObj [("unit", toJson s), ("verdict", Json.str "accept"), ("program", f.program),
          ("registered", f.registered)])
    | _, _, _ => throw ("error", "credit-live-verified: an answer is not {error}, {refused: [code, why]} or {findings}")
  if let some why := refused then throw why
  pure out

/-- `credit-live-verified`: on `{files, statements, descriptor, entries, ms, program, window, session, max_tokens,
runs}` (its period's committed circuit files, the window's live statements, its live descriptor in hex, its registered
entries, each step's m_s, the window Program's digest, its open root, its session and the max_tokens its period's open
committed), the live Program `liveOf` and its tables'
roots `liveRoots` first; with `runs` null, `{"run": [...]}`, the verifier runs `asks` names (`Verified.Ask.toJson`), or
the verdict when it names none; with `runs` the answers, the verdict: `{"outcomes": [...]}` (`judge`) or
`{"refused": [code, why]}`. -/
def liveVerified (j : Json) : R Json := do
  let some raw := (match get j "descriptor" with
    | .str s => if isHexBytes s then hex? (.str s) else none
    | _ => none) | throw ("error", "credit-live-verified: the live descriptor is not lowercase hex")
  let some ms := (Json.getArr? (get j "ms")).toOption.bind (·.toList.mapM nat?)
    | throw ("error", "credit-live-verified: the steps' served rows are not a list of natural numbers")
  let (some program, some window, some session) := (str? (get j "program"), str? (get j "window"),
      str? (get j "session"))
    | throw ("error", "credit-live-verified: the window's program, open root or session are malformed")
  let some maxTokens := (Json.getArr? (get j "max_tokens")).toOption.bind (·.toList.mapM nat?)
    | throw ("error", "credit-live-verified: the period's max_tokens are not a list of natural numbers")
  let got ← liveOf raw
  unless ms.length == got.s && ms.all (· < 2 ^ 32) do
    throw ("error", "credit-live-verified: the steps' served rows are not one u32 a round")
  unless (maxTokens.take got.t).all (· < 2 ^ 32) do
    throw ("error", "credit-live-verified: the period's public caps are not each a u32")
  let roots ← liveRoots (get j "entries") got.pre got.t got.s
  let (files, statements) := (get j "files", get j "statements")
  match get j "runs" with
  | .null =>
    let (as, refused) ← asks files statements got roots ms program window session maxTokens
    if as.isEmpty then
      if let some why := refused then throw why
      return Json.mkObj [("outcomes", Json.arr #[])]
    let partition := livePartition (Flock.hexOf (Flock.Sha512.hash raw)) got.unit
    return Json.mkObj [("run", Json.arr (as.map (·.toJson partition)))]
  | .arr runs =>
    let out ← judge files statements got roots ms program window session maxTokens runs
    return Json.mkObj [("outcomes", Json.arr out)]
  | _ => throw ("error", "credit-live-verified: the verifier's answers are neither null nor a list")

end PouwCheck.LiveVerified
