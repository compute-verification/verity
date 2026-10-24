import Std.Data.HashMap
import Flock.Keccak
import Flock.Program
import Flock.RegKnown
import PouwCommit.Frame
import PouwCheck.Json
import PouwCheck.Opening
import PouwCheck.Pass
import PouwCheck.Rounds
import PouwWindow.Draw
import PouwWindow.Rounds
import SampledProofs.ChallengerSession
import SampledProofs.PyJson
import SampledProofs.Registered

/-!
# A served window's credited set, its drawn units and their outcome, its A rows and its period (`credit.py`; RT4-INT-32, -34 and -42)

The verifier of record of `verity/core/protocols/pouw/credit.py`, which the difftest (`tests/test_difftest.py`) holds to
it, string for string:

**`window` and `periodCredit` are the only entries.** On one window's evidence and one round log, `window` runs every
step and credits only when all of them accept; `periodCredit` runs it on each window of a closed, complete period. The
steps are `Internal` (`credit.py`'s underscored ones), which only the difftest reaches, so no caller credits a window with
a step skipped: `Internal.credited` returns the credit bits whatever the proofs' outcome, and `Internal.aLink` trusts the
round log's roots, which only `roundsCheck` (`core_check.rounds`) holds to the ones the challenger received.

* `Internal.credited` is `credit._credited`: each manifest row (the window's calls, in order) carries its call's verdict as
  `credit`, a credited row its z table's root, and the draw root over the manifest (`callsRoot`, rows written as Python's
  `json.dumps(row, sort_keys=True)`, and the window's Commit record's digest, `commitOf?`), the last round's receipts and
  every round's root and coin (`drawRoot`) must be the one root the challenger's log shows its draw coin followed, read
  through the session the auditor's signed ledger opens for the window, run from a commit the host's pins log records on
  main's first-parent line (`trustedSessionOf`, over `pinnedSession`). So a credit bit changed after the draw coin is
  refused.
* `drawn` is `credit.drawn`: the units the draw coin draws from the window's own layout, `drawCheck`'s rule under the
  law `work:k` (`workUnits`), one stratum per K (`strata`, over `h1window.program`'s unit order) at the K the period's
  open committed (`law`; `committedK`). Never a list the prover hands in. `established` is the ε that draw established
  over the window's credited work at `DELTA` (`PouwWindow.epsAt` at the window's inputs), which `periodCredit` reports.
* `layoutOf` is `credit.layout_of`: the window's layout is its committed circuit's, read from the descriptor whose
  SHA-512 the period's open committed: its program function's record (`TILES`, `LINEARS`, `CALLS`, `M`), each K's
  template's A and B rows (R and C), and its root's nodes other than inputs, which must be each call's (M/R)·(n/C)
  instances of K's template in order. Never a layout the window's evidence hands in.
* `Internal.outcome` is `credit._outcome`: every unit `drawn` draws has exactly one proof outcome and it is an accept,
  else the whole window is refused (no partial credit). `window` is `credit.window`: `Internal.credited`, `roundsCheck`
  on the round log's roots and coins, `Internal.aLink` on that log and the window's registered entries, `layoutOf` of
  its descriptor, `periodWeights`, `Internal.arow`, then `Internal.outcome`'s rule under the served window's arguments
  (the stream its root, the registration its draw root, the receipt `{"receipts": …}`) and `Internal.registration`.
* `period` is `credit.period`: the windows presented are exactly those the auditor's ledger lists in the period's
  close, in its order, and the close is complete; `periodCredit` is `credit.period_credit`, every window's session run
  from main's code and the window proving the period's committed circuit (`creditedSession`, the pin first, then the
  circuit) and credited. Every path here that credits reads its session through `pinnedSession`, against the host's
  pins log (`pins`) and the verifier's own copy of main's first-parent history (`main`).
* `Internal.aLink` is `credit._a_link`: each registered A value (`aValue?`) is the frame-v3 SHA-512 tree over the
  aligned subtrees each round's root commits (`roundRoot`). Of S rounds, round s's tile a window's `…/a/<K>` of L rows
  over `[s·L/S, (s+1)·L/S)`, and round s's A table of K `…/a/<K>@<s>` (`h1window.table`) is committed whole by round s
  and listed by no other; round s's input-ids table `…/ids@<s>` is committed whole by round s.
* `Internal.arow` is `credit._arow` (ROOTS1): round s's A table of K `…/a/<K>@<s>` holds M rows for each of the round's
  calls of K (`stepTables`) and has the root, domain and leaf count of the Commit record's `row-table/<s>/k<K>`
  (`commitRows`), and round s's `…/ids@<s>` those of its `input-ids/<s>` (`commitIds`), so the A rows a call's units
  read are the served Commit's. Round 0's
  `…/ids@0` is also inside round 0's request table: round 0's `requests`, which the window's open root binds, is
  `requestsRoot` of its `request_table` and ids@0's root, so the prompt tokens are fixed before the salt coin.
* `Internal.registration` is `credit._registration`: each drawn unit's proof outcome is the verifier of record's
  verdict for its session (`PROOF_KEYS`: the SHA-512 of the descriptor it verified the session against, and the
  session's registered finding, each registered port's value and the root its reads were opened against). The entries
  are the window's registration's: each A value's domain, and each other value's a proof opened, is `domainOf` the
  committed descriptor's SHA-512 and, for every value but the weights' `…/b/<K>` (`isHeld`) and round 0's `…/ids@0`,
  the window's open root (`SampledProofs.Registered.domainAt`, PRF-127), for `…/ids@0` the window's session; each
  proof was verified against that descriptor, opened only values
  the window registers once, each at its registered root, and opened its unit's K's A value, the window's `…/a/<K>` or
  its call's round's `…/a/<K>@<s>`, and its K's weights `<PREFIX>b/<K>`. So the A rows `Internal.aLink` ties to the round roots are the ones each drawn unit's
  proof opened, under this window's registration.
* `Internal.weights` is `credit._weights`: each K's `<PREFIX>b/<K>` registered once, all of K's B rows under the
  registered domain of its name and the window's circuit, binding no window, and `weightsRoot` of their
  roots, the window's round-0 request table's root and its session are its open root (`windowOpenRoot`), so its B rows,
  codes and s_w, were fixed before its salt. `window` composes it right after `layoutOf` (`periodWeights`), and holds their root to the
  one the open of the window's period signed (`committedWeights`), so every window of a period opens the same B rows.
  That open names the model the root is of, a role and checkpoint `models.json` pins (`PINNED_MODELS`, the file
  `credit.py` reads, embedded here), else the period counts nothing.

Every refusal is (code, why), the first rule the input breaks, with Python's words.
-/

namespace PouwCheck.Credit

open Lean (Json)
open PouwCommit (Part tagged Domain)
open SampledProofs.Challenger (ChallengerKey LiveOpening trustedSessionOf liveReason)

def CALLS_TAG : ByteArray := "verity/pouw/h1/calls/v2".toUTF8.push 0
def DRAW_TAG : String := "verity/pouw/h1/draw/v1"
def ROUND_TAG : String := "verity/pouw/h1/round-a/v1"
def CREDIT : String := "credit"
def DEPTH : Nat := 64
def VERDICTS : List String := ["accept", "reject"]
def LEVELS : Nat := 64
def MAX_ROWS : Nat := 2 ^ 24

def SALT_FORM : String := "the window's Program salt is not a SHA-512 digest (128 lowercase hex)"
def CALLS_FORM : String := "the window's manifest is not a list of call rows"
def SERVED_FORM : String := "the served trees' digest is neither null nor a SHA-512 digest (128 lowercase hex)"
def DRAW_FORM : String := "the window's draw is not an object whose receipts are a SHA-512 digest"
def NO_DRAW : String :=
  "the challenger's session closed with no draw coin opened: the window opens its draw coin once, after its draw root"
def DRAW_ROOT : String :=
  "the window's draw root is not over its manifest (each call's credit bit among it), its Commit record, its last " ++
  "round's receipts and every round's root and coin"
def CHANGED : String :=
  "the draw root over the manifest (each call's credit bit among it), its Commit record, its last round's " ++
  "receipts and every round's root and coin is not the one root the challenger's log shows its draw coin " ++
  "followed: a credit bit, a call or the Commit record changed after the draw coin, or the draw opened after " ++
  "another root. Commit every call's credit bit in its manifest row, and the Commit record, before the draw " ++
  "root, and never change them after the draw coin"
def K_FIELD : String := "draw_k"
def WEIGHTS_FIELD : String := "weights_root"
def PERIOD_PIN_FIELD : String := "pinned"
def STALE_FIELD : String := "pin_floor"
def MAX_K : Nat := 2 ^ 24
def DELTA : Rat := mkRat 1 (2 ^ 40)
def K_FORM : String :=
  s!"the draw's K is not a positive integer at most 2^24 (the K its period's open committed as {K_FIELD})"
def LAYOUT_KEYS : List String := ["calls", "linears", "m", "tile"]
def LAYOUT_FORM : String :=
  "the window's layout is not {calls, linears, m, tile}, h1window.program's arguments: its calls " ++
  "[[linear, A block], ...] in order, its linears [[name, K, n], ...] each once, its A slot m and its tile " ++
  "[R, C], every number a natural number and R, C at least one"
def COIN_FORM : String := "the draw coin is not a SHA-512 digest (128 lowercase hex)"
def DRAW_ARGS : String :=
  "the draw's stream is not a string, its registration not a SHA-512 digest, or its receipt not an object of " ++
  "printable ASCII strings"
def WINDOW_KEYS : List String :=
  ["calls", "commit", "descriptor", "draw", "entries", "proofs", "rounds", "salt", "served", "session", "window"]
def PERIOD_KEYS : List String := WINDOW_KEYS
def WINDOWS_FORM : String :=
  "the period's windows are not a list of each window's evidence {" ++ ", ".intercalate PERIOD_KEYS ++ "}, " ++
  "in its close's order"
def DESCRIPTOR_HEX : String :=
  "the window's descriptor is not the bytes of the circuit its proofs were verified against in lowercase hex (its " ++
  "canonical JSON without annotations, the bytes program_digest hashes)"
/-- A drawn unit's proof outcome: the verifier of record's verdict for the session that proved it. -/
def PROOF_KEYS : List String := ["program", "registered", "unit", "verdict"]
def PROOFS_FORM : String :=
  "the drawn units' proof outcomes are not a list of {program, registered, unit, verdict}, each the verifier of " ++
  "record's verdict for a drawn unit's session: verdict accept or reject, program the SHA-512 of the circuit's " ++
  "descriptor it verified the session against, and registered its finding, each registered port's {value, root}, " ++
  "port and value printable ASCII and root a SHA-512 digest"
def LOG_FORM : String := "the round log is not a nonempty list"
def ENTRIES_FORM : String := "the window's registered entries are not a list of objects"
def NO_A : String :=
  "the window registers no A value (…/a/<K>, or a round's …/a/<K>@<s>), so nothing ties its units' A rows to " ++
  "the round roots: register the window's A values"
def COMMIT_TAG : ByteArray := "verity/pouw/h1/commit/v1".toUTF8.push 0
/-- The served Commit's record: one-stage's registration record with registered values. -/
def COMMIT_FORMAT : String := "verity/registration/v2"
def COMMIT_FORM : String :=
  "the window's Commit record is not the bytes in lowercase hex of a verity/registration/v2 record in canonical " ++
  "JSON whose registered values are a list of objects: the served Commit runs in the window's engine, and its " ++
  "record is bound under the window's calls_root before the draw coin"

/-- `credit.commit_rows`: the Commit's name for the A rows (its AROW) of the window's round `s`'s calls of contraction
`k`, its A table of K `…/a/<K>@<s>`. -/
def commitRows (s k : Nat) : String := s!"row-table/{s}/k{k}"

/-- `credit.commit_ids`: the Commit's name for its step's input token ids at the window's round `s`. -/
def commitIds (s : Nat) : String := s!"input-ids/{s}"

def noBit (i : Nat) : String :=
  s!"manifest row {i} has no credit bit: the serving process commits each call's verdict as `{CREDIT}` (true or " ++
  "false) in its manifest row, under calls_root, before the draw coin"

def noZ (i : Nat) : String :=
  s!"call {i} is credited with no z table root: a credited call's manifest row carries its z table's root " ++
  "(`tables.z`), or its credit bit is false"

/-- `_digest`: a SHA-512 digest written as 128 lowercase hex. -/
def digest? (j : Json) : Option ByteArray :=
  if (str? j).any isHexDigest then hex? j else none

def isObj : Json → Bool
  | .obj _ => true
  | _ => false

/-- An object's fields in key order (Python's `sorted`). -/
def fields : Json → List (String × Json)
  | .obj kvs => (kvs.toArray.toList.map fun (k, v) => (k, v)).mergeSort fun a b => a.1 ≤ b.1
  | _ => []

/-- `_plain`: JSON of objects, lists, strings, integers, booleans and null, nested at most `depth` deep. -/
def plain : Nat → Json → Bool
  | 0, _ => false
  | _ + 1, .null | _ + 1, .bool _ | _ + 1, .str _ => true
  | _ + 1, .num n => n.exponent == 0
  | d + 1, .arr a => a.all (plain d)
  | d + 1, .obj kvs => kvs.toArray.all fun (_, v) => plain d v

/-- `json.dumps(x, sort_keys=True)` of a value `plain` holds at depth `fuel`. -/
def dumps : Nat → Json → String
  | 0, _ => ""
  | _ + 1, .null => "null"
  | _ + 1, .bool b => if b then "true" else "false"
  | _ + 1, .num n => toString n.mantissa
  | _ + 1, .str s => SampledProofs.PyJson.str s
  | d + 1, .arr a => "[" ++ ", ".intercalate (a.toList.map (dumps d)) ++ "]"
  | d + 1, j@(.obj _) =>
    "{" ++ ", ".intercalate ((fields j).map fun (k, v) => SampledProofs.PyJson.str k ++ ": " ++ dumps d v) ++ "}"

/-- `calls_root`: SHA-512 over each row as `json.dumps(row, sort_keys=True)` and a newline, under `CALLS_TAG` and the
window's Program salt; with the served trees' digest, SHA-512 of that and it; with the Commit record's digest, SHA-512
under `COMMIT_TAG` of that and it. -/
def callsRoot (salt : ByteArray) (rows : List Json) (served : Option ByteArray) (commit : Option ByteArray := none) :
    ByteArray :=
  let h := Flock.Sha512.hash (rows.foldl (fun acc r => acc ++ (dumps DEPTH r ++ "\n").toUTF8) (CALLS_TAG ++ salt))
  let d := match served with
    | none => h
    | some s => Flock.Sha512.hash (h ++ s)
  commit.elim d fun c => Flock.Sha512.hash (COMMIT_TAG ++ d ++ c)

/-- `credit._HEX`: bytes as lowercase hex, at least one. -/
def isHexBytes (s : String) : Bool :=
  !s.isEmpty && s.length % 2 == 0 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

/-- `credit.commit_of`: the digest of the window's Commit record `commit` (the bytes in lowercase hex of its canonical
JSON, the bytes `registration.digest` hashes under its format) and its registered values; none when it is not one. -/
def commitOf? (commit : Json) : Option (ByteArray × Array Json) :=
  (str? commit).bind fun s =>
  if isHexBytes s then
    (hex? commit).bind fun raw =>
    (String.fromUTF8? raw).bind fun text =>
    (Json.parse text).toOption.bind fun j =>
    (SampledProofs.PyJson.canonical j).toOption.bind fun t =>
    (Json.getArr? (get j "registered")).toOption.bind fun xs =>
    if t == text && isObj j && get j "format" == Json.str COMMIT_FORMAT && xs.all isObj then
      some (Flock.Sha512.hash ((COMMIT_FORMAT.toUTF8.push 0) ++ raw), xs)
    else none
  else none

/-- `credit._credited`: the credited calls' indices, else the refusal. A step of `window`: its indices are no credit
until every other step accepts. -/
def Internal.credited (key ledger pins main window session salt calls served commit rounds draw : Json) :
    R (List Nat) := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let (c, received) ← (trustedSessionOf ledger window k pins main session).mapError fun e =>
    ("coin", s!"the challenger's session is not trusted: {e}")
  let some s := digest? salt | throw ("manifest", SALT_FORM)
  let .arr rows := calls | throw ("manifest", CALLS_FORM)
  let rows := rows.toList
  let mut out : List Nat := []
  for (row, i) in rows.zipIdx do
    unless isObj row && plain DEPTH row do
      throw ("manifest", s!"manifest row {i} is not an object of objects, lists, strings, integers, booleans and null")
    unless PouwWindow.intOf? (get row "index") == some (i : Int) do
      throw ("manifest", s!"manifest row {i} is not call {i}: the rows are the window's calls in order, index 0, 1, …")
    let .bool bit := get row CREDIT | throw ("credit", noBit i)
    if bit then
      if (digest? (get (get row "tables") "z")).isNone then throw ("credit", noZ i)
      out := out ++ [i]
  let servedB ← match served with
    | .null => pure none
    | j => match digest? j with
      | some b => pure (some b)
      | none => throw ("manifest", SERVED_FORM)
  let cd ← (commitOf? commit).elim (throw ("commit", COMMIT_FORM)) pure
  let .arr xs := rounds | throw ("rounds", "the round log is not a list")
  let mut parts : List Part := []
  for (x, r) in xs.toList.zipIdx do
    let some root := (if isObj x then digest? (get x "root") else none)
      | throw ("rounds", s!"round {r}'s root is not a SHA-512 digest")
    let o ← (LiveOpening.ofJson (get x "coin")).mapError fun e => ("rounds", s!"round {r}'s coin: {e}")
    parts := parts ++ [.bytes root, .bytes o.coin]
  let some receipts := (if isObj draw then digest? (get draw "receipts") else none) | throw ("draw", DRAW_FORM)
  let root := tagged DRAW_TAG ([.bytes (callsRoot s rows servedB (some cd.1)), .bytes receipts] ++ parts)
  unless (str? (get draw "root")) == some (Flock.hexOf root) do throw ("draw", DRAW_ROOT)
  let got := (received.lookup PouwWindow.DRAW).getD []
  if got.isEmpty then throw ("draw", NO_DRAW)
  unless got == [root] do throw ("draw", CHANGED)
  let why := match LiveOpening.ofJson (get draw "coin") with
    | .error e => some e
    | .ok o => match liveReason c o PouwWindow.DRAW 0 root with
      | .error e => some e
      | .ok w => w
  if let some w := why then throw ("coin", s!"the draw coin: {w}")
  pure out

def layoutRefuse {α : Type} (why : String) : R α := throw ("layout", why)

/-- `[a, b, …]`, as Python writes a list of integers. -/
def listStr (xs : List Nat) : String := "[" ++ ", ".intercalate (xs.map toString) ++ "]"

/-- `credit.strata`: the work strata of the served H1 window whose Program is `h1window.program` of `layout`
(`{calls, linears, m, tile}`), else the refusal. Its units are the Program's, call by call, each call's (m/R)·(n/C)
tiles row block major; K's stratum is the units of the calls of K's linears, template `k<K>`, its ranges in order
(adjacent ones merged), the strata in the order of their first unit, each unit's work R·C·K and every floor one. Its cost
is the layout's calls, never its units. -/
def strata (layout : Json) : R (Array Flock.Draw.WorkStratum) := do
  unless keys layout == some LAYOUT_KEYS do layoutRefuse LAYOUT_FORM
  let (.arr tile, some m, .arr linears, .arr calls) :=
      (get layout "tile", nat? (get layout "m"), get layout "linears", get layout "calls") | layoutRefuse LAYOUT_FORM
  let some [r, c] := tile.toList.mapM nat? | layoutRefuse LAYOUT_FORM
  unless 1 ≤ r && 1 ≤ c && !linears.isEmpty && !calls.isEmpty do layoutRefuse LAYOUT_FORM
  let mut ks : Array (String × Nat × Nat) := #[]
  for x in linears do
    let some (name, k, n) := (match x with
        | .arr a => match a.toList with
          | [.str s, k, n] => do pure (s, ← nat? k, ← nat? n)
          | _ => none
        | _ => none) | layoutRefuse LAYOUT_FORM
    if ks.any (·.1 == name) then layoutRefuse s!"linear {name} is named twice: a window names each of its weights once"
    ks := ks.push (name, k, n)
  let mut cs : Array (String × Nat) := #[]
  for x in calls do
    let some (lin, blk) := (match x with
        | .arr a => match a.toList with
          | [.str s, b] => (s, ·) <$> nat? b
          | _ => none
        | _ => none) | layoutRefuse LAYOUT_FORM
    unless ks.any (·.1 == lin) do
      layoutRefuse s!"a call of {lin}, which is no linear of the window: every call is of one of its linears"
    cs := cs.push (lin, blk)
  for (name, _) in ks do
    unless cs.any (·.1 == name) do layoutRefuse s!"linear {name} is never called: every linear of the window is called"
  if m < r || m &&& (m - 1) != 0 then
    layoutRefuse s!"the A slot {m} is no power of two of at least the tile's {r} rows (h1window.slot)"
  for (name, k, n) in ks do
    if k < 1 || n < c || n % c != 0 then
      layoutRefuse s!"linear {name}: K = {k} and n = {n} are not K >= 1 and n whole blocks of the tile's {c} columns"
  let kOf : String → Nat := fun lin => ((ks.find? (·.1 == lin)).map (·.2.1)).getD 0
  for k in (ks.toList.map (·.2.1)).eraseDups.mergeSort (· ≤ ·) do
    let used := ((cs.toList.filter fun (lin, _) => kOf lin == k).map (·.2)).eraseDups.mergeSort (· ≤ ·)
    unless used == List.range used.length do
      layoutRefuse s!"the A blocks of K = {k} are {listStr used}: each of 0, 1, ... is read by some call of K"
  let mut out : Array Flock.Draw.WorkStratum := #[]
  let mut at_ := 0
  for (lin, _) in cs do
    let some (_, k, n) := ks.find? (·.1 == lin) | layoutRefuse LAYOUT_FORM
    let size := (m / r) * (n / c)
    let t := s!"k{k}"
    match out.findIdx? (·.template == t) with
    | none => out := out.push { template := t, units := #[(at_, at_ + size)], k := 0, work := r * c * k, floor := 1 }
    | some i => out := out.modify i fun s =>
        match s.units.back? with
        | some (lo, hi) =>
          if hi == at_ then { s with units := s.units.pop.push (lo, at_ + size) }
          else { s with units := s.units.push (at_, at_ + size) }
        | none => { s with units := s.units.push (at_, at_ + size) }
    at_ := at_ + size
  pure out

/-- `credit.law`: the strata under the work law `work:k` at the committed K `k`, each stratum's `k_s`
`Flock.Draw.workK`; else the refusal (`budget`): `k` isn't a positive integer at most `MAX_K`, or the window has fewer
than K units. -/
def law (ss : Array Flock.Draw.WorkStratum) (k : Json) : R (Array Flock.Draw.Stratum) := do
  let some k := (nat? k).filter (fun k => 1 ≤ k && k ≤ MAX_K) | throw ("budget", K_FORM)
  let n := ss.foldl (fun a s => a + s.size) 0
  if n < k then
    throw ("budget", s!"the window's {n} units are fewer than K = {k}, the K its period committed, so its draw " ++
      s!"would not be K's: the proof service commits a K at most every window's units ({K_FIELD} in the period-open)")
  let total := Flock.Draw.totalWork ss
  pure (ss.map fun s => { s.toStratum with k := Flock.Draw.workK k s.work s.size total s.floor })

def SCHEMA : String := "verity-ir/descriptor/v1"
def WINDOW_FN : String := "PouwH1Window_v3"
def RECORD : List String := ["TILES", "LINEARS", "CALLS", "M", "PREFIX"]
def TILE_FN : String := "PouwH1WindowTile_v1"
def SHARED : List String := ["zy", "salt", "position", "roots"]
def DESCRIPTOR_FORM : String :=
  "the window's descriptor is not a verity-ir/descriptor/v1 descriptor in canonical JSON (keys sorted, compact, " ++
  "ASCII: the bytes program_digest hashes) with its callees, definitions, program function and root"
def NOT_WINDOW : String :=
  "the window's circuit is not an H1 window: its program function is not PouwH1Window_v3{TILES=…,LINEARS=…," ++
  "CALLS=…,M=…,PREFIX=…}, each binding canonical JSON, its TILES [[K, {\"fn\": template}], ...] by ascending K and " ++
  "its PREFIX a string (h1window.program's Program)"
def BODY : String :=
  "the window's circuit's units are not its record's tiles: its root's nodes other than its inputs are not, in " ++
  "order, each of its CALLS' (M/R)·(n/C) instances of K's template, each node a call of it or a batch of at least " ++
  "one (h1window.program's body)"

def noTemplate (k : Nat) : String :=
  s!"the window's template at K = {k} is not h1window's: a composite {TILE_FN} bound to its UNIT and K = {k}, " ++
  s!"its id {TILE_FN}\{UNIT=…,K={k}}, whose parameters are a_k{k} and b_k{k}, the tile's R and C rows, then " ++
  ", ".intercalate SHARED ++ ", at the same R × C at every K"

/-- `credit._loads`: `text` read as JSON, when it is that value's canonical JSON (`SampledProofs.PyJson.canonical`,
Python's `canonical_json`), so every reading of it is the same. -/
def loads? (text : String) : Option Json :=
  match Json.parse text with
  | .ok j => match SampledProofs.PyJson.canonical j with
    | .ok s => if s == text then some j else none
    | .error _ => none
  | .error _ => none

/-- `credit._tile`: the R × C of K = `k`'s template `dj` (its definition, under the id `tid`), or none when it is not
`h1window.template`'s. -/
def tileOf (dj : Json) (tid : String) (k : Nat) : Option (Nat × Nat) := do
  guard (isObj dj && get dj "form" == Json.str "composite" && get dj "id" == Json.str TILE_FN)
  let b := get dj "bindings"
  let body := get dj "body"
  guard (keys b == some ["K", "UNIT"] && nat? (get b "K") == some k && isObj body)
  let .ok u := SampledProofs.PyJson.canonical (get b "UNIT") | none
  guard (tid == s!"{TILE_FN}\{UNIT={u},K={k}}")
  let .arr ps := get body "params" | none
  let ps ← ps.toList.mapM fun (p : Json) => (match p with
    | .arr a => (match a.toList with
      | [n, t] => some (n, t)
      | _ => none)
    | _ => none)
  guard (ps.map (·.1) == ([s!"a_k{k}", s!"b_k{k}"] ++ SHARED).map Json.str)
  let rows ← (ps.take 2).mapM fun ((_, t) : Json × Json) => (match t with
    | .arr a => (match a.toList with
      | [.str "Array", n, _] => (nat? n).filter (1 ≤ ·)
      | _ => none)
    | _ => none)
  match rows with
  | [r, c] => some (r, c)
  | _ => none

/-- `credit._runs`: `(template, count)` pairs with adjacent ones of one template merged. -/
def runs (xs : List (String × Nat)) : List (String × Nat) :=
  (xs.foldl (fun acc (t, n) => match acc with
    | (u, m) :: rest => if u == t then (t, m + n) :: rest else (t, n) :: acc
    | [] => [(t, n)]) []).reverse

/-- `credit.layout_of`: the layout (`{calls, linears, m, tile}`, `strata`'s) of the H1 window whose circuit's
descriptor is `descriptor` (its bytes, canonical JSON without annotations, the bytes the period's `circuit` is the
SHA-512 of), else the refusal (`layout`). The calls, linears and slot are its program function's record (`WINDOW_FN`'s
`CALLS`, `LINEARS` and `M`), each K's template (`TILES`) is `h1window.template`'s, whose A and B rows give R and C, and
the root's nodes other than inputs are each call's (M/R)·(n/C) instances of its K's template, in the calls' order: a
call one instance and a batch its members, as the template query counts units. Its cost is the descriptor's bytes,
never the gates its nodes unroll to. -/
def layoutOf (descriptor : ByteArray) : R Json := do
  let some d := (String.fromUTF8? descriptor).bind loads? | layoutRefuse DESCRIPTOR_FORM
  let defs := get d "definitions"
  let (.str schema, .arr cs, .str fn, .str root) := (get d "schema", get d "callees", get d "program_fn", get d "root")
    | layoutRefuse DESCRIPTOR_FORM
  let some callees := cs.toList.mapM str? | layoutRefuse DESCRIPTOR_FORM
  unless schema == SCHEMA && isObj defs do layoutRefuse DESCRIPTOR_FORM
  let callees := callees.toArray
  let pairs := if fn.startsWith (WINDOW_FN ++ "{") then Flock.Program.recordPairs fn else #[]
  unless pairs.toList.map (·.1) == RECORD do layoutRefuse NOT_WINDOW
  let some vals := pairs.toList.mapM (fun (_, s) => loads? s) | layoutRefuse NOT_WINDOW
  let [tiles, linears, calls, m, prefix_] := vals | layoutRefuse NOT_WINDOW
  let some ts := (match tiles with
      | .arr a => a.toList.mapM fun (t : Json) => (match t with
        | .arr p => (match p.toList with
          | [k, f] => do
            let k ← (nat? k).filter (1 ≤ ·)
            guard (keys f == some ["fn"])
            pure (k, ← str? (get f "fn"))
          | _ => none)
        | _ => none)
      | _ => none) | layoutRefuse NOT_WINDOW
  unless !ts.isEmpty && (str? prefix_).isSome && (ts.zip ts.tail).all (fun (a, b) => a.1 < b.1) do
    layoutRefuse NOT_WINDOW
  let mut rc : Option (Nat × Nat) := none
  for (k, tid) in ts do
    let some got := tileOf (get defs tid) tid k | layoutRefuse (noTemplate k)
    if rc.any (· != got) then layoutRefuse (noTemplate k)
    rc := some got
  let some (r, c) := rc | layoutRefuse NOT_WINDOW
  let layout := Json.mkObj [("calls", calls), ("linears", linears), ("m", m),
    ("tile", Json.arr #[Lean.toJson r, Lean.toJson c])]
  discard <| strata layout
  let rows := fun (j : Json) => (match j with
    | .arr a => a.toList.filterMap fun (x : Json) => (match x with
      | .arr b => some b.toList
      | _ => none)
    | _ => ([] : List (List Json)))
  let lins : List (String × Nat × Nat) := (rows linears).filterMap fun (x : List Json) => (match x with
    | [.str s, k, n] => do pure (s, ← nat? k, ← nat? n)
    | _ => none)
  let tks := ts.map (·.1)
  let lks := (lins.map (·.2.1)).eraseDups.mergeSort (· ≤ ·)
  unless tks == lks do
    layoutRefuse (s!"the window's TILES are at K = {listStr tks}, not at its linears' K = {listStr lks}: h1window " ++
      "binds one template per K its linears have")
  let mv := (nat? m).getD 0
  let want := runs <| (rows calls).filterMap fun (x : List Json) => (match x with
    | [.str lin, _] => (lins.find? (·.1 == lin)).bind fun (_, k, n) =>
        (ts.find? (·.1 == k)).map fun (_, tid) => (tid, (mv / r) * (n / c))
    | _ => none)
  let rootDef := get defs root
  let .arr nodes := (if get rootDef "form" == Json.str "composite" then get (get rootDef "body") "nodes" else .null)
    | layoutRefuse BODY
  let mut got : Array (String × Nat) := #[]
  for x in nodes do
    let some f := (if isObj x then nat? (get x "fn") else none) | layoutRefuse BODY
    let some t := callees[f]? | layoutRefuse BODY
    if Flock.Program.isInputId t then continue
    let n := if get x "form" == Json.str "call" then some 1
      else if get x "form" == Json.str "batch" then nat? (get x "n") else none
    let some n := n.filter (1 ≤ ·) | layoutRefuse BODY
    got := got.push (t, n)
  unless runs got.toList == want do layoutRefuse BODY
  pure layout

/-- `credit.call_work`: each call's units and the work of each of them, `(size, work)` in call order, of a layout
`strata` accepted. -/
def callWork (layout : Json) : List (Nat × Nat) := Id.run do
  let num := fun j => (nat? j).getD 0
  let (r, c) := (match get layout "tile" with
    | .arr a => (num (a.getD 0 .null), num (a.getD 1 .null))
    | _ => (1, 1))
  let m := num (get layout "m")
  let rows := fun (j : Json) => (match j with
    | .arr a => a.toList.filterMap fun (x : Json) => (match x with
      | .arr b => some b.toList
      | _ => none)
    | _ => ([] : List (List Json)))
  let lins : List (String × Nat × Nat) := (rows (get layout "linears")).filterMap fun (x : List Json) => (match x with
    | [.str s, k, n] => some (s, num k, num n)
    | _ => none)
  pure <| (rows (get layout "calls")).filterMap fun (x : List Json) => (match x with
    | [.str lin, _] => (lins.find? (·.1 == lin)).map fun (_, k, n) => ((m / r) * (n / c), r * c * k)
    | _ => none)

/-- `credit.established`: the ε a draw of K = `k` under the work law over the window `layout` establishes at `DELTA`
for its credited calls `got`: `PouwWindow.epsAt K δ C W 1`, `C` the credited calls' units' work and `W` the window's
(the work law misses a set of work `εC` with probability at most `(1 − εC/W)^K`, its cap `K ≥ W`); none when it
establishes no ε ≤ 1. -/
def established (layout : Json) (got : List Nat) (k : Nat) : Option Rat :=
  let cw := callWork layout
  let credited := ((List.range cw.length).zip cw).foldl
    (fun a (i, s, w) => if got.contains i then a + s * w else a) 0
  let total := cw.foldl (fun a (s, w) => a + s * w) 0
  (PouwWindow.epsAt k DELTA credited total 1).toOption

/-- `credit._drawn`: the units the draw coin `coin` draws over `ss` (`workUnits`), else the refusal. -/
def drawnOf (coin : ByteArray) (stream registration receipt : Json) (ss : Array Flock.Draw.WorkStratum) (k : Json) :
    R (Array Nat) := do
  let some (st, reg, kvs) := (match stream, registration, digest? registration, receipt with
      | .str st, .str reg, some _, .obj kv => (st, reg, ·) <$> kv.toArray.toList.mapM fun (x, v) => match v with
        | .str s => if _root_.PouwCheck.plain x && _root_.PouwCheck.plain s then some (x, s) else none
        | _ => none
      | _, _, _, _ => none) | throw ("draw", DRAW_ARGS)
  let some units ← port (workUnits coin st reg kvs (← law ss k)) | throw ("sampler", "the draw's stream ran out")
  pure units

/-- `credit._finding`: a session's registered finding (flock-verify's `registered`, `{port: {value, root}}`: each
registered port's value and the root its reads were opened against) as `(port, value, root)` in port order. -/
def finding? (x : Json) : Option (List (String × String × String)) :=
  if isObj x then (fields x).mapM fun (port, e) =>
    match keys e, get e "value", get e "root" with
    | some ["root", "value"], .str v, .str r =>
      if _root_.PouwCheck.plain port && _root_.PouwCheck.plain v && (digest? (.str r)).isSome then some (port, v, r)
      else none
    | _, _, _ => none
  else none

/-- `credit._proved`: none when every unit of `drawn` has exactly one outcome and it is an accept, and no other unit
has one; else the refusal (`outcome`, `missing` or `failed`). -/
def proved (drawn : Array Nat) (proofs : Json) : R Unit := do
  let picked : Std.HashMap Nat Unit := drawn.foldl (fun m u => m.insert u ()) {}
  let .arr ps := proofs | throw ("outcome", PROOFS_FORM)
  let mut seen : Std.HashMap Nat String := {}
  for p in ps.toList do
    let some (u, v) := (if keys p == some PROOF_KEYS && (digest? (get p "program")).isSome
          && (finding? (get p "registered")).isSome then
        match nat? (get p "unit"), get p "verdict" with
        | some u, .str v => if VERDICTS.contains v then some (u, v) else none
        | _, _ => none
      else none) | throw ("outcome", PROOFS_FORM)
    if seen.contains u then
      throw ("outcome", s!"unit {u} has two proof outcomes: a drawn unit is proved and verified once, and a retry " ++
        "never reproves it")
    unless picked.contains u do
      throw ("outcome", s!"unit {u} has a proof outcome and was not drawn: only the draw's units are proved")
    seen := seen.insert u v
  for u in drawn do
    match seen.get? u with
    | none => throw ("missing", s!"drawn unit {u} has no proof outcome: every drawn unit's proof is verified, and a " ++
        "window with one missing is refused whole (no partial credit). Prove and verify every drawn unit")
    | some v =>
      unless v == "accept" do
        throw ("failed", s!"drawn unit {u}'s proof was rejected: a failed drawn unit refuses the whole window, not " ++
          "only its call (no partial credit, and no retry)")

/-- `credit.drawn`: the units the draw coin `coin` (its value, a SHA-512 digest in hex) draws from the window `layout`,
at the committed K `k`, ascending, else the refusal (`layout`, `draw`, `budget` or `sampler`). -/
def drawn (coin stream registration receipt layout k : Json) : R (Array Nat) := do
  let ss ← strata layout
  let some c := digest? coin | throw ("draw", COIN_FORM)
  drawnOf c stream registration receipt ss k

/-- `credit._outcome`: none when every unit the draw coin draws from the window (`drawn`) has exactly one outcome and
it is an accept, and no other unit has one; else the refusal. `window`'s rule on the draw coin it read. -/
def Internal.outcome (coin stream registration receipt layout k proofs : Json) : R Unit := do
  proved (← drawn coin stream registration receipt layout k) proofs

/-- An A value's segments by name, each `(level, index, root)` in row order. -/
abbrev Segments := List (String × List (Nat × Nat × ByteArray))

/-- `credit.round_root`: the root the challenger receives for round `r`, over the serving process's own round root and,
for each A value by name, its segments. -/
def roundRoot (r : Nat) (inner : ByteArray) (segs : Segments) : ByteArray :=
  tagged ROUND_TAG ([.int r, .bytes inner] ++ (segs.mergeSort fun a b => a.1 ≤ b.1).flatMap fun (v, ss) =>
    ss.flatMap fun (l, i, h) => [.str v, .int l, .int i, .bytes h])

def segment? (s : Json) : Option (Nat × Nat × ByteArray) :=
  match s with
  | .arr a => match a.toList with
    | [l, i, h] => do
      let l ← nat? l
      let i ← nat? i
      if l < LEVELS then (l, i, ·) <$> digest? h else none
    | _ => none
  | _ => none

/-- `credit._segments`. -/
def segments? (a : Json) : Option Segments :=
  if isObj a then (fields a).mapM fun (v, ss) => match ss with
    | .arr ss => (v, ·) <$> ss.toList.mapM segment?
    | _ => none
  else none

/-- `credit._DIGITS`: a contraction K in a registered value's name, a canonical decimal (no leading zero). -/
def canonicalK (k : String) : Bool := !k.isEmpty && k.all Char.isDigit && !k.startsWith "0"

def digits (s : String) : Bool := !s.isEmpty && s.all Char.isDigit

/-- `credit.a_value`: a registered A value's contraction and round, as their digits: a name whose last two parts are `a`
and either a contraction (`…/a/<K>`, the window's A rows of K), its round none, or a contraction and a round
(`…/a/<K>@<s>`, that round's A table of K), K a canonical decimal. -/
def aValue? : Json → Option (String × Option String)
  | .str v => match (v.splitOn "/").reverse with
    | last :: "a" :: _ => match last.splitOn "@" with
      | [k] => if canonicalK k then some (k, none) else none
      | [k, c] => if canonicalK k && digits c then some (k, some c) else none
      | _ => none
    | _ => none
  | _ => none

/-- `credit.is_a`: `…/a/<K>` or a round's `…/a/<K>@<s>` (`aValue?`). -/
def isA (v : Json) : Bool := (aValue? v).isSome

/-- `credit.ids_step`: an input-ids table's round, a name whose last part is `ids@<s>` (`<prefix>ids@<s>`, slot i's
input token id at round s, s a canonical decimal). -/
def idsStep? (v : Json) : Option Nat :=
  (str? v).bind fun v =>
  let last := (v.splitOn "/").getLastD ""
  let s := String.ofList (last.toList.drop 4)
  if last.startsWith "ids@" && (s == "0" || canonicalK s) then some s.toNat! else none

/-- `credit.receipt_call`: a receipt table's call, a name whose last part is `receipt@<c>` (`<prefix>receipt@<c>`, row t
call c's t-th unit's receipt, c a canonical decimal). -/
def receiptCall? (v : Json) : Option Nat :=
  (str? v).bind fun v =>
  let last := (v.splitOn "/").getLastD ""
  let c := String.ofList (last.toList.drop 8)
  if last.startsWith "receipt@" && (c == "0" || canonicalK c) then some c.toNat! else none

/-- `credit.pub_call`: a public table's call, a name whose last part is `pub@<c>` (`<prefix>pub@<c>`, row t call c's
t-th unit's public row, which `registeredFile` builds, c a canonical decimal). -/
def pubCall? (v : Json) : Option Nat :=
  (str? v).bind fun v =>
  let last := (v.splitOn "/").getLastD ""
  let c := String.ofList (last.toList.drop 4)
  if last.startsWith "pub@" && (c == "0" || canonicalK c) then some c.toNat! else none

/-- `credit.is_tied`: a value the round roots commit, an A value (`isA`), an input-ids table (`idsStep?`) or a receipt
table (`receiptCall?`). -/
def isTied (v : Json) : Bool := isA v || (idsStep? v).isSome || (receiptCall? v).isSome

def what (v : String) : String :=
  if isA (.str v) then "A value" else if (idsStep? (.str v)).isSome then "input-ids table" else "receipt table"

/-- `credit.is_held`: a value held across windows, the weights' rows (`…/b/<K>`, K a canonical decimal); every other
value a window registers is its own. -/
def isHeld : Json → Bool
  | .str v => match (v.splitOn "/").reverse with
    | k :: "b" :: _ => canonicalK k
    | _ => false
  | _ => false

/-- The node at (`level`, `index`) of `d`'s tree from the subtree roots `have_`, and its pads past its count. -/
def treeAt (d : Domain) (have_ : Std.HashMap (Nat × Nat) ByteArray) : Nat → Nat → Option ByteArray
  | 0, index => match have_.get? (0, index) with
    | some h => some h
    | none => if index ≥ d.count then some (d.pad index) else none
  | l + 1, index => match have_.get? (l + 1, index) with
    | some h => some h
    | none => do
      let left ← treeAt d have_ l (2 * index)
      let right ← treeAt d have_ l (2 * index + 1)
      pure (d.node l index left right)

/-- `credit.tree_root`. -/
def treeRoot (domain : ByteArray) (count : Nat) (have_ : Std.HashMap (Nat × Nat) ByteArray) : Option ByteArray :=
  treeAt ⟨.sha512, count, domain⟩ have_ (PouwCommit.depth count) 0

def aLinkRefuse {α : Type} (why : String) : R α := throw ("a-link", why)

/-- `credit._a_link`: none when every registered A value is the tree over the segments the round log commits, else the
refusal. Of S rounds, round s's segments of a window's A value `…/a/<K>` of L rows tile its rows `[s·L/S, (s+1)·L/S)`;
round s's A table of K `…/a/<K>@<s>` is committed whole by round s, whose segments tile all its rows, and no other
round lists it. Round s's input-ids table `…/ids@<s>` (`idsStep?`) is committed whole by round s alike, and so is
call c's receipt table `…/receipt@<c>` (`receiptCall?`) by c's round, of the window's `calls` calls (its manifest's
rows; 0 admits no receipt table) the rounds' equal blocks. A step of `window`, which first holds the round roots to the
ones the challenger received (`roundsCheck`). -/
def Internal.aLink (rounds entries : Json) (calls : Nat) : R Unit := do
  let some xs := (match rounds with
      | .arr a => if a.isEmpty then none else some a.toList
      | _ => none) | aLinkRefuse LOG_FORM
  let mut segs : List Segments := []
  for (x, r) in xs.zipIdx do
    let (some root, some inner) := (if isObj x then (digest? (get x "root"), digest? (get x "inner"))
        else (none, none)) | aLinkRefuse s!"round {r}'s root or inner root is not a SHA-512 digest"
    let some p := segments? (get x "a")
      | aLinkRefuse (s!"round {r}'s A segments are not an object from each A value to its [level, index, root] " ++
          s!"subtrees, each level below {LEVELS}")
    unless roundRoot r inner p == root do
      aLinkRefuse (s!"round {r}'s root is not over its inner root and its A segments: the serving process commits " ++
        "each round's segments of every registered A value under that round's root (`credit.round_root`), before " ++
        "that round's coin")
    segs := segs ++ [p]
  let .arr es := entries | aLinkRefuse ENTRIES_FORM
  unless es.all isObj do aLinkRefuse ENTRIES_FORM
  let a := es.toList.filterMap fun e => if isTied (get e "value") then (·, e) <$> str? (get e "value") else none
  unless a.any (isA <| .str ·.1) do aLinkRefuse NO_A
  let mut names : List String := []
  for (v, _) in a do
    if names.contains v then aLinkRefuse s!"the window registers {what v} {v} twice"
    names := names ++ [v]
  for (p, r) in segs.zipIdx do
    for (v, _) in p do
      unless names.contains v do
        aLinkRefuse (s!"round {r} lists segments of {v}, which is no registered A value, input-ids table or receipt " ++
          "table of the window")
  let steps := xs.length
  for (v, e) in a do
    let (some root, some domain, some count) := (digest? (get e "root"), digest? (get e "domain"), nat? (get e "leaves"))
      | aLinkRefuse (s!"registered {what v} {v}'s entry has no root, domain or leaf count from 1 to {MAX_ROWS} " ++
          "(the verifier hashes its pads)")
    unless 1 ≤ count && count ≤ MAX_ROWS do
      aLinkRefuse (s!"registered {what v} {v}'s entry has no root, domain or leaf count from 1 to {MAX_ROWS} " ++
        "(the verifier hashes its pads)")
    let (per, own) ← (match receiptCall? (.str v), idsStep? (.str v), (aValue? (.str v)).bind (·.2) with
      | some c, _, _ => do
        unless calls % steps == 0 && c < calls do
          aLinkRefuse (s!"registered receipt table {v} is call {c}'s, and the window's {calls} calls are not its " ++
            s!"{steps} rounds' equal blocks with a call {c}")
        let o := c / (calls / steps)
        pure (count, some (s!"call {c}'s receipt table, its round {o}'s", o))
      | none, some s, _ => do
        if s ≥ steps then
          aLinkRefuse s!"registered input-ids table {v} is round {s}'s, and the window has {steps} rounds"
        pure (count, some (s!"round {s}'s input-ids table", s))
      | none, none, none => do
        if count % steps != 0 then
          aLinkRefuse s!"registered A value {v}'s {count} rows are not the window's {steps} rounds' equal blocks"
        pure (count / steps, none)
      | none, none, some s => do
        let s := s.toNat!
        if s ≥ steps then
          aLinkRefuse s!"registered A value {v} is round {s}'s A table, and the window has {steps} rounds"
        pure (count, some (s!"round {s}'s A table", s)))
    let mut have_ : Std.HashMap (Nat × Nat) ByteArray := {}
    for (p, r) in segs.zipIdx do
      if let some (whose, o) := own then
        if r != o then
          unless ((p.lookup v).getD []).isEmpty do
            aLinkRefuse s!"round {r} lists segments of {v}, {whose}, which only round {o} commits"
          continue
      let (start, stop) := if own.isSome then (0, count) else (r * per, (r + 1) * per)
      let untiled := s!"round {r}'s segments of {v} don't tile its rows [{start}, {stop}) in order, each an aligned " ++
        "subtree: the serving process lists each round's rows of every A value as aligned subtrees (`credit.cover`)"
      let mut pos := start
      for (level, index, h) in (p.lookup v).getD [] do
        if index <<< level != pos || pos + 2 ^ level > stop then aLinkRefuse untiled
        have_ := have_.insert (level, index) h
        pos := pos + 2 ^ level
      if pos != stop then aLinkRefuse untiled
    unless treeRoot domain count have_ == some root do
      aLinkRefuse (s!"registered {what v} {v}'s root is not the tree over the segments the round roots commit: its " ++
        s!"rows are not the ones fixed under each round's root before that round's coin. Register the {what v} " ++
        "whose rows each round's root committed")

/-- `credit._entry`: a registered value's entry as one-stage's registration names it (`registered.ENTRY_KEYS`, with
`word_bits` only when it is not 16): its value printable ASCII, its root a SHA-512 digest, its rows and words from 1
to `MAX_ROWS`. -/
def entryOk (e : Json) : Bool :=
  let rows := fun (k : String) => (nat? (get e k)).any fun n => 1 ≤ n && n ≤ MAX_ROWS
  let shape := match keys e with
    | some ["domain", "leaves", "root", "value", "words"] => true
    | some ["domain", "leaves", "root", "value", "word_bits", "words"] => nat? (get e "word_bits") == some 32
    | _ => false
  shape && ((str? (get e "value")).any _root_.PouwCheck.plain) && (digest? (get e "root")).isSome
    && (str? (get e "domain")).isSome && rows "leaves" && rows "words"

/-- `credit.domain_of`: the domain (hex) one-stage's registration gives the registered value `e` (an entry `entryOk`
holds) of the window whose open root is `window`, under the circuit whose descriptor's SHA-512 is `program`
(`SampledProofs.Registered.domainAt`): the weights' (`…/b/<K>`, `isHeld`) are held across windows and bind none;
round 0's input-ids table (`…/ids@0`, `idsStep?` 0), whose root the window's open root contains (`requestsRoot`,
`Internal.arow`), binds the window's session `session` in that root's place (`domainBound` at `"session"`); every
other value's rows are the window's own, its A rows and its calls' A and z tables among them, and its domain binds the
window's open root (PRF-127). -/
def domainOf (program window session : String) (e : Json) : R String := do
  let v := (str? (get e "value")).getD ""
  let (words, leaves, bits) := ((nat? (get e "words")).getD 0, (nat? (get e "leaves")).getD 0,
    (nat? (get e "word_bits")).getD 16)
  let d ← (if idsStep? (.str v) == some 0 then
      SampledProofs.Registered.domainBound program v words leaves bits "session" session
    else SampledProofs.Registered.domainAt program v words leaves bits
      (if isHeld (.str v) then none else some window)).mapError fun why => ("registration", why)
  pure (Flock.hexOf d)

def otherRegistration (v : String) : String :=
  s!"registered value {v}'s domain is not the one the window's registration gives it (the registered domain of " ++
  s!"{v} under the circuit the window's proofs were verified against, the SHA-512 of its descriptor, and, for a " ++
  "value other than the weights' …/b/<K> and round 0's …/ids@0, the window's open root, the root the " ++
  "auditor's ledger opened its session for, for …/ids@0 that session): the entry is another registration's, " ++
  "another window's or another circuit's"

/-- How deep `leavesOf` reads a type: deeper than Python's recursion limit reads. -/
def TYPE_DEPTH : Nat := 1000

/-- `credit._leaves`: a descriptor type's leaves, `["Value", w]` one, `["Array", n, t]` n of t's, a `["Tuple", [t, ...]]`
or `["Record", [[name, t], ...]]` its fields'; zero for any other form, or past `fuel` levels. -/
def leavesOf : Nat → Json → Nat
  | 0, _ => 0
  | fuel + 1, .arr a => match a.toList with
    | .str "Value" :: _ => 1
    | [.str "Array", n, t] => match nat? n with
      | some n => n * leavesOf fuel t
      | none => 0
    | [.str "Tuple", .arr fs] => fs.foldl (fun s x => s + leavesOf fuel x) 0
    | [.str "Record", .arr fs] => fs.foldl (fun s x => match x with
      | .arr p => (match p.toList with
        | [_, t] => s + leavesOf fuel t
        | _ => s)
      | _ => s) 0
    | _ => 0
  | _ + 1, _ => 0

/-- K's template, as a statement over its units reads it: its id and the 16-bit words of its A row, its B row, its
z/y record and its receipt (its return's `receipt` field, the statement's output row o0; 0 when it returns none of
whole 16-bit words). -/
structure Tile where
  fn : String
  a : Nat
  b : Nat
  zy : Nat
  receipt : Nat := 0

/-- `credit.RECEIPT_PORT`: the statement's output row of its units' receipts, which reads its call's receipt table. -/
def RECEIPT_PORT : String := "o0"
/-- `credit.PUB_PORT`: the statement's public row, which reads its call's public table, the verifier's own. -/
def PUB_PORT : String := "pub"

/-- `credit.OK_PORT`: the statement's output row of its unit's ok, which reads the window's ok rows, the verifier's own. -/
def OK_PORT : String := "o1"
/-- `credit.OK_VALUE`: the window Program's output the ok rows are, after its prefix: row u unit u's. -/
def OK_VALUE : String := "ok"
/-- `core_check.OK_ROW`: the ok row the verifier accepts, a unit's one output bit, 1, in its one word (h1_prove's, #3124). -/
def OK_ROW : Array Nat := #[1]
/-- `core_check.OK_SALT_TAG`: the ok row's hm96 salt tag (h1_prove's, #3124). -/
def OK_SALT_TAG : ByteArray := "verity/pouw/h1-prove/ok-row-salt/v0".toUTF8.push 0

/-- `core_check.ok_salt`: SHAKE-256 of `OK_SALT_TAG` ‖ context ‖ 0 ‖ u64be(position), `PouwCheck.saltBytes` bytes. -/
def okSalt (context : String) (position : Nat) : ByteArray :=
  Flock.Keccak.shake256 ((OK_SALT_TAG ++ context.toUTF8).push 0 ++ PouwCommit.be8 position) PouwCheck.saltBytes

/-- `core_check.public_words`: `PouwCheck.publicRow` as a `row/v2` row of 16-bit leaves, word w its bytes 2w (low) and
2w + 1. -/
def publicWords (salt : ByteArray) (position : Nat) (roots : ByteArray × ByteArray) : Array Nat :=
  let x := PouwCheck.publicRow salt position roots
  (Array.range (x.size / 2)).map fun i => (x.get! (2 * i)).toNat + 256 * (x.get! (2 * i + 1)).toNat

/-- `credit._bits`: a descriptor type's bits, its leaves' widths: `["Value", w]` w, and `leavesOf`'s rule for the rest;
zero for any other form, or past `fuel` levels. -/
def bitsOf : Nat → Json → Nat
  | 0, _ => 0
  | fuel + 1, .arr a => match a.toList with
    | [.str "Value", w] => (nat? w).getD 0
    | [.str "Array", n, t] => match nat? n with
      | some n => n * bitsOf fuel t
      | none => 0
    | [.str "Tuple", .arr fs] => fs.foldl (fun s x => s + bitsOf fuel x) 0
    | [.str "Record", .arr fs] => fs.foldl (fun s x => match x with
      | .arr p => (match p.toList with
        | [_, t] => s + bitsOf fuel t
        | _ => s)
      | _ => s) 0
    | _ => 0
  | _ + 1, _ => 0

/-- `credit._receipt_words`: the 16-bit words of a template return's `receipt` field (`["Record", [["receipt", t],
...]]`, its first such field), its bits as the statement's output row o0 holds them (`bitsOf`); 0 for another form or
bits of no whole words. -/
def receiptWords (ret : Json) : Nat :=
  let fields : List Json := match ret with
    | .arr #[.str "Record", .arr fs] => fs.toList
    | _ => []
  let t := fields.findSome? fun x => match x with
    | .arr #[.str "receipt", t] => some t
    | _ => none
  let n := match t with
    | some t => bitsOf TYPE_DEPTH t
    | none => 0
  if n % 16 == 0 then n / 16 else 0

/-- `credit.shape_of`: of a descriptor `layoutOf` accepted, its parameters' prefix (`PREFIX`) and each K's `Tile`
(`TILES`, by ascending K), the widths `h1window.table_reads_file` reads. Its cost is the descriptor's bytes. -/
structure Shape where
  pre : String
  tiles : List (Nat × Tile)

def shapeOf (descriptor : ByteArray) : Shape := Id.run do
  let some d := (String.fromUTF8? descriptor).bind loads? | return ⟨"", []⟩
  let .str fn := get d "program_fn" | return ⟨"", []⟩
  let vals := (Flock.Program.recordPairs fn).toList.filterMap fun (k, s) => (loads? s).map (k, ·)
  let pre := ((vals.lookup "PREFIX").bind str?).getD ""
  let elem := fun (t : Json) => (match t with
    | .arr q => q.getD 2 .null
    | _ => .null)
  let tiles : List (Nat × Tile) := match vals.lookup "TILES" with
    | some (.arr ts) => ts.toList.filterMap fun (t : Json) => (match t with
      | .arr p => (match p.toList with
        | [k, f] => do
          let k ← nat? k
          let fid ← str? (get f "fn")
          let body := get (get (get d "definitions") fid) "body"
          let .arr ps := get body "params" | none
          let ty := fun (i : Nat) => (match ps[i]? with
            | some (.arr q) => q.getD 1 .null
            | _ => .null)
          pure (k, { fn := fid, a := leavesOf TYPE_DEPTH (elem (ty 0)) / 16, b := leavesOf TYPE_DEPTH (elem (ty 1)) / 16,
                     zy := leavesOf TYPE_DEPTH (ty 2) / 16, receipt := receiptWords (get body "ret_type") })
        | _ => none)
      | _ => none)
    | _ => []
  return ⟨pre, tiles⟩

def PARTITION_FORMAT : String := "verity/partition/v1"
def QUERY_NAME : String := "Q_template_instances"
def QUERY_VERSION : Nat := 0

/-- `credit.partition_of`: the partition object of the window whose circuit's descriptor has SHA-512 `program` and whose
`shapeOf` is `shape`: every instance of each K's template one unit, its templates by ascending K. -/
def partitionOf (program : String) (shape : Shape) : Json :=
  Json.mkObj [("format", Json.str PARTITION_FORMAT), ("program", Json.str program),
    ("query", Json.mkObj [("name", Json.str QUERY_NAME), ("version", Lean.toJson QUERY_VERSION),
      ("params", Json.mkObj [("templates",
        Json.arr ((shape.tiles.mergeSort (·.1 ≤ ·.1)).map (Json.str ·.2.fn)).toArray)])])]

/-- `registered.READS`: a `--registered` file's format. -/
def READS : String := "verity/registered-reads/v0"
def SCHEMA_KEY : String := "row_schema"

/-- `credit.step_tables`: each round's A tables of a layout `strata` accepted, over its `steps` rounds (its calls the
rounds' equal blocks, in order): `(s, K, calls)` for each K round s's calls have, in order of its first call there,
`calls` the round's calls of K in call order. Round s's A table of K `<prefix>a/<K>@<s>` holds the M rows of each of
their A blocks, call after call. Empty when `steps` doesn't divide the calls. -/
def stepTables (layout : Json) (steps : Nat) : List (Nat × Nat × List Nat) := Id.run do
  let calls : List String := match get layout "calls" with
    | .arr xs => xs.toList.filterMap fun x => match x with
      | .arr #[.str lin, _] => some lin
      | _ => none
    | _ => []
  let ks : List (String × Nat) := match get layout "linears" with
    | .arr xs => xs.toList.filterMap fun x => match x with
      | .arr #[.str name, k, _] => (name, ·) <$> nat? k
      | _ => none
    | _ => []
  if steps == 0 || calls.length % steps != 0 then return []
  let per := calls.length / steps
  let mut out : List (Nat × Nat × List Nat) := []
  for s in List.range steps do
    let mut mine : List (Nat × List Nat) := []
    for c in List.range' (s * per) per do
      let k := (ks.lookup (calls.getD c "")).getD 0
      mine := if (mine.lookup k).isSome then mine.map fun (k', cs) => (k', if k' == k then cs ++ [c] else cs)
        else mine ++ [(k, [c])]
    out := out ++ mine.map fun (k, cs) => (s, k, cs)
  return out

/-- `credit.step_row`: call `call`'s round of the layout's `steps`, its round's calls of its K, and its first row in
its round's A table of K (`stepTables`): `(s, n, M·j)`. -/
def stepRow (layout : Json) (steps call : Nat) : Nat × Nat × Nat :=
  match (stepTables layout steps).find? (·.2.2.contains call) with
  | some (s, _, cs) => (s, cs.length, ((nat? (get layout "m")).getD 0) * cs.idxOf call)
  | none => (0, 0, 0)

/-- `credit.registered_file`: the verifier's `--registered` file of call `call`'s statement in the window `layout`
(one `strata` accepted, its circuit's descriptor of SHA-512 `program` and `shapeOf` `shape`, its calls `steps` rounds'
equal blocks), else the refusal (`registration`: the window registers no root, or two, for a value the statement
reads). It is `h1window.table_reads_file`'s, from the descriptor alone: unit u, tile t = ti·(n/C) + tj of the call,
reads its A row i (port `a_k<K>_<i>`) at row o + R·ti + i of its round s's A table of K `<prefix>a/<K>@<s>` (M rows for
each of the round's calls of K, o the call's first, `stepRow`), its B row j (`b_k<K>_<j>`) at
row b0 + C·tj + j of the weights' `<prefix>b/<K>` (b0 the call's linear's first row in it), and its record (`zy`) at row
t of the call's z table `<prefix>zy@<call>`; and, when the window registers the call's receipt table
`<prefix>receipt@<call>`, the statement's output row o0 (`RECEIPT_PORT`, the unit's receipt, `shapeOf`'s width) at its
row t, so the receipts the round roots commit are the ones the units return; every other unit reads none of them, and
the file lists no unit past the call's last (`Flock.Registered.badInstance` reads none past `reads`). Each value is a
row/v2 value of 16-bit leaves under `SampledProofs.Registered.domainAt` of its name, rows and words, binding the
window's open root `window` but for the weights' (`isHeld`), at the root `roots` (`{value: [root, ...]}`) gives it. Its
cost is the window's units up to the call's last, once a port: the reads file's form. The statement's public row `pub`
(`PUB_PORT`) is the verifier's own: unit u0 + t reads row t of the call's public table `<prefix>pub@<call>`, which the
file names at its known content (`known`, `Flock.Registered.checkKnownAll`), the window's Program salt `salt` (hex), the
position u0 + t and the call's roots, root(`a/<K>@<s>`) and root(`b/<K>`) as the window registers them (`publicWords`),
each under `publicRowSalt` of `window` and its position, at the root of their registration; a root of the two that is
no SHA-512 digest is refused. The statement's
output row `ok` (`OK_PORT`, o1) is the verifier's own too: unit u reads row u of the window's ok rows `<prefix>ok`
(`OK_VALUE`, one row a unit of the window), named at its known content, the one word 1 (`OK_ROW`) under `okSalt` of
`window` and u; a window whose entries register those rows themselves is refused. -/
def registeredFile (program : String) (layout : Json) (shape : Shape) (call : Nat) (roots : Json) (window : String)
    (steps : Nat) (salt : String) : R Json := do
  let ov := s!"{shape.pre}{OK_VALUE}"
  if (PouwWindow.field? roots ov).isSome then
    throw ("registration", s!"the window registers {ov}, its units' ok rows, which are the verifier's own (the one " ++
      "word 1 under each unit's public ok salt): no entry registers them")
  let num := fun j => (nat? j).getD 0
  let (r, c) := (match get layout "tile" with
    | .arr a => (num (a.getD 0 .null), num (a.getD 1 .null))
    | _ => (1, 1))
  let m := num (get layout "m")
  let rows := fun (j : Json) => (match j with
    | .arr a => a.toList.filterMap fun (x : Json) => (match x with
      | .arr b => some b.toList
      | _ => none)
    | _ => ([] : List (List Json)))
  let lins : List (String × Nat × Nat) := (rows (get layout "linears")).filterMap fun (x : List Json) => (match x with
    | [.str s, k, n] => some (s, num k, num n)
    | _ => none)
  let calls : List (List Json) := rows (get layout "calls")
  let lin : String := (match calls[call]? with
    | some (Json.str s :: _) => s
    | _ => "")
  let kn : Nat × Nat := ((lins.find? (·.1 == lin)).map (·.2)).getD (0, 0)
  let (k, n) := kn
  let tile : Tile := (shape.tiles.lookup k).getD { fn := "", a := 0, b := 0, zy := 0 }
  let b0 : Nat := ((lins.takeWhile (·.1 != lin)).filter (·.2.1 == k)).foldl (fun s x => s + x.2.2) 0
  let sizes := (callWork layout).map (·.1)
  let u0 := (sizes.take call).foldl (· + ·) 0
  let size := sizes.getD call 0
  let per := n / c
  let (s, callsK, at_) := stepRow layout steps call
  let a := s!"{shape.pre}a/{k}@{s}"
  let zy := s!"{shape.pre}zy@{call}"
  let bv := s!"{shape.pre}b/{k}"
  let rc := s!"{shape.pre}receipt@{call}"
  let receipted := (PouwWindow.field? roots rc).isSome
  if receipted && tile.receipt == 0 then
    throw ("registration", s!"the window registers {rc}, and K = {k}'s template returns no receipt of whole 16-bit " ++
      "words for its statement's output row o0 to read")
  let shapes : List (String × Nat × Nat) :=
    [(a, m * callsK, tile.a), (zy, size, tile.zy), (bv, (lins.filter (·.2.1 == k)).foldl (fun s x => s + x.2.2) 0, tile.b)]
    ++ (if receipted then [(rc, size, tile.receipt)] else [])
  let mut root : List (String × String) := []
  for (v, _) in shapes do
    let got : List Json := (match get roots v with
      | .arr xs => xs.toList
      | _ => [])
    match got with
    | [.str x] => root := root ++ [(v, x)]
    | _ => throw ("registration", s!"call {call}'s statement reads {v}, which the window registers " ++
        s!"{if got.isEmpty then "not" else "twice"}: register each of a call's tables and its weights once")
  let (some ra, some rb) := (digest? (.str ((root.lookup a).getD "")), digest? (.str ((root.lookup bv).getD "")))
    | throw ("registration", s!"call {call}'s seed reads the roots of {a} and {bv}, which are not both SHA-512 " ++
        "digests (128 lowercase hex): register each table at its tree's root")
  let ports : List (String × String × (Nat → Nat)) :=
    (List.range r).map (fun i => (s!"a_k{k}_{i}", a, fun t => at_ + r * (t / per) + i)) ++
    (List.range c).map (fun j => (s!"b_k{k}_{j}", bv, fun t => b0 + c * (t % per) + j)) ++
    [("zy", zy, fun t => t)] ++ (if receipted then [(RECEIPT_PORT, rc, fun t => t)] else [])
  let mut out : List (String × Json) := []
  for (port, v, row) in ports do
    let reads := (List.range (u0 + size)).map fun u =>
      if u0 ≤ u && u < u0 + size then Lean.toJson (row (u - u0)) else Json.null
    let lw : Nat × Nat := ((shapes.find? (·.1 == v)).map (·.2)).getD (0, 0)
    let (leaves, words) := lw
    let d ← (SampledProofs.Registered.domainAt program v words leaves 16
      (if isHeld (.str v) then none else some window)).mapError fun why => ("registration", why)
    out := out ++ [(port, Json.mkObj [("value", Json.str v), ("root", Json.str ((root.lookup v).getD "")),
      ("leaves", Lean.toJson leaves), ("words", Lean.toJson words), ("word_bits", Lean.toJson (16 : Nat)),
      ("domain", Json.str (Flock.hexOf d)), (SCHEMA_KEY, Json.str Flock.HmRow.ROW_SCHEMA_V2),
      ("reads", Json.arr reads.toArray)])]
  let pv := s!"{shape.pre}pub@{call}"
  let words := PouwCheck.PUBLIC_ROW_BITS / 16
  let sb := (hex? (.str salt)).getD ByteArray.empty
  let ks : Array (Array Nat × ByteArray) := (Array.range size).map fun t =>
    (publicWords sb (u0 + t) (ra, rb), PouwCheck.publicRowSalt window (u0 + t))
  let d ← (SampledProofs.Registered.domainAt program pv words size 16 (some window)).mapError fun why =>
    ("registration", why)
  let reg : Flock.Registered.Port :=
    { name := PUB_PORT, value := pv, root := ByteArray.empty, leaves := size, words := some words, wordBits := 16,
      domain := d, readsRel := #[], schema := Flock.HmRow.ROW_SCHEMA_V2 }
  let pr := Flock.FV3.root d (Flock.Registered.knownLeaves reg ks) Flock.Sha512.hash
  out := out ++ [(PUB_PORT, Json.mkObj [("value", Json.str pv), ("root", Json.str (Flock.hexOf pr)),
    ("leaves", Lean.toJson size), ("words", Lean.toJson words), ("word_bits", Lean.toJson (16 : Nat)),
    ("domain", Json.str (Flock.hexOf d)), (SCHEMA_KEY, Json.str Flock.HmRow.ROW_SCHEMA_V2),
    ("reads", Json.arr ((List.range u0).map (fun _ => Json.null) ++ (List.range size).map Lean.toJson).toArray),
    ("known", Json.mkObj [("rows", Lean.toJson (ks.map (·.1))),
      ("salts", Json.arr (ks.map fun k => Json.str (Flock.hexOf k.2)))])])]
  let population := sizes.foldl (· + ·) 0
  let oks : Array (Array Nat × ByteArray) := (Array.range population).map fun u => (OK_ROW, okSalt window u)
  let od ← (SampledProofs.Registered.domainAt program ov OK_ROW.size population 16 (some window)).mapError fun why =>
    ("registration", why)
  let oreg : Flock.Registered.Port :=
    { name := OK_PORT, value := ov, root := ByteArray.empty, leaves := population, words := some OK_ROW.size,
      wordBits := 16, domain := od, readsRel := #[], schema := Flock.HmRow.ROW_SCHEMA_V2 }
  let opr := Flock.FV3.root od (Flock.Registered.knownLeaves oreg oks) Flock.Sha512.hash
  out := out ++ [(OK_PORT, Json.mkObj [("value", Json.str ov), ("root", Json.str (Flock.hexOf opr)),
    ("leaves", Lean.toJson population), ("words", Lean.toJson OK_ROW.size), ("word_bits", Lean.toJson (16 : Nat)),
    ("domain", Json.str (Flock.hexOf od)), (SCHEMA_KEY, Json.str Flock.HmRow.ROW_SCHEMA_V2),
    ("reads", Json.arr ((List.range u0).map (fun _ => Json.null) ++
      (List.range size).map (fun t => Lean.toJson (u0 + t))).toArray),
    ("known", Json.mkObj [("rows", Lean.toJson (oks.map (·.1))),
      ("salts", Json.arr (oks.map fun k => Json.str (Flock.hexOf k.2)))])])]
  pure (Json.mkObj [("format", Json.str READS), ("ports", Json.mkObj out)])

def registrationRefuse {α : Type} (why : String) : R α := throw ("registration", why)

/-- `_registration`'s `held`: the refusal of the registered value `v`'s entry `e`, none when it is one-stage's form and
its domain is `domainOf`'s. -/
def held (program window session v : String) (e : Json) : R (Option String) := do
  unless entryOk e do
    return some (s!"registered value {v}'s entry is not one-stage's \{value, root, leaves, words, domain} (with " ++
      s!"word_bits only when it is not 16), its rows and words from 1 to {MAX_ROWS}")
  let d ← domainOf program window session e
  pure (if str? (get e "domain") == some d then none else some (otherRegistration v))

/-- `_registration`'s call pointer: from call `c`, the last call whose first unit (`starts`, each call's in order) is at
most `u`. Over units in ascending order its steps total the calls, never the units. -/
def callAt (starts : Array Nat) (u c : Nat) : Nat := go starts.size c
where
  go : Nat → Nat → Nat
    | 0, c => c
    | n + 1, c => if c + 1 < starts.size && starts.getD (c + 1) 0 ≤ u then go n (c + 1) else c

/-- `credit._registration`: none when the window's registered `entries` are its own registration's and each drawn unit's
proof (`proofs`, one accepted outcome a unit, as `proved` holds them) is a session verified against it, else the
refusal (`registration`). Each A value's and input-ids table's domain, and each other value's a proof opened, is
`domainOf` the window's open root `window`, its session `session` and the descriptor's SHA-512 `program`; each proof
was verified against
`program`; every registered port its session opened (its `registered` finding) is of a value the window registers once,
at that value's registered root, but for its call's public table (`pubCall?`) and the window's ok rows
(`<prefix>ok`, `OK_VALUE`), which no entry registers and `credit._verified` holds to the rows `registeredFile` built; and it opened its unit's K's A value: the window's `…/a/<K>` (K the unit's stratum's
in `ss`) or its call's round's A table of K `…/a/<K>@<s>` (the call `layout`'s units place it in, `callAt`, s its round
of `steps`, its calls the rounds' equal blocks), and its K's B
rows, the window's `<prefix>b/<K>` (`prefix` the descriptor's `PREFIX`). A step of `window`, after `Internal.aLink` has
held each A value's root to the segments the round roots commit and `periodWeights` each `<prefix>b/<K>` to the
weights' root its period signed. -/
def Internal.registration (window program prefix_ : String) (entries proofs : Json) (ss : Array Flock.Draw.WorkStratum)
    (layout : Json) (steps : Nat) (session : String) : R Unit := do
  let es := match entries with
    | .arr a => a.toList
    | _ => []
  for e in es do
    if isTied (get e "value") then
      if let some why ← held program window session ((str? (get e "value")).getD "") e then registrationRefuse why
  let names := es.map fun e => str? (get e "value")
  let ps := (match proofs with
    | .arr a => a.toList
    | _ => []).filterMap fun p => (fun u f => (u, get p "program", f)) <$> nat? (get p "unit") <*> finding? (get p "registered")
  let mut checked : List String := []
  let starts := ((callWork layout).foldl (fun (xs, at_) (size, _) => (xs.push at_, at_ + size)) (#[], 0)).1
  let mut c := 0
  let per := max 1 ((match get layout "calls" with
    | .arr a => a.size
    | _ => 0) / steps)
  for (u, prog, f) in ps.mergeSort fun a b => a.1 ≤ b.1 do
    let k := ((ss.find? fun s => s.units.any fun (lo, hi) => lo ≤ u && u < hi).map
      fun s => String.ofList (s.template.toList.drop 1)).getD ""
    c := callAt starts u c
    unless str? prog == some program do
      registrationRefuse (s!"drawn unit {u}'s proof was verified against another circuit than the window's (its " ++
        "program is not the SHA-512 of the descriptor whose layout the window was drawn over): verify every drawn " ++
        "unit's session against the window's committed circuit")
    let mut opened := false
    let mut weights := false
    for (port, v, root) in f do
      if (pubCall? (.str v)).isSome || v == prefix_ ++ OK_VALUE then continue
      let n := names.count (some v)
      if n != 1 then
        registrationRefuse (s!"drawn unit {u}'s proof opened port {port} against {v}, which the window registers " ++
          s!"{if n > 1 then "twice" else "not"}: its session was verified against another registration than the " ++
          "window's")
      let e := (es.find? fun e => str? (get e "value") == some v).getD .null
      if !checked.contains v && !isTied (.str v) then
        if let some why ← held program window session v e then registrationRefuse why
      checked := v :: checked
      unless str? (get e "root") == some root do
        registrationRefuse (s!"drawn unit {u}'s proof opened port {port}'s rows of {v} against another root than the " ++
          s!"one the window registered for {v}: its session was verified against another registration, or the " ++
          "window's entry is not the root its sessions opened")
      opened := opened || (match aValue? (.str v) with
        | some (k', s') => k' == k && (s'.all (· == toString (c / per)))
        | none => false)
      weights := weights || v == s!"{prefix_}b/{k}"
    unless opened do
      registrationRefuse (s!"drawn unit {u}'s proof opened no A value …/a/{k} or …/a/{k}@{c / per} (its call's " ++
        "round's) of the window, so nothing ties the A rows it proved to the ones the round roots committed: verify each drawn unit's " ++
        "session with the window's A value registered")
    unless weights do
      registrationRefuse (s!"drawn unit {u}'s proof opened no weights {prefix_}b/{k} of the window, so nothing ties " ++
        "the B rows it proved to the ones its period registered: verify each drawn unit's session with the window's " ++
        "weights registered")

def arowRefuse {α : Type} (why : String) : R α := throw ("arow", why)

/-- `credit.REQUESTS_IDS_TAG`: round 0's request root (`requests`, the one the window's open root binds), its request
table's root with round 0's input-ids table's root inside (`requestsRoot`). -/
def REQUESTS_IDS_TAG : String := "verity/pouw/h1/requests/ids/v1"
def IDS_OUTSIDE : String :=
  "round 0's request root (requests, the one the window's open root binds) is not requests_root of its request " ++
  "table's root (request_table) and its input-ids table's root, so round 0's input ids are not fixed before the " ++
  "salt: commit round 0's input-ids table inside round 0's request table, under the window's open root"

/-- `credit.requests_root`: `tagged` of `REQUESTS_IDS_TAG`, round 0's request table's root `table` and its input-ids
table's root `ids`. So the window's open root contains ids@0's root, and its rows are fixed before the salt coin. -/
def requestsRoot (table ids : ByteArray) : ByteArray := tagged REQUESTS_IDS_TAG [.bytes table, .bytes ids]

/-- `credit._same`: `c` has the root, domain and leaf count of `e`, an entry `Internal.aLink` accepted. -/
def same (e c : Json) : Bool :=
  get c "root" == get e "root" && get c "domain" == get e "domain" && (nat? (get c "leaves")).isSome
    && nat? (get c "leaves") == nat? (get e "leaves")

/-- `credit._arow`: none when each round's A table of each K and each round's input-ids table among the window's
registered `entries` is the Commit's value of the same rows among its record's `registered` (`commitOf?`'s): same root,
domain and leaf count; else the refusal (`arow`). The window's `layout` (`strata`'s) calls are its round log `rounds`'
S rounds' equal blocks, and round s's `<prefix>a/<K>@<s>` (`prefix_` the descriptor's `PREFIX`, `stepTables`) holds M
rows for each of the round's calls of K, call after call, and is the Commit's `commitRows` of s and K; round s's
`<prefix>ids@<s>` is the Commit's `commitIds` of s. The Commit registers each of those values once. So the A rows a
call's units read are its linear's input as the served Commit commits it, and the input token ids under each round's
root are its step's (ROOTS1). The window registers no other input-ids table, and round 0's request root (`requests`,
the one its open root binds) is `requestsRoot` of its request table's root (`request_table`) and `<prefix>ids@0`'s, so
round 0's input ids are fixed before the salt coin. A step of `window`, after `Internal.aLink` has held each to the
segments the round roots commit, `periodWeights` round 0's request root to the open root, and the manifest to the
layout's calls. Its cost is the calls and the records' entries. -/
def Internal.arow (rounds : Array Json) (entries : Json) (registered : Array Json) (layout : Json) (prefix_ : String) :
    R Unit := do
  let steps := rounds.size
  let calls : Nat := match get layout "calls" with
    | .arr xs => xs.size
    | _ => 0
  if calls % steps != 0 then
    arowRefuse (s!"the window's {calls} calls are not its {steps} rounds' equal blocks, so no call's round, " ++
      "and no Commit value of its rows, is fixed")
  let ours : Std.HashMap String Json := match entries with
    | .arr es => es.foldl (fun m e => if isTied (get e "value") then (str? (get e "value")).elim m (m.insert · e)
        else m) {}
    | _ => {}
  let theirs : Std.HashMap String (Array Json) := registered.foldl (fun m e =>
    (str? (get e "value")).elim m fun v => m.insert v ((m.getD v #[]).push e)) {}
  let m := (nat? (get layout "m")).getD 0
  let pairs := (stepTables layout steps).map (fun (s, k, cs) =>
      (s!"{prefix_}a/{k}@{s}", commitRows s k, s!"round {s}'s A table of K = {k}", some (m * cs.length))) ++
    (List.range steps).map fun s => (s!"{prefix_}ids@{s}", commitIds s, s!"round {s}'s input-ids table", none)
  for (v, name, whose, rows) in pairs do
    let some e := ours.get? v
      | arowRefuse (s!"the window registers no {whose} {v}, so nothing ties it to the Commit's {name}: the served " ++
          "window registers each round's A table of each K and its input-ids table")
    if let some n := rows then
      unless nat? (get e "leaves") == some n do
        arowRefuse (s!"{whose} {v} has {(nat? (get e "leaves")).getD 0} rows, not {n}: it holds the M rows of each " ++
          "of the round's calls of K, call after call, so a call missing from it reads another call's rows (ROOTS1)")
    let got := theirs.getD name #[]
    if got.size != 1 then
      arowRefuse (s!"the Commit record registers {name} {if got.isEmpty then "not" else "twice"}: the served " ++
        "Commit registers each round's A table of each K as row-table/<round>/k<K> and each round's input-ids " ++
        "table as input-ids/<round>, once")
    unless same e got[0]! do
      arowRefuse (s!"{whose} {v} is not the Commit's {name}: their root, domain or leaf count differ, so its rows " ++
        "are not the ones the served Commit commits (ROOTS1). The Commit registers the window's own table as its " ++
        "value, with no tree of its own")
  let want := pairs.map (·.1)
  let es := match entries with
    | .arr a => a.toList
    | _ => []
  for e in es do
    if let some v := str? (get e "value") then
      if (idsStep? (.str v)).isSome && !want.contains v then
        arowRefuse (s!"the window registers input-ids table {v}, which is not {prefix_}ids@<s> of any of its " ++
          s!"{steps} rounds: it registers each round's input-ids table once, under its descriptor's PREFIX")
  let r0 := rounds[0]?.getD .null
  let (some table, some ids) := (digest? (get r0 "request_table"),
      (ours.get? s!"{prefix_}ids@0").bind fun e => digest? (get e "root")) | arowRefuse IDS_OUTSIDE
  unless get r0 "requests" == .str (Flock.hexOf (requestsRoot table ids)) do arowRefuse IDS_OUTSIDE

def WEIGHTS_TAG : String := "verity/pouw/h1/weights/v1"
def WINDOW_ROOT_FORM : String := "the window is not {root}, its open root a SHA-512 digest (128 lowercase hex)"
def REQUESTS_FORM : String := "the window's round-0 request table's root is not a SHA-512 digest (128 lowercase hex)"
def SESSION_FORM : String :=
  "the window's session is not the id of the session the auditor's ledger opened for it (1 to 64 of " ++
  "[A-Za-z0-9._:-], the first alphanumeric)"
def CIRCUIT_FORM : String :=
  "the window's circuit is not its descriptor's SHA-512 (128 lowercase hex) and its PREFIX (a string of printable " ++
  "ASCII but \" and \\)"
def OTHER_WEIGHTS : String :=
  "the window's registered weights are not the ones its open root binds: weights_root of each K's …/b/<K> root, its " ++
  "round-0 request table's root and its session are not its open root (salt.window_open_root), so a B row, its " ++
  "codes or its s_w, is not the one registered before the window's salt. Register each K's B rows once, before the " ++
  "window opens, and open them at that root"

/-- `weights.weights_root`: `tagged` of `WEIGHTS_TAG` and each K's registered `…/b/<K>` root by ascending K. -/
def weightsRoot (roots : List (Nat × ByteArray)) : ByteArray := weightsRootAt 64 WEIGHTS_TAG roots

def weightsRefuse {α : Type} (why : String) : R α := throw ("weights", why)

/-- `weights.b_words`: a B row's 16-bit words, its K int8 codes then the 4 bytes of its f32 s_w (`h1zy.ROW`). -/
def bWords (k : Nat) : Nat := (k + 5) / 2

def otherWeightsEntry (v : String) (k n : Nat) : String :=
  s!"registered weights {v}'s entry is not the registration of K = {k}'s B rows: one-stage's \{value, root, " ++
  s!"leaves, words, domain}, its root a SHA-512 digest, its {n} rows (K's linears' n in turn) of {bWords k} " ++
  s!"16-bit words (K codes then s_w, h1zy.ROW) under the registered domain of {v} and the window's circuit (the " ++
  "SHA-512 of its descriptor), binding no window (registered.domain)"

/-- `credit._weights`: the weights' root the window registered, exactly `<prefix>b/<K>` for each K of its layout once
and no other `…/b/<K>`, each the registration of K's B rows under `SampledProofs.Registered.domain` of its name and
the circuit `program`, binding no window, when `weightsRoot` of their roots, its round-0 request table's root
`requests` and its session `session` are its open root (`windowOpenRoot`); else the refusal (`weights` or
`layout`). -/
def Internal.weights (window layout entries requests session program prefix_ : Json) : R ByteArray := do
  let some r := digest? (get window "root") | weightsRefuse WINDOW_ROOT_FORM
  let some t := digest? requests | weightsRefuse REQUESTS_FORM
  let .str s := session | weightsRefuse SESSION_FORM
  unless sessionId? s do weightsRefuse SESSION_FORM
  let (some _, .str pre) := (digest? program, prefix_) | weightsRefuse CIRCUIT_FORM
  unless _root_.PouwCheck.plain pre do weightsRefuse CIRCUIT_FORM
  let prog := (str? program).getD ""
  discard <| strata layout
  let lins : List (Nat × Nat) := match get layout "linears" with
    | .arr xs => xs.toList.filterMap fun x => match x with
        | .arr #[_, k, n] => do pure (← nat? k, ← nat? n)
        | _ => none
    | _ => []
  let rows : List (Nat × Nat) := (lins.map (·.1)).eraseDups.map fun k =>
    (k, (lins.filter (·.1 == k)).foldl (fun a x => a + x.2) 0)
  let .arr es := entries | weightsRefuse ENTRIES_FORM
  unless es.all isObj do weightsRefuse ENTRIES_FORM
  let mut roots : List (Nat × ByteArray) := []
  for e in es do
    let v := get e "value"
    if isHeld v then
      let name := (str? v).getD ""
      let some (k, n) := rows.find? (fun (k, _) => s!"{pre}b/{k}" == name)
        | weightsRefuse (s!"the window registers weights {name}, which is not {pre}b/<K> of a K of its circuit: it " ++
            s!"registers each of its K's B rows as {pre}b/<K> ({pre} its descriptor's PREFIX), and no other …/b/<K>")
      if (roots.lookup k).isSome then
        weightsRefuse (s!"the window registers K = {k}'s weights twice ({name}): each K's B rows are one registered " ++
          s!"value {pre}b/<K>")
      let d ← (SampledProofs.Registered.domain prog name (bWords k) n 16).mapError fun why => ("weights", why)
      let some h := digest? (get e "root") | weightsRefuse (otherWeightsEntry name k n)
      unless keys e == some ["domain", "leaves", "root", "value", "words"] && nat? (get e "leaves") == some n
          && nat? (get e "words") == some (bWords k) && str? (get e "domain") == some (Flock.hexOf d) do
        weightsRefuse (otherWeightsEntry name k n)
      roots := roots ++ [(k, h)]
  for k in (rows.map (·.1)).mergeSort do
    unless (roots.lookup k).isSome do
      weightsRefuse (s!"the window registers no weights {pre}b/{k} of K = {k}: each K's B rows, their codes and " ++
        "s_w, are registered under the weights' root its open root binds, before its salt")
  let w := weightsRoot roots
  unless windowOpenRoot w t s == r do weightsRefuse OTHER_WEIGHTS
  pure w

def noWeights (name : String) : String :=
  s!"period {name}'s open signs no weights' root ({WEIGHTS_FIELD}, a SHA-512 digest, the gate's --model): " ++
  "the period registers each K's B rows before its first window's salt and opens with their root"

def MODEL_FIELD : String := "model"
def MODEL_FIX : String :=
  "open the period with the gate's --model, which signs the auditor's registration of a checkpoint credit pins " ++
  "(models.json)"

/-- `credit.PINNED_MODELS`: each checkpoint role a period's open may name, with its checkpoint's SHA-512, from the same
`models.json` `credit.py` reads. Credit counts only an int8 role's weights: an open naming another role signs no
weights' root, so `committedWeights` refuses its period. -/
def PINNED_MODELS : List (String × String) :=
  match Json.parse (include_str "../models.json") with
  | .ok j => (fields j).filterMap fun (r, e) => (str? (get e "checkpoint_sha512")).map (r, ·)
  | .error _ => []

def noModel (name : String) : String :=
  s!"period {name}'s open signs a weights' root but no model ({MODEL_FIELD}, \{name, checkpoint_sha512}), so " ++
  s!"nothing says which checkpoint it is the root of: {MODEL_FIX}"

def unpinnedModel (name model : String) : String :=
  s!"period {name}'s open names model {model}, which is no checkpoint role credit pins: {MODEL_FIX}"

def otherCheckpoint (name model : String) : String :=
  s!"period {name}'s open names model {model} with another checkpoint than the one credit pins for it: {MODEL_FIX}"

/-- `credit.committed_weights`: the weights' root period `name`'s open signed (`WEIGHTS_FIELD`, a SHA-512 digest) in
the auditor's ledger, one the challenger's checks have trusted, its open naming the model that root is of
(`MODEL_FIELD`), a role `PINNED_MODELS` pins with the same checkpoint; else the refusal (`weights`). -/
def committedWeights (ledger : Json) (name : String) : R String := do
  let .arr lines := ledger | throw ("weights", noWeights name)
  let body := ((lines.filterMap SampledProofs.Challenger.ledgerEntry?).find? fun e =>
    e.kind == "period-open" && get e.body "period" == Json.str name).map (·.body)
  let some w := body.bind fun b => if (digest? (get b WEIGHTS_FIELD)).isSome then str? (get b WEIGHTS_FIELD) else none
    | throw ("weights", noWeights name)
  let m := (body.map (get · MODEL_FIELD)).getD .null
  let some model := (if keys m == some ["checkpoint_sha512", "name"] then str? (get m "name") else none)
    | throw ("weights", noModel name)
  let some pin := PINNED_MODELS.lookup model | throw ("weights", unpinnedModel name model)
  unless get m "checkpoint_sha512" == Json.str pin do throw ("weights", otherCheckpoint name model)
  pure w

/-- `credit._prefix`: the `PREFIX` of the program function's record of a descriptor `layoutOf` accepted. -/
def prefixOf (descriptor : ByteArray) : Json :=
  let fn := (((String.fromUTF8? descriptor).bind loads?).bind fun d => str? (get d "program_fn")).getD ""
  (((Flock.Program.recordPairs fn).toList.lookup "PREFIX").bind loads?).getD .null

/-- `credit._period_weights`: the session the auditor's ledger opened for the window when the window's registered
weights (`Internal.weights`, over round 0's `requests` in its round log and that session) fold to the weights' root its
period's open signed (`committedWeights`); else the refusal (`weights`, `layout` or `coin`). -/
def periodWeights (key ledger window rounds layout entries : Json) (program : String) (prefix_ : Json) :
    R String := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let a ← (SampledProofs.Challenger.auditedSession ledger window k).mapError fun e =>
    ("coin", s!"the challenger's session is not trusted: {e}")
  let some name := a.period
    | weightsRefuse ("the window is in no period of the auditor's ledger, so no period-open signed the weights' " ++
        s!"root it opens ({WEIGHTS_FIELD}): reserve its session inside a period whose open signs it")
  let signed ← committedWeights ledger name
  let requests := match rounds with
    | .arr xs => (xs[0]?.map (get · "requests")).getD .null
    | _ => .null
  let w ← Internal.weights window layout entries requests (.str a.session) (.str program) prefix_
  let got := Flock.hexOf w
  unless got == signed do
    weightsRefuse (s!"the window's registered weights' root {got.take 16}... is not {signed.take 16}..., the one " ++
      s!"period {name}'s open signed: a B row's codes or s_w differ from the ones the period registered before its " ++
      "first window's salt. Open every window of a period on its registered B rows")
  pure a.session

/-- `credit._window`: `window`'s credited calls and the layout it read from the descriptor. -/
def windowLayout
    (key ledger pins main window session salt calls served commit rounds draw descriptor entries proofs k : Json) :
    R (List Nat × Json) := do
  let got ← Internal.credited key ledger pins main window session salt calls served commit rounds draw
  let log := match rounds with
    | .arr xs => xs
    | _ => #[]
  roundsCheck key ledger pins main window session (.arr (log.map (get · "root"))) (.arr (log.map (get · "coin")))
  Internal.aLink rounds entries (match calls with
    | .arr xs => xs.size
    | _ => 0)
  let some d := (match descriptor with
      | .str s => if isHexBytes s then hex? (.str s) else none
      | _ => none) | layoutRefuse DESCRIPTOR_HEX
  let layout ← layoutOf d
  let program := Flock.hexOf (Flock.Sha512.hash d)
  let sid ← periodWeights key ledger window rounds layout entries program (prefixOf d)
  let ss ← strata layout
  let (.arr rows, .arr lcalls) := (calls, get layout "calls") | layoutRefuse LAYOUT_FORM
  unless rows.size == lcalls.size do
    throw ("manifest", s!"the manifest's {rows.size} rows are not the circuit's {lcalls.size} calls: each call of " ++
      "the window's Program is one manifest row, in order")
  Internal.arow log entries (((commitOf? commit).map (·.2)).getD #[]) layout ((str? (prefixOf d)).getD "")
  let o ← (LiveOpening.ofJson (get draw "coin")).mapError fun e => ("coin", s!"the draw coin: {e}")
  proved (← drawnOf o.coin (get window "root") (get draw "root") (Json.mkObj [("receipts", get draw "receipts")]) ss
    k) proofs
  Internal.registration ((str? (get window "root")).getD "") program ((str? (prefixOf d)).getD "") entries proofs ss
    layout log.size sid
  pure (got, layout)

/-- `credit.window`, an entry: a served H1 window's credited calls, else the refusal of the first step that refuses,
every step over the same evidence and the one round log `rounds`: `Internal.credited` (the credit bits fixed before the
draw coin), `roundsCheck` (the log's roots the ones the challenger received, each round's coin opened once after its
root), `Internal.aLink` (each A value among the registered `entries` the tree over the segments those roots commit),
the layout of its circuit (`layoutOf` of `descriptor`, the bytes in hex of the circuit its proofs were verified
against), `periodWeights` (`Internal.weights`, its period's: the window's `<PREFIX>b/<K>` fold to the weights' root the
open of the period holding it signed, `committedWeights`, and with round 0's `requests` and its session to its open
root), its manifest one row per call of the layout, and `Internal.outcome`'s rule on the draw coin `credited` read at
its period's committed K `k`, the stream the window's root, the registration its draw root and the receipt
`{"receipts": …}`, its last round's receipts, each of `proofs` the verifier of record's verdict for a drawn unit's
session (`PROOF_KEYS`), and `Internal.registration` (the `entries` the window's own registration's, under its open root
and the descriptor's SHA-512, and each drawn unit's session verified against that descriptor, opening its registered
values at their registered roots, its K's A value and its K's `<PREFIX>b/<K>` among them). `periodCredit` ties
`descriptor` to the circuit the window's period committed. -/
def window
    (key ledger pins main window session salt calls served commit rounds draw descriptor entries proofs k : Json) :
    R (List Nat) :=
  (·.1) <$> windowLayout key ledger pins main window session salt calls served commit rounds draw descriptor entries
    proofs k

/-- `credit.period`: none when `roots` are exactly the windows the auditor's ledger lists in period `name`'s close, in
its order, and that close is complete (`checked`, under the pinned key `key`); else the refusal (`coin`, `period`,
`missing` or `order`). -/
def period (key ledger name roots : Json) : R Unit := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  discard <| (SampledProofs.Challenger.checked ledger k).mapError fun e =>
    ("coin", s!"the auditor's ledger is not trusted: {e}")
  let .str name := name | throw ("period", "the period is not named")
  let .arr lines := ledger | throw ("period", "the period is not named")
  let es := lines.filterMap SampledProofs.Challenger.ledgerEntry?
  let named := fun (e : SampledProofs.Challenger.LedgerEntry) => get e.body "period" == Json.str name
  unless es.any (fun e => e.kind == "period-open" && named e) do
    throw ("period", s!"the auditor's ledger opens no period {name}: credit counts only a period the ledger opens " ++
      "and closes")
  let some close := (es.find? fun e => e.kind == "period-close" && named e).map (·.body)
    | throw ("period", s!"period {name} has no close in the auditor's ledger, so it earns nothing: close it " ++
        "(period-close) and present the ledger after its close")
  let ws := match get close "windows" with
    | .arr a => a
    | _ => #[]
  unless get close "complete" == Json.bool true do
    match ws.findIdx? (fun w => get w "complete" != Json.bool true) with
    | none => throw ("period", s!"period {name} closed with no window, so it earns nothing")
    | some i => throw ("period", s!"period {name} closed incomplete: its window {i} (session " ++
        s!"{(str? (get ws[i]! "session")).getD ""}) never opened or didn't end complete before the close, so the " ++
        "period earns nothing")
  let some rs := (match roots with
      | .arr a => a.toList.mapM fun r => if (digest? r).isSome then str? r else none
      | _ => none) | throw ("period", "the period's windows are not a list of window roots (SHA-512 digests)")
  let want := ws.toList.map fun w => (str? (get w "root")).getD ""
  let wanted : Std.HashMap String Unit := want.foldl (fun m r => m.insert r ()) {}
  let presented : Std.HashMap String Unit := rs.foldl (fun m r => m.insert r ()) {}
  for i in [0:max want.length rs.length] do
    if i ≥ want.length then
      throw ("period", s!"the windows presented run past the {want.length} that period {name}'s close lists: only " ++
        "its close's windows count, each once")
    let w := want[i]!
    if i ≥ rs.length || (rs[i]! != w && !presented.contains w) then
      throw ("missing", s!"window {i} of period {name} ({w.take 16}...) is missing: every window between the " ++
        "period's open and its close counts, so present them all, in the close's order")
    let r := rs[i]!
    if r != w then
      if !wanted.contains r then
        throw ("period", s!"window {r.take 16}... is not one of period {name}'s windows: only the windows its close " ++
          "lists count")
      throw ("order", s!"window {i} of period {name} is {r.take 16}..., not {w.take 16}..., the close's window {i}: " ++
        "present the period's windows in the order its close lists them")

/-- `credit.committed_k`: the K period `name`'s open committed (`K_FIELD`, a positive integer at most `MAX_K`) in the
auditor's ledger, one `period` has trusted; else the refusal (`budget`). -/
def committedK (ledger : Json) (name : String) : R Nat := do
  let .arr lines := ledger | throw ("period", "the period is not named")
  let body := ((lines.filterMap SampledProofs.Challenger.ledgerEntry?).find? fun e =>
    e.kind == "period-open" && get e.body "period" == Json.str name).map (·.body)
  let some k := (body.bind fun b => nat? (get b K_FIELD)).filter (fun k => 1 ≤ k && k ≤ MAX_K)
    | throw ("budget", s!"period {name}'s open commits no draw K ({K_FIELD}, a positive integer at most 2^24), so " ++
        "its windows have no K to be drawn with: the proof service commits K in the period-open, beside its circuit " ++
        "and cap, before any draw")
  pure k

private def periodBody (ledger : Json) (name : String) : Option Json :=
  match ledger with
  | .arr lines => ((lines.filterMap SampledProofs.Challenger.ledgerEntry?).find? fun e =>
      e.kind == "period-open" && get e.body "period" == Json.str name).map (·.body)
  | _ => none

/-- `credit.committed_pin`: the commit period `name`'s open recorded it opened under (`PERIOD_PIN_FIELD`, a commit
string), or `none` when its open records none: `periodCredit` holds every window's session to within the freshness floor
of it on main's first-parent line (`pinHygiene`). Optional. -/
def committedPin (ledger : Json) (name : String) : Option String :=
  (periodBody ledger name).bind fun b => str? (get b PERIOD_PIN_FIELD)

/-- `credit.committed_floor`: the freshness floor period `name`'s open recorded (`STALE_FIELD`, a nonnegative integer),
or `none` when its open records none: `periodCredit` counts no window whose pin is more than this many first-parent
commits behind the period's open pin (`committedPin`), default 0 when the open pins but records no floor. Optional. -/
def committedFloor (ledger : Json) (name : String) : Option Nat :=
  (periodBody ledger name).bind fun b => nat? (get b STALE_FIELD)

/-- `credit._main_index`: `commit`'s position in the verifier's copy of main's first-parent history `main` (newest
first, so position 0 is main's head and position p is p commits behind it), or `none` when `main` doesn't hold it. -/
def mainIndex? (main : Json) (commit : String) : Option Nat :=
  match main with
  | .arr cs => cs.findIdx? (· == Json.str commit)
  | _ => none

/-- `credit._pin_hygiene`: why window `i` of period `name`, whose session ran commit `pinned` on main's first-parent line
`main`, is too old for the period: its pin more than the period's freshness floor of first-parent commits behind the
commit the period's open recorded (`periodPin`, `committedPin`; `floor`, `committedFloor`, default 0 when the open pins
but records no floor), measured from that pin and never from main's head; `none` when it is recent enough or the open
records no pin. The per-window pin and circuit are `creditedSession`'s; this is the period-level hygiene on top
(CMP74-FOLLOW). -/
def pinHygiene (name : String) (i : Nat) (main : Json) (pinned : String) (periodPin : Option String)
    (floor : Option Nat) : Option (String × String) :=
  match periodPin with
  | none => none
  | some pp =>
    match mainIndex? main pinned with
    | none => none
    | some pos =>
      match mainIndex? main pp with
      | none => some ("pin", s!"period {name}'s open pins commit {pp.take 16}..., which is not on main's first-parent " ++
          s!"line in the verifier's copy ({PERIOD_PIN_FIELD}): open the period under a commit " ++
          "git rev-list --first-parent origin/main holds")
      | some base =>
        let f := floor.getD 0
        if pos > base + f then
          some ("pin", s!"window {i} of period {name} is pinned {pos - base} first-parent commit(s) before the commit " ++
            s!"its period opened under, past its freshness floor of {f} ({STALE_FIELD}): {pinned.take 16}... is " ++
            s!"that far behind {pp.take 16}... on main's first-parent line, so the period earns nothing; " ++
            "open every window within the floor of its period's open pin")
        else none

/-- `credit.period_credit`'s credit: each window's credited calls in the close's order, the committed K, the δ the ε
is reported at and each window's ε. -/
structure PeriodCredit where
  windows : List (List Nat)
  k : Nat
  delta : Rat
  eps : List (Option Rat)

def PeriodCredit.toJson (c : PeriodCredit) : Json :=
  Json.mkObj [("windows", Lean.toJson c.windows), ("k", c.k), ("delta", toString c.delta),
    ("eps", Json.arr (c.eps.map fun e => (e.map fun e => Json.str (toString e)).getD .null).toArray)]

/-- `credit.period_credit`: period `name`'s credit: the period's windows are exactly its close's (`period`), its open
committed K (`committedK`) and signed the weights' root every window's registered B rows fold to (`committedWeights`),
each window's session ran a commit the host's pins log `pins` records on the verifier's copy of main's first-parent
history `main` and the window proves the circuit the period's open committed (`circuit`: the pin first, then its
descriptor hashing to it, `SampledProofs.Challenger.creditedSession`), each window's session pinned within the period's
freshness floor (`committedFloor`, first-parent commits, default 0) of the commit the period's open recorded
(`committedPin`), counted from that pin and never from main's head, both optional and refused with `pin` (`pinHygiene`),
and each window is credited (`window`) drawn at K, a refused window refusing the period; each window's ε is what its draw
of K established (`established`). -/
def periodCredit (key ledger pins main name windows : Json) : R PeriodCredit := do
  let some ws := (match windows with
      | .arr a => if a.all (fun w => keys w == some PERIOD_KEYS) then some a.toList else none
      | _ => none) | throw ("period", WINDOWS_FORM)
  let roots := Json.arr (ws.toArray.map fun w => match get w "window" with
    | j@(.obj _) => get j "root"
    | _ => .null)
  period key ledger name roots
  let nm := (str? name).getD ""
  let bigK ← committedK ledger nm
  discard <| committedWeights ledger nm
  let periodPin := committedPin ledger nm
  let floor := committedFloor ledger nm
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let mut out : List (List Nat) := []
  let mut eps : List (Option Rat) := []
  for (w, i) in ws.zipIdx do
    let f := get w
    let some d := (match f "descriptor" with
        | .str s => if isHexBytes s then hex? (.str s) else none
        | _ => none) | throw ("circuit", s!"window {i} of period {nm}: {DESCRIPTOR_HEX}")
    let a ← (SampledProofs.Challenger.creditedSession ledger (f "window") k pins main d).mapError fun e =>
      ("circuit", s!"window {i} of period {nm} doesn't count toward it, so the period earns nothing: {e}")
    if let some why := pinHygiene nm i main a.pinned periodPin floor then throw why
    let (got, layout) ← (windowLayout key ledger pins main (f "window") (f "session") (f "salt") (f "calls") (f "served")
      (f "commit") (f "rounds") (f "draw") (f "descriptor") (f "entries") (f "proofs") (Lean.toJson bigK)).mapError
      fun (c, why) => ("window", s!"window {i} of period {nm} is refused, so the period earns nothing: {c}: {why}")
    out := out ++ [got]
    eps := eps ++ [established layout got bigK]
  pure ⟨out, bigK, DELTA, eps⟩

end PouwCheck.Credit
