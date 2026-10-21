import Std.Data.HashMap
import PouwCommit.Frame
import PouwCheck.Json
import PouwCheck.Pass
import PouwWindow.Draw
import PouwWindow.Rounds
import SampledProofs.ChallengerSession
import SampledProofs.PyJson

/-!
# A served window's credited set, its drawn units and their outcome, its A rows and its period (`credit.py`; RT4-INT-32, -34 and -42)

The verifier of record of `verity/core/protocols/pouw/credit.py`, which the difftest (`tests/test_difftest.py`) holds to
it, string for string:

* `credited` is `credit.credited`: each manifest row (the window's calls, in order) carries its call's verdict as
  `credit`, a credited row its z table's root, and the draw root over the manifest (`callsRoot`, rows written as Python's
  `json.dumps(row, sort_keys=True)`), the last round's receipts and every round's root and coin (`drawRoot`) must be the
  one root the challenger's log shows its draw coin followed, read through the session the auditor's signed ledger opens
  for the window (`trustedSessionOf`). So a credit bit changed after the draw coin is refused.
* `drawn` is `credit.drawn`: the units the draw coin draws from the window's own layout, `drawCheck`'s rule under the
  law `work:k` (`workUnits`), one stratum per K (`strata`, over `h1window.program`'s unit order) at the K the period's
  open committed (`law`; `committedK`). Never a list the prover hands in. `established` is the ε that draw established
  over the window's credited work at `DELTA` (`PouwWindow.epsAt` at the window's inputs), which `periodCredit` reports.
* `outcome` is `credit.outcome`: every unit `drawn` draws has exactly one proof outcome and it is an accept, else the
  whole window is refused (no partial credit). `window` is `credit.window`: `credited`, then `outcome` under the served
  window's arguments (the stream its root, the registration its draw root, the receipt `{"receipts": …}`).
* `period` is `credit.period`: the windows presented are exactly those the auditor's ledger lists in the period's
  close, in its order, and the close is complete; `periodCredit` is `credit.period_credit`, every window of it proving
  the period's committed circuit (`creditedSession`) and credited.
* `aLink` is `credit.a_link`: each registered A value (`…/a/<K>`) is the frame-v3 SHA-512 tree over the aligned subtrees
  each round's root commits (`roundRoot`), round s's tiling the value's rows `[s·L/S, (s+1)·L/S)`.

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
  "the window's draw root is not over its manifest (each call's credit bit among it), its last round's receipts and " ++
  "every round's root and coin"
def CHANGED : String :=
  "the draw root over the manifest (each call's credit bit among it), its last round's receipts and every round's " ++
  "root and coin is not the one root the challenger's log shows its draw coin followed: a credit bit or a call " ++
  "changed after the draw coin, or the draw opened after another root. Commit every call's credit bit in its " ++
  "manifest row before the draw root, and never change it after the draw coin"
def K_FIELD : String := "draw_k"
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
  ["calls", "draw", "layout", "proofs", "rounds", "salt", "served", "session", "window"]
def PERIOD_KEYS : List String :=
  ["calls", "descriptor", "draw", "layout", "proofs", "rounds", "salt", "served", "session", "window"]
def WINDOWS_FORM : String :=
  "the period's windows are not a list of each window's evidence {" ++ ", ".intercalate PERIOD_KEYS ++ "}, " ++
  "in its close's order"
def DESCRIPTOR_HEX : String :=
  "the window's descriptor is not the bytes of the circuit its proofs were verified against in lowercase hex (its " ++
  "canonical JSON without annotations, the bytes program_digest hashes)"
def PROOFS_FORM : String := "the drawn units' proof outcomes are not a list of {unit, verdict}, verdict accept or reject"
def LOG_FORM : String := "the round log is not a nonempty list"
def ENTRIES_FORM : String := "the window's registered entries are not a list of objects"
def NO_A : String :=
  "the window registers no A value (…/a/<K>), so nothing ties its units' A rows to the round roots: register the " ++
  "window's A values"

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
window's Program salt; with the served trees' digest, SHA-512 of that and it. -/
def callsRoot (salt : ByteArray) (rows : List Json) (served : Option ByteArray) : ByteArray :=
  let h := Flock.Sha512.hash (rows.foldl (fun acc r => acc ++ (dumps DEPTH r ++ "\n").toUTF8) (CALLS_TAG ++ salt))
  match served with
  | none => h
  | some s => Flock.Sha512.hash (h ++ s)

/-- `credit.credited`: the credited calls' indices, else the refusal. -/
def credited (key ledger window session salt calls served rounds draw : Json) : R (List Nat) := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let (c, received) ← (trustedSessionOf ledger window k session).mapError fun e =>
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
  let .arr xs := rounds | throw ("rounds", "the round log is not a list")
  let mut parts : List Part := []
  for (x, r) in xs.toList.zipIdx do
    let some root := (if isObj x then digest? (get x "root") else none)
      | throw ("rounds", s!"round {r}'s root is not a SHA-512 digest")
    let o ← (LiveOpening.ofJson (get x "coin")).mapError fun e => ("rounds", s!"round {r}'s coin: {e}")
    parts := parts ++ [.bytes root, .bytes o.coin]
  let some receipts := (if isObj draw then digest? (get draw "receipts") else none) | throw ("draw", DRAW_FORM)
  let root := tagged DRAW_TAG ([.bytes (callsRoot s rows servedB), .bytes receipts] ++ parts)
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

/-- `credit._proved`: none when every unit of `drawn` has exactly one outcome and it is an accept, and no other unit
has one; else the refusal (`outcome`, `missing` or `failed`). -/
def proved (drawn : Array Nat) (proofs : Json) : R Unit := do
  let picked : Std.HashMap Nat Unit := drawn.foldl (fun m u => m.insert u ()) {}
  let .arr ps := proofs | throw ("outcome", PROOFS_FORM)
  let mut seen : Std.HashMap Nat String := {}
  for p in ps.toList do
    let some (u, v) := (if keys p == some ["unit", "verdict"] then
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

/-- `credit.outcome`: none when every unit the draw coin draws from the window (`drawn`) has exactly one outcome and it
is an accept, and no other unit has one; else the refusal. -/
def outcome (coin stream registration receipt layout k proofs : Json) : R Unit := do
  proved (← drawn coin stream registration receipt layout k) proofs

/-- `credit.window`: a served H1 window's credited calls (`credited`), its manifest one row per call of `layout`, and
`outcome` of the draw coin `credited` read at its period's committed K `k`, the stream the window's root, the
registration its draw root and the receipt `{"receipts": …}`, its last round's receipts; else the refusal. -/
def window (key ledger window session salt calls served rounds draw layout proofs k : Json) : R (List Nat) := do
  let got ← credited key ledger window session salt calls served rounds draw
  let ss ← strata layout
  let (.arr rows, .arr lcalls) := (calls, get layout "calls") | layoutRefuse LAYOUT_FORM
  unless rows.size == lcalls.size do
    throw ("manifest", s!"the manifest's {rows.size} rows are not the layout's {lcalls.size} calls: each call of the " ++
      "window's Program is one manifest row, in order")
  let o ← (LiveOpening.ofJson (get draw "coin")).mapError fun e => ("coin", s!"the draw coin: {e}")
  proved (← drawnOf o.coin (get window "root") (get draw "root") (Json.mkObj [("receipts", get draw "receipts")]) ss
    k) proofs
  pure got

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

/-- `credit._HEX`: bytes as lowercase hex, at least one. -/
def isHexBytes (s : String) : Bool :=
  !s.isEmpty && s.length % 2 == 0 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

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
committed K (`committedK`), each window proves the circuit the period's open committed (`circuit`: its descriptor
hashes to it, `SampledProofs.Challenger.creditedSession`) and each window is credited (`window`) drawn at K, a refused
window refusing the period; each window's ε is what its draw of K established (`established`). -/
def periodCredit (key ledger name windows : Json) : R PeriodCredit := do
  let some ws := (match windows with
      | .arr a => if a.all (fun w => keys w == some PERIOD_KEYS) then some a.toList else none
      | _ => none) | throw ("period", WINDOWS_FORM)
  let roots := Json.arr (ws.toArray.map fun w => match get w "window" with
    | j@(.obj _) => get j "root"
    | _ => .null)
  period key ledger name roots
  let nm := (str? name).getD ""
  let bigK ← committedK ledger nm
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let mut out : List (List Nat) := []
  let mut eps : List (Option Rat) := []
  for (w, i) in ws.zipIdx do
    let f := get w
    let some d := (match f "descriptor" with
        | .str s => if isHexBytes s then hex? (.str s) else none
        | _ => none) | throw ("circuit", s!"window {i} of period {nm}: {DESCRIPTOR_HEX}")
    discard <| (SampledProofs.Challenger.creditedSession ledger (f "window") k d).mapError fun e =>
      ("circuit", s!"window {i} of period {nm} doesn't count toward it, so the period earns nothing: {e}")
    let got ← (window key ledger (f "window") (f "session") (f "salt") (f "calls") (f "served") (f "rounds") (f "draw")
      (f "layout") (f "proofs") (Lean.toJson bigK)).mapError fun (c, why) =>
        ("window", s!"window {i} of period {nm} is refused, so the period earns nothing: {c}: {why}")
    out := out ++ [got]
    eps := eps ++ [established (f "layout") got bigK]
  pure ⟨out, bigK, DELTA, eps⟩

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

/-- `credit.is_a`: a name whose last two parts are `a` and a contraction (`…/a/<K>`). -/
def isA : Json → Bool
  | .str v => match (v.splitOn "/").reverse with
    | k :: "a" :: _ => !k.isEmpty && k.all Char.isDigit
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

/-- `credit.a_link`: none when every registered A value is the tree over the segments the round log commits, else the
refusal. -/
def aLink (rounds entries : Json) : R Unit := do
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
  let a := es.toList.filterMap fun e => if isA (get e "value") then (·, e) <$> str? (get e "value") else none
  if a.isEmpty then aLinkRefuse NO_A
  let mut names : List String := []
  for (v, _) in a do
    if names.contains v then aLinkRefuse s!"the window registers A value {v} twice"
    names := names ++ [v]
  for (p, r) in segs.zipIdx do
    for (v, _) in p do
      unless names.contains v do
        aLinkRefuse s!"round {r} lists segments of {v}, which is no registered A value of the window"
  let steps := xs.length
  for (v, e) in a do
    let (some root, some domain, some count) := (digest? (get e "root"), digest? (get e "domain"), nat? (get e "leaves"))
      | aLinkRefuse (s!"registered A value {v}'s entry has no root, domain or leaf count from 1 to {MAX_ROWS} (the " ++
          "verifier hashes its pads)")
    unless 1 ≤ count && count ≤ MAX_ROWS do
      aLinkRefuse (s!"registered A value {v}'s entry has no root, domain or leaf count from 1 to {MAX_ROWS} (the " ++
        "verifier hashes its pads)")
    if count % steps != 0 then
      aLinkRefuse s!"registered A value {v}'s {count} rows are not the window's {steps} rounds' equal blocks"
    let per := count / steps
    let mut have_ : Std.HashMap (Nat × Nat) ByteArray := {}
    for (p, r) in segs.zipIdx do
      let stop := (r + 1) * per
      let untiled := s!"round {r}'s segments of {v} don't tile its rows [{r * per}, {stop}) in order, each an aligned " ++
        "subtree: the serving process lists each round's rows of every A value as aligned subtrees (`credit.cover`)"
      let mut pos := r * per
      for (level, index, h) in (p.lookup v).getD [] do
        if index <<< level != pos || pos + 2 ^ level > stop then aLinkRefuse untiled
        have_ := have_.insert (level, index) h
        pos := pos + 2 ^ level
      if pos != stop then aLinkRefuse untiled
    unless treeRoot domain count have_ == some root do
      aLinkRefuse (s!"registered A value {v}'s root is not the tree over the segments the round roots commit: its " ++
        "rows are not the ones fixed under each round's root before that round's coin. Register the A value whose " ++
        "rows each round's root committed")

end PouwCheck.Credit
