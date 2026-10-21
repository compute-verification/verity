import Std.Data.HashMap
import PouwCommit.Frame
import PouwCheck.Json
import PouwCheck.Pass
import PouwWindow.Rounds
import SampledProofs.ChallengerSession
import SampledProofs.PyJson

/-!
# A served window's credited set, its drawn units' outcome and its A rows (`credit.py`; RT4-INT-32, RT4-INT-34)

The verifier of record of `verity/core/protocols/pouw/credit.py`, which the difftest (`tests/test_difftest.py`) holds to
it, string for string:

* `credited` is `credit.credited`: each manifest row (the window's calls, in order) carries its call's verdict as
  `credit`, a credited row its z table's root, and the draw root over the manifest (`callsRoot`, rows written as Python's
  `json.dumps(row, sort_keys=True)`), the last round's receipts and every round's root and coin (`drawRoot`) must be the
  one root the challenger's log shows its draw coin followed, read through the session the auditor's signed ledger opens
  for the window, its host at a commit of the verifier's copy of main (`trustedSessionOf`). So a credit bit changed
  after the draw coin is refused.
* `outcome` is `credit.outcome`: every drawn unit has exactly one proof outcome and it is an accept, else the whole
  window is refused (no partial credit).
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
def DRAWN_FORM : String := "the drawn units are not a nonempty list of distinct unit indices"
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
def credited (key main ledger window session salt calls served rounds draw : Json) : R (List Nat) := do
  let k ← (ChallengerKey.ofJson key).mapError fun e => ("coin", s!"the verifier's challenger key is malformed: {e}")
  let (c, received) ← (trustedSessionOf ledger window k main session).mapError fun e =>
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

/-- `credit.outcome`: none when every drawn unit has exactly one outcome and it is an accept, and no other unit has one;
else the refusal (`outcome`, `missing` or `failed`). -/
def outcome (drawn proofs : Json) : R Unit := do
  let some ds := (match drawn with
      | .arr a => a.toList.mapM nat?
      | _ => none) | throw ("outcome", DRAWN_FORM)
  unless !ds.isEmpty && ds.eraseDups.length == ds.length do throw ("outcome", DRAWN_FORM)
  let .arr ps := proofs | throw ("outcome", PROOFS_FORM)
  let mut seen : List (Nat × String) := []
  for p in ps.toList do
    let some (u, v) := (if keys p == some ["unit", "verdict"] then
        match nat? (get p "unit"), get p "verdict" with
        | some u, .str v => if VERDICTS.contains v then some (u, v) else none
        | _, _ => none
      else none) | throw ("outcome", PROOFS_FORM)
    if (seen.lookup u).isSome then
      throw ("outcome", s!"unit {u} has two proof outcomes: a drawn unit is proved and verified once, and a retry " ++
        "never reproves it")
    unless ds.contains u do
      throw ("outcome", s!"unit {u} has a proof outcome and was not drawn: only the draw's units are proved")
    seen := seen ++ [(u, v)]
  for u in ds do
    match seen.lookup u with
    | none => throw ("missing", s!"drawn unit {u} has no proof outcome: every drawn unit's proof is verified, and a " ++
        "window with one missing is refused whole (no partial credit). Prove and verify every drawn unit")
    | some v =>
      unless v == "accept" do
        throw ("failed", s!"drawn unit {u}'s proof was rejected: a failed drawn unit refuses the whole window, not " ++
          "only its call (no partial credit, and no retry)")

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
