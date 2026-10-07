import CoreProtocol.Protocol
import CoreProtocol.Profile
import Flock.Hash
import PouwCheck.Json

/-!
# The check's three inputs (`core_check.py`, "the three inputs")

* `Params`: the verifier's own parameters of the window: the check item, ε and δ, the grid's bits, the declared range
  of tile counts, the window's context and the check query's SHA-512.
* `Registration`: one-stage's record of the window (`registration.record`), canonical JSON, with what PoUW's rule reads;
  its digest is the record's own, `registration.digest`.
* `Evidence`: the proof service's answer for a registration, in the core interface's answer form: the protocol, the
  registration it is for, and what it said (on acceptance, the integrity profile and the subset draw).

Each decoder refuses with `core_check.py`'s message, and `Core.Protocol.answer` prefixes it with the input's name.
-/

namespace PouwCheck

open Lean (Json)
open Flock (V)

def scheme : String := "frame-v3-sha512/hm96-sha512"
def formats : List String := ["verity/registration/v1", "verity/registration/v2"]
def paramsKeys : List String := ["check", "context", "delta", "eps", "grid_bits", "query", "range"]
def checkKeys : List String := ["entries", "scheme", "template", "tile_work"]
def answerKeys : List String := ["commitment", "profile", "protocol", "public", "reason", "verdict"]

/-- The `check` item: the tile template, its work per tile, the commitment scheme and the public bound on entries. -/
structure CheckItem where
  entries : Nat
  scheme : String
  template : String
  tileWork : Nat
  deriving DecidableEq, Repr

def CheckItem.toJson (c : CheckItem) : Json :=
  Json.mkObj [("entries", c.entries), ("scheme", c.scheme), ("template", c.template), ("tile_work", c.tileWork)]

structure Params where
  check : CheckItem
  context : String
  delta : Core.Ratio
  eps : Core.Ratio
  gridBits : Nat
  /-- The check query's SHA-512, hex. -/
  query : String
  lo : Nat
  hi : Nat
  deriving DecidableEq, Repr

def isHexDigest (s : String) : Bool := s.length = 128 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

/-- A ratio-string field and its range, in `params`' order. -/
def ratioField (j : Json) (k : String) (ok : Core.Ratio → Bool) (what : String) : V Core.Ratio := do
  let some s := str? (get j k) | throw s!"{k} is not a string"
  let r ← Core.Ratio.ofString s k
  unless ok r do throw s!"{k} is not in {what}"
  pure r

def paramsOf (j : Json) : V Params := do
  unless keys j == some paramsKeys do throw s!"keys are not {", ".intercalate paramsKeys}"
  let c := get j "check"
  unless keys c == some checkKeys do throw s!"check's keys are not {", ".intercalate checkKeys}"
  let some entries := (nat? (get c "entries")).filter (2 ≤ ·)
    | throw "check.entries is not a natural number of at least 2"
  unless get c "scheme" == .str scheme do throw s!"check.scheme is not {scheme}"
  let some template := (str? (get c "template")).filter (!·.isEmpty) | throw "check.template is not a non-empty string"
  let some tileWork := (nat? (get c "tile_work")).filter (1 ≤ ·)
    | throw "check.tile_work is not a positive natural number"
  let some context := (str? (get j "context")).filter (!·.isEmpty) | throw "context is not a non-empty string"
  let delta ← ratioField j "delta" (fun r => 0 < r.num && r.num < r.den) "(0, 1)"
  let eps ← ratioField j "eps" (fun r => 0 < r.num && r.num ≤ r.den) "(0, 1]"
  let some gridBits := nat? (get j "grid_bits") | throw "grid_bits is not a natural number"
  let some query := (str? (get j "query")).filter isHexDigest | throw "query is not a SHA-512 digest (128 hex digits)"
  let range : Option (Nat × Nat) := match get j "range" with
    | .arr #[a, b] => do
      let lo ← nat? a
      let hi ← nat? b
      if 1 ≤ lo ∧ lo ≤ hi then some (lo, hi) else none
    | _ => none
  let some (lo, hi) := range | throw "range is not [lo, hi] with 1 <= lo <= hi"
  pure { check := ⟨entries, scheme, template, tileWork⟩, context, delta, eps, gridBits, query, lo, hi }

/-- One of the record's roots. -/
structure Root where
  port : String
  root : String
  leaves : Nat
  domain : String
  deriving DecidableEq, Repr

def rootOf? (j : Json) : Option Root := do
  guard (keys j == some ["domain", "leaves", "port", "root"])
  pure ⟨← str? (get j "port"), ← str? (get j "root"), ← nat? (get j "leaves"), ← str? (get j "domain")⟩

/-- The record's `window`: its kind, its context, and the items it registers. -/
structure Window where
  kind : String
  context : String
  items : Json

/-- The registration: the canonical bytes its digest hashes, and the record's fields PoUW's rule reads. -/
structure Registration where
  bytes : ByteArray
  format : String
  program : String
  query : String
  partition : String
  scheme : String
  population : Nat
  law : Json
  roots : Array Root
  window : Window

def registrationOf (b : ByteArray) : V Registration := do
  let rec_ ← Core.canonicalJson b
  let some format := (str? (get rec_ "format")).filter (formats.contains ·)
    | throw s!"format is not {" or ".intercalate formats}"
  let s (k : String) : V String := match str? (get rec_ k) with
    | some v => pure v
    | none => throw s!"{k} is missing or not a string"
  let program ← s "program"
  let query ← s "query"
  let partition ← s "partition"
  let scheme ← s "scheme"
  let some population := nat? (get rec_ "population") | throw "population is missing or not a natural number"
  let some law := PouwWindow.field? rec_ "law" | throw "law is missing"
  let roots : Option (Array Root) := match get rec_ "roots" with
    | .arr rs => rs.mapM rootOf?
    | _ => none
  let some roots := roots | throw "roots is not a list of {domain, leaves, port, root}"
  let w := get rec_ "window"
  let window : Option Window := do
    guard (keys w == some ["context", "kind", "public"])
    let items := get w "public"
    guard (keys items).isSome
    pure ⟨← str? (get w "kind"), ← str? (get w "context"), items⟩
  let some window := window | throw "window is not {context, kind, public} with strings and an object"
  pure { bytes := b, format, program, query, partition, scheme, population, law, roots, window }

/-- `registration.digest`: SHA-512(format ‖ 0 ‖ the canonical record). -/
def Registration.digest (c : Registration) : ByteArray :=
  Flock.Sha512.hash (c.format.toUTF8 ++ ByteArray.mk #[0] ++ c.bytes)

/-! The integrity profile read back from `Core.Integrity.toJson`'s form. -/

def strings? (j : Json) : Option (Array String) :=
  match j with
  | .arr xs => xs.mapM str?
  | _ => none

def drawOf? (j : Json) : Option Core.Draw :=
  match keys j with
  | some ["subset"] => (nat? (get j "subset")).map .subset
  | some ["bernoulli"] => do (Core.Ratio.ofString (← str? (get j "bernoulli")) "bernoulli").toOption.map .bernoulli
  | some ["law", "strata"] => do
    guard (get j "law" == .str "stratified")
    let .arr ss := get j "strata" | none
    let strata ← ss.mapM fun s => do
      guard (keys s == some ["k", "name", "size"])
      pure (⟨← str? (get s "name"), ← nat? (get s "size"), ← nat? (get s "k")⟩ : Core.Stratum)
    pure (.stratified strata)
  | _ => none

def levelOf? (j : Json) : Option Core.Level := do
  guard (keys j == some ["draw", "name", "query", "size"])
  let query ← match get j "query" with
    | .null => some none
    | .str q => some (some q)
    | _ => none
  let draw ← match get j "draw" with
    | .null => some none
    | d => (drawOf? d).map some
  pure { name := ← str? (get j "name"), query, size := ← nat? (get j "size"), draw }

def integrityOf? (j : Json) : Option Core.Integrity := do
  guard (keys j == some ["covered", "delta", "drawn", "levels", "linked", "prescribed", "programs"])
  let .obj drawn := get j "drawn" | none
  let drawn ← drawn.toArray.mapM fun (k, v) => do pure (k, ← nat? v)
  let .arr levels := get j "levels" | none
  let .bool linked := get j "linked" | none
  pure { programs := ← strings? (get j "programs"), prescribed := ← strings? (get j "prescribed"), linked,
         covered := ← strings? (get j "covered"), levels := ← levels.mapM levelOf?,
         delta := ← (Core.Ratio.ofString (← str? (get j "delta")) "delta").toOption, drawn }

/-- The service's draw: `subset:k` over a population, and the units it drew. -/
structure SubsetDraw where
  k : Nat
  population : Nat
  units : Array Nat
  deriving DecidableEq, Repr

def subsetDrawOf? (j : Json) : Option SubsetDraw := do
  guard (keys j == some ["draw"])
  let d := get j "draw"
  guard (keys d == some ["k", "law", "population", "units"] && get d "law" == .str "subset")
  let .arr units := get d "units" | none
  pure ⟨← nat? (get d "k"), ← nat? (get d "population"), ← units.mapM nat?⟩

/-- What the service said. -/
inductive Said where
  | accept (profile : Core.Integrity) (draw : SubsetDraw)
  | reject (reason : String)
  | undecided (reason : String)

structure Evidence where
  protocol : String
  /-- The registration digest the answer is for, hex, or `none` for `null`. -/
  commitment : Option String
  said : Said

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  unless keys j == some answerKeys do throw s!"keys are not {", ".intercalate answerKeys}"
  let some protocol := str? (get j "protocol") | throw "protocol is not a string"
  let verdict ← match get j "verdict" with
    | .str "accept" => pure Core.Verdict.accept
    | .str "reject" => pure .reject
    | .str "undecided" => pure .undecided
    | _ => throw "verdict is not accept, reject or undecided"
  let commitment ← match get j "commitment" with
    | .null => pure none
    | .str s => pure (some s)
    | _ => throw "commitment is not null or a string"
  let shape := "an accepted answer has public outputs and a profile and no reason; any other has a reason and neither"
  match verdict, get j "reason", get j "public", get j "profile" with
  | .accept, .null, pub, prof =>
    if pub == .null || prof == .null then throw shape
    let some profile := integrityOf? prof | throw "profile is not an integrity profile"
    let some draw := subsetDrawOf? pub | throw "public is not {draw} with a subset draw {k, law, population, units}"
    pure ⟨protocol, commitment, .accept profile draw⟩
  | .reject, .str r, .null, .null => pure ⟨protocol, commitment, .reject r⟩
  | .undecided, .str r, .null, .null => pure ⟨protocol, commitment, .undecided r⟩
  | _, _, _, _ => throw shape

end PouwCheck
