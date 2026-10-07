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
* `Evidence`: the proof service's answer for a registration, read by `Core.Answer.ofJson`: the protocol, the
  registration it is for, and what it said (on acceptance, the integrity profile and the verifier's draw as one-stage's
  `draw.py` writes it).

Each decoder refuses with `core_check.py`'s message, and `Core.Protocol.answer` prefixes it with the input's name.
-/

namespace PouwCheck

open Lean (Json)
open Flock (V)

def scheme : String := "frame-v3-sha512/hm96-sha512"
def formats : List String := ["verity/registration/v1", "verity/registration/v2"]
def paramsKeys : List String := ["check", "context", "delta", "eps", "grid_bits", "query", "range"]
def checkKeys : List String := ["entries", "scheme", "template", "tile_work"]

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

/-! The service's draw, as one-stage's `draw.py` writes it. -/

/-- A stratum of a stratified or work draw: its template, its units as `[lo, hi)` ranges, its `k`, and a work
stratum's work and floor. -/
structure DrawStratum where
  template : String
  units : Array (Nat × Nat)
  k : Nat
  work : Option (Nat × Nat)
  deriving DecidableEq, Repr

/-- The draw's law: `subset:k`, `bernoulli:p` (`p` as written), `stratified:k` or `work:k` with their strata, and a work
draw's closure. -/
inductive Law where
  | subset (k : Nat)
  | bernoulli (p : String)
  | stratified (k : Nat) (strata : Array DrawStratum)
  | work (k : Nat) (strata : Array DrawStratum) (closure : Option (Array Nat))
  deriving DecidableEq, Repr

/-- `parse_law`'s spelling. -/
def Law.name : Law → String
  | .subset k => s!"subset:{k}"
  | .bernoulli p => s!"bernoulli:{p}"
  | .stratified k _ => s!"stratified:{k}"
  | .work k _ _ => s!"work:{k}"

/-- The verifier's draw: its law, over a population, and the units it drew. -/
structure UnitDraw where
  law : Law
  population : Nat
  units : Array Nat
  deriving DecidableEq, Repr

def keysAre (j : Json) (ks : List String) (ctx : String) : V Unit :=
  unless keys j == some ks do throw s!"{ctx}: keys are not {", ".intercalate ks}"

def natsOf (j : Json) (k ctx : String) : V (Array Nat) := do
  (← Flock.J.arr j k ctx).mapM (Flock.J.asNat · s!"{ctx}.{k}")

def rangeOf (j : Json) (ctx : String) : V (Nat × Nat) :=
  match j with
  | .arr #[lo, hi] => do pure (← Flock.J.asNat lo ctx, ← Flock.J.asNat hi ctx)
  | _ => throw s!"{ctx}: not a range [lo, hi]"

def stratumOf (work : Bool) (j : Json) (ctx : String) : V DrawStratum := do
  keysAre j (if work then ["floor", "k", "template", "units", "work"] else ["k", "template", "units"]) ctx
  let template ← Flock.J.str j "template" ctx
  let units ← (← Flock.J.arr j "units" ctx).mapM (rangeOf · s!"{ctx}.units")
  let k ← Flock.J.nat j "k" ctx
  let w ← if work then do pure (some (← Flock.J.nat j "work" ctx, ← Flock.J.nat j "floor" ctx)) else pure none
  pure ⟨template, units, k, w⟩

def strataOf (work : Bool) (j : Json) (ctx : String) : V (Array DrawStratum) := do
  let mut strata := #[]
  for s in ← Flock.J.arr j "strata" ctx do
    strata := strata.push (← stratumOf work s s!"{ctx}.strata[{strata.size}]")
  pure strata

/-- The public outputs `{"draw": d}`, `d` one of `draw.py`'s laws with exactly its keys (`check_draw`'s). -/
def drawOf (j : Json) : V UnitDraw := do
  keysAre j ["draw"] "public"
  let d := get j "draw"
  let ctx := "public.draw"
  let law ← Flock.J.str d "law" ctx
  let ks ← match law with
    | "subset" => pure ["k", "law", "population", "units"]
    | "bernoulli" => pure ["law", "p", "population", "units"]
    | "stratified" => pure ["k", "law", "population", "strata", "units"]
    | "work" => pure ((if Flock.J.has d "closure" then ["closure"] else []) ++ ["k", "law", "population", "strata", "units"])
    | _ => throw s!"{ctx}.law: not subset, bernoulli, stratified or work"
  keysAre d ks ctx
  let population ← Flock.J.nat d "population" ctx
  let l ← match law with
    | "subset" => do pure (.subset (← Flock.J.nat d "k" ctx))
    | "bernoulli" => do pure (.bernoulli (← Flock.J.str d "p" ctx))
    | "stratified" => do pure (.stratified (← Flock.J.nat d "k" ctx) (← strataOf false d ctx))
    | _ => do
      let k ← Flock.J.nat d "k" ctx
      let strata ← strataOf true d ctx
      let closure ← if Flock.J.has d "closure" then some <$> natsOf d "closure" ctx else pure none
      pure (.work k strata closure)
  pure ⟨l, population, ← natsOf d "units" ctx⟩

/-- What the service said. -/
inductive Said where
  | accept (profile : Core.Integrity) (draw : UnitDraw)
  | reject (reason : String)
  | undecided (reason : String)

structure Evidence where
  protocol : String
  /-- The registration digest the answer is for, hex, or `none` for `null`. -/
  commitment : Option String
  said : Said

/-- `Core.Answer.ofJson`, then on acceptance `Core.Integrity.ofJson` of its profile and `drawOf` of its public outputs
(which `Answer.ofJson` has already found present, as it has the reason of any other answer). -/
def evidenceOf (b : ByteArray) : V Evidence := do
  let a ← Core.Answer.ofJson (← Core.canonicalJson b)
  let said ← match a.verdict with
    | .accept => do pure (.accept (← Core.Integrity.ofJson (a.profile.getD .null)) (← drawOf (a.published.getD .null)))
    | .reject => pure (.reject (a.reason.getD ""))
    | .undecided => pure (.undecided (a.reason.getD ""))
  pure ⟨a.protocol, a.digest.map Flock.hexOf, said⟩

end PouwCheck
