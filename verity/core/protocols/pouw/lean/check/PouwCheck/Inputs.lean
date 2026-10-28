import CoreProtocol.Protocol
import CoreProtocol.Profile
import Flock.Draw
import Flock.WindowRecord
import PouwCheck.Json
import PouwCheck.Tile
import PouwWindow.Length

/-!
# The check's three inputs (`core_check.py`, "the three inputs")

* `Params`: the verifier's own parameters of the window: the check item, ε and δ, the grid's bits, the declared range
  of tile counts, the window's context and the check query's SHA-512; its template one the verifier pins (`tiles`), with
  that template's tile work and query; and the rows' fixed length: stop mode `fixed` and each slot's max_tokens.
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
def paramsKeys : List String := ["check", "context", "delta", "eps", "grid_bits", "length", "query", "range"]
def lengthParamKeys : List String := ["max_tokens", "stop"]
def checkKeys : List String := ["entries", "scheme", "template", "tile_work"]
/-- `window.COINS` and `window.MODES`: the check item's optional key and the coin modes it may name. -/
def coinsKey : String := "coins"
def coinModes : List String := ["live", "seeded"]
/-- `core_check.LIVE_PARAM`, `LIVE` and `LIVE_PARAMS`: a live window's parameters' salt session inputs. -/
def liveKey : String := "live"
def liveKeys : List String := ["key", "ledger", "main", "pins", "session"]
def LIVE_PARAMS : String :=
  "live is exactly when check.coins is live, and is then {key, ledger, main, pins, session}: the pinned challenger " ++
    "key, the auditor's ledger, the host's pins log, main's first-parent history and the salt session's files"

/-- The `check` item: the tile template, its work per tile, the commitment scheme, the public bound on entries, and the
window's coin mode when the item names one (`window.mode`: seeded when it names none). -/
structure CheckItem where
  entries : Nat
  scheme : String
  template : String
  tileWork : Nat
  coins : Option String := none
  deriving DecidableEq, Repr

def CheckItem.toJson (c : CheckItem) : Json :=
  let item : List (String × Json) :=
    [("entries", c.entries), ("scheme", c.scheme), ("template", c.template), ("tile_work", c.tileWork)]
  Json.mkObj (match c.coins with
    | some m => (coinsKey, Json.str m) :: item
    | none => item)

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
  /-- Each slot's max_tokens, which a row of record serves whatever it samples (stop mode `fixed`, `ignore_eos`). -/
  maxTokens : List Nat
  /-- A live window's salt session's inputs as its verifier holds them (`liveKeys`); `none` for any other window. -/
  live : Option Json := none

def isHexDigest (s : String) : Bool := s.length = 128 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

/-- A ratio-string field and its range, in `params`' order. -/
def ratioField (j : Json) (k : String) (ok : Core.Ratio → Bool) (what : String) : V Core.Ratio := do
  let some s := str? (get j k) | throw s!"{k} is not a string"
  let r ← Core.Ratio.ofString s k
  unless ok r do throw s!"{k} is not in {what}"
  pure r

def paramsOf (j : Json) : V Params := do
  unless (keys j).map (·.filter (· != liveKey)) == some paramsKeys do
    throw s!"keys are not {", ".intercalate paramsKeys}, and optionally {liveKey}"
  let c := get j "check"
  unless (keys c).map (·.filter (· != coinsKey)) == some checkKeys do
    throw s!"check's keys are not {", ".intercalate checkKeys}, and optionally {coinsKey}"
  let coins : Option String ← match PouwWindow.field? c coinsKey with
    | none => pure none
    | some (.str s) => if coinModes.contains s then pure (some s) else throw s!"check.{coinsKey} is not live or seeded"
    | some _ => throw s!"check.{coinsKey} is not live or seeded"
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
  let some tile := tile? template | throw "check.template is not a tile template the verifier pins"
  unless tile.tileWork == tileWork do throw "check.tile_work is not its pinned template's tile work"
  unless tile.query == query do throw "query is not its pinned template's query digest"
  let ln := get j "length"
  let maxTokens : Option (List Nat) := do
    guard (keys ln == some lengthParamKeys && get ln "stop" == .str PouwWindow.stopFixed)
    let m ← PouwWindow.nats? (get ln "max_tokens")
    guard (!m.isEmpty && m.all (1 ≤ ·))
    pure m
  let some maxTokens := maxTokens
    | throw "length is not {max_tokens, stop} with stop fixed and max_tokens a non-empty list of positive natural numbers"
  let live := PouwWindow.field? j liveKey
  unless (coins == some "live") == live.isSome && live.all (fun l => keys l == some liveKeys) do throw LIVE_PARAMS
  pure { check := { entries, scheme, template, tileWork, coins }, context, delta, eps, gridBits, query, lo, hi, maxTokens,
         live }

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

/-- `core_check.WINDOW_VALUES`: the window's optional groups of committed values, the activations it committed before
its salt and the registered weights' entries (`valuesRoot?`). -/
def WINDOW_VALUES : List String := ["activations", "registered"]

/-- The record's `window` item, its execution block: its kind, its context, the items it registers, and its groups of
committed values (`WINDOW_VALUES`), each `some` exactly when the item has the key, whatever its value. -/
structure ExecutionBlockItem where
  kind : String
  context : String
  items : Json
  activations : Option Json
  registered : Option Json

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
  window : ExecutionBlockItem

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
  let window : Option ExecutionBlockItem := do
    guard ((keys w).map (·.filter (!WINDOW_VALUES.contains ·)) == some ["context", "kind", "public"])
    let items := get w "public"
    guard (keys items).isSome
    pure ⟨← str? (get w "kind"), ← str? (get w "context"), items, PouwWindow.field? w "activations",
      PouwWindow.field? w "registered"⟩
  let some window := window
    | throw ("window is not {context, kind, public}, with strings and an object, and optionally activations " ++
        "and registered")
  pure { bytes := b, format, program, query, partition, scheme, population, law, roots, window }

/-- `registration.digest`: the window record's digest (`Flock.WindowRecord.digestOf`) of its format and the canonical
bytes it was read from. -/
def Registration.digest (c : Registration) : ByteArray := Flock.WindowRecord.digestOf c.format c.bytes

/-! The service's draw, as one-stage's `draw.py` writes it: a `Flock.Draw.UnitDraw`, decoded with this check's own
refusals (`core_check.py`'s `unit_draw`), which read only its keys and leave its law to `outcome`. -/

deriving instance DecidableEq for Flock.Draw.Stratum, Flock.Draw.WorkStratum, Flock.Draw.Law

/-- `parse_law`'s spelling. -/
def lawName : Flock.Draw.Law → String
  | .subset k => s!"subset:{k}"
  | .bernoulli a b => s!"bernoulli:{a}/{b}"
  | .stratified k _ => s!"stratified:{k}"
  | .work k _ => s!"work:{k}"

def keysAre (j : Json) (ks : List String) (ctx : String) : V Unit :=
  unless keys j == some ks do throw s!"{ctx}: keys are not {", ".intercalate ks}"

def natsOf (j : Json) (k ctx : String) : V (Array Nat) := do
  (← Flock.J.arr j k ctx).mapM (Flock.J.asNat · s!"{ctx}.{k}")

def rangeOf (j : Json) (ctx : String) : V (Nat × Nat) :=
  match j with
  | .arr #[lo, hi] => do pure (← Flock.J.asNat lo ctx, ← Flock.J.asNat hi ctx)
  | _ => throw s!"{ctx}: not a range [lo, hi]"

/-- A stratum's template, units and `k`, and a work stratum's work and floor. -/
def stratumOf (work : Bool) (j : Json) (ctx : String) : V (Flock.Draw.Stratum × Nat × Nat) := do
  keysAre j (if work then ["floor", "k", "template", "units", "work"] else ["k", "template", "units"]) ctx
  let template ← Flock.J.str j "template" ctx
  let units ← (← Flock.J.arr j "units" ctx).mapM (rangeOf · s!"{ctx}.units")
  let k ← Flock.J.nat j "k" ctx
  let (w, f) ← if work then do pure (← Flock.J.nat j "work" ctx, ← Flock.J.nat j "floor" ctx) else pure (0, 0)
  pure (⟨template, units, k⟩, w, f)

def strataOf (work : Bool) (j : Json) (ctx : String) : V (Array (Flock.Draw.Stratum × Nat × Nat)) := do
  let mut strata := #[]
  for s in ← Flock.J.arr j "strata" ctx do
    strata := strata.push (← stratumOf work s s!"{ctx}.strata[{strata.size}]")
  pure strata

/-- A bernoulli draw's `p`: `num/den`, each a natural number in canonical decimal, so `lawName` writes it as written. -/
def pOf (s ctx : String) : V (Nat × Nat) :=
  let canon (t : String) : Option Nat := t.toNat?.bind fun n => if toString n == t then some n else none
  match s.splitOn "/" with
  | [a, b] => match canon a, canon b with
    | some a, some b => pure (a, b)
    | _, _ => throw s!"{ctx}.p: not num/den"
  | _ => throw s!"{ctx}.p: not num/den"

/-- The public outputs `{"draw": d}`, `d` one of `draw.py`'s laws with exactly its keys (`check_draw`'s), and optionally a
live window's `salt` (`saltOf`). -/
def drawOf (j : Json) : V Flock.Draw.UnitDraw := do
  unless keys j == some ["draw"] || keys j == some ["draw", "salt"] do
    throw "public: keys are not draw, and optionally salt"
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
  let (l, closure) ← match law with
    | "subset" => do pure (.subset (← Flock.J.nat d "k" ctx), #[])
    | "bernoulli" => do let (a, b) ← pOf (← Flock.J.str d "p" ctx) ctx; pure (.bernoulli a b, #[])
    | "stratified" => do
      pure (.stratified (← Flock.J.nat d "k" ctx) ((← strataOf false d ctx).map (·.1)), #[])
    | _ => do
      let k ← Flock.J.nat d "k" ctx
      let strata ← strataOf true d ctx
      let closure ← if Flock.J.has d "closure" then natsOf d "closure" ctx else pure #[]
      pure (.work k (strata.map fun (s, work, floor) => { toStratum := s, work, floor }), closure)
  pure ⟨l, population, ← natsOf d "units" ctx, closure⟩

/-- A live window's `salt` among the public outputs, an object (`windowSaltReason` reads it), or `none`. -/
def saltOf (j : Json) : V (Option Json) :=
  match PouwWindow.field? j "salt" with
  | none => pure none
  | some s@(.obj _) => pure (some s)
  | some _ => throw "public.salt: not an object"

/-- What the service said. -/
inductive Said where
  | accept (profile : Core.Integrity) (draw : Flock.Draw.UnitDraw)
  | reject (reason : String)
  | undecided (reason : String)

structure Evidence where
  protocol : String
  /-- The registration digest the answer is for, hex, or `none` for `null`. -/
  commitment : Option String
  said : Said
  /-- On acceptance, the salt among its public outputs (`saltOf`). -/
  salt : Option Json := none

/-- `Core.Answer.ofJson`, then on acceptance `Core.Integrity.ofJson` of its profile and `drawOf` of its public outputs
(which `Answer.ofJson` has already found present, as it has the reason of any other answer). -/
def evidenceOf (b : ByteArray) : V Evidence := do
  let a ← Core.Answer.ofJson (← Core.canonicalJson b)
  let (said, salt) ← match a.verdict with
    | .accept => do
      let prof ← Core.Integrity.ofJson (a.profile.getD .null)
      let draw ← drawOf (a.published.getD .null)
      pure (Said.accept prof draw, ← saltOf (a.published.getD .null))
    | .reject => pure (Said.reject (a.reason.getD ""), none)
    | .undecided => pure (Said.undecided (a.reason.getD ""), none)
  pure ⟨a.protocol, a.digest.map Flock.hexOf, said, salt⟩

end PouwCheck
