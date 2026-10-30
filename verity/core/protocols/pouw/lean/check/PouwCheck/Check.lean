import CoreProtocol.Protocol
import PouwWindow.Grid
import PouwWindow.Draw
import PouwWindow.Public
import PouwWindow.Length
import PouwCheck.Inputs
import PouwCheck.WindowSalt
import Identity.Domains

/-!
# PoUW's rule (`core_check.py`, "the check")

`registered` holds the registration to PoUW's rule: a served window in the verifier's context registering exactly
`window.REGISTERED` (a registration of `window.LEGACY_REGISTERED`, made before the fixed-length rule, is not of record),
its `length` item a fixed-length window's at the verifier's max_tokens (`window.length_problem`: no EOS-keyed row, each
slot served its max_tokens), a work level whose tiles lie in the declared range, each item the verifier's own at that level, the
population N = point(level + 1), the law subset:K′ with K′ the window's exact `k_prime`, and the check's query and
scheme. `outcome` then holds the service's answer to it: one-stage's, for this registration's digest, and on acceptance
a profile and draw of the registered partition's N units drawn subset:K′ at the window's δ, and a live window's salt its
live coin's (`windowSaltReason`; a seeded window's answer carries none). Every refusal is the first rule it breaks, in
this order, with `core_check.py`'s reason.

The service's reject is a reject and its undecided is undecided. On acceptance PoUW publishes `window.PUBLIC` and the
profile is the service's `Core.Integrity`, passed through.
-/

namespace PouwCheck

open Lean (Json)
open PouwWindow (Work Sampling)

def oneStage : String := "verity/sampled-proofs/one-stage/v1"

/-- The checked level's name in the service's profile. -/
def unitLevel : String := "unit"

def toRat (r : Core.Ratio) : Rat := mkRat r.num r.den

/-- `sorted(window.REGISTERED)`. -/
def registeredSorted : List String := (PouwWindow.registered.toArray.qsort (· < ·)).toList

/-- `sorted(window.LEGACY_REGISTERED)`. -/
def legacySorted : List String := (PouwWindow.legacyRegistered.toArray.qsort (· < ·)).toList

/-- What a registration that keeps PoUW's rule fixes: the level's work, the sampling item and K′, and `root_W`. -/
structure Fixed where
  work : Work
  sampling : Sampling
  rootW : Option String
  deriving DecidableEq, Repr

def rootWJson : Option String → Json
  | some r => .str r
  | none => .null

/-- The registered items as the verifier computes them at its level (`expected_items`), in sorted order. -/
def expected (p : Params) (f : Fixed) : List (String × Json) :=
  [("check", p.check.toJson), ("root_W", rootWJson f.rootW), ("sampling", f.sampling.toJson),
   ("work", f.work.itemJson p.check.tileWork)]

def subsetLaw (k : Nat) : Json := Json.mkObj [("k", k), ("law", "subset")]

/-- The registered work level. -/
def levelOf (items : Json) : Option Nat := PouwWindow.field? (get items "work") "level" >>= nat?

/-- The items that differ from the verifier's, by their canonical strings. -/
def wrongItems (p : Params) (items : Json) (f : Fixed) : List String :=
  ((expected p f).filter fun (k, v) => cs (get items k) ≠ cs v).map (·.1)

/-- PoUW's rule over the registration. -/
def registered (p : Params) (c : Registration) : Except String Fixed :=
  let win := c.window
  let ks := (keys win.items).getD []
  let bits := p.gridBits
  if ¬ (win.kind = "served" ∧ win.context = p.context) then
    .error s!"window: registered kind {win.kind} in context {win.context}, the verifier's served in {p.context}"
  else if ks ≠ registeredSorted then
    if ks = legacySorted then
      .error "window-form: no-fixed-length: registered before the fixed-length rule (19 Oct), so not of record"
    else .error s!"public-items: registered {", ".intercalate ks}, declared {", ".intercalate registeredSorted}"
  else match PouwWindow.lengthProblem (get win.items "length") p.maxTokens with
  | some r => .error r
  | none => match levelOf win.items with
  | none => .error "level: the registration names no work level"
  | some ell =>
    if PouwWindow.level bits (PouwWindow.floorPoint bits p.hi) < ell + 1 ∨ PouwWindow.point bits ell < p.lo then
      .error s!"level: level {ell} is outside the window's range {p.lo}..{p.hi}"
    else
      let work := Work.at bits ell
      match PouwWindow.sampling (toRat p.eps) (toRat p.delta) work p.check.tileWork bits with
      | .error e => .error e
      | .ok s =>
        let f : Fixed := ⟨work, s, (c.roots.find? (·.port = "W")).map (·.root)⟩
        let wrong := wrongItems p win.items f
        if wrong ≠ [] then .error s!"public-value: {", ".intercalate wrong}"
        else if c.population ≠ work.n then .error s!"population: registered {c.population}, the level's {work.n}"
        else if cs c.law ≠ cs (subsetLaw s.k) then
          .error s!"law: registered {cs c.law}, the window's {cs (subsetLaw s.k)}"
        else if c.query ≠ p.query then .error s!"query: registered {c.query}, the check's {p.query}"
        else if c.scheme ≠ p.check.scheme then .error s!"scheme: registered {c.scheme}, the check's {p.check.scheme}"
        else .ok f

/-- What an accepted window makes public: `window.PUBLIC`, its registered items at the verifier's values. -/
structure Public where
  check : CheckItem
  work : Work
  sampling : Sampling
  maxTokens : List Nat
  rootW : Option String
  drawn : Array Nat
  deriving DecidableEq, Repr

/-- The published items in `window.PUBLIC`'s order. -/
def Public.entries (pub : Public) : List (String × Json) :=
  [("verdict", true), ("work", pub.work.itemJson pub.check.tileWork), ("check", pub.check.toJson),
   ("sampling", pub.sampling.toJson), ("length", PouwWindow.lengthItem pub.maxTokens),
   ("drawn", Json.arr (pub.drawn.map fun (u : Nat) => Json.num u)),
   ("root_W", rootWJson pub.rootW)]

def Public.toJson (pub : Public) : Json := Json.mkObj pub.entries

/-- Strictly ascending. -/
def Ascending : List Nat → Prop
  | a :: b :: t => a < b ∧ Ascending (b :: t)
  | _ => True

instance instDecidableAscending : (l : List Nat) → Decidable (Ascending l)
  | [] => isTrue trivial
  | [_] => isTrue trivial
  | a :: b :: t =>
    have := instDecidableAscending (b :: t)
    inferInstanceAs (Decidable (a < b ∧ Ascending (b :: t)))

/-- The draw's units: K′ distinct units below N, ascending. -/
def UnitsOk (units : Array Nat) (k n : Nat) : Prop :=
  units.size = k ∧ Ascending units.toList ∧ ∀ u ∈ units.toList, u < n

instance instDecidableUnitsOk (units : Array Nat) (k n : Nat) : Decidable (UnitsOk units k n) := by
  unfold UnitsOk; infer_instance

def drawnJson (drawn : Array (String × Nat)) : Json := Json.mkObj (drawn.toList.map fun (k, v) => (k, Json.num v))

/-- The service's answer to a registration that keeps PoUW's rule. -/
def outcome (p : Params) (c : Registration) (f : Fixed) (e : Evidence) : Core.Outcome Public Core.Integrity :=
  let d := Flock.hexOf c.digest
  let k := f.sampling.k
  let n := f.work.n
  if e.protocol ≠ oneStage then .reject s!"outcome: protocol {e.protocol}, not one-stage's {oneStage}"
  else if e.commitment ≠ some d then
    let shown := match e.commitment with
      | some s => if s.isEmpty then "null" else s
      | none => "null"
    .reject s!"outcome: for registration {shown}, not this one {d}"
  else match e.said with
  | .reject r => .reject s!"service: {r}"
  | .undecided r => .undecided s!"service: {r}"
  | .accept prof draw =>
    if prof.programs ≠ #[c.program] then
      .reject s!"profile: programs {", ".intercalate prof.programs.toList}, not the registered {c.program}"
    else if prof.levels ≠ #[{ name := unitLevel, query := some c.partition, size := n, draw := some (.subset k) }] then
      .reject s!"profile: its levels are not the registered partition's {n} units drawn subset:{k}"
    else if prof.delta ≠ p.delta then
      .reject s!"profile: delta {prof.delta.toString}, not the window's {p.delta.toString}"
    else if prof.drawn ≠ #[(unitLevel, k)] then
      .reject s!"profile: drawn {cs (drawnJson prof.drawn)}, not {k} units"
    else if draw.law ≠ .subset k ∨ draw.population ≠ n then
      .reject s!"draw: {lawName draw.law} over {draw.population}, not subset:{k} over {n}"
    else if ¬ UnitsOk draw.units k n then
      .reject s!"draw: its {draw.units.size} units are not {k} ascending distinct units below {n}"
    else
      let why := windowSaltReason p c e.salt
      if why.isSome then .reject s!"salt: {why.getD ""}"
      else .accept ⟨p.check, f.work, f.sampling, p.maxTokens, f.rootW, draw.units⟩ prof

def check (p : Params) (c : Registration) (e : Evidence) : Core.Outcome Public Core.Integrity :=
  match registered p c with
  | .error r => .reject r
  | .ok f => outcome p c f e

/-! Each separator below keeps its literal while a lock reads its definition, whose record hashes the value as
written: tied here to its entry of the table (`Identity.Domains`), it reads the entry at its next record change. -/
example : oneStage = Domains.ONE_STAGE_PROTOCOL := rfl

end PouwCheck
