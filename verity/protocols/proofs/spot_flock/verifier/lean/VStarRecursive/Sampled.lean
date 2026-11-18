/-
SPOT (`verity/sampled-proofs/v1`), a `Core.BlobProtocol`: one-stage v2's audit on the proofs themselves
(`Spot.OneStageZk`). One-stage v1 is refused (`V1_REFUSED`): its verdicts are the evidence's, and this audit's
are its own run of the verifier. The recursive audit (`verity/sampled-proofs/recursive/v1`) is refused too, its params
and its evidence alike, by name (`RECURSIVE_REFUSED`), and none of its types is a case here; the
two-stage audit (`VStar.Recursive.SampledTwo`) is read and checked, then accepted while its switch
(`SampledTwo.FORM`) is on: SPOT's soundness covers the one-stage form only, and the verifier accepts the two-stage form
ahead of its proof.

`Core.Composition.product` reads every component's case, so SPOT is one component and the form is the params':

* **params**: `{"form": "one-stage" | "recursive" | "two-stage", ...}`, the rest that form's params. The auditor fixes
  the form with the rest of its params, before any coin. One-stage params naming no `proofs` are v1's, and refused; those
  naming one are v2's, the one statement's or the per-member form's (`OneStageZk.paramsEitherOfJson`); the recursive form
  is refused before its rest is read; the two-stage form reads the whole object, its `form` included
  (`SampledTwo.paramsOfJson`);
* **commitment**: the registration record (`OneStage.commitmentOf`), as every form reads it;
* **evidence**: `{"form", ...}`, the rest that form's evidence; the two-stage form reads the evidence's whole bytes
  (`SampledTwo.evidenceOf`). One-stage evidence naming no `proofs` is v1's, and refused; evidence naming one is v2's, in
  either of its forms (`OneStageZk.evidenceEitherOf`); recursive evidence is refused before its rest is read;
* **check**: that form's check on the blobs, its acceptance held to the caller's check (`Caller.gate`) against that
  form's verifier; evidence of another form than the params' rejects, so only one-stage v2's check and the two-stage
  form's run, and the two-stage form's check refuses a run only where its switch is off, which it is not.

The recursive audit's own command, `vstar-recursive` with any first word but `sampled`, is dispatched by the
executable's root, not here.
-/
module

public import VStarRecursive.SampledTwo
meta import VStarRecursive.SampledTwo

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace VStar.Recursive.Sampled

open Lean (Json)
open Flock (V)
open Spot

def PROTOCOL : String := "verity/sampled-proofs/v1"

def ONE_STAGE : String := "one-stage"
def RECURSIVE : String := "recursive"

/-- The refusal of a one-stage statement naming no `proofs`, one-stage v1's. -/
def V1_REFUSED : String :=
  s!"one-stage v1 ({Docket.PROTOCOL}) is refused: SPOT's one-stage form is v2 " ++
    s!"({OneStageZk.PROTOCOL}), whose verdicts are its own run of the verifier on the proofs"

/-- The refusal of the recursive form: SPOT's soundness covers the one-stage form only, and the verifier refuses
every form its soundness doesn't cover. -/
def RECURSIVE_REFUSED : String :=
  s!"the recursive audit (verity/sampled-proofs/recursive/v1) is refused: SPOT's soundness covers one-stage v2 " ++
    s!"({OneStageZk.PROTOCOL}) only"

abbrev Params := OneStageZk.Params ⊕ SampledTwo.Params
abbrev Evidence := OneStageZk.Evidence ⊕ SampledTwo.Evidence

/-- The forms, as a refusal names them. -/
def FORMS : String := s!"\"{ONE_STAGE}\", \"{RECURSIVE}\" or \"{SampledTwo.TWO_STAGE}\""

/-- The keys only two levels' parameters hold: the boundary docket, the interior and the two-level rule. -/
def TWO_LEVEL_KEYS : List String := ["boundary", "interior", "rule"]

/-- The form a `levels` field names: `1` one level's, `2` two levels'; any other count refused by name. -/
def levelsForm (v : Json) (what : String) : V String :=
  match v with
  | .num n =>
    if n.exponent != 0 then throw s!"levels-not-integer: the {what}' levels is {v.compress}, not a whole number of levels"
    else if n.mantissa == 1 then pure ONE_STAGE
    else if n.mantissa == 2 then pure SampledTwo.TWO_STAGE
    else if n.mantissa < 1 then throw s!"levels-not-positive: the {what}' levels is {v.compress}: an audit samples at least one level"
    else throw s!"levels-not-built: the {what}' levels is {v.compress}: only 1 or 2 levels are built"
  | _ => throw s!"levels-not-integer: the {what}' levels is {v.compress}, not a whole number of levels"

/-- The form the parameters' shape is: two levels' keys (`TWO_LEVEL_KEYS`) all or none. -/
def shapeForm (j : Json) : V String :=
  let held := TWO_LEVEL_KEYS.filter (fun k => (field? j k).isSome)
  if held.isEmpty then pure ONE_STAGE
  else if held.length == TWO_LEVEL_KEYS.length then pure SampledTwo.TWO_STAGE
  else throw s!"levels-shape: the parameters hold {held} of two levels' {TWO_LEVEL_KEYS}: neither shape"

/-- The form an object names and its rest, less `form` and `levels`. `levels` and `form`, where both are present, must
name one form, and an object naming neither is one level's. The parameters' (``params``) one- or two-level form must be
their shape's (`shapeForm`): the count is the parameters', fixed before any coin, and never a record's. -/
private def split (j : Json) (what : String) (params : Bool) : V (String × Json) := do
  let .obj kvs := j | throw s!"the {what} is not an object"
  let byLevels ← match field? j "levels" with
    | none => pure none
    | some v => some <$> levelsForm v what
  let form ← match getD j "form", byLevels with
    | .str f, some l =>
      if f == l then pure f
      else throw s!"levels-form-mismatch: the {what}' levels is {(getD j "levels").compress}, and their form is {f}"
    | .str f, none => pure f
    | .null, some l => pure l
    | .null, none => pure ONE_STAGE
    | v, _ => throw s!"the {what}' form is {v.compress}, not a string ({FORMS})"
  if params && (form == ONE_STAGE || form == SampledTwo.TWO_STAGE) then
    let s ← shapeForm j
    if s != form then throw s!"levels-params-mismatch: the parameters name the form {form}, and their shape is {s}'s"
  pure (form, Json.mkObj (kvs.toArray.filter (fun kv => kv.1 != "form" && kv.1 != "levels")).toList)

def paramsOfJson (j : Json) : V Params := do
  let (form, rest) ← split j "parameters" true
  if form == ONE_STAGE then
    if (field? rest "proofs").isNone then throw s!"the parameters name no proofs: {V1_REFUSED}"
    return .inl (← OneStageZk.paramsEitherOfJson rest)
  if form == RECURSIVE then throw RECURSIVE_REFUSED
  if form == SampledTwo.TWO_STAGE then return .inr (← SampledTwo.paramsOfJson j)
  throw s!"the parameters' form {form} is not {FORMS}"

def evidenceOf (b : ByteArray) : V Evidence := do
  let (form, rest) ← split (← Core.canonicalJson b) "evidence" false
  if form == RECURSIVE then throw RECURSIVE_REFUSED
  if form == SampledTwo.TWO_STAGE then return .inr (← SampledTwo.evidenceOf b)
  if form == ONE_STAGE && (field? rest "proofs").isNone then throw s!"the evidence names no proofs: {V1_REFUSED}"
  let rest := (← Flock.canon rest).toUTF8
  if form == ONE_STAGE then return .inl (← OneStageZk.evidenceEitherOf rest)
  throw s!"the evidence's form {form} is not {FORMS}"

/-- The form of a case of the parameters or the evidence. -/
def formOf {α β : Type} : α ⊕ β → String
  | .inl _ => ONE_STAGE
  | .inr _ => SampledTwo.TWO_STAGE

/-- The refusal of evidence of the form `ef` under parameters of the form `pf`. -/
def MISMATCH (pf ef : String) : String := s!"the evidence's form is {ef}, and the parameters' is {pf}"

/-- The name of a two-level accept's coarse level: the replay blocks. -/
def BOUNDARY_LEVEL : String := "replay"

/-- How many members a sampling design samples. -/
def designCount : Core.SamplingDesign → Nat
  | .subset k => k
  | .stratified ss => ss.foldl (fun n s => n + s.k) 0
  | .work k _ => k
  | .bernoulli _ => 0

/-- A two-level accept's coarse level and how many of its members are sampled: the replay blocks, the boundary
registration's population (`c`) under the replay grain's query, at the first stage's sampling design as the run of
record derives it (`Select.Rule.samplingDesignOver`, `Draw.derive`). None where that design has no profile. -/
def boundaryLevel (p : SampledTwo.Params) (c : OneStage.Commitment) : Option (Core.Level × Nat) := do
  let v := p.boundary.verifier
  let n ← nat? (getD c.record "population")
  let rule ← p.rule.over n (Select.strataSizes v.program v.query)
  let design ← (rule.samplingDesignOver n).toOption
  let table ← (p.rule.tableOver v.program v.query v.workTable).toOption
  let draw ← Record.profileSample (← (Draw.derive v.program v.query design table).toOption)
  let partition := match getD c.record "partition" with | .str s => s | _ => ""
  pure ({ name := BOUNDARY_LEVEL, query := some partition, size := n, draw := some draw }, designCount draw)

/-- A two-stage accept with its coarse level (`boundaryLevel`) before the interior's, coarse to fine. -/
def withBoundary (p : SampledTwo.Params) (c : OneStage.Commitment) :
    Core.Outcome OneStage.Public Core.Integrity → Core.Outcome OneStage.Public Core.Integrity
  | .accept pub prof =>
    match boundaryLevel p c with
    | some (l, k) => .accept pub { prof with levels := #[l] ++ prof.levels, drawn := #[(l.name, k)] ++ prof.drawn }
    | none => .reject "two-stage: the boundary's sampling design has no profile"
  | o => o

theorem withBoundary_accept {p : SampledTwo.Params} {c : OneStage.Commitment}
    {o : Core.Outcome OneStage.Public Core.Integrity} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : withBoundary p c o = .accept pub prof) : ∃ prof', o = .accept pub prof' := by
  cases o with
  | accept pub' prof' =>
    cases hb : boundaryLevel p c with
    | some lk =>
      obtain ⟨l, k⟩ := lk
      simp only [withBoundary, hb, Core.Outcome.accept.injEq] at h
      exact ⟨prof', by rw [h.1]⟩
    | none => simp [withBoundary, hb] at h
  | reject s => simp [withBoundary] at h
  | undecided s => simp [withBoundary] at h

/-- The refusal of an accept whose profile lists `got` levels where the parameters' form is `n` levels'. -/
def LEVELS_PROFILE (n got : Nat) : String :=
  s!"levels-profile-mismatch: the parameters' form is {n} level(s), and the accept's profile lists {got}"

/-- An accept whose profile's levels list is `n` long; any other refused (`LEVELS_PROFILE`). -/
def levelsHeld (n : Nat) : Core.Outcome OneStage.Public Core.Integrity → Core.Outcome OneStage.Public Core.Integrity
  | .accept pub prof => if prof.levels.size = n then .accept pub prof else .reject (LEVELS_PROFILE n prof.levels.size)
  | o => o

theorem levelsHeld_accept {n : Nat} {o : Core.Outcome OneStage.Public Core.Integrity} {pub : OneStage.Public}
    {prof : Core.Integrity} (h : levelsHeld n o = .accept pub prof) : o = .accept pub prof ∧ prof.levels.size = n := by
  cases o with
  | accept pub' prof' =>
    by_cases hs : prof'.levels.size = n
    · rw [levelsHeld, if_pos hs, Core.Outcome.accept.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨rfl, hs⟩
    · simp [levelsHeld, hs] at h
  | reject s => simp [levelsHeld] at h
  | undecided s => simp [levelsHeld] at h

/-- That form's check, its accept's profile listing that form's levels (`levelsHeld`): one, or two for the two-stage
form, whose coarse level is the boundary's (`withBoundary`); evidence of another form than the params' rejects
(`MISMATCH`). -/
def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match p, e with
  | .inl p, .inl e => levelsHeld 1 (Caller.gate c.record p.base.verifier (OneStageZk.checkBlobs p c e bs))
  | .inr p, .inr e => levelsHeld 2 (withBoundary p c (SampledTwo.checkBlobs p c e bs))
  | p, e => .reject (MISMATCH (formOf p) (formOf e))

def protocol : Core.BlobProtocol where
  id := PROTOCOL
  Params := Params
  Commitment := OneStage.Commitment
  Evidence := Evidence
  Public := OneStage.Public
  Profile := Core.Integrity
  params := paramsOfJson
  commitment := OneStage.commitmentOf
  evidence := evidenceOf
  digest c := c.digest
  check p c e := checkBlobs p c e Core.Blobs.empty
  publicJson := OneStage.publicJson
  profileJson := Core.Integrity.toJson
  checkBlobs := checkBlobs

#guard match paramsOfJson (Json.mkObj [("form", "other")]) with
  | .error e => e == s!"the parameters' form other is not {FORMS}"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("form", SampledTwo.TWO_STAGE)]) with
  | .error e => e.startsWith "levels-params-mismatch"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("form", SampledTwo.TWO_STAGE), ("boundary", Json.mkObj []),
    ("interior", Json.mkObj []), ("rule", Json.mkObj [])]) with
  | .error e => e.startsWith "the keys of the parameters"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("levels", Json.num 2), ("boundary", Json.mkObj []),
    ("interior", Json.mkObj []), ("rule", Json.mkObj [])]) with
  | .error e => e.startsWith "the keys of the parameters"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("levels", Json.num 2), ("form", ONE_STAGE)]) with
  | .error e => e == "levels-form-mismatch: the parameters' levels is 2, and their form is one-stage"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("levels", Json.num 3)]) with
  | .error e => e.startsWith "levels-not-built"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("levels", "2")]) with
  | .error e => e.startsWith "levels-not-integer"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("levels", Json.num 0)]) with
  | .error e => e.startsWith "levels-not-positive"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("levels", Json.num 1), ("boundary", Json.mkObj [])]) with
  | .error e => e.startsWith "levels-shape"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("values", Json.mkObj [])]) with
  | .error e => e == s!"the parameters name no proofs: {V1_REFUSED}"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("levels", Json.num 1), ("values", Json.mkObj [])]) with
  | .error e => e == s!"the parameters name no proofs: {V1_REFUSED}"
  | .ok _ => false
#guard match evidenceOf "{\"form\":\"two-stage\"}".toUTF8 with
  | .error e => e.startsWith "the keys of the evidence"
  | .ok _ => false
#guard match evidenceOf "{\"form\":\"recursive\"}".toUTF8 with
  | .error e => e == RECURSIVE_REFUSED
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("form", ONE_STAGE), ("values", Json.mkObj [])]) with
  | .error e => e == s!"the parameters name no proofs: {V1_REFUSED}"
  | .ok _ => false
#guard match evidenceOf "{\"form\":\"one-stage\",\"verdicts\":{}}".toUTF8 with
  | .error e => e == s!"the evidence names no proofs: {V1_REFUSED}"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("form", RECURSIVE), ("recursive", Json.mkObj [])]) with
  | .error e => e == RECURSIVE_REFUSED
  | .ok _ => false
#guard match levelsHeld 2 (.accept ⟨#[], #[], 0, none, true⟩
    { programs := #[], prescribed := #[], linked := false, covered := #[], delta := ⟨1, 2⟩, drawn := #[],
      levels := #[{ name := "unit", query := none, size := 4, draw := some (.subset 2) }] }) with
  | .reject r => r == "levels-profile-mismatch: the parameters' form is 2 level(s), and the accept's profile lists 1"
  | _ => false
#guard match levelsHeld 1 (.accept ⟨#[], #[], 0, none, true⟩
    { programs := #[], prescribed := #[], linked := false, covered := #[], delta := ⟨1, 2⟩, drawn := #[],
      levels := #[{ name := "unit", query := none, size := 4, draw := some (.subset 2) }] }) with
  | .accept _ _ => true
  | _ => false
#guard designCount (.stratified #[⟨"a", 3, 1⟩, ⟨"b", 5, 2⟩]) == 3

end VStar.Recursive.Sampled
