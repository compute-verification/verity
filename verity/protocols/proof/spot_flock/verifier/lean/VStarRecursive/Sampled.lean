/-
SPOT (`verity/sampled-proofs/v1`), a `Core.BlobProtocol` with one entry, whose case is as many levels as its `levels`
names: one level is one-stage v2's audit on the proofs themselves (`Spot.OneStageZk`), two levels the two-stage audit
(`VStar.Recursive.SampledTwo`), read and checked, and accepted ahead of its soundness proof; 3 or more are refused as not
built. One-stage v1 is refused (`V1_REFUSED`): its verdicts are the evidence's, and this audit's are its own run of the
verifier.

* **params**: `{"levels": 1 | 2, ...}`, one level where `levels` is absent, the rest that count's params. The auditor
  fixes the count with the rest of its params, before any coin, and it must be their shape's (`shapeLevels`). One-level
  params naming no `proofs` are v1's, and refused; those naming one are v2's, the one statement's or the per-member
  form's (`OneStageZk.paramsEitherOfJson`); two levels' read the whole object (`SampledTwo.paramsOfJson`);
* **commitment**: the registration record (`OneStage.commitmentOf`);
* **evidence**: `{"levels", ...}` likewise; two levels' read the evidence's whole bytes (`SampledTwo.evidenceOf`);
* **check**: that count's check on the blobs, its acceptance held to the caller's check (`Caller.gate`) and its profile
  listing that many levels (`levelsHeld`); evidence of another count than the params' rejects (`MISMATCH`).

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

/-- The refusal of a one-stage statement naming no `proofs`, one-stage v1's. -/
def V1_REFUSED : String :=
  s!"one-stage v1 ({Docket.PROTOCOL}) is refused: SPOT's one-stage form is v2 " ++
    s!"({OneStageZk.PROTOCOL}), whose verdicts are its own run of the verifier on the proofs"

abbrev Params := OneStageZk.Params ⊕ SampledTwo.Params
abbrev Evidence := OneStageZk.Evidence ⊕ SampledTwo.Evidence

/-- The keys only two levels' parameters hold: the boundary docket, the interior and the two-level rule. -/
def TWO_LEVEL_KEYS : List String := ["boundary", "interior", "rule"]

/-- The number of levels a `levels` field names, 1 or 2; any other count refused by name. -/
def levelsOf (v : Json) (what : String) : V Nat :=
  match v with
  | .num n =>
    if n.exponent != 0 then throw s!"levels-not-integer: the {what}' levels is {v.compress}, not a whole number of levels"
    else if n.mantissa == 1 then pure 1
    else if n.mantissa == 2 then pure 2
    else if n.mantissa < 1 then throw s!"levels-not-positive: the {what}' levels is {v.compress}: an audit samples at least one level"
    else throw s!"levels-not-built: the {what}' levels is {v.compress}: only 1 or 2 levels are built"
  | _ => throw s!"levels-not-integer: the {what}' levels is {v.compress}, not a whole number of levels"

/-- The number of levels the parameters' shape is: two levels' keys (`TWO_LEVEL_KEYS`) all or none. -/
def shapeLevels (j : Json) : V Nat :=
  let held := TWO_LEVEL_KEYS.filter (fun k => (field? j k).isSome)
  if held.isEmpty then pure 1
  else if held.length == TWO_LEVEL_KEYS.length then pure 2
  else throw s!"levels-shape: the parameters hold {held} of two levels' {TWO_LEVEL_KEYS}: neither shape"

/-- The levels an object names (`levels`, one where absent) and its rest, less `levels`. The parameters' (``params``)
levels must be their shape's (`shapeLevels`): the count is the parameters', fixed before any coin, and never a
record's. -/
private def split (j : Json) (what : String) (params : Bool) : V (Nat × Json) := do
  let .obj kvs := j | throw s!"the {what} is not an object"
  let n ← match field? j "levels" with
    | none => pure 1
    | some v => levelsOf v what
  if params then
    let s ← shapeLevels j
    if s != n then throw s!"levels-params-mismatch: the parameters name {n} level(s), and their shape is {s}'s"
  pure (n, Json.mkObj (kvs.toArray.filter (fun kv => kv.1 != "levels")).toList)

def paramsOfJson (j : Json) : V Params := do
  let (n, rest) ← split j "parameters" true
  if n == 2 then return .inr (← SampledTwo.paramsOfJson j)
  if (field? rest "proofs").isNone then throw s!"the parameters name no proofs: {V1_REFUSED}"
  return .inl (← OneStageZk.paramsEitherOfJson rest)

def evidenceOf (b : ByteArray) : V Evidence := do
  let (n, rest) ← split (← Core.canonicalJson b) "evidence" false
  if n == 2 then return .inr (← SampledTwo.evidenceOf b)
  if (field? rest "proofs").isNone then throw s!"the evidence names no proofs: {V1_REFUSED}"
  return .inl (← OneStageZk.evidenceEitherOf (← Flock.canon rest).toUTF8)

/-- The levels of a case of the parameters or the evidence. -/
def levelsOfCase {α β : Type} : α ⊕ β → Nat
  | .inl _ => 1
  | .inr _ => 2

/-- The refusal of evidence of `ef` levels under parameters of `pf`. -/
def MISMATCH (pf ef : Nat) : String := s!"the evidence is {ef} level(s), and the parameters {pf}"

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

/-- `prof` with level `l`, `k` of its members sampled, before its own levels, coarse to fine. -/
def coarser (l : Core.Level) (k : Nat) (prof : Core.Integrity) : Core.Integrity :=
  { prof with levels := #[l] ++ prof.levels, drawn := #[(l.name, k)] ++ prof.drawn }

/-- A two-stage accept with its coarse level (`boundaryLevel`) before the interior's, coarse to fine. -/
def withBoundary (p : SampledTwo.Params) (c : OneStage.Commitment) :
    Core.Outcome OneStage.Public Core.Integrity → Core.Outcome OneStage.Public Core.Integrity
  | .accept pub prof =>
    match boundaryLevel p c with
    | some (l, k) => .accept pub (coarser l k prof)
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

/-- The refusal of an accept whose profile lists `got` levels where the parameters name `n`. -/
def LEVELS_PROFILE (n got : Nat) : String :=
  s!"levels-profile-mismatch: the parameters name {n} level(s), and the accept's profile lists {got}"

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

/-- That count's check, its accept's profile listing that many levels (`levelsHeld`): one, or two, whose coarse
level is the boundary's (`withBoundary`); evidence of another count than the params' rejects
(`MISMATCH`). -/
def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match p, e with
  | .inl p, .inl e => levelsHeld 1 (Caller.gate c.record p.base.verifier (OneStageZk.checkBlobs p c e bs))
  | .inr p, .inr e => levelsHeld 2 (withBoundary p c (SampledTwo.checkBlobs p c e bs))
  | p, e => .reject (MISMATCH (levelsOfCase p) (levelsOfCase e))

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

#guard match paramsOfJson (Json.mkObj [("levels", Json.num 2)]) with
  | .error e => e.startsWith "levels-params-mismatch"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("levels", Json.num 2), ("boundary", Json.mkObj []),
    ("interior", Json.mkObj []), ("rule", Json.mkObj [])]) with
  | .error e => e.startsWith "the keys of the parameters"
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
#guard match evidenceOf "{\"levels\":2}".toUTF8 with
  | .error e => e.startsWith "the keys of the evidence"
  | .ok _ => false
#guard match evidenceOf "{\"verdicts\":{}}".toUTF8 with
  | .error e => e == s!"the evidence names no proofs: {V1_REFUSED}"
  | .ok _ => false
#guard match levelsHeld 2 (.accept ⟨#[], #[], 0, none, true⟩
    { programs := #[], prescribed := #[], linked := false, covered := #[], delta := ⟨1, 2⟩, drawn := #[],
      levels := #[{ name := "unit", query := none, size := 4, draw := some (.subset 2) }] }) with
  | .reject r => r == "levels-profile-mismatch: the parameters name 2 level(s), and the accept's profile lists 1"
  | _ => false
#guard match levelsHeld 1 (.accept ⟨#[], #[], 0, none, true⟩
    { programs := #[], prescribed := #[], linked := false, covered := #[], delta := ⟨1, 2⟩, drawn := #[],
      levels := #[{ name := "unit", query := none, size := 4, draw := some (.subset 2) }] }) with
  | .accept _ _ => true
  | _ => false
private def interiorProfile : Core.Integrity :=
  { programs := #[], prescribed := #[], linked := false, covered := #[], delta := ⟨1, 2⟩, drawn := #[("unit", 2)],
    levels := #[{ name := "unit", query := none, size := 4, draw := some (.subset 2) }] }
private def twoLevelProfile : Core.Integrity :=
  coarser { name := BOUNDARY_LEVEL, query := some "p", size := 6, draw := some (.subset 3) } 3 interiorProfile
#guard twoLevelProfile.levels.map (·.name) == #[BOUNDARY_LEVEL, "unit"]
#guard twoLevelProfile.levels.map (·.size) == #[6, 4]
#guard twoLevelProfile.levels.map (·.draw) == #[some (.subset 3), some (.subset 2)]
#guard twoLevelProfile.drawn == #[(BOUNDARY_LEVEL, 3), ("unit", 2)]
#guard twoLevelProfile.credited.isEmpty && twoLevelProfile.creditedProblem.isNone
#guard ({ twoLevelProfile with credited := #[0] }).creditedProblem.isSome
#guard match levelsHeld 2 (.accept ⟨#[], #[], 0, none, true⟩ twoLevelProfile) with
  | .accept _ prof => prof.levels.size == 2
  | _ => false
#guard match levelsHeld 1 (.accept ⟨#[], #[], 0, none, true⟩ twoLevelProfile) with
  | .reject r => r == LEVELS_PROFILE 1 2
  | _ => false
#guard designCount (.stratified #[⟨"a", 3, 1⟩, ⟨"b", 5, 2⟩]) == 3

end VStar.Recursive.Sampled
