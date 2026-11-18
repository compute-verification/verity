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

private def split (j : Json) (what : String) : V (String × Json) := do
  let .obj kvs := j | throw s!"the {what} is not an object"
  let .str form := getD j "form" | throw s!"the {what} names no form ({FORMS})"
  pure (form, Json.mkObj (kvs.toArray.filter (·.1 != "form")).toList)

def paramsOfJson (j : Json) : V Params := do
  let (form, rest) ← split j "parameters"
  if form == ONE_STAGE then
    if (field? rest "proofs").isNone then throw s!"the parameters name no proofs: {V1_REFUSED}"
    return .inl (← OneStageZk.paramsEitherOfJson rest)
  if form == RECURSIVE then throw RECURSIVE_REFUSED
  if form == SampledTwo.TWO_STAGE then return .inr (← SampledTwo.paramsOfJson j)
  throw s!"the parameters' form {form} is not {FORMS}"

def evidenceOf (b : ByteArray) : V Evidence := do
  let (form, rest) ← split (← Core.canonicalJson b) "evidence"
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

/-- That form's check; evidence of another form than the params' rejects (`MISMATCH`). -/
def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match p, e with
  | .inl p, .inl e => Caller.gate c.record p.base.verifier (OneStageZk.checkBlobs p c e bs)
  | .inr p, .inr e => SampledTwo.checkBlobs p c e bs
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
  | .error e => e.startsWith "the keys of the parameters"
  | .ok _ => false
#guard match evidenceOf "{\"form\":\"two-stage\"}".toUTF8 with
  | .error e => e.startsWith "the keys of the evidence"
  | .ok _ => false
#guard match evidenceOf "{\"form\":\"recursive\"}".toUTF8 with
  | .error e => e == RECURSIVE_REFUSED
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("form", ONE_STAGE), ("law", Json.mkObj [])]) with
  | .error e => e == s!"the parameters name no proofs: {V1_REFUSED}"
  | .ok _ => false
#guard match evidenceOf "{\"form\":\"one-stage\",\"verdicts\":{}}".toUTF8 with
  | .error e => e == s!"the evidence names no proofs: {V1_REFUSED}"
  | .ok _ => false
#guard match paramsOfJson (Json.mkObj [("form", RECURSIVE), ("recursive", Json.mkObj [])]) with
  | .error e => e == RECURSIVE_REFUSED
  | .ok _ => false

end VStar.Recursive.Sampled
