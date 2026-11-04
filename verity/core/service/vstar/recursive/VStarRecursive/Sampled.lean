/-
Sampled proofs (`verity/sampled-proofs/v1`), a `Core.BlobProtocol`: one-stage v1's audit at the challenger's coins
(`SampledProofs.OneStage.checkLive`) or the recursive audit (`VStar.Recursive`), whichever the auditor's params name.

`Core.Composition.product` reads every component's case, so sampled proofs is one component and the form is the params':

* **params**: `{"form": "one-stage" | "recursive", ...}`, the rest that form's params. The auditor fixes the form with the
  rest of its params, before any coin;
* **commitment**: v1's (the registration record), as both forms read it;
* **evidence**: `{"form", ...}`, the rest that form's evidence;
* **check**: that form's check, the recursive audit's on the blobs (one-stage v1 reads none); evidence of the other form
  rejects.
-/
import VStarRecursive.Protocol

namespace VStar.Recursive.Sampled

open Lean (Json)
open Flock (V)
open SampledProofs

def PROTOCOL : String := "verity/sampled-proofs/v1"

def ONE_STAGE : String := "one-stage"
def RECURSIVE : String := "recursive"

abbrev Params := OneStage.Params ⊕ Recursive.Params
abbrev Evidence := OneStage.Evidence ⊕ Recursive.Evidence

private def split (j : Json) (what : String) : V (String × Json) := do
  let .obj kvs := j | throw s!"the {what} is not an object"
  let .str form := getD j "form" | throw s!"the {what} names no form (\"{ONE_STAGE}\" or \"{RECURSIVE}\")"
  pure (form, Json.mkObj (kvs.toArray.filter (·.1 != "form")).toList)

def paramsOfJson (j : Json) : V Params := do
  let (form, rest) ← split j "parameters"
  if form == ONE_STAGE then return .inl (← OneStage.paramsOfJson rest)
  if form == RECURSIVE then return .inr (← Recursive.paramsOfJson rest)
  throw s!"the parameters' form {form} is not \"{ONE_STAGE}\" or \"{RECURSIVE}\""

def evidenceOf (b : ByteArray) : V Evidence := do
  let (form, rest) ← split (← Core.canonicalJson b) "evidence"
  let rest := (← Flock.canon rest).toUTF8
  if form == ONE_STAGE then return .inl (← OneStage.evidenceOf rest)
  if form == RECURSIVE then return .inr (← Recursive.evidenceOf rest)
  throw s!"the evidence's form {form} is not \"{ONE_STAGE}\" or \"{RECURSIVE}\""

def checkBlobs (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match p, e with
  | .inl p, .inl e => OneStage.checkLive p c e
  | .inr p, .inr e => Recursive.checkBlobs p c e bs
  | .inl _, .inr _ => .reject s!"the evidence's form is {RECURSIVE}, and the parameters' is {ONE_STAGE}"
  | .inr _, .inl _ => .reject s!"the evidence's form is {ONE_STAGE}, and the parameters' is {RECURSIVE}"

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

#guard (paramsOfJson (Json.mkObj [("form", "two-stage")])).toOption.isNone
#guard (evidenceOf "{\"form\":\"recursive\"}".toUTF8).toOption.isNone

end VStar.Recursive.Sampled
