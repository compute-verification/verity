/-
A composition with a registration check, onsite Verity's shape (`Core.Onsite.protocol`, whose registration check is
NCI's, `verity/nci/v1`): one `Core.Protocol` that runs the registration check's own `answer` on the registration's case
in its commit phase, before composing, and then composes its components as `product` does (`check`). The registration
check is not a component:

* It rejects when the registration check doesn't accept the registration, whatever the components answer, when the
  commitment doesn't bind the registration's digest, and when the component whose commitment is the registration (at
  onsite, sampled proofs' one-stage audit, whose commitment is the run's registration record) committed to another
  digest: one registration digest in both.
* Otherwise it answers as `product` of its components does on their cases, under the commitment's binding of their
  digests: an accepted registration adds nothing to what the composition makes public or keeps.

Its parameters are `product`'s with the registration's beside the components':
`{"components":[{"id":ID,"params":PARAMS},...],"registration":{"id":ID,"params":PARAMS}}`. Its commitment is `product`'s,
the 64-byte `binding` of the components' digests, followed by the registration's digest, all of which the prover sends
before the challenge; it is its own digest. Its evidence is `product`'s with the registration's case beside the
components': `{"components":[...],"registration":{"commitment":HEX,"evidence":HEX,"id":ID,"params":PARAMS}}`.

A composition with no registration check of its own can still hold its components to one record, remote Verity's shape
(`Core.Remote.protocol`): `same` is `product` whose components `ss` (at remote, PoUW's window and sampled proofs'
one-stage audit, each of whose commitment is the window record) must all have committed to the first one's digest. Once the cases fit, one that committed to another rejects whatever the components answer; one whose
commitment doesn't decode rejects as a component. Its parameters, commitment and evidence are `product`'s.
-/
import CoreComposition.Product

namespace Core.Composition

open Lean (Json)
open Flock (V)

def decodeRegisteredParams (j : Json) : V (Named × List Named) := do
  exactly j ["components", "registration"] "the composition's params"
  let r ← decodeNamed "the registration" (← Flock.J.get j "registration" "the composition's params")
  let xs ← Flock.J.arr j "components" "the composition's params"
  pure (r, ← xs.toList.mapM (decodeNamed "a component"))

/-- The commitment's first 64 bytes, the binding of the components' digests, and the rest, the registration's digest. -/
def decodeRegisteredCommitment (b : ByteArray) : V (ByteArray × ByteArray) :=
  if b.size < 64 then
    .error s!"the binding of the components' digests is 64 bytes, then the registration's digest: not {b.size} bytes"
  else .ok (b.extract 0 64, b.extract 64 b.size)

def decodeRegisteredEvidence (b : ByteArray) : V (Case × List Case) := do
  let j ← canonicalJson b
  exactly j ["components", "registration"] "the composition's evidence"
  let r ← decodeCase "the registration" (← Flock.J.get j "registration" "the composition's evidence")
  let xs ← Flock.J.arr j "components" "the composition's evidence"
  pure (r, ← xs.toList.mapM (decodeCase "a component"))

/-- The registration check's answer on the registration's case, under the parameters the composition names for it. -/
def registrationAnswer (R : Protocol) (n : Named) (k : Case) : Answer :=
  R.answer n.params.toUTF8 k.commitment k.evidence

/-- The digest of the commitment of the case `ev` carries in the place of the component `s` of `cs`, as that component
decodes it; `none` when no component is `s` or its commitment doesn't decode. -/
def committedBy (s : String) (cs : List Protocol) (ev : List Case) : Option ByteArray :=
  ((cs.zip ev).find? fun x => x.1.id == s).bind fun x => (x.1.commitment x.2.commitment).toOption.map x.1.digest

/-- Why a registered composition refuses when the registration check `r` checked the registration digest `d` and the
component `s` committed to `d'`. -/
def mismatch (r s : String) (d : ByteArray) (d' : Option ByteArray) : String :=
  s!"registration: {r} checked {Flock.hexOf d}, {s} committed {(d'.map Flock.hexOf).getD "no digest"}"

/-- The registration check `R` on the registration's entry `n` and case `k`, then `check` of the components `cs` on the
rest under the binding `b`: a registration that isn't `R`'s, that `R` doesn't accept, or whose digest isn't `d` rejects
before any component is run, and once the cases fit the composition, one that isn't the registration the component `s`
committed to rejects whatever the components answer. -/
def checkRegistered (id : String) (R : Protocol) (s : String) (cs : List Protocol) :
    Named × List Named → ByteArray × ByteArray → Case × List Case → Outcome Keyed Keyed
  | (n, ps), (b, d), (k, ev) =>
    if n.id ≠ R.id then .reject s!"params: the registration is {R.id}'s, not {n.id}'s"
    else if k.id ≠ R.id then .reject s!"evidence: the registration is {R.id}'s, not {k.id}'s"
    else if k.params ≠ n.params then
      .reject "evidence: the registration's case carries params other than the composition's"
    else if (registrationAnswer R n k).verdict ≠ .accept then
      let a := registrationAnswer R n k
      .reject s!"registration: {R.id} {a.verdict.name}: {a.reason.getD ""}"
    else if (registrationAnswer R n k).digest ≠ some d then
      .reject "the commitment does not bind the registration's digest"
    else match run cs ps ev with
      | .error r => .reject r
      | .ok answers =>
        if committedBy s cs ev ≠ some d then .reject (mismatch R.id s d (committedBy s cs ev)) else judge id b answers

/-- The composition `id` of the components `cs`, which must be listed in the order of their ids, each once, after the
registration check `R`, on the registration the component `s` commits to. -/
def registered (id : String) (R : Protocol) (s : String) (cs : List Protocol) : Protocol where
  id := id
  Params := Named × List Named
  Commitment := ByteArray × ByteArray
  Evidence := Case × List Case
  Public := Keyed
  Profile := Keyed
  params := decodeRegisteredParams
  commitment := decodeRegisteredCommitment
  evidence := decodeRegisteredEvidence
  digest c := c.1 ++ c.2
  check := checkRegistered id R s cs
  publicJson := Json.mkObj
  profileJson := Json.mkObj

/-- Why the components `ss` of `cs` didn't all commit to one record on the cases `ev`: the first whose digest isn't the
first one's, with both; `none` when they all committed to it, or when one of them has no digest, its commitment not
decoding, so that it rejects as a component. -/
def strays (ss : List String) (cs : List Protocol) (ev : List Case) : Option String :=
  match ss.mapM fun s => (committedBy s cs ev).map (s, ·) with
  | some ((s, d) :: rest) => (rest.find? (·.2 != d)).map fun (t, d') =>
    s!"record: {s} committed {Flock.hexOf d}, {t} committed {Flock.hexOf d'}"
  | _ => none

/-- `check` of the components `cs`, except that once the cases fit, components `ss` that didn't all commit to one
record reject whatever the components answer. -/
def checkSame (id : String) (ss : List String) (cs : List Protocol) (ps : List Named) (c : ByteArray) (ev : List Case) :
    Outcome Keyed Keyed :=
  match run cs ps ev with
  | .error r => .reject r
  | .ok answers => match strays ss cs ev with
    | some r => .reject r
    | none => judge id c answers

/-- The composition `id` of the components `cs`, which must be listed in the order of their ids, each once, whose
components `ss` must all commit to one record. -/
def same (id : String) (ss : List String) (cs : List Protocol) : Protocol :=
  { product id cs with check := checkSame id ss cs }

end Core.Composition
