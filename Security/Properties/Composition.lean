import CoreComposition.Registered

/-!
# The composition of core protocols: what its Lean code (`CoreComposition.Product`, `CoreComposition.Registered`) does

The properties of `Core.Protocol.answer`, the whole of what a caller of any core protocol runs, of
`Core.Composition.check`, the check of `Core.Composition.product` (remote Verity, and what onsite composes), and of
`Core.Composition.checkRegistered`, the check of `Core.Composition.registered` (onsite Verity, whose registration check
is NCI's), stated against the functions as they run, each in one shape: for every input, if the function accepts, its
named conclusion holds. `Proofs.Properties.Composition` proves them.

Together they reduce an acceptance of a composition's `answer` to its components' own: the three byte strings decode
and the composition's check accepts them (`AnswerAccepts`, at the product), so the cases fit the composition, every
component's `answer` accepts its own case, which `AnswerAccepts` (at the component) decodes to that component's check
accepting, and the commitment is the binding of the components' digests (`AcceptMeans`). At a registered composition
the check's acceptance is also the registration check's on the registration's case, whose digest the commitment binds
and the component whose commitment is the registration committed to, and the rest is an accepted composition of the
components (`AcceptRegistration`). The predicates below say what the
cases and digests are, in the components' own terms; none restates the composition's check.
-/

namespace Properties.Composition

open Core Core.Composition

/-- The three byte strings decode under `P`'s decoders (the parameters as canonical JSON and then as `P`'s parameters,
the commitment and the evidence), and `P`'s check accepts what they decode to. -/
def DecodedAccepted (P : Protocol) (a b e : ByteArray) : Prop :=
  ∃ p c ev, (canonicalJson a >>= P.params) = .ok p ∧ P.commitment b = .ok c ∧ P.evidence e = .ok ev ∧
    P.Accepts p c ev

/-- **What any protocol's `answer` accepts decodes, and its check accepts it.** -/
def AnswerAccepts : Prop :=
  ∀ (P : Protocol) (a b e : ByteArray), (P.answer a b e).verdict = .accept → DecodedAccepted P a b e

/-- Each component with the parameters the composition names for it and its case, in the composition's order. -/
def triples (cs : List Protocol) (ps : List Named) (ev : List Case) : List (Protocol × Named × Case) :=
  cs.zip (ps.zip ev)

/-- The cases fit the composition: the parameters and the evidence name exactly the components `cs`, in their order,
so that every component has its triple, and each case carries the parameters the composition names for it. -/
def Fits (cs : List Protocol) (ps : List Named) (ev : List Case) : Prop :=
  ps.map (·.id) = cs.map (·.id) ∧ ev.map (·.id) = cs.map (·.id) ∧ (triples cs ps ev).map (·.1) = cs ∧
    ∀ x ∈ triples cs ps ev, x.2.2.params = x.2.1.params

/-- The digest of the commitment `b` as component `P` decodes it; `none` when it doesn't decode. -/
def digestOf (P : Protocol) (b : ByteArray) : Option ByteArray :=
  (P.commitment b).toOption.map P.digest

/-- Each component's id with the digest of its case's commitment, in the composition's order; `none` when one of them
doesn't decode. -/
def digests (cs : List Protocol) (ps : List Named) (ev : List Case) : Option (List (String × ByteArray)) :=
  (triples cs ps ev).mapM fun x => (digestOf x.1 x.2.2.commitment).map (x.1.id, ·)

/-- An accepted composition: the cases fit it, every component's case decodes to what the component's check accepts
under the parameters the composition names for it, the commitment `c` is `bind` of the components' digests, and the
public outputs and profile are keyed by the components' ids, in order. -/
def Accepted (id : String) (cs : List Protocol) (ps : List Named) (c : ByteArray) (ev : List Case)
    (pub prof : Keyed) : Prop :=
  Fits cs ps ev ∧
    (∀ x ∈ triples cs ps ev, DecodedAccepted x.1 x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence) ∧
    (digests cs ps ev).map (bind id) = some c ∧ pub.map (·.1) = cs.map (·.id) ∧ prof.map (·.1) = cs.map (·.id)

/-- **An accepted composition is `Accepted`**: what the composition's check accepts fits it, each of its components
accepts its own case, and its commitment binds their digests. -/
def AcceptMeans : Prop :=
  ∀ (id : String) (cs : List Protocol) (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    check id cs ps c ev = .accept pub prof → Accepted id cs ps c ev pub prof

/-- An accepted registered composition: the registration's entry `n` and case `k` are the registration check `R`'s and
carry the same parameters, its case decodes to what `R`'s check accepts under them, the registration's digest is `d`,
the commitment's part after the components' binding `b`, the component `s` of `cs` committed on its case to the same
digest `d` (one registration digest in both), and the components' parameters `ps` and cases `ev` are an `Accepted`
composition of `cs` under `b`, with its public outputs and profile. -/
def RegisteredAccepted (id : String) (R : Protocol) (s : String) (cs : List Protocol) (n : Named) (ps : List Named)
    (b d : ByteArray) (k : Case) (ev : List Case) (pub prof : Keyed) : Prop :=
  n.id = R.id ∧ k.id = R.id ∧ k.params = n.params ∧ DecodedAccepted R n.params.toUTF8 k.commitment k.evidence ∧
    digestOf R k.commitment = some d ∧ (∃ x ∈ cs.zip ev, x.1.id = s ∧ digestOf x.1 x.2.commitment = some d) ∧
    Accepted id cs ps b ev pub prof

/-- **An accepted registered composition is `RegisteredAccepted`**: what its check accepts, the registration check
accepted on the registration's case, the commitment binds the registration's digest, the component `s` committed to
that digest, and the rest is an accepted composition of the components (`AcceptMeans`'s conclusion). At onsite
(`Core.Onsite.protocol`, whose check is `checkRegistered "verity/onsite/v0" NciCheck.Protocol.protocol
SampledProofs.OneStage.protocol.id Core.Onsite.components`), NCI accepted the registration, and it is the registration
record sampled proofs' one-stage audit committed to. -/
def AcceptRegistration : Prop :=
  ∀ (id : String) (R : Protocol) (s : String) (cs : List Protocol) (n : Named) (ps : List Named) (b d : ByteArray)
    (k : Case) (ev : List Case) (pub prof : Keyed),
    checkRegistered id R s cs (n, ps) (b, d) (k, ev) = .accept pub prof →
      RegisteredAccepted id R s cs n ps b d k ev pub prof

/-- **The bytes `bind` hashes**: the composition's id, a zero byte, and the canonical JSON
`{"components":[{"commitment":HEX,"id":ID},...]}` of the components' digests in order. -/
def bindMessage (id : String) (digests : List (String × ByteArray)) : ByteArray :=
  id.toUTF8 ++ ByteArray.mk #[0] ++ ("{\"components\":[" ++ ",".intercalate (digests.map fun (i, d) =>
    "{\"commitment\":" ++ Flock.canonStr (Flock.hexOf d) ++ ",\"id\":" ++ Flock.canonStr i ++ "}") ++ "]}").toUTF8

/-- **Equal compositions' commitments bind equal digests, or SHA-512 collides on their two messages**: `bind` of one
id over the digests `ds` and over `ds'` differ, or `ds = ds'`, or the two messages it hashed (`bindMessage`) differ and
have one compiled SHA-512. A finder that holds both lists outputs that pair. -/
def BindCollision : Prop :=
  ∀ (id : String) (ds ds' : List (String × ByteArray)), bind id ds ≠ bind id ds' ∨
    ds = ds' ∨ (bindMessage id ds ≠ bindMessage id ds' ∧
      Flock.Sha512.hash (bindMessage id ds) = Flock.Sha512.hash (bindMessage id ds'))

end Properties.Composition
