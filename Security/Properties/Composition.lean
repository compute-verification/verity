import CoreComposition.Product

/-!
# The composition of core protocols: what its Lean code (`CoreComposition.Product`) does

The properties of `Core.Protocol.answer`, the whole of what a caller of any core protocol runs, and of
`Core.Composition.check`, the check of `Core.Composition.product` (onsite and remote Verity), stated against the
functions as they run, each in one shape: for every input, if the function accepts, its named conclusion holds.
`Proofs.Properties.Composition` proves them.

Together they reduce an acceptance of a composition's `answer` to its components' own: the three byte strings decode
and the composition's check accepts them (`AnswerAccepts`, at the product), so the cases fit the composition, every
component's `answer` accepts its own case, which `AnswerAccepts` (at the component) decodes to that component's check
accepting, and the commitment is the binding of the components' digests (`AcceptMeans`). The predicates below say what
the cases and digests are, in the components' own terms; none restates the composition's check.
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

end Properties.Composition
