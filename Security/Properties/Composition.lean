import CoreComposition.Tie

/-!
# The composition of core protocols: what its Lean code (`CoreComposition.Product`, `Same`, `Tie`) does

The properties of `Core.Protocol.answer`, the whole of what a caller of any core protocol runs, of
`Core.Composition.check`, the check of `Core.Composition.product`, of `Core.Composition.checkSame`, the check of
`Core.Composition.same` (remote Verity's, and onsite's under its ties), whose components `ss` must commit to one record,
and of `Core.Composition.holdTies`, which `Core.Composition.tiedSame` (onsite Verity's) runs over `checkSame`'s outcome,
stated against the functions as they run, each in one shape: for every input, if the function accepts, its named
conclusion holds. `Proofs.Properties.Composition` proves them.

Together they reduce an acceptance of a composition's `answer` to its components' own: the three byte strings decode
and the composition's check accepts them (`AnswerAccepts`, at the product), so the cases fit the composition, every
component's `answer` accepts its own case, which `AnswerAccepts` (at the component) decodes to that component's check
accepting, and the commitment is the binding of the components' digests (`AcceptMeans`). At a `same` composition the
components `ss` also committed to one record (`AcceptSame`), and under `holdTies` the outcome it holds accepted and every
tie's components named one value (`AcceptTied`). The predicates below say what the cases and digests are, in the
components' own terms; none restates the composition's check.
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

/-- **The components `ss` committed to one record** on the cases `ev`: each that has a digest (`committedBy`, the digest
of its case's commitment as it decodes it) has the same one, `d`; or one of them has none, its case missing or its
commitment not decoding, so that it rejects as a component. -/
def OneRecord (ss : List String) (cs : List Protocol) (ev : List Case) : Prop :=
  (∃ s ∈ ss, committedBy s cs ev = none) ∨ ∃ d, ∀ s ∈ ss, committedBy s cs ev = some d

/-- An accepted `same` composition: an `Accepted` composition of `cs` whose components `ss` committed to one record. -/
def SameAccepted (id : String) (ss : List String) (cs : List Protocol) (ps : List Named) (c : ByteArray)
    (ev : List Case) (pub prof : Keyed) : Prop :=
  Accepted id cs ps c ev pub prof ∧ OneRecord ss cs ev

/-- **An accepted `same` composition is `SameAccepted`**: what `checkSame` accepts is an accepted composition of its
components (`AcceptMeans`'s conclusion), and the components `ss` committed to one record. At remote
(`Core.Remote.protocol`, `checkSame "verity/remote/v0"` over all three of its subprotocols) the policy, PoUW's window and
the one-stage audit committed to one record; at onsite (`Core.Onsite.protocol`, under its ties) the policy and the
one-stage audit (`Core.Onsite.tied`). -/
def AcceptSame : Prop :=
  ∀ (id : String) (ss : List String) (cs : List Protocol) (ps : List Named) (c : ByteArray) (ev : List Case)
    (pub prof : Keyed),
    checkSame id ss cs ps c ev = .accept pub prof → SameAccepted id ss cs ps c ev pub prof

/-- The value the component `n.1` of a tie names on the cases `ev`, read by `n.2` off the commitment of its case (the
first case under its id); `none` when it has no case or names nothing. -/
def named (ev : List Case) (n : String × (ByteArray → Option ByteArray)) : Option ByteArray :=
  (ev.find? fun k => k.id == n.1).bind fun k => n.2 k.commitment

/-- **The components of the tie `t` named one value** on the cases `ev`: each that names a value names the same one,
`d`; or one of them names none, its case missing or its commitment not decoding, so that it rejects as a component. -/
def Tied (t : Tie) (ev : List Case) : Prop :=
  (∃ n ∈ t.names, named ev n = none) ∨ ∃ d, ∀ n ∈ t.names, named ev n = some d

/-- The outcome `o` held by the ties `ties` on the cases `ev`, accepted: `o` itself accepted with these public outputs
and profile, and every tie's components named one value. -/
def TiedAccepted (ties : List Tie) (ev : List Case) (o : Outcome Keyed Keyed) (pub prof : Keyed) : Prop :=
  o = .accept pub prof ∧ ∀ t ∈ ties, Tied t ev

/-- **What `holdTies` accepts is `TiedAccepted`**: the outcome it holds accepted, unchanged, and every tie's components
named one value. At onsite (`Core.Onsite.protocol`, `holdTies [Core.Onsite.store]` over its `checkSame`) PoUS and the
remote bandwidth audit's P2 session are on one store. -/
def AcceptTied : Prop :=
  ∀ (ties : List Tie) (ev : List Case) (o : Outcome Keyed Keyed) (pub prof : Keyed),
    holdTies ties ev o = .accept pub prof → TiedAccepted ties ev o pub prof

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
