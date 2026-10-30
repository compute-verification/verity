/-
A composition whose components must commit to one record, remote's and onsite's shape (`Core.Remote.protocol`,
`Core.Onsite.protocol`): `same` is `product` whose components `ss` must all have committed to the first one's digest. At
remote they are the inference-only policy, PoUW's window and sampled proofs' one-stage audit, each of whose commitment is
the run's record; at onsite the policy and the one-stage audit. Once the cases fit, one that committed to another rejects
whatever the components answer; one whose commitment doesn't decode rejects as a component. Its parameters, commitment
and evidence are `product`'s.
-/
import CoreComposition.Product

namespace Core.Composition

/-- The digest of the commitment of the case `ev` carries in the place of the component `s` of `cs`, as that component
decodes it; `none` when no component is `s` or its commitment doesn't decode. -/
def committedBy (s : String) (cs : List Protocol) (ev : List Case) : Option ByteArray :=
  ((cs.zip ev).find? fun x => x.1.id == s).bind fun x => (x.1.commitment x.2.commitment).toOption.map x.1.digest

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
