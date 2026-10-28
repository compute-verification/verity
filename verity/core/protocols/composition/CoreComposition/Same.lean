/-
A composition whose components must commit to one record, remote's and onsite's shape (`Core.Remote.protocol`,
`Core.Onsite.protocol`): `same` is `product` whose components `ss`, each with its reading of the records its case
commits to (`Records`), must all have committed to the first record. At remote they are the inference-only policy's
commitment and sampled proofs' one-stage audit's, each the run's record, and the Commit record each window of PoUW's
period names; at onsite the policy's and the one-stage audit's commitments (`committed`). Once the cases fit, a record
that is another rejects whatever the components answer; a component whose records don't decode rejects as a
component. Its parameters, commitment and evidence are `product`'s.
-/
import CoreComposition.Product

namespace Core.Composition

/-- A component's reading of the records its case commits to: `none` when they don't decode. -/
abbrev Records := Protocol → Case → Option (List ByteArray)

/-- The record a component's commitment is: its digest, as the component decodes it. -/
def committed : Records := fun P k => (P.commitment k.commitment).toOption.map fun c => [P.digest c]

/-- The records `r` reads from the case `ev` carries in the place of the component `s` of `cs`; `none` when no
component is `s` or `r` reads none. -/
def recordsBy (s : String) (r : Records) (cs : List Protocol) (ev : List Case) : Option (List ByteArray) :=
  ((cs.zip ev).find? fun x => x.1.id == s).bind fun x => r x.1 x.2

/-- Why the components `ss` of `cs`, each with its reading of the records its case commits to, didn't all commit to one
record on the cases `ev`: the first record that isn't the first one's, with both; `none` when they all committed to it,
or when one of them reads no records, one not decoding, so that it rejects as a component. -/
def strays (ss : List (String × Records)) (cs : List Protocol) (ev : List Case) : Option String :=
  match (ss.mapM fun (s, r) => (recordsBy s r cs ev).map fun (ds : List ByteArray) => ds.map (s, ·)).map
      List.flatten with
  | some ((s, d) :: rest) => (rest.find? (·.2 != d)).map fun (t, d') =>
    s!"record: {s} committed {Flock.hexOf d}, {t} committed {Flock.hexOf d'}"
  | _ => none

/-- `check` of the components `cs`, except that once the cases fit, components `ss` that didn't all commit to one
record reject whatever the components answer. -/
def checkSame (id : String) (ss : List (String × Records)) (cs : List Protocol) (ps : List Named) (c : ByteArray)
    (ev : List Case) : Outcome Keyed Keyed :=
  match run cs ps ev with
  | .error r => .reject r
  | .ok answers => match strays ss cs ev with
    | some r => .reject r
    | none => judge id c answers

/-- The composition `id` of the components `cs`, which must be listed in the order of their ids, each once, whose
components `ss`, each read by its `Records`, must all commit to one record. -/
def same (id : String) (ss : List (String × Records)) (cs : List Protocol) : Protocol :=
  { product id cs with check := checkSame id ss cs }

end Core.Composition
