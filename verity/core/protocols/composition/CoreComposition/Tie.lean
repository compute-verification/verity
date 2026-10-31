/-
A tie: components of a composition that must name one value, each read off its case's commitment, the way onsite's
space and bandwidth components must be on one store (`Core.Onsite.store`). `tiedSame` is `same` with ties: once `same`
doesn't reject, components of a tie that didn't all name one value reject whatever the components answer, and any
rejection of `same`'s, a component's, the record's or the cases' misfit, stands. Its parameters, commitment and evidence
are `same`'s.
-/
import CoreComposition.Blobs

namespace Core.Composition

/-- The components `names` must all name one value, `what`, each as its function reads it off its case's commitment
(`none`: that commitment doesn't decode, and the component rejects). -/
structure Tie where
  what : String
  names : List (String × (ByteArray → Option ByteArray))

/-- Why the components of the tie `t` didn't all name one value on the cases `ev`: the first whose value isn't the first
one's, with both; `none` when they all named it, or when one of them has no case or names nothing. -/
def Tie.strays (t : Tie) (ev : List Case) : Option String :=
  let named (n : String × (ByteArray → Option ByteArray)) : Option (String × ByteArray) :=
    ((ev.find? fun (k : Case) => k.id == n.1).bind fun k => n.2 k.commitment).map (n.1, ·)
  match t.names.mapM named with
  | some ((s, d) :: rest) => (rest.find? fun (x : String × ByteArray) => x.2 != d).map fun x =>
    s!"{t.what}: {s} committed {Flock.hexOf d}, {x.1} committed {Flock.hexOf x.2}"
  | _ => none

/-- The outcome `o` on the cases `ev`, except that when it doesn't reject, the first tie of `ties` whose components
didn't all name one value rejects. -/
def holdTies (ties : List Tie) (ev : List Case) : Outcome Keyed Keyed → Outcome Keyed Keyed
  | .reject r => .reject r
  | o => match ties.findSome? (·.strays ev) with
    | some r => .reject r
    | none => o

/-- `same id ss cs` whose components of each tie of `ties` must name one value. -/
def tiedSame (id : String) (ss : List String) (ties : List Tie) (cs : List Protocol) : Protocol :=
  { same id ss cs with check := fun ps c ev => holdTies ties ev (checkSame id ss cs ps c ev) }

/-- `tiedSame` over components that read the caller's blobs: `sameB`, whose components of each tie of `ties` must
name one value. -/
def tiedSameB (id : String) (ss : List String) (ties : List Tie) (cs : List BlobProtocol) : BlobProtocol :=
  { tiedSame id ss ties (cs.map (·.toProtocol)) with
    checkBlobs := fun ps c ev bs => holdTies ties ev (checkSame id ss (cs.map (·.onBlobs bs)) ps c ev) }

/-- On the caller's blobs a `tiedSameB` is `tiedSame` of its components on them. -/
theorem tiedSameB_on (id : String) (ss : List String) (ties : List Tie) (cs : List BlobProtocol)
    (blobs : Array ByteArray) :
    (tiedSameB id ss ties cs).on blobs = tiedSame id ss ties (cs.map (·.onBlobs (Blobs.ofArray blobs))) := rfl

end Core.Composition
