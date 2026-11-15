/-
A tie: subprotocols of a composition that must name one value, each reading the values it names off its case, the way
onsite's space and bandwidth subprotocols must be on one store (`Core.Onsite.store`) and each window of remote's PoUW
period must name the policy's record (`Core.Remote.windows`). `tiedSame` is `same` with ties: once `same` doesn't
reject, subprotocols of a tie that didn't all name one value reject whatever the subprotocols answer, and any rejection of
`same`'s, a subprotocol's, the record's or the cases' misfit, stands. Its parameters, commitment and evidence are
`same`'s.
-/
module

public import CoreComposition.Blobs

@[expose] public section

namespace Core.Composition

/-- The subprotocols `names` must all name one value, `what`, each naming the values its function reads off its case
(`none`: the case doesn't decode, and the subprotocol rejects). -/
structure Tie where
  what : String
  names : List (String × (Case → Option (List ByteArray)))

/-- Why the subprotocols of the tie `t` didn't all name one value on the cases `ev`: the first value that isn't the first
one, with both; `none` when they all named it, or when one of them has no case or its case doesn't decode. -/
def Tie.strays (t : Tie) (ev : List Case) : Option String :=
  let named (n : String × (Case → Option (List ByteArray))) : Option (List (String × ByteArray)) :=
    ((ev.find? fun (k : Case) => k.id == n.1).bind n.2).map fun ds => ds.map (n.1, ·)
  match (t.names.mapM named).map List.flatten with
  | some ((s, d) :: rest) => (rest.find? fun (x : String × ByteArray) => x.2 != d).map fun x =>
    s!"{t.what}: {s} committed {Flock.hexOf d}, {x.1} committed {Flock.hexOf x.2}"
  | _ => none

/-- The outcome `o` on the cases `ev`, except that when it doesn't reject, the first tie of `ties` whose subprotocols
didn't all name one value rejects. -/
def holdTies (ties : List Tie) (ev : List Case) : Outcome Keyed Keyed → Outcome Keyed Keyed
  | .reject r => .reject r
  | o => match ties.findSome? (·.strays ev) with
    | some r => .reject r
    | none => o

/-- `same id ss cs` whose subprotocols of each tie of `ties` must name one value. -/
def tiedSame (id : String) (ss : List String) (ties : List Tie) (cs : List Protocol) : Protocol :=
  { same id ss cs with check := fun ps c ev => holdTies ties ev (checkSame id ss cs ps c ev) }

/-- `tiedSame` over subprotocols that read the caller's blobs: `sameB`, whose subprotocols of each tie of `ties` must
name one value. -/
def tiedSameB (id : String) (ss : List String) (ties : List Tie) (cs : List BlobProtocol) : BlobProtocol :=
  { tiedSame id ss ties (cs.map (·.toProtocol)) with
    checkBlobs := fun ps c ev bs => holdTies ties ev (checkSame id ss (cs.map (·.onBlobs bs)) ps c ev) }

/-- On the caller's blobs a `tiedSameB` is `tiedSame` of its subprotocols on them. -/
theorem tiedSameB_on (id : String) (ss : List String) (ties : List Tie) (cs : List BlobProtocol)
    (blobs : Array ByteArray) :
    (tiedSameB id ss ties cs).on blobs = tiedSame id ss ties (cs.map (·.onBlobs (Blobs.ofArray blobs))) := rfl

end Core.Composition
