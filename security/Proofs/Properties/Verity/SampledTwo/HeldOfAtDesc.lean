module

public import Proofs.Properties.Verity.SampledTwo.Reading
public import Proofs.Flock.SpotFlock.HeldOfDesc

/-!
# The held copies at the form's accept, from the descriptor check

The run of record's accept is read at one session of record `A`, whose held copies `A.h` are asked to be the ones
`SpotFlock.HeldOf.heldOf` builds at each accepted play: for every play `o` the form accepts on the case map `C`, with
run of record's input `i` and interior `it`, some `u` has `heldOf i it u = some A.h`. `heldOf` builds a copy only where
the descriptor's checks pass (`HeldOf.decided`), and the form's check decides none of them: it does not call the
descriptor check (`VStar.Recursive.SampledTwoDesc.descOk`). So the form's accept alone gives no held copy.

`heldOfAt_of_desc` gives it from two named premises, at any case map, so at the cases at one statement too:

* `DescAt C`: every play the form accepts on `C` passes the descriptor check at its parameters, its blobs and its run
  of record's input and interior. It is a check's verdict: once the form's check calls `descOk`, the form's accept
  gives it.
* `CopyAt A C u₀`: wherever `heldOf` builds a copy at an accepted play, at the reference statements the play's blobs
  give and `u₀`'s other fields, that copy is `A.h`. No check gives it. `heldOf` reads the play's parameters, its
  boundary record (the population and the links) and the reference statements on its blobs, which the prover sends, so
  two accepted plays have one copy only where they agree on all three.

The descriptor check gives the copy (`HeldOf.heldOf_of_descOk`), and `CopyAt` says which it is.

## What this does not show

* `CopyAt` at any case map: that the accepted plays carry one set of parameters, one boundary record and blobs
  giving one set of reference statements, and that `A.h` is the copy built from them.
* That the form's check calls `descOk`: it does not.
-/

@[expose] public section

namespace Verity.SampledTwo

open SpotFlock.HeldOf (Unread heldOf refStmtsOf)

/-- **Every play the form accepts passes the descriptor check** (`SampledTwoDesc.descOk`) at the parameters and blobs
its case decodes to and its run of record's input and interior. -/
def DescAt (C : CaseOf) : Prop :=
  ∀ o k p c e i it, FormAccepts C o → C o = some k → Decoded k p c e → inputOn C o = some i → i.interior = some it →
    VStar.Recursive.SampledTwoDesc.descOk p k.blobs i it = true

/-- **The session of record's held copies are the accepted plays'**: wherever `heldOf` builds a copy at a play the
form accepts, at the reference statements the play's blobs give (`refStmtsOf`) and `u₀`'s descriptor, query, code
and rounds, the copy is `A.h`. -/
def CopyAt (A : Live₂) (C : CaseOf) (u₀ : Unread) : Prop :=
  ∀ o k p c e i it xs h, FormAccepts C o → C o = some k → Decoded k p c e → inputOn C o = some i →
    i.interior = some it → refStmtsOf p k.blobs = some xs → heldOf i it { u₀ with xs } = some h → h = A.h

/-- **Each accepted play's held copies are the ones `heldOf` builds**, where every accepted play passes the
descriptor check (`DescAt`) and the copy built at it is the session of record's (`CopyAt`): the check gives a copy
at the play's reference statements (`HeldOf.heldOf_of_descOk`). The conclusion is the premise the run of record's
accept takes of the held copies, at the case map `C`. -/
theorem heldOfAt_of_desc {A : Live₂} {C : CaseOf} (u₀ : Unread) (hd : DescAt C) (hc : CopyAt A C u₀) :
    ∀ o i it, FormAccepts C o → inputOn C o = some i → i.interior = some it → ∃ u, heldOf i it u = some A.h := by
  intro o i it hA hi hit
  obtain ⟨k, hk, p, c, e, -, -, hdec, -⟩ := id hA
  obtain ⟨xs, h, hx, hh⟩ := SpotFlock.HeldOf.heldOf_of_descOk u₀ (hd o k p c e i it hA hk hdec hi hit)
  obtain rfl := hc o k p c e i it xs h hA hk hdec hi hit hx hh
  exact ⟨_, hh⟩

end Verity.SampledTwo
