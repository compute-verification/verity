module

public import Proofs.Properties.Verity.SampledTwo.AcceptsTop
public import Proofs.Flock.SpotFlock.HeldOfDesc

/-!
# The held copies at the form's accept, from the descriptor check

The run of record's accept is read at one session of record `A`, whose held copies `A.h` are asked to be the ones
`SpotFlock.HeldOf.heldOf` builds at each accepted play (`HeldOfAt`): for every play `o` the form accepts on the case
map `C`, with run of record's input `i` and interior `it`, some `u` has `heldOf i it u = some A.h`. `heldOf` builds a
copy only where the descriptor's checks pass (`HeldOf.decided`), and the form's check decides none of them: it does
not call the descriptor check (`VStar.Recursive.SampledTwoDesc.descOk`). So the form's accept alone gives no held copy.

`heldOfAt_of_desc` gives it from two named premises, at any case map, so at the cases at one statement too:

* `DescAt C`: every play the form accepts on `C` passes the descriptor check at its parameters, its blobs and its run
  of record's input and interior. It is a check's verdict: once the form's check calls `descOk`, the form's accept
  gives it.
* `CopyAt A C u₀`: wherever `heldOf` builds a copy at an accepted play, at the reference statements the play's blobs
  give and `u₀`'s other fields, that copy is `A.h`. No check gives it. `heldOf` reads the play's parameters, its
  boundary record (the population and the links) and the reference statements on its blobs, which the prover sends, so
  two accepted plays have one copy only where they agree on all three.

The descriptor check gives the copy (`HeldOf.heldOf_of_descOk`), and `CopyAt` says which it is.

## `CopyAt` at the cases at one statement

At the cases at the statement `x` (`atStmt C x`), `copyAt_of` gives `CopyAt` from three premises, one for each thing
`heldOf` reads, and `heldOfAt_at` is `HeldOfAt A (atStmt C x)` from them and `DescAt`:

* the boundary record needs none: every accepted play at `x` carries the record the proved verifier's reader reads
  from `x` (`inputOn_at`, `jsonMsg_of_readBack`);
* `DecodesAt C P`: every accepted play decodes the parameters `P`, which a case map over the plays at one
  parameters' bytes has;
* `RefsAt C xs₀`: at every accepted play, the blobs the parameters' digests name give the reference statements `xs₀`.
  A reference statement is its member's circuit file and public file, each the blob under a digest the parameters
  name, and the pinned lookup tables the blobs hold, each under a digest the verifier pins;
* `CopyOf A P j u`: `A.h` is the copy `heldOf` builds from the parameters `P`, the boundary record `j` and `u`, at
  every input and interior that carry them in the seven fields `heldOf` reads, which are the ones an accepted play's
  input has (`fields_of_accept`).

## What this does not show

* `DecodesAt`, `RefsAt` or `CopyOf` at any case map or session of record. `RefsAt` is no check's verdict.
* That one input with the seven fields building `A.h` gives `CopyOf`: that `heldOf` reads no other field.
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

/-! ## `CopyAt` at the cases at one statement -/

/-- **The accepted plays decode the parameters `P`.** -/
def DecodesAt (C : CaseOf) (P : VStar.Recursive.SampledTwo.Params) : Prop :=
  ∀ o k p c e, FormAccepts C o → C o = some k → Decoded k p c e → p = P

/-- **The accepted plays' blobs give the reference statements `xs₀`**: at every play the form accepts, the blobs the
parameters' digests name give the reference statements `xs₀` (`refStmtsOf`), wherever they give any. -/
def RefsAt (C : CaseOf) (xs₀ : List Claims.Msg) : Prop :=
  ∀ o k p c e xs, FormAccepts C o → C o = some k → Decoded k p c e → refStmtsOf p k.blobs = some xs → xs = xs₀

/-- **`A.h` is the copy `heldOf` builds from the parameters `P`, the boundary record `j` and `u`**: at every input
and interior that carry them in the seven fields `heldOf` reads (the boundary and interior verifiers, the boundary
record, the rule, no closure map, the replay block and the templates' circuits), the copy `heldOf` builds is `A.h`. -/
def CopyOf (A : Live₂) (P : VStar.Recursive.SampledTwo.Params) (j : Lean.Json) (u : Unread) : Prop :=
  ∀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (h : SpotFlock.Held),
    i.params.verifier = P.boundary.verifier → i.record = j → i.rule = P.rule → i.closure = none →
    it.params.verifier = P.interior.verifier → it.replay = P.replay → it.circuits = P.circuits →
    heldOf i it u = some h → h = A.h

/-- **The fields `heldOf` reads of an accepted play's input**: the boundary record is the one `readBack` returns of
the commitment's, the parameters, rule, replay block and circuits are the case's parameters', and there is no closure
map (`Steps.inputAt_ok`, `inputOf`). -/
theorem fields_of_accept {C : CaseOf} {o : Verity.Exchange} {k : Case} {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence} {i : Spot.Run.Input}
    {it : Spot.Run.Interior} (hA : FormAccepts C o) (hk : C o = some k) (hdec : Decoded k p c e)
    (hi : inputOn C o = some i) (hit : i.interior = some it) :
    ∃ j₁, VStar.Recursive.SampledTwo.readBack c.record = .ok j₁ ∧ i.record = j₁ ∧ i.params = p.boundary ∧
      i.rule = p.rule ∧ i.closure = none ∧ it.params = p.interior ∧ it.replay = p.replay ∧
      it.circuits = p.circuits := by
  obtain ⟨k', hk', p', c', e', pub, prof, ⟨h1, h2, h3⟩, hacc⟩ := hA
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  obtain ⟨g1, g2, g3⟩ := hdec
  obtain rfl : p = p' := Except.ok.inj (g1.symm.trans h1)
  obtain rfl : c = c' := Except.ok.inj (g2.symm.trans h2)
  obtain rfl : e = e' := Except.ok.inj (g3.symm.trans h3)
  obtain ⟨-, c₂, ξ₂, r, i', run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨-, -, -, -, vs, rc, j₁, j₂, -, hieq, hj₁, hj₂, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  have hik : inputOfCase k = some i' := by
    unfold inputOfCase
    rw [h1, h2, h3]
    simp only [bind, Except.bind, hin]
    rfl
  have hi' : inputOn C o = some i' := by
    unfold inputOn
    rw [hk]
    exact hik
  have hii : i = i' := Option.some.inj (hi.symm.trans hi')
  have hr : i.record = j₁ := by rw [hii, hieq]; rfl
  have hp : i.params = p.boundary := by rw [hii, hieq]; rfl
  have hrule : i.rule = p.rule := by rw [hii, hieq]; rfl
  have hcl : i.closure = none := by rw [hii, hieq]; rfl
  rw [hii, hieq] at hit
  have hip : it.params = p.interior := (congrArg Spot.Run.Interior.params (Option.some.inj hit)).symm
  have hrep : it.replay = p.replay := (congrArg Spot.Run.Interior.replay (Option.some.inj hit)).symm
  have hcirc : it.circuits = p.circuits := (congrArg Spot.Run.Interior.circuits (Option.some.inj hit)).symm
  exact ⟨j₁, hj₁, hr, hp, hrule, hcl, hip, hrep, hcirc⟩

/-- **`CopyAt` at the cases at the statement `x`**, where the accepted plays there decode the parameters `P`
(`DecodesAt`), their blobs give the reference statements `xs₀` (`RefsAt`), and `A.h` is the copy `heldOf` builds from
`P`, the record `j` the proved verifier's reader reads from `x` and `xs₀` (`CopyOf`): an accepted play at `x` carries
`j` (`inputOn_at`, `jsonMsg_of_readBack`) and `P`'s fields (`fields_of_accept`). -/
theorem copyAt_of {A : Live₂} {C : CaseOf} {x : Claims.Msg} {P : VStar.Recursive.SampledTwo.Params}
    {j : Lean.Json} {xs₀ : List Claims.Msg} (u₀ : Unread) (hP : DecodesAt (atStmt C x) P)
    (hR : RefsAt (atStmt C x) xs₀) (hj : SpotFlock.jsonMsg x = some j)
    (hA : CopyOf A P j { u₀ with xs := xs₀ }) : CopyAt A (atStmt C x) u₀ := by
  intro o k p c e i it xs h hacc hk hdec hi hit hx hh
  obtain rfl := hP o k p c e hacc hk hdec
  obtain rfl := hR o k p c e xs hacc hk hdec hx
  obtain ⟨j₁, hj₁, hrec, hpar, hrule, hcl, hip, hrep, hcirc⟩ := fields_of_accept hacc hk hdec hi hit
  have hmsg : recordMsg i.record = some x := (inputOn_at hi).2
  rw [hrec] at hmsg
  have hjj : j₁ = j := Option.some.inj ((jsonMsg_of_readBack hj₁ hmsg).symm.trans hj)
  exact hA i it h (by rw [hpar]) (hrec.trans hjj) hrule hcl (by rw [hip]) hrep hcirc hh

/-- **Each accepted play at the statement `x` has the held copies `heldOf` builds** (`HeldOfAt` at the cases at
`x`), where every accepted play there passes the descriptor check (`DescAt`), decodes the parameters `P` (`DecodesAt`)
and has blobs giving the reference statements `xs₀` (`RefsAt`), and `A.h` is the copy built from `P`, the record read
from `x` and `xs₀` (`CopyOf`). -/
theorem heldOfAt_at {A : Live₂} {C : CaseOf} {x : Claims.Msg} {P : VStar.Recursive.SampledTwo.Params}
    {j : Lean.Json} {xs₀ : List Claims.Msg} (u₀ : Unread) (hd : DescAt (atStmt C x))
    (hP : DecodesAt (atStmt C x) P) (hR : RefsAt (atStmt C x) xs₀) (hj : SpotFlock.jsonMsg x = some j)
    (hA : CopyOf A P j { u₀ with xs := xs₀ }) : HeldOfAt A (atStmt C x) :=
  heldOfAt_of_desc u₀ hd (copyAt_of u₀ hP hR hj hA)

end Verity.SampledTwo
