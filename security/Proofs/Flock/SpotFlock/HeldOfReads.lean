module

public import Proofs.Flock.SpotFlock.HeldOf
public import Proofs.Flock.SpotFlock.OwnReads

/-!
# The reference reads of the held copies `heldOf` builds are well-formed trees, where the params' are

`SpotFlock.ownReads_of` takes the premise `RefReadsWf h`: each template's reference reads, as `h` holds them, are
well-formed JSON trees. `HeldOf.heldOf` takes them from the interior's params (`Spot.Run.Interior.circuits`, each
template's `registered`), not from the reference statements (`Unread.xs`), so that they are the ones the second
stage's statement check reads (`HeldMatches.refReads`). So:

* `refReadsWf_of`: a held copy `heldOf` builds has `RefReadsWf` where the interior's templates' registered reads are
  well-formed trees (`ReadsWf`);
* `refReadsWf_form`: so has one `heldOfForm` builds at the two-stage form's run of record, where the form's params'
  templates' registered reads are (`inputOf` hands the run the params' `circuits`);
* `ownReads_form`: there `refReadsOwn` gives `OwnReads`.

## What this does not show

`ReadsWf` of the form's params. The form reads its params through `Core.canonicalJson`, which parses with
`Lean.Json.parse`, a `partial def` no proof reads through, so `FlockComplete.JsonRT.parse_wf` (of the verifier's own
parser, `Flock.JsonParse.parse`) does not reach them. `ReadsWf` is the premise that stays.
-/

@[expose] public section

namespace SpotFlock.HeldOf

open Lean (Json)
open Claims.Json (Wf)

/-- **The templates' registered reads are well-formed trees**: of the templates' circuits `cs` as the params give
them (template, circuit, header, registered reads), each one's registered reads. -/
def ReadsWf (cs : Array (String × String × Json × Json)) : Prop := ∀ c ∈ cs, Wf c.2.2.2

theorem wf_getD_reads {cs : Array (String × String × Json × Json)} (hw : ReadsWf cs) (t : ℕ) :
    Wf ((cs.map (·.2.2.2)).toList.getD t .null) := by
  rw [List.getD_eq_getElem?_getD]
  cases hg : (cs.map (·.2.2.2)).toList[t]? with
  | none =>
    rw [Option.getD_none]
    exact Wf.null
  | some x =>
    rw [Option.getD_some]
    have hx : x ∈ (cs.map (·.2.2.2)).toList := List.mem_of_getElem? hg
    simp only [Array.toList_map, List.mem_map] at hx
    obtain ⟨c, hc, rfl⟩ := hx
    exact hw c (Array.mem_toList_iff.1 hc)

/-- **A held copy `heldOf` builds has well-formed reference reads** (`RefReadsWf`) where the interior's templates'
registered reads are well-formed trees: `heldOf` takes each template's reference reads from them. -/
theorem refReadsWf_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (hw : ReadsWf it.circuits) : RefReadsWf h := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  intro t _
  exact wf_getD_reads hw t

/-- **`OwnReads` at a held copy `heldOf` builds**, from `refReadsOwn` and the interior's templates' registered reads
being well-formed trees. -/
theorem ownReads_of_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (hc : refReadsOwn h = true) (hw : ReadsWf it.circuits) : OwnReads h :=
  ownReads_of hc (refReadsWf_of hh hw)

/-- The interior of the form's run of record has the params' templates' circuits. -/
theorem circuits_inputOf {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ ξ₂ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {received₂ : String} {vs : Array (String × Json)}
    {it : Spot.Run.Interior}
    (hit : (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior = some it) :
    it.circuits = p.circuits := by
  have h1 : ((VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior.map (·.circuits)) =
      some p.circuits := rfl
  rw [hit] at h1
  exact Option.some.inj h1

/-- **A held copy `heldOfForm` builds at the form's run of record has well-formed reference reads**, where the form's
params' templates' registered reads are well-formed trees. -/
theorem refReadsWf_form {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ ξ₂ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {received₂ : String} {vs : Array (String × Json)} {u : Unread}
    {h : Held} (hh : heldOfForm p c ξ ξ₂ e received₂ vs u = some h) (hw : ReadsWf p.circuits) : RefReadsWf h := by
  obtain ⟨it, hit, hh', -⟩ := heldMatches_form hh
  refine refReadsWf_of hh' ?_
  rw [circuits_inputOf hit]
  exact hw

/-- **`OwnReads` at the form's run of record**: at a held copy `heldOfForm` builds, `refReadsOwn` gives `OwnReads`
where the form's params' templates' registered reads are well-formed trees. -/
theorem ownReads_form {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ ξ₂ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {received₂ : String} {vs : Array (String × Json)} {u : Unread}
    {h : Held} (hh : heldOfForm p c ξ ξ₂ e received₂ vs u = some h) (hc : refReadsOwn h = true)
    (hw : ReadsWf p.circuits) : OwnReads h :=
  ownReads_of hc (refReadsWf_form hh hw)

end SpotFlock.HeldOf
