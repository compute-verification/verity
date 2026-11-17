module

public import Proofs.Flock.SpotFlock.HeldOf
public import Proofs.Flock.SpotFlock.OwnReads

/-! The reference reads of the held copies `heldOf` builds are well-formed trees where the params' are (`refReadsWf_of`,
`ownReads_form`). -/

@[expose] public section

namespace SpotFlock.HeldOf

open Lean (Json)
open Claims.Json (Wf)

/-- A premise no proof discharges at the form's params: the form parses them with `Lean.Json.parse`, a `partial def` no
proof reads through, so `FlockComplete.JsonRT.parse_wf` does not reach them. -/
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

theorem refReadsWf_of {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (hw : ReadsWf it.circuits) : RefReadsWf h := by
  obtain ⟨first, ns, ks, l, hz, -, -, -, rfl, -⟩ := heldOf_spec hh
  intro t _
  exact wf_getD_reads hw t

theorem ownReads_of_heldOf {i : Spot.Run.Input} {it : Spot.Run.Interior} {u : Unread} {h : Held}
    (hh : heldOf i it u = some h) (hc : refReadsOwn h = true) (hw : ReadsWf it.circuits) : OwnReads h :=
  ownReads_of hc (refReadsWf_of hh hw)

theorem circuits_inputOf {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ ξ₂ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {received₂ : String} {vs : Array (String × Json)}
    {it : Spot.Run.Interior}
    (hit : (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior = some it) :
    it.circuits = p.circuits := by
  have h1 : ((VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior.map (·.circuits)) =
      some p.circuits := rfl
  rw [hit] at h1
  exact Option.some.inj h1

theorem refReadsWf_form {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ ξ₂ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {received₂ : String} {vs : Array (String × Json)} {u : Unread}
    {h : Held} (hh : heldOfForm p c ξ ξ₂ e received₂ vs u = some h) (hw : ReadsWf p.circuits) : RefReadsWf h := by
  obtain ⟨it, hit, hh', -⟩ := heldMatches_form hh
  refine refReadsWf_of hh' ?_
  rw [circuits_inputOf hit]
  exact hw

theorem ownReads_form {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment} {ξ ξ₂ : ByteArray}
    {e : VStar.Recursive.SampledTwo.Evidence} {received₂ : String} {vs : Array (String × Json)} {u : Unread}
    {h : Held} (hh : heldOfForm p c ξ ξ₂ e received₂ vs u = some h) (hc : refReadsOwn h = true)
    (hw : ReadsWf p.circuits) : OwnReads h :=
  ownReads_of hc (refReadsWf_form hh hw)

end SpotFlock.HeldOf
