/-
Δ's copies into the rows' circuits' inputs. Each of Δ's later writes is a copy `(dst, dst), (dst, src)` (`pair2`), and at
each `cv`, `pad` and chain bit of a row there is exactly one: the rows' wires (`rowWires_eq`) go into distinct inputs,
since port `p`'s compressions sit at the distinct places `at_p + k` (`rowSlots_eq`).
-/
import FlockSoundness.Discharge.Hm.Delta

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

/-- Port `p`'s chain wire into compression `k ≥ 1`, and its `hm96` wire. -/
def chainWire (ports : Array Port) (sha p k : ℕ) : Wire :=
  ⟨(sha, ((HmRow.rowSlots ports).2[p]!)[k]!.1, 2 * ((HmRow.rowSlots ports).2[p]!)[k]!.2),
    (sha, ((HmRow.rowSlots ports).2[p]!)[k - 1]!.1, ((HmRow.rowSlots ports).2[p]!)[k - 1]!.2)⟩

def hmWire (ports : Array Port) (sha hm p : ℕ) : Wire :=
  ⟨(hm, p, 0), (sha, ((HmRow.rowSlots ports).2[p]!)[((HmRow.rowSlots ports).2[p]!).size - 1]!.1,
    ((HmRow.rowSlots ports).2[p]!)[((HmRow.rowSlots ports).2[p]!).size - 1]!.2)⟩

theorem forIn_range1_append {γ : Type} (G : ℕ → List γ) (n : ℕ) {f : ℕ → Array γ → Id (ForInStep (Array γ))}
    (hf : ∀ a, ∀ d, ∃ d', f a d = ForInStep.yield d' ∧ d'.toList = d.toList ++ G a) (d : Array γ) :
    (forIn [1:n] d f).toList = d.toList ++ (List.range' 1 (n - 1)).flatMap G := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := forIn_id_append G (List.range' 1 [1:n].size) (fun a _ d => hf a d) d
  simpa [Std.Legacy.Range.size] using this

/-- **The rows' wires**, port by port: compression `k`'s `cv` from compression `k − 1`'s output for `k ≥ 1`, then the
`hm96` slot's `cv` from the last compression's output. -/
theorem rowWires_eq (ports : Array Port) (sha hm : ℕ) :
    (HmRow.rowWires ports sha hm).toList = (List.range (HmRow.rowSlots ports).2.size).flatMap fun p =>
      (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)).map (chainWire ports sha p) ++
        [hmWire ports sha hm p] := by
  unfold HmRow.rowWires
  simp only
  show (forIn (m := Id) [:(HmRow.rowSlots ports).2.size] (#[] : Array Wire) _).toList = _
  refine Eq.trans (forIn_range_append (fun p =>
      List.map (chainWire ports sha p) (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)) ++
        [hmWire ports sha hm p]) _ (fun p _ d => ⟨_, rfl, ?_⟩) #[]) ?_
  · rw [Array.toList_push, ← List.append_assoc]
    refine (List.append_left_inj _).mpr ?_
    refine Eq.trans (forIn_range1_append (fun k => [chainWire ports sha p k]) _ (fun k d => ⟨_, rfl, ?_⟩) d) ?_
    · rw [Array.toList_push]
      rfl
    · rw [List.map_eq_flatMap]
  · simp

end FlockSoundness.Discharge.Hm
