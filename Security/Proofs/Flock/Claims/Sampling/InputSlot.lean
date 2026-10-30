import Proofs.Flock.SamplingBoundary

/-!
# Input slots

The deployed verifier lays an input class after the population (`Slots.layout`): one stratum of a one-gate proof
block per input gate, in gate order, which `Slots.checked` lists on every draw. No run opens an input row on its own:
the session proves the drawn units (`HmRow.drawn`), and registration opens no row, so an input row is opened only
through a drawn unit that reads it.

* `faulty_input_iff`: at a slot whose proof block is `[p]`, `p` an input gate, the slot is faulty exactly when the
  witness opens no value at `p`. An input's computed value is its committed value, so an opened input
  is correct under every interior, and an unopened one under none.
* `withInputs P C`: `P`'s classes, then the input class of `C` (`inputClass`). `P`'s slots keep their indices
  (`withInputs_getElem?_of_lt`), slot `|P| + i` is the `i`th input gate's (`withInputs_getElem?_input`), and it is
  faulty exactly when that gate's value is unopened (`faulty_withInputs_iff`).
* `InScope`'s input clause: every input gate of `C` is in a proof unit of `withInputs P C` (`input_mem_withInputs`), so
  where `P` covers every other gate but the constants, `withInputs P C` covers every gate but the constants
  (`covers_withInputs`), `Partitions`' covering clause.
-/

namespace Claims.Sampling.InputSlot

open Claims.Sampling

/-- **An input slot is faulty exactly when its value is unopened.** -/
theorem faulty_input_iff {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {P : Partition} {w : Wit} {v p : ℕ}
    (hv : P.blocks[v]? = some [p]) (hp : s.prog.gates[p]? = some .input) :
    Faulty H Cm s P w v ↔ w.val H Cm s p = none := by
  have hcorr : ∀ ι : ℕ → Bool,
      s.prog.Correct (s.prog.refIn [p] (w.val H Cm s) ι) [p] ↔ w.val H Cm s p ≠ none := by
    intro ι
    have href : s.prog.refIn [p] (w.val H Cm s) ι p = w.val H Cm s p := Boundary.ref_input s.prog hp
    have hrun : s.prog.runIn (s.prog.refIn [p] (w.val H Cm s) ι) [p] p = w.val H Cm s p := by
      rw [Boundary.runIn_eq, hp]
      simpa [BGate.val] using href
    constructor
    · intro h
      obtain ⟨b, hb, -⟩ := (h p (List.mem_singleton_self p)).2 (Boundary.boundary_input s.prog hp)
      rw [href] at hb
      simp [hb]
    · intro h g hg
      obtain rfl : g = p := List.mem_singleton.1 hg
      refine ⟨fun x hx i hi _ => ?_, fun _ => ?_⟩
      · rw [hp] at hx
        obtain rfl := Option.some.inj hx
        simp [BGate.reads] at hi
      · cases hval : w.val H Cm s g with
        | none => exact absurd hval h
        | some b => exact ⟨b, by rw [href, hval], by rw [hrun, hval]⟩
  unfold Faulty
  constructor
  · intro hf
    by_contra hne
    exact hf ⟨[p], hv, fun _ => false, (hcorr _).2 hne⟩
  · rintro hnone ⟨u, hu, ι, hι⟩
    rw [hv] at hu
    obtain rfl := Option.some.inj hu
    exact (hcorr ι).1 hι hnone

/-- The input gates of `C`, in gate order. -/
def inputsAt (C : Prog) : List ℕ := (List.range C.gates.length).filter fun g => decide (C.gates[g]? = some .input)

/-- **The input class**, as the deployed verifier lays it: one stratum of a one-gate proof block per input gate; no
stratum when there is no input gate. -/
def inputClass (C : Prog) : List (List (List ℕ)) := if inputsAt C = [] then [] else [(inputsAt C).map fun g => [g]]

/-- `P`'s classes, then `C`'s input class. -/
def withInputs (P : Partition) (C : Prog) : Partition := ⟨P.classes ++ [inputClass C]⟩

theorem mem_inputsAt {C : Prog} {g : ℕ} : g ∈ inputsAt C ↔ C.gates[g]? = some .input := by
  simp only [inputsAt, List.mem_filter, List.mem_range, decide_eq_true_eq]
  exact ⟨fun h => h.2, fun h => ⟨(List.getElem?_eq_some_iff.1 h).1, h⟩⟩

theorem inputClass_flatten (C : Prog) : (inputClass C).flatten = (inputsAt C).map fun g => [g] := by
  unfold inputClass
  split <;> simp_all

theorem blocks_withInputs (P : Partition) (C : Prog) :
    (withInputs P C).blocks = P.blocks ++ (inputsAt C).map fun g => [g] := by
  simp [withInputs, Partition.blocks, inputClass_flatten]

/-- `P`'s slots keep their indices. -/
theorem withInputs_getElem?_of_lt (P : Partition) (C : Prog) {v : ℕ} (hv : v < P.blocks.length) :
    (withInputs P C).blocks[v]? = P.blocks[v]? := by
  rw [blocks_withInputs, List.getElem?_append_left hv]

/-- Slot `|P| + i` is the `i`th input gate's. -/
theorem withInputs_getElem?_input (P : Partition) (C : Prog) {i g : ℕ} (hi : (inputsAt C)[i]? = some g) :
    (withInputs P C).blocks[P.blocks.length + i]? = some [g] := by
  rw [blocks_withInputs, List.getElem?_append_right (by omega)]
  simp [hi]

/-- **An input slot of `withInputs P s.prog` is faulty exactly when its gate's value is unopened.** -/
theorem faulty_withInputs_iff {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {P : Partition} {w : Wit} {i g : ℕ}
    (hi : (inputsAt s.prog)[i]? = some g) :
    Faulty H Cm s (withInputs P s.prog) w (P.blocks.length + i) ↔ w.val H Cm s g = none :=
  faulty_input_iff (withInputs_getElem?_input P s.prog hi) (mem_inputsAt.1 (List.mem_of_getElem? hi))

/-- **`InScope`'s input clause**: every input gate of `C` is in a proof unit of `withInputs P C`. -/
theorem input_mem_withInputs (P : Partition) {C : Prog} {g : ℕ} (hg : C.gates[g]? = some .input) :
    g ∈ (withInputs P C).gates := by
  unfold Partition.gates
  rw [blocks_withInputs]
  exact List.mem_flatten.2 ⟨[g], List.mem_append_right _ (List.mem_map.2 ⟨g, mem_inputsAt.2 hg, rfl⟩),
    List.mem_singleton_self _⟩

/-- **`Partitions`' covering clause at `withInputs P C`**, where `P` covers every gate but the constants and the
inputs. -/
theorem covers_withInputs (P : Partition) (C : Prog)
    (h : ∀ g < C.gates.length, (∀ b, C.gates[g]? ≠ some (.const b)) → C.gates[g]? ≠ some .input → g ∈ P.gates) :
    ∀ g < C.gates.length, (∀ b, C.gates[g]? ≠ some (.const b)) → g ∈ (withInputs P C).gates := by
  intro g hg hc
  by_cases hin : C.gates[g]? = some .input
  · exact input_mem_withInputs P hin
  · have hP := h g hg hc hin
    unfold Partition.gates at hP ⊢
    rw [blocks_withInputs, List.flatten_append]
    exact List.mem_append_left _ hP

end Claims.Sampling.InputSlot
