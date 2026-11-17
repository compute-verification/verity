module

public import Definitions.Claims.Sampling
public import Proofs.Flock.SamplingBoundary
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-! The input class the deployed verifier lays after the population, one stratum of a one-gate proof block per input gate. -/

@[expose] public section

namespace Claims.Sampling.InputSlot

open Claims.Sampling

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

def inputsAt (C : Prog) : List ℕ := (List.range C.gates.length).filter fun g => decide (C.gates[g]? = some .input)

def inputClass (C : Prog) : List (List (List ℕ)) := if inputsAt C = [] then [] else [(inputsAt C).map fun g => [g]]

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

theorem withInputs_getElem?_of_lt (P : Partition) (C : Prog) {v : ℕ} (hv : v < P.blocks.length) :
    (withInputs P C).blocks[v]? = P.blocks[v]? := by
  rw [blocks_withInputs, List.getElem?_append_left hv]

theorem withInputs_getElem?_input (P : Partition) (C : Prog) {i g : ℕ} (hi : (inputsAt C)[i]? = some g) :
    (withInputs P C).blocks[P.blocks.length + i]? = some [g] := by
  rw [blocks_withInputs, List.getElem?_append_right (by omega)]
  simp [hi]

theorem faulty_withInputs_iff {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {P : Partition} {w : Wit} {i g : ℕ}
    (hi : (inputsAt s.prog)[i]? = some g) :
    Faulty H Cm s (withInputs P s.prog) w (P.blocks.length + i) ↔ w.val H Cm s g = none :=
  faulty_input_iff (withInputs_getElem?_input P s.prog hi) (mem_inputsAt.1 (List.mem_of_getElem? hi))

theorem input_mem_withInputs (P : Partition) {C : Prog} {g : ℕ} (hg : C.gates[g]? = some .input) :
    g ∈ (withInputs P C).gates := by
  unfold Partition.gates
  rw [blocks_withInputs]
  exact List.mem_flatten.2 ⟨[g], List.mem_append_right _ (List.mem_map.2 ⟨g, mem_inputsAt.2 hg, rfl⟩),
    List.mem_singleton_self _⟩

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
