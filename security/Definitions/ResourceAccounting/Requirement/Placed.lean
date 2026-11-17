module

public import Definitions.ResourceAccounting.Requirement
public import Definitions.Pouw.Confidence

/-! A registered circuit's placed execution as a run of the actual-circuit model; what such a node is credited is completeness, not what a node that passes PoUW ran. -/

@[expose] public section

namespace ResourceAccounting.Requirement

open Finset

variable {n : ℕ} {L : Type}

namespace Placement

variable (W : Wiring (Fin n)) (p : Placement L (Fin n))

/-- A gate's operation gives its value: the wiring carries no semantics. -/
@[reducible] def gates : Verity.Circuit where
  N := n
  w := W.width
  isInput g := decide (g ∉ p.ran ∨ g ∈ W.registered ∨ g ∈ W.inputs)
  reads g := if g ∉ p.ran ∨ g ∈ W.registered ∨ g ∈ W.inputs then ∅ else (W.uses p.value g).filter (· < g)
  reads_lt g a ha := by
    split_ifs at ha
    · simp at ha
    · exact (Finset.mem_filter.mp ha).2
  input_reads g hg := by simp only [decide_eq_true_eq] at hg; simp [hg]
  op g _ := BitVec.ofNat (W.width g) (p.value g)
  op_congr _ _ _ _ := rfl
  outputs := ∅

/-- Each gate is in cell `g`, its index, so no two gates share a cell whatever the ticks. -/
@[reducible] def placed : Placed L where
  toCircuit := p.gates W
  x := p.leaf
  t := p.tick
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun q => q.2.1) h)

open Classical in
noncomputable def heldAt (y : L) (s : ℕ) : Finset (Fin (p.placed W).N) :=
  univ.filter fun a => (p.placed W).x a = y ∧
    (((p.placed W).isInput a = true ∧ s = 0) ∨
      (((p.placed W).isInput a = true ∨ (p.placed W).t a < s) ∧
        ∃ g, a ∈ (p.placed W).reads g ∧ (p.placed W).x g = y ∧ s ≤ (p.placed W).t g))

@[reducible] noncomputable def actual : Actual.Run L where
  A := p.placed W
  val g := BitVec.ofNat (W.width g) (p.value g)
  faithful _ _ := rfl
  held := p.heldAt W
  holding := by
    classical
    refine ⟨fun y s a ha => ?_, fun a ha => ?_, fun g a ha hx s hs hg => ?_⟩
    · simp only [heldAt, mem_filter, mem_univ, true_and] at ha
      rcases ha with ⟨hy, ⟨hi, -⟩ | ⟨hi, -⟩⟩
      · exact ⟨hy, Or.inl hi⟩
      · exact ⟨hy, hi⟩
    · simp only [heldAt, mem_filter, mem_univ, true_and]
      exact Or.inl ⟨ha, trivial⟩
    · simp only [heldAt, mem_filter, mem_univ, true_and]
      exact ⟨hx, Or.inr ⟨hs, g, ha, rfl, hg⟩⟩

end Placement

end ResourceAccounting.Requirement
