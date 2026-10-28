import Definitions.Accounting.Requirement
import Definitions.Accounting.Confidence

/-!
# A placed execution as an actual run (trusted)

A declared circuit `W`'s placed execution (`Accounting.Requirement.Placement`) as a run of the actual-circuit model
(`Accounting.Placed`, `Actual.Run`), so that what PoUW credits and the help it bounds (`Accounting.PouwWork`'s `credits`
and `help`) read `W`'s gates. `W`'s values are `Fin n`, the IR's canonical order, in which an operand comes before its
gate (`Wiring.Topological`).

* `gates`: the circuit (`Verity.Circuit`) of `W`'s values, each of its width. The values the execution produces but the
  registered ones and the run's inputs are its computed gates, each reading the operands it uses (`Wiring.uses`, at the
  execution's values); the registered values, the run's inputs and the values the execution doesn't produce are input
  gates, which read nothing (one the execution doesn't produce is read by no gate it runs, so it is in no region and
  adds nothing to what crosses). A gate's operation gives its value: the wiring carries no semantics, and that the
  values are `W`'s is what the one-stage proofs show.
* `placed`: `gates` placed (`Placed`): each gate at its device (`Placement.leaf`), at the tick its device begins its
  engine step (`Placement.tick`), and in cell `g`, its index, so no two gates share a cell and `cell_inj` holds whatever
  the ticks.
* `actual`: the run (`Actual.Run`): the values, and each device holding what `Placed.Holding` asks and no more
  (`heldAt`): its inputs at tick 0, and each value it reads from a gate it placed, from the tick after the placement
  (from 0 for an input) to the read.

The three are reducible, so that the run's gates are `Fin n` to the elaborator as they are to the kernel. This is the
run of a node that runs `W` as placed, leaf, ticks and values as given, its operations constants, with no outputs and no
coins among its gates: `Definitions/Accounting/Guarantees/RequirementPouw.lean` states what such a node is credited and
helped, which is completeness, not what a node that passes PoUW ran.
-/

namespace Accounting.Requirement

open Finset

variable {n : ℕ} {L : Type}

namespace Placement

variable (W : Wiring (Fin n)) (p : Placement L (Fin n))

/-- **The circuit of `W`'s values**: the values the execution produces but the registered ones and the run's inputs are
computed gates, each reading the earlier operands it uses, and every other value is an input gate. A gate's operation
gives its value. -/
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

/-- **`W`'s gates placed**: each at its device, at the tick its device begins its engine step, in cell `g`. -/
@[reducible] def placed : Placed L where
  toCircuit := p.gates W
  x := p.leaf
  t := p.tick
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun q => q.2.1) h)

open Classical in
/-- What device `y` holds at the start of tick `s`: its input gates at tick 0, and each gate of `y` that a gate of `y`
reads, from the tick after its placement (from 0 for an input) to the read. -/
noncomputable def heldAt (y : L) (s : ℕ) : Finset (Fin (p.placed W).N) :=
  univ.filter fun a => (p.placed W).x a = y ∧
    (((p.placed W).isInput a = true ∧ s = 0) ∨
      (((p.placed W).isInput a = true ∨ (p.placed W).t a < s) ∧
        ∃ g, a ∈ (p.placed W).reads g ∧ (p.placed W).x g = y ∧ s ≤ (p.placed W).t g))

/-- **The actual run of `W`'s placed execution**: its gates placed, their values, and what each device holds. -/
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

end Accounting.Requirement
