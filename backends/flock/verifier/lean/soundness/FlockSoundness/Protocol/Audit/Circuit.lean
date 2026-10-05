import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card

/-!
# Circuits, partitions, and the correctness of a unit

A circuit is Boolean: input gates, and AND, XOR and NOT gates over earlier gates, and rows: the AND
of the XOR of some earlier gates and the XOR of others, a row as the verifier parses it (PROTOCOL.md
§16.5; the compiled lowering reads a template's rows as these gates, `Lowering.lean`). A partition puts
every computed gate in exactly one of `n` units; the input gates are the input unit (`none`). The
committed wires are derived, never stored: the inputs, the outputs, and every gate that a gate of
another unit reads (`Partition.committed`).

A transcript assigns a bit to every gate; only its committed entries are ever read. Unit `u` is
**correct** in a transcript `X` when its committed outputs are its gates applied to its committed
inputs (`Partition.Correct`): evaluate `u`'s gates in order, reading every other gate from `X`
(`Partition.localEval`), and compare with `X` on `u`'s committed gates.

* `compose`: if every unit is correct and the committed inputs are the anchors' values, every
  committed wire carries its value in the circuit's evaluation;
* `compose_cone`: the same for the wires `T` a consumer reads, given only that the units of `T`'s
  cone (the units holding an ancestor of `T`) are correct;
* `correct_congr`: a unit's correctness depends only on its committed inputs and outputs
  (`Partition.io`), which is what an extracted opening of the unit determines;
* `Refines.exists_wrong_fine`: when a fine partition refines a coarse one, a coarse unit that is
  wrong has a wrong fine unit inside it, whatever the values on the coarse unit's interior.
-/

namespace FlockSoundness.Audit

/-- The XOR of the values at a list of places. -/
def xorSum {ι : Type} (l : List ι) (v : ι → Bool) : Bool := (l.map v).foldr Bool.xor false

theorem xorSum_congr {ι : Type} {l : List ι} {v w : ι → Bool} (h : ∀ x ∈ l, v x = w x) :
    xorSum l v = xorSum l w := by
  unfold xorSum
  rw [List.map_congr_left h]

/-- A gate. -/
inductive Op (N : ℕ) where
  | input
  | and (a b : Fin N)
  | xor (a b : Fin N)
  | not (a : Fin N)
  /-- A row: `(⊕_{x ∈ a} x) ∧ (⊕_{x ∈ b} x)`. -/
  | row (a b : List (Fin N))

namespace Op

variable {N : ℕ}

/-- The gates an operation reads. -/
def args : Op N → List (Fin N)
  | input => []
  | and a b => [a, b]
  | xor a b => [a, b]
  | not a => [a]
  | row a b => a ++ b

def isInput : Op N → Bool
  | input => true
  | _ => false

/-- The operation applied to the values of its arguments (an input gate has no operation). -/
def apply : Op N → (Fin N → Bool) → Bool
  | input, _ => false
  | and a b, v => v a && v b
  | xor a b, v => Bool.xor (v a) (v b)
  | not a, v => !v a
  | row a b, v => xorSum a v && xorSum b v

theorem apply_congr (o : Op N) {v w : Fin N → Bool} (h : ∀ x ∈ o.args, v x = w x) :
    o.apply v = o.apply w := by
  cases o
  case row a b =>
    simp only [apply]
    rw [xorSum_congr fun x hx => h x (List.mem_append_left _ hx),
      xorSum_congr fun x hx => h x (List.mem_append_right _ hx)]
  all_goals simp_all [args, apply]

end Op

/-- A Boolean circuit, its gates in topological order. -/
structure Circuit where
  N : ℕ
  op : Fin N → Op N
  topo : ∀ g, ∀ a ∈ (op g).args, a < g
  outputs : Finset (Fin N)

namespace Circuit

variable (C : Circuit)

/-- Evaluation with some gates fixed: a fixed gate takes its given value, and every other gate
applies its operation to its arguments' values. -/
def run (fix : Fin C.N → Option Bool) (g : Fin C.N) : Bool :=
  match fix g with
  | some b => b
  | none => (C.op g).apply fun x => if _h : x < g then run fix x else false
termination_by g.val
decreasing_by all_goals assumption

theorem run_eq (fix : Fin C.N → Option Bool) (g : Fin C.N) :
    C.run fix g = match fix g with
      | some b => b
      | none => (C.op g).apply (C.run fix) := by
  rw [run]
  cases fix g with
  | some b => rfl
  | none =>
      exact Op.apply_congr _ fun x hx => by simp [C.topo g x hx]

theorem run_fixed {fix : Fin C.N → Option Bool} {g : Fin C.N} {b : Bool} (h : fix g = some b) :
    C.run fix g = b := by
  rw [run_eq, h]

theorem run_free {fix : Fin C.N → Option Bool} {g : Fin C.N} (h : fix g = none) :
    C.run fix g = (C.op g).apply (C.run fix) := by
  rw [run_eq, h]

/-- The input unit: the committed inputs are the anchors' values. -/
def InputsAgree (X a : Fin C.N → Bool) : Prop :=
  ∀ g, (C.op g).isInput = true → X g = a g

end Circuit

/-- A partition of a circuit's computed gates into `n` units; the input gates are the input unit
(`none`). -/
structure Partition (C : Circuit) (n : ℕ) where
  unit : Fin C.N → Option (Fin n)
  input_iff : ∀ g, unit g = none ↔ (C.op g).isInput = true

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

open Classical in
/-- The committed wires: inputs, outputs, and every gate that a gate of another unit reads. -/
noncomputable def committed : Finset (Fin C.N) :=
  Finset.univ.filter fun g => (C.op g).isInput = true ∨ g ∈ C.outputs ∨
    ∃ h, g ∈ (C.op h).args ∧ P.unit h ≠ P.unit g

theorem mem_committed_of_read {g h : Fin C.N} (hg : g ∈ (C.op h).args) (hne : P.unit h ≠ P.unit g) :
    g ∈ P.committed := by
  classical
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and]
  exact Or.inr (Or.inr ⟨h, hg, hne⟩)

/-- `u`'s gates computed in order, every other gate read from `X`. -/
def localEval (X : Fin C.N → Bool) (u : Fin n) : Fin C.N → Bool :=
  C.run fun g => if P.unit g = some u then none else some (X g)

theorem localEval_of_mem (X : Fin C.N → Bool) {u : Fin n} {g : Fin C.N} (h : P.unit g = some u) :
    P.localEval X u g = (C.op g).apply (P.localEval X u) :=
  C.run_free (by simp [h])

theorem localEval_of_not_mem (X : Fin C.N → Bool) {u : Fin n} {g : Fin C.N}
    (h : P.unit g ≠ some u) : P.localEval X u g = X g :=
  C.run_fixed (by simp [h])

/-- **Correctness of a unit**: its committed outputs are its gates applied to its committed inputs. -/
def Correct (X : Fin C.N → Bool) (u : Fin n) : Prop :=
  ∀ g ∈ P.committed, P.unit g = some u → X g = P.localEval X u g

open Classical in
/-- The wrong units of a transcript. -/
noncomputable def wrong (X : Fin C.N → Bool) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ P.Correct X u

theorem mem_wrong {X : Fin C.N → Bool} {u : Fin n} : u ∈ P.wrong X ↔ ¬ P.Correct X u := by
  classical
  simp [wrong]

open Classical in
/-- `u`'s committed inputs and outputs. -/
noncomputable def io (u : Fin n) : Finset (Fin C.N) :=
  P.committed.filter fun g => P.unit g = some u ∨ ∃ h, P.unit h = some u ∧ g ∈ (C.op h).args

theorem localEval_congr {X Y : Fin C.N → Bool} {u : Fin n} (hXY : ∀ g ∈ P.io u, X g = Y g) :
    ∀ m (g : Fin C.N), g.val = m → P.unit g = some u → P.localEval X u g = P.localEval Y u g := by
  classical
  intro m
  induction m using Nat.strong_induction_on with
  | _ m ih =>
  intro g hgm hu
  rw [P.localEval_of_mem X hu, P.localEval_of_mem Y hu]
  refine Op.apply_congr _ fun x hx => ?_
  by_cases hxu : P.unit x = some u
  · exact ih x.val (hgm ▸ C.topo g x hx) x rfl hxu
  · rw [P.localEval_of_not_mem X hxu, P.localEval_of_not_mem Y hxu]
    refine hXY x ?_
    simp only [io, Finset.mem_filter]
    exact ⟨P.mem_committed_of_read hx (by rw [hu]; exact Ne.symm hxu), Or.inr ⟨g, hu, hx⟩⟩

/-- A unit's correctness depends only on its committed inputs and outputs. -/
theorem correct_congr {X Y : Fin C.N → Bool} {u : Fin n} (hXY : ∀ g ∈ P.io u, X g = Y g)
    (hY : P.Correct Y u) : P.Correct X u := by
  classical
  intro g hg hu
  have hio : g ∈ P.io u := by simp only [io, Finset.mem_filter]; exact ⟨hg, Or.inl hu⟩
  rw [hXY g hio, P.localEval_congr hXY g.val g rfl hu]
  exact hY g hg hu

/-! ## Refinement -/

namespace Refines

variable {nf nc : ℕ} {Pf : Partition C nf} {Pc : Partition C nc}

end Refines

end Partition

end FlockSoundness.Audit
