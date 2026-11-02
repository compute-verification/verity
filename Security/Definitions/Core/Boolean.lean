import Definitions.Core.Circuit

/-!
# The Boolean profile

The circuits C-Flock proves are Boolean: input gates, and AND, XOR and NOT gates over earlier gates, and rows, the AND
of the XOR of some earlier gates and the XOR of others (a row as the verifier parses it, PROTOCOL.md §16.5; the compiled
lowering reads a template's rows as these gates, `Lowering.lean`). A Boolean circuit is a circuit whose gates are all one
bit wide (`toCircuit`), so every definition about circuits applies to it as it is.

This is the one Boolean circuit: C-Flock's audit (`Definitions.Flock.Circuit`) partitions it. It keeps the names
C-Flock's guarantees read it by (`FlockSoundness.Audit`), so their records name these declarations.
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

theorem args_of_isInput {o : Op N} (h : o.isInput = true) : o.args = [] := by
  cases o <;> simp_all [isInput, args]

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

/-- The circuit's evaluation on inputs `a`. -/
def eval (a : Fin C.N → Bool) : Fin C.N → Bool :=
  C.run fun g => if (C.op g).isInput then some (a g) else none

/-- A Boolean circuit is a circuit of one-bit gates. -/
def toCircuit (B : Circuit) : Verity.Circuit where
  N := B.N
  w _ := 1
  isInput g := (B.op g).isInput
  reads g := (B.op g).args.toFinset
  reads_lt g a ha := B.topo g a (List.mem_toFinset.1 ha)
  input_reads g h := by simp [Op.args_of_isInput h]
  op g v := BitVec.ofBool ((B.op g).apply fun a => (v a).getLsbD 0)
  op_congr g v v' h := by
    rw [Op.apply_congr (B.op g) fun x hx => by rw [h x (List.mem_toFinset.2 hx)]]
  outputs := B.outputs

end Circuit

end FlockSoundness.Audit
