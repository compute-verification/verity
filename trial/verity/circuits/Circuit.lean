/-! What a circuit is and what it computes, beside `circuit.py`, which implements it. A placeholder: the real
statement is today's `packages/verity/lean/Verity/Protocol/Circuit.lean`. -/

namespace Verity.Circuits

inductive Op | add | mul | xor

structure Circuit where
  inputs : Nat
  gates : List (Op × Nat × Nat)
  outputs : List Nat

def Op.apply : Op → Nat → Nat → Nat
  | .add, a, b => (a + b) % 2 ^ 32
  | .mul, a, b => (a * b) % 2 ^ 32
  | .xor, a, b => a ^^^ b

/-- Every wire's value: what `trace` in `circuit.py` returns. -/
def trace (c : Circuit) (inputs : List Nat) : List Nat :=
  c.gates.foldl (fun wires (op, a, b) => wires ++ [op.apply (wires.getD a 0) (wires.getD b 0)]) inputs

end Verity.Circuits
