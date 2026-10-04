/-! What a circuit is and what it computes, beside `circuit.py`, which implements it. A placeholder: the real
statement is today's `packages/verity/lean/Verity/Protocol/Circuit.lean`. -/

namespace Verity.Primitives

/-- The Boolean profile's gates, on bits. -/
inductive Op | xor | and | not

structure Circuit where
  inputs : Nat
  gates : List (Op × Nat × Nat)
  outputs : List Nat
  /-- What its first inputs hold in every execution; the rest vary. -/
  baseline : List Nat := []
  /-- Per input after the baseline that an earlier execution of the stream wrote: how many executions back, and which
  of its outputs. Where an input comes from, not what the circuit computes from it. -/
  state : List (Nat × Nat) := []

def Op.apply : Op → Nat → Nat → Nat
  | .xor, a, b => a ^^^ b
  | .and, a, b => a &&& b
  | .not, a, _ => a ^^^ 1

/-- Every wire's value, from the inputs after the baseline: what `transcript` in `circuit.py` returns. -/
def transcript (c : Circuit) (inputs : List Nat) : List Nat :=
  c.gates.foldl (fun wires (op, a, b) => wires ++ [op.apply (wires.getD a 0) (wires.getD b 0)]) (c.baseline ++ inputs)

end Verity.Primitives
