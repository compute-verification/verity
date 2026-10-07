import Evaluate.Prims
import Evaluate.Walk
import Evaluate.Serve

/-!
# `verity_evaluate`: the descriptor v1 evaluator in Lean

The Lean form of `verity.ml.circuits.evaluation.reference.evaluate` on a Program (`verity/ml/circuits/PROTOCOL.md`): decode,
bind the run inputs, evaluate gate by gate in canonical order, return the root's outputs or refuse. The Program is read
through C-Flock's verifier (`Flock.Program`), which evaluates no primitive; `Evaluate.Prims` adds the primitives, so far the
Boolean profile (E0). `verity-evaluate` serves it as JSON lines (`Evaluate.Serve`), and the suite beside this package holds
it to the Python, output for output and refusal for refusal.
-/
