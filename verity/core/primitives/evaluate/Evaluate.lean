import Evaluate.Scalar
import Evaluate.Fp32
import Evaluate.Casts
import Evaluate.TC.Spec
import Evaluate.TC.Fp8Atom
import Evaluate.TC.Fp4
import Evaluate.TC.Dots
import Evaluate.Prims
import Evaluate.Walk
import Evaluate.Serve

/-!
# `verity_evaluate`: the descriptor v1 evaluator in Lean

The Lean form of `verity.ml.circuits.evaluation.reference.evaluate` on a Program (`verity/ml/circuits/PROTOCOL.md`): decode,
bind the run inputs, evaluate gate by gate in canonical order, return the root's outputs or refuse. The Program is read
through C-Flock's verifier (`Flock.Program`), which evaluates no primitive; `Evaluate.Prims` adds the primitives: the
Boolean profile (E0), the constants and the scalar primitives (E1, `Evaluate.Scalar`), the FP32 and BF16 arithmetic and the
casts that read no table (E2, `Evaluate.Fp32`, `Evaluate.Casts`), the tensor-core dot steps (E3, `Evaluate.TC`), and the rest
of the registry (E4: `Evaluate.Words`, `Evaluate.Libdevice`, `Evaluate.Mufu`, `Evaluate.Decimal`, `Evaluate.Act`), each an
executable transcription of the Python that names the Security definition it must equal, or that Security has none yet.
`verity-evaluate` serves it through the difftest harness
(`Evaluate.Serve`), and the suite beside this package holds it to the Python, output for output and refusal for refusal.
-/
