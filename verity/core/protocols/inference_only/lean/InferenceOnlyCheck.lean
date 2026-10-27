import InferenceOnlyCheck.Check
import InferenceOnlyCheck.Difftest
import InferenceOnlyCheck.Desc
import InferenceOnlyCheck.DescDifftest
import InferenceOnlyCheck.DescLiveness
import InferenceOnlyCheck.DescFuzz
import InferenceOnlyCheck.Baseline
import InferenceOnlyCheck.BaselineDifftest

/-!
# The inference-only policy's check

`InferenceOnlyCheck.Baseline.check` (the `inference-only-check` exe) is the verifier's check that a declared division of a declaration into
isolation units meets `Nci.Complies Γ α` (`Security/Definitions/Nci.lean`), for IUs aligned to the root's Calls, at the
cost of the descriptor: `InferenceOnlyCheck.Desc.check`, one summary per Definition, never a walk over the unrolled circuit
(AGENTS.md "Design rules"), then each θ entry's provenance, registered public data that the first record of the
deployment's registration log registers and never an earlier run's output, against the service's log
(`InferenceOnlyCheck.Baseline`; `InferenceOnlyCheck.BaselineDifftest` holds it to the Python reference).
`InferenceOnlyCheck.check` is the same check gate by gate, with the input format and its mapping to the definition's circuit in
`InferenceOnlyCheck.Check`'s docstring, kept as the reference: `InferenceOnlyCheck.Difftest` holds it to the Python reference, and
`InferenceOnlyCheck.DescDifftest` holds `InferenceOnlyCheck.Desc` to it and to the Python reference, `InferenceOnlyCheck.DescLiveness` to it on
Program shapes past anything `InferenceOnlyCheck.Desc` walks one by one, and `InferenceOnlyCheck.DescFuzz` decides on generated Programs the
statements their agreement rests on, and the agreement itself. Code: it reads the Program with C-Flock's
decoder (`Flock.Program`) and imports nothing else but Lean core. It stays experimental until its agreement with the
definition is proved.

C-Flock's decoder doesn't check that a reference reads a leaf at the width it was committed with, which the
Python codec refuses as it decodes (`E_REINTERPRET`), so every Program the check reads goes through
`InferenceOnlyCheck.decodeProgram`, which refuses one that doesn't, with a message.
-/
