module

public import InferenceOnlyCheck.Check
public import InferenceOnlyCheck.Difftest
public import InferenceOnlyCheck.Desc
public import InferenceOnlyCheck.DescDifftest
public import InferenceOnlyCheck.DescLiveness
public import InferenceOnlyCheck.DescFuzz
public import InferenceOnlyCheck.Baseline
public import InferenceOnlyCheck.BaselineDifftest
public import InferenceOnlyCheck.Protocol
public import InferenceOnlyCheck.Count
public import InferenceOnlyCheck.CountDifftest

/-!
# The inference-only policy's check

`InferenceOnlyCheck.Baseline.check` (the `nci-check` exe) is the verifier's check that a declared division of a declaration into
isolation units meets `InferenceOnly.CompliesRate Γ α` (`Security/Definitions/InferenceOnly.lean`), for IUs aligned to the root's Calls, at
the cost of the descriptor: `InferenceOnlyCheck.Desc.check`, one summary per Definition, never a walk over the unrolled circuit
(AGENTS.md "Design rules"), then each θ entry's provenance, registered public data that the first record of the
deployment's registration log registers and never an earlier run's output, against the service's log
(`InferenceOnlyCheck.Baseline`; `InferenceOnlyCheck.BaselineDifftest` holds it to the Python reference).
`InferenceOnlyCheck.check` is the same check gate by gate, with the input format and its mapping to the definition's circuit in
`InferenceOnlyCheck.Check`'s docstring, kept as the reference: `InferenceOnlyCheck.Difftest` holds it to the Python reference, and
`InferenceOnlyCheck.DescDifftest` holds `InferenceOnlyCheck.Desc` to it and to the Python reference, `InferenceOnlyCheck.DescLiveness` to it on
Program shapes past anything `InferenceOnlyCheck.Desc` walks one by one, and `InferenceOnlyCheck.DescFuzz` decides on generated Programs the
statements their agreement rests on, and the agreement itself. Code: it reads the Program with Flock's
decoder (`Flock.Program`) and imports nothing else but Lean core. It stays experimental until its agreement with the
definition is proved.

`InferenceOnlyCheck.Protocol` is the policy's check as a core protocol, `verity/inference-only/v0` (the `inference-only` exe):
the inference-only policy's subprotocol of onsite and remote, on the core interface (`CoreProtocol`).

Flock's decoder doesn't check that a reference reads a leaf at the width it was committed with, which the
Python codec refuses as it decodes (`E_REINTERPRET`), so every Program the check reads goes through
`InferenceOnlyCheck.decodeProgram`, which refuses one that doesn't, with a message.

`InferenceOnlyCheck.Count.check` (`nci-check --count`) is the count of `(c, k)`-inference blocks: that a Boolean declaration's
computed gates divide into at most `n` proof blocks of a `(c, k)`-partition (`InferenceOnly.CompliesAtMost`), the blocks the
activations of the block Definitions the declaration names, with each block Definition's Boolean gates from the gate
table and its fresh bits from `InferenceOnlyCheck.Desc`'s summaries. It publishes only the judgment against `n`, `{complies}`;
the whole verdict, with the counts, is `InferenceOnlyCheck.Count.checkDetail`, for the private record. `InferenceOnlyCheck.CountDifftest`
holds both to the Python reference.
-/

@[expose] public section
