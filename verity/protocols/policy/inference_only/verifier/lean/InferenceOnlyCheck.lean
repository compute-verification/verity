module

public import InferenceOnlyCheck.Check
public import InferenceOnlyCheck.Difftest
public import InferenceOnlyCheck.Desc
public import InferenceOnlyCheck.DescDifftest
public import InferenceOnlyCheck.DescLiveness
public import InferenceOnlyCheck.DescFuzz
public import InferenceOnlyCheck.Theta
public import InferenceOnlyCheck.RecordText
public import InferenceOnlyCheck.Record
public import InferenceOnlyCheck.RecordCount
public import InferenceOnlyCheck.RecordDifftest
public import InferenceOnlyCheck.ThetaDifftest
public import InferenceOnlyCheck.Protocol
public import InferenceOnlyCheck.Count
public import InferenceOnlyCheck.CountDifftest
public import InferenceOnlyCheck.DefinitionDigest

/-!
# The inference-only policy's check

`InferenceOnlyCheck.Theta.check` (the `inference-only-check` exe) is the verifier's check that a declared division of a declaration into
inference blocks meets `InferenceOnly.CompliesRate Γ α` (`Security/Definitions/InferenceOnly.lean`), for inference blocks aligned to the root's Calls, at
the cost of the descriptor: `InferenceOnlyCheck.Desc.check`, one summary per Definition, never a walk over the unrolled circuit
(AGENTS.md "Design rules"), then each θ entry's provenance, registered public data that the first record of the
deployment's registration log registers and never an earlier run's output, against the service's log
(`InferenceOnlyCheck.Theta`; `InferenceOnlyCheck.ThetaDifftest` holds it to the Python reference).
`InferenceOnlyCheck.check` is the same check gate by gate, with the input format and its mapping to the definition's circuit in
`InferenceOnlyCheck.Check`'s docstring, kept as the reference: `InferenceOnlyCheck.Difftest` holds it to the Python reference, and
`InferenceOnlyCheck.DescDifftest` holds `InferenceOnlyCheck.Desc` to it and to the Python reference, `InferenceOnlyCheck.DescLiveness` to it on
Program shapes past anything `InferenceOnlyCheck.Desc` walks one by one, and `InferenceOnlyCheck.DescFuzz` decides on generated Programs the
statements their agreement rests on, and the agreement itself. Code: it reads the Program with Flock's
decoder (`Flock.Program`) and imports nothing else but Lean core. It stays experimental until its agreement with the
definition is proved.

`InferenceOnlyCheck.Record` is the policy's check of an execution phase's circuit record, whose Program holds no registered
weight and no wire between its units: it reads the record's wires through SPOT's reader (`Spot.Wires`, with
`Flock.TemplateQuery` the package's other import for it), classes every input row of every unit once, takes one
summary and decides the rate rule on it, counting as θ only what the registration log registers at the start of the
deployment. What the record's text declares, as functions of the Program and the text alone (the text's wiring,
each unit's block, what each row is declared to hold), is `InferenceOnlyCheck.RecordText`, which that check is built on and
which imports nothing else of this package. `InferenceOnlyCheck.RecordCount` is the count of `(c, k)`-inference blocks on that
summary. All three are a library: no executable, protocol or verdict reads them. `InferenceOnlyCheck.RecordDifftest` holds them to the Python reference, and the
wiring the form reads to the one SPOT's reader reads.

`InferenceOnlyCheck.Protocol` is the policy's check as a core protocol, `verity/inference-only/v0` (the `inference-only` exe):
the inference-only policy's subprotocol of onsite and remote, on the core interface (`CoreProtocol`).

Flock's decoder doesn't check that a reference reads a leaf at the width it was committed with, which the
Python codec refuses as it decodes (`E_REINTERPRET`), so every Program the check reads goes through
`InferenceOnlyCheck.decodeProgram`, which refuses one that doesn't, with a message.

`InferenceOnlyCheck.Count.check` (`inference-only-check --count`) is the count of `(c, k)`-inference blocks: that a Boolean declaration's
computed gates divide into at most `n` proof blocks of a `(c, k)`-partition (`InferenceOnly.CompliesAtMost`), the blocks the
activations of the block Definitions the declaration names, with each block Definition's Boolean gates from the gate
table and its fresh bits from `InferenceOnlyCheck.Desc`'s summaries. It publishes only the judgment against `n`, `{complies}`;
the whole verdict, with the counts, is `InferenceOnlyCheck.Count.checkDetail`, for the private record. `InferenceOnlyCheck.CountDifftest`
holds both to the Python reference.

`InferenceOnlyCheck.DefinitionDigest` holds a Definition's digests, read off its Program's descriptor with no decode: the circuit
format's own (`definitionDigest`), over the Definition's body, and the closure digest (`closureDigest`), over its body
and every body it reaches, which a check that takes a Definition by digest pins. It reads none of the modules above;
the period check takes its slot entries' Definitions by it.
-/

@[expose] public section
