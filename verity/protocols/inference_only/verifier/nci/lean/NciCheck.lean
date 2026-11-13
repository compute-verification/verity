module

public import NciCheck.Check
public import NciCheck.Difftest
public import NciCheck.Desc
public import NciCheck.DescDifftest
public import NciCheck.DescLiveness
public import NciCheck.DescFuzz
public import NciCheck.Baseline
public import NciCheck.BaselineDifftest
public import NciCheck.Protocol
public import NciCheck.Count
public import NciCheck.CountDifftest
public import NciCheck.DefinitionDigest

/-!
# The NCI policy's check

`NciCheck.Baseline.check` (the `nci-check` exe) is the verifier's check that a declared division of a declaration into
isolation units meets `Nci.CompliesRate Γ α` (`Security/Definitions/Nci.lean`), for IUs aligned to the root's Calls, at
the cost of the descriptor: `NciCheck.Desc.check`, one summary per Definition, never a walk over the unrolled circuit
(AGENTS.md "Design rules"), then each θ entry's provenance, registered public data that the first record of the
deployment's registration log registers and never an earlier run's output, against the service's log
(`NciCheck.Baseline`; `NciCheck.BaselineDifftest` holds it to the Python reference).
`NciCheck.check` is the same check gate by gate, with the input format and its mapping to the definition's circuit in
`NciCheck.Check`'s docstring, kept as the reference: `NciCheck.Difftest` holds it to the Python reference, and
`NciCheck.DescDifftest` holds `NciCheck.Desc` to it and to the Python reference, `NciCheck.DescLiveness` to it on
Program shapes past anything `NciCheck.Desc` walks one by one, and `NciCheck.DescFuzz` decides on generated Programs the
statements their agreement rests on, and the agreement itself. Code: it reads the Program with Flock's
decoder (`Flock.Program`) and imports nothing else but Lean core. It stays experimental until its agreement with the
definition is proved.

`NciCheck.Protocol` is the policy's check as a core protocol, `verity/inference-only/v0` (the `inference-only` exe):
the inference-only policy's subprotocol of onsite and remote, on the core interface (`CoreProtocol`).

Flock's decoder doesn't check that a reference reads a leaf at the width it was committed with, which the
Python codec refuses as it decodes (`E_REINTERPRET`), so every Program the check reads goes through
`NciCheck.decodeProgram`, which refuses one that doesn't, with a message.

`NciCheck.Count.check` (`nci-check --count`) is the count of `(c, k)`-inference blocks: that a Boolean declaration's
computed gates divide into at most `n` proof blocks of a `(c, k)`-partition (`Nci.CompliesAtMost`), the blocks the
activations of the block Definitions the declaration names, with each block Definition's Boolean gates from the gate
table and its fresh bits from `NciCheck.Desc`'s summaries. It publishes only the judgment against `n`, `{complies}`;
the whole verdict, with the counts, is `NciCheck.Count.checkDetail`, for the private record. `NciCheck.CountDifftest`
holds both to the Python reference.

`NciCheck.DefinitionDigest` holds a Definition's digests, read off its Program's descriptor with no decode: the circuit
format's own (`definitionDigest`), over the Definition's body, and the closure digest (`closureDigest`), over its body
and every body it reaches, which a check that takes a Definition by digest pins. It reads none of the modules above;
the period check takes its slot entries' Definitions by it.
-/

@[expose] public section
