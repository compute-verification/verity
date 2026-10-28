import NciCheck.Check
import NciCheck.Difftest
import NciCheck.Desc
import NciCheck.DescDifftest
import NciCheck.DescLiveness
import NciCheck.DescFuzz
import NciCheck.Baseline
import NciCheck.BaselineDifftest

/-!
# The NCI policy's check

`NciCheck.Baseline.check` (the `nci-check` exe) is the verifier's check that a declared division of a declaration into
isolation units meets `Nci.Complies Γ α` (`Security/Definitions/Nci.lean`), for IUs aligned to the root's Calls, at the
cost of the descriptor: `NciCheck.Desc.check`, one summary per Definition, never a walk over the unrolled circuit
(AGENTS.md "Design rules"), then each θ entry's provenance, registered public data that the first record of the
deployment's registration log registers and never an earlier run's output, against the service's log
(`NciCheck.Baseline`, CMP-66; `NciCheck.BaselineDifftest` holds it to the Python reference).
`NciCheck.check` is the same check gate by gate, with the input format and its mapping to the definition's circuit in
`NciCheck.Check`'s docstring, kept as the reference: `NciCheck.Difftest` holds it to the Python reference, and
`NciCheck.DescDifftest` holds `NciCheck.Desc` to it and to the Python reference, `NciCheck.DescLiveness` to it on
Program shapes past anything `NciCheck.Desc` walks one by one, and `NciCheck.DescFuzz` decides on generated Programs the
statements their agreement rests on, and the agreement itself. Code: it reads the Program with C-Flock's
decoder (`Flock.Program`) and imports nothing else but Lean core. It stays experimental until its agreement with the
definition is proved.

C-Flock's decoder refuses a Program with a reference that reads a leaf at another width than it was committed with
(`E_REINTERPRET`, `Flock.Program.Widths`) as the Python codec does, with the codec's message, which a `program` refusal
carries (`NciCheck.refusalOf`).
-/
