# The inference-only policy

The policy holds when the registered circuit's gates divide into at most n (c, k)-inference blocks, each of at most c
gates taking at most k unregistered input bits (inputs from outside the block that are not the registered information
θ); its rate form asks for at most Γ gates a block and α unregistered input bits a gate. Training hidden in a passing
monitoring period's declared inference is slowed by at least the factor of the training slowdown bound.

The guarantee: none is deployed or witnessed. `InferenceOnlyVerifier.Guarantees.ServedComplies` (`security/Properties/`
`InferenceOnly/Record.lean`): where `Record` accepts, every circuit its record lays out complies at Γ and α, on the open
premises `Record.Owns`, `Record.LaidOut`, `Record.Counts`, `Record.Rates`. `Properties.Remote.AcceptInferenceOnly`
(`security/Properties/Remote.lean`): remote's accept gives the policy, on the open premises `Divides`, `Expands`,
`LaysOut`, `Names`, `Parses`, `Properties.Spot.ExpectsV2`. `InferenceOnly.Guarantees.CompliesCK` (`security/`
`Definitions/InferenceOnly/Guarantees/`): the rate form gives a (c, ⌊αc⌋)-partition, on the open premise `CompliesRate`.

Not yet: on `served_verdicts.sh`'s road, `benchmarks/one_stage/inference_only.py` builds the input, chooses θ and
judges n in Python; Lean decides one window's blocks against c and k and holds no input of θ to a pinned root. The
protocol check runs only in the composite's case, at α and Γ the run sets or reads off its own division. The developer
writes every limit and no device signs one. The serving side's `judge` (`adapters/vllm/verity_vllm/records/`) is
Python. The protocol publishes the division and is not zero-knowledge. `Record` refuses a root that reads an input node
in part of its leaf width as `program`, where `Desc` and `Count` give `recut-unread`.

## Pieces
Each piece is a Lean file in `verifier/lean/InferenceOnlyVerifier/`, run by the auditor's verifier:
- `Count`: the count rule on a word-level Program; `inference_only.py` runs it as `inference-only-verifier --count`.
- `Protocol`: `Theta` as the core protocol `verity/inference-only/v0`; the composites run it as `inference-only`.
- `Theta`: `Desc`, then each input of θ's registration in the deployment's registration log.
- `Desc`: the rate rule at the description's size, on a Program holding every weight and wire it reads.
- `Record`: the rate rule on an execution phase's circuit record; `benchmarks/inference_only/report.py --from` runs it.
- `Check`: the unrolled reference, gate by gate, that the difference tests hold `Desc` to (`--unrolled`).

## What varies
A workload supplies its registered circuit, its division and θ's registrations; the rules and the check are fixed.
## Interfaces
It takes the Program, division and θ from the workload, the limits and log from `run`; it gives accept or refuse.
## Try it
~~~sh
(cd verity/protocols/policy/training_detection/inference_only/verifier/lean && lake exe inference-only id)
python3 tools/verity/lean/check.py --build verity/protocols/policy/training_detection/inference_only/verifier/lean
uv run tools/check/suites.py verity/protocols/accounting/network/verifier/network benchmarks/inference_only
~~~
