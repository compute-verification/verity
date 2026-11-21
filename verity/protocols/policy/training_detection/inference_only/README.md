# The inference-only policy

It establishes that the registered circuit is inference, not training: that it divides into inference blocks of at most
Γ gates, each reading at most α fresh input bits a gate, fresh meaning from outside the block and outside θ, the
registered information. Its check is Lean, and its one Python file is the verifier's manifest (`verifier/parties.py`).

The guarantee: `InferenceOnly.Guarantees.CompliesCK` (`security/Definitions/InferenceOnly/Guarantees/CompliesCK.lean`),
that a division complying at rate is a (c, ⌊αc⌋)-partition; the lock lists it as proved and not deployed, with no honest
witness, and its open premise `InferenceOnly.CompliesRate` is what the check decides, which no guarantee ties to an accept.

Not yet: on `verity run`'s road the inference-only verdict is Python's (`benchmarks/one_stage/inference_only.py`): the
Lean count decides a block's gates and fresh bits on an instance Python builds, Python decides n, and the core protocol
`verity/inference-only/v0` runs only in the composite, which that road doesn't reach.

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| the Lean check | the division at the size of the description (`Desc`), then θ's provenance (`Theta`) | `verifier/lean/InferenceOnlyVerifier/` |
| the core protocol | that check as `verity/inference-only/v0`, which both composites list (`Protocol`, `inference-only`) | `verifier/lean/` |
| the count | (c, k)-inference blocks against n (`Count`, `inference-only-verifier --count`) | `verifier/lean/` |
| the record form | the check on an execution phase's circuit record (`Record`, `inference-only-record-check`) | `verifier/lean/` |
| the Python references | the per-gate check, the unroll and the record form, which seven difference tests hold the Lean to | `verifier/lean/scripts/` |
| the manifest and scheme | the verifier, one party with no channel; the one table's entry | `verifier/parties.py`, `scheme.py` |

## What varies

| form of the check | on the road of `remote.toml` and `onsite_full.toml` |
| --- | --- |
| the core protocol, Lean | exists: the composite runs it, off the road |
| the count, Lean | served: each block's gates and fresh bits, at step `inference-only` |
| the instance, θ and n, Python | served: `benchmarks/one_stage/inference_only.py` builds the count's instance and decides n |
| the adapter's count | served: step `block-counts`, the prover's count (`verity_vllm`'s `block_counts`), no verifier's verdict |
| the record form, Lean | exists: `benchmarks/inference_only` alone |
| the Python references | exists: the difference tests alone |

A new implementation supplies: its circuit's division into inference blocks, θ and θ's provenance.
Fixed for every implementation: the rule (`InferenceOnly.CompliesRate`) and the Lean check.

## Interfaces

- From the workload: the registered circuit as a descriptor Program, its division into inference blocks, θ and its
  provenance in the deployment's registration log.
- From `run`: α and Γ, fixed before the serve, and the registration records before the run's own; no coins.
- It gives: the composite an accept or a refusal under `verity/inference-only/v0`, which publishes the division and each
  block's profile as a stated exception, not zero-knowledge.

## Try it

~~~sh
cd verity/protocols/policy/training_detection/inference_only/verifier/lean && lake exe inference-only id
~~~

It builds the check and prints the core protocol's identity, `verity/inference-only/v0`. It needs `lake` and no GPU.
