# The inference-only policy

The inference-only policy proves the registered circuit a (c, k, n)-inference circuit (`InferenceOnly.CompliesAtMost`):
its gates divide into at most n blocks of at most c gates, each reading at most k wires from outside θ, the registered
information. A circuit that is that and also trains has slowdown ρ ≥ cν/μ (`security/Definitions/Slowdown/README.md`).

This directory is the policy's check, in Lean only (`lean/`, the Lake package `inference_only_check`). Its one Python
file is its manifest, `parties.py`: one party, the verifier, with no channel. Both composites, onsite and remote,
compose the check as the core protocol `verity/inference-only/v0`.

Two things do not run yet. The check doesn't run inside a zero-knowledge proof, so its evidence reaches the verifier in
the clear. And remote doesn't run the check yet.

## Try it

~~~sh
cd verity/protocols/workload_classification/training_detection/inference_only/verifier/lean
lake exe inference-only id
~~~

This builds the check and prints the core protocol's identity, `verity/inference-only/v0`. It needs `lake` and no GPU.

## Commands

Each is a `lake exe` in `lean/`.

| Command | What it does |
| --- | --- |
| `inference-only id \| check \| serve` | the core protocol `verity/inference-only/v0`, which both composites compose |
| `inference-only-rows id \| check \| serve` | the core protocol `verity/inference-only/rows/v1`, which no system lists |
| `inference-only-check INSTANCE TRUSTED` | the verdict on one instance, as JSON; `--count` the count of (c, k)-inference blocks, `--policy` the policy alone |
| `inference-only-record-check CASE` | the record form on one case written from a served run's files, for the private record only |

The benchmark [`benchmarks/inference_only`](../../../../../../benchmarks/inference_only/README.md) runs the count on the
class this tree builds.

## Names

A few older names stay here until a change of their own renames each (`GLOSSARY.md`, Older names): `benchmarks/nci`;
a verdict's failure rule `"nci"`, its key `ius` for its inference blocks with the names that compute it, and the
refusal codes `input-in-iu`, `gate-in-no-iu` and `empty-iu`; the toy Definitions the vector generators build
(`NciHandBlk`, `NciDecl…`); `baseline_difftest/`, the θ difftest's directory, whose bytes the lock pins; "baseline"
for the registered information, also the θ difftest's name in its printed messages; "IU" and "inference block" for
an inference block, in the lock's reasons; "fresh input bits" for fresh input
bits, in the lock's reasons and the liveness difftest's message; and `novel` for fresh input: the variable of the
check's rate rule and of its Python reference, and a mode of the liveness difftest.
