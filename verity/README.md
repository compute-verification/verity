# Verity core (`verity`)

One distribution, one module per concern. Import from the module that owns an object (`from verity.primitives.circuits import Program`);
`verity` itself re-exports nothing. The architecture and the dependency direction between these modules are the root
`README.md`; the repository map with each module's contents is the root `AGENTS.md`.

| module | what it owns |
|---|---|
| `verity.primitives.circuits` | types, Definitions, Programs, the canonical layout, the reference evaluator, the descriptor codec, the partition references (`cut`, `partition`, `partition_object`, `units`), constants' binding times (`constants`) and annotations; the program format is specified in `primitives/circuits/PROTOCOL.md` |
| `verity.primitives.circuits.evaluation` | the one evaluator: the plain-integer reference and `evaluate_batch` over numpy arrays, with registered kernels |
| `verity_catalog.definitions` | exact hardware semantics of the tensor-core accumulator step (`verity_catalog.silicon`) and their IR registrations (`prims`, `gemm`), the other IR primitives (`scalar`: integer, bit, select and comparison operations; `fp32`: FP32/BF16 arithmetic), with numpy kernels, and the library of hardware and protocol objects core owns (`library`) |
| `verity.primitives.randomness` | `derive(source, domain, context)`, the one challenge derivation, and exact samplers |
| `verity_sp1.proofs` | backend-independent single statements, how a backend is judged on them, and `IntegrityProfile` |
| `verity.primitives.commitments` | application-level commitment schemes (`frame_v3`, `vllm_v1`, `hm96`, Merkle trees and openings) |
| `verity.claims` | the ids of security claims the tables cite |

`tests/test_boundaries.py` checks the direction of imports between core modules and a list of packages core may not
import, and keeps the plain-integer references (`verity.primitives.circuits`, `verity_catalog.silicon`, `verity.primitives.randomness`, `verity.claims`) free
of numpy.

~~~sh
uv run pytest verity/tests            # in `check`
~~~

The byte formats are pinned by vector files beside the tests: `tests/ir/format_vectors.json` (the program format),
`tests/ir/partition_vectors.json` and `template_instance_vectors.json` (partitions), `tests/randomness/vectors.json`
(the challenge derivation).
