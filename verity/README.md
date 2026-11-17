# `verity`

The distribution `verity`: the circuit language, commitments, the protocols, and the runtime that runs their parties.
Import from the module that owns an object (`from verity.circuits import Program`); `verity` itself re-exports nothing.

~~~sh
uv run tools/check/suites.py verity/circuits     # one package's tests; each directory with a pyproject.toml is a suite
~~~

Build a circuit, evaluate it, name it and query it:

~~~python
from verity.circuits import Program, query_ast
from verity.circuits.codec import program_digest
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.ops.gemm import GEMM_AMPERE, Gemm

gemm = bind(Gemm, K=64, N=8, **GEMM_AMPERE)       # x[64] @ W[8, 64]^T in BF16, as an A100's tensor cores compute it
x, W = [0x3F80] * 64, [0x3F80] * (8 * 64)         # BF16 encodings of 1.0, W flattened row by row
evaluate(gemm, x, W)                              # eight BF16 words 0x4280, that is 64.0
program = Program(gemm)
program_digest(program)                           # the circuit's name: its SHA-512 digest
query_ast.evaluate(program, "//GemmCoordinate")   # its 8 output coordinates, each a set of gates (`intervals()`)
~~~

## What is here

| Path | What it holds |
| --- | --- |
| [`circuits/`](circuits/README.md) | The circuit language: Definitions, Programs, the descriptor codec, queries and partitions. Its format is [`circuits/PROTOCOL.md`](circuits/PROTOCOL.md). Beside it, the circuits written in it: `ops/` (the primitive operations), `silicon/` (the GPUs' exact arithmetic), `kernels/` (fast code, each checked against its Definition) and `private_circuits/` (the universal circuit); the evaluator (`evaluate/`) and a circuit's requirement in Lean (`requirement/`). The served workloads' circuits are the vLLM adapter's (`adapters/vllm/verity_vllm/definitions/`). |
| `commitments/` | Hiding commitments ([frame-v3](commitments/frame_v3/PROTOCOL.md), [hm96](commitments/hm96/PROTOCOL.md)), shared hash code, hash-based signatures and transport (`crypto/`, with `lms.py`; its Lean in `lms/`), the one challenge derivation (`randomness/`), an attester's signed period roots (`attest/`), what an adapter hands Verity (`periods/`), the security claims' ids (`claim_ids.py`), content identities, the tagged SHA-512 of a canonical manifest (`identity.py`, its Lean in `identity/`), and the same commitments on a GPU (`cuda/`). |
| `protocols/` | The protocols, each in up to three parts: `verifier/` (the auditor's check, and the challenger, the verifier's online half), `prover/` (the developer's side) and `attester/` (an attester, the auditor's own hardware at the site). The layers are below. |
| `circuits/base/`, `circuits/lean/`, `commitments/lean/` | The three base Lean packages: bytes, hashes and parsing; the circuit format; frame-v3 and the sampling of proof blocks. Flock's verifier and every package that requires it load them, and they require nothing above them. |
| [`runtime/`](runtime/README.md) | Runs a protocol's parties over channels and keeps each party's log of the run. It knows no protocol. |
| `party.py`, `channel.py`, `protocol.py`, `errors.py` | What a party is (a next-message machine), what its messages declare, what a protocol declares with the placement check, and the shared exception classes. |
| `device.py` | The registered hardware (the Site), the check each registration entry passes, and the readers of the data sheets in [`calibration/hardware/`](../calibration/hardware/README.md). |
| `config.py` | The configuration a run starts from (`verity run CONFIG`). |

The protocols, by layer, with registration beside them:

| Layer | Directory | Protocols |
| --- | --- | --- |
| Proofs | `protocols/proofs/` | [`flock/`](protocols/proofs/flock/verifier/README.md) (the proof system: its Lean verifier and the ZK server in `verifier/`, its prover in `prover/`, the agreement with upstream's verifier in `prover/agreement/`), [`spot/`](protocols/proofs/spot/verifier/README.md) (sampling, with the integrity profile and the challenger in `verifier/challenger/`), `recursion/` (the recursion layer), `spot_flock/` (SPOT[Flock]), `spot_flock_recursion/` (SPOT[Flock] with the recursion) |
| Registration | `protocols/registration/` | [The registration log and its check](protocols/registration/README.md), run once, before any challenge |
| Resource accounting | `protocols/resource_accounting/` | [`compute/pouw/`](protocols/resource_accounting/compute/pouw/verifier/README.md), [`memory/pous/`](protocols/resource_accounting/memory/pous/verifier/README.md) and [`memory/poub/`](protocols/resource_accounting/memory/poub/verifier/README.md), [`network/`](protocols/resource_accounting/network/verifier/README.md) (the network attester) |
| Workload classification | `protocols/workload_classification/` | [`training_detection/inference_only/`](protocols/workload_classification/training_detection/inference_only/verifier/README.md), whose check is Lean only (`verifier/lean/`) |
| Compose | `protocols/compose/` | [`remote`](protocols/compose/verifier/remote/README.md) and `onsite`, each a list of protocols on one registration, with their Lean in `verifier/lean/`; `verifier/live_challenger/` is the auditor host's challenger service |
| Being tried | `protocols/experimental/` | `dither/`, [`pearl_c4/`](protocols/experimental/pearl_c4/README.md) |
| Shared | [`protocols/interface/`](protocols/interface/README.md) | The Lean interface each protocol's check implements |

Nine directories here are distributions of their own, each with its own import root (`tests/import_order.toml`'s
`[[distribution]]`), all under `protocols/`: `proofs/flock/`, `compose/verifier/live_challenger/`, and under
`resource_accounting/`, `memory/pous/verifier/check/`, `memory/pous/prover/p2/`, `memory/poub/prover/`,
`memory/poub/attester/live_verifier/`, `compute/pouw/prover/clear/`, `compute/pouw/prover/lowering/` and
`compute/pouw/verifier/receipts/`.

## Read next

- The root `AGENTS.md`: which folder may import which (`tests/import_order.toml`), and how to allow an import.
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) section 4: each part's interface. Each module's docstring says what it holds.
