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
| [`circuits/`](circuits/README.md) | The circuit language: Definitions, Programs, the descriptor codec, queries and partitions. Its format is [`circuits/PROTOCOL.md`](circuits/PROTOCOL.md). Beside it, the circuits written in it: `ops/` (the primitive operations), `silicon/` (the GPUs' exact arithmetic), `kernels/` (fast code, each checked against its Definition) and `private_circuits/` (the universal circuit). |
| `commitments/` | Hiding commitments ([frame-v3](commitments/frame_v3/PROTOCOL.md), [hm96](commitments/hm96/PROTOCOL.md)), shared hash code, hash-based signatures and transport (`crypto/`, with `lms.py`; its Lean in `lms/`), the one challenge derivation (`randomness/`), an attester's signed period roots (`attest/`), what an adapter hands Verity (`periods/`), the security claims' ids (`claim_ids.py`), and the same commitments on a GPU (`cuda/`). |
| `protocols/` | The protocols, each in up to three parts: `verifier/` (the auditor's check), `prover/` (the developer's side) and `device/` (an attester, the auditor's own hardware at the site, or the challenger, the verifier's online half). The layers are below. |
| [`runtime/`](runtime/README.md) | Runs a protocol's parties over channels and keeps each party's log of the run. It knows no protocol. |
| `party.py`, `channel.py`, `protocol.py` | What a party is (a next-message machine), what its messages declare, and what a protocol declares with the placement check. |
| `device.py` | The registered hardware (the Site), the check each registration entry passes, and the readers of the data sheets in [`hardware/`](../hardware/README.md). |
| `config.py` | The deployment configuration a run starts from (`verity run CONFIG`). |

The protocols, by layer, with registration beside them. After the landing, `accounting/` moves to
`resource_accounting/` by resource (`compute/pouw`, `memory/pous`, `memory/poub`, `network/certifier`), `policy/` to
`workload_classification/` and `accounting/registration/` to `protocols/registration/`; an attester's `device/` becomes
`attester/`, and the challenger's code moves under `verifier/`.

| Layer | Directory | Protocols |
| --- | --- | --- |
| Proofs | `protocols/proofs/` | [`flock/`](protocols/proofs/flock/verifier/README.md) (the proof system), [`spot/`](protocols/proofs/spot/verifier/README.md) (sampling), `recursion/`, `spot_flock/`, `spot_flock_recursion/` |
| Registration | `protocols/accounting/registration/` | [The registration log and its check](protocols/accounting/registration/README.md), run once, before any challenge |
| Resource accounting | `protocols/accounting/` | [`pouw/`](protocols/accounting/pouw/verifier/README.md) (compute), [`pous/`](protocols/accounting/pous/verifier/README.md) and [`poub/`](protocols/accounting/poub/verifier/README.md) (memory), [`certifier/`](protocols/accounting/certifier/verifier/README.md) (network) |
| Workload classification | `protocols/policy/` | [`inference_only/`](protocols/policy/inference_only/verifier/README.md) |
| Configurations | `protocols/compose/` | [`remote`](protocols/compose/verifier/remote/README.md) and `onsite`, each a list of protocols on one registration |
| Being tried | `protocols/experimental/` | `dither/`, [`pearl_c4/`](protocols/experimental/pearl_c4/README.md) |
| Shared | [`protocols/interface/`](protocols/interface/README.md) | The Lean interface each protocol's check implements |

## Rules that hold here

`tests/test_boundaries.py` holds the tree to the import order of [`tests/import_order.toml`](../tests/import_order.toml):
a file loads only its own row or an earlier one, no verifier loads prover code, a place of `verity` imports only the
places its `loads` lists, what the verifiers' side loads and reaches (the lists of its trusted places) only shrinks, and
`verity` imports no other distribution. To allow an import between two places, write it and run
`uv run tools/check/import_order.py`, which writes the line. The plain-integer references (`STDLIB_ONLY`:
`verity/circuits`, `verity/commitments/randomness`, `verity/commitments/claim_ids.py` among them) stay free of numpy.

Read next: [`ARCHITECTURE.md`](../ARCHITECTURE.md) section 4 for each part's interface, and each module's docstring
for what it holds.
