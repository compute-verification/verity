# `verity`

This is Verity itself: the code a developer and an auditor run. It has no notion of a model, a request or a serving
stack. Its vocabulary is circuits, values, commitments, proofs and verdicts, and an
[adapter](../adapters/vllm/README.md) translates a real system into it.

It has four parts:

1. **[Circuits](circuits/README.md).** A computation is written as a Boolean circuit: an exact description of what was
   computed, bit for bit, down to how a GPU's tensor cores round. A circuit is named by its hash.
2. **Commitments.** The developer commits to the values a circuit computed, without revealing them.
3. **Protocols.** Each protocol turns proofs about committed values into one thing the auditor learns: that the
   computation was done right, how much compute, memory or network the hardware used, or that the workload was
   inference.
4. **[Runtime](runtime/README.md).** Runs a protocol's parties on real machines and carries their messages. It knows no
   protocol.

## A first circuit

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
query_ast.evaluate(program, "//GemmCoordinate")   # its 8 output coordinates, each a set of gates
~~~

Import from the module that owns an object. `verity` itself re-exports nothing.

## Protocols

| Layer | Protocol | What the auditor learns |
| --- | --- | --- |
| Proofs | [Flock](protocols/proofs/flock/verifier/README.md) | One statement about one circuit holds of committed values, proved in zero knowledge. Flock is the proof system. |
| | [SPOT](protocols/proofs/spot/verifier/README.md) | At most a stated share of a computation's pieces is wrong, from proofs of a random sample of them. |
| | Recursion | One outer proof says the verifier accepted the developer's proofs. |
| Resource accounting | [Registration](protocols/registration/README.md) | What the protocols below account for: the hardware, the circuit compiler, and commitments to the rest, such as the weights. It runs once, before any challenge. |
| | [PoUW](protocols/resource_accounting/compute/pouw/verifier/README.md) | Compute: how many of the served matrix multiplies the GPU really computed, against its data sheet's capacity. |
| | [PoUS](protocols/resource_accounting/memory/pous/verifier/README.md) | Memory: how much of the registered store each device held. The memory wipe is part of it. |
| | [PoUB](protocols/resource_accounting/memory/poub/verifier/README.md) | Memory: how much of a node's memory read rate is left uncertified. |
| | [Network attester](protocols/resource_accounting/network/verifier/README.md) | Network: what left on a link is what the registered circuit produces from what entered. |
| Workload classification | [Inference-only](protocols/workload_classification/training_detection/inference_only/verifier/README.md) | The workload is inference: its circuit divides into blocks that each read little fresh input. |
| Compose | [Remote](protocols/compose/verifier/remote/README.md) and onsite | One verdict from a list of the protocols above. This is what an auditor runs. |

A protocol has up to three parts: its verifier, its prover, and an attester where the auditor has hardware at the
site. Two more protocols are being tried: Dither and [Pearl-C4](protocols/experimental/pearl_c4/README.md).

## Read next

- [`ARCHITECTURE.md`](../ARCHITECTURE.md): the three roles, and each part's interface.
- [The Lean interface](protocols/interface/README.md) that each protocol's verifier implements.
- [`demos/README.md`](../demos/README.md): the protocols run together, end to end.
