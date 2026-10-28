[quick overview written by Claude...]

# Verity

Verity lets an AI developer prove to an auditor what its datacenter's hardware computed, so that the auditor can bound
how much the developer could have trained in secret. It is for developers who must show what their hardware ran without
revealing their models or data, since every proof is zero-knowledge, and for the auditors who check those proofs. It has
two composite protocols: onsite, for an auditor with devices at the datacenter, and remote, for an auditor with none.
Start with the remote protocol in `verity/core/protocols/remote/`, then [ARCHITECTURE.md](ARCHITECTURE.md).

- [ARCHITECTURE.md](ARCHITECTURE.md): what Verity bounds and how, the design's layers and the Glossary.
- [GLOSSARY.md](GLOSSARY.md): the notation every guarantee is stated in, and the older names it replaces.
- `Security/`: the guarantees, in Lean. `Security/Definitions/` states them and their named assumptions,
  `Security/Proofs/` proves them, and a guarantee whose proof isn't there is open.
- [AGENTS.md](AGENTS.md): the code's layout and rules, for contributors and their agents.

## Subprotocols

Each subprotocol proves one kind of claim about what the developer's hardware did. Its Lean is its definition; where a Python
verifier still decides, a difftest holds it to the Lean until the Lean replaces it.

| Subprotocol | What an accepted run establishes | Where |
| --- | --- | --- |
| Sampled proofs | The auditor draws proof blocks of a committed circuit with live coins, and C-Flock proves each drawn block's values in the committed trace right. The result bounds how many proof blocks are faulty. | [`verity/core/protocols/sampled_proofs`](verity/core/protocols/sampled_proofs/README.md), C-Flock in [`verity/ml/flock`](verity/ml/flock/README.md) |
| Proof of useful work (PoUW) | A credited size S of noised matrix products, at least (1 − ε)·S of their AND gates computed right. Dither is its no-fold design, stated over the ACCEL model of the accelerator. | [`verity/core/protocols/pouw`](verity/core/protocols/pouw/README.md), [Dither](verity/core/protocols/pouw/dither/PROTOCOL.md) |
| Proof of useful space (PoUS) | A lower bound on the bits a device's memory held, from timed rounds the memory challenger runs and signs. | [`verity/core/protocols/pous`](verity/core/protocols/pous/README.md) |
| Proof of useful bandwidth (PoUB) | A lower bound on the bits read from a node's registered store. | [`verity/core/protocols/remote_bandwidth`](verity/core/protocols/remote_bandwidth/README.md) |
| Network certification | An auditor-trusted device on each node's uplink, the network certifier, certifies that the traffic carried only values in the committed trace, and bounds the bits that left the site. | [`verity/core/protocols/certifier`](verity/core/protocols/certifier/README.md) |
| Memory wipes | A registered server's life is cut into server blocks, each opened by a certified memory wipe, so state crosses a boundary only as a committed value. | `verity/core/devices/memory_challenger/lean` |
| The inference-only policy | Inference rather than training: the declared circuit divides into inference blocks of c gates, each reading at most k bits of fresh input. A protocol with no rounds, which the verifier checks from the declared circuit and the registered information alone. | [`verity/core/protocols/network/nci`](verity/core/protocols/network/nci/README.md) |

## The two composite protocols

The subprotocols run composed, as one protocol with one verdict
([`verity/core/protocols/composition`](verity/core/protocols/composition/README.md)): a run is accepted only if
every subprotocol accepts its own case, the commitment binds all of theirs, and the subprotocols whose commitment is
the run's record (the inference-only policy and sampled proofs at both, PoUW too at remote) commit to one record.

- **Onsite** (`verity/core/protocols/onsite`), for an auditor with devices at the datacenter: sampled proofs, PoUW,
  the inference-only policy, PoUS, PoUB, network certification and the memory wipe.
- **Remote** ([`verity/core/protocols/remote/README.md`](verity/core/protocols/remote/README.md)), for an auditor with
  no devices there: sampled proofs, PoUW and the inference-only policy. Start there for its lifecycle, guarantee,
  assumptions and demo.

## The vLLM integration

[`integrations/vllm`](integrations/vllm/README.md) attaches Verity to vLLM. It builds each served workload's circuit
from vLLM's own code and the model's checkpoint, checks it against what vLLM served, and commits the values while
serving through an OpenAI-compatible endpoint. Verity itself knows nothing of models: the integration hands it a
circuit, its values and their commitments ([ARCHITECTURE.md](ARCHITECTURE.md), "Integrations").

A workload of record ([`manifests/records.json`](integrations/vllm/manifests/records.json)) pins the circuit that vLLM's
code gives for one model, precision, parallelism and batch shape, with every rank's circuit digests. Its record, which
`verity-vllm records show WORKLOAD` prints, holds three results, and `verity-vllm records publish` refuses a record in
which any of them failed:

- **Build**: the circuit is built from vLLM's code, and every value it names at a call boundary is claimed by a
  Definition, with none uncovered. This is the integration's coverage, counted in values at call boundaries, not in
  gates.
- **Match**: on every rank, the circuit reproduces every value vLLM served at those boundaries, bit for bit, and the
  ranks agree.
- **Commit**: every value the circuit requires is committed, and the sampled openings verify (under SHA-512 roots for
  GLM-5.3 and DeepSeek-V4-Pro; the Llama-3.2-1B record predates SHA-512 and uses SHA-256).

| Workload | Hardware | Build, Match and Commit | Circuit pinned |
| --- | --- | --- | --- |
| GLM-5.3 FP8, batch 1, 256 tokens in and 32 out, greedy | 8 × H200, tensor-parallel | hold | yes |
| DeepSeek-V4-Pro FP8, the same shape | 8 × H200, tensor-parallel | hold | yes |
| Llama-3.2-1B BF16, the same shape, batch-invariant | H200 | hold | yes |
| GLM-5.3 FP8, batches of 8 and 32 with mixed arrivals, one with sampled decoding | 8 × H200, tensor-parallel | not yet published | yes |
| Qwen3-14B BF16 | 2 × RTX PRO 6000, tensor-parallel | not yet published | yes |
| Qwen3-0.6B BF16 | RTX PRO 6000 | not yet published | yes |

Rebuilding from vLLM's code gives every pinned circuit digest on every rank (`verity-vllm neutrality --tier fast|heavy`,
on a CPU machine). For GLM-5.3 and DeepSeek-V4-Pro, the circuits predicted ahead of time, with no tracing, equal the
traced Build on every rank (`verity-vllm predict compare`). `verity-vllm row chain WORKLOAD` reruns a workload's Build,
Match and Commit from scratch on its GPUs.

Llama-3.2-1B in bf16 on one RTX PRO 6000 has gone through sampled proofs end to end in a run of record: vLLM serves it,
its values are committed under SHA-512, live coins sample proof blocks, C-Flock `--zk` proves them, and the Lean verifier
accepts them and refuses every tampered case. Each sampled proof block is part of one serving step's circuit, not yet of
the whole served run's. The vLLM demo ([`examples/vllm`](examples/vllm/README.md)) serves Llama-3.2-1B INT8 on one H200,
the default of `benchmarks/one_stage/served_verdicts.sh`, through sampled proofs and PoUW, and has no passing run of
record yet.

Not yet:

- GLM-5.3 and DeepSeek-V4-Pro through sampled proofs and the Lean verifier.
- Coverage counted in AND gates, as a share of the served computation.
- The whole served run's circuit committed and proved, and the weights tied to a checkpoint in zero knowledge.
- PoUW, PoUS, PoUB and network certification composed on one served engine.

## Where to start

- **The sampled proof system**: `verity/core/protocols/sampled_proofs`, its guarantees in
  `Security/Definitions/Flock/Guarantees/`.
- **Hiding the circuit, by recursion**: an outer C-Flock proof checks the inner proofs, so the auditor sees neither the
  values nor the circuit. Circuit privacy is `Flock.SecurityProofs.RecursiveCircuitPrivate`, and its bound carries its
  terms, among them the commitment key's hiding gap, which isn't proved small for the pinned key. Recursion's knowledge
  soundness is not yet proved.
- **PoUW under the ACCEL model**: Dither's [protocol](verity/core/protocols/pouw/dither/PROTOCOL.md) and the Lean's
  `Pouw.Dither`.
- **Inference versus training**: the [inference-only policy](verity/core/protocols/network/nci/README.md), the
  feasible set, the admissible set and the overhead lower bound in [GLOSSARY.md](GLOSSARY.md), and its numbers in
  [`benchmarks/rho`](benchmarks/rho/README.md).

## Demos

~~~sh
uv sync --all-packages --extra torch-cpu    # a CPU machine; a GPU machine takes --extra torch
verity --help
~~~

You also need Rust (C-Flock's prover), Lean through elan (the C-Flock verifier and the protocol checks), and CUDA on a
GPU machine.

| Demo | What it serves and audits | Command | Details |
| --- | --- | --- | --- |
| The H200 demo | A fixed multi-query attention shape on one H200, every proof block's values committed. It runs sampled proofs, remote's sampled-proofs half: live coins draw proof blocks, which C-Flock `--zk` proves and the Lean verifier checks. Its other pieces are only a skeleton on fixture coins, and the demo claims none of them: PoUS, network certification and memory wipes (onsite's subprotocols), and the bandwidth audit over PoUS's store (a subprotocol of neither composite). | `verity example h200` | `examples/h200/README.md`, `examples/h200/interfaces/README.md` |
| vLLM | Llama-3.2-1B INT8, served by vLLM on one H200 through the committing endpoint. It runs the remote protocol's two subprotocols: live coins sample proof blocks, which C-Flock `--zk` proves and the Lean verifier checks, and PoUW runs on the served model as a development run that credits nothing. Onsite's subprotocols run on the same served run where each has its piece. | `verity example vllm` | `examples/vllm/README.md` |
| Lifecycle | Both composites, onsite and remote, stepped through one monitoring period on CPU in about 2 s: a toy two-layer int8 model, every party simulated in one process, a labelled stand-in for C-Flock's prover, and the verdict by the composition's Python mirror, with every deviation it lists refused. The only example that steps through the whole remote composite. It runs; nothing about it is proved. | `verity example lifecycle` | `examples/lifecycle/README.md` |
| Full utilization | An 8B model's linear layers in s8 at M = 16,384 on one H200, with PoUW crediting the work through Dither's no-fold receipt. No runner yet. | none yet | `examples/dither_full_util/README.md` |

The remote protocol's verifier, `remote check`, has no run of record yet; its first is due Wed 28 Oct, 11 AM PDT.
`verity example lifecycle` is the only example that steps through the whole remote composite, with a stand-in prover.
The remote protocol's demo is the vLLM demo (`verity example vllm`): sampled proofs and PoUW on one served run, with
PoUW's verdict a development run that credits nothing, while `verity example h200` runs remote's sampled-proofs half
([the remote protocol's demo](verity/core/protocols/remote/README.md#the-demo)).

The prover's side serves and proves. The auditor's side runs the verifier, the challenger that draws the live coins,
the network certifier and the memory challenger, and those must run on machines the prover doesn't control.
