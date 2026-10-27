# Verity

Verity verifies how AI datacenters are used. It represents a datacenter's computations exactly as Boolean circuits,
proves claims about them, and derives compute, memory and network accounting from those proofs. The intended
application is enforcing an international agreement to
[deliberately pace AI development](https://www.pacingthefrontier.com/), so it aims for security against a
developer who controls the hardware and its vendor.

This repository, its code, its Lean and its documents, is written by AI agents.

- [ARCHITECTURE.md](ARCHITECTURE.md): the design, its layers and the Glossary.
- [GLOSSARY.md](GLOSSARY.md): the notation every guarantee is stated in, and the older names it replaces.
- `Security/`: the guarantees, in Lean. `Security/Definitions/` states them and their named assumptions,
  `Security/Proofs/` proves them, and a guarantee whose proof isn't there is open.
- [AGENTS.md](AGENTS.md): the code's layout and rules, for contributors and their agents.

## Subprotocols

Each subprotocol proves one kind of claim about what the developer's hardware did. Its Lean is its definition; where a Python
verifier still decides, a difftest holds it to the Lean until the Lean replaces it.

| Subprotocol | What an accepted run establishes | Where |
| --- | --- | --- |
| Sampled proofs | The auditor draws proof blocks of a committed circuit with live coins, and C-Flock proves each drawn block's committed values right. The result bounds how many proof blocks are faulty. | [`verity/core/protocols/sampled_proofs`](verity/core/protocols/sampled_proofs/README.md), C-Flock in [`verity/ml/flock`](verity/ml/flock/README.md) |
| Proofs of useful work (PoUW) | A credited size S of noised matrix products, at least (1 − ε)·S of their AND gates computed right. Dither is its no-fold design, stated over the ACCEL model of the accelerator. | [`verity/core/protocols/pouw`](verity/core/protocols/pouw/PROTOCOL.md), [Dither](verity/core/protocols/pouw/dither/PROTOCOL.md) |
| Proofs of useful space (PoUS) | A lower bound on the bits a device's memory held, from timed rounds the memory challenger runs and signs. | [`verity/core/protocols/pous`](verity/core/protocols/pous/README.md) |
| The bandwidth audit | A lower bound on the bits read from PoUS's store. | [`verity/core/protocols/bandwidth`](verity/core/protocols/bandwidth/PROTOCOL.md) |
| The remote bandwidth audit | The same, over a node's registered store. | [`verity/core/protocols/remote_bandwidth`](verity/core/protocols/remote_bandwidth/README.md) |
| The network certifier | An auditor-trusted device on each node's uplink certifies that the traffic carried only committed values, and bounds the bits that left the site. | [`verity/core/protocols/certifier`](verity/core/protocols/certifier/README.md) |
| The wipe | A server's life is cut into server-time blocks, each opened by a certified wipe of its memory, so state crosses a boundary only as a committed value. | `verity/core/devices/memory_challenger/lean` |
| Novel communication intensity (NCI) | Inference rather than training: the committed circuit divides into blocks of bounded size, each reading a bounded amount of novel information. | `verity/core/protocols/network/nci/lean` |

## The two composite protocols

The subprotocols run composed, as one protocol with one verdict
([`verity/core/protocols/composition`](verity/core/protocols/composition/README.md)): a run is accepted only if
every component accepts its own case, and the commitment binds all of theirs.

- **Onsite** (`verity/core/protocols/onsite`), for an auditor with devices at the datacenter: sampled proofs, PoUW,
  PoUS, the remote bandwidth audit and the network certifier.
- **Remote** (`verity/core/protocols/remote`), for an auditor with no devices there: sampled proofs and PoUW.

The wipe, the bandwidth audit over PoUS's store and NCI are not components of either yet.

## Where to start

- **The sampled proof system**: `verity/core/protocols/sampled_proofs`, its guarantees in
  `Security/Definitions/Flock/Guarantees/`.
- **Hiding the circuit, by recursion**: an outer C-Flock proof checks the inner proofs, so the auditor sees neither the
  values nor the circuit. Circuit privacy is `Flock.SecurityProofs.RecursiveCircuitPrivate`, and its bound carries its
  terms, among them the commitment key's hiding gap, which isn't proved small for the pinned key. Recursion's knowledge
  soundness is not yet proved.
- **PoUW under the ACCEL model**: Dither's [protocol](verity/core/protocols/pouw/dither/PROTOCOL.md) and the Lean's
  `Pouw.Dither`.
- **Inference versus training**: NCI, and the physical model, envelope and overhead in
  [GLOSSARY.md](GLOSSARY.md).

## Demos

~~~sh
uv sync --all-packages --extra torch-cpu    # a CPU machine; a GPU machine takes --extra torch
verity --help
~~~

You also need Rust (C-Flock's prover), Lean through elan (the C-Flock verifier and the protocol checks), and CUDA on a
GPU machine.

| Demo | What it serves and audits | Command | Details |
| --- | --- | --- | --- |
| The toy | A two-layer W8A8 MLP on CPU. Each run is audited by every subprotocol, and the verdicts are composed. | `verity bench all-protocols` | `benchmarks/all_protocols/README.md` |
| The H200 demo | A fixed multi-query attention shape on one H200. Every proof block's values are committed, and the P2-encoded weights fill the card's memory. PoUS, the bandwidth audit, the network certifier and the wipe run. Live coins draw proof blocks, which C-Flock `--zk` proves and the Lean verifier checks. | `verity bench e2e_h200` | `benchmarks/e2e_h200/README.md` |
| vLLM | Llama-3.2-1B served by vLLM through the committing endpoint. A served run is audited by every subprotocol. | `verity example vllm` | `examples/vllm/README.md` |

The prover's side serves and proves. The auditor's side runs the verifier, the challenger that draws the live coins,
the network certifier and the memory challenger, and those must run on machines the prover doesn't control.
