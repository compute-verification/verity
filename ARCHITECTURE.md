# Verity

Verity is a system for verifying how AI datacenters are being used. It aims for security in extremely adversarial settings: the intended application is enforcing an international agreement to [deliberately pace AI development](https://www.pacingthefrontier.com/).

It is a streaming interactive protocol between a *developer* and an *auditor*. The developer keeps private records of the computations it performs in its datacenters and continuously proves properties of them. What Verity bounds is how much the developer could train in secret on hardware the auditors know about, in three steps ([GLOSSARY.md](GLOSSARY.md) has each term):

1. **Each protocol restricts what an accepted run can have computed.** An accepted protocol p gives its admissible set 𝒜_p, except with probability δ_p, and their intersection with the feasible set 𝓕, everything the hardware can compute, is the admissible set 𝒜.
2. **What the protocols establish.** [Sampled proofs](verity/core/protocols/sampled_proofs/README.md) and [proof of useful work](verity/core/protocols/pouw/README.md) (PoUW) prove that part of the actual circuit, what the hardware computed, matches the declared circuit, what the builder compiled from the developer's declared advice: that part is verified compute, up to the share the proofs could miss (ε_skip: faulty proof blocks and PoUW's misses). The [inference-only policy](verity/core/protocols/network/nci/README.md) requires the declared circuit to divide into inference blocks of c gates, each reading at most k bits of fresh input, anything but the registered information (the models, code and data committed up front); it is a policy with no rounds, which the verifier checks in validation from the declared circuit and the registered information alone. The memory wipe, [proof of useful space](verity/core/protocols/pous/README.md) (PoUS) and [network certification](verity/core/protocols/certifier/README.md) bound each server block's fresh-input budget, what a registered server takes in between two memory wipes. Hardware outside the auditors' inventory is unregistered hardware, which no protocol touches.
3. **The slowdown lower bound.** A bounding strategy, a necessary subcomputation that every successful training run must compute and a pricing argument for it, shows that training in the admissible set costs at least T_min/T_ref times the fleet-time it would cost with no protocol, fleet-time being elapsed time times the whole fleet, registered and unregistered. Verified compute contributes to training only through inference-block yields, free compute contributes at the hardware's rate, and each server block contributes at most its server-block yield. The protocols cap the compute rate available to training at R_free ≈ (U + ν)Φ, so a tiny run isn't slowed and a frontier run is slowed by about 1/(α(U + ν)): the result is a curve over run size.

The policy exploits an asymmetry: an inference episode (a chat session) reads little fresh input (say 100 KB of prompt tokens), a training episode (an RL rollout) an enormous amount (say 1 TB of new parameters), so inference fits blocks of a small fresh-input budget and training through them yields little.

The developer represents its computations as Boolean circuits that capture their numerical semantics down to the silicon (the logical circuit need not match the physical one's structure). Verity provides a [Python API](verity/README.md) for building and querying massive Boolean circuits, and a library of primitive circuits for NVIDIA accelerators' low-level semantics, tested with the [Hawkeye](https://arxiv.org/abs/2603.20421) methodology.

The subprotocols run composed in two composite protocols, onsite, for an auditor with devices at the datacenter, and remote, for one with none: the [composition README](verity/core/protocols/composition/README.md#subprotocols) is the one list of the subprotocols, what an accepted run of each establishes and the composites that include it. Proof of useful bandwidth (PoUB) is the one bandwidth subprotocol, onsite only for now.

## How an audit works

An audit runs registration (the site, the registered information θ and the builder B), then a loop of execution and verification, each verification in four steps (commitment, challenge, proof and validation, where the inference-only policy checks each execution block), then accounting once per monitoring period.

Sampled proofs run today as the [one-stage audit](verity/core/protocols/sampled_proofs/README.md). Both parties hold the circuit and the query that partitions it into proof blocks, and one round goes like this:

~~~text
developer                                          auditor
---------                                          -------
serve the workload, committing every value
that crosses a proof-block boundary
register the commitment roots   --------------->   check the registration
                                <---------------   draw proof blocks from its own randomness
prove the drawn blocks (C-Flock) -------------->   check the proofs (the Lean verifier)
                                                   record the integrity profile
~~~

The auditor never runs the computation; it only checks proofs. The integrity profile it records bounds how many proof blocks are faulty, given the committed trace.

## Architecture

Each layer may depend on the layers below it (the import rules: `AGENTS.md`, "Map and commands"):

~~~text
integrations/vllm, examples/, benchmarks/ (with the census)
---------------------------------------------------------------------------
verity/core/: the verifiers, trusted
  protocols: profile, sampled_proofs, pouw, pous, certifier; claims
  Lean, the verifiers, C-Flock's in service/flock
Security/: the definitions, models and proofs of the guarantees (the inference-only policy's among them)
---------------------------------------------------------------------------
verity/ml/: Python and CUDA: the provers, and Python that mirrors the Lean
  flock: C-Flock, the proof system
  circuits: evaluation, boolean, Definitions (ops), silicon semantics, targets
  kernels, commitments, crypto, randomness, private_circuits, definitions
~~~

`tools/` and `infra/` sit outside the layers: `tools/research` runs and records the work and depends on none of them, and nothing outside `infra/` imports `infra/`.

### Circuits

`verity.ml.circuits` defines circuits. A program is one circuit, stored compressed as Definitions called from a root body, with a canonical layout that locates any gate without expanding it and a SHA-512 digest that names it. `verity.ml.circuits.ops` holds the primitive circuits for NVIDIA accelerators, `verity.ml.circuits.silicon` their exact hardware semantics, and `verity.ml.circuits.evaluation` evaluates any circuit exactly. A circuit describes logical computation, never a GPU schedule, a proof system, a commitment layout or a sampling policy. Circuits are total: they match the hardware on every input bit pattern, NaN and infinity included. Applications define only their own Definitions, and every new or changed circuit passes `circuit-check`; that a circuit computes what the served model computes is completeness, validated empirically and never proved (`AGENTS.md`, "Design rules").

A partition divides a circuit's computed gates into proof blocks, the result of a named, versioned query the auditor evaluates itself (`verity/partition/v1`). It fixes what the developer commits: every value read across a proof-block boundary, plus the inputs and outputs.

### Commitments

`verity.ml.commitments` defines the commitment schemes: leaf rules, Merkle trees and openings, each with a specification and conformance vectors that fast committers match byte for byte; integrations never define their own. A commitment shows that the values a proof uses are the values bound earlier, and the proof shows how they are related: for a dot product, the commitment establishes that x, w and y are the values in the committed trace, and the proof that y = ExactDot(x, w). A proof system's internal commitments stay inside it.

### Proofs and protocols

`verity.core.protocols.profile` defines the integrity profile, what an accepted audit established. A proof counts only when it verifies against the intended statement and an approved prover identity, not merely some self-consistent statement. Which parts of a circuit are checked, how they are sampled and what is committed before a challenge are protocol questions, and none changes the circuit. Each protocol is a package under `verity.core.protocols` whose Lean is its definition and its verifier (`AGENTS.md`, "Design rules").

### The accounting layer

The logical layer is the cryptography: PoUW, PoUS and PoUB prove lower bounds on what the prover system did, with no device premise. The physical layer reads them against registered hardware, named assumptions and trusted devices' measurements: the device graph, whose zones (a node to start) are spatial units; placement; the timebase; trusted observers; boundary bounds; attribution; and the joint ledger.


- **Compute.** PoUW credits the noised products' size S, at least (1 − ε)·S of its AND gates right ([GLOSSARY.md](GLOSSARY.md), "Credit"). Traffic alone attributes nothing: a zone is credited only through an anchor, fresh trusted data it must consume (bounded by its egress), or the named premise that the fleet is closed.
- **Memory.** PoUB proves bits read, and PoUS bits held, at no location. Attribution places them in tiers, less what boundary links carry in.
- **Network.** Certifiers prove ports carried only values in the committed trace, timing aside; gaps cost line rate.

The joint ledger charges each device's time once. Without attribution, it subtracts only from the whole fleet's capacity, if no unregistered hardware helped.

### The proof system and the proof service

C-Flock (`verity/ml/flock`) is the proof system: it proves statements over arbitrary Boolean circuits, and its verifier is Lean (`verity/core/service/flock/`), with upstream Flock's Rust verifier as a cross-check. It consumes circuits and commitments, never branches on a model name or another source-system label, and is testable without an integration. Every accounting protocol uses the one proof service ([GLOSSARY.md](GLOSSARY.md)). A plain path (C-Flock without `--zk`) stays beside the zero-knowledge ones, for a configuration whose values needn't be hidden; its guarantee is soundness alone.

### Integrations

An integration makes Verity work end to end with a real system, and to Verity it is only a frontend: it hands over a circuit partitioned into proof blocks, the advice a registered compiler compiles, the committed trace and, for a sample, the sampled blocks' openings (`AGENTS.md`, "Integrations are frontends"). The proof service does the rest. Any serving stack plugs in without changing `verity/`, and nothing under `verity/` imports an integration (`tests/test_integration_boundaries.py` lists the sites that still do). The vLLM integration (`integrations/vllm/`), the only one so far, builds each served workload's circuit with an embedded Torch-to-circuit frontend, supplies its silicon-specific bindings, and commits the trace while serving.

## Documentation

- Specifications: the [program format](verity/ml/circuits/PROTOCOL.md), the commitment schemes ([frame-v3](verity/ml/commitments/frame_v3/PROTOCOL.md), [vllm-v1](verity/ml/commitments/vllm_v1/PROTOCOL.md)), the protocols, stated in Lean (`Security/`) with each one's README as its outline (a protocol's `PROTOCOL.md` remains only until its Lean covers it), and C-Flock's verifier, stated in Lean (`verity/core/service/flock/`), with [its outline](verity/ml/flock/verifier/PROTOCOL.md).
- Each subprotocol's README; [AGENTS.md](AGENTS.md) for contributors (the map, the rules, how a change lands).

## Glossary

The one glossary is [GLOSSARY.md](GLOSSARY.md): the terms, the one symbol table, the older names and the words that
keep a narrower sense.
