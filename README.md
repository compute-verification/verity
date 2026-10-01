# Verity

Verity is a system for verifying how AI datacenters are being used. It aims for security in extremely adversarial settings: the intended application is enforcing an international agreement to [deliberately pace AI development](https://www.pacingthefrontier.com/).

At a high level, it is a streaming interactive protocol between a *developer* and an *auditor*, in which the developer maintains private records of the computations it performs in its datacenters and continuously proves that these records have certain properties, which we have chosen as proxies for the following idealized properties:

- **Faithfulness:** the records accurately and comprehensively reflect the developer's actual computations.
- **Inference-only:** the records reflect only having performed inference, not training.

The developer represents its computations as Boolean circuits, which capture their numerical semantics down to the silicon level (though the logical circuit need not exactly match the structure of the physical circuit). To enable this, we provide a [Python API](packages/verity/README.md) for building and efficiently querying massive Boolean circuits, and a library of primitive circuits that capture the low-level semantics of NVIDIA accelerators, rigorously tested using the [Hawkeye](https://arxiv.org/abs/2603.20421) methodology.

To verify *faithfulness*, it currently implements a composition of the following protocols:

1. [Sampled proofs](protocols/sampled_proofs/README.md)
2. [Proofs of useful work](protocols/pouw/PROTOCOL.md)
3. [Proofs of useful space](protocols/pous/README.md)

To verify *inference-only*, it checks that the developer's committed circuits exhibit *bounded cross-episode novel communication intensity*: that is, that each can be partitioned into subcircuits, each of which has a bounded size and reads a bounded amount of novel information, defined as any information other than the models, code and data the developer committed up front. This exploits a fundamental asymmetry between inference and training: each inference episode (e.g., a chat session) depends on only a small amount of novel information (e.g., 100 KB worth of prompt tokens), whereas each training episode (e.g., an RL rollout) depends on an enormous amount of novel information (e.g., 1 TB of new parameters). It is conjectured that circuits with low cross-episode novel communication intensity are unable to advance frontier capabilities on relevant timescales (e.g., years).

## How an audit works

Sampled proofs run today as the [one-stage audit](protocols/one_stage/PROTOCOL.md). Both parties hold the circuit and the query that partitions it into proof units, and one round goes like this:

~~~text
developer                                          auditor
---------                                          -------
serve the workload, committing every value
that crosses a proof-unit boundary
register the commitment roots   --------------->   check the registration
                                <---------------   draw proof units from its own randomness
prove the drawn units (C-Flock) --------------->   check the proofs (the Lean verifier)
                                                   record the integrity profile
~~~

The auditor never runs any part of the computation; it only checks proofs. The integrity profile it records bounds how many proof units are faulty, given the committed values.

## Architecture

Each layer may depend on the layers below it:

~~~text
integrations/vllm, benchmarks/
---------------------------------------------------------------------------
protocols/                    backends/
  one_stage                     flock                    C-Flock and its Lean verifier
  sampled_proofs                gkr, direct, sp1, vole   frozen
  pous
  pouw
  network_warden
---------------------------------------------------------------------------
packages/verity, the core library:
  ir, ml, evaluation, commitments, randomness, proofs, claims
~~~

`tools/research` sits outside the layers: it runs and records the work, and depends on none of them.

### Circuits

`verity.ir` defines circuits. A program is one circuit, stored compressed as Definitions called from a root body, with a canonical layout that locates any gate without expanding the circuit and a SHA-512 digest that names it. `verity.ml` holds the primitive circuits for NVIDIA accelerators, and `verity.evaluation` evaluates any circuit exactly. A circuit describes logical computation, not a GPU schedule, a proof system, a commitment layout or a sampling policy. Circuits are total by default: they match the hardware on every input bit pattern, NaN and infinity included. Core owns every primitive circuit and all silicon semantics; applications define only their own Definitions. Every new or changed circuit passes `circuit-check`.

A partition divides a circuit's gates into proof units, each gate in exactly one. It is the result of a named, versioned query that the auditor evaluates itself (`verity/partition/v1`), and it fixes what the developer must commit: every value read across a unit boundary, plus the inputs and outputs.

### Commitments

`verity.commitments` defines the commitment schemes the records use: leaf rules, Merkle trees and openings, each with a specification and conformance vectors. Integrations never define their own, and fast committers match the vectors byte for byte. A commitment shows that the values a proof uses are the values bound earlier, not how they are related. A proof of a dot product, for example, establishes both

~~~text
x, w and y are the previously committed values
y = ExactDot(x, w)
~~~

The commitment establishes the first, and a backend's proof the second. A proof system's own internal commitments stay inside its backend.

### Proofs

`verity.proofs` defines backend-independent statements, how a backend's proof of them is judged, and the integrity profile, which records what an accepted audit established. A proof counts only when it verifies against the intended statement and an approved backend identity; a verifier that accepts some self-consistent statement is not enough.

### Protocols

Which parts of a circuit are checked, how they are sampled and what must be committed before a challenge are protocol questions, and none of them changes the circuit. Each protocol under `protocols/` is its own distribution and imports only the standard library, `verity` and itself.

### Backends

A backend proves statements. C-Flock (`backends/flock`) is the backend in active development; its verifier of record is written in Lean, and upstream Flock's Rust verifier is a cross-check. A-GKR, B-Ligero, D-SP1 and the VOLE backend are frozen: kept, and still benchmarked, but not developed. A backend consumes backend-independent objects, never branches on a frontend, a model name or another source-system label, and is testable without a frontend or an integration.

### Integrations

An integration makes Verity work end to end with a real system. The vLLM integration builds each served workload's circuit with an embedded Torch-to-IR frontend, supplies its silicon-specific bindings, acquires and commits runtime values while serving, and runs proving. It composes the components above without owning their semantics: circuits belong to core, and commitment schemes to `verity.commitments`. Standalone frontends, which would translate a source representation such as PyTorch, JAX or ONNX into circuits, don't exist yet.

## Repository structure

- [`packages/verity/`](packages/verity/README.md): the core library: `verity.ir`, the [circuit format](packages/verity/src/verity/ir/PROTOCOL.md), and `verity.ml`, exact hardware semantics
- `verity.commitments` and `verity.proofs`, also in [`packages/verity/`](packages/verity/README.md): commitments to the records, the statements a backend proves, and the integrity profile an audit returns
- [`backends/flock/`](backends/flock/README.md): C-Flock, the proof backend in active development; the frozen backends are listed in [`backends/`](backends/README.md)
- [`backends/flock/verifier/`](backends/flock/verifier/README.md): the verifier of record, written in Lean
- [`protocols/one_stage/`](protocols/one_stage/README.md) and [`protocols/sampled_proofs/`](protocols/sampled_proofs/README.md): sampled proofs, as the one-stage audit and the two-stage law
- [`protocols/pous/`](protocols/pous/README.md): proofs of useful space
- [`protocols/pouw/`](protocols/pouw/PROTOCOL.md): proofs of useful work
- [`protocols/network_warden/`](protocols/network_warden/README.md): the network warden, network transparency's timing channel
- [`integrations/vllm/`](integrations/vllm/README.md): the vLLM integration, a hooked vLLM engine that builds each served workload's circuit and commits its values while serving
- [`tools/research/`](tools/research/README.md): the research harness, for recorded runs on remote pods and the evidence store that keeps their results; it treats every workload as opaque and imports nothing from Verity
- [`tools/check/`](tools/check/check.py): `check`, the one check a merge requires, and `suites.py`, which runs the test suites
- `tools/circuit_check/`: `circuit-check`, the property checks every circuit passes ([rules](AGENTS.md#circuits))
- [`tools/lean/`](tools/lean/README.md): the audit every Lean package passes
- [`tools/cluster/`](tools/cluster/README.md): `cluster`, the shared machines' description, a pure per-node planner and the allocation ledger
- `tools/tc_probe/`, `tools/tc_probe_fp4/`, `tools/native_peak/`: GPU probes that compare tensor-core instructions bit for bit with the primitive circuits' models, and measure native peak throughput
- `benchmarks/`: reproducible workloads and measurements, including the [one-stage audit end to end](benchmarks/one_stage/README.md)
- [`census/`](census/README.md): data only: hardware, datatypes, networks, subcircuits, input sets and workloads
- `fixtures/` and `tests/`: repository-wide fixtures, and the repository's own suite of whole-tree invariants; replicated inputs are registered in `fixtures/artifacts.json`

## Getting started

Install the workspace with [uv](https://docs.astral.sh/uv/), then run the tests and checks:

~~~sh
uv sync --all-packages --extra torch-cpu # the workspace, every package editable, with CPU-only torch (GPU work: a member's `torch` extra)
uv run tools/check/suites.py             # every test suite whose inputs changed since it last passed; --quick skips slow tests
uv run circuit-check --all               # the property checks on every registered circuit
uv run --extra torch-cpu python tools/check/check.py  # the one check a merge requires; its Lean steps need lake
~~~

Each package's tests live in its own `tests/` and run as their own suite; the [testing skill](.agents/skills/testing/SKILL.md) has the rules. Serving, proving and hardware probes run on GPU pods through the research harness (`research run --on ...`), never on a laptop.

Build, evaluate and query a circuit from the primitive library:

~~~python
from verity.evaluation import evaluate
from verity.ir import Program, query
from verity.ir.defs import bind
from verity.ir.partition_object import program_sha512
from verity.ml.gemm import GEMM_AMPERE, Gemm

gemm = bind(Gemm, K=64, N=8, **GEMM_AMPERE)   # x[64] @ W[8, 64]^T in BF16, as an A100's tensor cores compute it
x, W = [0x3F80] * 64, [0x3F80] * (8 * 64)     # BF16 encodings of 1.0, with W flattened row by row
evaluate(gemm, x, W)                          # eight BF16 words 0x4280, that is 64.0
program = Program(gemm)
program_sha512(program)                       # the circuit's name: its SHA-512 digest
query(program.circuit, "//GemmCoordinate")    # its 8 output coordinates, each with its interval of gates
~~~

## Development

Dependencies point inward: integrations, backends, protocols and benchmarks depend on the core, and the core's modules depend on each other in one direction:

~~~text
ir  <-  evaluation  <-  ml
ir  <-  commitments
ir + ml + commitments  <-  proofs
randomness, claims: import no other core module
~~~

`packages/verity/tests/test_boundaries.py` and `tests/test_protocol_boundaries.py` enforce these rules. Import each object from the module that owns it (`from verity.ir import Program`); `verity` itself re-exports nothing.

There is no GitHub Actions CI. `check` is the one check, and a merge into `main` goes through `research merge`, which refuses without a passing `check` of that exact commit. [AGENTS.md](AGENTS.md) has the full rules for contributors and agents: commands, circuits, checking and merging, Lean, and where writing goes.

## Documentation

- Specifications: the [program format](packages/verity/src/verity/ir/PROTOCOL.md), the commitment schemes ([frame-v3](packages/verity/src/verity/commitments/frame_v3/PROTOCOL.md), [vllm-v1](packages/verity/src/verity/commitments/vllm_v1/PROTOCOL.md)), the protocols ([one-stage](protocols/one_stage/PROTOCOL.md), [two-stage](protocols/sampled_proofs/PROTOCOL.md), [proofs of useful space](protocols/pous/PROTOCOL.md), [proofs of useful work](protocols/pouw/PROTOCOL.md), [the network warden](protocols/network_warden/PROTOCOL.md)) and [C-Flock's verifier](backends/flock/verifier/PROTOCOL.md)
- Each component's README, linked under Repository structure
- [AGENTS.md](AGENTS.md) for contributors and agents, and the skills under [`.agents/skills/`](.agents/skills/) for recurring workflows such as Lean proofs

## Glossary

Code, docs, CLI help and output fields use these terms. When a term changes, this section changes with it, and the new term is rolled out in the same change.

- **Developer** and **auditor**: the protocol's two parties. The developer runs computations in its datacenters, keeps private records of them and proves properties of those records; the auditor checks the proofs, acting as the verifier.
- **Program**: the computation a proof claims something about (`verity.ir.Program`): one circuit, stored compressed as Definitions called from a root body, with the same interface at every level. The proof system is generic, so "model" never names it.
- **Circuit**: gates and the wires between them. A program is a compressed circuit; an *expanded* circuit is written out gate by gate, like the files C-Flock's circuit prover proves. A backend's lowering of a statement is also a circuit, as in A-GKR's pinned circuits or "SHA-256 in circuit".
- **Gate**: a node of a circuit: an input gate, the constant 0 or 1, AND, XOR or NOT. Each gate outputs one bit.
- **Wire**: an edge from a gate to a gate that reads it. A wire carries its source gate's value and has none of its own.
- **Port**: a named array of fixed-width elements on the boundary of a circuit definition; each call binds it to gates of the caller.
- **Element**: one fixed-width member of a port, such as 16 bits. A format labels elements, a leaf serializes whole elements, and no proof unit splits one.
- **Format**: a label on a port's elements, such as BF16, FP32, E4M3, int32 or token id, saying how an element's bits encode a number or symbol. Formats are metadata, outside the content digest (`verity.ir.annotations`).
- **Value**: the bit a gate takes in one run. The transcript is all of them (`verity.ir.evaluate.ExecutionTranscript`), and the value of an element or port is the transcript read at its gates, in order. Gates, wires, ports and elements are structure, the same in every run; values are one run's data. The messages between prover and verifier are the *proof transcript*.
- **Subcircuit template**: a named template with typed parameters, such as `gemm_coordinate(K, semantics)` or `rmsnorm(d, semantics)`. Shape and numeric (silicon) semantics are both parameters; a semantics value is an instruction id of `verity.ml.tc.instructions.INSTRUCTIONS`.
- **Subcircuit**: a subcircuit template with every parameter bound, such as `gemm_coordinate(K=1536, semantics=sm90.wgmma.m64n8k32.e4m3)`. A backend proves subcircuits, and a benchmark cell measures one.
- **Input**: one concrete assignment of values to a subcircuit's input gates, that is, to the elements of its input ports, with the output the chip produced recorded alongside.
- **Input set**: a batch of inputs for one subcircuit, with their provenance (captured or synthetic).
- **Commitment scheme**: how serving committed the values being proven, such as frame-v3 rows or `vllm-v1` (`verity.commitments`). Each scheme has a stable id and lists the hash assumptions it rests on. `frame-v3-sha512` and `vllm-v1-sha512` are the same framings through SHA-512, for the re-baseline's SHA-512 data commitments.
- **Hiding leaf**: a leaf whose published data hides the committed value, statistically given uniform salts, so a low-entropy value can't be searched from its digest. `hm96-sha256/v1` and `hm96-sha512/v1` (`verity.commitments.hm96`) wrap a scheme's SHA-256 or SHA-512 leaf digest with a salt stored beside the value; a leaf opens with its value and salt. With its pinned key the bound is N·2^-192 for N leaves, except for a 2^-64 fraction of keys (a common-reference-string step).
- **Statement**: what a backend proves, the full committed relation: a circuit (a subcircuit, or the proof units an audit drew) plus the commitment scheme that binds its inputs and outputs. A backend lowers a statement into its own circuit. Changing the scheme changes the statement, not the backend.
- **Backend family**: A-GKR, B-Ligero, C-Flock or D-SP1. The letter is part of the name. C-Flock is in active development; A-GKR, B-Ligero and D-SP1 are frozen: kept, and still in the benchmark tables, but not developed. **C-Flock's circuit prover** proves a statement over an expanded circuit (`verity/flock-circuit`); **C-Flock ZK** is its zero-knowledge version, in development.
- **Backend**: an implementation of the Backend interface, that is, one concrete member of a family: A-interactive, A-fs, B-interactive, B-fs, C-interactive or D-fs, where fs means Fiat–Shamir. A composed backend gets its own id: route (a), A-GKR with C-Flock and a link, is `A-route-a`. A scheme is not part of a backend's identity.
- **Cell**: one benchmark result, a backend × statement × input set × prover hardware.
- **Partition**: a division of a program's gates into proof units, each gate in exactly one (`verity/partition/v1`, `verity.ir.partition_object`). It is a named query on the program: the object holds the program's digest and the query's name, version and parameters, such as `Q_word` v1 `{X: 16, W: 32}`, and a verifier evaluates the query itself. The committed set, every gate read across a unit boundary plus the program's outputs, follows from the partition and is never stored.
- **Proof unit**: one part of a partition, proved on its own. A unit is incorrect when its committed outputs differ from its gates applied to its committed inputs. An audit's verifier draws the proof units to prove.
- **Replay unit (RU)**: in sampled proofs' two-stage law (`protocols/sampled_proofs/PROTOCOL.md`), a part of a Program that the prover replays to build the witness; the verifier draws replay units, then proof units inside each drawn one. "Unit" never names a subcircuit.
- **Instrumented program**: the program plus what serving adds to measure it: a tap on every committed element, and the leaf and tree hashing whose outputs are the roots. It is derived from the program, the partition and the commitment scheme, never authored, and no proof claims anything about it.
- **Integrity profile**: the audit's output, an object of its own (`verity.proofs.profile.IntegrityProfile`): per-stratum bounds on how many proof units of a partition are faulty (a unit is faulty when its committed outputs differ from its gates applied to its committed inputs), from the facts checked exhaustively, the verifier's draw law and its failure probability δ. Applications consume it with their own weights (`verity_one_stage.consumers`): exfiltration security weights faulty units by their output bits and adds which units are faulty, compute security weights them by their work.
- **Verifier**: the party that checks proofs. It issues each session's coins and draws the proof units to prove, in the one-stage audit from its own randomness; the coin server that hands out coins is part of it, not a separate party. C-Flock's verifier of record is written in Lean (`backends/flock/verifier/`), and upstream Flock's Rust verifier is a cross-check.
- **Frontend**: translates a source representation such as PyTorch, JAX or ONNX into Verity IR. The vLLM integration's `verity_vllm.program.frontend` is an embedded Torch→IR frontend, the seed of `frontends/torch` once a second consumer needs it.
- **Integration**: makes Verity work end to end with a real system or runtime, composing Verity's components around it. vLLM is an integration (`integrations/vllm`), an application of Verity, not a frontend.
- **Serving** and **proving**: serving is the live inference path, and its cost is serving overhead, such as computing commitments while serving; proving is proof generation, and its cost is proving overhead. Neither is called "production".

Older names stay where they are baked into a digest, a frozen fixture, a recorded artifact field or a pinned vector. They mean:

| Old name, where it stays | Term |
|---|---|
| "Candidate A", "Candidate B"; the `candidate` feature and label keys, `CandidateA`, `CandidateB`, `account_candidate_a` | backend family A-GKR, B-Ligero |
| the `candidate` label values `SP1` and `C` (`research.store.vocab.CANDIDATES`) | D-SP1, and the VOLE candidate (not C-Flock) |
| `verity.proofs.target.Target`, `TARGETS`, `FIRST`; the result fingerprint's `profile` | subcircuit (a `GemmCoordinate<K>` at one semantics) |
| `Target.family`, `FAMILY_DIGEST`, a gate family such as `veritor.tc-ampere-bf16@1` | the semantics parameter's gate set |
| "configured backend" | backend |
| "instance", "instance set"; the `bench-instances/*` datasets and `fixtures/bench-instances/`, the fingerprint's `instances` (`contract.instances_ref`) and its tier (`vu-k1536`, …), the `instance-equiv/v1` and `vllm-vu-set/v1` kinds, the `instances_dataset` / `instances_tier` labels, `instances_per_second` | input, input set |
| "configuration" meaning a backend with a scheme; `drilldown.Variant`, `VARIANTS`; `views.Configuration`, `CONFIGURATIONS` | a backend and a statement |
| `verity.proofs.statement.Statement` (statement formats 2 and 3 on the wire), its `KindProgram` and "kind" | the SP1-era wire encoding of a batch of proof-unit obligations, not a statement in this sense; a kind is a subcircuit |
| "verification unit" (VU); `vu_*` identifiers, `select_verification_units`, the `verity/sampled-proofs/vu-sample/v1` domain, the `vllm-vu-set/v1` kind, the `vu-k1536` tier | proof unit |
| "M0"; its first statement id, baked into the Lean verifier's `Flock/Tags.lean` | C-Flock's circuit prover: module `verity_flock.circuit`, CLI `flock-circuit`, statement `verity/flock-circuit` |
| "M1", "M2" (elsewhere in the code these also name unrelated things) | C-Flock ZK |
| "coin server"; the `round_digest` strings of statement identities in `Flock/Tags.lean` | the verifier |

Some words keep a different, narrower sense:

- `verity.ir`'s `Circuit` is a Program's canonical gate layout and a `GateSelection` is a selected set of its gates; its query `Family` is an indexed sequence of gate sets; a primitive's `family` is the id stem of lazily materialised primitives such as `Const<w>`. The `subcircuits` of a `typed-obligation/v0` object (`verity.proofs.typed_obligation.Subcircuit`) are its shared gate bodies, and a sampled-proofs obligation's JSON `subcircuit` key holds its gate selection; both names are serialized and stay. `verity.ir`'s gates are word-level: each outputs one `Value<w>` of w bits, not one bit.
- A **template instance** (the queries `Q_template_instance` and `Q_template_instances`) is one occurrence of a subcircuit template in a program, and in a claim id `property/instance`, such as `cr/sha-512`, the instance is the primitive an assumption is about; neither is an input.
- A **transition unit** is one tensor-core accumulator step; a GEMM-coordinate subcircuit chains `K/16` of them, and a "unit circuit" arithmetises one.
- "Candidate C" and "Candidate V" in older notes are the QuickSilver VOLE research backend in `backends/vole`, not C-Flock.
- In `integrations/vllm`, "family" and "kind" name IR definitions and query families, many of them serialized fields (see its README's vocabulary).
