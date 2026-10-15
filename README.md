# Verity

Verity is a system for verifying how AI datacenters are being used. It aims for security in extremely adversarial settings: the intended application is enforcing an international agreement to [deliberately pace AI development](https://www.pacingthefrontier.com/).

At a high level, it is a streaming interactive protocol between a *developer* and an *auditor*, in which the developer maintains private records of the computations it performs in its datacenters and continuously proves that these records have certain properties, which we have chosen as proxies for the following idealized properties:

- **Faithfulness:** the records accurately and comprehensively reflect the developer's actual computations.
- **Inference-only:** the records reflect only having performed inference, not training.

The developer represents its computations as Boolean circuits, which capture their numerical semantics down to the silicon level (though the logical circuit need not exactly match the structure of the physical circuit). To enable this, we provide a [Python API](verity/README.md) for building and efficiently querying massive Boolean circuits, and a library of primitive circuits that capture the low-level semantics of NVIDIA accelerators, rigorously tested using the [Hawkeye](https://arxiv.org/abs/2603.20421) methodology.

To verify *faithfulness*, it currently implements a composition of the following protocols:

1. [Sampled proofs](verity/core/protocols/sampled_proofs/README.md)
2. [Proofs of useful work](verity/core/protocols/pouw/PROTOCOL.md)
3. [Proofs of useful space](verity/core/protocols/pous/README.md)

To verify *inference-only*, it checks that the developer's committed circuits exhibit *bounded cross-episode novel communication intensity*: that is, that each can be partitioned into subcircuits, each of which has a bounded size and reads a bounded amount of novel information, defined as any information other than the models, code and data the developer committed up front. This exploits a fundamental asymmetry between inference and training: each inference episode (e.g., a chat session) depends on only a small amount of novel information (e.g., 100 KB worth of prompt tokens), whereas each training episode (e.g., an RL rollout) depends on an enormous amount of novel information (e.g., 1 TB of new parameters). It is conjectured that circuits with low cross-episode novel communication intensity are unable to advance frontier capabilities on relevant timescales (e.g., years).

## How an audit works

Sampled proofs run today as the [one-stage audit](verity/core/protocols/sampled_proofs/README.md). Both parties hold the circuit and the query that partitions it into proof units, and one round goes like this:

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
integrations/vllm, benchmarks/ (with the census)
---------------------------------------------------------------------------
verity/core/: the verifiers, trusted           archive/
  protocols: profile, sampled_proofs,          kept, not developed:
  pouw, pous, certifier; claims                gkr, direct, sp1, vole
  the Lean ports; service/flock: C-Flock's Lean verifier
Security/: the definitions, models and proofs of the guarantees (nci among them)
---------------------------------------------------------------------------
verity/ml/: Python and CUDA; what core's verifiers read of it is on the soundness path
  flock: C-Flock, the proof system
  circuits: evaluation, boolean, Definitions (ops), silicon semantics, targets
  kernels, commitments, crypto, randomness, private_circuits
~~~

`tools/` and `infra/` sit outside the layers: `tools/research` runs and records the work and depends on none of them, and
nothing outside `infra/` imports `infra/`.

### Circuits

`verity.ml.circuits` defines circuits. A program is one circuit, stored compressed as Definitions called from a root body, with a canonical layout that locates any gate without expanding the circuit and a SHA-512 digest that names it. `verity.ml.circuits.ops` holds the primitive circuits for NVIDIA accelerators, and `verity.ml.circuits.evaluation` evaluates any circuit exactly. A circuit describes logical computation, not a GPU schedule, a proof system, a commitment layout or a sampling policy. Circuits are total by default: they match the hardware on every input bit pattern, NaN and infinity included. `verity.ml.circuits` owns every primitive circuit and all silicon semantics; applications define only their own Definitions. Every new or changed circuit passes `circuit-check`.

A partition divides a circuit's gates into proof units, each gate in exactly one. It is the result of a named, versioned query that the auditor evaluates itself (`verity/partition/v1`), and it fixes what the developer must commit: every value read across a unit boundary, plus the inputs and outputs.

### Commitments

`verity.ml.commitments` defines the commitment schemes the records use: leaf rules, Merkle trees and openings, each with a specification and conformance vectors. Integrations never define their own, and fast committers match the vectors byte for byte. A commitment shows that the values a proof uses are the values bound earlier, not how they are related. A proof of a dot product, for example, establishes both

~~~text
x, w and y are the previously committed values
y = ExactDot(x, w)
~~~

The commitment establishes the first, and the proof system's proof the second. A proof system's own internal commitments stay inside it.

### Proofs

`verity.core.protocols.profile` defines the integrity profile, which records what an accepted audit established. A proof counts only when it verifies against the intended statement and an approved prover identity; a verifier that accepts some self-consistent statement is not enough.

### Protocols

Which parts of a circuit are checked, how they are sampled and what must be committed before a challenge are protocol questions, and none of them changes the circuit. Each protocol is a package under `verity.core.protocols` and imports only the standard library, the parts of `verity.ml` it is allowed, itself and the protocols it builds on (PoUW builds on sampled proofs, and reads the primitive circuits' silicon semantics). Its Lean port lands beside it in `verity/core/`, one verifier at a time.

### The proof system and the proof service

C-Flock (`verity/ml/flock`) is the proof system: it proves statements over arbitrary Boolean circuits, and its verifier of record is written in Lean (`verity/core/service/flock/`), with upstream Flock's Rust verifier as a cross-check. It consumes circuits and commitments, never branches on a model name or another source-system label, and is testable without an integration. The proof systems before it (A-GKR, B-Ligero, D-SP1 and the VOLE backend) are in `archive/`: kept, and still in the benchmark tables, but not developed.

Every accounting protocol (PoUW, PoUS, network accounting) uses one proof service. A protocol states what is committed, which units need proofs, the public Program each must pass and what it does with the outcome; the service owns commitments, live coins and the record, the selection, proving, verifying and the integrity profile. On the developer's side it is the workers behind the firewall; on the auditor's, the challenger and the verifier, with network certifiers and memory challengers placed in the developer's site. Every protocol is zero-knowledge, and everything hashes with SHA-512.

### Integrations

An integration makes Verity work end to end with a real system. The vLLM integration builds each served workload's circuit with an embedded Torch-to-circuit frontend, supplies its silicon-specific bindings, acquires and commits runtime values while serving, and runs proving. It composes the components above without owning their semantics: circuits belong to `verity.ml.circuits`, and commitment schemes to `verity.ml.commitments`.

## Repository structure

- [`verity/`](verity/README.md): the library, the distribution `verity`, in two parts:
  - `verity/core/` (`verity.core`): the verifiers, trusted code: the Python verifiers, `verity.core.protocols` (the protocols and the integrity profile an audit returns) and the claims, their Lean ports, and `verity/core/service/flock/`, C-Flock's verifier of record. `Security/Models/` states their guarantees, each open until its proof lands
  - `verity/ml/` (`verity.ml`): Python and CUDA: `verity.ml.circuits` and the [circuit format](verity/ml/circuits/PROTOCOL.md), with exact hardware semantics (`silicon`), the primitive circuits' Definitions (`ops`) and the targets; `verity.ml.kernels`, the fast code checked against a reference in core or the circuits; `verity.ml.commitments` (commitments to the records), `crypto` and `randomness`; and [`verity/ml/flock/`](verity/ml/flock/README.md), C-Flock, the proof system, whose verifier's specification and scripts are [`verity/ml/flock/verifier/`](verity/ml/flock/verifier/README.md); what core's verifiers read of it is on the soundness path, and the rest (C-Flock's prover, the private circuits) can't affect soundness
- [`verity/core/protocols/sampled_proofs/`](verity/core/protocols/sampled_proofs/README.md): sampled proofs, as the one-stage audit and two-stage sampling
- [`verity/core/protocols/pous/`](verity/core/protocols/pous/README.md): proofs of useful space
- [`verity/core/protocols/pouw/`](verity/core/protocols/pouw/PROTOCOL.md): proofs of useful work
- [`verity/core/protocols/certifier/`](verity/core/protocols/certifier/README.md): the network certifier, network accounting's timing channel
- `Security/`: the Lean every protocol states its guarantees in (`Definitions/`, `Models/`) and their proofs (`Proofs/`)
- the proof systems before C-Flock are in [`archive/`](archive/README.md)
- `integrations/`: one directory per outside system; [`integrations/vllm/`](integrations/vllm/README.md) is the vLLM integration, a hooked vLLM engine that builds each served workload's circuit and commits its values while serving
- `benchmarks/`: the measuring side: reproducible workloads and measurements, including the [one-stage audit end to end](benchmarks/one_stage/README.md), and the benchmark tools
- `benchmarks/harness/` (`verity-bench`, `verity_bench`): the shared top-level harness; `verity-bench vllm` measures integrations as a user and leaves the integration independent of benchmark policy
- [`benchmarks/census/`](benchmarks/census/README.md): data only: hardware, datatypes, networks, subcircuits, input sets and workloads
- `benchmarks/tc_probe/`, `benchmarks/tc_probe_fp4/`, `benchmarks/native_peak/`: GPU probes that compare tensor-core instructions bit for bit with the primitive circuits' models, and measure native peak throughput
- `tools/verity/`: the `verity` command (`tools/verity/cli/`) and the tools it runs:
  - [`tools/verity/check/`](tools/verity/check/check.py): `check`, the one check a merge requires, and `suites.py`, which runs the test suites; `rename_merge.py` merges a PR onto a train tip that moved files it edits, applying its changes at the new paths or refusing with the paths
  - `tools/verity/circuit_check/`: `circuit-check`, the property checks every circuit passes ([rules](AGENTS.md#circuits))
  - [`tools/verity/lean/`](tools/verity/lean/README.md): the Lean check every Lean package passes
- [`tools/research/`](tools/research/README.md): the research harness, for recorded runs on remote pods and the evidence store that keeps their results; it treats every workload as opaque and imports nothing from Verity
- `infra/`: our machines; [`infra/cluster/`](infra/cluster/README.md) is `cluster`, the shared machines' description, a pure per-node planner and the allocation ledger, and `control/` holds the control pod's boot script and loops
- `fixtures/` and `tests/`: repository-wide fixtures, and the repository's own suite of whole-tree invariants; replicated inputs are registered in `fixtures/artifacts.json`

## Getting started

Install the workspace with [uv](https://docs.astral.sh/uv/), then run the tests and checks:

~~~sh
uv sync --all-packages --extra torch-cpu   # the workspace, every package editable, with CPU-only torch (GPU work: a member's `torch` extra)
uv run tools/verity/check/suites.py        # every test suite whose inputs changed since it last passed; --quick skips slow tests
uv run circuit-check --all                 # the property checks on every registered circuit
uv run --extra torch-cpu verity check      # the one check a merge requires; its Lean steps need lake
~~~

Each package's tests live in its own `tests/` and run as their own suite; the [testing skill](.agents/skills/testing/SKILL.md) has the rules. Serving, proving and hardware probes run on GPU pods through the research harness (`research run --on ...`), never on a laptop.

Build, evaluate and query a circuit from the primitive library:

~~~python
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits import Program, query
from verity.ml.circuits.defs import bind
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.ops.gemm import GEMM_AMPERE, Gemm

gemm = bind(Gemm, K=64, N=8, **GEMM_AMPERE)   # x[64] @ W[8, 64]^T in BF16, as an A100's tensor cores compute it
x, W = [0x3F80] * 64, [0x3F80] * (8 * 64)     # BF16 encodings of 1.0, with W flattened row by row
evaluate(gemm, x, W)                          # eight BF16 words 0x4280, that is 64.0
program = Program(gemm)
program_digest(program)                       # the circuit's name: its SHA-512 digest
query(program.circuit, "//GemmCoordinate")    # its 8 output coordinates, each with its interval of gates
~~~

## Development

Dependencies point inward: integrations, the proof system and benchmarks depend on `verity`, `verity` imports no other distribution, `verity.core` may import the parts of `verity.ml` that `ALLOWED` lists, which puts them on the soundness path, and `verity.ml` never imports `verity.core` (apart from a few tests and V*'s builders), nothing outside `infra/` imports `infra/`, and `verity`'s modules depend on each other in one direction:

~~~text
ml.circuits  <-  ml.commitments  ->  ml.crypto
ml.circuits  <-  ml.circuits.silicon, .targets, .ops, ml.kernels  (the primitive circuits and the kernels)
ml.circuits  <-  core.protocols.profile  <-  the protocols
ml.randomness, ml.crypto, core.claims: import no other module of verity
~~~

`tests/test_boundaries.py` enforces these rules. Import each object from the module that owns it (`from verity.ml.circuits import Program`); `verity` itself re-exports nothing.

There is no GitHub Actions CI. `check` is the one check, and `main` always passes it, kept fast by reusing a part's verdict while its declared inputs are unchanged. Until that reuse exists, a change lands once the parts it touches pass, and a full `check` of `main` follows; a landing it fails is reverted at once. [AGENTS.md](AGENTS.md) has the full rules for contributors and agents: commands, circuits, checking and merging, Lean, and where writing goes.

## Documentation

- Specifications: the [program format](verity/ml/circuits/PROTOCOL.md), the commitment schemes ([frame-v3](verity/ml/commitments/frame_v3/PROTOCOL.md), [vllm-v1](verity/ml/commitments/vllm_v1/PROTOCOL.md)), the protocols, stated in Lean (`Security/`) with each one's README as its outline ([sampled proofs](verity/core/protocols/sampled_proofs/README.md), [proofs of useful space](verity/core/protocols/pous/README.md), [proofs of useful work](verity/core/protocols/pouw/PROTOCOL.md), [the network certifier](verity/core/protocols/certifier/README.md); a protocol's `PROTOCOL.md` remains only until its Lean covers it) and [C-Flock's verifier](verity/ml/flock/verifier/PROTOCOL.md)
- Each component's README, linked under Repository structure
- [AGENTS.md](AGENTS.md) for contributors and agents, and the skills under [`.agents/skills/`](.agents/skills/) for recurring workflows such as Lean proofs

## Glossary

Code, docs, CLI help and output fields use these terms. When a term changes, this section changes with it, and the new term is rolled out in the same change.

- **Developer** and **auditor**: the protocol's two parties. The developer runs computations in its datacenters, keeps private records of them and proves properties of those records; the auditor checks the proofs, with a challenger online and a verifier offline, and places network certifiers and memory challengers inside the developer's site. Those devices are the auditor's hardware, each on a machine the developer doesn't control; nothing the developer runs or hosts is trusted.
- **Program**: the computation a proof claims something about (`verity.ml.circuits.Program`): one circuit, stored compressed as Definitions called from a root body, with the same interface at every level. The proof system is generic, so "model" never names it.
- **Registered input** and **run input**: the two binding times of a program's input gates (`verity.ml.circuits.program`). A registered input (`RegisteredInput<w>_v1`) is bound when the program is registered and holds what the program brings: its weights, constants such as a norm's epsilon, the attention scale or a softcap, and tables such as `gelu_tanh_bf16`. A run input (`Input<w>_v1`) is bound per request and holds the request's data and parameters, such as temperature, top-p and the Philox positions. Both are values, committed under hiding leaves, so a program holds no data; a constant gate is an operation's own bits (`verity.ml.circuits.constants`).
- **Operation**: a Definition whose own definition is gates: the floating-point formats' constants, the round constants of SHA-512 and Keccak-f, and every measured MUFU table (the library, `verity.ml.circuits.library`) are its bits, not a program's data. The operations are one pinned, versioned list (`verity.ml.circuits.operations`, `verity/operations` v1), and its guard (`operations.program_data`) lists the data a program still holds: a constant outside any operation, and a table a primitive reads that the library does not list. No statement or table claims that a program's values are always hidden until the guard returns {} for it.
- **Circuit**: gates and the wires between them. A program is a compressed circuit; an *expanded* circuit is written out gate by gate, like the files C-Flock's circuit prover proves. A proof system's lowering of a statement is also a circuit, as in A-GKR's pinned circuits or "SHA-512 in circuit".
- **Gate**: a node of a circuit: an input gate, the constant 0 or 1 as one of an operation's own bits, the AND or XOR of two or more bits, or NOT. Each gate outputs one bit (`verity.ml.circuits.boolean`). An operation's own definition is gates, its constants and the hardware's measured tables included: the floating-point formats' constants, SHA-512's, and the MUFU tables for exp2, rcp, tanh and the rest (`verity.ml.circuits.library`). A constant that a program or a request brings is a value, never a gate, and so is every other table, even one that tabulates a fixed function with no program parameter (`TanhF32Rn_v1`'s RN32 tanh).
- **Wire**: an edge from a gate to a gate that reads it. A wire carries its source gate's value and has none of its own.
- **Port**: a named array of fixed-width elements on the boundary of a circuit definition, one of its parameters or results; each call binds it to gates of the caller. A Call's result ports are the first place a partition tries to cut: under `Q_call` a Call whose results fit in X bits is one proof unit.
- **Element**: one fixed-width member of a port, such as 16 bits. A format labels elements, and a leaf serializes whole elements. A partition does not read elements: a proof unit's boundary is counted in bits, so a unit may commit some of an element's bits and not others. The rule that bounds a unit is its output width (see Proof unit).
- **Format**: a label on a port's elements, such as BF16, FP32, E4M3, int32 or token id, saying how an element's bits encode a number or symbol. Formats are metadata, outside the content digest (`verity.ml.circuits.annotations`).
- **Boolean Definition**: a Definition whose gates are only gates in the sense above (`verity.ml.circuits.boolean.is_boolean`); its meaning is its gates. A *word Definition* has gates that output a w-bit word, whose meaning is registry code; the word library migrates to Boolean Definitions under new versions of the same names.
- **Word view**: a Boolean Definition read with each element's bits packed into one integer (`verity.ml.circuits.boolean.word_type`); under it, `GemmCoordinate_v3` takes and returns exactly what the word Definition `GemmCoordinate_v2` does.
- **Value**: the bit a gate takes in one run. The *trace* is all of them (`verity.ml.circuits.evaluate.ExecutionTranscript`), and the value of an element or port is the trace read at its gates, in order. Gates, wires, ports and elements are structure, and so are shapes such as batch size and sequence length, public until a private-circuit protocol; values are data, even those that are the same in every run. Everything else a program or a request brings is a value like any bit a run computes: the weights, a program's constants (eps, the attention scale, a softcap) and tables (`gelu_tanh_bf16`, `tanh_rn`), and a request's parameters (temperature, top-p, Philox positions). Values are always hidden: a value is committed under a hiding leaf and touched only inside a zero-knowledge proof, never treated as part of the program the verifier holds. The messages between prover and verifier are the *proof transcript*; "transcript" names only those.
- **Subcircuit template**: a named template with typed parameters, such as `gemm_coordinate(K, semantics)` or `rmsnorm(d, semantics)`. Shape and numeric (silicon) semantics are both parameters; a semantics value is an instruction id of `verity.ml.circuits.silicon.instructions.INSTRUCTIONS`.
- **Subcircuit**: a subcircuit template with every parameter bound, such as `gemm_coordinate(K=1536, semantics=sm90.wgmma.m64n8k32.e4m3)`. The proof system proves subcircuits, and a benchmark cell measures one.
- **Input**: one concrete assignment of values to a subcircuit's input gates, that is, to the elements of its input ports, with the output the chip produced recorded alongside.
- **Input set**: a batch of inputs for one subcircuit, with their provenance (captured or synthetic).
- **Commitment scheme**: how serving committed the values being proven, such as frame-v3 rows or `vllm-v1` (`verity.ml.commitments`). Each scheme has a stable id and lists the hash assumptions it rests on. `frame-v3-sha512` and `vllm-v1-sha512` are the same framings through SHA-512, for the re-baseline's SHA-512 data commitments.
- **Hiding leaf**: a leaf whose published data hides the committed value, statistically given uniform salts, so a low-entropy value can't be searched from its digest. `hm96-sha512/v1` (`verity.ml.commitments.hm96`) wraps a scheme's SHA-512 leaf digest with a salt stored beside the value; a leaf opens with its value and salt. With its pinned key the bound is N·2^-192 for N leaves, except for a 2^-64 fraction of keys (a common-reference-string step).
- **Statement**: what the proof system proves, the full committed relation: a circuit (a subcircuit, or the proof units an audit drew) plus the commitment scheme that binds its inputs and outputs. The proof system lowers a statement into its own circuit. Changing the scheme changes the statement, not the proof system.
- **Proof system**: what proves statements. C-Flock is the proof system: **C-Flock's circuit prover** proves a statement over an expanded circuit (`verity/flock-circuit`), and **C-Flock ZK**, its zero-knowledge version, is what the proof service runs. The proof systems before it, A-GKR, B-Ligero, D-SP1 and the VOLE backend, are in `archive/`: kept, and still in the benchmark tables, but not developed. Their recorded cells name a *backend*, one concrete member of a family (A-interactive, A-fs, B-interactive, B-fs, C-interactive or D-fs, where fs means Fiat–Shamir, or a composed id such as `A-route-a`); a scheme is not part of a backend's identity.
- **Cell**: one benchmark result, a proof system (an archived one's backend id included) × statement × input set × prover hardware.
- **Partition**: a division of a program's computed gates into proof units, each in exactly one (`verity/partition/v1`, `verity.ml.circuits.partition_object`). A gate with no inputs, a registered or run input or a constant, is in no unit: each is bound outside the partition (`Q_word` also leaves out its structure: wiring, and gates computed from constants alone). It is a named query on the program: the object holds the program's digest and the query's name, version and parameters, such as `Q_call` v1, which cuts each Call down its hierarchy until every unit has at most `X` output bits, or `Q_word` v1. A verifier evaluates the query itself. The committed set, every gate read across a unit boundary plus every computing gate a Call returns, follows from the partition and is never stored.
- **Proof unit**: one part of a partition, proved on its own: a set of gates u whose inputs in(u) and outputs out(u) (`verity.ml.circuits.parts`) are ordered bit strings, counted in bits at any gate width. Its outputs are at most 16 bits, a literal limit (a query's `X`), and its inputs are unbounded. A unit is *faulty* when its committed outputs differ from its gates applied to its committed inputs (see Faulty unit). An audit's verifier draws the proof units to prove.
- **Declaration**: the circuit the developer registers and its baseline, the input gates whose values it committed beforehand (its models, code and data in storage); every other value is *novel information* (`Verity.Declaration`, in `Security/Definitions/Core/`).
- **Policy**: a condition on declarations that the verifier checks from the circuit and its baseline alone, a protocol with no rounds of its own. NCI's is the first (`Nci.Complies`).
- **Isolation unit (IU)** and **novel communication intensity (NCI)**: an IU is a part of a division of a declared circuit's computed gates, and its NCI is the novel information it reads, in bits, per gate it holds. NCI's policy asks for IUs of bounded size, each with bounded NCI (`Security/`).
- **Replay unit (RU)**: in sampled proofs' two-stage sampling (`verity/core/protocols/sampled_proofs/README.md`), a part of a Program that the prover replays to build the witness; the verifier draws replay units, then proof units inside each drawn one. "Unit" never names a subcircuit.
- **Instrumented program**: the program plus what serving adds to measure it: a tap on every committed gate, and the leaf and tree hashing whose outputs are the roots. It is derived from the program, the partition and the commitment scheme, never authored, and no proof claims anything about it.
- **Integrity profile**: the audit's output, an object of its own (`verity.core.protocols.profile.IntegrityProfile`): per-stratum bounds on how many proof units of a partition are faulty (a unit is faulty when its committed outputs differ from its gates applied to its committed inputs), from the facts checked exhaustively, the verifier's sampling design and its failure probability δ. Applications consume it with their own weights (`verity.core.protocols.sampled_proofs.consumers`): exfiltration security weights faulty units by their output bits and adds which units are faulty, compute security weights them by their work.
- **Faulty unit**: a proof unit whose committed outputs differ from its gates applied to its committed inputs, in the trace the developer committed. A unit fed wrong inputs that computes faithfully from them isn't faulty; the values downstream of a faulty unit just differ from the honest evaluation, and have no term of their own.
- **Sampling design**: the rule by which the verifier draws the sample of proof units from the partition, its population: `subset:K`, `bernoulli:NUM/DEN`, `stratified:K` (the default), `work:K`, or *two-stage sampling*, which draws replay units and then proof units inside each drawn one. Prose writes it in full. Code and records call it `law` (`draw.parse_law`, a registration's `law` key, `TwoStageLaw`), and those names stay until a format version renames them.
- **Acceptance probability** P_a(B): the probability, under the sampling design, that the sample holds no unit of a set B, so that faulty units B go unproved and the audit can still accept. Under two-stage sampling with interiors committed after the first draw, its effective form is P̄_a (the Lean's `effEscape`).
- **Confidence set** M_δ, for error δ: every set B of proof units with P_a(B) > δ. Except with probability δ + ε_ks + ε_bind, an audit either rejects or the committed trace's set of faulty units is in M_δ. A consumer's upper confidence bound on a total is the largest sum of its per-unit quantity y_u over a set of M_δ, and exfiltration counts every output value some set of M_δ could deliver (the Lean's `influenceSet`).
- **Commit-and-prove argument**: what proves the sample. The developer commits the trace, the verifier draws, and the argument proves that each drawn unit computes its committed outputs from its committed inputs. Its errors are knowledge soundness ε_ks, binding ε_bind (that an accepted extraction disagrees with the committed values; the Lean's `LinkSound` and `δlink`) and zero knowledge ε_zk. C-Flock instantiates it.
- **Proof service**: what lets the auditor check what the developer's AI servers did, under PoUW, PoUS and network accounting, while learning nothing about the weights or data beyond each protocol's stated public outputs. The protocols describe their checks; the service handles commitments, coins, proving and verifying. On the developer's side it is the workers behind a firewall; on the auditor's, a challenger and a verifier, with network certifiers and memory challengers placed in the developer's site. A protocol states only a `Spec` (`verity.core.protocols.sampled_proofs.service`): what it commits, its selection rule, its check as a public Program, and its public items, each with its reason.
- **Workers**: the developer's servers and GPUs. They run the computation and nearly all proving, with zero knowledge off, trusted by neither party, and reach the auditor only through the firewall.
- **Firewall**: the developer-trusted only path from the workers to the auditor. It commits every outbound message under a fresh salt, masks and checks or fixes everything else, and releases outer proofs, which are zero-knowledge, one at a time on a fixed schedule; only outer proofs reach the auditor. The prover is the workers behind it (P = F ∘ W).
- **Network certifier**: an auditor-trusted device on each node's uplink, untrusted by the developer (`verity.core.protocols.certifier`). It attests the contents of the traffic it carries in network certificates and scrubs its timing, which backs the auditor's network bound; every node uplink passes through one.
- **Network certificate**: a network certifier's record of what crossed a link (`grid.Record`, committed as a `CommittedRecord`), signed under the certifier's key over its SHA-512 record digest (`certificate.CertifiedRecord`). Certificates never reach the auditor in the clear: they are witness, and the developer proves that its committed circuits are consistent with them.
- **Memory challenger**: an auditor-trusted device beside a network certifier, the auditor's hardware on a machine the developer doesn't control at the developer's site (never a process on the prover's node), that runs PoUS's timed rounds and signs each one (`verity.core.protocols.pous.memory_challenger`'s `MemoryChallenger`). The per-round on-time bits stay in its signed record; PoUS publishes the audit's verdict.
- **Challenger**: the auditor's online service. It issues live coins, committed in advance, draws, and keeps the record; the coins the auditor sees come only from it, and it never sees the witness. In C-Flock its live-coin part is the Rust `Server` of the patched checkout. Fiat–Shamir's challenger keeps its own sense.
- **Verifier**: the auditor's Lean verifier of record, run offline, and the outcome it writes. C-Flock's is written in Lean (`verity/core/service/flock/`), and upstream Flock's Rust verifier is a cross-check. As a party in a protocol's statements, such as the one-stage audit's draw of proof units or a game's `Verifier`, "the verifier" is the auditor's side as a whole.
- **Selection rule**: which proof units of a registration are proved (`Spec.selection`): all of them, a one-stage sampling design such as `stratified:K`, two-stage sampling, or a timed session's sequential draw. It is drawn from the challenger's randomness; a policy is a condition on declarations, not a draw.
- **Device model**: the one device the compute, bandwidth and capacity audits are all about (`Accounting.Device`, in `Security/Definitions/Accounting/`): on-chip state of at most S_chip bits and device memory of at most M bits; at most BW bits per tick from device memory to the chip, and at most a channel's capacity per tick over each side channel (PCIe from NVMe and the network); and each op class (tensor-core multiply-adds, F per tick, CUDA-core ops and oracle queries) at a rate of its own. A tick is its unit of time, and a window is T consecutive ticks.
- **Schedule**: what the device does at each tick (`Accounting.Schedule`): one step per tick, which charges each resource it uses (the ops of each class, the bits read from device memory and over each side channel, the state held at the end of the tick) to a resource account. A step fits the device when the charges of all accounts together are within its limit for every resource (`Step.Fits`), so no unit of a resource is charged twice. A schedule is the device's; a circuit's run is its trace (see Value).
- **Resource account**: an account a schedule charges a resource to, and in a joint audit the account each resource is certified for (`Accounting.Resource`): *compute*, work at a price per op (PoUW's compute audit); *bandwidth*, bits read from device memory (the bandwidth audit); and *capacity*, device memory held, in bit-ticks over a window (one bit held for one tick is one bit-tick; PoUS's capacity audit). What the other accounts use is neither credited to an audit's account nor needed by it.
- **Certificate**: the amount of its resource an audit's guarantee says an accepted run's account used, except with the audit's error: W_cert of work, Q_cert bits read from device memory, and S_cert bits of device memory on average over the capacity window (S_cert · T bit-ticks) (`Run.ComputeSound`, `Run.BandwidthSound`, `Run.CapacitySound`). Each holds against the other audits' coins as advice, and bounds the account's worst case over the audit's own coins.
- **Covert residual**: what the device's limits leave the other accounts once an account has its certificate: F·T − W_cert of work, BW·T − Q_cert bits read and M − S_cert bits of device memory on average, over each audit's window (`Accounting.Guarantees.CovertResidual`). Weak form, worst case over the audit's own coins: except with the errors summed, some coin of each audit leaves the other accounts within its residual, not necessarily the coin drawn. The per-run form, at the coins drawn, is not proved yet.
- **Joint audit**: the compute, bandwidth and capacity audits run on one device at once (`Accounting.Run`), each with coins, a resource account and a window of its own; the device sees every coin. Its guarantee (`Accounting.Guarantees.Joint`) bounds by the three errors summed the probability that every audit accepts and some account is below its certificate, in the same weak form as the covert residual.
- **Guarantee** (security guarantee): a proved `Prop` in which named assumptions imply a conclusion, such as the network certifier's bound on how much a covert sender can carry out through when its frames leave. A Lean package laid out under the four names (`AGENTS.md`, "Lean") states each as a `def G : Prop` in its `Guarantees` and proves it in `SecurityProofs.G`; the packages not yet moved state theirs in layouts of their own (PoUW's `Pinned` modules, C-Flock's theorems). The lock records each guarantee's statement. A theorem is a guarantee when something relies on it as proved: a ledger row, a published table, the docs site, a claim id in `verity.core.claims`, or code that reads its result. A mention in prose, a PR, a test or an internal table doesn't make one. `verity.core.claims` names the kinds Table 1 cites (soundness, zero knowledge, …).
- **Assumption**: a named `Prop` that guarantees take as a hypothesis and nothing proves, listed by name with its own record in its package's lock (`lean-audit.json`), stated once with its rationale in its area's `Models/<Area>/Assumptions` (package `security`; C-Flock's in `Proofs/Flock/Soundness/Assumptions` until its spec is extracted); the audit admits no other unproved fact (no `axiom`, no `sorry`). One about a primitive has a claim id `property/instance` in `verity.core.claims`, such as `cr/sha-512`, SHA-512's collision resistance. Proving an assumption removes it from every guarantee that took it.
- **Lemma**: any proved statement that isn't a guarantee, even one a package's record lists. A package laid out under the four names keeps its lemmas in `SecurityProofs`, outside its spec, and they change freely as long as every guarantee still proves the same statement.
- **Spec**: a Lean package's trusted text, everything a reader must trust to read its guarantees: its `Protocol` (definitions), `Assumptions` and `Guarantees`, which import only one another, the `Core` area's spec and Mathlib (`layers` in `lean-audit.json`). Every line of a spec is trusted, so it is kept as short as possible, and the proofs, the lemmas and as much machinery as possible go in `SecurityProofs`.
- **Lock**: the records of a package's guarantees in its `lean-audit.json` (`guarantees` and `reads`): each guarantee's statement and every definition the statement reads, written by `tools/verity/lean/check.py --update`. A changed record means a guarantee now says something else, and Daniel gets a DM showing it when it lands on `main`; a guarantee that only moved keeps its records (`--moved`). The lock protects as little as possible, the guarantees and what their statements read, though today it still lists many lemmas as well.
- **Trusted computing base (TCB)**: what a guarantee's conclusion about the running system depends on beyond its assumptions: each spec, and the code that implements it for every role, verifier and prover, reference and fast. The TCB is trusted, not certified trustworthy: if its code doesn't implement its spec, security degrades, whether soundness for the auditor or zero knowledge and completeness for the developer. Other code is untrusted or is the workload being verified. The *assurance TCB* is what makes "this guarantee is proved" and "this commit passed `check`" believable: Lean's kernel and toolchain, the pinned Mathlib and ArkLib, the Lean audit (`tools/verity/lean/`), `check`'s gates, and upstream's Rust verifier through lean-agreement.
- **Frontend**: translates a source representation such as PyTorch, JAX or ONNX into a circuit (a Program). The vLLM integration's `verity_vllm.program.frontend` is an embedded Torch-to-circuit frontend; there is no standalone one.
- **Integration**: makes Verity work end to end with a real system or runtime, composing Verity's components around it. vLLM is an integration (`integrations/vllm`), an application of Verity, not a frontend.
- **Serving** and **proving**: serving is the live inference path, and its cost is serving overhead, such as computing commitments while serving; proving is proof generation, and its cost is proving overhead. Neither is called "production".
- **Approach**: a line of work (a scheme, a route or an attack) with a hypothesis, an owning lane and a status: live, parked, killed, superseded or merged. Each is a record in the evidence store (`research notes claim`, `research notes approach`, `research notes approaches`; the research notes' `kb/onboarding.md`), rendered into the notes as `campaigns/<campaign>/APPROACHES.md`, and every status but live gives its reason and evidence. A dead end is a killed or superseded approach, kept so nobody goes down it again unknowingly.

Older names stay where they are baked into a digest, a frozen fixture, a recorded artifact field or a pinned vector. They mean:

| Old name, where it stays | Term |
|---|---|
| "Candidate A", "Candidate B"; the `candidate` feature and label keys, `CandidateA`, `CandidateB`, `account_candidate_a` | the archived proof systems A-GKR, B-Ligero |
| the `candidate` label values `SP1` and `C` (`research.store.vocab.CANDIDATES`) | D-SP1, and the VOLE candidate (not C-Flock) |
| `verity.ml.circuits.targets.target.Target`, `TARGETS`, `FIRST`; the result fingerprint's `profile` | subcircuit (a `GemmCoordinate<K>` at one semantics) |
| `Target.family`, `FAMILY_DIGEST`, a gate family such as `veritor.tc-ampere-bf16@1` | the semantics parameter's gate set |
| "backend", "backend family", "configured backend" for C-Flock; `backends/` (C-Flock's directory before the platform layout), `verity_flock.backend`, `BACKENDS` | the proof system |
| "IR", "Verity IR", "IR primitives", "IR evaluator"; `verity.ir` | circuit, Program; `verity.ml.circuits` |
| "instance", "instance set"; the `bench-instances/*` datasets and `fixtures/bench-instances/`, the fingerprint's `instances` (`contract.instances_ref`) and its tier (`vu-k1536`, …), the `instance-equiv/v1` and `vllm-vu-set/v1` kinds, the `instances_dataset` / `instances_tier` labels, `instances_per_second` | input, input set |
| "configuration" meaning a backend with a scheme; `drilldown.Variant`, `VARIANTS`; `views.Configuration`, `CONFIGURATIONS` | a proof system and a statement |
| `verity_sp1.proofs.statement.Statement` (statement formats 2 and 3 on the wire), its `KindProgram` and "kind" | the SP1-era wire encoding of a batch of proof-unit obligations, not a statement in this sense; a kind is a subcircuit |
| "pin", "pinned theorem", "pinned statement"; `pins` in a `lean-audit.json` written before the rename (read as `guarantees`, rewritten by its next `--update`) and in `tools/verity/lean/Facts.lean`'s facts | guarantee |
| "law", "draw law"; `law` identifiers and record keys (`parse_law`, `same_law`, `law_object`, `TwoStageLaw`, a registration's `law`), the Lean's `Law` | sampling design |
| "escape", "escape probability"; the Lean's `*_escape` theorems and `effEscape` | acceptance probability (P_a; `effEscape` is P̄_a) |
| "wrong unit", "incorrect unit"; the Lean's `wrong`, the audit record's `wrong_units` | faulty unit |
| "transcript" for the values of a run's gates; `ExecutionTranscript` | trace |
| "δ_link", "the link term"; the Lean's `LinkSound` and `δlink` | ε_bind, the binding error |
| "verification unit" (VU); `vu_*` identifiers, the `verity/sampled-proofs/vu-sample/v1` domain, the `vllm-vu-set/v1` kind, the `vu-k1536` tier | proof unit |
| "M0"; its first statement id, baked into the Lean verifier's `Flock/Tags.lean` | C-Flock's circuit prover: module `verity_flock.circuit`, CLI `flock-circuit`, statement `verity/flock-circuit` |
| "M1", "M2" (elsewhere in the code these also name unrelated things) | C-Flock ZK |
| `hm96-sha256/v1`, until one-hash deletes it | the hiding leaf `hm96-sha512/v1` |
| "coin server"; the `round_digest` strings of statement identities in `Flock/Tags.lean` | the challenger's live-coin part |
| "network transparency"; the `network-transparency/` paths in network accounting's project store that the `NetworkCertifier` Lean cites | network accounting |
| "warden", "network warden"; the domain tags `verity-network-warden/row/v1` and `verity-network-warden/record/v1`, the store label value `replay_set=warden/re-verify` | the network certifier; "the warden's records" are network certificates |
| "timed verifier"; the recorded config key `verifier_cpu` (`--verifier-cpu`) | the memory challenger |
| "prover gateway", "gateway"; one recorded statement string of recursive sessions | the firewall |
| "verifier gateway" | the challenger |
| "batch verifier" (outside `archive/`, where B-Ligero's is another thing) | the verifier |
| "GPU workers" | workers |
| "netlist"; C-Flock's `verity/flock-netlist/…` tags and domains and the `flock-netlist/v1` file format of its older statements (`netlist-vectors.patch`) | a private circuit, or the circuit's description |
| "trusted text", in the `layers` reasons of `lean-audit.json` and the module docs of the `Protocol.lean` files | spec |

The `veritor.*` gate families are grandfathered and the prefix is frozen (circuits, 14 Oct 2026, CIR-12): each existing id
declared in `verity/ml/circuits/targets/target.py` (`veritor.tc-ampere-bf16@1`, the `veritor.tc-bf16.*@1` and
`veritor.tensor-core.*@2` families) keeps its exact spelling, because the `*_FAMILY_DIGEST` pins, fixtures and recorded runs
bake it in, and renaming would invalidate every one of them for no semantic gain. No new family is created under
`veritor.*`: a new family takes a fresh id, and `verity/ml/circuits/tests/test_targets_frozen_families.py` fails on any
`veritor.*` id outside the grandfathered set.

Some words keep a different, narrower sense:

- `verity.ml.circuits`'s `Circuit` is a Program's canonical gate layout and a `GateSelection` is a selected set of its gates; its query `Family` is an indexed sequence of gate sets; a primitive's `family` is the id stem of lazily materialised primitives such as `Const<w>`. The `subcircuits` of a `typed-obligation/v0` object (`verity_sp1.proofs.typed_obligation.Subcircuit`) are its shared gate bodies, and a sampled-proofs obligation's JSON `subcircuit` key holds its gate selection; both names are serialized and stay. `verity.ml.circuits`'s gates are word-level: each outputs one `Value<w>` of w bits, not one bit, and so are the gates of the Lean model in `Security/Definitions/Core/` (`Verity.Circuit`), whose Boolean profile is width 1.
- A **template instance** (the queries `Q_template_instance` and `Q_template_instances`) is one occurrence of a subcircuit template in a program, and in a claim id `property/instance`, such as `cr/sha-512`, the instance is the primitive an assumption is about; neither is an input.
- A **universal unit** (`verity.ml.private_circuits.universal.UniversalUnit`) is a public Boolean Definition, one per size class, that
  runs any **private circuit** of the class, whose description (its gates and wiring) is given as a value; in private circuits it
  is the circuit of one proof unit, so the developer's circuit is hidden like any value.
- A **transition unit** is one tensor-core accumulator step; a GEMM-coordinate subcircuit chains `K/16` of them, and a "unit circuit" arithmetises one.
- "Candidate C" and "Candidate V" in older notes are the QuickSilver VOLE research backend in `archive/vole`, not C-Flock.
- In `integrations/vllm`, "family" and "kind" name circuit Definitions and query families, many of them serialized fields (see its README's vocabulary).
