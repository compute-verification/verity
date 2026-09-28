# Verity

Verity is a toolkit for representing computations precisely, verifying claims about them, and deriving compute, memory, and network resource guarantees.

**Verity IR is the common representation boundary.** Frontends translate other computation representations into Verity IR. Verification specifies what claims should be checked and what values must be bound. Commitments bind those values so they can later be authenticated. Backends prove or check the resulting obligations. Integrations compose these pieces around real systems such as vLLM.

Verity is under active development. The architecture below describes the intended component boundaries; it is not a claim that every frontend, backend, or integration is already implemented.

## Architecture

~~~text
External computation representations
   Torch          JAX          ONNX
      \             |            /
               Frontends
                   |
                   v
        +----------------------+
        |      Verity core     |
        |                      |
        |  IR                  |  What computation?
        |  ML semantics        |  What do exact ML/hardware operations mean?
        |  Proofs              |  What must be established?
        |  Commitments         |  What values are bound and authenticated?
        |  Resource accounting |  What resource claims follow?
        +-----------+----------+
                    |
              verification
               obligations
                    |
          +---------+---------+---------+
          |         |         |         |
       C-Flock    A-GKR   B-Ligero    D-SP1
                     Backends
~~~

Runtime integrations sit around this stack:

~~~text
                        Verity core
                     /      |       \
                    /       |        \
           frontend   commitments   backend
                    \       |        /
                     \      |       /
                      vLLM integration
                             |
                           vLLM
~~~

An integration may compose several Verity components to provide an out-of-the-box experience. That does not merge their implementation responsibilities.

## Core

### IR

`verity.ir` defines computations.

It owns typed values, value identities, exact functions, explicit dataflow, calls and other compact program structure, static bindings, and Programs.

The IR describes logical computation, not a physical GPU schedule, proof system, commitment layout, or verification sampling policy.

### ML semantics

`verity.ml` contains reusable exact definitions of ML and accelerator operations expressed using the IR.

Examples include exact floating-point operations, dot products, tensor-core arithmetic, normalization, attention components, and other numerical relations.

Hardware-specific semantics belong here when they are reusable facts about the computation itself. A backend should not invent a second definition of an H100 dot product, and a vLLM integration should not own its numerical meaning.

### Proofs

`verity.proofs` defines **backend-independent single statements over Verity programs**: the concrete relations and authenticated values a backend must establish, and how a backend's proof of them is judged.

Which parts of a Program are checked, how checks are selected or sampled, and which values must be bound before a challenge are protocol questions. They belong to the protocols under `protocols/`: `one_stage`, the one-stage audit, whose verifier draws the proof units to prove from its own randomness, and `sampled_proofs`, a two-stage law that draws its challenges through `verity.randomness`. Each summarises an accepted audit as a `verity.proofs` `IntegrityProfile`: what verification established, which resource accounting, policies and other protocols read.

Proofs are separate from computation semantics: changing which proof units are sampled or how they are grouped for checking does not redefine the Program.

Proofs are also separate from proof implementation: the same statement may be established by C-Flock, A-GKR, B-Ligero, D-SP1, or another backend.

### Commitments

`verity.commitments` binds logical Verity values and later authenticates openings of those values.

It owns application-level commitment layouts, roots, openings, multiproofs, and the mapping from Verity value identities or components to committed material.

Commitments establish that the values supplied to a check are the values that were previously bound. They do not by themselves establish the computational relation between those values.

For example, a proof of a dot product may need to establish both:

~~~text
x, w, and y are the previously committed values
~~~

and:

~~~text
y = ExactDot(x, w)
~~~

The first is commitment/authentication; the second is a verification obligation established by a backend.

Proof-system-internal commitments are different. A STARK's internal polynomial or FRI commitments belong inside that backend, not in `verity.commitments`.

### Resource accounting

`verity.resource_accounting` derives compute, memory, and network claims from Verity computations, verification guarantees, and the physical observations required by each protocol.

Resource accounting consumes the shared representation; it does not redefine numerical semantics or proof-backend behavior.

## Frontends

A frontend translates an external computation representation into Verity IR.

Expected examples include:

~~~text
frontends/
  torch/
  jax/
  onnx/
~~~

A Torch frontend, for example, accepts specified Torch graph representations and translates supported computations into ordinary Verity Programs and bindings.

A frontend should not silently guess exact numerical behavior that its source representation does not determine. Where necessary, the exact numeric (silicon) semantics must be bound explicitly.

Frontends stop at the common Verity representation. They do not implement backend-specific proving logic.

## Backends

A backend establishes Verity verification obligations. Each belongs to a backend family (see the Glossary):

~~~text
backends/
  flock/                 C-Flock, in active development; its Lean verifier is under verifier/
  gkr/                   A-GKR, frozen
  direct/                B-Ligero, frozen (verifiers: backends/ligero-verify/, backends/ligerito-verify/)
  sp1/                   D-SP1, frozen
~~~

Frozen backends stay in the repository and in the benchmark tables but are not developed.

A backend consumes backend-independent Verity objects: the selected computation or relation, its boundary values, the relevant bindings, and the authentication evidence required by the statement.

A backend may implement highly specialized proving algorithms for registered Verity definitions and may exploit generic structure such as shared operands, contiguous coordinates, or common authentication paths.

It must not require knowledge of which frontend or runtime produced the obligation.

In particular:

~~~text
Torch frontend  ─┐
JAX frontend    ─┼─> Verity representation ─> C-Flock
ONNX frontend   ─┘                         └─> D-SP1
~~~

Adding a new frontend should not require a new C-Flock implementation. Adding a new backend should not require understanding Torch or vLLM.

## Integrations

An integration makes Verity work end-to-end with a real system or runtime.

The initial integration is vLLM:

~~~text
integrations/
  vllm/
~~~

The vLLM integration may:

- construct or obtain the relevant Verity Program;
- use a frontend internally;
- supply vLLM- and silicon-specific static bindings;
- map a real vLLM execution onto Verity values;
- acquire and commit runtime values;
- choose vLLM-specific verification/query policy;
- invoke a backend;
- expose a user-facing wrapper around vLLM.

Users should eventually be able to use the vLLM integration without manually assembling these components.

An integration may ship with or install a default backend. Deployment composition does not create a source dependency from that backend back onto the integration.

## Queries and verification policy

Keep generic query and verification machinery in the core.

Place higher-level query recipes according to what they actually depend on:

~~~text
generic selection over Programs
    -> verification

selection based on reusable ML definitions
    -> shared ML/verification code

vLLM request/runtime-specific policy
    -> vLLM integration
~~~

Whatever its origin, the result presented to a backend should be expressed in backend-independent Verity terms.

Do not make a backend branch on `frontend="vllm"`, model names, Python module paths, or other source-system labels when the actual requirement can be represented as a general property of the obligation.

## Repository map

Create components as real code requires them; this map is not a request to populate empty directories.

~~~text
packages/
  verity/                Core Python library
    verity.ir
    verity.evaluation
    verity.ml
    verity.proofs
    verity.commitments
    verity.randomness
    verity.claims
    verity.resource_accounting

protocols/               Runtime protocols over the core (one_stage, sampled_proofs); each imports only the core
frontends/               External computation representations -> Verity IR
backends/                Verity obligations -> proofs/checks
integrations/            End-to-end integrations with real systems

benchmarks/              Reproducible workloads and performance measurements
census/                  Data only: hardware, datatypes, networks, subcircuits, input sets, workloads

tools/
  research/              Shared experiment execution, telemetry, and artifacts
  check/                 The one check a merge requires
  circuit_check/         The property checks every circuit passes
  lean/                  The Lean audit
~~~

## Python packaging

The core is initially one ordinary Python distribution:

~~~text
distribution: verity
~~~

with explicit imports such as:

~~~python
from verity.ir import Program
from verity.ml import ...
from verity.proofs import ...
from verity.commitments import ...
from verity.resource_accounting import ...
~~~

Do not build a large convenience API at `from verity import ...`; import objects from the module that owns them.

Core architectural modules are not separate distributions merely because they are separate modules.

Heavy frontends, backends, and integrations may become independently installable distributions when their dependency or deployment environments require it. For example:

~~~text
verity-flock
verity-sp1
verity-vllm
~~~

That decision should be driven by real installation boundaries rather than directory aesthetics.

`research` is separate from the Verity library:

~~~text
distribution / CLI: research
~~~

It is development infrastructure and must not import Verity packages.

## Development

Keep dependencies directional.

At the core:

~~~text
ir
^
|
evaluation
^
|
ml

ir
^
|
commitments

ir + ml + commitments
       ^
       |
proofs

ir + proofs + observations
       ^
       |
resource_accounting
~~~

This diagram communicates ownership, not a requirement that every possible dependency be used. `randomness` and `claims` import no other core module. `packages/verity/tests/test_boundaries.py` enforces these rules.

Higher-level components depend inward:

~~~text
frontends     -> Verity core
backends      -> Verity core
integrations  -> Verity core + selected frontend/backend interfaces
research      -> no Verity library packages
~~~

Backends must be testable without installing a frontend or runtime integration.

Frontends must be testable without installing a proof backend.

Package tests live with the code they exercise. Cross-component tests cover real end-to-end paths.

## First vertical slice

The first backend work used an existing exact dot-product relation, the existing Verity authentication scheme, and SP1 (D-SP1, `backends/sp1`; its baseline is `benchmarks/dot_product/baselines.json`).

Its goal was:

~~~text
exact reusable Verity dot-product semantics
        +
backend-independent verification obligation
        +
authenticated boundary values
        |
        v
      SP1
        |
        v
verified proof
~~~

This path works without vLLM installed.

Optimize proving only while preserving the declared relation and authentication semantics.

Keep relation-only performance measurements distinct from authenticated proof measurements.

A proof is accepted only when it is verified against the intended statement and approved proof-program/backend identity; "the proof verifier accepted some self-consistent statement" is insufficient.

The proof campaign should help reveal the smallest useful backend interface. Do not design a universal frontend/backend framework ahead of the backends that need it.

## Research tooling

Use `research` for substantial recorded experiments, remote execution, telemetry, comparison, preservation, and coordination.

`research` treats the workload as opaque. Workload-specific code supplies its own fingerprint, validation result, measurements, and artifacts through the language-neutral run-result contract.

The proof benchmark decides what makes two proof runs semantically comparable. `research` records and exposes differences; it does not understand SP1, Verity IR, or proof semantics.

Architectural explanation, consequential design decisions, and research interpretation live in Notion. Raw experiment evidence and telemetry live in artifact storage. The repository contains executable code and concise current operational guidance.

## Glossary

Code, docs, CLI help and output fields use these terms. When a term changes, this section changes with it, and the new term is rolled out in the same change.

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
- **Integrity profile**: what an accepted audit established (`verity.proofs.profile.IntegrityProfile`): the facts checked exhaustively and the verifier's draw law over named levels of a partition hierarchy, with its failure probability δ. From it a consumer bounds how many proof units of the partition can be incorrect, given serving's committed values.
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
| "M0"; the statement `verity/flock-netlist/v1`, baked into the Lean verifier's `Flock/Tags.lean` | C-Flock's circuit prover: module `verity_flock.circuit` (lands with PR #192), CLI `flock-circuit`, statement `verity/flock-circuit` |
| "M1", "M2" (elsewhere in the code these also name unrelated things) | C-Flock ZK |
| "coin server"; the `round_digest` strings of statement identities in `Flock/Tags.lean` | the verifier |

Some words keep a different, narrower sense:

- `verity.ir`'s `Circuit` is a Program's canonical gate layout and a `GateSelection` is a selected set of its gates; its query `Family` is an indexed sequence of gate sets; a primitive's `family` is the id stem of lazily materialised primitives such as `Const<w>`. The `subcircuits` of a `typed-obligation/v0` object (`verity.proofs.typed_obligation.Subcircuit`) are its shared gate bodies, and a sampled-proofs obligation's JSON `subcircuit` key holds its gate selection; both names are serialized and stay. `verity.ir`'s gates are word-level: each outputs one `Value<w>` of w bits, not one bit.
- A **template instance** (the queries `Q_template_instance` and `Q_template_instances`) is one occurrence of a subcircuit template in a program, and in a claim id `property/instance`, such as `cr/sha-512`, the instance is the primitive an assumption is about; neither is an input.
- A **transition unit** is one tensor-core accumulator step; a GEMM-coordinate subcircuit chains `K/16` of them, and a "unit circuit" arithmetises one.
- "Candidate C" and "Candidate V" in older notes are the QuickSilver VOLE research backend in `backends/vole`, not C-Flock.
- In `integrations/vllm`, "family" and "kind" name IR definitions and query families, many of them serialized fields (see its README's vocabulary).
