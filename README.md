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
          +---------+---------+
          |         |         |
         SP1      STARK     direct/replay
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

Which parts of a Program are checked, how checks are selected or sampled, and which values must be bound before a challenge are protocol questions. They belong to `protocols/sampled_proofs`, which draws its challenges through `verity.randomness` and summarises an accepted audit as a `verity.proofs` `IntegrityProfile`: what verification established, which resource accounting, policies and other protocols read.

Proofs are separate from computation semantics: changing which verification units are sampled or how they are grouped for checking does not redefine the Program.

Proofs are also separate from proof implementation: the same statement may be established by SP1, a custom STARK, direct evaluation, or another backend.

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

Frontends stop at the common Verity representation. They do not implement SP1-specific proving logic.

## Backends

A backend establishes Verity verification obligations.

Expected examples include:

~~~text
backends/
  sp1/
  stark/
  replay/
~~~

A backend consumes backend-independent Verity objects: the selected computation or relation, its boundary values, the relevant bindings, and the authentication evidence required by the statement.

A backend may implement highly specialized proving algorithms for registered Verity definitions and may exploit generic structure such as shared operands, contiguous coordinates, or common authentication paths.

It must not require knowledge of which frontend or runtime produced the obligation.

In particular:

~~~text
Torch frontend  ─┐
JAX frontend    ─┼─> Verity representation ─> SP1 backend
ONNX frontend   ─┘                         └─> STARK backend
~~~

Adding a new frontend should not require a new SP1 implementation. Adding a new backend should not require understanding Torch or vLLM.

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
    verity.ml
    verity.proofs
    verity.commitments
    verity.randomness
    verity.resource_accounting

protocols/               Runtime protocols over the core (sampled_proofs); each imports only the core
frontends/               External computation representations -> Verity IR
backends/                Verity obligations -> proofs/checks
integrations/            End-to-end integrations with real systems

benchmarks/              Reproducible workloads and performance measurements

tools/
  research/              Shared experiment execution, telemetry, and artifacts
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
verity-sp1
verity-vllm
verity-torch
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
ml

ir
^
|
commitments

ir + commitments
       ^
       |
proofs

ir + proofs + observations
       ^
       |
resource_accounting
~~~

This diagram communicates ownership, not a requirement that every possible dependency be used.

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

The first backend work uses an existing exact dot-product relation, the existing Verity authentication scheme, and SP1.

The initial goal is:

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

This path must work without vLLM installed.

First reproduce the existing known-correct SP1 baseline. Then optimize proving while preserving the declared relation and authentication semantics.

Keep relation-only performance measurements distinct from authenticated proof measurements.

A proof is accepted only when it is verified against the intended statement and approved proof-program/backend identity; "the proof verifier accepted some self-consistent statement" is insufficient.

The proof campaign should help reveal the smallest useful backend interface. Do not design a universal frontend/backend framework before this vertical slice works.

## Research tooling

Use `research` for substantial recorded experiments, remote execution, telemetry, comparison, preservation, and coordination.

`research` treats the workload as opaque. Workload-specific code supplies its own fingerprint, validation result, measurements, and artifacts through the language-neutral run-result contract.

The proof benchmark decides what makes two proof runs semantically comparable. `research` records and exposes differences; it does not understand SP1, Verity IR, or proof semantics.

Architectural explanation, consequential design decisions, and research interpretation live in Notion. Raw experiment evidence and telemetry live in artifact storage. The repository contains executable code and concise current operational guidance.

## Glossary

Code, docs, CLI help and output fields use these terms. When a term changes, this section changes with it, and the new term is rolled out in the same change.

- **Subcircuit template**: a named template with typed parameters, such as `gemm_coordinate(K, semantics)` or `rmsnorm(d, semantics)`. Shape and numeric (silicon) semantics are both parameters; a semantics value is an instruction id of `verity.ml.tc.instructions.INSTRUCTIONS`.
- **Subcircuit**: a subcircuit template with every parameter bound, such as `gemm_coordinate(K=1536, semantics=sm90.wgmma.m64n8k32.e4m3)`. A backend proves subcircuits, and a benchmark cell measures one.
- **Backend family**: A-GKR, B-Ligero, C-Flock or D-SP1. The letter is part of the name.
- **Backend**: an implementation of the Backend interface, that is, one concrete, configured member of a family: A-interactive, A-fs, B-interactive, B-fs, C-interactive or D-fs, where fs means Fiat–Shamir.
- **Replay unit (RU)** and **verification unit (VU)**: roles that sampled proofs assigns to parts of a Program (`protocols/sampled_proofs/PROTOCOL.md`). "Unit" never names a subcircuit.

Older names stay where they are baked into a digest, a frozen fixture, a recorded artifact field or a pinned vector. They mean:

| Old name, where it stays | Term |
|---|---|
| "Candidate A", "Candidate B"; the `candidate` feature and label keys, `CandidateA`, `CandidateB`, `account_candidate_a` | backend family A-GKR, B-Ligero |
| the `candidate` label values `SP1` and `C` (`research.store.vocab.CANDIDATES`) | D-SP1, and the VOLE candidate (not C-Flock) |
| `verity.proofs.target.Target`, `TARGETS`, `FIRST`; the result fingerprint's `profile` | subcircuit (a `GemmCoordinate<K>` at one semantics) |
| `Target.family`, `FAMILY_DIGEST`, a gate family such as `veritor.tc-ampere-bf16@1` | the semantics parameter's gate set |
| "configured backend"; `drilldown.Variant`, `VARIANTS`; `views.Configuration`, `CONFIGURATIONS` | backend |

Some words keep a different, narrower sense:

- `verity.ir`'s `Circuit` and `Subcircuit` are a Program's canonical gate layout and a selected set of its gates; its query `Family` is an indexed sequence of gate sets; a primitive's `family` is the id stem of lazily materialised primitives such as `Const<w>`. The `subcircuits` of a `typed-obligation/v0` object are its shared gate bodies (digest-bearing).
- A **circuit** on its own is a backend's arithmetisation of a subcircuit, as in A-GKR's pinned circuits or "SHA-256 in circuit".
- A **transition unit** is one tensor-core accumulator step; a GEMM-coordinate subcircuit chains `K/16` of them, and a "unit circuit" arithmetises one.
- An **instance** is one input to a subcircuit, as in instance sets and "4,096 instances".
- "Candidate C" and "Candidate V" in older notes are the QuickSilver VOLE research backend in `backends/vole`, not C-Flock.
- In `integrations/vllm`, "family" and "kind" name IR definitions and query families, many of them serialized fields (see its README's vocabulary).
