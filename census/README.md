# Census

Data only: the fixed quantities benchmarks are computed against. It will grow into a knowledge base of hardware, ML
models and workloads. The renderer and benchmark code read these files directly, and so does the docs website. Nothing
imports them as code, so the directory can move to its own repository unchanged.

| File | Entries | Schema |
| --- | --- | --- |
| `hardware.json` | GPU lines: a device (full `device` and short `name`, e.g. "H100 SXM5") running one operand dtype, with its dense tensor-core peak at FP32 accumulate | `schemas/hardware.schema.json` |
| `datatypes.json` | operand formats, displayed as the common name first, then the precise format: `BF16`, `FP8 (E4M3)`, `FP4 (NVFP4, E2M1)` | `schemas/datatypes.schema.json` |
| `networks.json` | network profiles: the reference (same datacenter, 1 ms, 100 Gb/s) and the sensitivity profiles | `schemas/networks.schema.json` |
| `subcircuits.json` | subcircuit templates, each with typed parameters (e.g. `gemm-coordinate` with `K`, `datatype` and `semantics`, the exact instruction), and under each the subcircuits: fully bound templates, each with an id and a display name, such as `gemm-coordinate/k1536/sm90-wgmma-e4m3` ("GemmCoordinate<1536> · sm90 wgmma · FP8 (E4M3)") | `schemas/subcircuits.schema.json` |
| `input_sets.json` | the frozen input set of each subcircuit (an input is one instance of the subcircuit): dataset, tier, size, digest, `provenance` (captured or synthetic), and the subcircuit its inputs are of | `schemas/input_sets.schema.json` |

Each file is a JSON array of entries. Each entry validates against its file's JSON Schema (draft 2020-12). The tests
in `backends/numerical/tests/bench/test_census_data.py` check that.

## Ids

- Every entry has an `id` that is unique within its file.
- A hardware id is `<device slug>/<dtype>`, for example `a100-sxm4-80gb/bf16` or `rtx-4090/e4m3`. The dtype is named as
  the instruction registry names operand dtypes (`bf16`, `e4m3`, `e2m1`); each is a `datatypes.json` id.
- A subcircuit template id is kebab-case, for example `gemm-coordinate`. A subcircuit (a fully bound template) is
  `<template>/<params>/<semantics>`, for example `gemm-coordinate/k1536/sm90-wgmma-e4m3`: the semantics are the exact
  instruction's architecture, operation and types without its shape, and the full instruction is its `semantics` parameter; `legacy_target` keeps the
  target name it was known by, and `fidelity_notes` record known gaps from what production runs. An input set's id is its dataset, slugged, for example `bench-instances-fp8-ada-v1`. Every
  other id is kebab-case, for example `dc-1ms-100gbps`.
- The census holds workloads, subcircuit templates and subcircuits, input sets, hardware and networks. It never encodes replay or verification
  units, which applications choose per run.
- Ids are stable. Renders cite them, and sites build history over them. A changed quantity gets a new entry with a new
  id, and an id that has been cited is never renamed or reused.

## Input sets and recorded names

"Instance set" is now "input set", and a single instance is an "input". Recorded names keep the old word: the
`instance-equiv/v1` artifact kind, a result's `instances` ref, and the `tier` field (`vu-k1536…`) are what fixtures and
recorded results carry, so they are unchanged.

## Sourcing

- Every entry has `sources`: a non-empty list of `{label, href?}`. The label names the document and the table or row
  the value was read from.
- Hardware lines also state their `methodology`. Peaks are datasheet **dense** figures at the accumulate width the
  relations use (FP32). An FMA counts as two FLOP, and sparsity figures are never used.
- A measured value (rather than a datasheet one) says how it was measured, and names the artifact that holds the
  measurement.

## Dependency direction

- Verity core never reads `census/`. `packages/verity/tests/test_boundaries.py` enforces this.
- Core takes such numbers as parameters: a target names only its `anchor_device`. The benchmark
  (`verity_numerical.bench.census`) reads the peak of that device and dtype from `hardware.json`.
