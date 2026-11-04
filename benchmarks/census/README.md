# Census

Data only: the fixed quantities benchmarks are computed against. It will grow into a knowledge base of hardware, ML
models and workloads. The renderer and benchmark code read these files directly, and so does the docs website. Nothing
imports them as code, so the directory can move to its own repository unchanged.

| File | Entries | Schema |
| --- | --- | --- |
| `hardware.json` | GPU lines: a device (full `device` and short `name`, e.g. "H100 SXM5") running one operand dtype, with its dense tensor-core peak at FP32 accumulate | `schemas/hardware.schema.json` |
| `datatypes.json` | operand formats, displayed as the common name first, then the precise format: `BF16`, `FP8 (E4M3)`, `FP4 (NVFP4, E2M1)` | `schemas/datatypes.schema.json` |
| `networks.json` | network profiles: the reference (same datacenter, 1 ms, 100 Gb/s) and the sensitivity profiles | `schemas/networks.schema.json` |
| `deployments.json` | deployments, whose figure a result gives beside the adversary resource model's (`adversary.json`): the machines of a datacenter and the window each machine is audited in, stated parameters (results per monitoring period) | `schemas/deployments.schema.json` |
| `subcircuits.json` | subcircuit templates, each with typed parameters (e.g. `gemm-coordinate` with `K`, `datatype` and `semantics`, the exact instruction), and under each the subcircuits: fully bound templates, each with an id and a display name, such as `gemm-coordinate/k1536/sm90-wgmma-e4m3` ("GemmCoordinate<1536> · sm90 wgmma · FP8 (E4M3)") | `schemas/subcircuits.schema.json` |
| `workloads.json` | served configurations: the model, the serving config (engine, batch, TP, datatype, hardware class) and the template mix, the proof blocks (`vus`) per subcircuit template and bound parameters, with the source run | `schemas/workloads.schema.json` |
| `input_sets.json` | the frozen input set of each subcircuit (an input is one instance of the subcircuit): dataset, tier, size, digest, `provenance` (captured or synthetic), and the subcircuit its inputs are of | `schemas/input_sets.schema.json` |
| `adversary.json` | the adversary resource model, the compute every result is stated against: the reference is `world-inference-2027`, all inference worldwide over a one-year monitoring period (requests, FLOP a request, about 9.4e30 FP8 FLOP in all), with the H100-equivalent base its source states | `schemas/adversary.schema.json` |

Each file is a JSON array of entries. Each entry validates against its file's JSON Schema (draft 2020-12). The tests
in `benchmarks/numerical/tests/bench/test_census_data.py` check that.

## Ids

- Every entry has an `id` that is unique within its file.
- A hardware id is `<device slug>/<dtype>`, for example `a100-sxm4-80gb/bf16` or `rtx-4090/e4m3`. The dtype is named as
  the instruction registry names operand dtypes (`bf16`, `e4m3`, `e2m1`); each is a `datatypes.json` id.
- A subcircuit template id is kebab-case, for example `gemm-coordinate`. A subcircuit (a fully bound template) is
  `<template>/<params>/<semantics>`, for example `gemm-coordinate/k1536/sm90-wgmma-e4m3`,
  `attention-head/d64-bn128/sm80-fa2-bf16` or `rmsnorm-fused-cuda/n2048-eps1e-05/bf16` (the forms bench-spine's generator writes). The
  params and semantics segments are lowercase and may contain `.`, `_` and `-`. For GEMM, the semantics are the exact
  instruction's architecture, operation and types without its shape, and the full instruction is its `semantics` parameter; `legacy_target` keeps the
  target name it was known by, and `fidelity_notes` record known gaps from what serving runs. An input set's id is its dataset, slugged, for example `bench-instances-fp8-ada-v1`. Every
  other id is kebab-case, for example `dc-1ms-100gbps`.
- The census holds workloads, subcircuit templates and subcircuits, input sets, hardware, networks, deployments and the adversary resource model. It never encodes replay or proof
  units, which applications choose per run.
- Ids are stable. Renders cite them, and sites build history over them. A changed quantity gets a new entry with a new
  id, and an id that has been cited is never renamed or reused.

## Input sets and recorded names

"Instance set" is now "input set", and a single instance is an "input". Recorded names keep the old word: the
`instance-equiv/v1` artifact kind, a result's `instances` ref, and the `tier` field (`vu-k1536…`) are what fixtures and
recorded results carry, so they are unchanged.

## Work models

Each subcircuit template carries a `work_model`: the arithmetic operations per input (one subcircuit instance) as an
integer expression over its bound parameters, for example `2*K` for `gemm-coordinate` and `3*N` plus one rsqrt for
`rmsnorm-triton`. An FMA counts as 2 ops, and special functions are counted apart in `special_per_input`. `variables`
names per-input quantities that are not parameters, such as an attention head's key count `T`. `version` is the one the
tables' native N uses (`work-model/v0`), `source` cites the IR definition or kernel, and `basis` says what is counted.
Weighting a workload's template mix by these counts gives its real work. Counting proof blocks or inputs does not.

A memory-bound template also carries `bytes_per_input`, the bytes read and written per input over its ports, with
`bytes_basis` and `bytes_source`. Together with a hardware line's `memory_bytes_per_second` (cited in `memory_sources`),
this is its memory-bandwidth work model: the renderer's native N for its statements is bandwidth ÷ bytes per input.
`BITS` in an expression is the bit width of the bound `datatype`. A GEMM-like template's bytes count its weight row only
(the activation row is shared by the row's coordinates): the batch-1 decode reading. `variable_bounds` bounds a per-input
variable from the parameters of the proof block it serves in; attention's `T` is `NB*BN` when the unit's last key
block is full, and between `(NB - 1)*BN + 1` and `NB*BN - 1` when it is partial.

A bound subcircuit may record `variable_ranges`: the range a per-input variable takes over the workloads it serves
(attention's `T`). A template marked `commitment_opening` (embedding) is opened against a commitment, not proved by a proof
backend. The per-workload headline lists it as uncovered for that reason and never counts it as covered. `port_datatypes`
records the datatype of each port a bytes model counts. For example, the MoE router reads bf16 logits.

A template the proof backends only partly prove records `verifier_evaluated`: the parts the verifier checks natively, their
share of the template, whether the headline counts a cell as covering the template (with a footnote), and the ruling.
Attention's softmax is counted, per red-team-flock-3's grant. Sampling is not counted; one
renderer switch counts it.

## Workloads

A workload is a served configuration: today the vLLM regression rows that record a replay partition. Its `template_mix`
counts the proof blocks per family (`vu_family`) and bound parameters (`replay_partition.strata`), the complete counts of
the run. This is the mix that weights backend overheads into a per-workload headline (the frozen tables' "Workload headline"). Every family maps to a census
subcircuit template bound from its IR Definition, with the number of the template's subcircuits one proof block holds (`subcircuits_per_vu`):
- `Gemm_v1` and `Gemm_v2` are N `gemm-coordinate`, on Ampere-class mma or Hopper wgmma;
- `Attention_v3`, `Attention_v2` and `Attention_v4` are NH `attention-head` (FA2, or FA3; `Attention_v4` is FA3 under its per-iteration `Check_inf` construction, whose `MASKED_FROM` names the first masked key block);
- `RoPE_v1` is NHEADS `rope-head`;
- the MoE expert GEMMs and the FP8 block-scaled GEMM are N coordinates;
- the other families are one row each.

The captured input sets (vllm-vu-export) sample these counts. They cross-check the mix and are linked to their workload
once registered; they don't replace the counts. An attention entry of a batch-1 workload also records `variable_counts`: the proof blocks at each key count T
(runs of T with units per value). These come from its one request's positions and are checked against the stratum's population
by phase. The headline weights attention cells by them, one T at a time. `source` names the row, its expected file and records artifact, and the commit it was read at. A proof block is a role
that sampled proofs assigns in the source run; the census records the run's counts, not a unit definition.

## Sourcing

- Every entry has `sources`: a non-empty list of `{label, href?}`. The label names the document and the table or row
  the value was read from.
- Hardware lines also state their `methodology`. Peaks are datasheet **dense** figures at the accumulate width the
  relations use (FP32). An FMA counts as two FLOP, and sparsity figures are never used.
- A measured value (rather than a datasheet one) says how it was measured, and names the artifact that holds the
  measurement.

## Dependency direction

- Verity core never reads `benchmarks/census/`. `tests/test_boundaries.py` enforces this.
- Core takes such numbers as parameters: a target names only its `anchor_device`. The benchmark
  (`verity_numerical.bench.census`) reads the peak of that device and dtype from `hardware.json`.
