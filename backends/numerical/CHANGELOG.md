# Changelog: `verity_numerical` (the bench renderers and their rules)

## 2026-09-27: the placement probe finds the pod id in an ssh-started job

- `bench.placement.probe` reads `RUNPOD_POD_ID`, `RUNPOD_PUBLIC_IP` and `RUNPOD_DC_ID` from the container's init process
  (`/proc/1/environ`) when the job's own environment lacks them. A job started over ssh does not inherit the container's
  `RUNPOD_*`, so its `placement.json` had a null pod id, and the shared-NAT re-check at `register` refused every such cell
  (flock-backend, 852816d6). Only those three keys are read. The job's own environment wins, and an unreadable file leaves
  the value unrecorded, as before.

## 2026-09-26: bench.cell accepts a shared datacenter NAT IP, given full machine-identity proof

- `bench.placement.assess` treats a public IP shared by the prover and verifier (a datacenter's egress NAT) as separate machines only
  when every condition holds (coordinator 22:00Z, with red-team-flock-3's conditions S1-S5):
  - the RunPod machine id and `boot_id` are recorded on both pods and differ. The DMI `product_uuid` is compared whenever it is
    read, and a shared one refuses. It may be unreadable only when both pods pass the bare-metal check: RunPod containers
    can't read it, and bare-metal containers share their host's kernel, so the `boot_id` identifies the host (coordinator
    22:40Z, red-team-flock-3 U2). The evidence records an unread uuid as `"unreadable"`;
  - both pods are bare metal (S1): no `hypervisor` CPU flag, and a DMI `sys_vendor` that is not a hypervisor's
    (`HYPERVISOR_VENDORS`); a missing flag or vendor refuses;
  - the link is RunPod's global network (S3): `<verifier pod id>.runpod.internal` resolving into 10/8;
  - a TCP-connect RTT on that route is at least `RTT_FLOOR_MS` = 0.1 ms, with all 30 connects succeeding (S4, R1). The
    co-resident pair measured about 0.03 ms.
    `placement.tcp_rtt` gives the median of 30 kernel-level connects, to the verifier pod's sshd on the same name and address
    (`rtt_target`); it does not use the session port, because the verifiers count every connection on it as a session. The
    session's `net.rtt_ms` and a round trip timed via the public IP do not count.
- **Host bridges and hairpins:** loopback, link-local and the docker pool 172.16/12 are always refused, as is dialling the prover's
  own public IP. A datacenter's routed private network (RunPod's 10.x) is accepted between distinct machine ids.
- **`plan`:** when the API shows a shared IP, it probes both pods over ssh (`placement.remote_probe`; the prover's probe includes
  its link and RTT) and applies the full rule. If the pods cannot be probed, the plan is refused.
- **`register` (S2):** re-checks the rule on the runs' own `placement.json` records and refuses when either run lacks any value
  (pod id, boot id, hypervisor flag, sys_vendor, and for the prover the link and its RTT), or when a run's boot id differs from what
  the plan probed on that pod (a reboot since the plan). A run's uuid is not held to the plan's (U3).
- **Recording (S5):** `cell.placement.shared_public_ip` and, when it is true, `separation`: the three ids of both pods (an unread uuid
  as `"unreadable"`), the shared IP, both pods' bare-metal facts, the link's name and address, the RTT measurement and the floor.

## 2026-09-26: class-pin attention cells credited per listed key count

- **Class cells:** flock-ir-lowering's class-pin statement proves a batch spanning several key counts under one pin per
  class (T 1–128, 129–256, 257–512; red-team-flock-3's grant, condition CP7). Such a cell records `key_counts` {T: heads} and
  `key_class` {T, pin} in its fingerprint, and `per_key_count` {T: heads, e2e_s, ...} on the result. `views.key_class_of`
  reads that shape; before, the one-T matcher rejected these cells X.
- **What is credited:** each T listed in `key_counts` that lies inside the class and has a positive `heads` and `e2e_s` in
  `per_key_count`, at P = heads / e2e_s and native N at that T. Other listed T are shown as `not_credited` with the reason;
  unlisted T are never credited. A class cell with nothing creditable is rejected X.
- **Table 2:** a row per class (`· T = 1–128`). The cell's N is work-weighted over its credited T
  (heads / Σ heads_T / N_T), so N / P is its overhead on its own inputs.
- **Headline:** per T, the fastest of the single-T and class cells covers #101's units at that T, still weighted by the
  template mix; the † footnote stays.
- **Unchanged:** tables, raw JSON, parity and the headline without class cells are byte-identical.

## 2026-09-26: bench.cell requires the verifier on another machine, not another container

- `bench.placement` identifies machines by the RunPod machine id, the public IP, the DMI `product_uuid`, the kernel `boot_id` and the
  host CPU model, rather than by container hostname. This follows red-team-flock's 12:00Z ruling that a co-resident verifier is not
  a separate verifier.
- **`plan`:** it reads both pods' machine id and public IP from the RunPod API (or from `--placement FILE`), and records them in
  `cell.placement`. It refuses a pair that shares a machine id, public IP, product uuid or boot id. It also refuses a pair that talks
  over a host-private address (loopback, link-local, RFC 1918, 100.64/10, ULA) unless both machine ids are known and differ, and a
  pair whose identity isn't enough to tell the machines apart.
- **Jobs:** every job first runs the stdlib-only probe, `placement.py probe`, which writes the container's identity (and, on the
  prover, the route to the verifier) to the run's `placement.json`.
- **`check` and `register`:** they compare both runs' identities (`placement.json`, else `job.json`'s boot id) merged with the plan's.
  They flag a run whose pod differs from the plan's, and stamp `cell.placement` into the registered document. The hostname
  comparison is gone.

## 2026-09-26: attention counted per key count; one verifier-evaluated switch

- **Attention rows per T:** an attention cell's key count T is read wherever the cell records it (`views.input_variables`):
  - the fingerprint's or the cell stamp's `statics` / `variables`;
  - the lowered unit's name (flock-ir-lowering's `.../d64-bn128/t4`);
  - the input set's name;
  - the registered set's parameters.

  Two differing records reject the cell X. A cell gets a Table 2 row per (subcircuit, T), with N at that T
  (864 GB/s ÷ (4D + 4TD)). This replaces 89ba3a65's blanket X for attention.
- **Per-T counts in the census:** batch-1 workloads (#101, #11, #39) give their attention strata `variable_counts`: the VUs
  at each T. With one request, prefill position p attends T = p + 1 and decode step s attends T = input_len + s, once per
  layer. For #101 that is T 1–287 with 16 VUs each, matching every stratum's population by phase. Their native work is
  exact, where it was an interval.
- **Headline coverage per T:** a cell covers only the units at its own T, weighted by the template mix. The headline shows:
  - the covered T;
  - the drawn T (from the registered captured sets of the workload's row);
  - drawn but uncovered T;
  - T values the run does not serve.

  Units at other T are uncovered ("not at every T"), never extrapolated. Without per-T counts (the batch-8 and batch-32
  rows), attention cells are listed as "no per-T counts" and not counted.
- **Verifier-evaluated parts:** templates record them in the census (`verifier_evaluated`: parts, share, counted, ruling).
  - Attention's softmax (4–11% of a head's scalar ops, all of its exp2 and reciprocal work) is counted with a † footnote,
    per red-team-flock-3's grant.
  - Sampling is not counted, pending Daniel.
  - The existing switch (`--preview-count-omitted`, `Rules.headline_count_omitted`) flips every not-counted template and
    every `omitted` label at once.
- **Unchanged:** tables, raw JSON and parity are byte-identical; only the headline section changes.

## 2026-09-26: an A-route-a bench.cell driver; A-GKR's route (a) at K = 2048 and 8192

- **Driver:** `bench/drivers/a_route_a.py` runs route (a) (A-GKR + C-Flock's keyed-BLAKE3 rows + the sigma link) through
  `backends/gkr/cell.sh`. The verifier pod builds its own statement of every sweep size from the shipped input set and serves
  them in order on one port. The prover pod runs the correctness gate, the RTT probe, and per size a warm-up session against a
  loopback verifier followed by the timed live sessions. `tools/cell.py finish` then writes the plateau's result with the sweep,
  protocol, loopback, commit and placement blocks, and `tools/cell.py register` stamps the lane. The driver takes input sets only.
- **Lowering:** A-GKR lowers `gemm-coordinate/k2048` and `k8192` on sm80 BF16 with `frame-v3/blake3-keyed` to
  `bf16-ampere-k<K>+blake3`. These are the K = 1536 circuit files at `steps` = K / 16, pinned in `backends/gkr/verifier/pins.txt`,
  and the lowering has a `stage()`.
- **Unchanged:** the renderers and their rules.

## 2026-09-26: the spine generates every served template

- `bench.templates` binds and generates every template of every census workload. `test_headline.NO_SPINE_GENERATOR` is empty.
  The new templates:
  - **FA3 attention heads:** `attention-head` with semantics `sm90.fa3.bf16`, that is `AttentionHead_v2` on the H100's wgmma step,
    evaluated as `Attention_v2{NH=1, KVH=1}`. The set name and relation are the vLLM exporter's.
  - **Row-shaped templates:** `token-select-greedy` (`TokenSelect_v1`), `moe-router-topk` and `moe-router-topk-norm`
    (`MoeRouterTopK(Norm)_v1`), `moe-sum` (`MoeSum_v1`), `bias-add` (`BiasAdd_v1`) and `bf16-mul-scalar`
    (`Bf16MulScalarTensor_v1`). Their ports are the exporter's `in<g>` / `out.<member>`.
  - **Templates whose Definition takes a table or slab, of which an input reads one row:**
    - `embedding-row` (`Embedding_v1{V, H}`) stores the token id and the row it opens.
    - `moe-expert-coordinate` and `moe-expert-coordinate-weighted` (`MoeExpertCoordinate(W)_v1{E, K}`) store the expert id and its
      row (plus the routing weight for the weighted form).
    - The other rows are +0. Embedding is evaluated one input at a time through its registered row kernel; the experts go through
      `MoeExpertGemm(W)_v1{N=1}`. Tests check both against their coordinate Definitions.
  - **The FP8 block-scaled GEMM:** `scaled-mm-fp8-block-coordinate` (`ScaledMmFp8BlockCoordinate_v1{K, G}`). Its operands are E4M3
    codes and its block scales positive normal f32.
- `gemm-coordinate` generates NVFP4 at `sm120.mma.m16n8k64.e2m1.nvf4`.
  - Ports: `x` and `w` are E2M1 codes (`u8`, one per byte); `sx` and `sw` are UE4M3 scale bytes per block of 16; `y` is the FP32
    accumulator word of `verity.ml.kernels.SM120_NVF4_DOT` (`models.BLACKWELL_SM120_NVF4`).
  - `generate --suite nvfp4` covers K = 1536 at 8,192 and 16,384 VUs.
- `generate --suite workloads` makes one set per census-bound subcircuit of vLLM rows #39, #57, #60, #67, #73 and #74, 45 in all.
  Each attention head uses its census `T` range.
- `bench.lowerings`: a template without a module in a family is `n/s`, and its reason names the file to create. It is no longer a
  registry problem.

## 2026-09-26: every served workload's subcircuits bound

- **Census:** every (template, parameters) in every census workload now has a bound subcircuit. That adds 42 subcircuits
  to #101's:
  - GEMM coordinates at K 1536–14336 on sm80 mma and sm90 wgmma;
  - FA2 and FA3 attention heads, each with its key-count range over the workloads it serves (#101's head now spans #11's
    too: T 1–4607);
  - RoPE at D 128 and 256;
  - RMSNorms (fused and Triton, eps 1e-05 and 1e-06, including Qwen3's N = 128 head norm);
  - SiLU·mul;
  - greedy selection;
  - embedding (a commitment opening);
  - the MoE router, expert coordinates and sum;
  - the FP8 block-scaled GEMM;
  - bias-add and the scalar multiply.
- **Ids and displays:** the bench spine's where it binds the template (`bench.templates.subcircuit`), otherwise its
  `<template>/<parameters>/<semantics or datatype>` form.
- **No spine generator yet:** FA3 attention (`sm90.fa3.bf16`), `token-select-greedy`, `embedding-row`, `moe-router-topk`,
  `moe-expert-coordinate`, `moe-expert-coordinate-weighted`, `moe-sum`, `scaled-mm-fp8-block-coordinate`, `bias-add` and
  `bf16-mul-scalar`. The list is pinned in `test_headline.NO_SPINE_GENERATOR`.
- **Headline:** it names every workload's gaps by subcircuit (`no result` or `commitment opening`); `no census subcircuit`
  no longer occurs.
- **Unchanged:** tables, raw JSON and parity are byte-identical.

## 2026-09-26: #101's attention, sampling and embedding subcircuits; an L40S hardware line

- **Census:** #101's remaining templates are bound, with bench-spine's ids:
  - `attention-head/d64-bn128/sm80-fa2-bf16`, whose key count T is recorded as 1–383 over #101's strata in `variable_ranges`;
  - `gumbel-top-p-token-select/v128256/fp32`;
  - `embedding-row/v128256-h2048/bf16`.
- **Embedding:** the template is marked `commitment_opening`. The ontology has no covered-by-commitment state, so the headline
  lists embedding as uncovered, with the reason `commitment opening`. Its native work stays in the denominator and is never
  imputed.
- **MoE router:** `moe-router-topk` gains `port_datatypes`: bf16 logits, f32 weights, i32 ids. Its bytes model is
  `2*E + 8*TOPK`, from its Definition's ports, so every template of every census workload is now modelled.
- **L40S line:** `views.L40S_BF16` puts the first target's instruction (`sm80.mma.m16n8k16.bf16`) on the L40S. It is a
  hardware-only line, so it has rows only where it has results, each keyed by its explicit subcircuit, with N from the
  `l40s-48gb/bf16` line: 362.05 TFLOPS for GEMM, 864 GB/s for the rest.
- **Rehoming:** under the spec rules, a result whose prover ran on another device than its profile's target moves to the
  line of the same instruction on that device (GEMM) or to the line its prover GPU picks (other templates). Before, it was
  rejected K. Only the L40S changes today, because no other device has a second line of an existing instruction.
  `hardware_target` also finds a line for a subcircuit with no tensor datatype (fp32 sampling), since its N is memory-bound.
- **Unchanged:** the legacy rules, the targets' rows and parity. Existing tables are byte-identical; only the headline
  section changes.

## 2026-09-26: previews of the two pending acceptance decisions (never published)

- `views --preview-one-sided-interaction`: the interaction check passes when the measured wall is at most 1.1 × the model. A
  run under the model no longer fails.
- `views --preview-cached-weights`: non-GEMM native N counts activations only, with per-row weights and tables cached. It
  uses the census's new `work_model.bytes_per_input_cached`: RoPE 4D, fused RMSNorm 8N, Triton RMSNorm 4N, SiLU·mul 6I
  unchanged.
- Both are `Rules` fields, off in `SPEC`. The published render and parity are unchanged. Daniel decides both; the previews
  render beside the published JSON in the notes' `renders/previews/`.

## 2026-09-26: the per-workload headline

- **What:** for each census workload and each backend, the share of the served run's native work that published, cleared
  cells cover, and the work-weighted overhead over that share (`bench.headline`). It renders as the "Workload headline"
  section after Table 3, and as `workload_headline` in the entities JSON. It is absent under the legacy rules. Existing
  tables, the raw views JSON and parity are byte-identical.
- **Units and native work:** units come from the workload's template mix (VUs × subcircuits per VU, per bound subcircuit).
  Native seconds come from each template's census work model on the served hardware line, in two bases, as sp1-evaluator's
  cover summary stated them:
  - tensor-peak GEMM: GEMM-like templates at the dense tensor-core peak, everything else memory-bound;
  - all memory-bound: a GEMM coordinate reads its weight row.
- **Coverage:** a subcircuit counts as covered only when the backend has an admissible result on exactly that subcircuit
  (template and bound parameters, semantics included). The result must be cleared (not provisional, no open finding) and
  its prover must have run on the served hardware. Overhead = the covered units' proving time (units ÷ the highest P) ÷
  their native time.
- **Uncovered and unmodelled:** uncovered subcircuits are listed with the reason: no census subcircuit, no result, results
  only on other hardware, or not cleared. Nothing is imputed. A template with no work model in a basis is listed and left
  out of both sums.
- **Intervals:** attention's key count is known only to its block (`NB`, `last`), so native work, shares and overheads are
  intervals.
- **Census:** L40S memory bandwidth (864 GB/s) is added. Byte models are added for `attention-head`,
  `gumbel-top-p-token-select`, `token-select-greedy`, `embedding-row`, `gemm-coordinate` (`K*BITS//8`), the MoE expert
  coordinates, `moe-sum`, `scaled-mm-fp8-block-coordinate`, `bias-add` and `bf16-mul-scalar`. Attention's
  `variable_bounds` is added, and the entities census now includes `workloads`. `moe-router-topk` has no byte model yet.

## 2026-09-26: non-GEMM statement rows, with native N from a memory-bandwidth work model

- **Statements:** a result whose subcircuit is another template (its `bench.cell` stamp, or the fingerprint's
  `subcircuit`) is its own statement. This covers C-Flock's frame-v3 cells on #101's `rope-head/d64/neox-bf16`,
  `silu-mul/i8192/bf16`, `rmsnorm-fused-cuda/n2048-eps1e-05/bf16` and `rmsnorm-triton/n2048-eps1e-05/bf16` on the H100.
  These rows follow the GEMM rows.
- **Native N:** it comes from the template's memory-bandwidth work model, the line's memory bandwidth divided by
  `work_model.bytes_per_input`:
  - `rope-head` 6D;
  - `silu-mul` 6I;
  - `rmsnorm-fused-cuda` 10N;
  - `rmsnorm-triton` 6N.

  The bytes are counted from each template's ports, with every row read or written once per input. The row's footnote
  states the basis. The census gains `bytes_per_input`, `bytes_basis` and `bytes_source` on those four templates, and
  `memory_bytes_per_second` (3.35 TB/s, from the H100 datasheet) with `memory_sources` on the H100 lines. The four bound
  subcircuits use bench-spine's ids and display names.
- **Admission:** these statements are rejected until the result carries a red-team `proof_class` verdict, in addition to
  a non-producer verification. They are rejected with reason U, not just marked provisional. Their inputs must be a
  registered input set of that subcircuit (reason I).
- **Entities:** the render has the statements, their subcircuits (template and parameters) and their results.
- **Unchanged:** existing renders and parity.

## 2026-09-26: real-K rows: a result at another K (or semantics) is its own statement

- **Statements:** K and the semantics are part of the subcircuit (docs ontology, "A result"). A result whose GEMM
  coordinate is not its target's own K = 1536 subcircuit is its own statement, with its own Table 2 rows after the
  K = 1536 rows, one per scheme with a result. Native N is computed at that K. It is not a new column.
- **Resolving a result's statement:** `views.result_subcircuit` takes:
  - K from the fingerprint;
  - the semantics from the `bench.cell` stamp's subcircuit, else the result's `relation` (`RELATION_SEMANTICS`), else the
    target's instruction.

  Semantics the census records as one (`stood_for_by`) are one statement: H100 BF16 `wgmma` stays on the `mma` row at
  K = 1536, and at real K uses the `wgmma` subcircuit's id (root's ruling, 02:44Z).
- **Census:** `census/subcircuits.json` gains the eight real-K bound subcircuits: `gemm-coordinate/k{2048,8192}/`
  `{sm80-mma-bf16, sm90-wgmma-bf16, sm89-mma-e4m3, sm90-wgmma-e4m3}`.
- **Inputs:** a real-K result's inputs must be a registered `input-set/v1` or `vllm-vu-set/v1` artifact of that K,
  matched by the stamp's art or the set's content digest, over a range inside the set. Otherwise it gets reason I.
  Red-team verdicts count per statement.
- **Entities:** the render keys real-K results, rows and Table 3 by their statement, and lists the real-K subcircuits.
- **Unchanged:** the legacy rules and parity (K != 1536 is still B). Existing rows, cells and the raw JSON are unchanged;
  only the embedded census section gains the new subcircuits. `superseded_by` already rejected a result (above).

## 2026-09-26: the below-bar drill-down

- A result labelled `below_bar=true` (the vocabulary's verification key, added to the store contract with this change)
  is no longer rejected. The entities render lists it under `below_bar` with:
  - its backend family, IR Definition and gate family, B, input set and prover GPU;
  - per proof mode (core, compressed): its own soundness bits, prove seconds, seconds per instance, proof bytes and verify
    seconds;
  - `bar_bits: 128`, and who verified it.

  It is never a Table 2 cell. The first entries are sp1-evaluator's nine SP1 proofs (RMSNorm fused and Triton, RoPE,
  GemmCoordinate at K = 2048 and 8192), at 2^-93.7 to 2^-100, verified by verify-night-3.

## 2026-09-26: a `SUPERSEDED …` finding rejects a result

- Any `superseded_by` label, the vocabulary's key naming the replacement, rejects the result the same way in `tables` and
  the views. `store_tables` keeps its frozen rows and flags them "superseded by …".

- A `finding` label starting with `SUPERSEDED` now rejects the result (reason D), as `PULLED` does. Its producer uses it when
  a re-run replaces a cell, for example flock-backend's four FP8 real-K cells re-run on the spine's input sets. Without the
  rejection, the old and new results would both publish, because their input sets differ.

## 2026-09-26: FP8 GEMM coordinate input sets (fp8-ada, fp8-hopper)

- `gemm-coordinate` generates the two FP8 lines, `sm89.mma.m16n8k32.e4m3` (RTX 4090) and `sm90.wgmma.m64n8k32.e4m3` (H100), at any
  K that is a multiple of 32.
  - Ports: `x` and `w` are E4M3 bytes (`u8`), uniform over the 254 finite codes; `y` is the FP32 accumulator word (`u32`). There is
    no epilogue, as the backends' fp8 relations take it.
  - The relation's `definition` is `GemmAccumulatorE4m3`, and its `dot` names the IR reference.
  - Outputs come from `verity.ml.kernels.HOPPER_E4M3_DOT` and from the new `ADA_E4M3_DOT` (sm_89's step, finite domain, over
    `models.ADA_E4M3_M16N8K32`). Both equal the `verity.ml.tc` chains the backends use.
  - A scalar reference's provenance entry is `templates.reference_digest`: the sha256 of its id, bindings and core sources.
- `generate --suite fp8` covers both lines at K = 2048 and 8192, at the captured #101 sizes (6,272 and 1,920) and at the RTX 4090's
  largest batches (16,384 and 8,192). Each smaller set is the larger one's prefix.
- `generate --suite` writes each set under `<out>/n<n>/sets/<name>/`, so a suite can hold one subcircuit at several sizes.

## 2026-09-26: the family columns are named A-GKR, B-Ligero, C-Flock and D-SP1 (labelled change to the published entities)

- The published entities render names each family column by its Glossary name: `table2.column_labels` maps every family
  to it, and `backend_families[].column` is that name (`SP1` is now `D-SP1`, `Flock` is now `C-Flock`; A-GKR and B-Ligero
  are unchanged).
- The markdown developer render prints the same names in Table 2's header, Table 3's cell labels and Table 1's family rows.
- Column ids stay `SP1` and `Flock`: they key the cells, `--format json`'s `table2.columns` and the parity gate against
  the frozen tables (`tables.py`, whose own headers are unchanged), so `views --parity` is unaffected. The first published
  entities render after this change differs from the previous one only in those two fields.

## 2026-09-26: the markdown tables are retired; backend profiles match main

- The docs site replaces the markdown Tables 1–3 and the table digests for reading (Daniel, 2026-09-26). The steward's
  `[[render]]` entry is now `published = "entities"`: it renders only the entities JSON as `<stamp>-tables.json`, and runs
  `views --parity` as an internal gate that publishes nothing (a difference is a `RENDER-FAILED` line). The markdown
  renderers stay as developer tools.
- Entities backend profiles, following the docs site's gaps note of 20260926T0210Z:
  - the SHA-256 binding the live transcript (and route (a)'s `root_F`) maps to `cr/sha-256`;
  - Flock's Ligerito Merkle trees are a proof-internal hash: BLAKE3 in the CPU prover, SHA-256 in Flock-CUDA
    (`cr/blake3`, `cr/sha-256`);
  - B-Ligero's malicious-verifier ZK also rests on `random-oracle`, because its Merkle leaves are unsalted
    (ligero PROTOCOL.md, red-team-zk-2 G3);
  - an `unknown` ZK text (SP1) maps to `no-zk`: Verity claims no ZK for it;
  - B-fs's `honest-verifier` has the role "zero knowledge only (soundness does not rest on it)";
  - a backend's claims and `implementations` cover only its configurations with a result in the render, when any has one;
    the others are listed as `declared_without_results`, such as SP1's committed guests, which aren't on main yet.
- Poseidon2 row leaves are admissible, flagged as algebraic, and never count toward a standard-hash claim. The render
  already did this; `verity.commitments.rowleaf`'s wording now matches.

## 2026-09-26: the README Glossary's terms in docstrings and help

- Docstrings, comments and `--help` say subcircuit, backend, statement and input set where they said target, candidate,
  configuration and instance set. Output is unchanged: printed tables, entity fields, fingerprint fields (`profile`,
  `instances`) and the frozen tables' headers keep their names, mapped in the Glossary.
## 2026-09-26: census workloads (served configurations and their template mix)

- `census/workloads.json` and its schema hold the served configurations. There are nine vLLM regression rows with a
  replay partition: #11, #39, #57, #60, #67, #68, #73, #74 and #101. Each entry records:
  - the model;
  - the serving config: engine, batch, TP, datatype, hardware class (`hardware` is null for L40S, which the census has no
    line for), input and output lengths, arrivals, sampling and mode;
  - the template mix: the verification units per VU family and bound parameters. `Gemm_v1{K,N}` maps to
    `gemm-coordinate`, with N coordinates per VU;
  - the source run: row, expected file, records and programs artifacts, program digest, reference, and the commit it
    was read at.
- The reader is `census.WORKLOADS`. A test keeps the file equal to the regression expectations (#101: 46,558 VUs, 18,400
  of them `Gemm_v1`).
- Every VU family in the mix maps to a census subcircuit template. `subcircuits.json` adds 15 templates beside
  `gemm-coordinate`, each with typed parameters, a one-line definition and its IR Definitions:
  - `attention-head` (FA2 and FA3), `rope-head`, `rmsnorm-fused-cuda`, `rmsnorm-triton`, `silu-mul` and
    `gumbel-top-p-token-select`, the ids bench-spine's generator keys on (PR #47);
  - `token-select-greedy`, `moe-router-topk`, `moe-expert-coordinate`, `moe-expert-coordinate-weighted`, `moe-sum`,
    `scaled-mm-fp8-block-coordinate`, `bias-add`, `bf16-mul-scalar` and `embedding-row`.

  Hopper `Gemm_v2` binds `gemm-coordinate` with semantics `sm90.wgmma.m64n8k16.bf16`.
- Each template carries a `work_model`: ops per input as an expression over its bound parameters (`gemm-coordinate` 2K;
  `rmsnorm-triton` 3N plus one rsqrt), with special functions apart. Its version, `work-model/v0`, is the one the
  tables' N uses, and it cites its IR definition. A test evaluates every mix's parameters and checks GEMM against N
  (3,072 ops at K = 1536).
- Bound subcircuit ids validate as `<template>/<params>/<semantics>` for every template, not only GEMM (e.g.
  `attention-head/d64-bn128/sm80-fa2-bf16`). `input_sets.json`'s subcircuit field uses the same pattern.
- `hardware.json` adds the L40S 48GB lines from NVIDIA's datasheet, with dense figures: BF16 362.05 TFLOPS and FP8 733
  TFLOPS. The datasheet does not state the accumulate width; its methodology records full-rate FP32 accumulate as an
  assumption, to be confirmed by a measurement. The L40S workloads carry `l40s-48gb/bf16`. `tools/native_peak` quotes the
  same figures.
- Nothing renders workloads yet. The per-workload headline (backend overheads weighted by the mix) is a later change.

## 2026-09-26: file-verified interactive results pay their protocol's rounds

- Bug: `views.interaction` gave every file-verified result `rounds = 0`, so all 14 B-interactive cells (B-Ligero, a 3-round
  interactive protocol, verified from its replayed transcript) were modelled as non-interactive. The reference network
  undercharged them and RTT sweeps left them flat.
- Root's ruling (2026-09-26): a file-verified result of an interactive protocol (its configuration is not transferable) uses the
  protocol's sequential depth as its rounds: the per-proof depth (`rounds.sequential_depth`, else `rounds.protocol`) times the
  sub-batch proofs that run one after another. With `software.backend.pipeline` proofs in flight, that is
  ceil(`N_subbatches` / pipeline) waves; without a pipeline, all `N_subbatches`. The measured RTT stays null. Fiat–Shamir
  results (A-GKR, SP1, B-Ligero FS) keep 0.
- The interaction record, and the entities render's `interaction`, gain `rounds_source`: `measured`,
  `derived: file-verified` or `fiat-shamir`. Markdown shows `<rounds> (derived)`, and the Table 3 interaction note now
  counts rounds × 1 ms.
- Effect at the reference network: the 14 B-interactive results rise by 0.9–5.5% (12 to 147 rounds each). The legacy
  parity is unchanged.
- Entities: `throughput.instances` is now `throughput.inputs`, following `interaction.inputs`.
- Entities: `interaction.measured_rtt.method` is now an enum (`views.RTT_METHODS`: `tcp-connect`,
  `open-connection-ping`, `unclassified`). The recorded `rtt_method` text names hosts and ports, so it stays in the result's
  own record in the evidence store. The markdown still prints the recorded text and is byte-identical.

## 2026-09-26: "instance set" becomes "input set"

- `census/instance_sets.json` and its schema are now `census/input_sets.json` and `schemas/input_sets.schema.json`. The
  reader's `census.INSTANCE_SETS` is now `census.INPUT_SETS`.
- In the entities render:
  - results, their `key` and subcircuits carry `input_set` (and `input_set_provenance`) instead of `instance_set`;
  - a result's `interaction.instances` (the inputs per timed run) is now `interaction.inputs`;
  - the embedded census section is now `input_sets`;
  - result ids are unchanged in shape: `<backend>@<statement>@<input set>@<prover hardware>`.
- Recorded names keep the old word, because fixtures and results carry them: the `instance-equiv/v1` artifact kind, a
  result's `instances` ref, and the `tier` field. Printed markdown strings such as reason I, "another instance set", are
  also unchanged, so renders stay byte-identical.

## 2026-09-26: the benchmark spine (input sets from the IR, one cell entry point, per-template lowerings)

- `bench.templates`: the subcircuit templates of row #101, each a registered IR Definition with typed parameters:
  `gemm-coordinate` (`GemmCoordinate_v2{K, DOT}`, any K, Ampere mma and Hopper wgmma BF16; the census's other semantics
  bind but cannot be generated), `attention-head` (FA2 `AttentionHead_v3`, T per instance), `rope-head`,
  `rmsnorm-fused-cuda`, `rmsnorm-triton`, `silu-mul` and `gumbel-top-p-token-select` (the vLLM registry's Definitions,
  imported lazily). A subcircuit carries the relation, set name and ports the vLLM VU exporter (PR #42) writes, and
  `identify(relation)` maps any set back to its subcircuit.
- `bench.input_sets`: one layout for captured and synthetic sets (PR #42's `vllm-vu-set/v1`; synthetic sets are
  `input-set/v1` with the same fields), `verify` re-evaluates every instance through `verity.evaluation`. All 11,040
  instances of the eight captured #101 sets (`art:b5bb0ca9`) re-evaluate bit-exactly.
- `bench.generate`: seeded synthetic sets. Draw `j` reads each port from `verity.randomness` counter streams at
  `(port, j)`; outputs come from the IR evaluator; instance `i` is the `i`-th draw in the subcircuit's domain, so sets are
  prefix-stable. `--register` stores a set as an `input-set/v1` artifact.
- `bench.cell plan | run | check | register`: one entry point per (backend, statement, input set). It resolves the
  family's lowering, plans the backend driver's `research run`s (`bench.drivers`, one module per backend; today
  `c_interactive`, wrapping `backends/flock/pod/30-cell.sh` and `verity_flock.register`), checks a result against the
  protocol (contract, warm/runs/uncontended/plateau, interaction record, verifier on another host, RTT and loopback
  probes, statement and instance set) and registers it with the cell stamped and the running lane as its producer.
- `bench.lowerings`: each family (A-GKR, B-Ligero, C-Flock, D-SP1) keeps one module per template under `templates/`
  (`supports(subcircuit, scheme)` with the reason, `relation`, optional `stage`); the reason is the cell's `n/s`.

## 2026-09-25: `views --format entities`, the docs site's render

- `python -m verity_numerical.bench.views --root <store> --format entities [--published]` prints the render as entities
  (schema `verity/tables-entities/v1`; the site's spec is the ontology's "A result" and "Presentation rules"):
  - `subcircuits`: one per subcircuit, a fully bound subcircuit template. Each has its census id and display name (e.g.
    `gemm-coordinate/k1536/sm90-wgmma-e4m3`, "GemmCoordinate<1536> · sm90 wgmma · FP8 (E4M3)"), its
    `subcircuit_template` with its bound `parameters` (`K`, `datatype`, and `semantics`: the exact instruction),
    `legacy_target`, `fidelity_notes`, the instance set with its provenance (captured or synthetic), the census hardware
    line, the datatype display and native N;
  - `backend_families`: A-GKR, B-Ligero, C-Flock and D-SP1, each with its markdown column header as `column`. The
    markdown itself is unchanged;
  - `backends`: one per backend family and interaction mode (`A-fs`, `B-interactive`, `B-fs`, `C-interactive`, `D-fs`),
    plus composed backends under their own id. Route (a) is `A-route-a`: it renders in the A-GKR column, and its profile
    adds C-Flock's assumptions and the sigma link's. Each backend's claims are written out in full;
  - `statements`: one per subcircuit and commitment scheme, `<subcircuit>+<scheme>`: the full committed relation a backend
    lowers and proves, with its scheme's assumptions. The renderer's configurations are an implementation detail
    (`implementations` names them) and are not emitted. A backend's profile leaves out statement-commitment hashes, which
    belong to the statement;
  - `schemes`: one per scheme, under ids that name the leaf hash (e.g. `frame-v3/blake3-keyed`; `legacy_id` is the
    renderer's). Each has its tree and leaf hashes, a structured `algebraic` flag, its assumptions (taken from the code)
    and the `leaf_circuits` that commit it. `blake3-xob` commits keyed BLAKE3's bytes (same schema and params in
    `ligero-verify`), so it is a circuit of `frame-v3/blake3-keyed`, not a scheme of its own;
  - `results`: the best result of every admissible backend, not only each family's pick (`best` marks that pick), keyed by
    backend × statement × instance set × prover hardware. Each result carries:
    - its `backend`, its `statement` and the `key` of those four;
    - its full assumption set: the backend's plus the statement's;
    - `hardware` (the semantics chip, which supplies N) and `prover_hardware` (where the prover ran, which supplies P);
    - the instance set with its provenance;
    - its soundness: the whole-cell bound, a union over the sub-batch proofs, plus the backend's per-proof bound where the
      result records it;
    - `N` (the native peak: a work-model version plus a census peak id) and `P` (instances per second at the reference
      network) beside the overhead;
    - its `interaction` record, enough to recompute the overhead at any network (`formula`):
      - `rounds`, `rounds_protocol` and `sequential_depth`;
      - `bytes_up` and `bytes_down`;
      - `commit_s`, `prover_compute_s` and `verifier_compute_s`;
      - `measured_rtt` and `loopback_per_round`, each with its method;
    - its phases: stages, fused stages as one span, "Other" clamped to 0 in [−0.5%, 0) and flagged when it is more
      negative, with the raw value kept as `other_raw`, and one-time setup in its own field;
    - its state, artifact and render time;
  - `best_standard_hash`: for each backend family and subcircuit, the lowest-overhead cell on standard hashes only;
  - `tables`: Table 1 lists backends (guarantees, assumptions, unmapped texts). Table 2 and 3 carry every printed cell's
    exact display string, value, marks and footnote; their rows are keyed by statement and prover hardware, and each cell
    names its backend;
  - `claims`: the `verity.claims` entries cited; `census`: the `census/` entries as read at render time;
  - `provenance`: the render time and the verity commit.
- Ids are stable across renders: subcircuits by census id, backends as `<letter>-<mode>` or a composed backend's own id,
  statements as `<subcircuit>+<scheme>`, and results as `<backend>@<statement>@<instance set>@<prover hardware>`.
- The markdown is unchanged byte for byte. Table 2's cells and footnotes and Table 3's rows now come from shared helpers
  (`table2_grid`, `table3_row`) that both renders use.
- The steward's `published = "views"` entry also writes `<stamp>-tables.json` (the entities render).

## 2026-09-25: peaks, networks and instance sets come from the census (`census/`, data only)

- The renderers read these values from `census/` at the repository root:
  - the native peaks;
  - the reference network (1 ms, 100 Gb/s) and the interaction drill-down's sensitivity profiles;
  - K and B;
  - the frozen instance sets.

  `census/` is data only: five JSON files, each with a JSON Schema, and a README. Each entry has a stable id, such as
  `a100-sxm4-80gb/bf16`, `dc-1ms-100gbps` or the subcircuit template `gemm_coordinate` with its bound subcircuits, and
  its sources.
- `verity_numerical.bench.census` reads the files directly (`$VERITY_CENSUS` overrides the location), and
  `tests/bench/test_census_data.py` validates them against their schemas.
- No value changed. Renders are byte-identical, and `--parity` is unchanged.
- Core holds no peak. `verity.proofs.target.Target` names only its `anchor_device`; `native_peak` and `NativePeak`
  are gone.
- `contract.native_peak_of(target)` returns the census line for the target's anchor device and operand dtype.
- `latency.pipeline` takes `native_s_per_vu` as a parameter.

## 2026-09-25: route (a) is its own full-relation A-GKR configuration

- Route (a) (A-GKR's prime proof, the operands as keyed-BLAKE3 rows proved in Flock flock-128-r2 with live coins, joined by the
  sigma link; PROTOCOL 17.5–17.6) gets a drill-down variant and a Table 1 configuration: full committed relation over
  `frame-v3/blake3-row`, NON_ZK_PROOF, 2^-130.19 (red-team-flock's fifth audit), live coins, collision resistance a Table 1
  assumption (decision 57). Its results fill the A-GKR column; plain A-GKR GPU results still read "weaker statement".
- The spec views read a result registered with a lane summary as its meta and the envelope as its JSON payload (as
  `cell.py rederive` registers art:4b52879f), as the drill-down already did. The legacy rules still read the meta only, so
  `--parity` is unchanged (15/15).

## 2026-09-25: live cells count prover compute once, and loopback coin handling as compute

- A live result whose `interaction` block says `t_total_includes_wait` (Flock's live coins) takes its prover compute from
  `live.prover_compute_seconds` (else `t.total` minus the wait). Before this, the network time was counted twice: once
  inside `t.total` and again as rounds × RTT.
- The prover's own per-round live-coin handling, measured on loopback (`live.loopback_round_seconds`), is prover compute:
  rounds × that, at most the measured wait, moves out of the network wait. Only the excess over loopback counts as
  network wait, and the ±10% tolerance check compares the measured wall against compute + rounds × 1 ms + bytes ÷ 100 Gb/s.
  Applies to every live cell (route (a), Flock). The tolerance is unchanged (root's ruling).
- The ±10% check validates the model against the run itself: the formula is evaluated at the run's measured RTT
  (`net.rtt_ms`, which must be the network round trip, not a figure derived from the prover's coin waits) and measured
  bandwidth (`net.bandwidth_bps`, else 100 Gb/s), and compared with the measured wall time. The published cell is still
  the formula at the 1 ms / 100 Gb/s reference. A cell's footnote names any loopback coin-handling figure counted as
  prover compute and how it was measured (`interaction.loopback_method`; a figure from another run says BORROWED).

## 2026-09-25: the switch to the approved tables spec (a change of method)

From the 6:00 PM PT digest on 2026-09-25, the published tables are the approved spec's views (`kb/TABLES.md`),
rendered by `python -m verity_numerical.bench.views --published`. The frozen 2026-09-22 tables
(`verity_numerical.bench.tables`) are still rendered, byte for byte as before, as a drill-down beside them.

- **Method.** Table 1 lists backend configurations and their security. Table 2 is the end-to-end overhead
  N ÷ P per subcircuit, hardware class and commitment scheme. N is the compute-bound datasheet estimate. P is
  instances per second, counting commitment, proving and the reference network (1 ms RTT, 100 Gb/s). A cell must
  prove the full committed relation over a scheme defined in `verity.commitments`. Table 3 splits each cell into
  serving (commitment), interaction and proving overhead. 2^-128 becomes the view's default filter, not a validity
  rule. The fixed B = 4096 and K = 1536 checks give way to the plateau of a recorded batch sweep. Algebraic hashes
  are admitted and flagged `(alg.)`.
- **Old numbers.** None of the 15 frozen cells meets the new rules as it stands. The A-GKR and bare B-Ligero
  cells prove a weaker statement (reason R). The B-Ligero + in-proof hash cells have no recorded protocol or plateau
  sweep (reason M). The switch digest (`python -m verity_numerical.bench.switch`)
  shows each old number beside its replacement, or beside the reason it left.
- **Parity.** Under today's parameters (`--rules legacy`), the new views select exactly the frozen cells, with the
  same artifacts and ratios. `views --parity --take-snapshot --record` checks this against a fresh
  `tables --snapshot` and records the check as a `tables-parity/v1` artifact.
- **Flock column.** Flock is its own backend family and gets its own column in Tables 1–3 (Daniel, 2026-09-25).
  That column holds pure-Flock configurations only: the relation and its row hashes in one binary-field circuit.
  Route (a) (the A-GKR prime side, Flock's row hashes and the sigma link) is an A-GKR configuration and stays in
  the A-GKR column. Until a Flock result qualifies, the column renders `—` with its reason.
- **Steward.** A `[[render]]` entry with `published = "views"` writes the views as `<stamp>-tables.md` and the
  frozen tables as `<stamp>-tables-frozen.md`. Without it, the entry renders the frozen tables as before.
