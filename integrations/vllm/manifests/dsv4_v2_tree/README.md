# DSV4 v2 tree (`cursor/dsv4-v2-21f3`)

The integration tree that DeepSeek-V4-Pro's v2 row of record is built, served and judged from (G1). Like the GLM v2 tree
(`../glm_v2_tree/README.md`), it is never a PR: each change lands through its own PR, and the row of record cites this tree as
`cursor/dsv4-v2-21f3@SHA`. Merges only, both sides kept; never `next`.

Row: `dsv4-pro__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager` (I256/O32, TP8, H200, eager, at its public cap). Its v1
record art:e81265e7b65773352945b6b85612d6429a0b212353bd18d615281ad0b7940f04 (fold Match r20261019-061046-dad5, Commit
r20261018-023527-3dc1) predates the padded static circuit and the SHA-512 TP Commit.

## Merged

Base `origin/cursor/glm-v2-21f3` @ 1d70a06e23. Its MANIFEST, `../glm_v2_tree/README.md` "Merged", lists what that carries:
main c0ebeb8b93, #2729 (Boolean-first, with #2765 dec1238773), #2836, #2837, #2838, #2852, #2824, #2894 (TP Commits at
rows-sha512 under the SHA-512 TP run root), #2906, #2883, #2888, #2893 and #2935 (each rank's execution read back). Then, in
this order:

| PR | head | carries |
|---|---|---|
| #2765 (`cursor/dsv4-land-21f3`) | dec1238773 | DeepSeek-V4-Pro's Definitions, frontend and row. Already in the base through #2729, so it has no merge commit of its own (`git merge-base --is-ancestor dec1238773 HEAD`) |
| #2845 (`cursor/land-dsv4-aot-21f3`) | 67916f1be5 | the row switched to the AOT Build (`aot`: comparison r20261018-222738-a095, trace art:826e06ff60bf5ede2823cc3a859fb5f9e6c9b867ea0ffc47be17634f7362e5a4); `test_predict`'s mHC step pins read from `records.json`. One commit on dec1238773; merged clean |
| `cursor/glm-v2-21f3` | 5a31cfd737 | V2PREP's runbook since the base: window B on a preemptible node, any vy-glm53-* GPU node |

This tree's own commits change `../glm_v2_tree/jobs/`, this directory, and `verity_vllm/pipeline/memory_map.py`'s
`census_entry` (with its test), which is none of `build.py`'s construction sources, so no Program digest or artifact
identity moves. The jobs now read the row's model from its name
(`jobs/model.sh`: role, repo, revision, config, metadata snapshot, checkpoint paths, row dirs, tree, post-load tensors). GLM-5.3
keeps every default it had, so V2PREP can merge them into `cursor/glm-v2-21f3` as they are:

- `stage_cpu.sh` readies a fresh CPU node: the venv, the H200 hosts' `libcuda.so.1` from the store
  (art:b56ac0392465cf8e67426eaf2794ebec3aaada861e9a4309061eb3828c3ccec7, cuda-driver/v1, 580.173.02, from vy-glm53-cpu-2), and
  the row's checkpoint metadata. `trace_compare.sh` stages the metadata where it is missing: the pin's non-shard files from the
  hub, each checked against `checkpoints.json`.
- `stage_ckpt.sh` stages the row's whole checkpoint on a GPU node from the hub, every file checked against `checkpoints.json`,
  then `.verified`, served at the pin's `local_path`.
- `match_gpu.sh` refuses a post-load model's legs unless the row dir's Build records `POST_LOAD` on every rank (below).
- `match_cpu.sh` hashes what the Match CPU phase writes into `match_cpu.sha256`, the list for the copy to window B's node.
- `gpu_batch.sh` takes `MODEL=dsv4-pro`.
- `model.sh`'s `DERIVE_GB` sets `BUILD_JOBS` from the run's `--mem-gb`, and its `CENSUS=skip` (`VERITY_BUILD_CENSUS`, read by
  `census_entry`) leaves the request Build's census out (CPU gates, below).
- `replay_publish.sh` refuses a non-GLM row: its conformance captures are GLM-5.3's.

## Not in v2

- CIR-44's split of `MhcFusedPostPre_v2`, the 19.5M-gate mHC post+pre Call: the row carries its named exception, as main does
  (`OVER_PARTITION_BUDGET` in `tools/verity/circuit_check/tests/test_circuit_check.py`).
- T8b's prefix commits, as in the GLM v2 tree.

## CPU gates

The jobs are `../glm_v2_tree/jobs/`, with `REF` and `SW` defaulting to this tree and `/workspace/cp/circuits/dsv4-v2/rows`.

A B1 request derive takes 35 min and peaks at 21.8 GB (`research mem`'s GB, 10^9 B; `result.json`'s `max_rss_mb` 20,785
MiB). The census the Build then writes beside it (`pipeline.memory_map.census_entry`, which v1 predates) is the Commit's opt-in memory
map's and decides no Build. On this Program it ran past 50 min at 161 GB without finishing (GLM-5.3's takes 2 min), in
`census.census`'s pass-through loop, which pairs each node's reader spans with its pass-through runs. In r20261020-071632-1bcf,
`row_records.derive_bytes` planned 13.5 GB a derive, so `--build-jobs auto` started all 8 ranks at once and cpu-3's cgroup OOM
killer (its /user.slice caps at 497 GB) took five of them. So `model.sh` sets `CENSUS=skip` (`VERITY_BUILD_CENSUS`: the Build
writes no census, its `result.json`'s `census` says it skipped, and `records.json` has no census digest) and `DERIVE_GB=32`,
over the census-free trace's largest process (27.6 GB, r20261020-085740-90ff): every job that derives runs `BUILD_JOBS` = its
`--mem-gb` over 32, so 9 at once in 300 GB. The GPU leg's Commit runs without `MEMORY_MAP`; one with it computes the census
then (`memory_map.of_build`).

1. `trace_compare.sh` (`SKIP_AOT1=1`): the v2 trace, `predict compare` against it, and the AOT Build against it, on
   vy-nebius-cpu-3 after `stage_cpu.sh`. r20261020-071632-1bcf is the first run (cancelled in its request derives' census),
   whose 8 step derives seed r20261020-085740-90ff (`SEED_FROM`, `SKIP_AOT2=1`): its predict compare is EQUAL on all 24
   Programs, and its trace is art:b889027e16ceef9c2c8d625008f3360ca7b86dcf7d22b9f7d8ee2153dbf6e52b.
2. `repin.sh ROW RUN vy-nebius-cpu-3`, committed as "re-pin ROW": the step, request and workload Programs of all 8 ranks from
   the v2 trace, `records.json`'s `aot` switched to its comparison and trace art.
3. `build_record.sh` at that commit (`AOT_TRACE=<the trace's row dir>` on the same node): the Build of record, with its
   manifest and `rowdir.sha256`. This is the row's first job, so it writes `<row>.tree`.

## GPU leg

The node is a parameter, `NODE`: vy-glm53-eu, vy-glm53-h8 (preemptible), or a RunPod 8x H200 registered as vy-glm53-rp1.
Every job runs from this tree's pushed head, `--source . --cwd source`, and the row dir is
`/workspace/cp/circuits/dsv4-v2/rows/<row>` on every node it visits.

Staged on the node before window A:

- the checkpoint: `deepseek-ai/DeepSeek-V4-Pro` at b5968e9190ef611bbf34a7229255be88a0e937c1, 864.7 GB (64 safetensors shards,
  the largest 14.0 GB, and 5 small files), not gated. `research run --on NODE --mem-gb 32 --timeout 4h ... -- env ROW=<row>
  bash integrations/vllm/manifests/glm_v2_tree/jobs/stage_ckpt.sh` writes it to `/workspace/cp/dsv4-ckpt` and links it at the
  pin's `local_path`, `/workspace/fs/dsv4/b5968e9190ef611bbf34a7229255be88a0e937c1`. From the hub at the measured 1.0 GB/s
  (r20261020-072311-c012 on vy-nebius-cpu-3: 55.5 GB in 55 s, checked in 10 s), that is about 15 min to download and 3 min
  to check. It needs 865 GB free on `/workspace`, besides the row dir. The only other copy is eu2's data disk
  (`/workspace/cp/dsv4-ckpt`, verified), which serves only while eu2 runs; eu's was deleted on 18 Oct (CIR-34).
- the Build of record: `copy_rowdir.sh <CPU node> NODE rowdir.sha256 /workspace/cp/circuits/dsv4-v2/rows`. Unlike GLM-5.3,
  DeepSeek-V4's loader has a post-load hook. Each rank's Build records `model.layers.0.hc_attn_fn_broadcast` ([24, 7168] f32,
  mHC's `hc_attn_fn` summed over the residual streams), and the capture registers it as a root from the Build's request
  Programs (`--build-rank`, `component_programs.post_load_tensors`, which reads each Program's `result.json` beside its
  `instances.json.gz`). An absent Build registers nothing, so `match_gpu.sh` refuses the legs (exit 2) until every rank's Build
  records it.
- `/workspace/cp/venv312` (pod_bootstrap.sh), `/usr/local/bin/gpu-lease`, `nvidia-smi`, `/usr/local/cuda`, and a research
  machine of project `verity`, as `../glm_v2_tree/README.md` lists for a RunPod node.

Then:

1. Window A: `MODEL=dsv4-pro gpu_batch.sh a NODE`, which runs `match_gpu.sh`: capture and control, gpu_memory_utilization
   0.92 read back from the driver and each rank, `match.sha256`.
2. CPU: `copy_rowdir.sh NODE <CPU node> match.sha256 /workspace/cp/circuits/dsv4-v2/rows`, then `match_cpu.sh`
   (`BUILD_RUN`, `--send match.sha256`), then `copy_rowdir.sh <CPU node> NODE match_cpu.sha256 ...` back to window B's node.
3. Window B: `MODEL=dsv4-pro gpu_batch.sh b ROWS.env NODE`, where ROWS.env is the line `<row> BUILD_RUN MATCH_RUN`. It runs
   `commit.sh`: rows-sha512 under `verity/tp-run-root/v2`, REPLAY_DEFERRED=1, RETAIN=host, PAIRS=1, gpu_memory_utilization
   0.88 read back, with the bundles and `_rowmeta` preserved. A preemptible node may run it. An eviction fails the run, which
   stays published as failed, and the Commit reruns whole from a new commitment with new draws
   (`gpu_batch.sh b ROWS.env NODE <row>`); `commit_attempts.jsonl` counts the attempts.
4. CPU: the Commit's replay and publish. `replay_publish.sh` is GLM-5.3's (its conformance captures), so DeepSeek-V4-Pro's
   publish step is still to be written. Its Boolean replay (`pipeline.boolean_replay`) builds each picked row's Boolean
   version, and some are large to build at the served statics: `BL.is_boolean` on `Dsv4QNormRopeKvInsert_v2` took 45 min at
   225 GiB, and on `FlashMlaDecode_v2` passed 424 GB in 90 min without finishing, in `boolean_dsa.build`'s `bits.compile`
   (r20261020-083555-d46d, cancelled). Its host needs more than cpu-3's 497 GB.

GPU time (wall of the GPU job, all 8 GPUs):

| job | wall | measured by |
|---|---|---|
| checkpoint staging (no GPU) | ~18 min | r20261020-072311-c012's rate, scaled to 864.7 GB |
| Match GPU legs B1 | ~23 min | v1 r20261018-023527-3dc1: match 1379 s, the check included |
| Commit B1 | ~1 h | GLM-5.3 B1's deferred one-pair Commit r20261019-074856-7c3a (58 min). v1's DSV4 Commit, 3 pairs replayed inline, took 3.5 h (12586 s) |
