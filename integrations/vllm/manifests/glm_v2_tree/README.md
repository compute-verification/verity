# GLM v2 tree (`cursor/glm-v2-21f3`)

The integration tree the GLM-5.3 v2 rows of record are built, served and judged from (CIR-25, G1). It is never a PR: each
change lands through its own PR, and a row of record cites this tree as `cursor/glm-v2-21f3@SHA`. Merges only, both sides
kept; never `next`.

Rows (each I256/O32, TP8, H200, eager, at its public cap):

- `glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager`
- `glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager`
- `glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__stoch-t0.8-p0.95__eager`
- `glm53__fp8__h200__tp8__b32__i256__o32__mixed-arrivals__greedy__eager`

## Merged

Base `origin/main` @ c0ebeb8b93 (0a0782abed through #2893), then, in this order:

| PR | head | carries |
|---|---|---|
| #2729 | f2629dde38 | Boolean-first; #2738 eead9401a9, #2694 f031a376f4, #2765 dec1238773, #2828 85d2659299, #2848 aaccf5e3d7 |
| #2836 | e2ef5e7ba1 | m1-rows (B8 multi-request path); #2831 3f13f1e1f2, #2828 85d2659299 |
| #2837 | d0fc072205 | a row of record done only at its public cap |
| #2838 | 615e13bdc3 | served-default capture harnesses (bf16 v6); #2828 85d2659299 |
| #2852 | ba1aa4b2d2 | TP served windows refused unless the run root is SHA-512; #2842 9ef0bf1d46 |
| #2824 | 5b5840cf7f | the committer's rows-sha512 iterations |
| #2894 (`cursor/tp-sha512-21f3`, CIR-63 part 1) | 210a785aff | TP Commits at rows-sha512 under the SHA-512 TP run root (`verity/tp-run-root/v2`, over 64-byte rank roots; a 32-byte root refused by name); #2852 ba1aa4b2d2, #2824 5b5840cf7f |
| #2906 (`cursor/boolv-21f3`, CIR-47) | 6ed1e0333d | CIR-47's Boolean versions are #2729's σ-lifts and on-demand constants; past #2729 it adds tests of GLM-5.3 TP8's constants, residual add, embedding shard and fused norm through them, circuit-check of a Definition without parameters, the pins of `Const32[0x00000001]_v2` and `Const32[0x0000011f]_v2`, and #2785 1c2e9db6bd's smell-check entries (no op or predictor code) |
| #2883 (`cursor/rows64-land-21f3`) | 1d8869d1e4 | `MoeSum_v3` (`MoeSum_v1` on bits as a batch of `MoeSum_v2{TOPK,64}`), A11's Boolean-first probes |
| #2888 (`cursor/multimem-land-21f3`) | af95f837ad | `AllReduceMultimem_v1` as a batch of 16-byte vectors (`AllReduceMultimemVector_v2`); its merges of #2836 and #2765 dec1238773 and its `test_predict` pin are this tree's already |
| #2893 (`cursor/private-one-class-21f3`, CIR-64) | 57081b13db | `partition_v2.build` refuses a v2 private partition whose units span more than one size class; it is on main 0a0782abed, so it brings this tree to there |
| `cursor/tp-readback-21f3` (#2935) | 2cf3c2c788 | tp-capture's `topology.json` and tp-commit's `versions.json` record `effective_execution`, read back from the engine's `vllm_config` as the single-GPU paths do, the driver's and (`tp2_probe`) each rank's own, so the Match and Commit jobs can show `gpu_memory_utilization`; record files only, no digest moves (the coordinator, 20 Oct 00:40Z) |
| `cursor/dsv4-v2-21f3` (DSV4V2), `jobs/` only | 8b7254f55b | the jobs read the row's model from its name (`jobs/model.sh`), so DeepSeek-V4-Pro's B1 runs on them from its own tree; `stage_cpu.sh`, `stage_ckpt.sh`, `match_gpu.sh`'s post-load check, `match_cpu.sha256`, `gpu_batch.sh`'s `MODEL`. GLM-5.3 keeps every default (revision, tree, row dirs, snapshot, checkpoint paths, config; no census skip, no derive sizing). Taken as files, not merged: DSV4's pins, its MANIFEST and the census skip in `memory_map.census_entry` stay on that tree (the coordinator, 20 Oct 09:10Z) |

ROWS64 and the multimem batch joined when ASK-61 was provisionally accepted (Daniel, 19 Oct 21:59Z, option A: public
power-of-two output classes up to 262,144 bits). Two conflicts, both sides kept:

- `integrations/vllm/verity_vllm/pipeline/serving_rows.py` `after_commit` (TPSHA512 against #2848, and again at #2894's head): the
  row-table layout's `row_tables` and `CS.commit`'s row salt key and step roots, beside #2894's `commitment` set after the
  provenance from the reader's committer's scheme.
- `integrations/vllm/tests/predict/test_predict.py`, GLM-5.3 B8 top-p's LP25/T6 request (multimem against the realized-arrival pin): this tree's
  predictor gives `b632583a…`, the realized-arrival pin; the vector form does not move that Program.

## To merge when their PR opens

- The PRs above re-merge at their head when it moves.
- The static MoE frontend switch for GLM's expert GEMMs. The ops landed in #2665; no switch commit is on any branch.
  GLM's FP8 MoE in this tree is already the unpermuted pair set: per (token, slot) pair, the expert id is a value operand
  and the slab is gathered inside the unit (`MoeExpertFp8Gemm(W)_v1`, the `indexed-read` construction).

## Not in v2

- CIR-44's fused-norm split: the GLM rows carry its named exception, as main does (the coordinator, 20 Oct 02:20Z).
- T8b's prefix commits (`cursor/prefix-tp-6f04`): ASK-42 is provisionally A (the prefix length is private), so a prefix row
  in a batch is proved cold and T8b's C-public commits stay out.

## GPU batch

Every script is in `jobs/` and runs from this tree's pushed head, `--cwd source` (publish: `--cwd clone`); every row dir is
`/workspace/cp/circuits/glm-v2/rows/<row>` on each node it visits, because the Commit's replay reads its Build and required
manifest at the paths the Commit recorded. `/workspace/fs` is not shared between vy-glm53-eu and the CPU nodes, so row dirs move
with `copy_rowdir.sh SRC DST LIST.sha256`: tar streams through the VM, every file then checked against the list the source job
wrote (T1's B1 legs: 104 GB, 1.1M files, about 11 min). The GPU jobs are submitted by `gpu_batch.sh a [NODE]` and `gpu_batch.sh b ROWS.env [NODE]`;
each waits for its GPUs on its node's lease, so a window runs back to back. eu is held only for a window: the lease, and the node, are
released where a CPU phase sits between (CIR-51).

One runbook for both models: each window runs GLM-5.3's four rows in value order from this tree, then DeepSeek-V4-Pro's B1
(`dsv4-pro__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager`) with `MODEL=dsv4-pro` from a checkout of
`cursor/dsv4-v2-21f3`, whose MANIFEST has that row's CPU gates and what its GPU leg stages on the node. The jobs read the row's
model from its name (`jobs/model.sh`), and `gpu_batch.sh` refuses a HEAD that is not a pushed commit of the model's tree.
Submitted in that order, the jobs take the node's lease in that order. A fresh CPU node is readied by `stage_cpu.sh` (the venv,
`libcuda.so.1`, the pin's metadata); GLM-5.3's pin names only `config.json` and `tokenizer.json`, so on a node other than
vy-glm53-cpu-2 its `model.safetensors.index.json`, which `build_record.sh` requires, is not staged.

A row is built, served and judged by one code tree (`records publish --tree` is not checked against the stage runs):
its first job writes `<row>.tree` (`cursor/glm-v2-21f3@SHA`) beside the row dir, and every later job (`jobs/tree_guard.sh`)
runs at that commit or at a descendant that changes only `manifests/records.json` and this directory. The Build of record and
the Match GPU legs are each their node's first job, so a copy settles the tree: the destination ends with the older of the two,
and `copy_rowdir.sh` copies nothing (exit 2) unless the newer changes only those since.
Only the Build and publish read a row's pins, so a moved row's re-pin may land between its Match GPU legs and its Build of
record. A PR head merged after a row's first job reaches that row only by rerunning it; its later jobs are launched from a
worktree at a commit the guard accepts (`git worktree add /tmp/glm-v2-SHA SHA`).

1. CPU, before window B (any order, overlapping window A): `build_record.sh` per row on vy-glm53-cpu-2 (the AOT Build of record:
   predict emit refused unless equal to the row's `aot.trace` Build, with its manifest). A row this tree moves is re-pinned
   first, since its AOT Build reads its trace and comparison from `records.json`: `trace_compare.sh` (`SKIP_AOT1=1` once the row
   is known to move) traces it, preserves the trace (`trace_art.sh`), and compares this tree's predictor with it beside an AOT
   Build against it (`SKIP_AOT2=1` leaves that out: the Build of record refuses the same mismatch); `repin.sh ROW RUN` then
   re-pins the row to the trace and switches it to that comparison as soon as the comparison is equal (a run still in its AOT
   Build is read over ssh), committed as "re-pin ROW", and `build_record.sh` runs at that commit (`AOT_TRACE=<the rehearsal's
   row dir>` on cpu-2 saves the fetch). A long trace splits in two runs: `TRACE_ONLY=1` ends after the trace, and
   `TRACE_RUN=<that run>` compares it; a trace restarted wider (`BUILD_JOBS=N`) takes `SEED_FROM=<the earlier row dir>`, whose
   finished derives are hard-linked in after `seed_trace.py` checks each (its `.rc`, result.json, whole files, descriptor
   digests). row_tp's `BUILD_TIMEOUT` bounds each derive and also the whole manifest (the word check, then one build per rank and
   request Program), so `build_record.sh` sets 24 h: the default 7200 s holds B1's manifest (85 min) but not B8's (about 5 h at 2
   processes; r20261020-084508-06dd and r20261020-084527-5384 failed at it). It runs the manifest in 8 processes and shares the
   word check's unit rules through `glm-v2/unit-rule-cache` (keyed by content, so a read rule is the computed one). `RESUME=1`
   reruns a failed Build in its row dir, keeping its derives and composed workloads and moving its verdicts and manifest into
   the run's `prior/`.
2. Window A (`gpu_batch.sh a [NODE] [ROW]`, NODE vy-glm53-eu by default, vy-glm53-h8 or any vy-glm53-* 8x H200 node): the four
   rows' Match GPU legs (`match_gpu.sh`), in value order (B1 greedy, B8 greedy, B8 stochastic, B32) so a short window lands the
   most. On a preemptible node an evicted leg reruns with `gpu_batch.sh a NODE ROW`, which moves the evicted `match/` aside (a
   `match/` whose legs ended is refused). Each leg's
   engines read `gpu_memory_utilization` back from their `vllm_config`, the driver and each rank (the no-observer control: the
   driver alone; `$OUT/gpu_memory_utilization.json`: 0.92, vLLM's default, exit 12 otherwise). They need no Build, so nothing
   is copied to the node first: the capture's `--build-rank` registers a Build's
   post-load tensors, an absent Build registers none, and GLM-5.3 has no model-level post-load hook (`loader_parity`; every
   rank's v2 trace records `post_load_tensors` `applies: false`). T1R takes
   the bf16 v6 capture (`BF16_ART`, from `verity-vllm records capture RUN`) and the G2 point (`benchmarks/served_commit/tp_decode.sh`,
   `ARM=rows-4k`: rows-sha512's 4 KiB rows, stock against Commit in one TP8 engine, TTFT, prefill and decode TPOT per batch 1, 8, 32)
   on eu first; `gpu_batch.sh t1r` (`cap_bf16_v6.sh`, then the G2 point) runs them only if those did not hold. Then
   DeepSeek-V4-Pro's B1 (`MODEL=dsv4-pro gpu_batch.sh a NODE` from its tree), after its checkpoint is staged on the node
   (`stage_ckpt.sh`, about 18 min) and its Build of record copied in: its loader has a post-load hook, and `match_gpu.sh`
   refuses its legs (exit 2) unless every rank's Build records the tensor the capture registers. Stop eu.
3. CPU: each row's legs copied from window A's node to the Build's (`copy_rowdir.sh vy-glm53-h8 vy-glm53-cpu-2 match.sha256`,
   or `vy-glm53-eu`), then `match_cpu.sh` (MATCH_PHASE=cpu, `--build-run`
   the Build of record, `--send match.sha256`); then the row dir (its `rowdir.sha256` and the `match_cpu.sha256` that
   `match_cpu.sh` writes, every file the Match CPU phase wrote) copied to window B's node.
4. Window B (`gpu_batch.sh b ROWS.env [NODE] [ROW]`, NODE vy-glm53-eu by default, or a preemptible 8x H200 node: vy-glm53-h8 or
   another): each row's Commit (`commit.sh`: rows-sha512 under `verity/tp-run-root/v2`, PENDING by design), in value order
   whatever ROWS.env's order, its bundles and row records preserved in the store. Stop the node. An eviction fails the
   Commit's run, which is published as failed, and that row's Commit reruns whole (`gpu_batch.sh b ROWS.env NODE ROW`) from a
   new commitment with new draws: the unfinished `commit/` and `commit.log` are moved aside (`commit.failed-*`), never resumed.
   Every attempt is a line of the row dir's `commit_attempts.jsonl` (run, node, tree, start, what it moved aside), which the
   row's `_rowmeta` and the Commit's outputs carry, and its number is in the bundles' meta; a Commit rerun on another node
   takes that file with the row dir. It commits against the Build of record in
   the same row dir, since publish refuses a Commit whose required manifest or per-rank Programs are not that Build's. Its modes
   are B1's Commit r20261019-074856-7c3a's and B8's r20261019-091846-1452's: REPLAY_DEFERRED=1, RETAIN=host (the default, not
   the G2 curve's `--retain none`) and PAIRS=1. Deferred, each rank's bundle `store/` holds its retained bytes, so the committed
   values are on disk for the replay and the audits; v1's B1 Commit r20261016-213336-6c35 was not deferred and kept none. The
   config's default is 3 pairs, which that v1 Commit ran, so a v2 Commit's overhead rests on one pair (G2's point measures
   overhead). Its engine reads `gpu_memory_utilization` back, the driver and each rank (`$OUT/gpu_memory_utilization.json`: 0.88, as the
   v1 Commits ran). Then DeepSeek-V4-Pro's B1 (`MODEL=dsv4-pro gpu_batch.sh b ROWS.env NODE` from its tree).
5. CPU, vy-nebius-cpu-3 (the verified checkpoint): the Build of record copied in, then `replay_publish.sh` per row (`--send
   bundles.json`): `tp-commit-replay` of every rank, then `records publish --tier heavy` with the nine captures of record cited
   by id (#2838's eight served-default captures, which this tree's `check_capture` accepts, and `BF16_ART`), the Commit's run
   record and the Match's record (its three tp2 files, which the job stores as the v1 records' were). A rank's replay is
   about 70 min at about 265 GB (B8 greedy, r20261019-123645-e69d: 8 ranks in 9 h in one job), so a row runs two `RANKS=` jobs
   side by side and then one without `RANKS` that publishes. `rejudge` of the decided Commit passed on that replay; on a
   PENDING Commit it reports `ranks` and `verdict` as FAIL, so it runs only after the replay.

On a node that is not one of circuits' Nebius H200 VMs (a RunPod 8x H200 as vy-glm53-rp1), the windows run the same way with
NODE as the parameter, but the GPU jobs assume what those VMs have:

- a research machine of project `verity` (`research pods register`; `research run --on` refuses an unlisted one), runs root
  `/workspace/research`;
- `/usr/local/bin/gpu-lease` and `nvidia-smi`: `match_gpu.sh` and `commit.sh` wrap the lease (every GPU at 0 MiB, no
  preemptible lease held). With no user systemd (a container), gpu-lease runs its command without a memory cap and says so;
- `/workspace/cp/venv312` (vLLM and the pipeline's Python), `/workspace/cp/hf` (GLM-5.3's snapshot at `REV`), `/usr/local/cuda`;
- the checkpoint read through the HF snapshot from `/workspace/fs/glm53/$REV`, the Nebius shared filesystem, or a verified copy at
  `/workspace/cp/glm53-ckpt/$REV` (`stage_ckpt.sh` stages one from the hub, every file checked against `checkpoints.json`)
  served there by `model.sh`'s `ckpt_bind`: bound over it with `sudo -n mount --bind` where that directory exists, linked to
  it where it does not (a container without the shared filesystem, sudo or the mount capability);
- row dirs at `/workspace/cp/circuits/glm-v2/rows/<row>`, the path the Commit records and its replay reads, with room for a
  row's Build of record, legs and bundles (B1's legs alone: 104 GB), and `/tmp` for the jobs' scratch.

GPU time (wall of the GPU job, which holds all 8 GPUs but for the capture):

| job | GPUs | wall | measured by |
|---|---|---|---|
| bf16 v6 capture | 1 | ~10 min | not yet run at v6 |
| Match GPU legs B1 | 8 | 23 min | r20261018-094537-15b3 (1402 s) |
| Match GPU legs B8 greedy | 8 | 28 min | B8 greedy's row, match-gpu 1655 s (17 Oct) |
| Match GPU legs B8 stochastic | 8 | 17 min | r20261018-071806-5fac (997 s) |
| Match GPU legs B32 | 8 | ~40 min | never run; B8's scaled |
| G2 rows-4k point | 8 | 5-9 min | r20261018-194044-414f, r20261018-195722-b284 |
| Commit B1 | 8 | 58 min | r20261019-074856-7c3a (lease 08:11-09:09Z) |
| Commit B8 greedy | 8 | 110 min | r20261019-091846-1452 (lease 109.9 min) |
| Commit B8 stochastic | 8 | ~110 min | never run; B8 greedy's |
| Commit B32 | 8 | ~4 h | never run; its bundles ~4x B8's |
| DeepSeek-V4-Pro checkpoint staging | 0 | ~18 min | its MANIFEST (r20261020-072311-c012's rate, scaled to 864.7 GB) |
| Match GPU legs DeepSeek-V4-Pro B1 | 8 | ~23 min | its v1 Match r20261018-023527-3dc1 (1379 s) |
| Commit DeepSeek-V4-Pro B1 | 8 | ~1 h | never run deferred; GLM-5.3 B1's |

Window A (the four GLM legs) is about 1.8 node-hours and window B about 8.6, so about 10.4 node-hours (83 GPU-hours), of which
6.5 are estimates (B32's two jobs, B8 stochastic's Commit); T1R's capture and G2 point add about 0.3 if `t1r` runs them.
DeepSeek-V4-Pro's B1 adds about 1.7 (its staging, legs and Commit, all estimates), so about 12.1 with both models. The
measured Commits ran under vllm-v1; rows-sha512's cost at TP8 is what the G2 point measures.
