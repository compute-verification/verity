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

## Builds of record

On vy-glm53-cpu-2, row dirs `/workspace/cp/circuits/glm-v2/rows/<row>`. Each passed (`build PASS`, manifest rc 0), its manifest
complete (no unmodelled value, no unbound TP peer), every Program `aot`, and every rank's step, workload and request digest equal
to the row's pin in `records.json` at its tree, which is this tree's head's. `rowdir.sha256` is in the run's outputs: the list
`copy_rowdir.sh` takes (its sha256 and file count here); the size is the row dir without `match/` and `commit*` (`du -sb`).

| row | `build_record.sh` run, its outputs | `vllm.build` Attempt | tree | manifest digest | identities | call boundaries | `rowdir.sha256` | size |
|---|---|---|---|---|---|---|---|---|
| B1 greedy | r20261020-062225-e880, art:b145a0da2a75f45bebb9928352b5527445506715d7d36c1e79509d424a37b723 | r20261020-062246-b987, art:32c488d8ca7bb12f80b38e45ec5a97d95312bcb62974a4d776da0b58d1adec35 | e2c986e7ba | `9cc899187e87230548b6c00a7b5390e78187dd23a56d29ebaad848a1b2a2a77a` | 670,120 | 39,936 of 39,936 | `8d3a09c2a33d2fad0f4b39d2b88c8dac379eb51b10abe63b39c2c09bc68c0600`, 215 files | 7,491,220,559 B |
| B8 greedy | r20261020-132536-b2d0, art:3727b9d6083b726afc997d68f0a3c1ccff949bcde87c0dc6f9cc505a19afd419 | r20261020-132554-c189, art:3bed6c69048dc98305cfca262f2aee4bc76f21a98d48690badb6babc97464dfe | 6fcebc0433 | `9e85828772b8aa55a05c6e145e64178e4e53880086997b63764de959ad4a9ef1` | 2,925,528 | 173,472 of 173,472 | `579139933be78d33bc869687fd97544716553f26211dae6b7b92b789f3f02c1e`, 775 files | 27,804,214,059 B |
| B8 stochastic | r20261020-132545-cc45, art:5d0c575053d2280b25076e8b963cf82004fa6e417028535df962e25c29b04b8f | r20261020-132557-d5f6, art:54d43fba35acf3a56af7b9779929c4cabbf2f0c6fce2b53359dbcce64446296c | 6fcebc0433 | `9003863bfe2eedb99a660522552b7b4866975da168ea08bc65d5fb1a0c6ff2af` | 2,529,368 | 149,760 of 149,760 | `376debc7048c57a3d65159fbce511b18251fac78c0d704b087b15e37cc92aeaa`, 775 files | 27,166,896,631 B |

The B8 rows are `RESUME=1` reruns (each about 60 min) of r20261020-084508-06dd
(art:4e82f2aee40dc827c8ff33749c0ecd3f319246feb30fe31f292db9b27dd8a5f0) and r20261020-084527-5384
(art:764b8ee91900985294eaa2cdea9522852cc367c830141e267a997b627226cd18), whose manifests overran row_tp's 7200 s; their verdicts and
manifest logs are in each rerun's `prior/`.

B32 is re-pinned (5af673ff05) to its v2 trace r20261020-060948-f9c7
(art:da5479fee29a71d62fe67e22cefdd73a853cba54e22e97b70c9097501f8435de, tree 0094d2ef10) and switched to its predict compare
r20261020-130447-1c49 (tree 8ca6f9878d, 5 h 34 min): EQUAL on every rank, 272 of 272 Programs traced equal to predicted, and
every new pin equal to the trace's `build_summary.json` digest. 264 pins moved (each rank's workload and 32 request Programs);
the 8 step Programs did not. Its Build of record, r20261020-192024-d435 on vy-glm53-cpu-2 after that node's restart for its
grown disk ("GPU batch" step 1), runs at de4c6e3620 with `AOT_TRACE` the trace's row dir, `RESUME=1`, `BUILD_JOBS=24` and
`--mem-gb 470`, from the B8 Builds' measured peaks (each run's process tree, 2 s samples):

| phase | B8 Builds of record | B32 |
|---|---|---|
| request derives (`predict emit`) | 7 at once, at most 17.7 GB each (median 9), about 20 min each; 75 GB | 24 at once: at most 432 GB if every one peaks; its trace's 24 derives peaked at 202 GB |
| GP-01 composes | 2 at once, 57 GB each: 111 to 116 GB, the Builds' peak | 2 at once (the default 400 GB budget, `compose_bytes` 145 GiB): 153 GB each in its trace r20261020-060948-f9c7, so 306 GB; B8's 116 GB scaled by batch is 464 GB |
| manifest, 8 processes | 48 GB, 59 min (r20261020-132536-b2d0) | B8's scaled by batch: 192 GB |

While it holds its 470 GB, vy-glm53-cpu-2 has about 285 GB left (its 773 GB limit less 18.5 GB in use outside reservations):
B1's Match CPU phase fits there at `--mem-gb 280` (three of its 60 GB GM-01 legs at once beside two 40 GB folds), and B32's
Build, best effort, yields to any G1 row that needs the room (`RESUME=1` keeps its derives and composes).

## Window A

On vy-glm53-eu from 23:31Z 20 Oct (GO 23:30Z), each row's Match GPU legs (`gpu_batch.sh a vy-glm53-eu ROW`). Each engine read
`gpu_memory_utilization` back as 0.92, the driver and all 8 ranks (the control: its driver), and vLLM's own sizing line on every
rank of both legs reads "Desired GPU memory utilization is (0.92, 128.62 GiB)": 89.39 GiB weights and non-torch memory, 0.52 GiB
peak activation and a 38.71 GiB KV cache; nvidia-smi read 133,911 MiB a GPU during B1's capture. The read-back check exits 1
on a rank off by 0.02 or a missing rank.

| row | `match_gpu.sh` run | tree | legs | `match/` | `match.sha256` |
|---|---|---|---|---|---|
| B1 greedy | r20261020-233119-e40c (`vllm.match` r20261020-233150-30e7, art:4cf0a6018fe689dd8622cac371ecd114475f612f43f542e21b23530181b5dd54) | c8695f2187 | PASS, 805 s (the first load from a cold page cache took 416 s) | 104 GiB | 1,098,214 files |
| B8 greedy | r20261020-235054-0dfc (`vllm.match` r20261020-235116-20b7, art:8c26cb7f88573090904af9763fba5703eae0d1443141bc4e1a37119b5d4c3fd9) | de6ca12b5f | PASS, 677 s | 312 GiB | 1,193,569 files |
| B8 stochastic | r20261020-235116-5968 (`vllm.match` r20261020-235125-02be, art:fa65f85a4ff07e1501f974d151a5169ee8eb90ef8a271591c1a187f52340cd6e) | de6ca12b5f | PASS, 3920 s from 23:51Z: about 13 min on the GPUs from 00:43Z, once the v1 rows' deletion made 360 GiB | 341 GiB | 1,192,954 files |
| DeepSeek-V4-Pro B1 | r20261021-010646-2249 (`vllm.match` r20261021-010655-e910, art:c376282619cc3cfc8bcfd5675e6253cac7224634cb0204789950ea56f3431a92), `MODEL=dsv4-pro` | dsv4 47ddb6246e | PASS, 1000 s; the post-load tensor on all 8 ranks' Builds | 139 GiB | 1,343,105 files |

Window A ended there at 01:25Z: B32's legs (about 1100 GiB) wait for room, which eu's disk does not have beside window B's.

eu's `/workspace` (3.0 TB) had 477 GiB free at the start: the GLM-5.3 checkpoint (704 GiB) and @circuits' v1 rows held the rest.
The v1 rows (1.4 TiB: `fs/glm53-m1` 965 GiB, `cp/circuits/glm-aot` 181, `cp/circuits-prefix-tp` 176, `fs/glm53-failfast`
110) were deleted at 00:42Z 21 Oct with `research retention rm --approved-by @circuits --ref CIR-25-v2-batch`
(r20261021-004135-4cb2, logged in eu's `research/retention/deletions.jsonl`), once its check found no live run, job script,
process or outside symlink that names or holds them; two run records from 16 and 18 Oct still read `running`, their processes
gone since eu's boot at 23:24Z 20 Oct. `cp/glm53-ckpt` stays, and so do the rows' legs, which window B reads on eu. Window B
also needs room (a B8 row's dir after the CPU phase, about 580 GiB, and its Commit's bundles, 244 GiB in v1's B8 greedy), so
DeepSeek-V4-Pro's checkpoint (864.7 GB, staged from the hub by r20261021-005054-35cc in 604 s at 1431 MB/s, every file checked)
left eu at 01:26Z once its B1 legs passed (r20261021-012549-7e39, `research retention rm`), leaving 990 GiB free; it is staged
again for its window B. Its Match CPU phase's start resolves the checkpoint dir (`config.checkpoint_dir`: the pin's
`local_path`, else the hub snapshot under `HF_HOME`), as the CPU nodes do from the metadata `stage_cpu.sh` puts at SNAP, and eu
had no DeepSeek-V4-Pro snapshot: the first attempt r20261021-012525-f740 stopped there (rc 10) and appended a `build FAIL` line
that every later reader of `stages.txt` would take for the Build's verdict. r20261021-013423-fed6 copied that attempt's records
aside (`match_cpu.failed-r20261021-012525-f740/` in the row dir), put `stages.txt`, `row.log`, `row.json` and
`effective_config.json` back to the legs' (each the attempt's prefix), and staged the pin's metadata at SNAP (`stage_meta`);
the rerun's start resolves the snapshot.

Each row's Match CPU phase (`match_cpu.sh`), placed by circuits (23:40Z 20 Oct and 00:30Z 21 Oct): CPU Match may run on eu's own
CPUs while the GPU batch holds it, 750 GB in all at most, and stops if a GPU leg or a Commit needs the memory. A B8 row runs
one GM-01 at a time at 450, as it would at 750: (750 − 80) / 370 = 1.

| row | Match CPU phase | node, `--mem-gb` | legs on the node |
|---|---|---|---|
| B1 greedy | r20261021-001940-db82, from 00:19Z | vy-glm53-cpu-2, 280 (three ranks at once) | copied from eu, every file checked |
| B8 greedy | r20261021-012405-e484, from 01:24Z | vy-glm53-cpu-2, 450 | 1,011,902 of its files copied again after the VM's restore cut the first copy, then all checked |
| B8 stochastic | r20261021-011240-7676, from 01:12Z | vy-glm53-eu, 450 | its own; its Build of record (774 files and `.tree`) copied in from cpu-2 |
| DeepSeek-V4-Pro B1 | r20261021-013528-0d47, from 01:35Z (the first attempt's start failed, below) | vy-glm53-eu, 290 (the dsv4 tree's three ranks) | its own; its Build of record (214 files and `.tree`) copied in from vy-nebius-cpu-3 |

B32's Build of record r20261020-192024-d435 ended before the standing rule (19:25Z 20 Oct) had to cancel it: build PASS at
00:57Z (wall 20,186 s, digest 28e931b1de151fd5), call boundaries PASS at 00:59Z (671,424 identities); BUILD_RUN
r20261020-192046-c5a4, BUILD_ART art:3e79e2ccc4c7159a3218717433be5bf6ba9f51c8c599009aabeacd599140a992, `rowdir.sha256`
`d457bdd6477b0308fd104608e3ba112622fd0b9ce7dd108fd8344043ae9e60aa` (2,695 files). Nothing to restart.

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
`libcuda.so.1` from `CUDA_DRIVER_ART`, the model's metadata): `CUDA_DRIVER_ART=art:b56ac0392465cf8e67426eaf2794ebec3aaada861e9a4309061eb3828c3ccec7`
is `libcuda.so.580.173.02`, the driver vy-glm53-cpu-2 and vy-nebius-cpu-3 load. GLM-5.3's `checkpoints.json` pin names only
`config.json` and `tokenizer.json`, so `model.sh`'s `META_EXTRA` names its other non-shard files (`model.safetensors.index.json`,
which `build_record.sh` requires, `tokenizer_config.json`, `generation_config.json`, `chat_template.jinja`) at the sha256
vy-glm53-cpu-2's snapshot held for the B1 and B8 Builds of record; `stage_meta` downloads each at `REV` and refuses any other.

A row is built, served and judged by one code tree (`records publish --tree` is not checked against the stage runs):
its first job writes `<row>.tree` (`cursor/glm-v2-21f3@SHA`) beside the row dir, and every later job (`jobs/tree_guard.sh`)
runs at that commit or at a descendant that changes only `manifests/records.json` and this directory. The Build of record and
the Match GPU legs are each their node's first job, so a copy settles the tree: the destination ends with the older of the two,
and `copy_rowdir.sh` copies nothing (exit 2) unless the newer changes only those since.
Only the Build and publish read a row's pins, so a moved row's re-pin may land between its Match GPU legs and its Build of
record. A PR head merged after a row's first job reaches that row only by rerunning it; its later jobs are launched from a
worktree at a commit the guard accepts (`git worktree add /tmp/glm-v2-SHA SHA`).

1. CPU, before window B (any order, overlapping window A): `build_record.sh` per row (the AOT Build of record: predict emit
   refused unless equal to the row's `aot.trace` Build, with its manifest), B1's and the B8 rows' on vy-glm53-cpu-2 ("Builds of
   record" below), B32's on vy-glm53-cpu-2 too, after the node's restart for its grown disk, with `AOT_TRACE=` its trace's row
   dir there (`/workspace/cp/circuits/glm-v2/r20261020-060948-f9c7/<row>`, 79 GB): vy-glm53-cpu-3 (cpu-d3: 256 vCPU, 1 TiB, 3 TB
   disk) could not be launched when B32 was re-pinned. On another node a Build of record follows `stage_cpu.sh`, with the
   trace's row dir copied (`copy_rowdir.sh SRC DST LIST.sha256 /workspace/cp/circuits/glm-v2`). A row this tree moves is re-pinned
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
   `match/` whose legs ended is refused). A row's capture also waits for its room on `/workspace` (`DISK_NEED_GIB`, from the v1
   captures: B1 104 GiB, B8 greedy 312, B8 stochastic 341, B32 about 1100 scaled), checked once the GPUs are free; eu had 477
   GiB free at window A's start, which holds B1's and B8 greedy's, so the later rows' legs start once room is made. Each leg's
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
   `match_cpu.sh` writes, every file the Match CPU phase wrote) copied to window B's node. The phase's bottleneck is the
   per-rank GM-01 (with its shared X-09): 339 GiB and about 1.5 h a B8 rank, beside folds of about 40 GB and 2 h two at a time
   (WSBSTOCH's B8 stochastic rehearsal r20261019-162751-9b01: 17 h, one GM-01 at a time). `match_cpu.sh` runs as many ranks'
   GM-01 at once as the run's `--mem-gb` holds: one B8 rank at 450 on vy-glm53-cpu-2 (755 GiB) or vy-nebius-cpu-3 (497 GB
   cap), two at 1000 with `FOLD_WORKERS=4` on vy-glm53-cpu-3 (1 TiB, 1099 GB: three need 1110 GB with no fold beside them),
   three at 1300 on vy-nebius-1 (1690 GiB). Scaled from the 17 h at one, two take about 9 h and three about 7 h (unmeasured). B8 greedy's runs on vy-glm53-cpu-3 if it is up, and B8 stochastic's on vy-nebius-1 or
   vy-glm53-cpu-2. B1's 60 GB and B32's 1500 GB a rank are B8's scaled by batch, unmeasured: B32's runs only on an otherwise
   empty vy-nebius-1, one rank at a time, and refuses a `--mem-gb` that cannot hold one. A row dir after the phase is about
   580 GiB (B8) and 1.19M inodes.
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
