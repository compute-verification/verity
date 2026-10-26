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
| `cursor/verity-circuits-6f04` (V2PREP), cherry-picked as 15ea726d5b | a371c5a7f1 | `binding.build_binding_map` indexes leaves in each layout meta's own `chunk` and refuses a `chunk` that differs. The TP worker's map (`rank_worker._tp2_binding`) kept the old 256 B default after 008faa17c3 moved rows-sha512 to 4 KiB rows, so every TP Commit's population openings raised IndexError ("Window B"). Only the Commit calls `build_binding_map`, so `jobs/tree_guard.sh` lets a row's later jobs run past exactly this fix (`COMMIT_FIXES`, by blob) |
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
| B32 greedy | r20261021-052310-44ec (`vllm.match` r20261021-052332-1be9, art:3e1434447f4a3963b534848a5ede7929a02124b46be19facde0eea9a24f32897), `ROW_ON=/workspace/extra` | 6d850cae23 | PASS, 2198 s | 904 GiB, on `/workspace/extra` | 1,534,973 files |
| DeepSeek-V4-Pro B1, attempt 3 | r20261021-065033-b2ef (`vllm.match` r20261021-065042-373e, art:2a239c99a2c131c4c782f6550717a18d82dee7f97fde2927d994d714d0b10da9), `MODEL=dsv4-pro` | dsv4 582c1317f1 | PASS, 1302 s; the post-load tensor on all 8 ranks' Builds; `capture/align_schedule.rank<r>.pt` on all 8 ranks (#3028) | 139 GiB | 1,343,220 files |

Window A ended there at 01:25Z: B32's legs (about 1100 GiB) waited for room, which eu's disk did not have beside window B's,
and ran from 05:23Z on `/workspace/extra` (infra, 01:55Z 21 Oct). DeepSeek-V4-Pro B1's attempt 3 needs a capture with the align
schedule (`../dsv4_v2_tree/README.md`, window B), so its legs ran again at 06:50Z, once T1R freed the GPUs at 06:33Z; attempt 2's
`match/` was renamed `match.attempt2-r20261021-010646-2249` in the row dir first.

eu's `/workspace` (3.0 TB) had 477 GiB free at the start: the GLM-5.3 checkpoint (704 GiB) and @circuits' v1 rows held the rest.
The v1 rows (1.4 TiB: `fs/glm53-m1` 965 GiB, `cp/circuits/glm-aot` 181, `cp/circuits-prefix-tp` 176, `fs/glm53-failfast`
110) were deleted at 00:42Z 21 Oct with `research retention rm --approved-by @circuits --ref CIR-25-v2-batch`
(r20261021-004135-4cb2, logged in eu's `research/retention/deletions.jsonl`), once its check found no live run, job script,
process or outside symlink that names or holds them; two run records from 16 and 18 Oct still read `running`, their processes
gone since eu's boot at 23:24Z 20 Oct. `cp/glm53-ckpt` stays, and so do the rows' legs, which window B reads on eu. Window B
also needs room (a B8 row's dir after the CPU phase, about 580 GiB, and its Commit's bundles, 244 GiB in v1's B8 greedy), so
DeepSeek-V4-Pro's checkpoint (864.7 GB, staged from the hub by r20261021-005054-35cc in 604 s at 1431 MB/s, every file checked)
left eu at 01:26Z once its B1 legs passed (r20261021-012549-7e39, `research retention rm`), leaving 990 GiB free; its window B
serves eu2's verified copy instead ("Window B"). Its Match CPU phase's start resolves the checkpoint dir (`config.checkpoint_dir`: the pin's
`local_path`, else the hub snapshot under `HF_HOME`), as the CPU nodes do from the metadata `stage_cpu.sh` puts at SNAP, and eu
had no DeepSeek-V4-Pro snapshot: the first attempt r20261021-012525-f740 stopped there (rc 10) and appended a `build FAIL` line
that every later reader of `stages.txt` would take for the Build's verdict. r20261021-013423-fed6 copied that attempt's records
aside (`match_cpu.failed-r20261021-012525-f740/` in the row dir), put `stages.txt`, `row.log`, `row.json` and
`effective_config.json` back to the legs' (each the attempt's prefix), and staged the pin's metadata at SNAP (`stage_meta`);
the rerun's start resolves the snapshot.

At 15:21Z and 15:22Z 21 Oct, with eu's `/workspace` at 43 GB free and `research data evict` finding nothing to evict, `research
retention rm --approved-by @circuits` deleted two superseded dirs of ours (logged in eu's `research/retention/deletions.jsonl`):
B1 attempt 1's moved-aside `commit.failed-20261021T141525Z` (57.8 GiB; its bundles are in the store, and vy-nebius-cpu-3 keeps
its own copy) and DeepSeek-V4-Pro attempt 2's `match.attempt2-r20261021-010646-2249` (148.3 GiB, superseded by r20261021-071904-0dc9),
leaving 250 GB free.

That room went before DeepSeek-V4-Pro's attempt 4 wrote its bundles. eu's sampler (`resources.jsonl`, kind `storage`) read
`/workspace` falling from 248 GiB free at 15:42Z to 0 at 15:58Z, about 15 GiB a minute, from the start of TPEVAL's re-judge
of attempt 3 on eu (r20261021-154123-222c and r20261021-154150-cdb2, staging into `cp/circuits/dsv4-v2/rejudge-a3/`). Over
those 16 minutes the Commit's ranks wrote about 33 GiB and 7676 about 34 (per-process `io.write_bytes`). B1 attempt 2 and
DeepSeek-V4-Pro attempt 3 show no such transient of their own. To make room for attempt 5 and the B8 Commits, B8 greedy's row moved to
`/workspace/extra`: r20261021-153850-3245 copied its legs there (1,193,568 files, each equal to 0dfc's `match.sha256`), `research
retention rm --approved-by @circuits` removed the `/workspace` original (309.6 GiB, logged in `deletions.jsonl`), and extra's
`circuits/glm-v2/rows/<B8 greedy>` is bound at the row's path (`row_on`). Its Build of record and ranks 0 to 6's folds
(r20261021-154235-427a hashed them on cpu-2: 63 files, 203 GB) are copied there from cpu-2 with `copy_rowdir.sh`. The rest of e484's
records follow when it ends, and its Commit runs with `ROW_ON=/workspace/extra` (647d2cc451).

Each row's Match CPU phase (`match_cpu.sh`), placed by circuits (23:40Z 20 Oct and 00:30Z 21 Oct): CPU Match may run on eu's own
CPUs while the GPU batch holds it, 750 GB in all at most, and stops if a GPU leg or a Commit needs the memory. A B8 row runs
one GM-01 at a time at 450, as it would at 750: (750 − 80) / 370 = 1.

| row | Match CPU phase | node, `--mem-gb` | legs on the node |
|---|---|---|---|
| B1 greedy | r20261021-001940-db82, from 00:19Z | vy-glm53-cpu-2, 280 (three ranks at once) | copied from eu, every file checked |
| B8 greedy | r20261021-012405-e484, from 01:24Z: match PASS at 18:06Z 21 Oct (`vllm.match` r20261021-012811-2bfd, art:b3e33742f8b75c2cb3bf353cb54979c388847c99fc3e0ff061ca05cdf9db2315), 59,670 s, every rank's fold, compare and GM-01 and the cross-rank leg | vy-glm53-cpu-2, 450 | 1,011,902 of its files copied again after the VM's restore cut the first copy, then all checked |
| B8 stochastic | r20261021-011240-7676, from 01:12Z: match PASS at 18:16Z 21 Oct (`vllm.match` r20261021-012926-a901, art:de4e3e1c5a29201c4369143e880ab55aa6885a3c5d81b89953b32b40307df005), 60,066 s, every rank's fold, compare and GM-01 and the cross-rank leg | vy-glm53-eu, 450 | its own; its Build of record (774 files and `.tree`) copied in from cpu-2 |
| DeepSeek-V4-Pro B1 | r20261021-013528-0d47, from 01:35Z (the first attempt's start failed, below) | vy-glm53-eu, 290 (the dsv4 tree's three ranks) | its own; its Build of record (214 files and `.tree`) copied in from vy-nebius-cpu-3 |
| DeepSeek-V4-Pro B1, attempt 3 | r20261021-071645-a966, from 07:16Z at dsv4 582c1317f1 | vy-glm53-eu, 100: 0d47's cgroup peaked at 53.2 GiB (`resources.jsonl`), and 450 + 100 is inside the 750 | its own (b2ef's) |

B32's Build of record r20261020-192024-d435 ended before the standing rule (19:25Z 20 Oct) had to cancel it: build PASS at
00:57Z (wall 20,186 s, digest 28e931b1de151fd5), call boundaries PASS at 00:59Z (671,424 identities); BUILD_RUN
r20261020-192046-c5a4, BUILD_ART art:3e79e2ccc4c7159a3218717433be5bf6ba9f51c8c599009aabeacd599140a992, `rowdir.sha256`
`d457bdd6477b0308fd104608e3ba112622fd0b9ce7dd108fd8344043ae9e60aa` (2,695 files). Nothing to restart.

## Window B

On vy-glm53-eu once T1R freed its GPUs (02:43Z 21 Oct), each row's Commit as its Match CPU phase ends (circuits, 01:55Z and
04:05Z 21 Oct): DeepSeek-V4-Pro B1 first, since its Match ended before B1's.

Retain mode. `commit.sh` exports `RETAIN=host` (with REPLAY_DEFERRED=1 and PAIRS=1), and nothing in `jobs/` passes
`--retain none`; the driver logs `commit staging: window_mb=256 slots=8 retain=host`. Both trees carry #2824's iteration 4
(590902f955, which keeps the prefill's hashes on the GPU and runs out of GPU memory under `--retain none` at B8 and B32 on
GLM-5.3 TP8), as `main` does. Under host retention T1R's B8 rehearsal on eu (r20261021-022741-21d6, rows-4k) peaked at
135.1 GiB a GPU of 139.8 and 514.7 GB of host (64.2 GB a rank), inside `commit.sh`'s `--mem-gb 600`, beside the Match CPU
phase's 450 on eu: 1050 GB of eu's 1610 GB admission. Its prefill ran about 9 times stock.

`MATCH_RUN` is the Match CPU phase's `vllm.match` Attempt, the line `match RUN art:… rc=0` in `match_cpu.sh`'s `chain.out`,
not the `research run` job: DeepSeek-V4-Pro's first attempt named the job (r20261021-013528-0d47) and `row chain` refused it
at its start ("not a passing Match Attempt"), before the GPUs or any `commit/`; it is attempt 1 in `commit_attempts.jsonl`.

DeepSeek-V4-Pro's checkpoint is eu2's verified copy, on eu2's data disk hot-attached to eu at `/workspace/eu2data` (infra,
01:50Z 21 Oct): `/workspace/eu2data/cp/dsv4-ckpt`, staged by infra on 17 Oct (`hf/hub` there holds only links to eu2's
Qwen and Llama snapshots). r20261021-040920-b9af hashed every file of the pin against `checkpoints.json` with no download
(69 files, 864.7 GB, 410 s, none bad) and then linked `/workspace/cp/dsv4-ckpt` (`model.sh`'s `LOCAL_CK`) to it; the pin's
`local_path` already linked `LOCAL_CK`, so `ckpt_bind` serves it unchanged.

The Commits' row dirs stay at their paths on eu's `/workspace`, which the Commit records and its replay reads (916 GiB free
at DeepSeek-V4-Pro's start). `/workspace/extra` (2 TiB, infra 01:55Z 21 Oct) takes B32's legs and the room the B8 Commits need.
B32's legs run with `ROW_ON=/workspace/extra gpu_batch.sh a vy-glm53-eu <B32>`: `match_gpu.sh` binds
`/workspace/extra/circuits/glm-v2/rows/<B32>` at the row's path (`model.sh` `row_on`, an absent or empty row dir only), and the
capture waits for its 1100 GiB on the row dir's own disk. A bound row's Commit (`ROW_ON=/workspace/extra commit.sh`, which
reports the bind or makes it) hard-links the row's records into `_rowmeta`'s temporary dir inside the row dir: under `$SW`, on
another disk, `cp -l` failed with "Invalid cross-device link" and, its status unchecked, would have preserved an empty `_rowmeta`.

| row | `BUILD_RUN` | `MATCH_RUN` | Commit | attempt | result |
|---|---|---|---|---|---|
| DeepSeek-V4-Pro B1 | r20261020-154507-f208 | r20261021-013752-f5ed (art:5f5d56d8b4fd8a324a16941242765d756952c8aeec3dc56422a0e83353ac461a), of r20261021-013528-0d47: match PASS 03:30Z, 6761 s, every rank's fold, compare and GM-01 and the cross-rank check | r20261021-041858-f624 from 04:19Z, its `vllm.commit` Attempt r20261021-041907-b7a8 (verdict art:01daa488cb9b35fc400c983a2db7ae5d4155e06f8de2b91c6143e02df6a260f6): weights loaded from eu2data in 7.5 min; the driver and all 8 ranks read back 0.88 (`commit/versions.json`); vLLM's sizing line 123.02 GiB, 102.67 GiB weights and non-torch | 2 | FAIL rc 12 at 04:54Z, no `commit/summary.json`: the match oracle (a value disagreement) and the population openings (IndexError). Stopped, not rerun (below) |
| B1 greedy | r20261020-062246-b987 | r20261021-002121-ef31 (art:630ee63db089112d635a45768487b27c1a2fd31ace6697c88937e3d5b66073c4), of r20261021-001940-db82: match PASS 07:50Z, 26,923 s, every rank's fold, compare and GM-01 and the cross-rank check; its 111 files copied to eu by 08:25Z, all checked | r20261021-082530-c049 from 08:25Z at 5ca9a5818a (tree_guard: "changes only pins, this runbook and COMMIT_FIXES"), its `vllm.commit` Attempt r20261021-082555-c2fa (verdict art:8a9bae5b849ed81765d419ea06386801fe7eba19a044308ffa110e91b6358771): the driver and all 8 ranks read back 0.88; the GPUs held 35.5 min | 1 | PENDING by design (the C2 sampled replay and boundary linkage deferred), rc 0 at 09:27Z: the match oracle PASS on all 8 ranks (5,056 tensors each), value check PASS, weights pin, cross-rank collectives 476 of 476 equal, 512 openings verified, 283,136 fold-match bindings with none bad. Bundles 8 x 7.2 GiB (`bundles.json` in art:41a43881fb83837a0d04a26ec5bfff6015a9256d5399fe07a9d500e54342f156); replay and publish r20261021-092916-8e81 on vy-nebius-cpu-3: ranks 0 to 4 replayed by 12:48Z, each 32,748 of 32,748 equal with boundary linkage 32/32 (1,600 to 2,160 s, peak RSS up to 113 GiB); the CPU decision's per-rank gates, run in memory over those 5 ranks with 8e81's own code (5ca9a5818a), fail each one on the cuBLAS route pin alone (below), the no-evaluator gate passing (`EmbeddingShard_v1` and the collectives waived by the cross-rank pass); 8e81 ended rc 3 at 13:55Z, the FAIL on the route pin, so nothing is published. All 8 ranks replayed, 32,748 of 32,748 equal with linkage 32/32 on each (ranks 5 to 7 in 1,608 to 1,722 s, peak RSS 113 GiB), and every other gate of its `commit_summary.json` passes (oracle compare, boundary linkage, cross-rank collectives, fold-match binding, query population, weights pin, value check): each rank fails the route pin alone. Rerun as attempt 2 |
| B1 greedy, attempt 2 | r20261020-062246-b987 | r20261021-002121-ef31, as attempt 1 | r20261021-141533-38b2 at 77c62e506e (the route pin's fix, below; tree_guard: "changes only pins, this runbook and COMMIT_FIXES"), after attempt 1's `commit/` and `commit.log` were moved aside by hand to `commit.failed-20261021T141525Z` (`commit.sh` refuses a `commit/` with a `summary.json`); its `vllm.commit` Attempt r20261021-142644-628a (verdict art:ef914ef2a81319b1a149847433db95ef71fd3727c1c2e5b62f93ebf75f14b1cc): the driver and all 8 ranks read back 0.88 | 2 | PENDING by design, rc 0 at 15:09Z (admitted at 14:26:43Z beside 0b1c): the match oracle PASS on all 8 ranks (5,056 tensors each), value check PASS, weights pin, cross-rank collectives 476 of 476 equal, 512 openings, 283,136 fold-match bindings with none bad. Every rank's facts of record name the venv's `libcublas.so.12` at 120901 (cuBLASLt 120901), with no route departure; flashinfer logs `cublas_version: saved=13.1.1 vs current=12.9.1` and ignores the empty cache on each rank. Bundles 8 x 7,334 MiB (`_rowmeta` art:76127c8195d394bc2944cef8113d322b891298d29ea9f42031240528e78fb379). Replay on vy-nebius-cpu-3 from 15:10Z, attempt 1's `commit/` moved aside there under the same name: `RANKS` 0,1,2 r20261021-151039-390b, 3,4,5 r20261021-151058-76fa and 6,7 r20261021-151106-e85b, every rank 32,748 of 32,748 equal with boundary linkage 32/32 (1,704 to 2,523 s, peak RSS up to 113 GiB). PASS decided on CPU by 76fa, the last of them, at 16:59:08Z from all 8 ranks' replays: sampled replay, boundary linkage and C2 population. Published by r20261021-171723-31d3 at f8fd971662 (`records publish --tier heavy`): record art:74c6c3ccd452e00ba6d2215c4282d9ae75a44b7da6d6d96861bdef5b1d661d69, match record art:69ad691edce688f08427e85929489ebbf54355e0aeb63828c897a60e7e7226f8, commit record art:83eea7d38a22b20d352bc378223400acf852da75cc45218e9a58faa010ce3fe5; its `records.json` change (manifest digest 9cc899187e87230548b6c00a7b5390e78187dd23a56d29ebaad848a1b2a2a77a) is this tree's |
| DeepSeek-V4-Pro B1, attempt 3 | r20261020-154507-f208 | r20261021-071904-0dc9 (art:ecd31bd53bfa895ea07a5fbf371e07ab0e82128a9da477d19ad20cae2e5684d8), of r20261021-071645-a966: match PASS 09:10Z, 6647 s, every rank's legs and the cross-rank check | r20261021-092814-0b1c from 09:28Z at dsv4 582c1317f1, attempt 2's `commit/` moved aside to `commit.failed-20261021T092823Z`: the driver and all 8 ranks read back 0.88 | 3 | PENDING by design at about 10:05Z: the match oracle PASS on all 8 ranks (5,920 tensors each), each rank's `schedule_replay` 1,952 calls replayed with `missing` 0 and `routing_differs` 0; value check PASS, weights pin, cross-rank collectives 374 of 374 equal, 512 openings verified, 222,208 fold-match bindings with none bad; `moe_block_stream` the harmless pattern on all 8 ranks (bound-as-leaf, 23,616 tensors, `at_calls`, `hash_layers` [0, 1, 2]). Bundles 8 x 8.6 GiB. Its `_rowmeta` swept in attempt 2's renamed `match.attempt2-r20261021-010646-2249/` (1,343,350 files, 150 GiB), which `commit.sh` did not exclude (fixed in 8a1874456a), so the job preserved it until 14:50Z (below) and ended rc 0 at 14:50:46Z, `_rowmeta` art:22da9ae9481be9867f0d81f7a77a92bb75b16bb91b5d0b0cc22ff2181cd8969e PRESERVED; its `bundles.json` is the one the replay jobs were sent (same sha256). FAIL decided on CPU at 12:43Z by r20261021-114900-5d55 (rc 3), once all 8 ranks' sampled replays had passed (r20261021-114850-5bfd ranks 0 to 3, 5d55 ranks 4 to 7: 14,167 of 14,167 equal on each, boundary linkage 32/32, 676 to 1,136 s, peak RSS 116 to 119 GiB). Every rank fails two gates: no-evaluator families, 567,483 VUs in 19 families this tree registers no evaluator for, which TP form (B) never covers (`Dsv4InvRopeQuantHead_v1` 280,112, `SiluMulClamp_v1` 122,549, `TritonBlockRMSNorm_v1` 35,014, `MhcPostPreTf32_v1` 30,976, `FlashMlaGatherRow_v1` 17,598, `Dsv4QNormRopeKvInsert_v1` 17,507, `MoeRouterSqrtSoftplus_v1` 16,646, `Dsv4CompressorScore_v1` 16,456, `FlashMlaPrefill_v1` 15,616, `GatherI32x129280_v1` 5,166, `MhcFusedPostPre_v1` 3,751, `Dsv4MlaCompressC4_v1` 2,130, `FlashMlaDecode_v1` 1,891, `MoeRouterHash_v1` 861, `MhcPreBroadcast_v1`, `MhcResidualBroadcast_v1`, `MhcPost_v1` and `HcHead_v1` 287 each, `Dsv4MlaCompressC128_v1` 62); and the cuBLAS route pin (below). Rerun as attempt 4 |
| DeepSeek-V4-Pro B1, attempt 4 | r20261020-154507-f208 | r20261021-071904-0dc9, as attempt 3 (b2ef's capture, its 8 `align_schedule` files) | r20261021-152252-6748 from 15:23Z at dsv4 4f78cf55e4 (17019d4282, the route pin's fix, with 8a1874456a cherry-picked, so `match.*/` stays out of `_rowmeta`; tree_guard: "changes only pins, this runbook and COMMIT_FIXES"), after attempt 3's `commit/` and `commit.log` were moved aside by hand to `commit.failed-20261021T152234Z`; its `vllm.commit` Attempt r20261021-152313-25f3 (verdict art:b51ef00a400467725fe7f4af4db5924159bd2f2ff03530397fd8205556bb2b74) | 4 | FAIL rc 12 at 16:00:52Z, eu's `/workspace` full (below): pair 0 committed with the match oracle PASS on all 8 ranks (5,920 tensors each, 1,952 MoE align witnesses equal and none differing) and 512 of 512 openings, then every rank's replay-bundle write raised OSError 28 and the Commit failed fast before the value check, with no bundles. Rerun as attempt 5, a whole new commitment, as for an eviction |
| DeepSeek-V4-Pro B1, attempt 5 | r20261020-154507-f208 | r20261021-071904-0dc9, as attempt 3 | r20261021-161703-abfb from 16:17Z at dsv4 f5892498fd (TPEVAL's evaluators for the 19 families, #3075, #3076, #3077 and #3079; tree_guard passes from 1bbf831029), after attempt 4's `commit/` and `commit.log` were moved aside by hand to `commit.failed-20261021T161359Z`, with 557 GiB free; its `vllm.commit` Attempt r20261021-161715-e71e (verdict art:a143231724c0cc49f8427e632b75eab29788c7247ace77a4308aaead3d074075): the driver and all 8 ranks read back 0.88 | 5 | PENDING by design at 16:53:45Z: the match oracle PASS on all 8 ranks (5,920 tensors each), each rank's `schedule_replay` 1,952 calls replayed with `missing` 0 and `routing_differs` 0 and its 1,952 MoE align witnesses equal; value check PASS, weights pin, cross-rank collectives 374 of 374 equal, 512 openings verified, 222,208 fold-match bindings with none bad. Every bundle's engine of record names the venv's `libcublas.so.12` at 120901 (cuBLASLt 120901). Bundles 8 x 8,739 MiB, sealed by 16:55Z, when the GPUs were free; the job ended rc 0 at 17:20:33Z, every bundle preserved and `_rowmeta` art:14cf49b91233416102200a20b5a9897f22baa2530e59ced727ed162afc8995d9 (147 files, 795 MB) |
| B8 stochastic | r20261020-132557-d5f6 | r20261021-012926-a901, of r20261021-011240-7676 | r20261021-181707-464b from 18:17Z at 09ac5ac3a5, in place on eu's `/workspace` (427 GiB free); its `vllm.commit` Attempt r20261021-181730-2f80 (verdict art:4d58f17c14d1b898dd991c0cc3916ac3507aa0c9b1705e524fb75e1f9e915cf5); `cublas_names.txt` the venv's two `.so.12`; the driver and all 8 ranks read back 0.88 | 1 | FAIL rc 12 at 19:06:42Z, no bundles: pair 0's instrumented run (53 steps, tokens equal to its control's, 512 openings verified, 442,400 fold-match bindings with none bad) fails identity coverage on every rank, 52,005 of 316,171 required identities unbound and 55,970 not in the manifest (the first r1's `logits_processor` output at step 9), so the Commit stops before the value check. A visible failure, not rerun unchanged. Its cause is the declaration's idle gaps, not the sampling: the Match capture (7676) and both arms of this pair served the same 120 tokens in the same 50 forwards (`tokens_equal_ref` on both arms; attribution 50 steps, 120 rows, none differing from the capture), each with seeded requests and `ignore_eos`, every request ending at its cap (53 is the Commit's `step()` iterations, 43 GM-01's count of non-empty declared steps). r0 (cap 2) ends at forward 1 before r1's declared step 9, and r1 ends at 8 before r2's declared 13; the serving loop admits the next request once the engine idles (`engine/arrivals.py`, `realized_arrive_steps`), so every run served r1 at 2 and r2 at 9, while the TP Commit places the Build's manifest at the declared steps (`pipeline/tp/commit.py`). circuits (19:20Z 21 Oct): B8 stochastic stays in G1, its schedule reproducible from the request itself, so that the AOT Build's (request, step) manifest is the served one; a schedule-adaptive manifest is post-G1. Attempt 2 declares the arrivals the serving loop realizes (e52b596c3c, `arrive_steps` 0, 2, 9, 13, 15, 16, 18, 18) and reruns the whole chain ("B8 stochastic attempt 2", below). This attempt's row dirs are moved aside: cpu-2's `<row>.at-6fcebc0433-d5f6` (its Build of record) and eu's `<row>.at-6fcebc0433-464b`, whose `match/` (555.3 GiB) `research retention rm --approved-by @circuits` removed (r20261021-202743-c31e), its 116 non-tensor files (each step's sampled tokens and rows, the logs) kept in `match.kept/` |
| B8 greedy | r20261020-132554-c189 | r20261021-012811-2bfd, of r20261021-012405-e484; its last 24 files (rank 7's folds and the rewritten records, 29.2 GB) copied to extra by 18:35Z, all equal | r20261021-183934-d980 from 18:39Z at 2b354e82de, `ROW_ON=/workspace/extra` (`disk_at_start.txt` reads `/dev/vde`, 454 GiB free); its `vllm.commit` Attempt r20261021-183958-acbe (verdict art:89040a317102c342d33376640fa845b134899f9e27ad526afebf164383f27041); `cublas_names.txt` the venv's two `.so.12` | 1 | FAIL rc 12 at 19:47:05Z, no bundles: pair 0's instrumented run (48 iterations, tokens equal to its control's, 512 openings verified, cross-rank collectives 478 of 478 equal) fails identity coverage on every rank, 494 of 365,691 required identities unbound, each bound as an ambiguous second copy: r1's MLA `self_attn/bmm*` outputs at engine steps 9 to 12 and 15 to 18, from step 9 (r1's prefill beside r0's decode) on, steps where two requests at different steps of their own run together. Cause: `query/manifest/bind.py` spelt a call boundary for every request in one step-wide order, using the first request's position for a key both requests spell (a layer's own `self_attn` output, a `BatchedGemm_v1` Call's output such as the W_UK_T absorb), so r0's `bmm_1404` was paired with r1's `self_attn` output. Fixed in 190ee66678 (each request's own Program order, with a test that fails on the old binder) and named in `tree_guard.sh`'s COMMIT_FIXES (d03373a350); the bug is on `main` too, where #3128 carries the fix (3efbf9688c, marked ready 21:21Z). Offline on eu (r20261021-200706-e93d), d980's rank 0 and rank 7 binding maps re-spelt by the fixed binder bind 365,691 of 365,691, none ambiguous, changing exactly those 494 entries. Rerun as attempt 2 |
| B8 greedy, attempt 2 | r20261020-132554-c189 | r20261021-012811-2bfd, as attempt 1 | r20261021-200958-ca80 from 20:10Z at d03373a350 (side branch `cursor/glm-v2-bind-order-5795`: 2b354e82de and the binder's fix; tree_guard passes from 6fcebc0433 by COMMIT_FIXES, while the tree's head also carries e52b596c3c's workload change, which is not one), `ROW_ON=/workspace/extra`, after attempt 1's `commit/` and `commit.log` were moved aside by hand to `commit.failed-20261021T200939Z`; its `vllm.commit` Attempt r20261021-201021-6b4e (art:921aa5b0acd06047c40e4a422716314615374703646aa283d15df40123eb4aa1) | 2 | PENDING by design at 20:52:59Z (2,554 s): value check PASS on 8 of 8 ranks, cross-rank collectives 478 of 478 equal, 512 openings, 415,856 fold-match bindings with none bad, identity coverage PASS. Off eu's GPUs ~20:53Z, the last B8 Commit there tonight. Bundles 8 x 26,043 MiB, all PRESERVED, the job rc 0 at ~21:29Z (`_rowmeta` art:475a7b51817d6e0e9a4f648185c528717c539b2c1c2c5aae0b0d6831936faf22). Replay on vy-glm53-cpu-2 at 1afe24b078: d03373a350 plus #3118's carry and its tree_guard entries, blob-identical to d38ccf5aaa's, where tree_guard refuses this row (e52b596c3c changes `pipeline/workload.py`). `RANKS` 0,1,2,3 r20261021-213022-e72f from 21:30Z; 4,5,6,7 r20261021-213033-aa93, admitted once the Build of record frees its 320 GB. Every rank replayed, 141,860 of 141,860 sampled VUs equal with boundary linkage 139/139 (3,898 to 4,587 s, peak RSS 131 GiB); e72f ended rc 4 (ranks remain). FAIL decided on CPU by aa93, the last of them, at 04:01:15Z (rc 3, so nothing is published), on one gate alone: the no-evaluator families. `LlBf16SplitKGemm_v1`, vLLM's cuteDSL MoE router gate for 5 to 16 tokens in a step (`LlBf16SplitKGemm_v1{N=256,K=6144,SPLIT_K,TILE_K=256,WARPS=4}`), is in the Build's step Program, and the replay has no row kernel for it: 3,975 VUs on each rank (31,800 in all) are not evaluable by name, and no other pass covers them, so C2 of record is not established on any rank. Sampled replay, boundary linkage, C2 population, value check, attribution, the match oracle (7,426 of 7,426 on each rank) and the cross-rank collectives all pass. B1 never reaches this family: its steps carry at most 4 tokens, where the gate is `LlBf16DotprodGemm_v1`, which `program/kernels/glm53_rows.py` evaluates. `bf16_gemm_check.splitk_rows`, the family's CPU model, exists on `main` and in this tree, but no row kernel registers it on any ref. B8 stochastic's step Program holds the same family (SPLIT_K=6), so its replay needs that row kernel first. A visible failure, not rerun. circuits (05:45Z 22 Oct): a fresh Commit, attempt 3, with new live coins. Re-judging attempt 2 with the split-K strata added would change the sample after its outcome is known (`challenge.stratum_picks` keys each stratum by its sorted index), so it is out |
| B8 greedy, attempt 3 | r20261020-132554-c189 | r20261021-012811-2bfd, as attempt 1 | r20261024-232308-142f from 23:23Z 24 Oct on vy-glm53-eu at the side branch's head ae3c360e04 (#3163's carry 38126ff6d0, the replay's row kernel for `LlBf16SplitKGemm_v1` wrapping `bf16_gemm_check.splitk_rows` with #3130's self_check, and its tree_guard entries), which tree_guard admits from 6fcebc0433, so its Commit and replay run at one head; `ROW_ON=/workspace/extra`, the job moving attempt 2's `commit/` and `commit.log` aside to `commit.failed-20261024T232324Z` (`commit.sh` refuses a `commit/` with a `summary.json`); its `vllm.commit` Attempt r20261024-232324-9337 (verdict art:777fbe16d501de102613e55807a5f06e5ec057b925a6c7dff1cedd80831c7304); `cublas_names.txt` the venv's two `.so.12`; the driver and all 8 ranks read back 0.88. eu had been out of eu-north1 H200 capacity from 10:35Z 22 Oct | 3 | PASS. The Commit was PENDING by design, rc 0 at 00:17:57Z 25 Oct (run record art:b232637cf89eeadaaef2c0a463329cf846302557d2443bd629a819b5b639548a; 9337 rc 0 at 00:06:59Z, 2,613 s): value check PASS on 8 of 8 ranks, the match oracle 7,426 of 7,426 tensors on each, cross-rank collectives 478 of 478 equal, 415,856 fold-match bindings with none bad, weights pin. Bundles 8 x 26 GiB, all preserved (`_rowmeta` art:9f260e846f09c1091ab70e8edc756a0172b89e2349af99bdb3b128f8b1b17215, 114 files, 3.1 GB). Replay on vy-glm53-cpu-2 at ae3c360e04 (circuits, 00:41Z 26 Oct), 190 GB a job from attempt 2's replay peaks: the prep r20261026-004349-6900 moved attempt 2's replayed `commit/` aside to `commit.failed-20261026T004406Z` and fetched `_rowmeta` and the 8 bundles (rc 0 at 03:32Z); then `RANKS` 0,1,2,3 r20261026-004407-835c and 4,5,6,7 r20261026-004413-89f8, every rank 142,910 of 142,910 sampled VUs equal with boundary linkage 139/139 (3,944 to 4,407 s, peak RSS 131 GiB). PASS decided on CPU by 835c, the last of them, at 08:06:13Z 26 Oct from all 8 ranks' replays: sampled replay 1,143,280 of 1,143,280 equal (`LlBf16SplitKGemm_v1`'s 31,800 VUs evaluated; the families with no evaluator are the collectives and `EmbeddingShard_v1`, which the cross-rank pass covers), boundary linkage 1,112 of 1,112, C2 population, query population, weights pin, over the Commit's own run root, per-rank roots and manifest digest (each replayed bundle's `MANIFEST.json` the sha256 the Commit's `bundles.json` names). Published by r20261026-004419-55d8 at ae3c360e04 (`records publish --tier heavy`, rc 0 at 08:07:19Z): record art:3556aa9dd6d0233428de823495d324c1c8a873ace1cf537f52c8c7946424bea4, match record art:5eea1e829dd1dcc1795007f984fad34f52ec73fb6cd04bcb2a2835f2d7dea1b7, commit record art:b232637cf89eeadaaef2c0a463329cf846302557d2443bd629a819b5b639548a; its `records.json` change (manifest digest 9e85828772b8aa55a05c6e145e64178e4e53880086997b63764de959ad4a9ef1) is its `out/records_json.diff`, for the re-pin |

B8 stochastic attempt 2 (circuits, 19:20Z 21 Oct) reruns the whole chain from a new row start at the tree's head: its first
job writes the row's `.tree` at ed92872cc1, which carries the declaration (e52b596c3c) and the binder's fix (190ee66678). No
trace or re-pin comes first: each request and step derive of its AOT Build predicts its launch batch from the arrivals the
serving loop realizes (`realized_arrive_steps`), which the new declaration leaves as they were, so each must still equal the
`aot.trace` Build's; the GP-01 workload Programs and the manifest carry `arrive_steps` and move, and `records publish` indexes
the Build's Programs as the row's pins. The Match GPU legs are the GPU proof of the declaration: their capture and its
no-observer control are two served runs, which must serve the same tokens in the same engine steps, each request first served
at its declared step.

| step | run | node | state |
|---|---|---|---|
| Build of record | r20261021-202321-42f1 from 20:23Z at ed92872cc1: `AOT_TRACE` the trace's row dir on cpu-2 (`r20261019-235311-53f8/<row>`), `BUILD_JOBS=14` (of 72 derives), `BUDGET_GB=260` (4 composes at once, 57 GB each measured), `VERITY_MANIFEST_JOBS=16`, `--mem-gb 320`, which leaves B8 greedy's first replay job its 280 | vy-glm53-cpu-2 | PASS at 21:58:07Z, rc 0 (1:35:23 wall, peak RSS 54 GiB): its `vllm.build` Attempt r20261021-202343-1cbc (art:220e18dad5c772a73dfc7e12c49653b5b2d56f091ea96e4ebbbbd609e90fb26a), construction aot, 8 shapes, digest a26c43faa02a1efc; build-call-boundaries PASS, 149,760 identities all claimed. `rowdir.sha256` 775 files, copied to cpu-3 and eu from ~22:01Z |
| Match GPU legs | r20261021-205420-2513 at ed92872cc1, its `vllm.match` Attempt r20261021-205442-b36b (art:66471ebcaf47f13926a31479f5beba0d309dfe3221e66e5ff1cfa20485fe23a4), on the GPUs ~20:55Z to 21:09Z right after ca80 (DSV4V2's legs not yet queued); `gpu_memory_utilization` 0.92 read back on the driver and all 8 ranks; 341 GiB, `match.sha256` 1,192,873 files | vy-glm53-eu | PASS, capture_rc 0 and control_rc 0. The declaration holds: capture and no-observer control served the same 120 tokens in the same 50 engine steps (51 iterations, was 53), all 8 ranks equal, and every request's declared arrival, realized arrival and first served step are equal (0, 2, 9, 13, 15, 16, 18, 18); the control's tokens equal attempt 1's. Copied to cpu-2 from 21:11Z (`copy_rowdir.sh`, the trees equal) |
| Match CPU phase | split by `match_cpu.sh`'s `RANKS` (280642f9cd), the two halves at one commit: `RANKS=4,5,6,7` on vy-nebius-cpu-3 (circuits, 21:05Z: 450 GB, 23:15Z to 08:00Z, DSV4V2's jobs first on its 503 GB), over the Build of record and the legs' 92 small files (those ranks' observer logs, no tensors), ~8.6 h from B8 greedy's per-rank times (a fold 2.1 h, two at once; a GM-01 1.5 h, one at a time); then cpu-2's full run with `MATCH_REUSE=legs` after B8 greedy's replay, which keeps the ranks cpu-3's documents cover and runs the rest and the cross-rank leg | vy-nebius-cpu-3, vy-glm53-cpu-2 | legs subset on cpu-3 by 21:31Z (92 files equal eu's). cpu-3's first start, r20261021-231331-885d, failed at its Build check before any fold: its `vllm.match` r20261021-231403-b64f ended rc 10, "no local checkpoint dir for zai-org/GLM-5.3 @ aca966…", since cpu-3 had no GLM-5.3 metadata (a setup failure, no proof or coins). `stage_cpu.sh` staged it (r20261021-231625-9bdd, rc 0: `config.json`, the tokenizer files, `model.safetensors.index.json`, `generation_config.json` and `chat_template.jinja`, each equal to its pin). Relaunched as r20261021-231741-4076 at 575f91cf30 (`vllm.match` r20261021-231757-5916), folding ranks 4 and 5 from 23:18Z, ~08:00Z projected; its cross-rank check reads `pass False` as designed (structure, provenance and tokens each pass, provenance over 0 collectives, no tensors there), and its `tp2_match.json` is never copied out. Its lease reads no sooner than 23 Oct 11:46:51Z (the run's `--timeout 30h`). cpu-2's full run r20261021-232124-2123 at 575f91cf30 (`MATCH_REUSE=legs FOLD_WORKERS=3`, `--mem-gb 520`) waited for memory from 23:21Z behind B8 greedy's two replay jobs (280 GB each) and r20261021-221921-a898 (96 GB, not ours), admitted at 03:47:33Z; its `vllm.match` r20261022-041131-656c from 04:11Z: the cross-rank check PASS at 04:12:53Z (structure, tokens, provenance over 7,120 collectives with no mismatch), then ranks 0, 1 and 2 folding at once. Folds of ranks 0 to 2 ended ~06:16Z; GM-01 takes ~99 min a rank here, one at a time (rank 1 06:17Z to 07:56Z, then rank 2 to ~09:36Z, then rank 0 to ~11:15Z); rank 3 folds from 07:56Z and its GM-01 follows rank 0's, ~11:15Z to ~12:55Z, then the cross-rank leg: ~13:15Z to 13:30Z projected (was 12:30Z, from a 1.5 h GM-01 and a different order). 4076 ended rc 0 at 07:39:16Z: its per-rank judge reads fold, compare and GM-01 PASS on ranks 4, 5, 6 and 7 (its `vllm.match` r20261021-231757-5916 ended rc 11 as designed, ranks 0 to 3 absent there), `match_ranks.sha256` 36 files. Ranks 4 and 5 (copied 05:12Z to 05:36Z after each leg's document read PASS) equal that list on cpu-2; ranks 6 and 7 were copied 08:07Z to 08:30Z with `copy_rowdir.sh`, all 18 equal, before 2123's threads reach rank 4 (~09:36Z, when rank 2's GM-01 ends). GM-01 ran 89 to 99 min a rank (rank 2 to 09:27Z, rank 0 09:28Z to 10:57Z, rank 3 10:58Z to 12:27Z), and ranks 4 to 7 were reused (each fold ~1 s, GM-01 and compare kept). PASS at 13:22:29Z, the job rc 0 at 13:25:44Z: `vllm.match` r20261022-041131-656c (art:75545bcab80d99fd345347229077690895f0b4c7e604b9bd0844f7375b432f90), 33,056 s, every rank's fold, compare and GM-01 and the cross-rank leg (`tp2_rank_match.json`: count, order, op, segmentation, site, shard_linkage, producer, step, recorder, value, provenance, links and complete, 7,900 collectives a rank) PASS; `match_cpu.sha256` 87 files |
| Commit | waits for a GPU node: vy-glm53-eu is out of eu-north1 H200 capacity since 10:35Z (it holds this row's Build and legs; the 87 files of `match_cpu.sha256` go there from cpu-2 while B8 greedy's attempt 3 runs), or vy-glm53-h8 after its preflight and the whole row (~580 GB) from cpu-2; `MATCH_RUN` r20261022-041131-656c | vy-glm53-eu | not run: attempt 3 (below) rebuilt the row on #3652's splits |
| replay and publish | | vy-glm53-cpu-2 | not run, as the Commit |

B8 stochastic attempt 3 (circuits, 11:40Z 24 Oct) rebuilds the row on #3652's registered top-p splits (S = 32 on H200), merged
into this tree as d3a094a398. #3652 moves every step and request Program, so a trace comes first: r20261024-114706-e589
(EQUAL on every rank), then the re-pin 0b241a11a1, the row's tree. The Match GPU legs and the Commit run on RunPod
vy-glm53-h200-1, whose CUDA toolkit is 12.9.1 (installed by r20261024-150302-044c: vLLM's DeepGEMM JIT compiles with
`/usr/local/cuda`'s nvcc, and the pod's own 12.4 refuses `ld_st.cuh`'s 128-bit `q` operand); the Build pins cc 9.0 and 132 SMs,
not the toolkit or the driver (595.91.07 there).

Named gap: at S = 32, GLM-5.3's sampler, `GumbelTopPTokenSelect_v2{V=154880,S=32}`, is over Q_word_v1's default gate limit (24M),
and the Build of record runs with the row's allowance `VERITY_QWORD_MAX_GATES=GumbelTopPTokenSelect_v2=140000000` (and
`VERITY_QWORD_MAX_GATES_ALLOWED` the same, which the strict word check before the Commit reads). Its source: the 30 Sep option-3
ruling (raise the sampler's limit per row) and circuits' confirmation for this row at 15:10Z 24 Oct. 140M scales the 151936
vocabulary's 136M to V = 154880. The sampler call stays one unit, not provable in practice, until a chunked GumbelTopP split exists.

| step | run | node | state |
|---|---|---|---|
| Build of record | r20261024-130454-422b at 0b241a11a1 (`vllm.build` r20261024-130513-3a11) | vy-glm53-cpu-2 | FAIL rc 10 at 14:08Z: all 72 derives passed (rank 0's step 0801afbcf2b8b218), then the manifest's word check refused the sampler as too-large |
| Build of record, resumed | r20261024-145656-e91b at 0b241a11a1, `RESUME=1` (derives and composed workloads kept) with the allowance above | vy-glm53-cpu-2 | PASS at 15:33Z: `vllm.build` r20261024-145707-7573 (art:f187eb912167ba6625f542ddb47531b88a47bf8f01035cfdb16c2f47c6c6576e), manifest digest 6e1d2cfd…, complete; the allowance is a `qword.allowance` label on both runs and the art |
| Match GPU legs | r20261024-131914-76bd at 0b241a11a1 | vy-glm53-h200-1 | FAIL at 13:52Z, both legs at engine warm-up (the DeepGEMM JIT under nvcc 12.4); no capture, no coins; its row dir moved aside to `<row>.legs-cuda124-r20261024-131914-76bd` |
| Match GPU legs | r20261024-150302-044c at 0b241a11a1, after the toolkit and a tap rebuild with it | vy-glm53-h200-1 | PASS at 16:06Z: both legs rc 0 (capture 15:54Z, control 16:02Z), `vllm.match` r20261024-152827-6a6c (art:487919e1c2f643b1b31cca9a8d41c9fc81a58442b81c53ba37f99f9de5d22476); 341 GB, 1,192,898 files in `match.sha256`; nvcc 12.9.86 in `nvcc.txt` and `device_facts.txt` |
| the Build into the store | r20261024-154123-3460 (`store_tar.sh put`, NAME=build) | vy-glm53-cpu-2 | PASS at 15:50Z: 774 files, 2 parts, 25.3 GiB, parts art:8e0b581e8faaccff2984b17f2dad07f5f44a989c5ecf077f1403b0e490d210b1, every file checked against e91b's `rowdir.sha256` |
| the legs into the store | r20261024-154133-f0d3 (waits for 044c; `legs-small`, the files but the tensors, then `legs-full`) | vy-glm53-h200-1 | `legs-small` PASS at 16:29Z: 504 files, 3 parts, 47.7 GiB, parts art:ef44b8efa767c42098418d6a17f254ecc1123916b67d78c92ce437aab72e723a; then cancelled at 16:30Z: one stream from the pod to the store ran at 22-80 MB/s, so `legs-full` (341 GB again) would have landed after the pod's balance |
| the tensors into the store | r20261024-162504-a5e2 (`values-r0` .. `values-r7`, each rank's `capture/rank<r>/values`, four `store_tar.sh put`s at once) | vy-glm53-h200-1 | four streams 170-290 MB/s; r0-r3 PASS at 16:55Z (36.6 GiB each), r4-r7 running. With `legs-small` these are the whole legs |
| Match CPU phase, rank 4 | r20261024-154212-ddfc (`match_rank.sh`, the Build in place) | vy-glm53-cpu-2 (450 GB) | FAIL at 20:15Z (rc 1, set `ranks-4` labelled failed): fold PASS, compare and GM-01 FAIL, as on every rank (below); `vllm.match` r20261024-164419-f0c8 (art:3a1c84ce454ea4bf972a8b159647e11aa6e4324427f7693a7bcb8cbc325dfb41) |
| Match CPU phase, rank 5 | r20261024-154712-05a2 (`match_rank.sh`; attempt 2's row dir moved aside) | vy-nebius-cpu-3 (400 GB) | FAIL at 20:13Z (rc 1, set `ranks-5` labelled failed): fold PASS, compare and GM-01 FAIL, as on every rank (below); `vllm.match` r20261024-164807-7e15 (art:f6d7edfd5670412f94dc6df9db42e87d3fd29221981904e57c212780b261e3c0) |
| Match CPU phase, rank 6 | r20261024-154534-a841 (`match_rank.sh`) | vy-nebius-cpu-4 (380 GB, infra 15:25Z) | cancelled at about 20:35Z: its fold ran without the Build's construction (below) |
| Match CPU phase, rank 7 | r20261024-154554-7297 (`match_rank.sh`) | vy-nebius-cpu-5 (380 GB, infra 15:25Z) | cancelled at 16:46Z, no label: its first Build part's fetch read at 13-170 KB/s for 55 min over three connections (R2 sending slowly: rcv_rtt 475 ms against rtt 2 ms), while fresh processes on the node fetched it at 45-70 MB/s |
| Match CPU phase, rank 7 | r20261024-164708-d47e at 59249bbd54 (`LABEL_ALSO=r20261024-154554-7297`, which `da02`'s `RANK_SETS` names; `store_tar.sh get` refetches a part slower than 16 MB/s) | vy-nebius-cpu-5 (380 GB) | its parts prefetched into the node's store by r20261024-165043-9579 at 40-60 MB/s, after the same slow read on its second Build part; cancelled at about 20:35Z, as a841 |
| ranks 0-3, the full Match, the Commit | r20261024-154813-da02 (`pod_chain.sh`: waits for 044c, ranks 0-3 in `rows-side` three GM-01 at once, ranks 4-7's documents from the store, `MATCH_REUSE=legs`, then `commit.sh`) | vy-glm53-h200-1 (1300 GB) | the Build in the row dir at 16:20Z; ranks 0-3's folds from about 16:27Z, each PASS; ranks 0 and 3's compare and GM-01 FAIL by 19:40Z (below), so the chain ends at the Match with no Commit. Cancelled at about 20:35Z with r20261024-200936-de28, which was putting ranks 0-3's folds into the store as `side-folds` (f1a4 had failed at 19:15Z: its list sat at the path `store_tar.sh` writes its filtered list to, `$OUT/$NAME.sha256`, which emptied it). Those folds, like every fold of this attempt so far, ran without the Build's construction, so none is kept, and the pod's `rows-side` copy was removed |
| Match CPU phase, rank 4, at the fix | r20261024-205211-4005 at d7e2ff5532 (`match_rank.sh`, set `ranks-4`) | vy-nebius-cpu-4 (380 GB) | fold, compare and GM-01 PASS by 00:50Z 25 Oct, the first rank through them at #3897's fix; its set `ranks-4` put from 00:50Z |
| ranks 0-3 and 5-7, the full Match, the Commit, at the fix | r20261024-205656-8dd5 at d7e2ff5532 (`pod_chain.sh`, `RANKS=0,1,2,3,5,6,7`, `FOLD_WORKERS=7`, `RANK_SETS=r20261024-205211-4005:ranks-4`) | vy-glm53-h200-1 (1450 GB) | from 20:57Z; cancelled at about 21:30Z, when circuits terminated the pod (top's standing rule: CPU-only work on an H200 counts as idle). Nothing of it is kept |
| ranks 0-3 and 5-7's folds | r20261024-212829-de32 (0), r20261024-213149-7d26 (1), r20261024-213509-937d (2), r20261024-213829-6237 (3), r20261024-212504-d135 (5), r20261024-212616-04a4 (6), r20261024-212711-f94f (7) at 928cd29607 (`fold_rank.sh`, sets `fold-<r>`, each in its own sweep, removed after its put) | vy-nebius-cpu-5 (48 GB each) | FAIL at 23:45-23:55Z, each with ENOSPC in `persist_program`: `fold_rank.sh` called `tp.fold_match.fold_rank` with no `VERITY_INSTANCES_FORM`, so `match.py`'s library default (`runs`, `verity-gen/instances/v1`) wrote each rank's `instances.jsonl`, about 205 GB a rank against the 28 GB of the form of record (`progressions`, v2, which `row_tp` sets and 4005's fold wrote), and seven such sweeps filled the node's 2 TB. `fold_rank.sh` folds in the v2 form since a48bdaec96 (the fold digest is over the Program descriptor, so the form changes only the bytes stored). circuits freed the seven sweeps (1.3 TiB) with `research retention rm` |
| ranks 0, 1 and 2's compare and GM-01 | r20261024-224411-b6d5 (0), r20261024-224417-3a81 (1), r20261024-224424-0d01 (2) at e16afc195b (`match_full.sh` with `FULL=0` and one `OWN` rank each, through `slot.py --by circuits`; sets `ranks-<r>` on each) | vy-nebius-3, circuits slots c0, c1, c2 (380 GB each) | FAIL rc 5 at about 23:55Z, with ranks 3 and 5's and 4ace: their fold's get found the fold's set labelled failed |
| ranks 3 and 5's compare and GM-01 | r20261024-224443-5108 (3), r20261024-224504-3cfa (5), as above | vy-nebius-1, circuits slots c3, c5 (`--mem-gb 380`) | FAIL rc 5 likewise |
| rank 6's compare and GM-01 | r20261024-214603-a84b at 928cd29607 (`match_rank.sh` with `FOLD_PARTS`, set `ranks-6`) | vy-nebius-cpu-4 (380 GB) | withdrawn before admission once `fold-6` had failed, labelled `custody never-ran` |
| rank 7's compare and GM-01 | r20261024-232833-4ace at f681b95c3f, as a84b (`LABEL_ALSO=r20261024-214743-1664`, which 341b's `RANK_SETS` names) | vy-nebius-cpu-2 (380 GB) | from 23:28Z (circuits 23:20Z: DSV4V2's 224 GB run there had ended). r20261024-214743-1664, queued behind a84b on vy-nebius-cpu-4, withdrawn; r20261024-231432-f583 (from 23:14Z) ended by SIGTERM at 23:18:13Z, when the node was powered off from outside (logind: "Power key pressed short"; its deadline is 26 Oct) with every run on it, before it labelled anything. 4ace FAIL rc 5 as b6d5; its sweep (73 GB) freed by `research retention rm` |
| the full Match | r20261024-224959-341b at e16afc195b (`match_full.sh`: the Build and `legs-small` checked against 044c's `match.sha256` less its values, the eight ranks' sets, `MATCH_REUSE=legs`; puts `match-cpu` and labels `glm_v2.match_run`) | vy-nebius-cpu-5 (410 GB: `match_cpu.sh` sizes for one GM-01 leg even when it reuses the legs) | withdrawn before admission (`custody never-ran`): the node's disk was full |
| the restart: ranks 0-3 and 5-7's folds | r20261025-002320-6645 (0), r20261025-002341-d603 (1), r20261025-002348-f6a2 (2) on vy-nebius-cpu-5; r20261025-002603-3ea5 (3), r20261025-002716-16f7 (5) on vy-nebius-cpu-4; r20261025-004614-b7c9 (6) on vy-nebius-cpu-3; r20261025-010811-b9c2 (7) on vy-nebius-cpu-4, at 39a5ebc1b3 (`fold_rank.sh`: the v2 form, its own sweep `rows-fold<r>` kept as @circuits's at start and freed by `research retention rm` once `fold-<r>` is put) | 48 GB each | from 00:23Z 25 Oct (circuits 00:20Z), each admitted at once, but 16f7, queued at the head of vy-nebius-cpu-4's line until 00:53Z; b9c2 from 01:08Z (vy-nebius-cpu-5's memory queue was over-full) |
| the restart: ranks 0-3 and 5-7's compare and GM-01 | `match_rank.sh` with `FOLD_PARTS` and `CLEAN=1` (sweep `rows-gm<r>`, kept and freed as the folds' are), one a rank, launched by V2PREP's waiter once the rank's fold set is in or its fold has run 90 min | the first of vy-glm53-cpu-2, vy-nebius-cpu-4, -5 and -2 with 380 GB free beyond its memory queue, 380 GB plus 30 GB in MemAvailable (on vy-nebius-cpu-4 also what proofs' unledgered p0 may still grow by: 300 GB, its peak, less its tree's PSS, read at each look; nothing more once its slot is free with nobody in its line), (our ranks there, this one included) x 210 GB + 150 GB of disk free (circuits 00:20Z), and its `/etc/research/deadline`, if any, at least 2.5 h away; on vy-glm53-cpu-2 one of ours at a time and only once the rank's fold set is in; none on vy-nebius-cpu-3, proofs' pool, where b7c9 (launched before that rule, unslotted) finishes and any later fold of ours goes through its research slots (circuits 01:00Z) | every rank's fold, compare and GM-01 PASS, its set `ranks-<r>` put and both sweeps freed: 0 r20261025-024925-10c5, 1 r20261025-045327-f079, 3 r20261025-065036-ab58, 5 r20261025-085002-a420, 7 r20261025-105342-d81a and 6 r20261025-131931-cdf7 on vy-glm53-cpu-2 (infra moved its deadline to 16:00Z, then 19:00Z), 2 r20261025-051555-6263 on vy-nebius-cpu-4, the last set in at 15:25Z |
| the restart: the full Match | r20261025-152519-2397 at 39a5ebc1b3 (`match_full.sh` as 341b, its sweep `rows-full` kept as @circuits's) | vy-glm53-cpu-2 (410 GB; at 15:25Z vy-nebius-cpu-2 had 250 GB free beyond its queue and 280 GB of disk, against the 373 GB its sets take) | PASS at 17:18Z, 3240 s from 16:24Z: `vllm.match` r20261025-162425-bb4e (art:99a3c87fe1cc658368bbb6f5efec2dd57a1c4c2ea330edfaa95532b613d43928), the TP2 match with no mismatches, every rank's fold, compare and GM-01, and the cross-rank checks; `match-cpu` put by 18:37Z (87 files, 14 parts, 216.1 GiB, parts art:0d92b9a993b299695e8b9412b926d9c6d9cbdea73f5a9cb75cbb554e06b5b049) and `glm_v2.match_run` labelled, then `rows-full` (289.2 GiB) freed |
| the Commit | r20261025-204617-3850 (M1MAIN's `pod_commit_eu.sh`: `pod_commit.sh` with its device probe under `gpu-lease 8` and `ROW_ON=/workspace/extra`; `SETUP=0`, `MATCH_FULL=r20261025-152519-2397`, `MATCH_RUN=r20261025-162425-bb4e`, the Build r20261024-145707-7573 and the qword allowance): the checkpoint beside `build`, `legs-small`, `values-r0` .. `values-r7` and `match-cpu`, then `commit.sh`. Its first launch, r20261025-194113-fe7f from 19:41Z, failed before the GPUs, so before any commitment or draw: M1MAIN's download watchdog killed a stalled fetch. 3850 relaunched it with the same inputs | vy-glm53-eu (infra started it at 19:35Z, deadline 23:06Z, after M1MAIN freed greedy's and attempts 1 and 2's rows there) | PENDING by design, rc 0 at 22:20:23Z (run record art:1ae53c59969a16c405a82cb82eda482c5a9ab1a8bd1d7283242156d1df8267a2): its GPU run r20261025-211221-11e8 ended rc 0 at 21:49:50Z (2,243 s), the ranks agreed, and the driver and every rank read back 0.88. Bundles 8 x 27.3 GB, all preserved: rank 0 art:bf31ec184d68263a975cf2c0e25c81fddaa8fe00a1b24fb4a4a61402daa6c3d5, 1 art:0bd091fa13e7719fd133bef83dbb521d78e48ae996b3e61d9def5dc4fea32cd6, 2 art:87dd669863b309cfe8f5b681f236d6869cd0aaedc2b149e5ddb52976c85ab828, 3 art:7497f0f9399d718fdf2c528de37d7b28cfd3c753fb84123b59956a3b2cc79b81, 4 art:22921306a2e1dbfe2323f9e4db8154d0a0b19c60b1666e09c021e2fa6bf50717, 5 art:aabeb85bf1ca5e224896a7694fd232c11630dd870afe285204085147bbaabd5e, 6 art:56244af5c2c7789c6256abb7041a6dd7fd8f522bb9694ce6dc5a2608a192cc44, 7 art:421523de188b31b8c32f3dd8c4774f12c8453b2dcddd4d4589cd07b6291c7bbf; `_rowmeta` art:e606da4b35814cd4a684c83b9252d8f5892417d440fc2a57aa1e83277d9315b5 (114 files, 2.9 GB). Value check PASS on 8 of 8 ranks, the match oracle 7,900 of 7,900 tensors on each, cross-rank collectives 478 of 478 equal, 442,400 fold-match bindings with none bad, weights pin. Its CPU replay decided it PASS (next row). RunPod had no 8x H200 from 15:25Z (NO-STOCK on every create), and no pod was created |
| replay and publish | `replay_publish.sh` at 39a5ebc1b3 (circuits, 22:35Z 25 Oct), 190 GB a job from B8 greedy attempt 2's replay peaks, after the prep r20261025-231203-b5c5 (rc 0 at 02:23:49Z 26 Oct: the checkpoint staged and checked at `/workspace/cp/circuits/glm53-ckpt/aca966e4e02791568aa6a4ced368624b3d897f42`, 704 GiB; the Build of record from `build`'s parts, the earlier row dir moved aside; `_rowmeta` and the 8 bundles fetched): `RANKS=0,1,2,3,4,5,6,7` r20261026-003112-b8fc, then the publish r20261026-003703-723c | vy-nebius-cpu-4 | PASS. Every rank 138,705 of 138,705 sampled VUs equal with boundary linkage 120/120 (2,744 to 3,280 s, peak RSS 131 GiB); PASS decided on CPU by b8fc at 08:58:31Z 26 Oct from all 8 ranks' replays (rc 0 at 08:58:40Z): sampled replay 1,109,640 of 1,109,640 equal (`LlBf16SplitKGemm_v1`'s 27,000 VUs and `GumbelTopPTokenSelect_v2`'s 960 evaluated; the families with no evaluator are the collectives and `EmbeddingShard_v1`, which the cross-rank pass covers), boundary linkage 960 of 960, C2 population, query population, weights pin, over the Commit's own run root, per-rank roots and manifest digest (each replayed bundle's `MANIFEST.json` the sha256 the Commit's `bundles.json` names). Published by 723c (`records publish --tier heavy`, rc 0 at 09:00:58Z): record art:2def11ad48fa75bcebe6953021920d1c8f4878986c6efd5c6ee3bffa9002aa57, match record art:de21e03c8c3bbfb4814be5116638bcf05924703466ca49818807ea0f671e917b, commit record art:1ae53c59969a16c405a82cb82eda482c5a9ab1a8bd1d7283242156d1df8267a2; its `records.json` change (manifest digest 6e1d2cfd4d9085b6bd46ac06fb7d84b3033ffc6f246c0f723672809e9a16e9ee) is its `out/records_json.diff`, for the re-pin. The two earlier row dirs moved aside there (`<row>.prior-r20261024-205211-4005` and `<row>.prior-r20261025-231203-b5c5`, 214.5 GB) were removed with `research retention rm --approved-by @circuits` (CIR-110) |
| withdrawn or cancelled before they started | r20261024-215103-c7f2, r20261024-215243-1ab0, r20261024-215423-2693 (vy-nebius-cpu-5) and r20261024-214923-2cab (vy-nebius-cpu-4): ranks 0-3's GM-01, replaced by 77df and 39fc; then r20261024-221203-77df (vy-nebius-cpu-2, cancelled while it waited for `fold-0`), r20261024-214250-5e87 (vy-nebius-cpu-4) and r20261024-221304-39fc (vy-nebius-cpu-5, rank 2 and the full Match), replaced by the slot runs and 341b; r20261024-214743-1664 (vy-nebius-cpu-4, rank 7), replaced by 4ace; those that never ran labelled `custody never-ran` | | |
| ranks 0, 1 and 3, first try | r20261024-221021-09df at daff6a6b38 | vy-nebius-cpu-2 | FAIL at once (rc 5): the node had no venv for `store_tar.sh`, so `match_full.sh` now readies a node that lacks one |
| the pod's store copies | r20261024-165801-02e5 (after a5e2, `research data evict --target-free-gb 700`: every part above is preserved) | vy-glm53-h200-1 | |

The CPU phase is split over five nodes (circuits, 14:46Z and 15:10Z; vy-glm53-cpu-3 couldn't be created, so infra offered
vy-nebius-cpu-4 and -5 at 15:25Z), carried through the store rather than through an agent's VM: each node's inputs and outputs
are `store_tar.sh` sets, and each consumer waits for its producer's `store_tar.parts.<set>` label. A failed rank labels its set
failed, and the chain stops there with no Commit. Seven GM-01 legs ran at once (the pod three, each other node one), so ranks
0-3 on the pod set the length: a fold 2.1 h, then two rounds of GM-01 at 1.6 h. With the pod gone, the relaunch at the fix
runs on CPU nodes only. Off the pod, the Commit's inputs are the sets `build`, `legs-small` and
`values-r0` .. `values-r7` (their `get`s into one sweep give back the row dir and the legs file for file) and the ranks' sets. One `put` stream from
the pod to the store ran at 22-80 MB/s and four at once at 170-290 MB/s, so a large set goes in as several sets put together.

The Match failed on every rank the CPU phase judged (0 and 3 on the pod, 4 and 5 off it), the same way on both. Each rank's fold passed. The compare (X-09 `batch_decomp`) found none of the 8 requests canonical-equal to its Build component, with equal live and dead counts (r0: 1,154,616 and 38,034). GM-01 failed G3: every executed step of each request had one fold instance and one component instance unmatched, from its first step; G4 (516,368 identity-map problems) and G7 (10 boundary probes) followed from it. On rank 0 the first unmatched instance was request r0's sampler at step 0, where the Build's Program carries `GumbelTopPTokenSelect_v2{V=154880,S=32}` (`greedy-check-per-stage`). The cause is the TP fold driver: `tp.fold_match` ran each rank's fold without `--sampler-construction`, `--registered-values` or `--moe-construction`, which the single-rank driver passes. Each `fold_summary.json` records the sampler construction as None, and the fold bound the profile's default select, `GumbelTopPTokenSelectSharedGreedy_v1{V=154880}`. #3897 fixes it (on this tree as 9096c004bf and fede99cc68, which `tree_guard.sh`'s COMMIT_FIXES admits): each rank folds under the construction its Build dirs declare, `--reuse-fold` keeps only a fold made under it, and `--reuse-legs` keeps legs only over a kept fold. Rank 4, run whole on vy-nebius-cpu-4, passed its fold, compare and GM-01 at the fix. The other ranks' first folds failed for disk (the table), so they were restarted at 00:23Z 25 Oct: each rank's fold (about 44 GB) and then its compare and GM-01 leg (about 370 GB, which fetches the Build and the legs while its fold ends), each job in a sweep of its own, recorded as @circuits's and freed once its set is in the store, and placed only where `research mem` admits it at once and the disk holds 210 GB a rank of ours there plus 150 GB (circuits 00:20Z). vy-nebius-1 and -3 (stop at 03:00Z) take none; vy-glm53-cpu-2, once infra moved its deadline, took six GM-01 legs and the full Match. Every rank passed, and the full Match passed at 17:18Z 25 Oct. RunPod had no 8x H200 for the Commit, so it ran on vy-glm53-eu over the Match's pass, PENDING by design at 22:20Z 25 Oct; its CPU replay on vy-nebius-cpu-4 decided it PASS at 08:58Z 26 Oct, and it is published (the table).

DeepSeek-V4-Pro B1's Commit holds more host memory than `commit.sh`'s `--mem-gb 600` reserves: its cgroup's anonymous memory
rose through warm-up at up to 1.6 GiB/s and levelled off at about 800 GiB (about 100 GiB a rank) by 04:37Z, beside B8
stochastic's Match CPU phase (7676, about 300 GiB then), leaving eu about 450 GB. So nothing else is placed on eu beside it,
whatever its admission says. 7676's own anonymous memory peaks at 390 GiB in a rank's GM-01 (its 657 GiB peak counts page
cache), so attempt 3 fits beside it.

B1's replay host is vy-nebius-cpu-3 (the verified checkpoint): its Build of record and the Match's records (220 files and
`.tree` e2c986e7ba, without the per-rank fold outputs the replay does not read) were copied from cpu-2 at 08:40Z, all checked;
the Commit's `_rowmeta` brings the row records. `COMMIT_RUN` is the Commit's job (c049), whose `run_record` the publish cites:
the `vllm.commit` Attempt stores no run record. `BF16_ART` is T1R's bf16 v6 capture r20261021-014319-7d3f
(art:414f78ffe20a6cb27b61e33fc144f11ebf5680cb1007557802d0a532bbb59afc, `glm53_bf16_gemm_capture`).

DeepSeek-V4-Pro B1's replay runs on vy-glm53-eu, which holds everything it reads: the verified checkpoint
(`/workspace/cp/dsv4-ckpt`), the Build of record, the Match and the Commit's 8 bundles in place. vy-glm53-cpu-3 does not
exist and vy-glm53-cpu-2 holds B8 greedy's Match CPU phase. Two jobs side by side, `RANKS=0,1,2,3` r20261021-114850-5bfd and
`RANKS=4,5,6,7` r20261021-114900-5d55, at `--mem-gb 150` each (B1 GLM-5.3's replay peaks at 111 GiB a rank, and a DeepSeek-V4-Pro
rank's share of the checkpoint is about 14% larger), so 450 (7676) + 300 is the 750 for CPU work on eu. Its Commit job was
still preserving the 150 GiB `_rowmeta` (1,343,350 files in manifest order, about 60 a second, each a request the object store
takes about 0.37 s over; from 12:28Z a second pusher of the same art, r20261021-122822-7f8f, `research data push --jobs 192`,
left it mostly HEADs, 110 to 145 a second, to 14:50Z; the pusher itself ended rc 1 at 13:29Z, 325,513 uploaded and 1,017,834
present, with 3 HTTP 500s it left to the put; killing the put fails the Commit, rc 5), so the replay did not
wait for its `bundles.json`: the bundle lines the job had written, with `_rowmeta` art:22da9ae9481be9867f0d81f7a77a92bb75b16bb91b5d0b0cc22ff2181cd8969e,
the manifest the put had written to eu's store. The row dir is that `_rowmeta`'s source (hard links): its 145 records match the
manifest by sha256 and all 1,343,205 of attempt 2's files are there, so `.rowmeta-<hex>` was set in the row dir and both jobs
printed "already in" and every bundle "in place", with no 150 GiB fetch. Two earlier launches (r20261021-114748-d947,
r20261021-114800-b5cd) exited 2 at the input check, before anything ran: the file was sent under another name than `bundles.json`.
The decision is a FAIL, so nothing is published (`replay_publish.sh` exits with `tp-commit-replay`'s rc 3 before it publishes).

Attempt 5's replay was to run on vy-glm53-cpu-2 once e484 ended, not on eu's or vy-nebius-cpu-3's CPUs (circuits, 15:47Z and
16:01Z 21 Oct). cpu-2's `/workspace/cp/dsv4-ckpt` had a `.verified` from 17 Oct but none of the shards, so `stage_ckpt.sh` was
to stage the checkpoint there (864.7 GB, every file checked) once e484, which must not run out of disk, had ended. To make room, `research
retention rm --approved-by @circuits --ref CIR-25-v2-batch` deleted cpu-2's copy of the v1 rows at 16:33Z (`fs/glm53-m1`'s
`sweep/`, `sweep-v2/`, `records/` and `unit-rules/`, 319.4 GiB, logged in cpu-2's `deletions.jsonl`; eu's copy went at 00:42Z),
leaving 822 GiB free. The tool kept `fs/glm53-m1/work` (11 GiB), which holds retained passes. While e484 ran,
r20261021-173416-6183 (`ONLY='model-000[0-2]?-*'`) staged 29 of the 64 shards (390.6 GB at 1,508 MB/s, all checked, none bad),
leaving e484 452 GiB. The Build of record (ab0f's `rowdir.sha256`, 214 files, and the `.tree` at 1bbf831029) was copied from
vy-nebius-cpu-3 with `copy_rowdir.sh`, every file equal. The replay fetches the bundles and `_rowmeta` from the store.

Attempt 5 is not replayed (circuits, 18:52Z 21 Oct). TPEVAL's re-judge of attempt 3 at dsv4 fab2cefa87 (r20261021-164913-4f2b
and r20261021-164937-8ec8 on eu) found a Definition bug: `TritonBlockRMSNorm_v1`'s q-norm at N=1536 (rank 0, step 17,
`model.layers.6.attn/q` row 0: "F_V(committed inputs) != committed output"), fixed by #3111. So attempt 5's Programs are wrong
for those picks. DSV4V2 runs attempt 6 from a re-pin with #3111: a Build on vy-nebius-cpu-3, Match legs and Commit on eu, and the
Match CPU phase and replay on vy-nebius-cpu-3. cpu-2 is for the two B8 replays only, so the full stage (r20261021-183804-65d2) was
cancelled during its check, and the 64 shards both stages fetched were deleted, with their `.cache` metadata (only files from
17:34Z on), leaving cpu-2's `cp/dsv4-ckpt` as it was on 17 Oct and 1,339 GiB free.

Both Commits fail the cuBLAS route pin (`integrations/vllm/verity_vllm/check/cublas_route_pin.py`) on every rank. Their Programs bind Definitions only the pin
binds (B1 rank 0: `CublasSplitKGemmF32Out_v1` 19,200 and `CublasSplitKGemm_v1` 6,027), so each rank's facts of record must be
the pin's scan's (r20261017-191356-49a9, cuBLAS 120901). But each Commit's engine process had two cuBLAS libraries mapped, the
venv's `nvidia/cublas/lib/libcublas.so.12` that torch links and eu's system `/usr/local/cuda-13.0/.../libcublas.so.13.1.1.3`.
So `integrations/vllm/verity_vllm/check/replay/driver.py`'s probe (`_mapped_library`) refuses to pick one and records `cublas_version` as "unavailable:
LookupError: 2 libcublas.so* mapped", and `route_departures` counts an unavailable fact as a departure.

The second library comes from vLLM's served warm-up. On sm90 it runs `flashinfer_autotune`, whose autotune-cache load and save
(`AutoTuner.load_configs`, `save_configs`) record the environment with flashinfer's `_collect_metadata`, and its
`_get_cublas_version` loads cuBLAS with ctypes by the unversioned name `libcublas.so`. The venv has only `libcublas.so.12`, and
eu's `/etc/ld.so.conf.d` puts CUDA 13.0's library dir (on eu since 6 Oct) on the system search path, so the name gets 13.1 and
its cuBLASLt, on every rank, since each loads the leader's cache file. It only calls `cublasGetProperty` through its own handle:
no library on eu links a cuBLAS (`readelf -d` over the venv's JIT caches, `torch_extensions`, flashinfer's `cached_ops` and
`nc_build`), and torch's GEMMs run on the venv's 12.9.1. eu's five autotune cache files hold no tuned tactic (metadata only,
`cublas_version` 13.1.1), so correcting the version changes none.

The fix is `jobs/commit.sh`'s `cublas-names` dir (77c62e506e; dsv4-v2 17019d4282): first on `LD_LIBRARY_PATH`, its
`libcublas.so` and `libcublasLt.so` are the venv's `.so.12`, which the loader finds already mapped. The probe
r20261021-141321-af45 on eu (one H200, `commit.sh`'s environment) calls the served path after a torch GEMM. Without the dir,
four cuBLAS files are mapped, `cublas_versions()` is unavailable and `route_departures` names `cublas_version`. With it, only the
venv's two are mapped, both versions are 120901, and there is no departure. The facts are read in the serving process, so the Commits
before the fix (B1 attempt 1, DeepSeek-V4-Pro attempt 3) can't be re-decided: B1 reruns as attempt 2, and the B8 Commits run
at this head. A Commit launched on eu any other way (main's jobs included) still maps both; main's fix is VLLM-69 (vllm lane).

DeepSeek-V4-Pro B1's attempt 2 failed twice over, both read from `commit.log` and `commit/finalize_pair0.json`:

- The `moe_block_stream` watch item (`../dsv4_v2_tree/README.md` window B step 3) is the harmless pattern: the MoE block source
  attached on every rank, and each rank declares `moe_block_stream` bound-as-leaf with 23616 tensors, `at_calls` true and
  `hash_layers` [0, 1, 2]. Every rank's identity check, weights pin and token attribution pass (32 of 32 sampled tokens).
- The match oracle fails on all 8 ranks, a value disagreement. The committed TP partials and MoE outputs, read back through
  `open_range` against the run root (13728 reads, none failed), differ from the Match capture's (`match/capture`) in 5697 of
  5920 tensors a rank, and 959 MoE align witnesses a rank differ. At step 0, layer 0's MoE output and every partial up to it
  are equal on every rank. Layer 1's rank-local MoE partial (k 4) differs on ranks 2, 4 and 5 and is equal on the other five,
  so its all-reduced output differs in 12 of 3670016 bytes; the differences grow layer by layer (979504 bytes at layer 4), and
  from layer 10 `align.par` differs, a routing difference. At the hash-routed layers 0 to 2, where the routing cannot differ,
  `align.rank`, `align.start` and `align.blk` differ: `moe_align_block_size` placed the pairs in another atomicAdd order. The
  Match (`match/tp2_match.json`) compares structure, cross-rank provenance and tokens, never a second run's values, so this is
  the first comparison of two runs' MoE values on this row. The served Marlin MoE runs with `use_atomic_add=False`; which
  kernel makes the two runs differ is not established, and finding it is a debugging run.
- Then every rank raised IndexError in `tp2_commit_open_population` (`identity_of`, rank 0 at (10, 108414)). The TP worker's
  binding map indexed leaves in 256 B units against rows-sha512's 4 KiB rows: rank 0's draw is `model.layers.59.attn/kv` at
  stream offset 27753748, leaf 108414 in 256 B units, while step 10's stream holds about 7077 rows. That is a371c5a7f1's fault, above,
  and it fails every rows-sha512 TP Commit, GLM-5.3's included; it is independent of the values, since the oracle reads through
  the committer's own layouts.

DSV4V2 settled the divergence: the only difference is `moe_align_block_size`'s atomicAdd schedule, and each run's values
reproduce bit for bit under its own schedule. #3028 (822660cef7) records the capture's schedule and replays it in the Commit
wherever the top-k ids match, and it is carried into `cursor/dsv4-v2-21f3` at 582c1317f1 with #3020, both by blob in COMMIT_FIXES.
So DeepSeek-V4-Pro B1 runs a third attempt there (circuits, 06:48Z 21 Oct), after B1's Commit on the GPUs: its legs and Match
CPU phase above, then `commit.sh`, whose oracle names the differing tensor if one still differs (`../dsv4_v2_tree/README.md`).

B1's Build of record is on eu (copied from cpu-2 at 04:25Z, its 208 files checked and `.tree` e2c986e7ba): of `rowdir.sha256`,
the six row records the Match CPU phase rewrites (`stages.txt`, `row.log`, `effective_config.json`, `source_identity.json`,
`target_family.json`, `timeline.jsonl`) come with db82's `match_cpu.sha256`.

## GPU batch

Every script is in `jobs/` and runs from this tree's pushed head, `--cwd source` (publish: `--cwd clone`); every row dir is
`/workspace/cp/circuits/glm-v2/rows/<row>` on each node it visits, because the Commit's replay reads its Build and required
manifest at the paths the Commit recorded. `/workspace/fs` is not shared between vy-glm53-eu and the CPU nodes, so row dirs move
with `copy_rowdir.sh SRC DST LIST.sha256`: tar streams through the VM, every file then checked against the list the source job
wrote (T1's B1 legs: 104 GB, 1.1M files, about 11 min). Between nodes that can't reach each other, or off a node that may go
away, `store_tar.sh put|get` carries the same lists through the evidence store instead, in checked tar parts, with each
consumer waiting for its producer run's `store_tar.parts.<set>` label (B8 stochastic attempt 3: the Build at 25.3 GiB in about
6 min). A split Match CPU phase runs `match_rank.sh` on each CPU node and `pod_chain.sh` on the GPU node that holds the legs,
which joins the ranks' documents, runs the full Match and then the Commit; `wait_run.sh` lets a job launch before the run it
consumes on the same node ends. The GPU jobs are submitted by `gpu_batch.sh a [NODE]` and `gpu_batch.sh b ROWS.env [NODE]`;
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
runs at that commit or at a descendant that changes only `manifests/records.json` and this directory, or a file by exactly a
Commit-only fix the guard names (`COMMIT_FIXES`, the file's blob before and after: a371c5a7f1). The Build of record and
the Match GPU legs are each their node's first job, so a copy settles the tree: the destination ends with the older of the two,
and `copy_rowdir.sh` copies nothing (exit 2) unless the newer changes only those since.
Only the Build and publish read a row's pins, so a moved row's re-pin may land between its Match GPU legs and its Build of
record. A PR head merged after a row's first job reaches that row only by rerunning it; its later jobs are launched from a
worktree at a commit the guard accepts (`git worktree add /tmp/glm-v2-SHA SHA`). A row's Commit runs at a head that carries
the COMMIT_FIXES, since a commit before them fails its openings ("Window B").

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
   `match/` whose legs ended is refused). A row's capture also waits for its room on the row dir's disk (`DISK_NEED_GIB`, from the v1
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
   another; a ROWS.env line's `BUILD_RUN` and `MATCH_RUN` are the `vllm.build` and `vllm.match` Attempts, not their jobs): each row's Commit (`commit.sh`: rows-sha512 under `verity/tp-run-root/v2`, PENDING by design), in value order
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
