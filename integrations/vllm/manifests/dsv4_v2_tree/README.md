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
| #3020 (`cursor/verity-circuits-6f04`) | a371c5a7f1 | the binding map's leaf unit. Cherry-picked from `cursor/glm-v2-21f3`'s 15ea726d5b, not merged (a merge would bring main), and allowed by blob (COMMIT_FIXES) |
| #3028 (`cursor/moe-align-schedule-replay-6f04`) | a521e5ed60 | the Commit replays the Match capture's MoE align schedule. Cherry-picked, not merged, and allowed by blob (COMMIT_FIXES) |

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
  `census_entry`) leaves the request Build's census out (CPU gates, below). Its `MANIFEST_JOBS=8` and `BUILD_TO=14400` run the
  Build's required-value manifest in 8 processes under a 4 h timeout, with the unit rules kept in
  `/workspace/cp/circuits/dsv4-v2/unit-rule-cache`.
- `replay_publish.sh` reads the row's model from `model.sh` (repo, revision, `REPLAY_CK`: the verified checkpoint its replay
  reads) and cites each model's captures of record; `BF16_ART` is GLM-5.3's alone, and the Match's record is stored as
  `<model>/tp8-match-record/v1` (`glm53/` as before, `dsv4-pro/`).

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
then (`memory_map.of_build`). `census_entry` reads `VERITY_BUILD_CENSUS` from the environment, which the P7 lint
(`tests/lint/test_p07_declared_inputs.py`) refuses, so that test fails on this tree. Taking it as a `build` option would edit
`build.py`, one of the construction sources whose hash every artifact identity carries. The failure stays on this tree only: it
is never a PR, and on `main` CIR-73's census fix removes the need for the switch.

The Build's required-value manifest then cuts each Definition's unit rule (`word.prefetch`), and each 384-expert MoE GEMM's
takes about 45 min. At `build_record.sh`'s 2 processes the first two took the manifest's first 46 min, before the rest of the
word check and the component builds (GLM-5.3's took 56 and 29 min), against the Build's 2 h timeout. So r20261020-121943-5683
was cancelled in its manifest, and `model.sh` sets `MANIFEST_JOBS=8` and `BUILD_TO=14400`.

Every Call Definition the Programs derive has a Boolean version that keeps its boundary (the G1 probe). Of the 53 step Calls, 30
were checked deep, building each Boolean body. The other 23 are deferred, best effort and not a G1 gate, since circuit-check's suite
already checks those families: building `FlashMlaDecode_v2`'s Boolean body passed 424 GB (r20261020-083555-d46d, cancelled), so
they need a host with at least 755 GB, and run on vy-glm53-cpu-3 in a gap after the GLM-5.3 rows' CPU Match phases.

1. `trace_compare.sh` (`SKIP_AOT1=1`): the v2 trace, `predict compare` against it, and the AOT Build against it, on
   vy-nebius-cpu-3 after `stage_cpu.sh`. r20261020-071632-1bcf is the first run (cancelled in its request derives' census),
   whose 8 step derives seed r20261020-085740-90ff (`SEED_FROM`, `SKIP_AOT2=1`): its predict compare is EQUAL on all 24
   Programs, and its trace is art:b889027e16ceef9c2c8d625008f3360ca7b86dcf7d22b9f7d8ee2153dbf6e52b.
2. `repin.sh ROW RUN vy-nebius-cpu-3`, committed as "re-pin ROW": the step, request and workload Programs of all 8 ranks from
   the v2 trace, `records.json`'s `aot` switched to its comparison and trace art.
3. `build_record.sh` at that commit, or a descendant changing only this runbook (`tree_guard.sh`), with `AOT_TRACE=<the
   trace's row dir>` on the same node: the Build of record, with its manifest and `rowdir.sha256`. This is the row's first job,
   so it writes `<row>.tree`. r20261020-134932-4e9a's Build passed (digest 84c39fac2fc20187, 78 min) and stopped at the
   call-boundary check (exit 21): 92,928 of its 233,472 call boundaries had no claim, every one at a Marlin block port that
   `moe_source.PORTS` declares. The MoE source binds them on every TP rank but claimed none, so the call-boundary source, which
   serves one rank, was left with them. 1bbf831029 adds the claim (`moe_source.claims`, `taps.attached`), and the gate at that
   tree passes 4e9a's manifest with every call boundary claimed (r20261020-154201-a617). A code change moves the row's tree, so
   the Build of record is r20261020-154450-ab0f at 1bbf831029, with 4e9a's row dir under `attempts/`.

   ab0f passed in 29.5 min (1772 s), its largest process at 27.6 GB. Its Program digests and its manifest (648,552 identities,
   digest f32c76d659f58fd3) are 4e9a's. Its step manifest read 4e9a's unit rules from the cache, in 17 s against 2,893 s.
   Its call-boundary check claims all 233,472 call boundaries (sources `vocab_range`, `dsv4_attention`, `moe_align`, `moe`).
   - `BUILD_RUN`, its `vllm.build` Attempt: r20261020-154507-f208.
   - `BUILD_ART`: art:2122a7f8f506acce54471de447d4fd21101c6cc6bf7e00f5cc1625bb45001819.
   - `rowdir.sha256` lists 215 files and is in ab0f's `out/` (`research fetch r20261020-154450-ab0f --all` brings it to
     `~/.research/runs/r20261020-154450-ab0f/out/`).
   - The row dir is 1.76 GB (1,762,126,162 B), and r20261020-162800-6cbc checks the list against it.

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
- the Build of record: `copy_rowdir.sh vy-nebius-cpu-3 NODE ~/.research/runs/r20261020-154450-ab0f/out/rowdir.sha256
  /workspace/cp/circuits/dsv4-v2/rows` (CPU gates, step 3). Unlike GLM-5.3,
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
   (`BUILD_RUN=r20261020-154507-f208`, `--send match.sha256`), then `copy_rowdir.sh <CPU node> NODE match_cpu.sha256 ...`
   back to window B's node.
3. Window B: `MODEL=dsv4-pro gpu_batch.sh b ROWS.env NODE`, where ROWS.env is the line `<row> r20261020-154507-f208 MATCH_RUN`
   (`ROW BUILD_RUN MATCH_RUN`). It runs
   `commit.sh`: rows-sha512 under `verity/tp-run-root/v2`, REPLAY_DEFERRED=1, RETAIN=host, PAIRS=1, gpu_memory_utilization
   0.88 read back, with the bundles and `_rowmeta` preserved. A preemptible node may run it. An eviction fails the run, which
   stays published as failed, and the Commit reruns whole from a new commitment with new draws
   (`gpu_batch.sh b ROWS.env NODE <row>`); `commit_attempts.jsonl` counts the attempts.

   Watch in the first Commit: `moe_block_stream` among its required classes. The TP driver adds the class once the MoE source
   attaches on every rank (`pipeline/tp/commit.py`), although this manifest names no `moe_block_stream` identity. Here the
   source runs `at_calls` and commits the Marlin block's call boundaries instead, the ones its claim covers (1bbf831029, #2959).
   `required_classes_check` passes the class when every rank declares it with more than zero tensors.
   - Harmless: `commit.log` has `[tp2commit] MoE block source attached on every rank ({'layers': 61, ...}); required classes
     [..., 'moe_block_stream']`. Each `commit/declaration_rank<r>.json` declares `moe_block_stream` `bound-as-leaf`, with
     `at_calls` true, `hash_layers` [0, 1, 2] and `tensors` above 0. The rank's verdict passes, or fails for a reason unrelated
     to the MoE block.
   - Harmful: a rank's verdict (`commit.log`, `commit/summary.json`) fails for one of these reasons:
     - `required class 'moe_block_stream' not declared`: a rank attached no MoE source. The driver usually refuses first,
       with `the MoE block source attached on ranks [...] of 8`.
     - `class 'moe_block_stream' declared bound-as-leaf with tensors=0`: vLLM ran none of the launch sites the source wraps (another
       Marlin path), so the claimed call boundaries are unbound too.
     - `identity coverage: N of M required identities have no binding` or `identity check: ... step(s) differ`, first naming
       a `model.layers.K.ffn.experts/` Value: the source committed the block, but not at the ports the gate's claim assumes.

   Attempt 2, r20261021-041858-f624 (inner r20261021-041907-b7a8, verdict
   art:01daa488cb9b35fc400c983a2db7ae5d4155e06f8de2b91c6143e02df6a260f6), ran at 47ddb6246e against the Match
   r20261021-010646-2249.
   - `moe_block_stream` was harmless on all 8 ranks: `bound-as-leaf`, 23,616 tensors, `at_calls` true and `hash_layers`
     [0, 1, 2].
   - It failed on two bugs, both fixed here by blob (`../glm_v2_tree/jobs/tree_guard.sh` COMMIT_FIXES, which `copy_rowdir.sh` reads too):
     - The sampled openings raised IndexError in `identity_of`: the binding map's leaves were in 256 B units under rows-sha512's
       4 KiB rows. Fixed by #3020 (a371c5a7f1, cherry-picked from 15ea726d5b).
     - The match oracle failed on every rank: 5,697 of 5,920 committed tensors differ from the capture's.
       `moe_align_block_size` hands out each expert's slots in atomicAdd order, and a pair's block decides the Marlin expert
       GEMM's split-K cuts. So the low bits differ from layer 1 at step 0 and drift, until the learned routing from layer 10
       on differs too. Fixed by #3028 (a521e5ed60): the capture records each align call's schedule
       (`match/capture/align_schedule.rank<r>.pt`), and the Commit replays it.
   - The CPU experiment r20261021-061937-9868 evaluated layer 1's MoE rows (after r20261021-055246-b82b) on ranks 2, 4 and 5,
     whose partials differ:
     - The routing is equal. Under its committed schedule, every committed word reproduces.
     - The capture's differing row (one per rank) reproduces bit for bit with its one expert-365 pair in another of the
       expert's blocks, and the committed block does not.

   Attempt 3 needs a new capture, since the old one has no schedule file and so is not replayed. At this tree's head:
   1. move the attempt's `match/` aside (`mv $D/match $D/match.attempt2-r20261021-010646-2249`: a rename, not a delete, since
      `match_gpu.sh` refuses a `match/` whose legs ended);
   2. run window A's `match_gpu.sh` and the CPU phase's `match_cpu.sh` (steps 1 and 2);
   3. run window B's `commit.sh`.

   In the Commit's `commit/summary.json`, each rank's `match_oracle.<r>.schedule_replay` must be non-null, with `missing` 0,
   and the oracle must PASS. A null means no schedule was replayed. A nonzero `routing_differs` means a call routed otherwise
   than the capture's, and its first [step, module] is named in `routing_differs_first`.
4. CPU, on the replay host (below): `replay_publish.sh` per `RANKS` job, then one without `RANKS` that publishes, as GLM-5.3's
   rows run it (`../glm_v2_tree/README.md` step 5): `tp-commit-replay` of every rank from the bundles window B preserved, then
   `records publish --tier heavy`. It needs from the GPU legs:
   - window B's `$OUT/bundles.json` (`--send` it), naming each rank's bundle and the `_rowmeta` tree, and `COMMIT_RUN`, the
     Commit's run, whose run record the record cites;
   - the Match's `match/tp2_match.json`, `match/fold_match/tp2_fold_match.json` and `match/fold_match/tp2_rank_match.json` in the
     row dir (step 2's `match_cpu.sh`), and `MATCH_RUN`;
   - `BUILD_ART`, the Build of record's artifact (its `vllm.build` run's, art:2122a7f8f506acce…), whose construction manifests
     `due` reads.

   It cites nine of DeepSeek-V4-Pro's ten capture properties, preserved from their GO runs (13–14 Oct) with `records capture`,
   and names `dsv4_moe_capture` a gap: its capture r20261013-205619-5c39 ran with `VLLM_BATCH_INVARIANT=1`, where vLLM's default
   is 0. The other nine set no vLLM switch away from its default, but none reads the switches back as GLM-5.3's #2838 captures
   do. The capture map (`deepseek-v4-eager-hopper-tp8-v0`) names no Program's captures, so publish requires none of the ten
   (`records.due` is empty). Naming them in the map changes `manifests/capture-maps/`, which `tree_guard.sh` lets no later job of
   this row carry, so that waits for the next tree.

   Dry run, without GPU output: the job's nine citations pass `check_capture`, its evidence passes `evidence_of`, the row is no
   `index_refusal`, its tier is heavy and its `aot` is r20261020-085740-90ff's. `due` against the Build of record's artifact
   (16 construction manifests), as against the v2 trace's Build before it, is empty under that map. In the Build of record's
   row dir (r20261020-162800-6cbc), every rank's two derived Programs are `aot` (publish refuses a switched row whose Build
   traced one). `config_record.record` finds the Build ok on all 8 ranks, its stage PASS with manifest f32c76d659f58fd3, and its
   outcome "word check not run" until the Match and Commit are in.

The replay host. The Commit's replay of record evaluates its picks with the word row kernels (`check.replay.evaluate`'s
`ROW_EVALUATORS` is empty on every record), so it builds no Boolean body. `boolean-replay`, which would, is a single-device
re-verification (`row_dir/build_request`, no TP weights) and takes no TP8 bundle, so it is not run for this row. A rank's replay
holds its share of the checkpoint, opened against the committed weights root, beside a forked pool fitted to the job's memory
(`driver.fit_pool`). GLM-5.3 B8's peaked at 257–259 GiB a rank with 32 workers, 66–71 min a rank with 9 min of it opening the
weights (r20261019-123645-e69d). DeepSeek-V4-Pro's checkpoint is 805 GiB against GLM-5.3's 704 GiB, so a rank's share is about
14% larger, and its replay is estimated at 270–300 GiB a rank. That is not measured: the first job runs `RANKS=0` alone at
`--mem-gb 330` and records its peak (`sampled_replay_p0_rank0.json`'s `peak_rss_bytes`), and the rest are packed by it. A job
replays its ranks in turn, so the jobs at once are the host's memory over that peak:

| host | memory | jobs at once | 8 ranks |
|---|---|---|---|
| vy-glm53-cpu-3 (cpu-d3, being brought up by infra) | 1 TiB, else 768 GB | 3, else 2 | ~3.5 h, else ~4.5 h |
| vy-glm53-cpu-2, after its disk restart | 755 GiB | 2 | ~4.5 h |
| vy-nebius-cpu-3 | 497 GB (/user.slice) | 1 | ~9 h |

The plan is vy-glm53-cpu-3, else vy-glm53-cpu-2. Either needs the checkpoint staged and verified at `REPLAY_CK`,
`/workspace/cp/dsv4-ckpt`: `stage_ckpt.sh` with `PY=/workspace/cp/circuits/venv312/bin/python` (a CPU node's venv), 864.7 GB in
about 18 min, with 865 GB free. It also needs the row dir: the Build of record (`copy_rowdir.sh vy-nebius-cpu-3 HOST
~/.research/runs/r20261020-154450-ab0f/out/rowdir.sha256 /workspace/cp/circuits/dsv4-v2/rows`) and the Match's files.

GPU time (wall of the GPU job, all 8 GPUs):

| job | wall | measured by |
|---|---|---|
| checkpoint staging (no GPU) | ~18 min | r20261020-072311-c012's rate, scaled to 864.7 GB |
| Match GPU legs B1 | ~23 min | v1 r20261018-023527-3dc1: match 1379 s, the check included |
| Commit B1 | ~1 h | GLM-5.3 B1's deferred one-pair Commit r20261019-074856-7c3a (58 min). v1's DSV4 Commit, 3 pairs replayed inline, took 3.5 h (12586 s) |
