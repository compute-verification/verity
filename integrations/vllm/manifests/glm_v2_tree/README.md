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

Base `origin/main` @ c0ebeb8b93, then, in this order:

| PR | head | carries |
|---|---|---|
| #2729 | f2629dde38 | Boolean-first; #2738 eead9401a9, #2694 f031a376f4, #2765 dec1238773, #2828 85d2659299, #2848 aaccf5e3d7 |
| #2836 | e2ef5e7ba1 | m1-rows (B8 multi-request path); #2831 3f13f1e1f2, #2828 85d2659299 |
| #2837 | d0fc072205 | a row of record done only at its public cap |
| #2838 | 615e13bdc3 | served-default capture harnesses (bf16 v6); #2828 85d2659299 |
| #2852 | ba1aa4b2d2 | TP served windows refused unless the run root is SHA-512; #2842 9ef0bf1d46 |
| #2824 | 5b5840cf7f | the committer's rows-sha512 iterations |
| `cursor/tp-sha512-21f3` (CIR-63 part 1, no PR yet) | d86c42a576 | TP Commits at rows-sha512 under the SHA-512 TP run root (`verity/tp-run-root/v2`); #2852 ba1aa4b2d2, #2824 5b5840cf7f |
| `cursor/boolv-21f3` (CIR-47, no PR yet) | a326a9ce23 | main c0ebeb8b93 merged with #2729's head: CIR-47's Boolean versions are #2729's σ-lifts and on-demand constants, so it changes nothing past #2729 |
| `cursor/rows64-land-21f3` (no PR yet) | 1d8869d1e4 | `MoeSum_v3` (`MoeSum_v1` on bits as a batch of `MoeSum_v2{TOPK,64}`), A11's Boolean-first probes |
| `cursor/multimem-land-21f3` (no PR yet) | 7338aa92c8 | `AllReduceMultimem_v1` as a batch of 16-byte vectors (`AllReduceMultimemVector_v2`) |

ROWS64 and the multimem batch joined when ASK-61 was provisionally accepted (Daniel, 19 Oct 21:59Z, option A: public
power-of-two output classes up to 262,144 bits). Two conflicts, both sides kept:

- `integrations/vllm/verity_vllm/pipeline/serving_rows.py` `after_commit` (TPSHA512 against #2848): the row-table layout's `row_tables`, and the
  provenance's `commitment` read from the committer's scheme.
- `integrations/vllm/tests/predict/test_predict.py`, GLM-5.3 B8 top-p's LP25/T6 request (multimem against the realized-arrival pin): this tree's
  predictor gives `b632583a…`, the realized-arrival pin; the vector form does not move that Program.

## To merge when their PR opens

- The branches above re-merge at their PR's head once it opens.
- CIR-44: the two Call splits. No branch yet.
- The static MoE frontend switch for GLM's expert GEMMs. The ops landed in #2665; no switch commit is on any branch.
  GLM's FP8 MoE in this tree is already the unpermuted pair set: per (token, slot) pair, the expert id is a value operand
  and the slab is gathered inside the unit (`MoeExpertFp8Gemm(W)_v1`, the `indexed-read` construction).

## Not in v2

- T8b's prefix commits (`cursor/prefix-tp-6f04`): ASK-42 is provisionally A (the prefix length is private), so a prefix row
  in a batch is proved cold and T8b's C-public commits stay out.

## GPU batch

Every script is in `jobs/` and runs from this tree's pushed head, `--cwd source` (publish: `--cwd clone`); every row dir is
`/workspace/cp/circuits/glm-v2/rows/<row>` on each node it visits, because the Commit's replay reads its Build and required
manifest at the paths the Commit recorded. `/workspace/fs` is not shared between vy-glm53-eu and the CPU nodes, so row dirs move
with `copy_rowdir.sh SRC DST LIST.sha256`: tar streams through the VM, every file then checked against the list the source job
wrote (T1's B1 legs: 104 GB, 1.1M files, about 11 min). The GPU jobs are submitted by `gpu_batch.sh a` and `gpu_batch.sh b ROWS.env`;
each waits for its GPUs on eu's lease, so a window runs back to back. eu is held only for a window: the lease, and the node, are
released where a CPU phase sits between (CIR-51).

A row is built, served and judged at one tree, the one its record cites (`records publish --tree` is not checked against the
stage runs): its first job writes `<row>.tree` (`cursor/glm-v2-21f3@SHA`) beside the row dir, the copies carry it, and every
later job of the row refuses another tree. Once the branch moves past a row's tree, that row's later jobs are launched from a
worktree at it (`git worktree add /tmp/glm-v2-SHA SHA`); a PR head merged after window A starts reaches a row only by rerunning it.

1. CPU, before window B (any order, overlapping window A): `build_record.sh` per row on vy-glm53-cpu-2 (the AOT Build of record:
   predict emit refused unless equal to the row's `aot.trace` Build, with its manifest), `AOT_TRACE=<the v2 trace>` for a row whose
   rehearsal moved a Program (that row is re-pinned with its comparison first).
2. Window A, eu (`gpu_batch.sh a`): the four rows' Match GPU legs (`match_gpu.sh`). They need no Build: the capture's
   `--build-rank` registers a Build's post-load tensors, and GLM-5.3 has no model-level post-load hook (`loader_parity`). T1R takes
   the bf16 v6 capture (`BF16_ART`, from `verity-vllm records capture RUN`) and the G2 point (`benchmarks/served_commit/tp_decode.sh`,
   `ARM=rows-4k`: rows-sha512's 4 KiB rows, stock against Commit in one TP8 engine, TTFT, prefill and decode TPOT per batch 1, 8, 32)
   on eu first; `gpu_batch.sh t1r` (`cap_bf16_v6.sh`, then the G2 point) runs them only if those did not hold. Stop eu.
3. CPU: each row's legs copied eu -> the Build's node (its `match.sha256`), then `match_cpu.sh` (MATCH_PHASE=cpu, `--build-run`
   the Build of record, `--send match.sha256`); then the row dir (its `rowdir.sha256` and the Match's files) copied to eu.
4. Window B, eu (`gpu_batch.sh b ROWS.env`): each row's Commit (`commit.sh`: rows-sha512 under `verity/tp-run-root/v2`,
   REPLAY_DEFERRED=1 RETAIN=host PAIRS=1, PENDING by design), its bundles and row records preserved in the store. Stop eu.
5. CPU, vy-nebius-cpu-3 (the verified checkpoint): the Build of record copied in, then `replay_publish.sh` per row (`--send
   bundles.json`): `tp-commit-replay` of every rank, then `records publish --tier heavy` with the nine captures of record cited
   by id (#2838's eight served-default captures, which this tree's `check_capture` accepts, and `BF16_ART`).

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

Window A (the four legs) is about 1.8 node-hours and window B about 8.6, so about 10.4 node-hours (83 GPU-hours), of which
6.5 are estimates (B32's two jobs, B8 stochastic's Commit); T1R's capture and G2 point add about 0.3 if `t1r` runs them. The
measured Commits ran under vllm-v1; rows-sha512's cost at TP8 is what the G2 point measures.
