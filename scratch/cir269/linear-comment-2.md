Written Fri 07:11 AM PST [15:11Z]. CATALOGHAND's return: I stopped at your third stop condition, because shard 1/4's peak passed 56 GB.

**Records pass at #9660 `1333bd53f693`** (`r20261113-142348-fa99`, step 1) [run]: 886 catalog bindings built, 0 of them held a record that did not count, 0 reused, 886 records stored (38,720,051 bytes) in 1,578.1 s. It gave no reason line. The tracer saw everything, under the handed functions.

**Base `8854b8249777`** (`r20261113-142346-aa58`, declared 48 GB, `--mem-gb 40`, peak 33.9 GiB) [run]: 886 built, 886 stored, in 1,437.1 s.

**Comparison** [run]:
- No binding's record at the head reads fewer entries than at the base, of any kind, and none lost a read.
- New at the head: the reads of `registry.catalog` and its new sidecar.
- Every changed digest is one of three things: circuit-check's own code; `catalog.py` or `call_scope.py`; or an import site that resolves to a changed file (vLLM's sidecar to `program_view`, `boolean` to `catalog`, `word` to `call_scope`).
- Not compared: 8 of 887 manifest rows. Four of Flock's `TEMPLATE_PARAMS` bindings repeat under one name, and my dump kept one row a name.

**Shard 1/4, step 2** (`--jobs 8 --mem-gb 48`, declared 64 GB, 16 cores) [run]:
- Its targets were forked from a process of 1.75 to 5.7 GiB, and it printed no `warn circuit-check holds` line.
- Two targets `heavy.json` doesn't list peaked at 36.0 GiB, about 30.7 GiB past their fork: `DsaTopKReadRows_v1{N=131072,TOPK=2048,BLOCK=16384,READ=EmbeddingRead_v1{V=131072,H=576}}` and the same with `N=40960`.
- Each was counted at 4 GiB, so both ran at once. At 15:02:04Z two processes held 34.2 GiB each, and the bound reached 85.3 GiB.
- I cancelled it about 20 minutes into step 2, at 72 targets, 12 of them with one failure each: 8 `AllReduceMnnvl_v1`, 3 `MoeExpertFp8Gemm*`, 1 `LogSoftmaxNllBackwardRow_v1`. No report was written, so I don't know which check failed, or whether `main` fails them too.
- Shards 2/4 to 4/4 and the combine were not run.

**Over 16 GiB and not in `heavy.json`, from a thin parent:** those two `DsaTopKReadRows_v1` targets, among 72 targets [run]. `Dsv4SwaWindow_v2{SW=128,BS=64,NB=16384}` (9.06 GiB) and `DsaPositionsMerge_v1{M=8,K=2048}` (5.42 GiB) grow past 4 GiB unlisted too.

**To go on under 64 GB** [inferred]: either list the two in `heavy.json` at about 31 GiB, so the scheduler runs them alone (a data edit to circuit-check, yours to allow), or run the shard that holds them at `--jobs 1`.

#9660's body has your two lines. Both PRs are unmarked, with no labels, and nothing was restacked or merged.
