# gate4-h100 lane -- HANDOFF

Branch `lane/gate4-h100` (worktree `/Users/danielreuter/projects/verity-gate4`), pod `vy-g4` (H100 80GB, id uktx6i8dejuunk,
ssh helper `/tmp/g4.sh` on the laptop = `ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p 11461 root@64.247.201.61`).
Ledger `backends/numerical/reports/ledger/gate4-h100.jsonl`, track `shared`.  Deliverable: `backends/numerical/reports/GATE4_H100.md/.json`; raw results under `backends/numerical/reports/gate4/`.

## Pod state

* `/workspace/venv312` (torch 2.6 cu124, Triton 3.2); `cupy-cuda12x` + `blake3` were NOT installed on this pod (the brief said
  they were) -- installed by this lane's setup step.
* `/workspace/gate4/src` = `git archive HEAD` of this branch (tar-synced); `/workspace/gate4/setup.log` = deps + instance build.
* `/workspace/bench-instances/v1` = `verity_numerical.bench.instances build --procs 48` (persistent; every prover takes `--root`).
* BabyBear export for the a-gpu prover: `/workspace/bb/pos4096`, `/workspace/bb/neg` (built by this lane; see step 1).

## Rows: measured / next

(kept current after every row; see the table in GATE4_H100.md)

| row | status | run id |
|---|---|---|
| A end-to-end (a-gpu v2e, main 38757ed) | DONE: 5.02 s median of 5 warm (1.23 ms/VU, 1.24e8), verified 4096/4096; cold run r20260922-090931-b5be (rep0 316 s compile) | r20260922-091718-45c3 |
| A end-to-end (a-gpu2 head f87b096) | FAILED at B=4096: `logup_packed._walk_fs: LogUp: fractional sum is not zero (a query is not in its table)` -- the lane's packed LogUp (tested at 64 VUs) does not prove the 4096 batch; not fixed here (measurement lane) | r20260922-091947-822f |
| A end-to-end (a-gpu2 pre-graph merge 179d509) | FAILED, same error in `logup_packed._walk` | r20260922-092252-3ebd |
| A buckets v1 / v2 (a-fusion `bench_fused` 2^26/2^28 k=3 + `logup_graph_bench` v1 v2 level/round + v2 concurrent) | running | r20260922-092450-5b79 |
| A buckets v1 / v2 (a-fusion kernels) | pending | |
| B end-to-end non-ZK / HVZK (b-ligero2) | pending | |
| B buckets v1 / v2 (encode) | pending | |
| SP1, C | cited only | -- |

## Next steps

1. instance build -> bb_export -> a-gpu prove (reps 3) as a recorded run.
2. a-fusion `bench_fused` + `logup_graph_bench` re-run.
3. b-ligero2 bench-vu non-ZK and `--zk`.
4. encode `buckets_bench` + `v2_bench`.
