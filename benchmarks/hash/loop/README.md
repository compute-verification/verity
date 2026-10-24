# The hash loop: served commit overhead, iteration by iteration

Each iteration is one commit on the loop branch with a one-line hypothesis, measured the same way on three fixed GPUs (H200 on
`vy-dc-h200`, B200 on `vy-dc-b200`, RTX PRO 6000 on `vy-nebius-2`), so that a plot of iteration against each measure shows which
ideas moved the cost of committing every value of served vLLM. The measurement is circuits' `benchmarks/served_commit/prof.sh`,
unmodified: Qwen3-8B, B32, vLLM compiled, 4 prefill steps of 2,560 tokens and 64 decode steps, `stock`, `stock-prof` and the
commit arms interleaved rep by rep (3 reps) under nsys, each commit arm gated token for token against `stock` first. On the H200
every arm runs under Qwen3-8B's Build of record (`PIN_BUILD`: r20261018-062957-da83's `out/build.json`, artifact digest 3bd5dcb1,
seeded into one fresh Inductor dir and checked after). The RTX PRO 6000 (cc 12.0) runs pinned under circuits' Qwen3-8B Build once
`iterate.sh`'s `PIN_ART_RTX` names it; the B200 (cc 10.0) is on the compiled Builds' refusal list, so its rows are unpinned, and the
table says which is which.

One iteration, end to end, from a clean checkout of the loop branch with the iteration committed and pushed:

~~~sh
bash benchmarks/hash/loop/iterate.sh N "one-line hypothesis"
~~~

It launches `job.sh` on the three machines through `research run --on M --project verity --campaign hash-loop --source . --cwd
source`, waits for them, and labels each run and its result (`label.py`): campaign=hash-loop, sweep=iteration, row=N,
arm=workload (qwen3-8b), label=hypothesis, source=branch@commit, hardware=GPU. A run's result (`result.json`, published as one `bench-result/v1`) holds every
arm's measures, each measurement naming its `arm` (stock, sha512, blake3, sha512-layer: circuits' per-layer acquisition at 4 KiB rows, no scheme's roots). The verdict on an iteration is a label too:
`python3 benchmarks/hash/loop/label.py RUN... --row N --hypothesis "..." --outcome kept|dropped --reason "..."`.

- `job.sh`: the workload on one GPU: the model check, then in the lease the committer's GPU byte-identity tests (among them
  `test_rows_cuda.py`'s: a learning step's two-half leaves, `rows_h` then `rows_frame`, are `rows_leaves`' bytes) once in each rows-sha512 leaf form (`VERITY_ROWS_FORM=fused` and `split`, `native_rows.cu`; the timed arms run the device's
  own: fused on sm_120, split elsewhere; a failure is no timing), `prof.sh` under `gpu-lease 1 --wait` (the node's lease flags, timeouts inside the lease), the SM
  clocks read before and after, then `collect.py`.
- The BLAKE3 sibling arm (`served-commit-rows-blake3`, `arms.py`): circuits' rows arm under the committer's research scheme
  `rows-blake3` (`verity_vllm.commit.step_rows_b3`, `native_rows_b3.cu`: the same step, keyed BLAKE3 in every hash position;
  it never serves, and stays off main under the one-hash rule, so the arm runs only from a branch that carries it and its checks
  (the scheme and its kernels against the hash lane's hm96-blake3 reference, `benchmarks/hash/intpipe/hm96_ref.py`), with `ARMS`
  naming it and `CHECK_TESTS` adding the checks). `job.sh` names it to a copy of `prof.sh` with its one case line added.
- What-if arm (a) (`served-commit-rows-whatif-q`, `arms.py`): the rows arm with the post-RoPE q out of the Commit, as if q norm
  and RoPE ran in the attention unit's prologue; k, v and qkv stay committed. Its roots are no scheme's and it never serves; run
  it in its own labeled run (`SWEEP=what-if ARMS=stock,stock-prof,served-commit-rows,served-commit-rows-whatif-q
  ARM_NAMES=served-commit-rows-whatif-q=whatif-q bash iterate.sh what-if-a "..."`), so its failure costs no iteration.
- `collect.py`: the run's measures per arm (its docstring defines them): TPOT and throughput overhead against stock with their 95%
  intervals (the headline), prefill and decode step time, the model's own GPU time per step and its rise over stock, hash GPU time
  per step, the noise (stock's TPOT spread over its interleaved reps), and the first profiled run's wall-time rise and bytes
  copied: the timed runs replay its requests and hit its plans, so only that run takes the learning path every live step takes.
- `overlap.py`: from a run's `served.nsys-rep`, each of the model's kernels (and the gather) timed alone and beside the hash
  kernels: the contention, kernel by kernel.
- `kernels.py`: from a run's `served.nsys-rep`, one arm's commit kernels in one phase: launches a step, mean time, grid, stream, and
  when each starts against the step's model kernels, with the GPU's idle time between them.
- `label.py`: the labels, idempotent; rerun it on runs whose launching VM stopped waiting.
- `glm_rows.py`: a GLM-5.3 TP8 row (workload `glm-5.3`, hardware `8x H200`) from one of circuits' served runs (WS-D's
  `summary.json` and `runs.jsonl` at B32), as the loop's result doc, put as `bench-result/v1` and labeled with `label.py
  --workload glm-5.3 --result-art`; iteration 0 is circuits' r20261018-194044-414f (rows-sha512) and -195722-b284 (rows-layer-4k).
- `plot.py`: iteration against TPOT and throughput overhead (`--measures` for more), one row of panels per workload, one line per
  (GPU, workload:arm, served-pin state), pinned and unpinned never averaged together (pinned markers filled, unpinned hollow),
  from the evidence store only (`research data select --label campaign=hash-loop --kind bench-result/v1`, after `research data
  refresh`, since each run publishes its result from its node); a maxtput workload's throughput panel plots its loss against
  its card's native reference run (`native_loss_pct`, which maxtput's `row.py` publishes); a result labeled
  `kernel_outcome=voided` or `superseded_by` is not plotted; `--table` prints the iteration table, with the same-cycle
  overhead and the loss against native side by side.

~~~sh
OUT=/tmp/loop uv run --with matplotlib python3 benchmarks/hash/loop/plot.py --table
~~~
