# The hash loop: served commit overhead, iteration by iteration

Each iteration is one commit on the loop branch with a one-line hypothesis, measured the same way on three fixed GPUs (H200 on
`vy-dc-h200`, B200 on `vy-dc-b200`, RTX PRO 6000 on `vy-nebius-2`), so that a plot of iteration against each measure shows which
ideas moved the cost of committing every value of served vLLM. The measurement is circuits' `benchmarks/served_commit/prof.sh`,
unmodified: Qwen3-8B, B32, vLLM compiled, 4 prefill steps of 2,560 tokens and 64 decode steps, `stock`, `stock-prof` and the
commit arms interleaved rep by rep (3 reps) under nsys, each commit arm gated token for token against `stock` first.

One iteration, end to end, from a clean checkout of the loop branch with the iteration committed and pushed:

~~~sh
bash benchmarks/hash/loop/iterate.sh N "one-line hypothesis"
~~~

It launches `job.sh` on the three machines through `research run --on M --project verity --campaign hash-loop --source . --cwd
source`, waits for them, and labels each run and its result (`label.py`): campaign=hash-loop, sweep=iteration, row=N,
label=hypothesis, source=branch@commit, hardware=GPU. A run's result (`result.json`, published as one `bench-result/v1`) holds every
arm's measures, each measurement naming its `arm` (stock, sha512, blake3). The verdict on an iteration is a label too:
`python3 benchmarks/hash/loop/label.py RUN... --row N --hypothesis "..." --outcome kept|dropped --reason "..."`.

- `job.sh`: the workload on one GPU: the model check, then `prof.sh` under `gpu-lease 1 --wait` (the node's lease flags, timeouts
  inside the lease), the SM clocks read before and after, then `collect.py`.
- `collect.py`: the run's measures per arm (its docstring defines them): TPOT and throughput overhead against stock with their 95%
  intervals (the headline), prefill and decode step time, the model's own GPU time per step and its rise over stock, hash GPU time
  per step, and the noise (stock's TPOT spread over its interleaved reps).
- `label.py`: the labels, idempotent; rerun it on runs whose launching VM stopped waiting.
- `plot.py`: iteration against each measure, one line per (GPU, arm), from the evidence store only
  (`research data select --label campaign=hash-loop --kind bench-result/v1`, after `research data refresh`, since each run
  publishes its result from its node); `--table` prints the iteration table.

~~~sh
OUT=/tmp/loop uv run --with matplotlib python3 benchmarks/hash/loop/plot.py --table
~~~
