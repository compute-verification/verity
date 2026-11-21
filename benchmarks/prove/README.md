# prove

`verity bench prove` is the one command for what a proof system's prover costs on a workload: per block type, the blocks
proved per served token and the prover's ms per block, then the prover GPUs a serving GPU needs, from the evidence
store's labelled records. It reads records and proves nothing. The rules and the table are
`benchmarks/harness/verity_bench/prove*.py`'s; this directory holds the runner.

~~~text
verity bench prove --system flock --workload glm53      every block proved
verity bench prove --system spot --workload glm53       Flock at the workload's sampling plan of record
    [--provers N] [--cores N] [--rows N] [--json] [--local-only]
verity bench prove lean-verify --lean F --proof TREE    the Lean verifier timed on one kept proof
~~~

## Try it

~~~sh
uv run verity bench prove --system flock --workload glm53 --local-only
~~~

Prints the inputs (each a labelled record of the store), then a row per block type with its figure and the run it is
from, then the totals, marked measured, modeled or partial. It needs `research` and the store, no GPU and no node. Under
a `research run`, `--system spot` also writes the headline metrics of `metrics.toml`.
