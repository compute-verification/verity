# verity_vllm.check.match

The Match's checks. The adapter's Build declares a workload Program: what vLLM is meant to execute. The Match records
what vLLM actually executed and folds that record into the same form. These checks hold the fold to the Program it
should equal.

It is four commands of `verity-vllm`, and the pipeline's stages load it: `global-match` (a batched execution
against the workload Program), `batch-decomp` (a batch of more than one request, checked one request at a time),
`program-compare` (two Programs, structurally) and `program-dag` (a derived Program and a fold). `verity-vllm COMMAND
--help` gives each one's options. Under tensor parallelism it also checks that the ranks' folds state the same
collectives, and replays a sample of the collectives' outputs from the ranks' committed partials.

Each module's docstring says what it holds. [`DESIGN.md`](DESIGN.md) says what the package loads, what loads it, and
which names other code imports from each module.
