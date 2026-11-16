# verity_vllm.pipeline.tp

The adapter's stages when several GPUs serve one model together (tensor parallelism, where each GPU is a rank). A
workload served on one GPU is captured, matched, committed and replayed by the pipeline's stages; this package runs
each of those stages on every rank and holds the ranks to each other, through the collectives they share.

The stages are commands of `verity-vllm` whose names start with `tp-`, such as `tp-match` and `tp-commit`;
`verity-vllm --help` lists them. `pipeline/row_tp.py` runs them in order for a workload of any world size from two up.

The package's docstring says what it holds, and [`DESIGN.md`](DESIGN.md) lists which names other code imports from
each module.
