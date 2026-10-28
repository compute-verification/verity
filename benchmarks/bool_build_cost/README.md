# The Boolean Build's cost against the word Build's

`run.py` takes one workload's Calls as their word specializations (`ROWS`: Qwen3-0.6B's B1 / 256-token prompt / 32-token output
workload,
`qwen3-06b`, or the `small` case), and their Boolean versions (`verity.ml.definitions.boolean.boolean_version`, as the Boolean Build's lift
calls them). For each of the two, word and Boolean (`--ir`), it measures:

- `descriptor`: the descriptor of one Program calling every specialization once, which writes each Definition once, as a
  workload's does. It records the bytes the Program digest hashes (`codec.descriptor_size`), the digest, and the largest Definitions
  and families.
- `rules`: the `Q_word_v1{16,32}` proof-block rule of every Definition under them, scheduled as the manifest's prefetch schedules them
  (`verity.ml.definitions.word.prefetch`, by depth, largest first), timed per Definition in its worker. It records the verdict and
  a digest of each rule.

The root body of a workload's Program, with its KV-cache references, is not in this Program. `rules-<ir>.jsonl` gets one line per
rule as it finishes.

A Boolean head is tens of millions of bit gates, so run the `rules` part on a machine with 16 cores and 160 GB:

~~~text
uv sync --all-packages --extra torch-cpu
uv run benchmarks/bool_build_cost/run.py --row qwen3-06b --jobs 16 --out DIR
~~~

`--t 1,52,65` keeps only the attention specializations of those key lengths. `--part descriptor` is light enough for a VM.
