# The Boolean Build's cost against the word Build's

`run.py` takes one row's Calls as their word specializations (`ROWS`: Qwen3-0.6B's B1 / 256-token prompt / 32-token output row,
`qwen3-06b`, or the `small` case), and their Boolean versions (`program.boolean.boolean_version`, as the Boolean Build's lift
calls them). For each IR it measures:

- `descriptor`: the descriptor of one Program calling every specialization once, which writes each Definition once, as a
  row's does. It records the bytes the Program digest hashes (`codec.descriptor_size`), the digest, and the largest Definitions
  and families.
- `rules`: the `Q_word_v1{16,32}` unit rule of every Definition under them, scheduled as the manifest's prefetch schedules them
  (`verity.ml.definitions.word.prefetch`, by depth, largest first), timed per Definition in its worker. It records the verdict and
  a digest of each rule.

The root body of a row's Program, with its KV-cache references, is not in this Program. `rules-<ir>.jsonl` gets one line per
rule as it finishes.

A Boolean head is tens of millions of bit gates, so run the `rules` part on a node:

~~~text
research run --on auto --cpus 16 --mem-gb 160 --max-min 150 -- bash -lc \
  'uv sync -q --all-packages --extra torch-cpu && uv run benchmarks/bool_build_cost/run.py --row qwen3-06b --jobs 16 --out $OUT'
~~~

`--t 1,52,65` keeps only the attention specializations of those key lengths. `--part descriptor` is light enough for a VM.
