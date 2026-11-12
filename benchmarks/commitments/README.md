# Commitment costs

What committing values costs under each commitment scheme, on the host and on the GPU:

- `commit_cost.py`: each named scheme (`frame-v3`, its SHA-256, BLAKE3 and SHA-512 row and word variants, `vllm-v1` and
  `vllm-v1-sha512`) hashing N leaves of B bytes and building its tree, through the plain-Python references on one core, leaf
  and tree phases timed apart; one opening per scheme is checked through its `verify_path`.
- `hiding_leaf_cost.py`: per row, the hm96 hiding leaves against unsalted ones: stored and hashed bytes, compressions,
  in-circuit AND and XOR gates, and single-core host time.
- `serving_commit_cost.py`: a cell's `commit.seconds`, the serving hook committing the first `--n` instances of an input set
  afresh (`frame-v3-sha512` with `hm96-sha512/v1` row leaves), with two rows recomputed by core as its validation.
- `step_root_cost.py`: the serving loop's per-step row-table hashing for one int8 decode step of Llama-3.1-8B, at several worker
  counts, against a SHA-512 floor over the same bytes.
- `step_root_gpu.py`: the same hashing on the GPU, overlapped with a graphed proxy of the decode step's linears.
- `table_read_commit.py`: the same committer over one decode step's DSA table reads (each read's opened row, salt, path and
  output row, per layer) at a batch of long-row requests: GB/s alone, beside a decode-shaped GEMM loop, and the CPU twin's;
  and the committer's history tables those reads open (creation, appends, paths) as `history.HistoryTable`, the device
  tables' CPU twin and on the GPU. Its arms run the cheapest first (`--arms` picks some), and `table_read_commit.json` is
  rewritten after each one with the arms still to run, so a run that is stopped keeps what it measured.

`tool.py` declares the first three as research tools (`commit_cost`, `hiding_leaf_cost`, `serving_commit_cost`), so their runs
are keyed by their parameters:

~~~text
research run --tool commit_cost --project verity -- python benchmarks/commitments/commit_cost.py --scheme all --leaves 1048576
~~~

The others run as plain `research run` jobs (each file's docstring has its command; `step_root_gpu.py` and
`table_read_commit.py`'s GPU arms need a GPU).
There is no `run.py` or `run.sh`, so `verity bench commitments` does not apply. Each script reads only the tree (and, for
`serving_commit_cost.py`, an input set by path or `art:` id) and writes `result.json` (`research/result/v0.1`) to `--out`, else
`$RESEARCH_RUN_DIR`.

Status: in use. The suite (`tests/`) runs `commit_cost.py`, `hiding_leaf_cost.py` and `step_root_cost.py` small, holds their
roots to the references' and their results to the result schema, and loads the `commit_cost` tool by name; it stops
`table_read_commit.py` after one arm and reads that arm's result from disk.
