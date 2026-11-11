# Commitment costs

What committing values costs under each commitment scheme, on the host and on the GPU:

- `commit_cost.py`: each named scheme (`frame-v3`, its SHA-256, BLAKE3 and SHA-512 row and word variants, `vllm-v1` and
  `vllm-v1-sha512`) hashing N leaves of B bytes and building its tree, through the plain-Python references on one core, leaf
  and tree phases timed apart; one opening per scheme is checked through its `verify_path`.
- `hiding_leaf_cost.py`: per row, the hm96 hiding leaves against unsalted ones: stored and hashed bytes, compressions,
  in-circuit AND and XOR gates, and single-core host time.
- `serving_commit_cost.py`: a cell's `commit.seconds`, the serving hook committing the first `--n` instances of an input set
  afresh (`frame-v3-sha512` with `hm96-sha512/v1` row leaves), with two rows recomputed by core as its validation.

`tool.py` declares all three as research tools (`commit_cost`, `hiding_leaf_cost`, `serving_commit_cost`), so their runs
are keyed by their parameters:

~~~text
research run --tool commit_cost --project verity -- python benchmarks/commitments/commit_cost.py --scheme all --leaves 1048576
~~~

There is no `run.py` or `run.sh`, so `verity bench commitments` does not apply. Each script reads only the tree (and, for
`serving_commit_cost.py`, an input set by path or `art:` id) and writes `result.json` (`research/result/v0.1`) to `--out`, else
`$RESEARCH_RUN_DIR`.

Status: in use. The suite (`tests/`) runs `commit_cost.py` and `hiding_leaf_cost.py` small, holds their
roots to the references' and their results to the result schema, and loads the `commit_cost` tool by name.
