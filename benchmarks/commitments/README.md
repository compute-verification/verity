# Commitment costs

Before anything is proved, the developer commits to every value a proof may need, while it serves. That commitment is
paid on every value, whether or not an audit later samples it. This benchmark measures what it costs under each
commitment scheme, on the host and on the GPU, from hashing leaves and building trees to the serving loop's per-step
hashing of a decode step's rows.

## Try it

~~~sh
uv run python benchmarks/commitments/commit_cost.py --scheme frame-v3-sha512 --leaves 4096 --out DIR
~~~

This hashes 4,096 leaves of 256 bytes under frame-v3 with SHA-512, builds the tree, and prints one JSON line: the leaf
and tree phases' seconds, timed apart, and whether one opening verified. It runs on one core in well under a second and
writes `result.json` to `DIR`.

## The scripts

| Script | What it measures |
| --- | --- |
| `commit_cost.py` | each named scheme hashing N leaves and building its tree, through the plain-Python references on one core |
| `hiding_leaf_cost.py` | hiding leaves against unsalted ones, per row: bytes, compressions, in-circuit gates and host time |
| `serving_commit_cost.py` | the serving hook committing an input set afresh |
| `interior_commit_peak.py` | the most memory an interior's commitment holds |
| `step_root_cost.py` | the serving loop's per-step row-table hashing for one int8 decode step of Llama-3.1-8B, against a SHA-512 floor |
| `step_root_gpu.py` | the same hashing on the GPU, beside a proxy of the decode step's linears |
| `table_read_commit.py` | the committer over one decode step's table reads and their history tables, alone, beside a GEMM loop, and on the CPU |

Each script's docstring has its command; `step_root_gpu.py` and `table_read_commit.py`'s GPU arms need a GPU. The first
three are research tools (`tool.py`), so their runs are keyed by their parameters:

~~~text
research run --tool commit_cost --project verity -- python benchmarks/commitments/commit_cost.py --scheme all --leaves 1048576
~~~

There is no `run.py` or `run.sh`, so `verity bench commitments` does not apply.

Status: in use. Its suite runs `commit_cost.py`, `hiding_leaf_cost.py`, `step_root_cost.py` and `interior_commit_peak.py`
small and holds their roots to the references'.
