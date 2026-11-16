# commit_gpu: hm96-sha512 rows in frame-v3-sha512 trees on the GPU

While serving, the developer commits every value a proof may later need, and the commitment has to keep up with the
GPU that produced the values. `commit_gpu` is a CUDA committer whose levels, root and openings equal
`verity.commitments`' bit for bit: `hm96.SHA512` hiding leaves over `sha512/row/v2` or `sha512/row-seg/v1` row
digests, in a frame-v3-sha512 tree. This is a benchmark: it measures the committer on the e2e slice's o_proj operand
(8192 x 4096 int8) and on tile leaves of a prefill chunk's operand, beside cuBLASLt's int8 GEMM at the same shape.
`commit_gpu.cuh`'s header has the construction.

## Try it

The reference side runs on the CPU. Prepare the quick check's cases, every case's inputs and the reference's levels,
b || c and leaves, the way a GPU run does before its lease:

~~~sh
PYTHONPATH=. python3 verity/commitments/cuda/commit_gpu/reference.py prepare /tmp/commit_gpu quick
~~~

It prints one JSON line a case (mode, leaves, row bits, salts) in a few seconds. On a GPU, `reference.py check` then
holds a build's output to these cases.

On an RTX PRO 6000 node (MACHINE):

~~~text
research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  [--env BUILDS="a0r0l a0r0 a1r0 a0r3"] -- bash verity/commitments/cuda/commit_gpu/gpu.sh
~~~

`BUILDS` defaults to `a0r0l`, the fastest form; the other tags, which `gpu.sh`'s header lists, rebuild forms to time
beside it.

The job writes under the run's `$OUT`: `rc.txt` (each step's exit code, the builds that were timed and the fastest of
them), `commit.txt`, `nproc.txt` and `gpu.txt` (the commit, the host's cores and the leased GPU), `build-TAG.log` a
build, `prepare.jsonl` (the reference's cases, a line each), `selftest.jsonl` (a line a build) and `check-TAG.jsonl`
(the bit-for-bit check, a line a case and one for the build), and the measurements, a JSON line each: `sha-TAG.jsonl`
(the raw compression rate), `bench-e2e-TAG.jsonl`, `bench-sweep-TAG.jsonl` and `bench-seg-TAG.jsonl` (the two
leaf-size sweeps) and `bench-tile-TAG.jsonl` (the tile sweep). Beside them are `smi.csv` (the SM clock, power and
temperature every half second), `nsys*` (an nsys trace's report of the kernels' launch shapes) and the steps' standard
error (`*.err`). It writes no `result.json`. Its builds, cases and parameter files are in a directory under `/dev/shm`,
which it removes. Its last lines are `summary.py`'s, one a measurement.

