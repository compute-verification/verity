# The PoUW bench harness (sm_120)

One harness for every PoUW prover arm on the RTX PRO 6000 Blackwell (sm_120), FP8 and FP4 streams alike: each arm and each
plain GEMM baseline is gated at the exact shape and flag set it is timed at, then everything is timed the same way,
interleaved, and reported over the fastest plain GEMM of the arm's precision on the same card and shape.

It is independent of the serving stack: stdlib, numpy and ctypes (no torch), with a small native library for the device side.
Its timing discipline: bursts over distinct operand sets past twice the L2, a read-only L2 flush, per-item sensors,
versions, the consistency gate. It uses the CUDA primary context through ctypes, so driver-API arms plug in.

## Try it

Plan the shapes' operand sets, bursts and memory, with no GPU:

~~~sh
cd benchmarks/pouw/harness
python bench.py --plan
~~~

It prints one line a shape, from `m2048-n2048-k2048` to the models' linears, with its sets, burst and memory.

## Running it

The runner is `run.sh JOB [ARGS]`, which `verity bench pouw harness JOB [ARGS]` runs: it starts one job as it is, with
ARGS and the environment. JOB is `build` (`native/build.sh`), `pin_target` (`jobs/pin_target.sh`), or `bench`,
`characterize`, `verify`, `ledger`, `price_twins` or `ieee_pin`, the Python script of that name, under `"$PY"` where
`jobs/env.sh` has set it and under `python3` else. The commands below start the same scripts by their paths.

Build (no GPU needed; nvcc 12.9, about 7 GB of memory per CUTLASS 3.x unit):

```bash
export POUW_HARNESS_LIB="$(benchmarks/pouw/harness/native/build.sh)"   # prints the library's path, nothing else
```

The build lands in `native/build/<key>/` (`POUW_HARNESS_BUILD` moves the cache), keyed by the sources, nvcc's version and the
cuBLAS headers', with `build.json` (the key, versions, the library's hash, each kernel's SASS MMA counts); an existing build is
reused, and so is each object whose unit, headers and flags are unchanged. A run refuses a library its `build.json` doesn't
describe. Where there is no CUDA 12.9 (the RTX PRO 6000 node has 13.0), `. jobs/toolkit.sh` first assembles it from
NVIDIA's pinned redistributable archives and the cuBLASLt wheel `env.sh` pins. `LT2_HOME` names a second cuBLAS install
(that node's `/usr/local/cuda`, whose `libcublasLt.so.13` is 13.1.1.3): `lt.cu` is built alone against it with the host
compiler into `libpouw_lt2.so`, which a run loads beside the library (`RTLD_LOCAL`; `POUW_HARNESS_LT2` names another, or
`none`), so every family also gets that cuBLASLt's candidates, named `lt<major>_…`. On that node the build is a CPU job,
at `nice 19` on the fill queue's cores:

```bash
research run --on MACHINE --project verity --source . --cwd source -- bash -c \
  '. benchmarks/pouw/harness/jobs/toolkit.sh && export POUW_HARNESS_BUILD=/workspace/pouw/harness/build JOBS=8 LT2_HOME=/usr/local/cuda &&
   L="$(nice -n 19 taskset -c 96-191 benchmarks/pouw/harness/native/build.sh)" &&
   cp "${L%/*}"/{libpouw_harness.so,libpouw_lt2.so,build.json} "$RESEARCH_RUN_DIR/"'
```

`build.json` also records each kernel's registers, stack and local memory (`resources`; a spill shows as local bytes).

Before a shape's reps, the same interleaved schedule runs as a warm-up (kept apart, in `warmup_items`) for at least
`--warmup-s` of device time and until a round's median SM clock is within 0.5% of the round before (`--warmup-max-s` bounds
it; the shape's `warmup` says whether it got steady). Every entry's item is sized to the same device time, at least
`--item-ms` (40): calls for a baseline or an arm, replays for a chain, from the warm-up's first round (so its later rounds
run the timed items) and again from its last (`sizing`). Every item samples NVML while the GPU
runs it (`sensors.during`), and each panel result carries both sides' reps in the same run (`panel.reps`: per-call and item ms
and SM clock of the arm's and the baseline's items, rep by rep), `sm_clock_arm` and `sm_clock_base` for `panel.py`, and
`clock_gap` (medians more than 0.5% apart).

Every reference verifier an arm names today is a replay: it opens the run's rows and words and recomputes the scheme in the
clear. So an accepted result is replay-verified (development shortcut; the protocol's check is a ZK proof, PouwCheck). The
records and `panel.py`'s arguments keep the name `verifier`.

[`DESIGN.md`](DESIGN.md) has how to run it on a GPU and as a timed run, the verify step, the decode chain, the SASS gate
and the reference verifier an arm names, the poisoned transcripts, the ledger and the exit codes. Each module's docstring
says what the module is.
