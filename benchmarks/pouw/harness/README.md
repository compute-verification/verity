# The PoUW bench harness (sm_120)

One harness for every PoUW prover arm on the RTX PRO 6000 Blackwell (sm_120), FP8 and FP4 streams alike: each arm and each
plain GEMM baseline is gated at the exact shape and flag set it is timed at, then everything is timed the same way,
interleaved, and reported over the fastest plain GEMM of the arm's precision on the same card and shape.

It is independent of the NCP stack: stdlib, numpy and ctypes (no torch), with a small native library for the device side.
It takes #475's timing discipline (bursts over distinct operand sets past twice the L2, a read-only L2 flush, per-item
sensors, versions, the consistency gate) and #449's primary-context ctypes pattern, so driver-API arms plug in.

| File | What |
|---|---|
| `arm.py` | the arm interface (v0.x): what an arm provides and what the harness checks; the contract is published in the project store, `internal/pouw/rtx-pro/workers/6-harness.md` |
| `bench.py` | the run: identity, fill and derive preflights, per shape gates (negative controls at the first shape), the baselines' autotune, interleaved timing, the consistency gate, the report |
| `chain.py`, `derive.py` | decode as a dependent chain (`--decode-method dependent-chain`, the default): call i+1's A is derived on the device from call i's y, for the plain GEMM and for arms alike; an arm's decode split into before, during and after the GEMM, with and without a side stream; `derive.py` is the derive's bit-exact numpy twin, which gates every chain |
| `characterize.py` | the card: identity, tensor-core peak rates per MMA kind, DRAM and L2 bandwidth, operand-entropy control, sustained clocks under a held GEMM; the lock-clock sweep only when asked (`--sections lock`), never where someone else fixes the clocks (node 2) |
| `baselines.py` | the baseline families (BF16, FP8 per-tensor and rowwise, MXFP8, FP8 1×128/128×128, NVFP4, MXFP4, int8): cuBLASLt's top heuristic algorithms and CUTLASS's sm_120 kernels, and the bit-exact gate on ternary operands |
| `example_arm.py` | an example arm, **not a PoUW scheme** (plain cuBLASLt FP8 and a stand-in read of D): the smoke test of the arm path and a template |
| `schedule.py`, `report.py`, `shapes.py`, `identity.py`, `fill.py` | the rules as pure functions (tested on CPU in `../tests/test_harness.py`) |
| `native/` | `libpouw_harness.so`: fills, flushes, events, graphs, cuBLASLt, the CUTLASS kernels (`build.sh`, CUTLASS pinned at v4.8.0) |

Build (no GPU needed; nvcc 12.9, about 7 GB of memory per CUTLASS 3.x unit):

```bash
export POUW_HARNESS_LIB="$(benchmarks/pouw/harness/native/build.sh)"   # prints the library's path, nothing else
```

The build lands in `native/build/<key>/` (`POUW_HARNESS_BUILD` moves the cache), keyed by the sources, nvcc's version and the
cuBLAS headers', with `build.json` (the key, versions, the library's hash, each kernel's SASS MMA counts); an existing build is
reused, and so is each object whose unit, headers and flags are unchanged. A run refuses a library its `build.json` doesn't
describe. Where there is no CUDA 12.9 (node 2 has 13.0), `. jobs/toolkit.sh` first assembles it from NVIDIA's pinned
redistributable archives and the cuBLASLt wheel `env.sh` pins. On node 2 the build is a CPU job, at `nice 19` on the fill
queue's cores:

```bash
research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source -- bash -c \
  '. benchmarks/pouw/harness/jobs/toolkit.sh && export POUW_HARNESS_BUILD=/workspace/pouw/harness/build JOBS=8 &&
   L="$(nice -n 19 taskset -c 96-191 benchmarks/pouw/harness/native/build.sh)" && cp "${L%/*}"/{libpouw_harness.so,build.json} "$RESEARCH_RUN_DIR/"'
```

Run on your GPU only (`CUDA_VISIBLE_DEVICES` one index; the UUID from `server.md`'s table):

```bash
cd benchmarks/pouw/harness
python bench.py --plan                                              # sets, bursts, memory per shape; no GPU
CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID python bench.py --gpu 5 --expect-uuid GPU-... \
    --shapes headline --arm example_arm:ExampleFp8 --out run.json
CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID python characterize.py --gpu 5 --expect-uuid GPU-... --out char.json
```

Where there is no CUDA 12.8+ toolkit, or it isn't the build's (node 2 has 13.0), `jobs/env.sh BUILD` gives the run NVIDIA's
cuBLASLt and runtime wheels, a library built elsewhere (a build directory, or `art:<id>` of one preserved with
`research data put --kind pouw-harness-build/v1 --tree DIR --preserve`), and this tree's `verity` and `verity_pouw` on
`PYTHONPATH`. On node 2, `gpu-lease 1` picks the GPU, so the run names it `--gpu lease` (the one index the lease set, checked
against `server.md`'s GPU table); a timed run holds the node with `gpu-lease 8 --wait` and times on its first GPU:

```bash
R='research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source --env GPU_LEASE_WHO=<your bc-id>
   --send BUILD/libpouw_harness.so --send BUILD/build.json --send GPUS.md'          # GPUS.md: server.md (its GPU table)
H='. benchmarks/pouw/harness/jobs/env.sh "$RESEARCH_RUN_DIR/inputs" && "$PY" benchmarks/pouw/harness/bench.py --gpu lease
   --server-md "$RESEARCH_RUN_DIR/inputs/GPUS.md" --clock-label locked-2100 --shapes headline
   --arm-path benchmarks/pouw/harness --arm example_arm:ExampleFp8 --out "$RESEARCH_RUN_DIR/bench.json"'
$R -- gpu-lease 1 --wait -- bash -c "$H"                                                           # untimed: gates, autotune
$R -- gpu-lease 8 --wait -- timeout 1140 bash -c 'export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} && '"$H"  # timed
```

An arm on another branch runs from a tree that holds both: its branch with this directory and `../tests/test_harness.py`
checked out from the harness's commit, `--arm-path` its own directory. `jobs/harness.yaml` and `jobs/harness-timed.yaml` are
node 2's queue jobs, for when its Kueue is up.

Decode rows count on the panel only as a dependent chain, which needs a library with the derive kernel (`ph_derive`);
`--decode-method independent` times the old independent graph with any library. An arm's calls expose `phases` (before, gemm,
after) to get the split; `arm.py` has the contract.

Exit codes: 0 ok; 2 identity or usage; 3 a device, build or harness error; 5 a known-bad variant wasn't rejected; 6 the
consistency gate rejected an arm. The JSON is written in every case, with every rep.
