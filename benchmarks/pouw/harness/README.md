# The PoUW bench harness (sm_120)

One harness for every PoUW prover arm on the RTX PRO 6000 Blackwell (sm_120), FP8 and FP4 streams alike: each arm and each
plain GEMM baseline is gated at the exact shape and flag set it is timed at, then everything is timed the same way,
interleaved, and reported over the fastest plain GEMM of the arm's precision on the same card and shape.

It is independent of the NCP stack: stdlib, numpy and ctypes (no torch), with a small native library for the device side.
It takes #475's timing discipline (bursts over distinct operand sets past twice the L2, a read-only L2 flush, per-item
sensors, versions, the consistency gate) and #449's primary-context ctypes pattern, so driver-API arms plug in.

| File | What |
|---|---|
| `arm.py` | the arm interface (v0.x): what an arm provides and what the harness checks, the contract in code |
| `bench.py` | the run: identity, fill and derive preflights, per shape gates (negative controls at the first shape), the baselines' autotune (its fastest `--finalists` per family and each library's fastest go on to the timing, and at decode each gets a chain), a warm-up to steady state, interleaved timing in items of equal device time, the consistency gate, the report |
| `chain.py`, `derive.py` | decode as a dependent chain (`--decode-method dependent-chain`, the default): call i+1's A is derived on the device from call i's y, for the plain GEMM and for arms alike; an arm's decode split into before, during and after the GEMM, with and without a side stream; `derive.py` is the derive's bit-exact numpy twin, which gates every chain |
| `characterize.py` | the card: identity, tensor-core peak rates per MMA kind, DRAM and L2 bandwidth, operand-entropy control, sustained clocks under a held GEMM; the lock-clock sweep only when asked (`--sections lock`), never where someone else fixes the clocks (node 2) |
| `baselines.py` | the baseline families (BF16, FP8 per-tensor and rowwise, MXFP8, FP8 1×128/128×128, NVFP4, MXFP4, int8): cuBLASLt's top heuristic algorithms (for the families `--lt-enumerate` names, every configuration its capabilities admit at the shape, screened by one graph rep before the tune; a space too big for one run is searched in slices, `--lt-enumerate-range START:STOP`, and its best tuned together, `--lt-enumerate-names FILE`), from each cuBLASLt loaded, CUTLASS's sm_120 kernels, and the plain NVFP4 and FP8 GEMMs of `../nvfp4_sm120/nvf4_plain.cu` (#543, #570; source `verity`, `verity_nvf4_256x128*` and `verity_fp8_256x128*`: 256×128 tiles, one persistent CTA an SM, m a multiple of 256, n of 128, k of 128 or for FP8 64, one tile order, a workspace of their own for their tensor maps), and the bit-exact gate on ternary operands |
| `example_arm.py` | an example arm, **not a PoUW scheme** (plain cuBLASLt FP8 and a stand-in read of D): the smoke test of the arm path and a template |
| `sass_gate.py` | the SASS gate, before anything is timed: no instruction with `.FTZ` and no FP32 `MUFU` (so no `-ftz=true`, `--use_fast_math` or approximate div/sqrt build) in any binary an arm declares (`binaries` in its build record) or any library with device code mapped into the run, except inside ptxas's IEEE div/sqrt/rcp and integer-division sequences as a ptxas pinned in `sass_pins.json` (version, build, arguments, numeric run) builds them; nvdisasm and `cuobjdump -sass` must agree; also a CLI (`python sass_gate.py BINARY...`, `--pin`) |
| `ieee_pin.py`, `ieee_pin.cu` | the gate's numeric pin, and where its pins come from: builds `ieee_pin.cu` with each pinned toolkit and flag set and three negative controls, checks each toolkit's `sass_pins.json` entry is what its builds pin, gates them, runs them under `gpu-lease 1` (rcp and sqrt on all 2^32 inputs, div on 2^32 pairs weighted to subnormals, integer division exhaustively or weighted, every subnormal input against the host too) and writes `ieee-pin.json`; its run is each toolkit's `numeric` |
| `verify.py` | the verify step, after the lease in the same run: each arm's reference verifier, a replay, on the transcripts its timed code emitted; writes the verifier's commit, its accept line and the transcript (`<run id>:<path> sha256:<hex>`) beside each panel row, and prints `panel.py append`'s arguments. An accepted row is replay-verified (development shortcut; the protocol's check is a ZK proof, PROTOCOL.md) |
| `ledger.py` | the attempt ledger: `verify.py` writes one `kernel-attempt/v1` record per arm and shape under the run's `ledger/` (the variant, parent, tier, gate version, metrics, step times, γ, the verifier's lines, the hypothesis) and publishes it with `--publish`; `ledger.py publish RUN_DIR` later, `ledger.py outcome ART OUTCOME --reason ...`, and `ledger.py rows`, what `panel.py` reads beside `attempts.jsonl` |
| `price_twins.py`, `price_twins.json` | γ's Lean-pinned price twins, read from the POUS Project store's pins (`python price_twins.py STORE_ROOT`, or `--repository-only` when only `Security/lean-audit.json` changed): the ledger looks an attempt's γ up at both FP32 prices on the priced path its variant declares, at its cube or its linear (n, k) (`lookup(..., layout=)` for a served layout's), and never takes one typed in |
| `schedule.py`, `report.py`, `shapes.py`, `identity.py`, `fill.py` | the rules as pure functions (tested on CPU in `../tests/test_harness.py`) |
| `native/` | `libpouw_harness.so`: fills, flushes, events, graphs, cuBLASLt (`lt.cu`), the CUTLASS kernels (`build.sh`, CUTLASS pinned at v4.8.0) and `../nvfp4_sm120/nvf4_plain.cu`'s in the same registry (`cutlass_api.h`; `PH_NO_CUTLASS=1` builds only these); optionally `libpouw_lt2.so`, `lt.cu` alone against a second cuBLASLt |

Build (no GPU needed; nvcc 12.9, about 7 GB of memory per CUTLASS 3.x unit):

```bash
export POUW_HARNESS_LIB="$(benchmarks/pouw/harness/native/build.sh)"   # prints the library's path, nothing else
```

The build lands in `native/build/<key>/` (`POUW_HARNESS_BUILD` moves the cache), keyed by the sources, nvcc's version and the
cuBLAS headers', with `build.json` (the key, versions, the library's hash, each kernel's SASS MMA counts); an existing build is
reused, and so is each object whose unit, headers and flags are unchanged. A run refuses a library its `build.json` doesn't
describe. Where there is no CUDA 12.9 (node 2 has 13.0), `. jobs/toolkit.sh` first assembles it from NVIDIA's pinned
redistributable archives and the cuBLASLt wheel `env.sh` pins. `LT2_HOME` names a second cuBLAS install (node 2's
`/usr/local/cuda`, whose `libcublasLt.so.13` is 13.1.1.3): `lt.cu` is built alone against it with the host compiler into
`libpouw_lt2.so`, which a run loads beside the library (`RTLD_LOCAL`; `POUW_HARNESS_LT2` names another, or `none`), so every
family also gets that cuBLASLt's candidates, named `lt<major>_…`. On node 2 the build is a CPU job, at `nice 19` on the fill
queue's cores:

```bash
research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source -- bash -c \
  '. benchmarks/pouw/harness/jobs/toolkit.sh && export POUW_HARNESS_BUILD=/workspace/pouw/harness/build JOBS=8 LT2_HOME=/usr/local/cuda &&
   L="$(nice -n 19 taskset -c 96-191 benchmarks/pouw/harness/native/build.sh)" &&
   cp "${L%/*}"/{libpouw_harness.so,libpouw_lt2.so,build.json} "$RESEARCH_RUN_DIR/"'
```

`build.json` also records each kernel's registers, stack and local memory (`resources`; a spill shows as local bytes).

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
`research data put --kind pouw-harness-build/v1 --tree DIR --preserve`), and this tree's `verity` and `verity.core.protocols.pouw` on
`PYTHONPATH`. On node 2, `gpu-lease 1` picks the GPU, so the run names it `--gpu lease` (the one index the lease set, checked
against `server.md`'s GPU table); a timed run holds the node with `gpu-lease 8 --wait --timed` and times on its first GPU
(`--timed` pauses the fill queue: without it, fill jobs keep running beside the timed run). A timed run also passes
`research run --no-sampler`, so `research`'s own telemetry sampler (`nvidia-smi` every 5 s) stays off the node while it
times (17:27Z); the harness records clocks and the UUID itself:

```bash
R='research run --on vy-nebius-2 --project verity --campaign pouw --source . --cwd source --env GPU_LEASE_WHO=<your bc-id>
   --send BUILD/libpouw_harness.so --send BUILD/libpouw_lt2.so --send BUILD/build.json --send GPUS.md'  # GPUS.md: server.md's table
H='. benchmarks/pouw/harness/jobs/env.sh "$RESEARCH_RUN_DIR/inputs" && "$PY" benchmarks/pouw/harness/bench.py --gpu lease
   --server-md "$RESEARCH_RUN_DIR/inputs/GPUS.md" --clock-label locked-2100 --shapes headline
   --arm-path benchmarks/pouw/harness --arm example_arm:ExampleFp8 --out "$RESEARCH_RUN_DIR/bench.json"'
$R -- gpu-lease 1 --wait -- bash -c "$H"                                                           # untimed: gates, autotune
$R --no-sampler -- gpu-lease 8 --wait --timed -- timeout 1140 bash -c 'export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} && '"$H"  # timed
```

A measured panel row also needs the verify step in the same run, after the lease, so no GPU waits on the CPU verifier:

```bash
V='. benchmarks/pouw/harness/jobs/env.sh "$RESEARCH_RUN_DIR/inputs" && "$PY" benchmarks/pouw/harness/verify.py "$RESEARCH_RUN_DIR/bench.json"'
$R --no-sampler -- bash -c "gpu-lease 8 --wait --timed -- timeout 1140 bash -c 'export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES%%,*} && $H'; b=\$?; $V; v=\$?; exit \$(( b ? b : v ))"
```

`H` and `V` go to `bash -c`, so each must be one line (`H` is wrapped above for reading).

An arm on another branch runs from a tree that holds both: its branch with this directory and `../tests/test_harness.py`
checked out from the harness's commit, `--arm-path` its own directory. `jobs/harness.yaml` and `jobs/harness-timed.yaml` are
node 2's queue jobs, for when its Kueue is up.

Before a shape's reps, the same interleaved schedule runs as a warm-up (kept apart, in `warmup_items`) for at least
`--warmup-s` of device time and until a round's median SM clock is within 0.5% of the round before (`--warmup-max-s` bounds
it; the shape's `warmup` says whether it got steady). Then every entry's timed item is sized to the same device time, at
least `--item-ms` (40): calls for a baseline or an arm, replays for a chain (`sizing`). Every item samples NVML while the GPU
runs it (`sensors.during`), and each panel row carries both sides' reps in the same run (`panel.reps`: per-call and item ms
and SM clock of the arm's and the baseline's items, rep by rep), `sm_clock_arm` and `sm_clock_base` for `panel.py`, and
`clock_gap` (medians more than 0.5% apart; 08:50Z).

Decode rows count on the panel only as a dependent chain, which needs a library with the derive kernel (`ph_derive`);
`--decode-method independent` times the old independent graph with any library. An arm (`arm.BenchArm`, arm interface v0.6;
`Arm` is its deprecated alias, since the vLLM API's `Arm` is a serving bench mode) gets the split when its calls expose
`phases` named by the step groups the executor schedules, `arm.GROUPS` (`a_tree`, `forming`, `gemm`, `cleanup`,
`tile_hash`, `screen`, as `verity.core.protocols.pouw.serving`'s): the chain times before (a_tree, forming), during (gemm, cleanup) and
after (tile_hash, screen, the movable groups), and the report's `split_groups` names the groups each time covers. The
side-stream variant times the served `deferred` schedule (`verity.core.protocols.pouw.serving.DEFERRED`): tile hashing (`arm.DEFERRED`)
off the main lane, the screen on it. v0.4's `before`, `gemm` and `after` still work, each as its part, and are deprecated;
their one `after` can't be split that way, so they get no side-stream variant. `arm.py` has the contract.

A search's best, timed later with `--lt-enumerate-names FILE`, needs an `--lt-enumerate-max` that reaches every configuration
FILE names: the enumeration keeps only its first `--lt-enumerate-max` (8,192) per library and swap, and cuBLASLt 13's FP8
space at `m32-n8192-k8192` is 134,216. A run that doesn't take every named configuration at every shape stops with exit 2.

An arm's `build` names every cubin, fatbin or library its kernels come from in `binaries` (arm interface v0.3; a cubin loaded
with `cuModuleLoadData` is never mapped, so the gate can't find it alone). The gate needs `cuobjdump` and `nvdisasm`
(`POUW_CUDA_BIN`, `CUDA_HOME/bin`, the PATH or `/usr/local/cuda/bin`) and caches passes by the binary's sha256
(`POUW_SASS_GATE_CACHE`). To check a build before a run: `python sass_gate.py my_kernels.cubin --json gate.json`. Build
sm_120a with a pinned toolkit (CUDA 12.9.1 from `jobs/toolkit.sh`, or 13.0.1, node 2's `/usr/local/cuda`) and without
`-ftz=true`, `--use_fast_math`, `-prec-div=false` or `-prec-sqrt=false`; `-fmad=false` is pinned too. Any other toolkit,
architecture or ptxas argument fails closed until it is pinned (`--pin ... --toolkit`, from `ieee_pin.cu`'s builds) with a
numeric run (`ieee_pin.py`).

An arm's `build` also names its reference verifier in `verifier` (arm interface v0.4, `arm.check_verifier`, checked before
anything is timed): the transcript file per shape, relative to the run's directory (e.g.
`transcripts/{arm}/{shape}/manifest.json`; the arm writes it under `ctx["transcripts_dir"]`), the commands that check it,
run once per shape (`per: "shape"`, the default) or once per run (`per: "run"`), and the verifier's `commit` if it isn't the
run's source commit (`RESEARCH_SOURCE_SHA`). For example, GPU 1's `verify.py` over the whole run directory:

```python
"verifier": {"per": "run", "transcript": "transcripts/{arm}/{shape}/manifest.json",
             "commands": [["{python}", str(HERE / "verify.py"), "{run_dir}", "--commit", "{commit}"]]}
```

`verify.py RESULT_JSON` runs them from the run's working directory, keeps their output under `verify/`, and accepts a
transcript only by an output line that starts with `ACCEPT` and names `sha256:<hex>` of the transcript file as it hashes it
itself, with no `REJECT` line naming it (per shape, every command must also exit 0). Each panel row gets `verification`,
and when accepted `verifier_commit`, `verifier_accept` (the line verbatim), `transcript` and `panel_args`; the result JSON
gets `verification` with every command's exit code. It exits 0 only if every panel row was accepted and every command
exited 0, 1 otherwise, 2 on a usage error (no run id, no commit). An arm without a verifier gets no measured row.

Every reference verifier an arm names today is a replay: it opens the run's rows and words and recomputes the scheme in the
clear. So an accepted row is replay-verified (development shortcut; the protocol's check is a ZK proof, PROTOCOL.md). The
records and `panel.py`'s arguments keep the name `verifier`.

Every gate's record and every negative control carry `wall_s`, the host's seconds in them (a gate's summed over its runs),
and each shape its `wall_s` by stage (`operands`, `baseline_gates`, then `screen` alone under `--screen-only`, else
`arm_gates`, `tune`, `chain_gates`, `sass_gate`, `warmup`, `timed`, `transcripts`, `report`), so a lease's idle minutes
show where they went: operands are filled on the device in milliseconds, while an arm's CPU reference inside its gates can
take minutes. So `refs.py` gives arms `pool()`, a forked process pool of the job's CPUs (`OMP_NUM_THREADS`, a fill job's
`cpus=`), for such references, and an arm with `prefetch(dev, call, shape, flags)` (optional, arm interface v0.6) is told
every call a flag set's gate is about to check before the first of them, so their references compute side by side; the gate still compares against the
inputs as the device holds them at the gate.

**Poisoned transcripts and the no-write control** (panel rule, 11:50Z). An arm with `dump` (arm interface v0.5)
lets the harness make its transcripts after the timed items: every buffer the transcript reads (`call.transcript_buffers`,
default `call.outputs`; never an operand, since poisoning one changes what the call computes, so an overlap is refused and the
row gets no control) is filled with 0xA5, the weight side and the call run, and the arm's `dump` writes
`transcripts/<arm>/<shape>/`; then the buffers are poisoned again, nothing is launched, and `dump` writes
`negative/<arm>/<shape>/`, which the verifier must reject. The run's JSON says what was poisoned and how many bytes the call
wrote (`poisoning`). `verify.py` judges the control at the transcript's path under `negative/` unless the declaration names
another `control`; an accepted control rejects the row, and a rejected one's line is the row's `--negative-control`. An arm
without `dump` still writes its own transcripts, and its record says the harness didn't poison them.

**Every attempt is recorded.** `verify.py` writes `ledger/<arm>/<shape>.json` for every arm row, replay-verified or not; pass
`--tier 2b` for a timed whole-node run (the default, 2a, is a screen that never plots), `--hypothesis`, `--parent`, and
`--publish` to put the records into the evidence store. The arm's build record names its `variant`
(`ledger.variant_record`: the sources' digest, the binary's, the priced path) and its `lineage` (parent, approach,
mechanism tags, hypothesis, panel line and version). A knob that picks code at run time (`PEARLC_A_FORMS`) is part of the
variant: pass it as `variant_record(..., config={...})`. γ is looked up from `price_twins.json` on the variant's priced path, at
both FP32 prices, the larger published; a lineage that types γ in gets no record. The records are also declared in the run's
`outputs.json`, so under `research run --on`, whose workload holds no store key, the runner publishes them with the Attempt;
`ledger.py publish RUN_DIR --by LANE` afterwards only asserts their outcome labels.

Exit codes: 0 ok; 2 identity or usage; 3 a device, build or harness error; 5 a known-bad variant wasn't rejected; 6 the
consistency gate rejected an arm; 7 the SASS gate refused a binary. The JSON is written in every case, with every rep, and
`sass_gate` holds every binary's verdict.
