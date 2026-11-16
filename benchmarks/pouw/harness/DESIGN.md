# The PoUW bench harness: design

How a run goes on a GPU, what an arm declares, and what a run records, beside [the README](README.md).

Run on your GPU only (`CUDA_VISIBLE_DEVICES` one index; the UUID from `server.md`'s table):

```bash
cd benchmarks/pouw/harness
python bench.py --plan                                              # sets, bursts, memory per shape; no GPU
CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID python bench.py --gpu 5 --expect-uuid GPU-... \
    --shapes headline --arm example_arm:ExampleFp8 --out run.json
CUDA_VISIBLE_DEVICES=5 CUDA_DEVICE_ORDER=PCI_BUS_ID python characterize.py --gpu 5 --expect-uuid GPU-... --out char.json
```

Where there is no CUDA 12.8+ toolkit, or it isn't the build's (the RTX PRO 6000 node has 13.0), `jobs/env.sh BUILD`
gives the run NVIDIA's cuBLASLt and runtime wheels, a library built elsewhere (a build directory, or `art:<id>` of one
preserved with `research data put --kind pouw-harness-build/v1 --tree DIR --preserve`), and this tree's `verity` and
`verity.protocols.resource_accounting.compute.pouw.verifier` on `PYTHONPATH`. On that node, `gpu-lease 1` picks the GPU, so the run names it
`--gpu lease` (the one index the lease set, checked against `server.md`'s GPU table); a timed run holds the node with
`gpu-lease 8 --wait --timed` and times on its first GPU (`--timed` pauses the fill queue: without it, fill jobs keep
running beside the timed run). A timed run also passes `research run --no-sampler`, so `research`'s own telemetry
sampler (`nvidia-smi` every 5 s) stays off the node while it times; the harness records clocks and the UUID itself:

```bash
R='research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=<you>
   --send BUILD/libpouw_harness.so --send BUILD/libpouw_lt2.so --send BUILD/build.json --send GPUS.md'  # GPUS.md: server.md's table
H='. benchmarks/pouw/harness/jobs/env.sh "$RESEARCH_RUN_DIR/inputs" && "$PY" benchmarks/pouw/harness/bench.py --gpu lease
   --server-md "$RESEARCH_RUN_DIR/inputs/GPUS.md" --clock-label locked-2100 --shapes headline
   --arm-path benchmarks/pouw/harness --arm example_arm:ExampleFp8 --out "$RESEARCH_RUN_DIR/bench.json"'
$R -- gpu-lease 1 --wait -- bash -c "$H"                                                           # untimed: gates, autotune
$R --no-sampler -- gpu-lease 8 --wait --timed -- timeout 1140 bash -c 'export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} && '"$H"  # timed
```

A measured panel result also needs the verify step in the same run, after the lease, so no GPU waits on the CPU verifier:

```bash
V='. benchmarks/pouw/harness/jobs/env.sh "$RESEARCH_RUN_DIR/inputs" && "$PY" benchmarks/pouw/harness/verify.py "$RESEARCH_RUN_DIR/bench.json"'
$R --no-sampler -- bash -c "gpu-lease 8 --wait --timed -- timeout 1140 bash -c 'export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES%%,*} && $H'; b=\$?; $V; v=\$?; exit \$(( b ? b : v ))"
```

`H` and `V` go to `bash -c`, so each must be one line (`H` is wrapped above for reading).

An arm on another branch runs from a tree that holds both: its branch with this directory and `../../../tests/benchmarks/pouw/test_harness.py`
checked out from the harness's commit, `--arm-path` its own directory. `jobs/harness.yaml` and `jobs/harness-timed.yaml` are
the RTX PRO 6000 node's queue jobs, for when its Kueue is up.

Decode results count on the panel only as a dependent chain, which needs a library with the derive kernel (`ph_derive`);
`--decode-method independent` times the old independent graph with any library. An arm (`arm.BenchArm`, arm interface
v0.6) gets the split when its calls expose
`phases` named by the step groups the executor schedules, `arm.GROUPS` (`a_tree`, `forming`, `gemm`, `cleanup`,
`tile_hash`, `screen`, as `verity.protocols.resource_accounting.compute.pouw.verifier.serving`'s): the chain times before (a_tree, forming), during (gemm, cleanup) and
after (tile_hash, screen, the movable groups), and the report's `split_groups` names the groups each time covers. The
side-stream variant times the served `deferred` schedule (`verity.protocols.resource_accounting.compute.pouw.verifier.serving.DEFERRED`): tile hashing (`arm.DEFERRED`)
off the main lane, the screen on it. v0.4's `before`, `gemm` and `after` still work, each as its part, and are deprecated;
their one `after` can't be split that way, so they get no side-stream variant. `arm.py` has the contract.

A search's best, timed later with `--lt-enumerate-names FILE`, needs an `--lt-enumerate-max` that reaches every configuration
FILE names: the enumeration keeps only its first `--lt-enumerate-max` (8,192) per library and swap, and cuBLASLt 13's FP8
space at `m32-n8192-k8192` is 134,216. A run that doesn't take every named configuration at every shape stops with exit 2.

An arm's `build` names every cubin, fatbin or library its kernels come from in `binaries` (arm interface v0.3; a cubin loaded
with `cuModuleLoadData` is never mapped, so the gate can't find it alone). The gate needs `cuobjdump` and `nvdisasm`
(`POUW_CUDA_BIN`, `CUDA_HOME/bin`, the PATH or `/usr/local/cuda/bin`), and the system's `libzstd.so.1` for a binary whose
device code nvcc compressed (build with `--no-compress` and it needs none), and caches passes by the binary's sha256
(`POUW_SASS_GATE_CACHE`). To check a build before a run, from the repository's root:
`python verity/protocols/resource_accounting/compute/pouw/prover/sass_gate.py my_kernels.cubin --json gate.json`. Build
sm_120a with a pinned toolkit (CUDA 12.9.1 from `jobs/toolkit.sh`, or 13.0.1, the RTX PRO 6000 node's `/usr/local/cuda`)
and without `-ftz=true`, `--use_fast_math`, `-prec-div=false` or `-prec-sqrt=false`; `-fmad=false` is pinned too. Any
other toolkit, architecture or ptxas argument fails closed until it is pinned (`--pin ... --toolkit`, from
`ieee_pin.cu`'s builds) with a numeric run (`ieee_pin.py`).

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
itself, with no `REJECT` line naming it (per shape, every command must also exit 0). Each panel result gets `verification`,
and when accepted `verifier_commit`, `verifier_accept` (the line verbatim), `transcript` and `panel_args`; the result JSON
gets `verification` with every command's exit code. It exits 0 only if every panel result was accepted and every command
exited 0, 1 otherwise, 2 on a usage error (no run id, no commit). An arm without a verifier gets no measured result.

Every gate's record and every negative control carry `wall_s`, the host's seconds in them (a gate's summed over its runs),
and each shape its `wall_s` by stage (`operands`, `baseline_gates`, then `screen` alone under `--screen-only`, else
`arm_gates`, `tune`, `chain_gates`, `sass_gate`, `warmup`, `timed`, `transcripts`, `report`), so a lease's idle minutes
show where they went: operands are filled on the device in milliseconds, while an arm's CPU reference inside its gates can
take minutes.

**Poisoned transcripts and the no-write control** (a panel rule). An arm with `dump` (arm interface v0.5)
lets the harness make its transcripts after the timed items: every buffer the transcript reads (`call.transcript_buffers`,
default `call.outputs`; never an operand, since poisoning one changes what the call computes, so an overlap is refused and the
result gets no control) is filled with 0xA5, the weight side and the call run, and the arm's `dump` writes
`transcripts/<arm>/<shape>/`; then the buffers are poisoned again, nothing is launched, and `dump` writes
`negative/<arm>/<shape>/`, which the verifier must reject. The run's JSON says what was poisoned and how many bytes the call
wrote (`poisoning`). `verify.py` judges the control at the transcript's path under `negative/` unless the declaration names
another `control`; an accepted control rejects the result, and a rejected one's line is the result's `--negative-control`. An arm
without `dump` still writes its own transcripts, and its record says the harness didn't poison them.

**Every attempt is recorded.** `verify.py` writes `ledger/<arm>/<shape>.json` for every arm and shape, replay-verified or not; pass
`--tier 2b` for a timed whole-node run (the default, 2a, is a screen that never plots), `--hypothesis`, `--parent`, and
`--publish` to put the records into the evidence store. The arm's build record names its `variant`
(`ledger.variant_record`: the sources' digest, the binary's, the priced path) and its `lineage` (parent, approach,
mechanism tags, hypothesis, panel line and version). A knob that picks code at run time (`PEARLC_A_FORMS`) is part of the
variant: pass it as `variant_record(..., config={...})`. The gap is looked up from `price_twins.json` on the variant's priced path, at
both FP32 prices, the larger published; a lineage that types the gap in gets no record. The records are also declared in the run's
`outputs.json`, so under `research run --on`, whose workload holds no store key, the runner publishes them with the Attempt;
`ledger.py publish RUN_DIR --by LANE` afterwards only asserts their outcome labels.

Exit codes: 0 ok; 2 identity or usage; 3 a device, build or harness error; 5 a known-bad variant wasn't rejected; 6 the
consistency gate rejected an arm; 7 the SASS gate refused a binary. The JSON is written in every case, with every rep, and
`sass_gate` holds every binary's verdict.
