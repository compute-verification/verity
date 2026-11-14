# tensor_kinds: what every native MMA form costs in issue

Each native `mma.sync` form of a card, timed alone and beside a lane of scalar instructions, and from those timings
the form's joint issue coefficient `a`: on the facet `a·#MMA + #other <= cycles` (one issue a SMSP-cycle for every
other instruction), each timed program bounds `a <= (cycles - #other) / #MMA`, and a form's coefficient is the least
bound over its programs. A form's issue price an output slot is `a / slots`, read against IMMA s8 m16n8k32's. The
instruction counts are read from each kernel's SASS, so a program counts what ptxas emitted. Timings only; clocks are
read, never set.

The ACCEL model reads it: `../../machine_model/data/forms.toml` takes its forms from `KIND` in `tensor_kinds.py` and names
these runs as its source, `tensor_kinds.py refit` prints the model's joint issue rows, and `sm80.py census` writes
sm_80's device files.

## Jobs

Each is the command of a `research run` on a node with the card, from a pushed commit; its header has the full
command and its knobs. The build and the SASS dump run outside the lease, and the timed work under `gpu-lease 1`, with
nvidia-smi sampling the leased GPU's clock and power every 200 ms.

| Job | Card | What it times |
| --- | --- | --- |
| `run.sh` | one RTX PRO 6000 (sm_120, the default), one L40S (`TK_ARCH=89`) or one B200 (`TK_ARCH=100`, the legacy `mma.sync` forms) | `tensor_kinds.cu`: every form alone and beside IADD3 and FFMA alternating, and beside IADD3, at 4 to 16 lane ops a MMA, warp-specialized and interleaved; then the fit |
| `search.sh` | one RTX PRO 6000 | `search.cu`: the forms that set the model's joint rows over schedules (the lane op, its register layout, the warps a SMSP and their roles, where the lane ops sit), with the two facet programs first and last as the control |
| `run_sm80.sh [GROUP...]` | one A100 (sm_80) | `sm80.cu`: every form alone and beside a scalar lane, the scalar pipes alone, in pairs and beside a saturated MMA, duo beside IMMA, and which integer forms leave the exact s32 word; `../gpu_kit/sass_gate.py` gates every timed kernel before the lease |

~~~text
research run --on MACHINE --project verity --source /workspace --cwd clone --custody-ttl 12h \
  --env GPU_LEASE_WHO=LANE -- bash benchmarks/pouw/tensor_kinds/run.sh
~~~

## What a run writes

Under the run's `$OUT`:

- every job: `meta.json` (the run's id, the host, the start time and nvcc's version), `lease.json`, `sass/` (each
  binary's `cuobjdump -sass`), the build logs, `gpu.log`, and nvidia-smi's `smi-before.txt`, `smi-samples.csv` and
  `smi-after.txt`;
- `run.sh`: `tensor_kinds.jsonl` (a line a timed case) and, from the fit, `tensor_kinds.json`, `tensor_kinds.md` and
  `fit.log`;
- `search.sh`: `search-TAG.jsonl` a binary and `search.log`;
- `run_sm80.sh`: `g*.jsonl` a group, `gate.json`, `loops.json` and `gate.log` from the gate, and `sm80.json`,
  `sm80.md` and `fit.log` from the analysis.

## Reading runs

~~~text
python3 tensor_kinds.py OUT                             # the fit of a run.sh run; exits 3 when a gate fails
python3 tensor_kinds.py search OUT                      # a search.sh run: every schedule's program, the control, the gates
python3 tensor_kinds.py refit OUT... [--search OUT...]  # the ACCEL model's joint issue rows from these runs, as JSON
python3 sm80.py gate OUT BIN...                         # the timing gate on every timed kernel of sm80.cu
python3 sm80.py analyze OUT                             # a run_sm80.sh run's table
python3 sm80.py census OUT... --extension OUT... [--to DIR]   # sm_80's rows.csv and census.csv for the ACCEL model
~~~

`tensor_kinds.py` counts a fit only when the lane op alone issues at 0.9 a SMSP-cycle or more, and takes a 2% noise
margin; `sm80.py` reports no number for a kernel the gate refused or whose poisoned sink failed. Their module
docstrings give each quantity. One caller reads this directory: `../opcode_probe/sm89/census.py` loads `tensor_kinds.py`
by its path for the sm_89 census, and its `__all__` lists what that reads. The tests are
`../tests/test_tensor_kinds.py` and `../tests/test_tensor_kinds_sm80.py`.
