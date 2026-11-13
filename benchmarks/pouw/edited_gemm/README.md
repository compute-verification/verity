# edited_gemm: the served edited GEMM and attention, with receipts, against plain int8

The kernels here serve a linear or an attention product on dithered or edited int8 operands, with receipts of the
running sums. `egemm.cu` and `prof.cu` are built on `masked_gemm/lpgemm.cu`'s mainloop, the other sm_120 kernels on
`egemm.cu`, and `egemm80.cu` is `egemm.cu`'s port to sm_80. The directory measures what serving that way costs over the plain product:

- **time over plain** for each arm: `egemm.cu`'s kernels against cuBLASLt's int8 GEMM at Llama-8B's four linears,
  as hot items and interleaved call by call, and the other kernels' arms against their own plain baselines;
- **where the time goes**: Nsight Compute's sections for one call of an arm, which `eps_profile.py` reads into executed
  instructions a MMA by class beside the ACCEL model's bound, and the kernels' SASS;
- **that every arm is right first**: a run times nothing until its gates pass. The SASS gate reads the binary, the
  device gate runs every arm on poisoned output buffers with a no-write control, and a CPU reference recomputes the
  dumped tiles from the inputs and the key alone.

## Hardware

One leased sm_120 GPU (RTX PRO 6000) with the CUDA 13.0 toolkit, `ncu` and `cuobjdump`; `run80.sh` runs on one A100
(sm_80). Clocks are read, never set.

## The runner: `run.sh`

`run.sh` is a GPU job: a `research run` command starts it, from the source tree's root.

~~~text
research run --on MACHINE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  -- bash benchmarks/pouw/edited_gemm/run.sh
~~~

It builds `egemm` and runs the SASS gate outside any lease. Lease 1 runs the device gates at a small shape and each
linear and dumps tiles, which `egemm_ref.py` recomputes on the CPU. Only if both pass, lease 2 times the arms at
o_proj and the other linears and runs `ncu`. It exits 4 when the build fails and 5 when a gate fails.

| Variable | Default | Meaning |
|---|---|---|
| `EG_SET` | `all` | the set of arms and shapes: `best`, `x13`, or one of the fold-cost sets `eps1`, `eps1lag`, `eps1sleep`, `eps1wave`, `eps1pred`, `eps1uni`, `eps1fill` (the script's header says what each times) |
| `REPS`, `IL_REPS` | `15`, `41` | repetitions of a hot item, and of an interleaved pair |
| `O_SHAPE`, `LAYERS` | o_proj, and the other three linears, at m = 8192 | the shapes timed |
| `EG_ARMS`, `LAYER_ARMS` | the set's | the arms timed at o_proj, and at each other linear |
| `IL`, `IL_COLD`, `LAYER_IL` | the set's | the interleaved pairs `ARM:BASE`; `IL_COLD` flushes L2 before each call |
| `NCU`, `NCU_STALL` | the set's | the arms whose counters, and whose stall reasons, `ncu` reads at o_proj; the fold-cost sets' sections under `ncu/` are the set's alone |
| `EG_PYTHON` | `python3` | the Python that runs the SASS gate, the reference and the summary |

What a run writes under `$OUT`:

| File | Holds |
|---|---|
| `commit.txt`, `build.log`, `egemm.sass.gz` | the commit, the build's log and the disassembly |
| `sass_gate.json`, `sass_gate.txt`, `sass_scope.txt`, `sass_local_mem_ops.txt` | the SASS gate's verdict, which kernels its findings are in, and each kernel's local-memory loads and stores |
| `gate.jsonl`, `gate.err`, `gate_NAME_meta.json`, `ref.jsonl`, `ref.err` | the device gates, each gate's shape and key, and `egemm_ref.py`'s checks, one JSON line a shape |
| `gpu.txt`, `smi.csv` | the GPU under the gate lease, and `nvidia-smi`'s samples under the timing lease |
| `bench.jsonl`, `il.jsonl`, their `.err`, `rc.txt` | the hot items and the interleaved pairs, one JSON line a shape; the exit codes |
| `ncu-*`, `ncu-stall-*`, `ncu/` | Nsight Compute's counters and sections for one call of an arm |
| `summary.md` | `summary.py`'s tables of the above |

## The other jobs

Each is a GPU job started as `run.sh` is, with its own name in place of `run.sh`; its header lists its variables
and what it writes.

| Job | Kernel | Reference | What it gates and times |
|---|---|---|---|
| `crun.sh` | `cgemm.cu` | `cgemm_ref.py` | Dither-C's linear at Llama-3.1-8B's four linears at 2048 tokens and 8192³; ends with `eps_profile.py`'s profile and TOML |
| `cwrun.sh` | `cwgemm.cu` | `cwgemm_ref.py` | the same linear with the weight edit formed inside the GEMM, against the edit as a pass |
| `cpvrun.sh` | `cpv.cu` | `cpv_ref.py` | Dither-C's P V, a unit at a time |
| `aprun.sh` | `apv.cu` | `apv_ref.py` | the adopted P V and its V producers alone; ends with `eps_profile.py`'s and `vprod_profile.py`'s profiles |
| `fattnrun.sh` | `fattn.cu`, with `apv.cu`, `cgemm.cu` and `cwgemm.cu` | `fattn_ref.py`, `apv_ref.py`, `cgemm_ref.py` | attention with S never stored, the four linears, and plain attention by `flash_ref.py` (torch) |
| `d2srun.sh` | `d2sgemm.cu` | `d2sgemm_ref.py` | Dither-2S's linear, and Q Kᵀ with both operands dithered |
| `run80.sh` | `egemm80.cu` | `egemm80_ref.py` | the edited GEMM on one A100: `egemm80.py`'s SASS gate, the bit-exact gate, the bench, and each receipted arm held for its steady clock |
| `profile.sh` | `prof.cu` | none | where `lpgemm.cu`'s edited GEMM spends its time at o_proj: its parts, the interleaving against plain cuBLASLt, and `ncu`; `prof_summary.py` renders it |
| `egemm_fill_probe.sh` | `egemm.cu` | `egemm_ref.py` | duo with x extra dependent scalar operations a word, beside the kernel without them and plain |
| `egemm_dial_probe.sh` | `egemm.cu` built with `-DEG_ROUNDS=R` | `egemm_ref.py` | the fold dial below, once for each R in `EG_R_LIST` |

## The fold dial, `EG_ROUNDS`

`egemm.cu`'s XOR-in absorb folds each committed word's four s32 limbs `EG_ROUNDS` times through the XOR-in(13, 19)
step, with no extra MMAs, so the rounds are scalar work and the tensor time is untouched. `EG_ROUNDS == 1`, the
default, is byte-identical to the fold of record. The loop is `#pragma unroll 1`, so the SASS opcode count is the same
at every R, and a `static_assert` bounds R to 1..64.

- **The CPU reference has no dial.** `egemm_ref.py` recomputes the one-round fold. A receipt at R > 1 is checked on the
  device only (the product against cuBLASLt and the poisoned control), until `xin_receipts` and `xin13_receipts` loop
  their absorb as the kernel does.
- **What `egemm_depthr_check.py` tests.** For the same-word R-round fold it searches one family of changes to a committed
  word (5n + 2 a word at width n: each limb's single bit flips, a +1 on the two added limbs, a paired flip on the two
  XOR-in limbs) for a change that no state sees: at width 8 over every odd key pair, at widths 10 and 12 over keys the
  reduced `key_ok_r11` admits, and at 32 bits on sampled states, for R up to 5. It passes when it finds none on a key
  `key_ok_r11` admits. It does not search the word space. `benchmarks/pouw/theory/fold_kernel_check.py` does, and reports
  a 16-element group of changes that fixes the fold at every key and every R; `benchmarks/pouw/theory/fold_open_shape_check.py`
  gives width-8 keys the reduced `key_ok_r11` admits at which further changes fix it. `EG_ROUNDS == 1`, the default, is
  the fold of record, and a depth R ≥ 2 rests on nothing here. `egemm.cu` derives its keys under `key_ok_r9`, which has
  no KJ19 clause (`key_ok_r11` is `key_ok_r9` with it).

### Measured, with no run in the store

`egemm_dial_probe.sh` at 8192³, `x13_l2o` over the no-fold GEMM `e_plain`: device-correct at R = 1, 2 and 4
(`device_ok` and `control_rejected`), at 1.22×, 2.07× and 3.29× the no-fold GEMM's time.

## Files

| File | What it is |
|---|---|
| `egemm.cu`, `cgemm.cu`, `cwgemm.cu`, `cpv.cu`, `apv.cu`, `fattn.cu`, `d2sgemm.cu`, `egemm80.cu`, `prof.cu` | the kernels and their drivers; each header defines its arms and its commands |
| `*_ref.py` | the CPU references: each recomputes its kernel's gate dump from the inputs and the key alone |
| `receipt_folds_r9.py` | the reference of the two receipt folds below duo, which `egemm_ref.py` loads only at its pinned sha256: it is not edited |
| `egemm_depthr_check.py` | the search above; `python3 egemm_depthr_check.py` runs it and exits 0 when every check passes |
| `eps_profile.py`, `vprod_profile.py` | a run's `ncu` sections as executed instructions by class, against the ACCEL model |
| `summary.py`, `prof_summary.py`, `egemm80.py` | `run.sh`'s, `profile.sh`'s and `run80.sh`'s summaries; `egemm80.py gate` is the sm_80 SASS gate |
| `flash_ref.py` | plain attention's time under torch, `fattn.cu`'s baseline |
| `tests/` | the suite `uv run tools/verity/check/suites.py benchmarks/pouw/edited_gemm` runs: the eight CPU judges of the kernels' gates (`egemm_ref.py`, `cgemm_ref.py`, `cwgemm_ref.py`, `cpv_ref.py`, `apv_ref.py`, `fattn_ref.py`, `egemm80_ref.py`, `d2sgemm_ref.py`) on made-up gate directories, `egemm_ref.py` against the receipts reference, the summaries, `eps_profile.py` (priced with the ACCEL model's census) and `vprod_profile.py` on made-up run directories, `egemm80.py`'s SASS gate, the depth-R check, and that a reference started by its path leaves `sys.path` and the module names `derive` and `ref` alone |

## What other code reads

`egemm.cu` and `prof.cu` include `masked_gemm/lpgemm.cu`; `cgemm.cu` and `d2sgemm.cu` include `egemm.cu`; `cwgemm.cu`,
`cpv.cu` and `apv.cu` include `cgemm.cu`; `fattn.cu` includes `apv.cu`. Four references (`egemm_ref.py`, `cgemm_ref.py`,
`d2sgemm_ref.py`, `egemm80_ref.py`) load `floored_gemm/derive.py` from its file for its `tile_receipts`, putting
nothing on `sys.path`, and the Dither-C and P V references recompute through
`benchmarks/pouw/theory/checked_strip.py`, which `cgemm_ref.py` loads the same way. Inside the directory,
`cwgemm_ref.py`, `cpv_ref.py` and `apv_ref.py` import `cgemm_ref.py`, `apv_ref.py` imports `cpv_ref.py`'s
`patch_cols`, and `vprod_profile.py` imports `eps_profile.py`; each module another script imports lists in `__all__`
what they read of it (the suite's tests read more of a module than its list). From outside the directory,
`verity/protocols/pouw/receipts/receipt_conform.py --adopter egemm_ref` checks `egemm_ref.py`'s `xin_keys`,
`xin13_receipts` and `tile_receipts` (the suite runs that check too), `benchmarks/pouw/theory/`'s fold checks import
`egemm_depthr_check.py`'s `key_ok_r11` and `key_ok_r11_32`, `adapters/vllm/tests/engine/` loads `fattn_ref.py` by path
as the int8 attention's oracle, and the ACCEL model's honest profiles under `benchmarks/machine_model/data/` are
written from `eps_profile.py OUT --toml` and `egemm80.py summary`. The Python here touches no GPU but `flash_ref.py`;
the job scripts do the rest.
