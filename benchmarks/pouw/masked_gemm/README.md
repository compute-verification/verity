# masked_gemm: what hiding the operands costs the served GEMM

Each kernel here is `floored_gemm/fgemm.cu`'s GEMM with its operands hidden one way, timed against one plain GEMM
(cuBLASLt's, or the plain mainloop) on the same shapes. The directory measures, for each way:

- **time over the plain GEMM** for each arm, at the served linears' shapes and the attention products' (Q Kᵀ and P V
  at 32 heads), with the geometric means the tables print;
- **which part of an arm's time is which**: forming the hidden operands, generating the noise, the GEMM, the peel and
  the receipts, as differences between arms of one interleaved run;
- **that every arm is right**: the gate dumps every arm's inputs, outputs and receipts at small shapes on poisoned
  output buffers, and a CPU reference recomputes each bit. A table marks an arm whose gate failed or was not checked.

The five ways, each defined in its kernel's header:

| Kernel | What is hidden, and how |
|---|---|
| `mgemm.cu` | masked operands: A + R and B + S at full rank, or low-rank masks at ranks 8 to 128, as one GEMM over stacked operands; int8 and E4M3's exact integer grid |
| `egemm.cu` | edited operands: fresh noise from {5, 7, -5, -7} added to both, as a separate pass or fused into the mainloop, with no peel |
| `sgemm.cu` | the edit with its noise expanded by SHA-512 in counter mode (`sha_noise.cuh`), over the alphabets ±1 and {-8..8} |
| `pvgemm.cu` | P V's noise: one shared edit a layer or fresh noise a head, added in a pass, in the mainloop, or where P is produced |
| `lpgemm.cu` | one-side-peeled noise: A edited by a rank-32 term that a tail stage or an extended cuBLASLt call peels, B by fresh ±1 |

## Hardware

One leased sm_120 GPU (RTX PRO 6000) with the CUDA 13.0 toolkit. Clocks are read, never set.

## The runner: `run.sh`

`run.sh` is a GPU job: a `research run` command starts it, from the clone's root.

~~~text
research run --on MACHINE --project verity --source . --cwd clone --env GPU_LEASE_WHO=LANE \
  --send benchmarks/pouw/masked_gemm/run.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
~~~

It builds `mgemm` outside any lease, runs the gate at three small shapes under a short lease and has `mref.py` check the
dumps, then times the arms interleaved at the served shapes under a second lease, whatever the gate said. It exits 4
when the build fails.

| Variable | Default | Meaning |
|---|---|---|
| `STAGE` | `all` | `build` stops after the build, `gate` after the gate |
| `REPS` | `11` | timed repetitions a shape |
| `SHAPES` | the four served linears at m = 8192, and 4096³ | `name:m:n:k` a shape |
| `MG_ARMS` | every arm | the arms timed |

What a run writes under `$OUT`:

| File | Holds |
|---|---|
| `commit.txt`, `nvcc.txt`, `build.log`, `regs.txt` | the commit, the compiler, the build's log, and ptxas's register and spill lines |
| `gpu-gate.txt`, `gate-gpu.txt`, `gate.jsonl`, `gate.err` | the gate: the GPU it ran on, the device's output, and `mref.py`'s checks, one JSON line a shape |
| `gpu.txt`, `smi.csv` | the GPU under the timing lease and `nvidia-smi`'s samples of its clock, power and temperature |
| `bench.jsonl`, `bench.err`, `rc.txt` | one JSON line a shape with each arm's time and clock; the exit codes |
| `table.md`, `table.err` | `mtable.py`'s tables of the above |

## The other jobs

Each is started as `run.sh` is, with its own name in place of `run.sh`, and takes `STAGE`, `REPS` (default 11 for
`erun.sh`, 7 for the others) and `SHAPES`, a shape being `name:m:n:k:batch`. `pvrun.sh` and `lprun.sh` gate and time under
one lease, so their `STAGE` is `build` or `all`.

| Job | Kernel | Reference | Table | Its own variables |
|---|---|---|---|---|
| `erun.sh` | `egemm.cu` | `eref.py` | `etable.py` | `EG_ARMS` |
| `srun.sh` | `sgemm.cu` | `sref.py` | `stable.py` | `SG_ARMS` |
| `pvrun.sh` | `sgemm.cu` and `pvgemm.cu` | `sref.py`, `pvref.py` | `stable.py`, `pvtable.py` | `SG_ARMS` (default: the plain arms and each alphabet's separate-pass arms), `PV_ARMS`, `PV_S` (the sequence lengths, default `2048 8192`) |
| `lprun.sh` | `lpgemm.cu` | `lpref.py` | `lptable.py` | `LP_ARMS` |

`erun.sh` and `srun.sh` write the files `run.sh` writes. `pvrun.sh` and `lprun.sh` write `gate-KERNEL.txt`,
`build-KERNEL.log` and `regs-KERNEL.txt` a kernel and no `gpu-gate.txt`, and `pvrun.sh` adds `pvgate.jsonl`,
`pvbench.jsonl` and `pvtable.md` for `pvgemm`.

## Files

| File | What it is |
|---|---|
| `mgemm.cu`, `egemm.cu`, `sgemm.cu`, `pvgemm.cu`, `lpgemm.cu` | the kernels and their drivers: `NAME gate DIR ...` dumps every arm at one small shape, `NAME bench REPS SHAPE...` times them interleaved |
| `sha_noise.cuh`, `sha_pass.cuh` | SHA-512 in counter mode as noise, and the passes that add it to an operand |
| `mref.py`, `eref.py`, `sref.py`, `pvref.py`, `lpref.py` | the CPU references: `python3 NAME DIR` recomputes a gate dump, prints one JSON line of checks, and exits 0 only if every check passes; all but `pvref.py` take `--no-receipts`, for a dump without them |
| `mtable.py`, `etable.py`, `stable.py`, `pvtable.py`, `lptable.py` | `python3 NAME BENCH.jsonl GATE.jsonl`: a run's tables as Markdown |

## What other code reads

Later kernels are built on these: `edited_gemm/egemm.cu` and `prof.cu` include `lpgemm.cu`, and `floored_gemm/fgemm.cu`
includes `sha_noise.cuh` for its own noise. `mgemm.cu` includes `fgemm.cu`, and the other four kernels include
`mgemm.cu`. Among the references, `eref.py`, `sref.py`, `pvref.py` and `lpref.py` import `mref.py`'s `decode`, `words`
and `duo_receipts`, and `pvref.py` and `lpref.py` import `sref.py`'s `sigma`, `noise` and `fold`; each runs with this
directory as its script directory. The Python here starts no process and touches no GPU: only the five job scripts do.
The tables' test is `benchmarks/pouw/tests/test_masked_gemm_tables.py`.
