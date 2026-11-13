# floored_gemm: the useful int8 GEMM with receipts of its running sums

`fgemm.cu` is the served GEMM on the exact int8 grid: the operands are noised a chunk of 32 columns at a time, the tensor
cores multiply the noised operands, and an exact peel returns the product of the codes. The directory measures what a
receipt of the running sums costs that GEMM, against cuBLASLt's int8 GEMM on the same shapes:

- **η and time over cuBLASLt** for each arm: the plain mainloop, the noised call without receipts, the scalar receipts
  duo and duo2 at strides 32 and 64, and the tensor-core receipts TB4, TB2 and TB1 with their chained kinds
  (`fgemm.cu`'s header defines each);
- **where the time goes**: Nsight Compute's tensor-pipe, ALU, issue and DRAM shares for one call of an arm, and the
  kernels' SASS opcode counts;
- **that every arm is right first**: a run times nothing until the gate passes, which dumps every arm's outputs and
  receipts at six small shapes on poisoned output buffers and has `ref.py` recompute each bit on the CPU.

## Hardware

One leased sm_120 GPU (RTX PRO 6000) with the CUDA 13.0 toolkit, `ncu` and `cuobjdump`. Clocks are read, never set.

## The runner: `run.sh`

`run.sh` is a GPU job: a `research run` command starts it, from the clone's root.

~~~text
research run --on MACHINE --project verity --source . --cwd clone --env GPU_LEASE_WHO=LANE \
  --send benchmarks/pouw/floored_gemm/run.sh -- bash -c 'bash "$RESEARCH_RUN_DIR/inputs/run.sh"'
~~~

It builds `fgemm` and counts its SASS outside any lease, runs the gate under a short lease, and only if every check
passes times the arms interleaved at the served shapes and runs `ncu` under a second lease. It exits 4 when the build
fails, 5 when the binary holds no IMMA and 6 when the gate fails.

| Variable | Default | Meaning |
|---|---|---|
| `STAGE` | `all` | `build` stops after the build, `gate` after the gate |
| `REPS` | `15` | timed repetitions a shape |
| `SHAPES` | the served linears on four tile configs | `name:m:n:k:C` a shape, `C` the tile config `P`, `Q`, `W` or `D` |
| `NCU_SHAPES` | five of them | the shapes `ncu` profiles |

What a run writes under `$OUT`:

| File | Holds |
|---|---|
| `commit.txt`, `nvcc.txt`, `build.log` | the commit, the compiler and the build's log with ptxas's register and spill lines |
| `sass-mix.json`, `fgemm.sass.gz` | `sass_mix.py`'s opcode counts a kernel, and the disassembly |
| `gpu-gate.txt`, `gate-gpu.txt`, `gate.jsonl`, `gate.err` | the gate: the GPU it ran on, the device's output, and `ref.py`'s checks, one JSON line a shape |
| `gpu.txt`, `smi.csv` | the GPU under the timing lease and `nvidia-smi`'s samples of its clock, power and temperature |
| `bench.jsonl`, `bench.err`, `rc.txt` | one JSON line a shape with each arm's time, η and clock; the exit codes |
| `ncu-*.csv` | Nsight Compute's counters for one call of an arm |
| `table.md`, `table.err` | `table.py`'s tables of all of the above |

## The other job: `tb_run.sh`

The same stages for the tensor-core receipts beside duo and duo2, started the same way with `tb_run.sh` in place of
`run.sh`. It also takes `FG_ARMS` (the arms timed), `NCU_ARMS` (the arms profiled) and `DIFF=0` (skip the differential
check; `DIFF_TRIALS` sets its trials), and writes `regs.txt` (ptxas's register and spill lines), `tb_table.py`'s
`table.md`, and `diff.json` with `diff.err`.

## Files

| File | What it is |
|---|---|
| `fgemm.cu` | the kernels and their driver: `fgemm gate`, `fgemm bench`, `fgemm one` |
| `ref.py` | the CPU reference of the gate: `python3 ref.py DIR` recomputes every output and receipt of a dump |
| `derive.py` | the duo receipts of one 128 x 128 tile under `ref.py`'s keys, which the edited GEMMs' references share |
| `sass_mix.py` | `python3 sass_mix.py DUMP`: opcode counts a kernel of a `cuobjdump -sass` dump, as JSON |
| `table.py`, `tb_table.py` | `python3 table.py BENCH.jsonl [RUN_OUT_DIR]`: a run's tables as Markdown |
| `diff.py` | `python3 diff.py [--trials N] [--seed S] [--json OUT]`: on the CPU, how often a tensor-core receipt is unchanged when one or two accumulator words are off by a fixed difference |

## What other code reads

Later kernels are built on this one: `masked_gemm/mgemm.cu` includes `fgemm.cu`. Their CPU references import `ref.py`
(the names in its `__all__`: the duo keys and rotations, `mix32`, and the tensor-core receipts); `derive.py` and
`edited_gemm/egemm_ref.py` load it from its file. Four references of `edited_gemm/` and
`benchmarks/pouw/tests/test_segment_h1.py` load `derive.py` from its file for `tile_receipts` and the tile's shape (its
`__all__`, with the `mix32` and `key_mul` it passes on), so the name `derive` is `harness/`'s alone. `tb_table.py` imports
`table.py`'s `LINEAR` and `ncu_tensor`. The Python here starts no process and touches no GPU: only the two job scripts
do. Its tests are `benchmarks/pouw/tests/test_floored_gemm_table.py` and `test_floored_gemm_scripts.py`.
