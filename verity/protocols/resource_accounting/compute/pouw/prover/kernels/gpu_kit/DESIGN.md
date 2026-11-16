# The GPU timing kit: design

The kit's files, its contract with its callers and its SASS gate's checks, beside [the README](README.md).

## The files

| File | What |
|---|---|
| `bench.cuh` | Header only (C++17). `gk::poison_gate`: 0xA5 the output on the kernel's own stream, run, judge; then the no-write control, which the judge must reject. `gk::timed_items`: every arm's item sized to 40 ms, a warm-up until the SM clock is steady, then rep by rep with the arms' order rotating, NVML's SM clock and throttle reasons before, during and after every item. `gk::summary`: a `GK arm=...` line an arm, tagged `[clock-gap]`, `[throttled]`, `[unsteady]` or `[no-clock]`. `gk::Dump`: the one dump format |
| `sass_gate.py` | The timing gate, on the kernel's SASS before it is timed: the timed region's loops and each one's exact opcode counts (`--expect`), no LDL/STL, a live sink, and with `--smem` the bank check on the kernel's own addresses. It reuses `verity/protocols/resource_accounting/compute/pouw/prover/sass_gate.py`'s cubin reader and instruction model; `--numeric` also runs that gate |
| `banks.py` | Shared-memory bank conflicts of one warp-wide access (LDS/STS/ATOMS phases of 128 B, LDSM/STSM phases of 8 rows) |
| `dump.py` | The dump format: `NAME.bin` and a `dump.jsonl` line a buffer, layout `bytes` or `sha512-be`; the reader checks sizes and names |
| `compare.py` | The twin compare: a device dump against its twin's, naming the first difference and the known faults (not written, byte order, layout) |

**What a caller may import.** A bench includes `bench.cuh` (`../edited_gemm/egemm80.cu`, `../../../../../../../../benchmarks/pouw/opcode_probe/probe.cuh`,
`../../../../../../../../benchmarks/pouw/tensor_kinds/sm80.cu`). A script loads `sass_gate.py` from its file and reads the names in its `__all__`
(`../../../../../../../../benchmarks/pouw/opcode_probe/probe.py`, `../../../../../../../../benchmarks/pouw/tensor_kinds/sm80.py`). `banks.py`, `dump.py` and `compare.py` state an `__all__` each;
their readers are the kit's own files and its tests. `sass_gate.py` and `compare.py` are commands too (the gate below, `compare.py` in the README).

**What it never does.** It holds no kernel and no twin of one: the bench brings both. Its Python launches nothing on a
GPU: the gate reads a built binary, and the other three read and write files where their caller says. It imports the
standard library alone, but for the SASS gate beside PoUW's prover, which `sass_gate.py` loads from its file.

## The gate

`python sass_gate.py BINARY --kernels K... --expect OP=N[,OP=N...]... [--smem FILE] [--numeric]` exits 0 when every
kernel passes and 8 when one is refused. Each refusal names the kernel, the check and the fix:

- **clock:** fewer than two `SR_CLOCKLO` reads, so there is no timed region.
- **loops:** the timed region holds more or fewer loops than `--expect` declares (ptxas unrolled the loop or added a
  remainder), a loop is nested or straddles a clock read, or a loop's count of an opcode isn't the declared one (ptxas
  merged, hoisted or duplicated instructions). The fix is one iteration a trip: `#pragma unroll 1`.
- **forbid:** an LDL or STL anywhere in the kernel (a spill).
- **sink:** nothing the timed instructions compute reaches memory, so ptxas may drop them. The gate traces the kernel's
  reaching definitions from every store after the timed region back to an expected instruction.
- **smem:** an access whose phase puts two words on one bank (with the worst phase's byte offsets), or a sample whose
  opcodes aren't the loop's shared-memory instructions.

`../../../../../../../../benchmarks/pouw/opcode_probe/` is the kit's first user. The tests are `../../../../../../../../tests/benchmarks/pouw/test_gpu_kit.py`.
