# gpu_kit: the timing protocol as code

The kernel-engineering skill's timing protocol (`.agents/skills/kernel-engineering/SKILL.md`), so a bench includes it
instead of writing its own poisoning, clock reads or SASS checks. `harness/` is the same protocol for an arm of the
harness; this kit is for a standalone `.cu` bench, and its constants are the harness's.

| File | What |
|---|---|
| `bench.cuh` | Header only (C++17). `gk::poison_gate`: 0xA5 the output on the kernel's own stream, run, judge; then the no-write control, which the judge must reject. `gk::timed_items`: every arm's item sized to 40 ms, a warm-up until the SM clock is steady, then rep by rep with the arms' order rotating, NVML's SM clock and throttle reasons before, during and after every item. `gk::summary`: a `GK arm=...` line an arm, tagged `[clock-gap]`, `[throttled]`, `[unsteady]` or `[no-clock]`. `gk::Dump`: the one dump format |
| `sass_gate.py` | The timing gate, on the kernel's SASS before it is timed: the timed region's loops and each one's exact opcode counts (`--expect`), no LDL/STL, a live sink, and with `--smem` the bank check on the kernel's own addresses. It reuses `harness/sass_gate.py`'s cubin reader and instruction model; `--numeric` also runs that gate |
| `banks.py` | Shared-memory bank conflicts of one warp-wide access (LDS/STS/ATOMS phases of 128 B, LDSM/STSM phases of 8 rows) |
| `dump.py` | The dump format: `NAME.bin` and a `dump.jsonl` line a buffer, layout `bytes` or `sha512-be`; the reader checks sizes and names |
| `compare.py` | The twin compare: a device dump against its twin's, naming the first difference and the known faults (not written, byte order, layout) |

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

## A bench on the kit

~~~cpp
#include "../gpu_kit/bench.cuh"

gk::PoisonGate g = gk::poison_gate(out, bytes, s, [&](cudaStream_t st) { kernel<<<grid, block, 0, st>>>(out); },
                                   [&](const void* h, size_t n) { return matches_twin(h, n); });
if (!g.ok()) return 3;  // the output wasn't written, or the control passed: the timing would show nothing
gk::Nvml nv(0);
std::vector<gk::Arm> arms = {{"baseline", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) base<<<...>>>(); }},
                             {"candidate", [&](cudaStream_t st, int n) { for (int i = 0; i < n; i++) cand<<<...>>>(); }}};
gk::summary(gk::timed_items(arms, 10, s, nv), arms);
gk::Dump("out").digests("leaves", leaves, count, s);  // SHA-512 bytes, whatever word order the device keeps
~~~

Run `sass_gate.py` on the binary in the same job, before the timed run. Write the twin's output with `dump.write` under
the same names and compare with `python compare.py DEVICE_DIR TWIN_DIR` (exit 9 on a difference).

## What it would have caught

- **The memset race (44878194f):** a poison memset on the default stream raced a GEMM on another stream. `gk::poison`
  takes the kernel's stream, and `compare.py` names a difference whose words still hold 0xA5 as "not written".
- **EGEMM's digest byte order (63f78c35a):** the device dumped its digests' native uint64 words, so no reference matched.
  `gk::Dump::digests` writes SHA-512's big-endian bytes, and `compare.py` names a word-reversed difference as byte order.

`../opcode_probe/` is the kit's first user. The tests are `../tests/test_gpu_kit.py`.
