# The GPU timing kit

The kit is the kernel-engineering skill's timing protocol (`.agents/skills/kernel-engineering/SKILL.md`) as code, for a
standalone `.cu` bench: a bench includes it instead of writing its own poisoning, clock reads or SASS checks.
`bench.cuh` poisons the kernel's output and runs the no-write control the judge must reject, times the arms with their
order rotating and NVML's SM clock read around every item, and dumps buffers; `sass_gate.py` refuses a binary whose
timed region is not what the bench declared; `compare.py` compares a device dump against its twin's. `harness/` is the
same protocol for an arm of the harness, and the kit's constants are the harness's.

## Try it

Without a GPU, run the kit's tests:

~~~sh
uv run pytest tests/benchmarks/pouw/test_gpu_kit.py -q
~~~

It prints `19 passed`.

## A bench on the kit

~~~cpp
#include "bench.cuh"

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

`python sass_gate.py BINARY --kernels K... --expect OP=N[,OP=N...]... [--smem FILE] [--numeric]` exits 0 when every
kernel passes and 8 when one is refused, naming the kernel, the check and the fix.
