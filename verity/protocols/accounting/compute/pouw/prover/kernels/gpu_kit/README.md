# The GPU timing kit

The kernel-engineering skill's timing protocol (`.agents/skills/kernel-engineering/SKILL.md`) as code, for a standalone
`.cu` bench: a bench includes it instead of writing its own poisoning, clock reads or SASS checks. `bench.cuh` (its API
in its header) poisons the kernel's output and runs the no-write control, times the arms with their order rotating and
NVML's SM clock read around every item, and dumps buffers; `sass_gate.py` refuses a binary whose timed region is not
what the bench names (`--expect`); `compare.py` compares a device dump against its twin's. `harness/` is the same protocol for an
arm of the harness, and the kit's constants are the harness's. The subsystem is [`../README.md`](../README.md).

## Try it

~~~sh
uv run pytest tests/benchmarks/pouw/test_gpu_kit.py -q
~~~

It runs the kit's tests with no GPU.

`python sass_gate.py BINARY --cuda-bin DIR --kernels K... --expect OP=N[,OP=N...]... [--smem FILE] [--numeric]` exits 0
when every kernel passes and 8 when one is refused, naming the kernel, the check and the fix; DIR is the bin directory
of the toolkit that built the binary. `python compare.py DEVICE_DIR TWIN_DIR` exits 9 on a difference.
