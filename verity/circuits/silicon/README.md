# verity.circuits.silicon

This is what a GPU's arithmetic does, written down exactly. A tensor core does not add the way IEEE arithmetic does:
it sums products in groups, truncates them at a fixed width, and has its own rules for NaN, infinities and
saturation, and the instruction classes differ. Here each class's accumulator step, the scalar FP32 pipe and the casts are integer arithmetic on
the encodings, and the same steps are also written as exact GF(2) circuits, which the Boolean Definitions are built
from. The semantics import nothing but the standard library.

## Try one

~~~python
from verity.circuits.silicon.instructions import INSTRUCTIONS
from verity.circuits.silicon.total import tc_dot_total

ins = INSTRUCTIONS["sm80.mma.m16n8k16.bf16"]         # an A100's BF16 mma
ins.status                                           # 'pinned': the model matched every hardware probe
tc_dot_total(ins.model, 0x3F800000, [0x3F80] * 16, [0x4000] * 16)   # 0x42040000: 1 + 16 * (1 * 2) = 33.0
~~~

The step takes an FP32 accumulator word and sixteen BF16 words a side, and returns the accumulator word the silicon
computes. The package's docstring lists every module and the step of each pipeline.

## Held to the hardware

`instructions.INSTRUCTIONS` gives each PTX instruction class its model and its status: pinned (the model matched every
hardware probe), hypothesis or unprobed, with the evidence. [`captures/`](captures/README.md) holds replicas of evidence
measured on silicon elsewhere, [`../../../tests/verity/circuits/ops/fixtures/`](../../../tests/verity/circuits/ops/fixtures/README.md)
the captures the tests replay, `hawkeye/PIN.json` the pinned source-close reference, and `discrepancy_log.json` each
place where the paper, the pinned simulator, the reference and the hardware disagree, open until its resolution names
its evidence. The benchmarks [`tc_probe`](../../../benchmarks/tc_probe/README.md) and
[`tc_probe_fp4`](../../../benchmarks/tc_probe_fp4/README.md) capture the hardware.

The package's tests are in `verity/circuits/ops`'s suite, under `tests/verity/circuits/ops/`.
