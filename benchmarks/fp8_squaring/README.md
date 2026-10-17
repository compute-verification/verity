# FP8-matmul modular squaring (stacking loophole 4)

Workstream G, stacking campaign (17 Oct 2026).  Decides loophole (4): does an
FP32 accumulator inside an FP8 matmul give unbudgeted ("free") carry roundings?

`fp8_squaring.py` builds a full modular squaring mod `2^1024 - 105` from three
primitives, each one real `mma.sync.m16n8k32.f32.e4m3.e4m3` step evaluated
through the repo's bit-exact sm_120 accumulation model
(`verity.ml.circuits.silicon.models.BLACKWELL_SM120_E4M3_M16N8K32`):

- `exact_sum`  -- column sum, acc = 0 (no rounding; exact on the FP32 int grid).
- `prod_floor` -- floor(x*y / 2^b), acc = 2^(b+23): the accumulation truncates
  toward zero, and that truncation is the carry rounding, done for free.
- `val_floor`  -- floor(v / 2^b) for a single value.

`fp8_squaring_test.py` checks every primitive bit-exact and the full
`a^2 mod (2^1024 - 105)` against `pow`.

Finding: the free rounding is real but a throughput dead end.  The sm_120 step
floors only a *single* E4M3-range operand per instruction, so a squaring needs
O(limbs^2) tensor-core floor steps (~131k per 1024-bit squaring, ~1021/byte),
all data-dependent and LSU-bound.  The fastest known big-integer squaring does
carries on the CUDA cores (`kv.cu`, 1.5e10/s); routing them through tensor-core
accumulations to harvest the "free" rounding is far slower (int8-TC products
alone are 0.25x kv.cu, note:20261015T0731Z-draft-model-constants / W12).
