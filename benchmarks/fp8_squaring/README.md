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

Finding: the free rounding is real but it does not help the cheater.  The sm_120
step floors only a *single* E4M3-range operand per instruction (<=448, exact
integers only 0..16), not a column sum, so a squaring needs O(limbs^2)
tensor-core floor steps (~1.4e5 per 1024-bit squaring, ~1100/byte for a dense
input; data-dependent because zero nibbles are skipped), against the
carry-bound encoder's O(limbs).  So counted as roundings it is *worse*, not
cheaper.  And on throughput the fastest big-integer squaring does its carries on
the CUDA cores (`kv.cu`, 1.499e10/s/GPU at 1024 bits; the adversary's best is
FOLD 6 at 1.970e10/s, ra-squaring-ceiling r20261012-194557-c72f): that report
measured that an int8/FP8 tensor-core (Toeplitz) squaring is *slower* than the
CUDA-core column form (~8 SM-clocks vs FOLD 5's 6.0 at 512 bits), so routing the
encoder through FP8/FP32 tensor cores lowers r_enc.  Only a single pinned
m16n8k32 step is a circuit; library split-K/stream-K GEMMs reassociate the
reduction and are not the attested chain.
