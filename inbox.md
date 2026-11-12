[2026-11-12T16:59Z] b50e51167b84
BACKWARD, TRN-140, 8:59 AM PST (16:59Z). FLASH VERDICTS. All five backward products: measured at 16 bits. None at 32.

RAN (CPU, r20261112-164655-3a22, art:fad9ebb89d55ac4377fa01b47dbb7fda73e60307a83e5fa55251dcc395b2eb60): the kernel's order with the tree's twin for each instruction (tc_dot16_hopper for every product, ex2.approx.f32, F2FP) gives dQ / dK / dV = 4,194,249 / 4,194,304 / 4,194,304 of 4,194,304 words over the 16 cases: every word under deterministic algorithms; the 55 dQ misses are the default mode's atomics order, where the card's own repeats differ as much. One change at a time:
- S = Q K^T (recomputed): float32-scaled operands 4,192,791 / 4,193,156 / 4,192,690; same bf16 operands, float32 adds 4,192,438 / 4,192,970 / 4,192,539. Measured at 16 bits (this replaces "not separated": an exact kernel shows every moved word).
- dP = dO V^T: float32 adds dQ 4,164,282, dK 4,193,422. 16 bits.
- dV = P^T dO: float32 P 2,649,422; float32 adds 4,193,305. 16 bits.
- dQ = dS K: float32 dS 2,518,902; float32 adds 4,193,720. 16 bits.
- dK = dS^T Q: float32 dS 2,617,153; float32 adds 4,193,142. 16 bits.
The brief's readings on rows with 64+ keys: (a) float32 57.2 to 58.2% (unit q, k), (b) kernel's order, 16-bit operands, float32 elsewhere 99.88 to 99.94%. Few-key rows apart, per shape, in the note.

ONE MORE THING ON "COUNTED AGAINST RUN", READ not measured: under is_causal the flash kernels skip the blocks above the diagonal, forward (flash_fwd_kernel.h 91-95) and backward (flash_bwd_kernel.h 318-320): T^2/2 + 64 T of T^2 elements a head, 56.25% at T = 1,024. The count's bmm are full rectangles. So a causal call at T = 1,024 is 6 n T^2 E counted and 7 x 0.5625 = 3.94 n T^2 E run. Words cannot show it (a masked block adds zeros).

Your four items are in the note: (1) formulas and numbers; (2) S: same gemm, atom, head-dim order, scale and mask order by the source; a circuit can wire one S and hold six; (3) flash computes all three gradients always, no grad mask (READ; I run a one-operand call in the second lease); (4) efficient kernel: next, second lease now.

Note: training-20261112T1655Z-finding-fused-backward-five-products-16-bits (notes 2d20ab2815). Capture art:6a0d7e798b0d04914013112c84cfde313b0e4ba84dc8fd0ac47491c75b5667d1.

