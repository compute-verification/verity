# s8_orbit: one Strassen level of the exact int8 GEMM on the H200 (sm_90a)

O = X·Wᵀ in int32, bit for bit, with seven tensor-core products of half size in place of eight. X (M×K) is A and W (N×K)
is Bᵀ, both int8. Every Strassen form has at most two terms, so a form is its int8 wrap plus 256 times a carry in
{-1, 0, 1}, and each product is the wraps' product plus sparse carry corrections (`orbit.py`'s docstring has the
identity). B's forms and carries are made once a weight; A's are made per call, in shared memory.

- `orbit.py`: the numpy reference: the forms (`A_FORMS`, `B_FORMS`, `C_SUMS`), B's carry metadata, and `orbit(x, w)`.
- `cpu_check.py`: the reference against the plain int8 GEMM on a checkpoint's linears (fused qkv, o, fused gate_up,
  down), at three kinds of activation, with B's carry densities and metadata sizes. Exits 4 on any difference.
- `s8_orbit.cu`, `s8_orbit_acc.h`: the kernel. `s8orbit_<BN>` (BN 64, 80 or 96) runs the seven products as `wgmma
  m64nBNk32 .s32.s8.s8` on the wrapped forms, persistent, with no carry corrections. `s8orbit_bfix` then adds B's
  carries' corrections from lists made once a weight. `build.sh` builds the cubin off the card and refuses a spill,
  local memory, a serialized wgmma, or an orbit kernel without IGMMA.
- `orbit_gpu.py`: the cubin from Python on torch tensors, and the plain int8 GEMM entry:

  ```python
  p = orbit_gpu.prepare(w)        # once at load: w [N, K] int8 -> B's forms, carries and correction lists
  y = orbit_gpu.int_mm(x, p)      # x [M, K] int8 -> x·wᵀ [M, N] int32, exact (the cubin at $S8_ORBIT_CUBIN)
  ```

  `int_mm` runs the orbit and `s8orbit_bfix`, and adds A's carry terms only when the kernel's flags say A's forms
  carried.
- `gate.py`: the kernel against `torch._int_mm` (cuBLASLt's exact s32 GEMM): no-carry shapes with tails, full-range
  operands with the corrections, the B correction pass, and the entry at M 1, 7, 34 and 2,048.
- `bench.py`: the wall time at M 16,384 on a checkpoint's four linears against `torch._int_mm` and s8_linear's
  `wgmma.scaled_mm`, alone and with SHA-512 rows hashed beside it on the SMs the orbit leaves free. It records θ against
  989.4 T int8 MACs a second at 1,830 MHz, the SM clock, and κ: `s8orbit_bfix`'s time per correction MAC in tensor-MAC
  time.
- `subset.py`: one layer's linears in one safetensors file, for a node without the snapshot.
- `run.sh`: builds both cubins in the run, then runs the GPU steps under gpu-lease.

On a GPU node: `bash run.sh "gate.py --cubin @CUBIN@ --snapshot DIR" "bench.py --cubin @CUBIN@ --snapshot DIR
--wgmma-cubin @WGMMA@ --sha"` (`run.sh`'s header has the rest). The CPU tests (`tests/`) hold the reference and the
correction lists to the plain product.
