"""`engine.int8_fused`'s row quantizer plans: per GPU architecture (`int8_fused.arch`, from the device's compute capability)
and K, (M floor, num_warps, rows a program) rows, `benchmarks/hash/exactlin/ksweep.py --quant-only`'s gated winners over
16 warps a row; a launch takes the first row whose floor M reaches, else 16 warps and one row a program
(`int8_fused.quant_plan`).  A plan never changes a value: each row's arithmetic is the same in any plan."""
from __future__ import annotations

# The quantizer is ALU-bound on its per-element division, so fewer warps a row win once M fills the SMs.
QUANT_PLANS: dict[str, dict[int, tuple[tuple[int, int, int], ...]]] = {
    "sm_90": {2048: ((2048, 4, 1), (1024, 8, 1)), 4096: ((1024, 8, 1),), 8192: ((1024, 8, 1),)},
}
