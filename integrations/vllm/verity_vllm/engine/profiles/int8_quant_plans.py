"""`engine.int8_fused`'s row quantizer plans: per GPU architecture (`int8_fused.arch`, from the device's compute capability)
and K, (M floor, num_warps, rows a program) rows, `benchmarks/hash/exactlin/ksweep.py --quant-only`'s gated winners over
16 warps a row; a launch takes the first row whose floor M reaches, else 16 warps and one row a program
(`int8_fused.quant_plan`).  A plan never changes a value: each row's arithmetic is the same in any plan."""
from __future__ import annotations

# The quantizer is ALU-bound on its per-element division, so fewer warps a row win once M fills the SMs.
QUANT_PLANS: dict[str, dict[int, tuple[tuple[int, int, int], ...]]] = {
    "sm_90": {2048: ((2048, 4, 1), (1024, 8, 1)), 4096: ((1024, 8, 1),), 8192: ((1024, 8, 1),)},
}

# The fused producers' plans (one row a program) per architecture, kernel and row width: (M floor, num_warps) rows, the
# first whose floor M reaches, `int8_producers_gpu.py`'s fastest, at the widths where that beats vLLM's op and the
# quantizer apart; the engine fuses only where a plan applies (`int8_fused.producer_warps`, `int8_producers`).  At 8 or 16
# warps the add-norm is three times slower than the two apart, at 1 warp twice as fast.  The silu at I 8,192 is faster at
# 16 warps up to M 640 and at 8 from M 768 on.  The silu at I 14,336 (Llama-3.1-8B, a masked 16,384-wide tile) is level
# with the two apart at every warps tried, so it isn't listed.
PRODUCER_WARPS: dict[str, dict[str, dict[int, tuple[tuple[int, int], ...]]]] = {
    "sm_90": {"norm": {2048: ((0, 1),), 4096: ((0, 1),)}, "silu": {8192: ((768, 8), (0, 16))}},
}
