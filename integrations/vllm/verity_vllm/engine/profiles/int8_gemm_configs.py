"""`engine.int8_fused`'s GEMM configs: per GPU architecture (`int8_fused.arch`, from the device's compute capability) and
(N, K), (m_max, config) rows, `benchmarks/hash/exactlin/ksweep.py`'s gated winners where they beat vLLM's tiles; a call
takes the first row with M <= m_max, else vLLM's tiles (`int8_fused.pick`).  A config is `int8_fused.Cfg`'s text: tile
BMxBNxBK, warps, stages, split-K and the TMA mode.  The config never changes a value (int32 sums and a per-element
epilogue), only the time."""
from __future__ import annotations

# Per architecture and (N, K): (m_max, config) rows, ksweep.py's winners where they beat vLLM's tiles (best.json).
CONFIGS: dict[str, dict[tuple[int, int], list[tuple[int, str]]]] = {
    "sm_120": {  # RTX PRO 6000: M <= 64; M 256 to 8192 (pointer and TMA)
        (6144, 4096): [(1, "16x64x128:w4:s5:k1"), (16, "16x32x128:w4:s4:k1"), (32, "32x32x128:w4:s4:k1"),
                       (64, "32x64x128:w4:s4:k1"), (256, "128x64x128:w4:s3:k1:t1"), (2048, "128x128x64:w4:s3:k1:t1"),
                       (8192, "128x128x64:w4:s3:k1:t1")],
        (4096, 4096): [(1, "16x32x256:w4:s5:k1"), (16, "16x16x256:w4:s3:k1"), (32, "32x32x256:w4:s4:k1"),
                       (64, "32x64x256:w8:s4:k1"), (256, "128x64x128:w4:s4:k1"), (2048, "128x128x64:w4:s3:k1:t1"),
                       (8192, "128x128x64:w4:s3:k1:t1")],
        (28672, 4096): [(1, "16x32x128:w4:s4:k1"), (16, "16x32x128:w4:s4:k1"), (32, "32x32x256:w4:s2:k1"),
                        (64, "64x64x128:w4:s3:k1"), (256, "128x64x64:w4:s3:k1:t1"), (2048, "128x128x64:w4:s4:k1:t1"),
                        (8192, "128x128x64:w4:s3:k1:t1")],
        (4096, 14336): [(1, "16x32x512:w4:s4:k1"), (16, "16x32x512:w4:s4:k1"), (32, "32x32x512:w8:s4:k1"),
                        (64, "32x64x512:w8:s3:k1"), (256, "128x64x128:w4:s5:k1"), (2048, "128x128x64:w4:s3:k1:t1"),
                        (8192, "128x128x64:w4:s3:k1:t1")],
        (24576, 4096): [(1, "16x32x128:w4:s4:k1"), (16, "16x32x128:w4:s3:k1"), (32, "32x32x256:w4:s2:k1"),
                        (64, "32x64x128:w8:s2:k1"), (256, "128x128x64:w4:s4:k1:t1"), (2048, "128x128x64:w4:s3:k1:t1"),
                        (8192, "128x128x64:w4:s3:k1:t1")],
        (4096, 12288): [(1, "16x32x512:w4:s5:k1"), (16, "16x32x512:w4:s4:k1"), (32, "32x32x512:w4:s4:k1"),
                        (64, "32x64x512:w8:s3:k1"), (256, "128x64x128:w4:s5:k1"), (2048, "128x128x64:w4:s3:k1:t1"),
                        (8192, "128x128x64:w4:s3:k1:t1")],
    },
    "sm_90": {  # H200: M <= 32; M 64 to 8192 (pointer and TMA); the 1B's M 8192 to 32768 (pointer and TMA)
        (6144, 4096): [(1, "16x64x512:w4:s4:k1"), (16, "16x64x512:w4:s4:k1"), (32, "32x32x512:w4:s3:k1"),
                       (64, "64x32x256:w4:s4:k1:t1"), (256, "128x128x256:w4:s3:k1:t1"),
                       (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (4096, 4096): [(1, "16x32x512:w4:s4:k1"), (16, "16x32x512:w4:s4:k1"), (32, "32x32x256:w4:s5:k1"),
                       (64, "64x16x256:w4:s5:k1:t1"), (256, "128x64x256:w4:s4:k1:t1"),
                       (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (28672, 4096): [(1, "16x32x256:w4:s3:k1"), (16, "16x32x128:w4:s3:k1"), (32, "32x128x256:w4:s3:k1"),
                        (64, "64x128x128:w8:s4:k1:t1"), (256, "128x128x128:w4:s3:k1:t1"),
                        (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (4096, 14336): [(1, "16x64x256:w4:s3:k4"), (16, "16x32x512:w4:s3:k2"), (32, "32x32x512:w4:s5:k1"),
                        (64, "64x32x256:w4:s5:k1:t2"), (256, "128x64x256:w4:s4:k1:t1"),
                        (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (24576, 4096): [(1, "16x64x256:w4:s3:k1"), (16, "16x64x256:w4:s3:k1"), (32, "32x64x256:w4:s3:k1"),
                        (64, "64x64x128:w4:s4:k1:t1"), (256, "128x64x128:w4:s3:k1:t1"),
                        (2048, "128x128x128:w4:s3:k1:t1"), (8192, "128x128x128:w4:s3:k1:t1")],
        (4096, 12288): [(1, "16x32x256:w4:s3:k4"), (16, "16x32x512:w4:s4:k2"), (32, "32x32x512:w4:s5:k1"),
                        (64, "64x32x256:w4:s5:k1:t2"), (256, "64x128x256:w8:s4:k1:t2"),
                        (2048, "128x128x128:w8:s3:k1:t1"), (8192, "128x128x128:w8:s3:k1:t1")],
        # Llama-3.2-1B: vLLM's tiles to M 4096 (not swept); M 8192 to 32768 swept (pointer and TMA, the zy path gated)
        (3072, 2048): [(64, "64x64x256:w4:s3:k1"), (128, "64x128x128:w4:s3:k1"), (4096, "128x128x128:w4:s3:k1"),
                       (8192, "128x128x128:w4:s3:k1:t1"), (16384, "128x128x128:w4:s3:k1:t1"),
                       (32768, "128x128x128:w4:s3:k1:t1")],
        (2048, 2048): [(64, "64x64x256:w4:s3:k1"), (128, "64x128x128:w4:s3:k1"), (4096, "128x128x128:w4:s3:k1"),
                       (8192, "128x128x128:w4:s3:k1:t1"), (16384, "128x128x128:w4:s3:k1:t1"),
                       (32768, "128x128x128:w4:s3:k1:t1")],
        (16384, 2048): [(32, "64x128x256:w4:s3:k1"), (64, "64x64x256:w4:s3:k1"), (128, "64x128x128:w4:s3:k1"),
                        (4096, "128x128x128:w4:s3:k1"), (8192, "128x128x128:w4:s3:k1:t1"),
                        (16384, "128x128x128:w4:s3:k1:t1"), (32768, "128x128x128:w4:s3:k1:t1")],
        (2048, 8192): [(64, "64x64x256:w4:s3:k1"), (128, "64x128x128:w4:s3:k1"), (4096, "128x128x128:w4:s3:k1"),
                       (8192, "128x128x128:w4:s3:k1:t1"), (16384, "128x128x128:w4:s3:k1:t1"),
                       (32768, "128x128x128:w4:s3:k1:t1")],
    },
    "sm_100": {  # B200: M <= 64; M 256 to 8192 (pointer and TMA)
        (6144, 4096): [(1, "16x64x512:w4:s4:k1"), (16, "16x64x512:w4:s4:k1"), (32, "32x16x256:w4:s4:k1"),
                       (64, "64x64x512:w4:s4:k1"), (256, "64x64x256:w4:s3:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                       (8192, "128x64x128:w4:s3:k1:t1")],
        (4096, 4096): [(1, "16x32x512:w4:s4:k1"), (16, "16x32x512:w4:s4:k1"), (32, "32x32x512:w4:s4:k1"),
                       (64, "64x32x512:w4:s4:k1"), (256, "64x64x256:w4:s3:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                       (8192, "128x128x128:w4:s3:k1:t1")],
        (28672, 4096): [(1, "16x64x256:w4:s3:k1"), (16, "16x64x256:w4:s3:k1"), (32, "32x128x256:w4:s3:k1"),
                        (64, "64x128x256:w4:s3:k1"), (256, "128x64x128:w4:s4:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                        (8192, "128x128x128:w4:s4:k1:t1")],
        (4096, 14336): [(1, "16x64x256:w4:s5:k4"), (16, "16x64x512:w4:s3:k4"), (32, "32x64x256:w4:s4:k4"),
                        (64, "64x64x512:w4:s4:k2"), (256, "64x64x256:w4:s4:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                        (8192, "128x128x128:w4:s4:k1:t1")],
        (24576, 4096): [(1, "16x64x256:w4:s4:k1"), (16, "16x64x256:w4:s4:k1"), (32, "32x64x256:w4:s3:k1"),
                        (64, "64x64x256:w4:s3:k1"), (256, "64x128x128:w4:s4:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                        (8192, "128x128x128:w4:s3:k1:t1")],
        (4096, 12288): [(1, "16x64x256:w4:s5:k4"), (16, "16x64x512:w4:s3:k4"), (32, "32x64x256:w4:s5:k4"),
                        (64, "64x32x512:w4:s4:k1"), (256, "64x64x256:w4:s3:k1:t1"), (2048, "128x64x128:w4:s3:k1:t1"),
                        (8192, "128x128x128:w4:s3:k1:t1")],
    },
}

# `int8_fused.product`'s sm_90a GEMM (`engine.int8_wgmma`) per architecture and (N, K): (m_min, config) rows, the last whose
# m_min <= M taken (`int8_wgmma.row`); a shape with no row runs CONFIGS' Triton plan.  A config is `int8_wgmma.WgCfg`'s
# text: kernel, raster group, CTAs.  `benchmarks/hash/exactlin/wgbench.py`'s gated winners against the Triton plan, by MACs a
# SM clock at a served step's clock (`--rest 1.0`) at M 8,192, 16,384 and 32,768 (r20261028-063549-1b46); below M 8,192 the
# Triton plan is unmeasured against it.  WGMMA_ARCH is the target `int8_wgmma.cu` is built for (wgmma and setmaxnreg are
# sm_90a's).
WGMMA_ARCH = "sm_90a"
WGMMA: dict[str, dict[tuple[int, int], list[tuple[int, str]]]] = {
    "sm_90": {  # H200, Llama-3.2-1B
        (3072, 2048): [(8192, "pp_128x128:g8"), (32768, "ppm2_128x128:g8")],
        (2048, 2048): [(8192, "pp_128x128:g8")],
        (16384, 2048): [(8192, "pp_128x128:g8")],
        (2048, 8192): [(8192, "pp_128x128:g8"), (16384, "co_128x256:g8"), (32768, "ppm2_128x128:g8")],
    },
}
