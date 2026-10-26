"""`kernels.deepgemm_schedule` against DeepGEMM 8b1392b9's `get_best_config<SM90ArchSpec>`, pinned at the capture box's 132 SMs:
the stage formula at its documented points, the m-dependence (only through cdiv(m, 16), so every boundary is 16t + 1, checked
exhaustively on one shape), the boundary lists of every served TP8 shape and the grouped einsum, the sweep (base, both sides of
every boundary, the endpoint), and the 20 template-argument strings `generate_impl` would write."""
from __future__ import annotations

import pytest

from verity.ml.definitions.numerics import deepgemm_schedule as S

SMS = 132
#: (name, K, N) of every FP8 linear the deepgemm capture runs (DeepSeek-V4-Pro's and GLM-5.3's TP8 shapes)
LINEARS = (("dsv4/fused_wqa_wkv", 7168, 2048), ("dsv4/wq_b", 1536, 8192), ("dsv4/indexer.wq_b", 1536, 8192),
           ("dsv4/wo_b", 2048, 7168), ("dsv4/se.gate_up", 7168, 768), ("dsv4/se.down", 384, 7168),
           ("glm53/fused_qkv_a", 6144, 2624), ("glm53/q_b", 2048, 2048), ("glm53/kv_b", 512, 3584),
           ("glm53/o_proj", 2048, 6144), ("glm53/indexer.wq_b", 2048, 4096), ("glm53/mlp.gate_up", 6144, 3072),
           ("glm53/mlp.down", 1536, 6144), ("glm53/se.gate_up", 6144, 512), ("glm53/se.down", 256, 6144))
#: every m in (32, 1024] where the chosen layout changes, per shape, at 132 SMs: where the capture's sweep must sit on both sides
BOUNDARIES = {"dsv4/fused_wqa_wkv": [33, 65, 129, 193, 257, 321, 385, 513, 641, 769, 833],
              "dsv4/wq_b": [33, 65, 129, 193, 257, 321, 385, 513, 641, 769],
              "dsv4/indexer.wq_b": [33, 65, 129, 193, 257, 321, 385, 513, 641, 769],
              "dsv4/wo_b": [33, 65, 129, 193, 257, 321, 385, 513, 641, 769],
              "dsv4/se.gate_up": [33, 129, 321, 513, 705, 833],
              "dsv4/se.down": [33, 65, 129, 193, 257, 321, 385, 513, 641, 769],
              "glm53/fused_qkv_a": [33, 65, 129, 193, 257, 321, 385, 513, 577, 641, 705, 769, 897],
              "glm53/q_b": [33, 65, 129, 193, 257, 321, 385, 513, 641, 769, 833],
              "glm53/kv_b": [33, 65, 129, 193, 257, 321, 385, 513, 577, 641, 705, 769, 833],
              "glm53/o_proj": [33, 65, 129, 193, 257, 385, 513, 641, 769],
              "glm53/indexer.wq_b": [33, 65, 129, 193, 257, 321, 385, 449, 513, 641, 769, 897],
              "glm53/mlp.gate_up": [33, 65, 129, 193, 257, 321, 385, 513, 577, 641, 769],
              "glm53/mlp.down": [33, 65, 129, 193, 257, 321, 385, 513, 641, 769],
              "glm53/se.gate_up": [33, 257, 513, 769],
              "glm53/se.down": [33, 65, 129, 193, 257, 321, 385, 513, 577, 641, 769]}
#: the grouped o-projection einsum (`sm90_fp8_bmm`, GemmType::Batched): H = 2, K = 4096, N = 1024, swept from M = 1
EINSUM = dict(num_groups=2, batched=True)
EINSUM_K, EINSUM_N = 4096, 1024
EINSUM_BOUNDARIES = [17, 33, 65, 129, 193, 257, 321, 385, 449, 513, 641, 769]


def test_num_stages():
    """The smem pipeline formula: capacity 232448, aligned cd/sfa, kNumMaxStages 16 (sm90.hpp get_pipeline_config)."""
    assert S.num_stages(32, 16, 7168) == 16
    assert S.num_stages(64, 128, 7168) == 8
    assert S.num_stages(128, 128, 7168) == 5
    assert S.num_stages(256, 128, 1536) == 3
    assert S.num_stages(128, 192, 1536) == 4
    # block_n > 128 keeps two sfb words per k-block (128 % block_n != 0)
    assert S.num_stages(64, 192, 512) == 6


def test_m_dependence_is_cdiv_16():
    """On one shape, every m in one ceil(m/16) step picks the same layout: the boundary scan may step 16t + 1."""
    K, N = 2048, 2048
    for t in range(0, 64):
        lo = 16 * t + 1
        configs = {S.best_config(m, N, K, num_sms=SMS) for m in range(lo, min(lo + 16, 1025))}
        assert len(configs) == 1, f"configs differ inside ({lo}, {lo + 15}]"


@pytest.mark.parametrize("name,K,N", LINEARS, ids=[n for n, _, _ in LINEARS])
def test_linear_boundaries(name, K, N):
    """Each served shape's layout changes exactly at the pinned m (fp8_gemm_nt computes [M,K] x [N,K] -> [M,N]: n = N, k = K)."""
    assert S.boundaries(N, K, num_sms=SMS, lo=32, hi=1024) == BOUNDARIES[name]


def test_einsum_boundaries():
    assert S.boundaries(EINSUM_N, EINSUM_K, num_sms=SMS, lo=1, hi=1024, **EINSUM) == EINSUM_BOUNDARIES


def test_small_m_configs():
    """The small-block candidates: m <= 16 admits block_m 16, m <= 32 admits 32; the batched einsum takes them at decode sizes."""
    assert S.best_config(1, EINSUM_N, EINSUM_K, num_sms=SMS, **EINSUM).block_m == 16
    assert S.best_config(17, EINSUM_N, EINSUM_K, num_sms=SMS, **EINSUM).block_m == 32
    assert S.best_config(32, 2048, 7168, num_sms=SMS) == S.Config(block_m=32, block_n=16, cluster_m=1, cluster_n=1,
                                                                 num_stages=16, swizzle_cd=32, num_math_threads=128)


def test_endpoint_config():
    """At the 1024 endpoint fused_wqa_wkv runs the large tile, 256 math threads."""
    assert S.best_config(1024, 2048, 7168, num_sms=SMS) == S.Config(block_m=128, block_n=128, cluster_m=1, cluster_n=1,
                                                                    num_stages=5, swizzle_cd=128, num_math_threads=256)


def test_multicast_appears_only_past_one_wave():
    """A multicast layout with at most one wave scores INT64_MAX, so small m never multicasts; fused_wqa_wkv's 769..832 does."""
    for m in (33, 64, 256, 512):
        c = S.best_config(m, 2048, 7168, num_sms=SMS)
        assert (c.cluster_m, c.cluster_n) == (1, 1)
    c = S.best_config(769, 2048, 7168, num_sms=SMS)
    assert (c.cluster_m, c.cluster_n) == (2, 1)


def test_batched_never_multicasts():
    """GemmType::Batched disables multicast whatever the wave count."""
    for m in (1, 64, 512, 1024):
        c = S.best_config(m, EINSUM_N, EINSUM_K, num_sms=SMS, **EINSUM)
        assert (c.cluster_m, c.cluster_n) == (1, 1)


def test_sweep_ms():
    """The sweep is the base (capped at hi), both sides of every boundary, and the endpoint, sorted without duplicates."""
    base = (1, 8, 16, 31, 32, 64, 256)
    ms = S.sweep_ms(base, 2048, 7168, num_sms=SMS, lo=32, hi=1024)
    assert ms == tuple(sorted(set(base) | {1024} | {x for b in BOUNDARIES["dsv4/fused_wqa_wkv"] for x in (b - 1, b)}))
    assert ms == (1, 8, 16, 31, 32, 33, 64, 65, 128, 129, 192, 193, 256, 257, 320, 321, 384, 385, 512, 513,
                  640, 641, 768, 769, 832, 833, 1024)
    short = S.sweep_ms(base, 2048, 7168, num_sms=SMS, lo=32, hi=64)
    assert short == (1, 8, 16, 31, 32, 33, 64)


def test_template_args():
    """The 20 strings of `sm90_fp8_gemm_1d2d_impl`'s instantiation: m runtime ("0"), n and k compiled, the layout's own fields."""
    cfg = S.best_config(1024, 2048, 7168, num_sms=SMS)
    args = S.template_args(2048, 7168, num_groups=1, num_sms=SMS, gemm_type="Normal", config=cfg)
    assert args == ["cute::UMMA::Major::K", "0", "2048", "7168", "1", "128", "128", "128", "128", "128", "128",
                    "5", "128", "256", "1", "false", "132", "GemmType::Normal", "cutlass::bfloat16_t",
                    "epilogue::transform::EpilogueIdentity"]
    cfg = S.best_config(1, EINSUM_N, EINSUM_K, num_sms=SMS, **EINSUM)
    args = S.template_args(EINSUM_N, EINSUM_K, num_groups=2, num_sms=SMS, gemm_type="Batched", config=cfg)
    assert args[:8] == ["cute::UMMA::Major::K", "0", "1024", "4096", "2", "16", "16", "128"]
    assert args[13:18] == ["128", "1", "false", "132", "GemmType::Batched"]
    cfg = S.best_config(769, 2048, 7168, num_sms=SMS)
    args = S.template_args(2048, 7168, num_groups=1, num_sms=SMS, gemm_type="Normal", config=cfg)
    assert args[5:8] == ["64", "112", "128"] and args[14] == "2" and args[15] == "false" and args[13] == "128"
