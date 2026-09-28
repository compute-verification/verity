"""``benchmarks/tc_probe/tc_probe.py`` tile generation without a device (lane pin-bf16): the FP8 families of tc-probe-fp8 are byte-identical
for a fixed seed, the accumulator / adder families now exist for BF16 with shapes, dtypes and determinism, the adder-width ratios
bracket every BF16 pipeline's width so the survive/truncate boundary is observable, and the D15(b) floor probe cannot tell any
adder floor in [-152, -126] apart under the pinned Hopper FP8 model (the structural reason is tested on ``group_sum`` directly).
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.silicon import silicon as S
from verity.ml.circuits.silicon.models import group_sum
from verity.ml.circuits.silicon.term import Term, e4m3_product, fp32_term

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "uv.lock").exists())
SEED = 20260902
N_RANDOM = 64

# sha256[:16] of A || Bt || C for families(rng(SEED), N_RANDOM, fmt, M, K), recorded on main 156bf21 before pin-bf16 touched
# families(): the FP8 tile generation must stay byte-identical (the H100 / 4090 FP8 sweeps and canaries are keyed on these draws).
FP8_BASELINE = {
    ("e4m3", 16, 32): {
        "randn_zero_acc": "809bc060a317c7e3", "randn_random_acc": "f820887b9092ddbe", "uniform_bits": "aa40694d1545c47c",
        "subnormal": "aa217bde5c952b12", "cancellation": "e2c7e90eeb65f80b", "mixed_magnitude": "f6606bca7dcf3746",
        "near_overflow": "a239d74aa7689a1c", "signed_zero": "9c9099f476293844", "tiny_acc_zero_products": "6ee5f666c50dd6d3",
        "zero_products_nonzero_acc": "1c0e37b3fa25d7a5", "acc_only_lsb_walk": "6899e75285657c42", "specials": "6bb6adf3f07a4900",
        "adder_width_cancellation": "3f5cdac8b29e861e",
    },
    ("e5m2", 16, 32): {
        "randn_zero_acc": "2e2e24062a2d5a2c", "randn_random_acc": "1b36e6c525e712f9", "uniform_bits": "2726932187993b75",
        "subnormal": "8fe041275721c8a7", "cancellation": "20c4430d9039babe", "mixed_magnitude": "bff02d83a6c7168c",
        "near_overflow": "32ea194fc0d6fde1", "signed_zero": "b85834ff0941b5dd", "tiny_acc_zero_products": "6ee5f666c50dd6d3",
        "zero_products_nonzero_acc": "d369bfd0e658ea5e", "acc_only_lsb_walk": "6899e75285657c42", "specials": "edd7399f793a2b5f",
        "adder_width_cancellation": "86b0d535103911b1",
    },
    ("e4m3", 64, 32): {
        "randn_zero_acc": "39cddeaba67aa8d5", "randn_random_acc": "a659ca54b062798d", "uniform_bits": "63760938a116ca07",
        "subnormal": "bb3634fc46a12cc6", "cancellation": "c52e591e69524f99", "mixed_magnitude": "dad32dceecd95d04",
        "near_overflow": "b33f7ec00ec7ece0", "signed_zero": "8c33a6aef1140017", "tiny_acc_zero_products": "44363c1270674bae",
        "zero_products_nonzero_acc": "bb897966a751ca12", "acc_only_lsb_walk": "c2a29e09ba2d5b49", "specials": "e3b9e20e4cc9ffba",
        "adder_width_cancellation": "b74cfd4ff9245a3e",
    },
}
# the ten BF16 families that existed before pin-bf16, same provenance
BF16_BASELINE = {
    "randn_zero_acc": "2ee9898f47695ef8", "randn_random_acc": "40f9b47c15d1cceb", "uniform_bits": "8968208f734ca3ec",
    "subnormal": "84a044e4c2bce486", "cancellation": "72c235e327e45a46", "mixed_magnitude": "4a4d207b4afb87dc",
    "near_overflow": "d09a8a9068dc1304", "signed_zero": "f70141b62b27a4c7", "tiny_acc_zero_products": "3f847e8dc1e15d38",
    "zero_acc_tiny_cancellation": "252570b4f8f2455a",
}
CANONICAL = ("zero_products_nonzero_acc", "acc_only_lsb_walk", "specials", "adder_width_cancellation")


@pytest.fixture(scope="module")
def P():
    sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "tc_probe"))
    try:
        import tc_probe
    finally:
        sys.path.pop(0)
    return tc_probe


def _digests(P, fmt, M, K, seed=SEED, n_random=N_RANDOM):
    rng = np.random.default_rng(seed)
    out = {}
    for name, A, Bt, C, _ in P.families(rng, n_random, fmt, M, K):
        h = hashlib.sha256()
        h.update(A.tobytes()), h.update(Bt.tobytes()), h.update(C.tobytes())
        out[name] = (h.hexdigest()[:16], A, Bt, C)
    return out


@pytest.mark.parametrize("key", sorted(FP8_BASELINE))
def test_fp8_tile_generation_is_byte_identical_for_a_fixed_seed(P, key):
    fmt_name, M, K = key
    fmt = {"e4m3": P.E4M3, "e5m2": P.E5M2}[fmt_name]
    got = _digests(P, fmt, M, K)
    assert {n: v[0] for n, v in got.items() if n in FP8_BASELINE[key]} == FP8_BASELINE[key]
    assert [n for n in got if n not in FP8_BASELINE[key]] == ["fp8_floor_probe"]              # appended after the baseline families
    assert list(got)[: len(FP8_BASELINE[key])] == list(FP8_BASELINE[key])                    # in the same order


def test_bf16_families_keep_the_old_draws_and_add_the_canonical_set(P):
    got = _digests(P, P.BF16, 16, 16)
    assert {n: v[0] for n, v in got.items() if n in BF16_BASELINE} == BF16_BASELINE
    assert list(got)[len(BF16_BASELINE):] == list(CANONICAL)
    assert "fp8_floor_probe" not in got
    for name in CANONICAL:
        _, A, Bt, C = got[name]
        n = A.shape[0]
        assert A.shape == (n, 16, 16) and Bt.shape == (n, 8, 16) and C.shape == (n, 16, 8)
        assert A.dtype == Bt.dtype == np.uint16 and C.dtype == np.uint32
    assert got["acc_only_lsb_walk"][3].size == 64 * 1024 and got["zero_products_nonzero_acc"][3].size == 512 * 128
    assert got["specials"][1].shape[0] == 48 and got["adder_width_cancellation"][1].shape[0] == 32
    again = _digests(P, P.BF16, 16, 16)
    assert {n: v[0] for n, v in again.items()} == {n: v[0] for n, v in got.items()}          # deterministic
    other = _digests(P, P.BF16, 16, 16, seed=SEED + 1)
    assert other["zero_products_nonzero_acc"][0] != got["zero_products_nonzero_acc"][0]        # ... and seeded
    assert other["adder_width_cancellation"][0] == got["adder_width_cancellation"][0]          # designed tiles do not depend on the seed
    assert other["specials"][0] == got["specials"][0]


def test_bf16_accumulator_families_have_zero_products_and_the_specials_table_is_bf16(P):
    got = _digests(P, P.BF16, 16, 16)
    for name in ("zero_products_nonzero_acc", "acc_only_lsb_walk"):
        _, A, Bt, C = got[name]
        assert not A.any()                                                                     # every product is +-0 * x
        assert np.count_nonzero(C & 0x7FFFFFFF) > 0.99 * C.size
    _, A, Bt, C = got["specials"]
    words = set(np.unique(A).tolist()) | set(np.unique(Bt).tolist())
    assert {0x3F80, 0x7F80, 0x7FC0, 0x0001, 0x7F7F} <= words and max(words) <= 0xFFFF
    assert "bf16" in P.SPECIAL_OPERANDS and P.SPECIAL_OPERANDS["bf16"]["one"] == 0x3F80


@pytest.mark.parametrize("pipeline, boundary", [(S.AMPERE_BF16_M16N8K16, -24), (S.HOPPER_BF16_M16N8K16, -25), (S.ADA_BF16_M16N8K16, -24)])
def test_bf16_adder_width_ratios_bracket_the_pipeline_width(P, pipeline, boundary):
    """The pinned pipeline keeps the residual iff ratio >= -(width - 1) inside the pair's group: that boundary must lie strictly inside
    the probed ratio range, with the grouping visible (Ampere's second group of 8 keeps every residual)."""
    pas, ratios = P.adder_width_parameters(P.BF16)
    assert boundary - 1 in ratios and boundary in ratios and min(ratios) < boundary - 3 and max(ratios) > boundary + 3
    assert all(abs(2 * pa) <= 126 for pa in pas) and min(2 * pa + r for pa in pas for r in ratios) >= -149
    A, Bt, C, meta = P.adder_width_tiles(P.BF16, 16, 16)
    model = P.Model(pipeline)
    D = np.zeros(C.shape, np.uint32)
    for t in range(A.shape[0]):
        for i in range(16):
            for j in range(8):
                D[t, i, j] = model.evaluate(int(C[t, i, j]), A[t, i].tolist(), Bt[t, j].tolist())
    summary = P.adder_width_summary(meta, D)
    for variant in ("prod", "acc"):
        first_group = {s: v for s, v in summary[variant].items() if int(s) < pipeline.groups[0]}
        assert first_group and all(v["min_surviving_ratio"] == boundary and v["max_truncated_ratio"] == boundary - 1 and not v["other"] for v in first_group.values())
        later = {s: v for s, v in summary[variant].items() if int(s) >= pipeline.groups[0]}
        assert all(v["max_truncated_ratio"] is None and v["min_surviving_ratio"] == min(ratios) for v in later.values())
        assert bool(later) == (len(pipeline.groups) > 1)


def test_fp8_adder_width_parameters_are_hawkeyes(P):
    assert P.adder_width_parameters(P.E4M3) == ((0, 4, 8), list(range(-26, -4)))


def test_sm120_bf16_instruction_is_registered(P):
    ins = P.INSTRUCTIONS["sm120.mma.m16n8k16.bf16"]
    assert ins.arch == "sm_120a" and ins.fmt is P.BF16 and ins.k == 16
    assert [m.pipeline.name for m in ins.models] == [S.HOPPER_BF16_M16N8K16.name] and ins.models[0].declared


def test_floor_probe_tiles_are_designed_and_deterministic(P):
    A, Bt, C, meta = P.floor_probe_tiles(P.E4M3, 64, 32)
    A2, Bt2, C2, _ = P.floor_probe_tiles(P.E4M3, 64, 32)
    assert (A == A2).all() and (Bt == Bt2).all() and (C == C2).all()
    assert A.shape == (7, 64, 32) and Bt.shape == (7, 8, 32) and C.shape == (7, 64, 8) and A.dtype == np.uint8 and C.dtype == np.uint32
    kinds = [t["kind"] for t in meta["tiles"]]
    assert kinds == ["acc_bits", "acc_bits", "acc_bits", "min_products", "min_products", "cancelling_pair", "product_window"]
    acc_bits = C[0].ravel()
    assert {1 << b for b in range(24)} <= set(acc_bits.tolist()) and {(1 << 31) | (1 << b) for b in range(24)} <= set(acc_bits.tolist())
    assert not A[:3].any()                                                                      # acc_bits: zero products
    assert meta["product_exponent"] == -18 and meta["floors"] == list(range(-152, -125))
    assert not Bt[0].any() and Bt[2].any()                                                      # zero_b, zero_b, random_b


def test_no_adder_floor_in_the_h100_fit_range_is_observable_under_the_hopper_fp8_model(P):
    """D15(b): every floor in [-152, -126] gives the same words on the floor-probe tiles (a small M=16 copy: 896 elements, five floors
    spanning the range); the structural reason -- with any nonzero term the group maximum exponent is >= -126, the FP32 subnormal
    grid, and with none the sum is zero -- is checked on group_sum with random terms."""
    base = S.HOPPER_E4M3_K32
    A, Bt, C, meta = P.floor_probe_tiles(P.E4M3, 16, 32)
    words = {}
    for floor in (-152, -145, -139, -133, -126):
        model = P.Model(dataclasses.replace(base, name=f"{base.name}@{floor}", zero_exponent=floor))
        D = np.zeros(C.shape, np.uint32)
        for t in range(A.shape[0]):
            for i in range(16):
                for j in range(8):
                    D[t, i, j] = model.evaluate(int(C[t, i, j]), A[t, i].tolist(), Bt[t, j].tolist())
        words[floor] = D
    assert all((words[f] == words[-139]).all() for f in words)
    summary = P.floor_probe_summary(meta, words[-139])
    assert summary["min_products+"]["elements_equal_to_n_p"] == 128 and summary["cancelling_pair"]["elements_equal_to_plus_zero"] == 128
    assert summary["product_window"] == {"max_surviving_k": 7, "min_truncated_k": 8, "unclassified": 0}
    # group_sum: the floor is max'ed with the exponents of the nonzero terms; E4M3 products sit at exponent >= -12 and FP32
    # accumulators at >= -126, so no floor <= -126 can be the maximum unless every term is zero (and then the sum is ZERO).
    assert e4m3_product(0x01, 0x01).exponent == -12 and fp32_term(0x00000001).exponent == -126
    rng = np.random.default_rng(1)
    for _ in range(200):
        try:
            acc = fp32_term(int(rng.integers(0, 1 << 31)) | (int(rng.integers(0, 2)) << 31)) if rng.random() < 0.7 else Term(False, -126, int(rng.integers(0, 1 << 23)))
            prods = [e4m3_product(int(rng.integers(0, 256)), int(rng.integers(0, 256))) for _ in range(int(rng.integers(0, 5)))]
        except Exception:  # noqa: BLE001 - a NaN operand / non-finite accumulator in the random draw
            continue
        terms = [acc, *prods]
        ref = group_sum(terms, width=base.width, zero_exponent=-139)
        assert all(group_sum(terms, width=base.width, zero_exponent=f) == ref for f in (-152, -140, -126))
    assert group_sum([fp32_term(0x00000400)], width=base.width, zero_exponent=-125) != group_sum([fp32_term(0x00000400)], width=base.width, zero_exponent=-126)


def test_floor_probe_is_fp8_only_and_last(P):
    names = [n for n, *_ in P.families(np.random.default_rng(0), N_RANDOM, P.E5M2, 64, 32)]
    assert names[-1] == "fp8_floor_probe" and names[-5:-1] == list(CANONICAL)
    assert "fp8_floor_probe" not in P.UNASSERTED_FAMILIES                                       # asserted against the pinned model
