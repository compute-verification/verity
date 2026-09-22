"""Hawkeye's structured test sequence on one ``mma.sync.m16n8k16`` (GPU host only).

Ports ``experiments/GPU_reproduction.py`` of https://github.com/badasherez/gpu-simulator
(Badash, Boneh, Komargodski, Srivastava, arXiv 2603.20421) onto the single-instruction
probe of :mod:`verity_vllm_numerics.probe`, for BF16 and FP16, and evaluates every probe
tile through the CPU model too, so each structured test is simultaneously a
characterisation of the silicon and a check of the model.

Tests (per dtype): fragment layout; full-precision product; exhaustive neutral-subgroup
search over {acc, 0..15} (group structure, accumulator position); internal adder width
per residual slot; shift-rounding mode (5 candidate-elimination probes); normalisation
overflow; final-summation rounding; subnormal handling; extended-range accumulation and
minimal max exponent (BF16); accumulator-magnitude and cross-group cancellation probes;
and the paper's introductory vector ``[65504, 1, -65504, 1] . [1, 0.001, 1, 0.001]``.

Usage: python -m verity_vllm_numerics.hawkeye_sequence --out results/hawkeye_sequence.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from itertools import combinations

import numpy as np

from . import probe as G
from .mma import MODELS, TCModel, model_for, tc_dot

K = 16


class Probe:
    """Collects (acc, a, b) records, runs them in one GPU call, evaluates the model on each."""

    def __init__(self, dtype: str, model: TCModel):
        self.dtype = dtype
        self.model = model
        self.records: list[tuple[int, list[int], list[int]]] = []

    def add(self, acc: float | int, a: dict[int, float] | list, b: dict[int, float] | list, *, raw=False) -> int:
        """Queue one record; ``a``/``b`` map k-slot -> value (fp32 floats or raw words if ``raw``)."""

        aw = [0] * K
        bw = [0] * K
        if isinstance(a, dict):
            for k_, v in a.items():
                aw[k_] = int(v) if raw else G.scalar_operand(self.dtype, v)
            for k_, v in b.items():
                bw[k_] = int(v) if raw else G.scalar_operand(self.dtype, v)
        else:
            aw = [int(v) if raw else G.scalar_operand(self.dtype, v) for v in a] + [0] * (K - len(a))
            bw = [int(v) if raw else G.scalar_operand(self.dtype, v) for v in b] + [0] * (K - len(b))
        acc_bits = int(acc) if raw else int(G.f32_bits(np.array([acc], np.float32))[0])
        self.records.append((acc_bits, aw, bw))
        return len(self.records) - 1

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.records)
        A = np.zeros((n, 16, K), np.uint16)
        Bt = np.zeros((n, 8, K), np.uint16)
        C = np.zeros((n, 16, 8), np.uint32)
        for i, (acc, a, b) in enumerate(self.records):
            A[i, 0, :] = a
            Bt[i, 0, :] = b
            C[i, 0, 0] = acc
        D = G.mma_tiles(self.dtype, A, Bt, C)[:, 0, 0].copy()
        M = np.zeros(n, np.uint32)
        for i, (acc, a, b) in enumerate(self.records):
            try:
                M[i] = tc_dot(self.model, acc, a, b)
            except Exception:  # noqa: BLE001 - outside the modelled domain
                M[i] = 0xFFFFFFFF
        return D, M


def f(bits: int) -> float:
    return float(np.array([bits], np.uint32).view(np.float32)[0])


def hx(bits: int) -> str:
    return f"{int(bits):08x}"


def layout_test(dtype: str, rng: np.random.Generator, n: int = 1024) -> dict:
    a = rng.integers(-3, 4, size=(n, 16, K)).astype(np.float32)
    b = rng.integers(-3, 4, size=(n, 8, K)).astype(np.float32)
    c = rng.integers(-8, 9, size=(n, 16, 8)).astype(np.float32)
    want = c + np.einsum("nik,njk->nij", a, b)
    got = G.bits_f32(G.mma_tiles(dtype, G.to_operand(dtype, a), G.to_operand(dtype, b), G.f32_bits(c)))
    return {"tiles": n, "elements": n * 128, "mismatches": int(np.sum(got != want))}


def neutral_subgroups_exhaustive(dtype: str, model: TCModel) -> dict:
    """Hawkeye ``find_accumulation_group_size``: every subset of {acc(-1), 0..15} with >= 2 elements, batched."""

    v_large, v_small = 2.0**20, 2.0**-20
    sl, ss = math.sqrt(v_large), math.sqrt(v_small)
    subsets = [list(s) for size in range(2, 18) for s in combinations(range(-1, 16), size)]
    n = len(subsets)
    A = np.zeros((2 * n, 16, K), np.float32)
    B = np.zeros((2 * n, 8, K), np.float32)
    C = np.zeros((2 * n, 16, 8), np.float32)
    for idx, S in enumerate(subsets):
        Sset = set(S)
        acc_in = -1 in Sset
        for kk in range(K):
            v = sl if kk in Sset else ss
            A[idx, 0, kk] = v
            B[idx, 0, kk] = v
        C[idx, 0, 0] = v_large if acc_in else v_small
        m = len(S)
        neg = (m + 1) // 2
        for i in range(neg):
            j = S[i]
            if j == -1:
                C[idx, 0, 0] *= -1.0
            else:
                A[idx, 0, j] = -sl
        if m % 2 == 1:
            j = S[neg]
            if j == -1:
                C[idx, 0, 0] *= 2.0
            else:
                A[idx, 0, j] = 2.0 * sl
        for kk in range(K):
            v = 0.0 if kk in Sset else ss
            A[n + idx, 0, kk] = v
            B[n + idx, 0, kk] = v
        C[n + idx, 0, 0] = 0.0 if acc_in else v_small
    Aw, Bw, Cw = G.to_operand(dtype, A), G.to_operand(dtype, B), G.f32_bits(C)
    D = G.mma_tiles(dtype, Aw, Bw, Cw)[:, 0, 0]
    neutral = [S for idx, S in enumerate(subsets) if D[idx] == D[n + idx]]
    # the model's prediction of the same experiment
    Dm = np.zeros(2 * n, np.uint32)
    for idx in range(2 * n):
        Dm[idx] = tc_dot(model, int(Cw[idx, 0, 0]), Aw[idx, 0].tolist(), Bw[idx, 0].tolist())
    neutral_model = [S for idx, S in enumerate(subsets) if Dm[idx] == Dm[n + idx]]
    smallest = min(neutral, key=len) if neutral else None
    fmt = lambda S: [("acc" if x == -1 else x) for x in S]
    maximal = [S for S in neutral if not any(set(S) < set(T) for T in neutral)]
    return {
        "subsets_tested": n,
        "neutral_count": len(neutral),
        "neutral_count_model": len(neutral_model),
        "neutral_sets_identical_to_model": neutral == neutral_model,
        "gpu_vs_model_mismatching_tiles": int(np.sum(D != Dm)),
        "smallest_neutral": fmt(smallest) if smallest else None,
        "products_in_smallest": (sum(1 for x in smallest if x != -1) if smallest else None),
        "accumulator_position": ("beginning" if smallest and -1 in smallest else "end"),
        "maximal_neutral": [fmt(S) for S in maximal][:8],
        "neutral_first_20": [fmt(S) for S in neutral[:20]],
    }


def structured_probes(dtype: str, model: TCModel) -> dict:
    """The single-tile tests of GPU_reproduction.py plus accumulator/cross-group probes, batched."""

    P = Probe(dtype, model)
    tests: dict[str, dict] = {}
    idx: dict[str, list[int]] = {}

    def add(name: str, acc, a, b, **meta):
        i = P.add(acc, a, b)
        idx.setdefault(name, []).append(i)
        tests.setdefault(name, {"meta": []})["meta"].append(meta)

    # Full-precision product (Alg. 3): the widest product of two finite operands, alone.
    big = (2.0**11 - 1) if dtype == "f16" else 255.0
    add("full_precision_product", 0.0, {0: big}, {0: big}, expected=big * big)

    # Internal precision (Alg. 5): 1*1 + 1*(-1) + 2^-c for c = 1..60 with the residual in slot s.
    for slot in (2, 7, 8, 15):
        for c in range(1, 61):
            up, fl = math.ceil(-c / 2), math.floor(-c / 2)
            if dtype == "f16" and fl < -24:
                continue
            add(
                f"internal_precision_slot{slot}",
                0.0,
                {0: 1.0, 1: 1.0, slot: 2.0**up},
                {0: 1.0, 1: -1.0, slot: 2.0**fl},
                c=c,
                expected=2.0**-c,
            )

    # Shift rounding (Alg. 6/7) with c = 24 (Ampere).
    c = 24
    ea, eb = -math.ceil((c + 1) / 2), -math.floor((c + 1) / 2)
    add("shift_round_1_pos_half_ulp", 0.0, {0: 1.0, 1: 1.0, 2: 2.0**ea}, {0: 1.0, 1: -1.0, 2: 2.0**eb})
    add("shift_round_2_neg_half_ulp", 0.0, {0: 1.0, 1: 1.0, 2: -(2.0**ea)}, {0: 1.0, 1: -1.0, 2: 2.0**eb})
    eb2, ebb = -math.ceil(c / 2), -math.floor(c / 2)
    add(
        "shift_round_3_above_half",
        0.0,
        {0: 1.0, 1: 1.0, 2: 2.0**eb2 + 2.0 ** (eb2 - 1)},
        {0: 1.0, 1: -1.0, 2: 2.0**ebb},
        expected_if_no_even=2.0**-c,
    )
    add(
        "shift_round_4_above_half_b",
        0.0,
        {0: 1.0, 1: 1.0, 2: 2.0**ea},
        {0: 1.0, 1: -1.0, 2: 2.0**eb + 2.0 ** (eb - 1)},
    )
    add(
        "shift_round_5_tie_vs_trunc",
        0.0,
        {0: 1.0, 1: 1.0, 2: 3.0 * 2.0**ea},
        {0: 1.0, 1: -1.0, 2: 2.0 ** (eb - 1)},
    )

    # Normalisation overflow (Alg. 8): 1.5*1.5 - 1.5*1.5 + 2^-c.
    up, fl = math.ceil(-c / 2), math.floor(-c / 2)
    add("normalization_overflow", 0.0, {0: 1.5, 1: 1.5, 2: 2.0**up}, {0: 1.5, 1: -1.5, 2: 2.0**fl}, expected=2.0**-c)

    # Final summation rounding (Alg. 9): 2.25*2^24 + 3 and 2.25*2^24 - 1.
    add("final_round_ref_plus_3", 0.0, {0: 1.5 * 2.0**12, 1: 3.0}, {0: 1.5 * 2.0**12, 1: 1.0}, ref=2.25 * 2.0**24)
    add("final_round_ref_minus_1", 0.0, {0: 1.5 * 2.0**12, 1: 1.0}, {0: 1.5 * 2.0**12, 1: -1.0}, ref=2.25 * 2.0**24)

    # Subnormal handling (Alg. 10): subnormal * normal, alone and with a residual.
    if dtype == "f16":
        sub_e = -14
        for k_ in range(1, 11):
            add("subnormal_product", 0.0, {0: 2.0 ** (sub_e - k_)}, {0: 2.0**14}, expected=2.0**-k_)
        up, fl = math.ceil((-1 - c) / 2), math.floor((-1 - c) / 2)
        add(
            "subnormal_normalization",
            0.0,
            {0: 2.0 ** (sub_e - 1), 1: 2.0 ** (sub_e - 1), 2: 2.0**up},
            {0: 2.0**-sub_e, 1: -(2.0**-sub_e), 2: 2.0**fl},
            hawkeye_yes_means=2.0 ** (-1 - c),
        )
    else:
        for k_ in range(1, 8):
            add("subnormal_product", 0.0, {0: 2.0 ** (-127 - k_ + 1)}, {0: 2.0**126}, expected=2.0 ** (-k_))
        # products landing below the FP32 normal range: 2^-100 * 2^-30 = 2^-130 (denormalised, not flushed)
        add("subnormal_result", 0.0, {0: 2.0**-100}, {0: 2.0**-30}, expected=2.0**-130)
        add("subnormal_result_sum", 2.0**-140, {0: 2.0**-100, 1: 2.0**-100}, {0: 2.0**-30, 1: 2.0**-31})

    if dtype == "bf16":
        add("extended_range", 0.0, {0: 2.0**127, 1: 2.0**127, 2: 2.0**127}, {0: 2.0, 1: -2.0, 2: 1.0}, expected=2.0**127)
        add("overflow_across_groups", 0.0, {0: 2.0**64, 8: 2.0**64}, {0: 2.0**64, 8: -(2.0**64)}, note="Ada: +inf")
        add("overflow_same_group", 0.0, {0: 2.0**64, 1: 2.0**64}, {0: 2.0**64, 1: -(2.0**64)}, note="Ada: 0")
        for e in range(0, -160, -1):
            add(
                "minimal_max_exponent",
                0.0,
                {0: 2.0 ** math.ceil(e / 2), 1: 2.0 ** math.ceil((e - c) / 2)},
                {0: 2.0 ** math.floor(e / 2), 1: -(2.0 ** math.floor((e - c) / 2))},
                e=e,
                expected=2.0**e,
            )

    # Accumulator-magnitude and group-structure probes (operand magnitudes within the dtype's range).
    big = 2.0**20 if dtype == "bf16" else 2.0**10
    add("acc_dominates_truncation", 2.0**23, {0: 0.75}, {0: 1.0}, rne_would_give=2.0**23 + 1, trunc=2.0**23)
    add("acc_dominates_truncation_neg", -(2.0**23), {0: 0.75}, {0: -1.0}, rne_would_give=-(2.0**23 + 1))
    add("acc_exact_ulp", 2.0**23, {0: 1.0}, {0: 1.0}, expected=2.0**23 + 1)
    add("acc_lsb_kept", f(0x3F800001), {0: 2.0**-30 if dtype == "bf16" else 2.0**-14}, {0: 1.0}, note="acc not truncated when products tiny")
    add("cancel_acc_group1_residual_group2", big, {0: 1.0, 8: 2.0**-10}, {0: -big, 8: 2.0**-10}, expected=2.0**-20)
    add("cancel_acc_group1_residual_group1", big, {0: 1.0, 1: 2.0**-10}, {0: -big, 1: 2.0**-10}, expected_trunc=0.0 if dtype == "bf16" else 2.0**-20, note="bf16: residual 40 bits below the max exponent is truncated; f16: 30 bits, also truncated (25-bit window)")
    add("cancel_group2_residual_group1", 0.0, {0: 2.0**-10, 8: 1.0, 9: 1.0}, {0: 2.0**-10, 8: big, 9: -big}, note="residual first, then big cancel")
    add("acc_sticky_through_group2", big, {0: 1.0, 8: 1.0}, {0: -big, 8: 2.0**-10}, expected=2.0**-10)

    # The paper's introductory vector, FP16-exact; BF16 rounds 65504 to 65536 and 0.001 to 0.0010009765625.
    va = [65504.0, 1.0, -65504.0, 1.0]
    vb = [1.0, 0.001, 1.0, 0.001]
    add("intro_vector_slots0123", 0.0, {i: va[i] for i in range(4)}, {i: vb[i] for i in range(4)}, paper_A100=0.002, paper_L40S=0.0)
    add("intro_vector_slots_0_1_8_9", 0.0, {0: va[0], 1: va[1], 8: va[2], 9: va[3]}, {0: vb[0], 1: vb[1], 8: vb[2], 9: vb[3]})
    add("intro_vector_slots_0_8_1_9", 0.0, {0: va[0], 8: va[1], 1: va[2], 9: va[3]}, {0: vb[0], 8: vb[1], 1: vb[2], 9: vb[3]})
    add("intro_vector_pairs_then_sum", 0.0, {0: va[0], 1: va[2], 8: va[1], 9: va[3]}, {0: vb[0], 1: vb[2], 8: vb[1], 9: vb[3]})

    D, Mo = P.run()
    out = {}
    for name, ids in idx.items():
        recs = []
        for j, i in enumerate(ids):
            acc, a, b = P.records[i]
            r = {
                "acc": hx(acc),
                "a": "".join(f"{x:04x}" for x in a),
                "b": "".join(f"{x:04x}" for x in b),
                "gpu": hx(D[i]),
                "gpu_value": f(int(D[i])),
                "model": hx(Mo[i]),
                "match": bool(D[i] == Mo[i]),
            }
            r.update(tests[name]["meta"][j])
            recs.append(r)
        out[name] = {"records": recs, "all_match": all(r["match"] for r in recs)}

    # Derived characterisation values (as GPU_reproduction.py would print them).
    derived = {}
    for slot in (2, 7, 8, 15):
        prec = None
        for r in out[f"internal_precision_slot{slot}"]["records"]:
            if r["gpu_value"] != r["expected"]:
                prec = r["c"] - 1
                break
        derived[f"internal_precision_bits_slot{slot}"] = prec
    cands = [
        "nearest_tie_towards_0",
        "nearest_tie_away_from_0",
        "nearest_tie_towards_even",
        "towards_0",
        "away_from_0",
        "towards_even",
    ]
    r1 = out["shift_round_1_pos_half_ulp"]["records"][0]["gpu_value"]
    cands = [m for m in cands if "away_from_0" not in m] if r1 == 0.0 else ["nearest_tie_away_from_0", "away_from_0"]
    r2 = out["shift_round_2_neg_half_ulp"]["records"][0]["gpu_value"]
    cands = [m for m in cands if "away_from_0" not in m] if r2 == 0.0 else [m for m in cands if "away_from_0" in m]
    r3 = out["shift_round_3_above_half"]["records"][0]["gpu_value"]
    cands = [m for m in cands if "even" not in m] if r3 == 2.0**-c else [m for m in cands if "even" in m]
    r4 = out["shift_round_4_above_half_b"]["records"][0]["gpu_value"]
    cands = [m for m in cands if "towards_0" in m] if r4 == 0.0 else [m for m in cands if "towards_0" not in m]
    if "nearest_tie_towards_0" in cands and "towards_0" in cands:
        r5 = out["shift_round_5_tie_vs_trunc"]["records"][0]["gpu_value"]
        cands = ["towards_0"] if r5 == 0.0 else ["nearest_tie_towards_0"]
    derived["shift_rounding"] = cands
    derived["normalization_overflow"] = out["normalization_overflow"]["records"][0]["gpu_value"] != 2.0**-c
    ref = 2.25 * 2.0**24
    base = out["final_round_ref_plus_3"]["records"][0]["gpu_value"]
    act = out["final_round_ref_minus_1"]["records"][0]["gpu_value"]
    derived["final_rounding"] = (
        "truncation (towards_0)" if base == ref and base - act > 0 else f"other (baseline={base}, minus1={act})"
    )
    if dtype == "f16":
        derived["subnormal_normalized"] = (
            out["subnormal_normalization"]["records"][0]["gpu_value"] == 2.0 ** (-1 - c)
        )
    if dtype == "bf16":
        derived["extended_range_supported"] = out["extended_range"]["records"][0]["gpu_value"] == 2.0**127
        mme = None
        for r in out["minimal_max_exponent"]["records"]:
            if r["gpu_value"] == r["expected"]:
                mme = r["e"] + 1
                break
        derived["minimal_max_exponent"] = mme
    out["_derived"] = derived
    out["_all_match"] = all(v["all_match"] for k_, v in out.items() if not k_.startswith("_"))
    out["_records"] = len(P.records)
    out["_mismatching_records"] = int(sum(1 for i in range(len(P.records)) if D[i] != Mo[i]))
    return out


def torch_intro_vector() -> dict:
    """The paper's vector through cuBLAS (torch.matmul) in FP16 and BF16 at a few shapes."""

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch unavailable: {e}"}
    res = {}
    for dt, name in ((torch.float16, "f16"), (torch.bfloat16, "bf16")):
        a = torch.tensor([[65504.0, 1.0, -65504.0, 1.0]], dtype=dt, device="cuda")
        b = torch.tensor([[1.0, 0.001, 1.0, 0.001]], dtype=dt, device="cuda")
        r = {}
        r["1x4_dot_4x1"] = float((a @ b.t())[0, 0])
        A = torch.zeros(16, 16, dtype=dt, device="cuda")
        B = torch.zeros(16, 16, dtype=dt, device="cuda")
        A[0, :4] = a[0]
        B[:4, 0] = b[0]
        r["16x16_matmul"] = float((A @ B)[0, 0])
        A = torch.zeros(128, 128, dtype=dt, device="cuda")
        B = torch.zeros(128, 128, dtype=dt, device="cuda")
        A[0, :4] = a[0]
        B[:4, 0] = b[0]
        r["128x128_matmul"] = float((A @ B)[0, 0])
        r["fp32_sequential"] = float(sum(float(x) * float(y) for x, y in zip(a[0].float().tolist(), b[0].float().tolist())))
        res[name] = r
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtypes", default="bf16,f16")
    ap.add_argument("--skip-exhaustive", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(20260907)
    t0 = time.time()
    out = {"device": G.device_info(), "dtypes": {}}
    for dtype in args.dtypes.split(","):
        model = model_for(dtype)
        r = {"model": {"name": model.name, "groups": list(model.groups), "width": model.width, "zero_exponent": model.zero_exponent, "instruction": model.instruction, "version": model.version}}
        r["layout"] = layout_test(dtype, rng)
        print(dtype, "layout", r["layout"], flush=True)
        if not args.skip_exhaustive:
            t1 = time.time()
            r["neutral_subgroups"] = neutral_subgroups_exhaustive(dtype, model)
            r["neutral_subgroups"]["seconds"] = time.time() - t1
            print(dtype, "neutral", {k_: v for k_, v in r["neutral_subgroups"].items() if "first_20" not in k_ and "maximal" not in k_}, flush=True)
        r["structured"] = structured_probes(dtype, model)
        print(dtype, "structured derived", r["structured"]["_derived"], "all_match", r["structured"]["_all_match"], "mismatching", r["structured"]["_mismatching_records"], flush=True)
        for name in ("intro_vector_slots0123", "intro_vector_slots_0_1_8_9", "acc_dominates_truncation", "cancel_acc_group1_residual_group1", "cancel_acc_group1_residual_group2"):
            rec = r["structured"][name]["records"][0]
            print("   ", name, "gpu", rec["gpu"], rec["gpu_value"], "model", rec["model"], "match", rec["match"], flush=True)
        out["dtypes"][dtype] = r
    out["torch_intro_vector"] = torch_intro_vector()
    print("torch intro vector", out["torch_intro_vector"], flush=True)
    out["seconds"] = time.time() - t0
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
