"""anchor-a100 lane: the measured A100 column for every GPU kernel of the campaign, and A vs B re-priced on it.

    uv run --frozen --package verity-numerical python archive/shared/anchor_a100/report.py \
        [--results DIR] [--out backends/numerical/reports/anchor_a100]

Inputs (fetched research runs, under ``--results/<kernel>/``): ``microbench/microbench.json``,
``hash_gpu/bench.json``, ``encode/bench.json``, ``fused/bench.json``, ``fused_logup/fused_logup.json``,
``packed/bench.json``, ``tensor/bench.json``.  Missing inputs leave their section out (the report is regenerated after
every kernel lands).  The recorded 4090 / H100 numbers are the lanes' own reports (``backends/numerical/reports/*.json``,
``backends/*/results_*.json``).  All of them are the campaign's frozen evidence, read from the evidence store
(``verity_numerical.bench.frozen``) at their former repository paths; ``--results`` defaults to the lane's recorded runs there.

Floors on the A100 use the *measured* copy bandwidth (``microbench.json``) beside the datasheet 2,039 GB/s; tensor floors
use the datasheet INT8 peak 312 T MAC/s (the cuBLAS ``_int_mm`` ceiling measured here is reported next to it).

Section "A vs B on the A100": ``explore.tensor`` with the A100-measured buckets substituted (``encoder_seconds_per_row``,
``hash_rate``; A's checker sumcheck + logUp buckets replaced by the measured fused / packed sequences), every other
bucket modelled at 0.172 of the A100's INT8 peak and listed under ``omitted``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from verity_numerical.bench import contract, frozen
from verity_numerical.explore import tensor as tx
from verity_numerical.security import accounting as acc

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "direct"))
from encode import (
    reprice_measured as rm,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPORTS = "backends/numerical/reports"
A100 = "NVIDIA A100 SXM4 80GB"
DATASHEET_BW = 2.039e12
DATASHEET_INT8_MACS = 312.0e12
BW_4090 = 1.008e12
BW_H100 = 3.35e12
INT8_MACS_4090 = 330.3e12
INT8_MACS_H100 = 989.5e12
UTIL = 0.172
UNITS = 4096
LAYER_WIDTHS = acc.CHECKER_LAYER_WIDTHS_A
WOULD_NOT_RUN = [
    "Nothing was modified for sm_80: every kernel (cupy RawKernels, Triton 3.2 `tl.dot` int8 paths, torch `_int_mm`) compiled and ran bit-exact on the A100 as recorded.",
    "The fused lane's `tc` variant of the logUp round is not built on any card (`fused_logup` has SIMT only); the fused SIMT logUp round *was* run (section 3b).",
    "The packed lane's plain k=0 direct 2^28 (140 GB) does not fit the 80 GB card on any GPU; k=2..4 at 2^28 were run (17.8-34.4 GB).",
    "`cupy` and `blake3` were absent from the pod's venv and were installed (`cupy-cuda12x`, `blake3`); the hash_gpu / encode benches otherwise ran unchanged.",
    "The A100's `torch._int_mm` reaches 0.12 of the datasheet INT8 peak at 8192^3 (the 4090 got 0.25): the 'fraction of INT8 peak' floors on this card are against a ceiling cuBLAS itself does not approach.",
    "b-ligero needed `PYTHONPATH=.:benchmarks/numerical/python` and `verity_numerical.bench.instances build` on the pod first (the operand arrays are built, not committed); `bench-vu` has no `--vus` flag (85 VUs come from `--batch 8192`).  Run r20260922-080532-9bbb is the failed attempt.",
    "Model-only rows (`explore.tensor` at 0.172 utilisation) are ESTIMATES; the A vs B section labels every still-modelled bucket.  No proof was produced in this lane except the b-ligero prover runs (section 7, `NON_ZK_PROOF_DIAGNOSTIC`).",
]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if isinstance(x, str):
        return x
    if abs(x) >= 1e5 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.3g}"
    return f"{x:,.{nd}f}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------------------------------


def sec_microbench(mb: dict) -> tuple[str, dict]:
    bw = mb["bandwidth"]
    i8 = mb["int8"]
    bf = mb["bf16"]
    rows = [["HBM copy (read+write), 1 GiB", f"{bw['copy']['gb_per_s']:.0f} GB/s", "2,039 GB/s", f"{bw['copy']['fraction_of_datasheet']:.2f}"],
            ["HBM copy, 4 GiB", f"{bw['copy_4gib']['gb_per_s']:.0f} GB/s", "2,039 GB/s", f"{bw['copy_4gib']['fraction_of_datasheet']:.2f}"],
            ["HBM write only (`fill_`)", f"{bw['write']['gb_per_s']:.0f} GB/s", "2,039 GB/s", f"{bw['write']['fraction_of_datasheet']:.2f}"],
            ["HBM read only (`sum`, int32 -> int64: a torch reduction, not a bandwidth number)", f"{bw['read']['gb_per_s']:.0f} GB/s", "2,039 GB/s", f"{bw['read']['fraction_of_datasheet']:.2f}"]]
    for k, v in i8.items():
        if "error" in v:
            rows.append([f"`_int_mm` {k}", v["error"], "312 T MAC/s", "--"])
        else:
            rows.append([f"`_int_mm` {k}", f"{v['t_macs_per_s']:.1f} T MAC/s ({v['seconds']*1e3:.3f} ms)", "312 T MAC/s", f"{v['fraction_of_int8_peak']:.3f}"])
    for k, v in bf.items():
        rows.append([f"BF16 matmul {k}", f"{v['tflops']:.0f} TFLOPS ({v['seconds']*1e3:.2f} ms)", "312 TFLOPS", f"{v['fraction_of_bf16_peak']:.2f}"])
    txt = md_table(["microbench (CUDA events, median of 9)", "A100 measured", "datasheet", "fraction"], rows)
    mp = mb["measured_peaks"]
    summary = {"hbm_copy_bytes_per_s": mp["hbm_copy_bytes_per_s"], "int8_macs_per_s_cublas_best": mp["int8_macs_per_s_cublas_best"],
               "bf16_flops_per_s_cublas_best": mp["bf16_flops_per_s_cublas_best"], "hardware": mb["hardware"], "run_id": mb.get("run_id")}
    return txt, summary


def sec_hash(a100: dict, ref: dict, bw_meas: float) -> tuple[str, list[dict]]:
    """Full-tree GB/s at the Ligero shapes; the floor is one HBM read of the matrix at the measured copy bandwidth."""
    def pick(doc, cand, hash, mds=None):
        for r in doc["ligero"]:
            if r.get("candidate") == cand and r["hash"] == hash and (mds is None or r.get("mds") == mds):
                return r
        return None
    rows, data = [], []
    for hash, mds in (("sha256", None), ("sha256-chunked", None), ("blake3", None), ("poseidon2-16", None), ("poseidon2-24", None),
                      ("poseidon2-16", "blocks"), ("poseidon2-16", "int8")):
        for cand in ("B", "A"):
            r4, ra = pick(ref, cand, hash, mds), pick(a100, cand, hash, mds)
            if not r4 or not ra:
                continue
            key = "full_shape_tree_ms_scaled" if mds else "tree_ms_median"
            nbytes = r4.get("full_shape_bytes", r4["bytes"])
            ms4, msa = r4[key], ra[key]
            gbs4, gbsa = nbytes / ms4 / 1e6, nbytes / msa / 1e6
            floor_ms = nbytes / bw_meas * 1e3
            ratio = msa and ms4 / msa
            label = f"{hash}" + (f" / torch {mds}" if mds else " / cuda" if hash.startswith("poseidon2") else "")
            rows.append([label, cand, f"{gbs4:.1f} GB/s ({ms4:,.1f} ms)", f"{gbsa:.1f} GB/s ({msa:,.1f} ms)", f"{ratio:.2f}x",
                         f"{floor_ms:.1f} ms (read once at {bw_meas/1e9:.0f} GB/s)", f"{floor_ms/msa:.2f}"])
            data.append({"kernel": "hash_gpu", "hash": hash, "mds": mds, "candidate": cand, "bytes": nbytes, "ref_gpu": "RTX 4090", "ref_ms": ms4, "ref_gb_per_s": gbs4,
                         "a100_ms": msa, "a100_gb_per_s": gbsa, "ratio_ref_over_a100_time": ratio, "a100_floor_ms": floor_ms, "fraction_of_floor": floor_ms / msa,
                         "scaled_from_columns": ra.get("scaled_from_columns")})
    txt = md_table(["hash (leaf function)", "shape", "RTX 4090 recorded", "A100 measured", "speed ratio A100/4090", "A100 floor", "fraction of floor"], rows)
    # generic trees and permutation
    g_rows = []
    for lg in (20, 23, 25, 27):
        for hash in ("sha256", "blake3", "poseidon2-16", "poseidon2-24"):
            r4 = next((r for r in ref["generic"] if r.get("log2_leaves") == lg and r["hash"] == hash and "mds" not in r), None)
            ra = next((r for r in a100["generic"] if r.get("log2_leaves") == lg and r["hash"] == hash and "mds" not in r), None)
            if r4 and ra:
                g_rows.append([f"2^{lg}", hash, f"{r4['leaves_per_s']/1e6:,.0f} M leaves/s", f"{ra['leaves_per_s']/1e6:,.0f} M leaves/s", f"{ra['leaves_per_s']/r4['leaves_per_s']:.2f}x"])
                data.append({"kernel": "hash_gpu.generic", "hash": hash, "log2_leaves": lg, "ref_leaves_per_s": r4["leaves_per_s"], "a100_leaves_per_s": ra["leaves_per_s"],
                             "ratio": ra["leaves_per_s"] / r4["leaves_per_s"]})
    txt += "\n\nGeneric trees (32-byte leaves, tree-dominated):\n\n" + md_table(["leaves", "hash", "RTX 4090", "A100", "ratio"], g_rows)
    p_rows = []
    for ra in a100["permutation"]:
        r4 = next((r for r in ref["permutation"] if r["width"] == ra["width"] and r["mds"] == ra["mds"]), None)
        if r4:
            p_rows.append([f"w{ra['width']} {ra['mds']}", f"{r4['permutations_per_s']/1e6:,.1f} M perm/s", f"{ra['permutations_per_s']/1e6:,.1f} M perm/s", f"{ra['permutations_per_s']/r4['permutations_per_s']:.2f}x"])
            data.append({"kernel": "hash_gpu.permutation", "width": ra["width"], "mds": ra["mds"], "ref_perm_per_s": r4["permutations_per_s"], "a100_perm_per_s": ra["permutations_per_s"]})
    txt += "\n\nPoseidon2 permutation throughput:\n\n" + md_table(["variant", "RTX 4090", "A100", "ratio"], p_rows)
    return txt, data


def sec_encode(a100: dict, ref: dict, bw_meas: float) -> tuple[str, list[dict]]:
    rows, data = [], []
    for shape in ("B", "A"):
        for name in ("ntt_simt_v2_t512", "ntt_simt_v2_t512_nosys", "ntt_simt_v1_t512", "ntt_simt_v2_t128", "ntt_simt_v2_t256", "ntt_simt_v2_t1024",
                     "matdft_int8_triton", "matdft_int8_cublas"):
            r4 = next((r for r in ref["encoders"] if r["shape"] == shape and r["encoder"] == name), None)
            ra = next((r for r in a100["encoders"] if r["shape"] == shape and r["encoder"] == name), None)
            if not r4 or not ra:
                continue
            hbm_floor = (ra["bytes_in"] + ra["bytes_out"]) / bw_meas * 1e3
            tc_floor = ra["limb_macs"] / DATASHEET_INT8_MACS * 1e3 if "limb_macs" in ra else None
            floor = hbm_floor if ra["kind"] == "simt" else max(hbm_floor, tc_floor)
            floor_label = f"{hbm_floor:.1f} ms HBM@{bw_meas/1e9:.0f}" if ra["kind"] == "simt" else f"{tc_floor:.1f} ms TC@312T (HBM {hbm_floor:.1f})"
            rows.append([name, shape, f"{r4['ms_median']:,.1f} ms ({r4['us_per_row']:.3f} us/row)", f"{ra['ms_median']:,.1f} ms ({ra['us_per_row']:.3f} us/row)",
                         f"{r4['ms_median']/ra['ms_median']:.2f}x", floor_label, f"{floor/ra['ms_median']:.2f}"
                         + (f"; {ra['t_macs_per_s']/DATASHEET_INT8_MACS*1e12:.3f} of INT8 peak" if "t_macs_per_s" in ra else "")])
            data.append({"kernel": "encode", "encoder": name, "shape": shape, "ref_gpu": "RTX 4090", "ref_ms": r4["ms_median"], "a100_ms": ra["ms_median"],
                         "a100_us_per_row": ra["us_per_row"], "ratio_ref_over_a100_time": r4["ms_median"] / ra["ms_median"],
                         "a100_hbm_floor_ms_measured_bw": hbm_floor, "a100_hbm_floor_ms_datasheet": (ra["bytes_in"] + ra["bytes_out"]) / DATASHEET_BW * 1e3,
                         "a100_tc_floor_ms": tc_floor, "fraction_of_floor": floor / ra["ms_median"], "a100_hbm_gb_per_s": ra["hbm_gb_per_s"],
                         "a100_fraction_of_int8_peak": (ra["t_macs_per_s"] * 1e12 / DATASHEET_INT8_MACS) if "t_macs_per_s" in ra else None,
                         "scaled_from_rows": ra.get("scaled_from_rows"), "bit_exact": a100["checks"]})
    txt = md_table(["encoder", "shape", "RTX 4090 recorded", "A100 measured", "speed ratio A100/4090", "A100 floor", "fraction of floor"], rows)
    txt += f"\n\nBit-exact checks on the A100 (64 full rows vs the numpy reference): `{a100['checks']}`."
    return txt, data


def _fused_kernel(doc, impl, n, deg_in, mode, flushed=True):
    for k in doc["kernels"]:
        if k["impl"] == impl and k["n"] == n and k["deg_in"] == deg_in and k["mode"] == mode and k.get("flush", flushed) == flushed:
            return k
    return None


def sec_fused(a100: dict, ref: dict, bw_meas: float) -> tuple[str, list[dict]]:
    rows, data = [], []
    for impl in ("simt", "tc"):
        for n in (1 << 23, 1 << 22, 1 << 21, 1 << 20, 1 << 18):
            for deg_in, mode in ((6, "fold+eval"), (1, "fold+eval")):
                k4 = _fused_kernel(ref["measured"], impl, n, deg_in, mode)
                ka = _fused_kernel(a100, impl, n, deg_in, mode)
                if not k4 or not ka:
                    continue
                floor_us = ka["bytes"] / bw_meas * 1e6
                t4, ta = k4["seconds"] * 1e6, ka["seconds"] * 1e6
                rows.append([f"{impl} round, 2^{int(math.log2(n))}, {'ext' if deg_in == 6 else 'base'} input, {mode}", f"{t4:,.0f} us ({k4['gb_per_s']:.0f} GB/s)",
                             f"{ta:,.0f} us ({ka['gb_per_s']:.0f} GB/s)", f"{t4/ta:.2f}x", f"{floor_us:,.0f} us", f"{floor_us/ta:.2f}"])
                data.append({"kernel": "fused", "impl": impl, "n": n, "deg_in": deg_in, "mode": mode, "ref_gpu": "RTX 4090", "ref_us": t4, "a100_us": ta,
                             "ratio_ref_over_a100_time": t4 / ta, "a100_floor_us_measured_bw": floor_us, "a100_floor_us_datasheet": ka["bytes"] / DATASHEET_BW * 1e6,
                             "fraction_of_floor": floor_us / ta, "a100_gb_per_s": ka["gb_per_s"]})
    for impl in ("simt", "tc"):
        L4, La = ref["measured"]["layers"].get(impl), a100["layers"].get(impl)
        if L4 and La:
            floor_ms = La["bytes"] / bw_meas * 1e3
            rows.append([f"{impl} 6-layer / 130-round layer model, graphed", f"{L4['graphed_seconds']*1e3:.2f} ms", f"{La['graphed_seconds']*1e3:.2f} ms",
                         f"{L4['graphed_seconds']/La['graphed_seconds']:.2f}x", f"{floor_ms:.2f} ms", f"{floor_ms/(La['graphed_seconds']*1e3):.2f}"])
            rows.append([f"{impl} layer model, eager", f"{L4['eager_seconds']*1e3:.2f} ms", f"{La['eager_seconds']*1e3:.2f} ms",
                         f"{L4['eager_seconds']/La['eager_seconds']:.2f}x", "", ""])
            data.append({"kernel": "fused.layers", "impl": impl, "ref_gpu": "RTX 4090", "ref_graphed_s": L4["graphed_seconds"], "a100_graphed_s": La["graphed_seconds"],
                         "ref_eager_s": L4["eager_seconds"], "a100_eager_s": La["eager_seconds"], "ratio_ref_over_a100_time": L4["graphed_seconds"] / La["graphed_seconds"],
                         "a100_floor_s_measured_bw": La["bytes"] / bw_meas, "fraction_of_floor": La["bytes"] / bw_meas / La["graphed_seconds"], "bytes": La["bytes"]})
    if "launch" in a100:
        for impl in ("simt", "tc"):
            if impl in a100["launch"] and impl in ref["measured"]["launch"]:
                data.append({"kernel": "fused.launch", "impl": impl, "ref": ref["measured"]["launch"][impl], "a100": a100["launch"][impl]})
    txt = md_table(["fused checker round", "RTX 4090 recorded", "A100 measured", "speed ratio A100/4090", "A100 HBM floor (measured BW)", "fraction of floor"], rows)
    if a100.get("agreement_test_tail"):
        txt += f"\n\nBit-exact tests on the A100 (`TC_DEVICE=cuda pytest tensor/fused/tests`): {a100['agreement_test_tail']}"
    return txt, data


def sec_logup(a100: dict, ref: dict, bw_meas: float) -> tuple[str, list[dict]]:
    rows, data = [], []
    for ra in a100["rows"]:
        r4 = next((r for r in ref["rows"] if r["n"] == ra["n"] and r["block"] == ra["block"] and r["num_warps"] == ra["num_warps"]), None)
        if not r4:
            continue
        floor_us = ra["bytes"] / bw_meas * 1e6
        rows.append([f"logUp round 2^{int(math.log2(ra['n']))}, block {ra['block']} x {ra['num_warps']} warps", f"{r4['fold_eval_us']:,.0f} us ({r4['gbps']:.0f} GB/s)",
                     f"{ra['fold_eval_us']:,.0f} us ({ra['gbps']:.0f} GB/s)", f"{r4['fold_eval_us']/ra['fold_eval_us']:.2f}x", f"{floor_us:,.0f} us", f"{floor_us/ra['fold_eval_us']:.2f}"])
        data.append({"kernel": "fused_logup", "n": ra["n"], "block": ra["block"], "num_warps": ra["num_warps"], "ref_gpu": "RTX 4090", "ref_us": r4["fold_eval_us"],
                     "a100_us": ra["fold_eval_us"], "ratio_ref_over_a100_time": r4["fold_eval_us"] / ra["fold_eval_us"], "a100_floor_us_measured_bw": floor_us,
                     "fraction_of_floor": floor_us / ra["fold_eval_us"], "a100_gb_per_s": ra["gbps"], "a100_gmul_per_s": ra["gmul_per_s"]})
    txt = md_table(["fused logUp round (fold+eval)", "RTX 4090 recorded", "A100 measured", "speed ratio A100/4090", "A100 HBM floor (measured BW)", "fraction of floor"], rows)
    if a100.get("agreement"):
        txt += f"\n\nBit-exact vs the reference on the A100: {a100['agreement']}"
    return txt, data


def sec_packed(a100: dict, ref: dict) -> tuple[str, list[dict]]:
    rows, data = [], []
    for lg in (26, 24, 22, 20):
        for k in range(5):
            rh = next((r for r in ref["results_summary"] if r["log2_t"] == lg and r["k"] == k), None)
            ra = next((r for r in a100["results"] if r["log2_t"] == lg and r["k"] == k), None)
            if not rh or not ra:
                continue
            util_a = ra["macs_performed"] / DATASHEET_INT8_MACS / ra["seconds"]
            rows.append([f"2^{lg}, k={k}", f"{rh['seconds']*1e3:.2f} ms ({rh['utilisation.round_graphed']:.2%} of H100 INT8 peak)",
                         f"{ra['seconds']*1e3:.2f} ms ({util_a:.2%} of A100 INT8 peak)", f"{rh['seconds']/ra['seconds']:.2f}x",
                         f"{ra['macs_performed']/DATASHEET_INT8_MACS*1e3:.2f} ms TC@312T", f"{util_a:.4f}", f"{ra['mem.peak_device_bytes']/1e9:.1f} GB"])
            data.append({"kernel": "packed", "log2_t": lg, "k": k, "ref_gpu": "H100", "ref_s": rh["seconds"], "a100_s": ra["seconds"], "ratio_ref_over_a100_time": rh["seconds"] / ra["seconds"],
                         "a100_macs": ra["macs_performed"], "a100_fraction_of_int8_peak": util_a, "a100_peak_mem_bytes": ra["mem.peak_device_bytes"]})
    txt = md_table(["packed sumcheck (whole sumcheck, all rounds, CUDA-graphed)", "H100 recorded", "A100 measured", "speed ratio A100/H100", "A100 TC floor", "fraction of INT8 peak", "A100 peak mem"], rows)
    lrows = []
    for k in range(5):
        Lh, La = ref["layers"].get(str(k)), a100["layers"].get(str(k)) or a100["layers"].get(k)
        if Lh and La:
            lrows.append([f"k={k}", f"{Lh['batch_seconds.padded']*1e3:.0f} / {Lh['batch_seconds.unpadded']*1e3:.0f} ms", f"{Lh['per_vu_seconds.padded']*1e6:.1f} us",
                          f"{La['batch_seconds.padded']*1e3:.0f} / {La['batch_seconds.unpadded']*1e3:.0f} ms", f"{La['per_vu_seconds.padded']*1e6:.1f} us",
                          f"{Lh['batch_seconds.padded']/La['batch_seconds.padded']:.2f}x", f"{La['fit.seconds_per_element']*1e9:.3f} ns/elem + {La['fit.intercept_seconds']*1e3:.2f} ms"])
            data.append({"kernel": "packed.layers", "k": k, "ref_gpu": "H100", "ref_batch_s_padded": Lh["batch_seconds.padded"], "ref_per_vu_s": Lh["per_vu_seconds.padded"],
                         "a100_batch_s_padded": La["batch_seconds.padded"], "a100_batch_s_unpadded": La["batch_seconds.unpadded"], "a100_per_vu_s": La["per_vu_seconds.padded"],
                         "ratio_ref_over_a100_time": Lh["batch_seconds.padded"] / La["batch_seconds.padded"], "a100_fit_s_per_elem": La["fit.seconds_per_element"]})
    txt += "\n\nSix-layer batch model (linear fit over the measured sizes; padded / un-padded widths):\n\n" + md_table(
        ["k", "H100 per batch", "H100 per VU", "A100 per batch", "A100 per VU", "speed ratio A100/H100", "A100 fit"], lrows)
    if a100.get("agreement_test_tail"):
        txt += f"\n\nAgreement test on the A100: {a100['agreement_test_tail']}"
    return txt, data


def sec_packed2(a26: dict, h26: dict, a28: dict, h28: dict, alog: dict, hlog: dict) -> tuple[str, list[dict]]:
    """a-packed2 second pass: 32-bit Montgomery packed sumcheck (2^26 table, direct 2^28) and the logUp fractional GKR at the full-VU shape."""
    rows, data = [], []
    for lg, A, H in ((26, a26, h26), (28, a28, h28)):
        for k in range(5):
            ra = next((r for r in A["results"] if r["log2_t"] == lg and r["k"] == k), None)
            rh = next((r for r in H["results"] if r["log2_t"] == lg and r["k"] == k), None)
            if not ra or not rh:
                continue
            util_a = ra["macs_performed"] / DATASHEET_INT8_MACS / ra["seconds"]
            rows.append([f"Montgomery packed sumcheck 2^{lg}, k={k}", f"{rh['seconds']*1e3:.2f} ms", f"{ra['seconds']*1e3:.2f} ms ({util_a:.2%} of A100 INT8 peak)", f"{rh['seconds']/ra['seconds']:.2f}x",
                         f"{ra['macs_performed']/DATASHEET_INT8_MACS*1e3:.2f} ms TC@312T", f"{util_a:.4f}", f"{ra['mem.peak_device_bytes']/1e9:.1f} GB"])
            data.append({"kernel": "packed2", "log2_t": lg, "k": k, "ref_gpu": "H100", "ref_s": rh["seconds"], "a100_s": ra["seconds"], "ratio_ref_over_a100_time": rh["seconds"] / ra["seconds"],
                         "a100_fraction_of_int8_peak": util_a, "a100_peak_mem_bytes": ra["mem.peak_device_bytes"]})
    for lg, A, H in ((26, a26, h26), (28, a28, h28)):
        for k in sorted(A["layers"], key=int):
            La, Lh = A["layers"][k], H["layers"].get(k)
            if not Lh:
                continue
            rows.append([f"six-layer batch from the 2^{lg} fit, k={k} (padded / un-padded)", f"{Lh['batch_seconds.padded']*1e3:.0f} / {Lh['batch_seconds.unpadded']*1e3:.0f} ms = {Lh['per_vu_seconds.padded']*1e6:.1f} us/VU",
                         f"{La['batch_seconds.padded']*1e3:.0f} / {La['batch_seconds.unpadded']*1e3:.0f} ms = **{La['per_vu_seconds.padded']*1e6:.1f} us/VU**", f"{Lh['batch_seconds.padded']/La['batch_seconds.padded']:.2f}x", "", "", ""])
            data.append({"kernel": "packed2.layers", "log2_t_fit": lg, "k": int(k), "ref_gpu": "H100", "ref_batch_s_padded": Lh["batch_seconds.padded"], "a100_batch_s_padded": La["batch_seconds.padded"],
                         "a100_batch_s_unpadded": La["batch_seconds.unpadded"], "a100_per_vu_s": La["per_vu_seconds.padded"], "ratio_ref_over_a100_time": Lh["batch_seconds.padded"] / La["batch_seconds.padded"]})
    for ra in alog["results"]:
        rh = next((r for r in hlog["results"] if r["k"] == ra["k"]), None)
        if not rh:
            continue
        rows.append([f"logUp fractional GKR, full-VU shape ({alog['units']:,} x {alog['lookups']} lookups, 2^{alog['n']} leaves), k={ra['k']}", f"{rh['total_ms']:.0f} ms = {rh['us_per_vu']:.0f} us/VU",
                     f"**{ra['total_ms']:.0f} ms = {ra['us_per_vu']:.0f} us/VU** (20 smallest levels {ra['tail_levels_ms']:.0f} ms)", f"{rh['total_ms']/ra['total_ms']:.2f}x", "", "", f"{ra['mem.peak_device_bytes']/1e9:.1f} GB"])
        data.append({"kernel": "logup_gkr", "k": ra["k"], "ref_gpu": "H100", "ref_s": rh["total_ms"] * 1e-3, "a100_s": ra["total_ms"] * 1e-3, "ratio_ref_over_a100_time": rh["total_ms"] / ra["total_ms"],
                     "a100_us_per_vu": ra["us_per_vu"], "a100_tail_levels_ms": ra["tail_levels_ms"], "ref_tail_levels_ms": rh["tail_levels_ms"], "a100_peak_mem_bytes": ra["mem.peak_device_bytes"],
                     "split_ms": {k: ra[k] for k in ("histogram_ms", "tree_ms", "big_packed_ms", "big_fold_k_ms", "big_ext_ms", "tail_levels_ms")}})
    txt = md_table(["a-packed2 kernels (whole sumcheck, CUDA-graphed rounds)", "H100 recorded", "A100 measured", "speed ratio A100/H100", "A100 TC floor", "fraction of INT8 peak", "A100 peak mem"], rows)
    if a26.get("agreement_test_tail"):
        txt += f"\n\nAgreement test on the A100 (Montgomery kernels): {a26['agreement_test_tail']}"
    return txt, data


def sec_bwave(commit_a: dict, commit_4: dict, buckets_a: dict, buckets_4: dict, lint_a: dict, lint_4: dict, bw_meas: float) -> tuple[str, list[dict]]:
    """b-commit (fused commitment, prover buckets) and b-lintest (structured linear test) on the A100 vs the recorded 4090 rows."""
    rows, data = [], []
    for shape in ("B", "A"):
        for hash in ("blake3", "sha256"):
            for r4 in sorted((r for r in commit_4["results"] if r["shape"] == shape and r["hash"] == hash and r["mode"] == "fused"), key=lambda r: (r["slice_rows"], r["overlap"], r["keep"])):
                ra = next((r for r in commit_a["results"] if r["name"] == r4["name"] and r["shape"] == shape), None)
                if not ra:
                    continue
                floor = (ra["bytes_in"] + ra["codeword_bytes"]) / bw_meas * 1e3
                rows.append([f"fused commitment (encode -> {hash} -> root), {shape}, {r4['slice_rows']}-row slices{', overlap' if r4['overlap'] else ''}{', cosets kept' if r4['keep'] else ''}",
                             f"{r4['ms_median']:.1f} ms ({r4['us_per_row']:.3f} us/row)", f"{ra['ms_median']:.1f} ms ({ra['us_per_row']:.3f} us/row)", f"{r4['ms_median']/ra['ms_median']:.2f}x",
                             f"{floor:.1f} ms (read W + write codeword at {bw_meas/1e9:.0f} GB/s)", f"{floor/ra['ms_median']:.2f}"])
                data.append({"kernel": "commit", "name": r4["name"], "shape": shape, "hash": hash, "ref_gpu": "RTX 4090", "ref_ms": r4["ms_median"], "a100_ms": ra["ms_median"],
                             "ratio_ref_over_a100_time": r4["ms_median"] / ra["ms_median"], "a100_floor_ms": floor, "fraction_of_floor": floor / ra["ms_median"]})
        for r4 in (r for r in commit_4["results"] if r["shape"] == shape and r["mode"] != "fused"):
            ra = next((r for r in commit_a["results"] if r["name"] == r4["name"] and r["shape"] == shape), None)
            if ra:
                rows.append([f"{r4['name']} ({shape}, unfused reference pass)", f"{r4['ms_median']:.1f} ms", f"{ra['ms_median']:.1f} ms", f"{r4['ms_median']/ra['ms_median']:.2f}x", "", ""])
                data.append({"kernel": "commit", "name": r4["name"], "shape": shape, "ref_gpu": "RTX 4090", "ref_ms": r4["ms_median"], "a100_ms": ra["ms_median"], "ratio_ref_over_a100_time": r4["ms_median"] / ra["ms_median"]})
    for r4 in buckets_4["results"]:
        ra = next((r for r in buckets_a["results"] if r["name"] == r4["name"] and r["shape"] == r4["shape"]), None)
        if not ra:
            continue
        fl = ra.get("read_floor_ms")
        floor_meas = ra["bytes_read"] / bw_meas * 1e3 if ra.get("bytes_read") else None
        rows.append([f"{r4['bucket']}: `{r4['name']}` ({r4['shape']})", f"{r4['ms_median']:.2f} ms", f"{ra['ms_median']:.2f} ms", f"{r4['ms_median']/ra['ms_median']:.2f}x",
                     f"{floor_meas:.2f} ms read floor" if floor_meas else "", f"{floor_meas/ra['ms_median']:.2f}" if floor_meas else ""])
        data.append({"kernel": "buckets", "bucket": r4["bucket"], "name": r4["name"], "shape": r4["shape"], "ref_gpu": "RTX 4090", "ref_ms": r4["ms_median"], "a100_ms": ra["ms_median"],
                     "ratio_ref_over_a100_time": r4["ms_median"] / ra["ms_median"], "a100_floor_ms": floor_meas, "fraction_of_floor": floor_meas / ra["ms_median"] if floor_meas else None, "datasheet_floor_ms_4090": fl})
    for r4 in lint_4["results"]:
        ra = next((r for r in lint_a["results"] if r["name"] == r4["name"]), None)
        if not ra:
            continue
        floor_meas = ra["read_floor_ms"] * 1.008e12 / bw_meas if ra.get("read_floor_ms") else None
        rows.append([f"linear test, structured: `{r4['name']}`", f"{r4['ms_median']:.2f} ms", f"{ra['ms_median']:.2f} ms", f"{r4['ms_median']/ra['ms_median']:.2f}x",
                     f"{floor_meas:.2f} ms read floor" if floor_meas else "", f"{floor_meas/ra['ms_median']:.2f}" if floor_meas else ""])
        data.append({"kernel": "lintest", "name": r4["name"], "M": r4["M"], "lazy": r4["lazy_reduction"], "ref_gpu": "RTX 4090", "ref_ms": r4["ms_median"], "a100_ms": ra["ms_median"],
                     "ratio_ref_over_a100_time": r4["ms_median"] / ra["ms_median"], "a100_floor_ms": floor_meas, "fraction_of_floor": floor_meas / ra["ms_median"] if floor_meas else None})
    txt = md_table(["B prover bucket", "RTX 4090 recorded", "A100 measured", "speed ratio A100/4090", "A100 floor (measured BW)", "fraction of floor"], rows)
    txt += (f"\n\nBit-exact on the A100: commitment roots `{sum(commit_a['checks'].values())}/{len(commit_a['checks'])}` variants equal the reference; "
            f"lintest checks `{lint_a['checks']}`.")
    return txt, data


def sec_ligero(a: dict, ref: dict, bw_meas: float) -> tuple[str, list[dict]]:
    """b-ligero: the torch direct-Ligero prover (an actual NON_ZK_PROOF_DIAGNOSTIC proof, verifier-accepted) on the A100 vs the L40S record."""
    rows, data = [], []
    for name, label in (("ligero_unit_b4096", "unit prover, 4,096 units (tu-k16), l=4096"), ("ligero_vu_42", "VU chain prover, 42 VUs, l=4096"), ("ligero_vu_85", "VU chain prover, 85 VUs, l=8192")):
        A, L = a.get(name), ref.get(name)
        if not A or not L:
            continue
        ma = {m["name"]: m["value"] for m in A["measurements"]}
        ml = {m["name"]: m["value"] for m in L["measurements"]}
        per = "unit.seconds_per_unit" if "unit.seconds_per_unit" in ma else "vu.seconds_per_vu"
        rows.append([f"**{label}**: t.total", f"{ml['t.total']*1e3:.1f} ms ({ml[per]*1e6:.1f} us per {'unit' if 'unit' in per else 'VU'})", f"**{ma['t.total']*1e3:.1f} ms** ({ma[per]*1e6:.1f} us)", f"{ml['t.total']/ma['t.total']:.2f}x",
                     f"{ma['census.bytes_encoded']*2/bw_meas*1e3:.2f} ms (encode read+write at {bw_meas/1e9:.0f} GB/s)", f"{ma['census.bytes_encoded']*2/bw_meas/ma['t.total']:.3f}"])
        for k, lab in (("split.hints_seconds", "hints (torch, graph-replayed)"), ("split.witness_torch_seconds", "witness rows (torch program)"), ("split.encode_seconds", "encode (NTT) + commit"),
                       ("split.merkle_seconds", "Merkle"), ("split.tests_seconds", "linear/quadratic tests"), ("split.openings_seconds", "openings + serialisation"), ("verifier.seconds", "verifier (host)")):
            if k in ma and k in ml:
                rows.append([f"  {lab}", f"{ml[k]*1e3:.2f} ms", f"{ma[k]*1e3:.2f} ms", f"{ml[k]/ma[k]:.2f}x", "", ""])
        data.append({"kernel": "ligero_prover", "bench": name, "ref_gpu": "L40S", "ref_total_s": ml["t.total"], "a100_total_s": ma["t.total"], "ratio_ref_over_a100_time": ml["t.total"] / ma["t.total"],
                     "a100_seconds_per": ma[per], "per": per, "validation": A["validation"], "a100_split": {k: ma[k] for k in ma if k.startswith("split.")}, "proof_class": A["workload_fingerprint"]["proof_class"],
                     "overhead_vs_native_peak": (ma[per] / contract.native_seconds_per_vu(tx.K_TARGET)) if "vu" in per else None, "transcript_bytes": ma.get("transcript.bytes"), "peak_mem_bytes": ma.get("mem.peak_device_bytes")})
    txt = md_table(["b-ligero torch prover (`NON_ZK_PROOF_DIAGNOSTIC`, verifier accepted every rep)", "L40S recorded", "A100 measured", "speed ratio A100/L40S", "A100 HBM floor (encode traffic)", "fraction"], rows)
    return txt, data


def sec_tensor(a100: dict, ref: dict) -> tuple[str, list[dict]]:
    rows, data = [], []
    for r4 in ref["measured"]["rounds"]:
        ra = next((r for r in a100["rounds"] if r["t"] == r4["t"] and r["deg"] == r4["deg"]), None)
        if not ra or r4["t"] < 4096:
            continue
        t4, ta = r4["seconds"]["total_graphed"] * 1e6, ra["seconds"]["total_graphed"] * 1e6
        frac = ra["macs_performed"] / DATASHEET_INT8_MACS / ra["seconds"]["total_graphed"]
        rows.append([f"t=2^{int(math.log2(r4['t']))} deg {r4['deg']} round (graphed)", f"{t4:,.0f} us", f"{ta:,.0f} us", f"{t4/ta:.2f}x",
                     f"{ra['macs_performed']/DATASHEET_INT8_MACS*1e6:.1f} us TC@312T", f"{frac:.4f}"])
        data.append({"kernel": "tensor", "t": r4["t"], "deg": r4["deg"], "ref_gpu": "RTX 4090", "ref_us": t4, "a100_us": ta, "ratio_ref_over_a100_time": t4 / ta,
                     "a100_fraction_of_int8_peak_round": frac, "a100_matmul_us": ra["seconds"]["matmul"] * 1e6, "ref_matmul_us": r4["seconds"]["matmul"] * 1e6})
    L4, La = ref["measured"]["layer_model_seconds"], a100["layers"]["seconds"]
    K4, Ka = ref["measured"].get("layer_model_kernel_seconds") or {}, a100["layers"]["kernel_seconds"]
    for key, label in (("total_graphed", "layer model graphed wall (130 rounds)"), ("total", "layer model, CUDA-event segments (launch gaps included)"),
                       ("matmul", "  of which matmul kernels (events)"), ("tensor_bucket", "  tensor bucket (events)")):
        if key in L4 and key in La:
            rows.append([label, f"{L4[key]*1e3:.1f} ms", f"{La[key]*1e3:.1f} ms", f"{L4[key]/La[key]:.2f}x", "", ""])
            data.append({"kernel": "tensor.layers", "key": key, "ref_s": L4[key], "a100_s": La[key], "ratio_ref_over_a100_time": L4[key] / La[key]})
    if "gpu_busy" in Ka:
        rows.append(["layer model GPU kernel time (profiler)", "87.4 ms (report)", f"{Ka['gpu_busy']*1e3:.1f} ms", f"{87.4e-3/Ka['gpu_busy']:.2f}x", "", ""])
        data.append({"kernel": "tensor.layers", "key": "kernel.gpu_busy", "ref_s": 87.4e-3, "a100_s": Ka["gpu_busy"], "ratio_ref_over_a100_time": 87.4e-3 / Ka["gpu_busy"]})
    for ca in a100.get("int_mm_ceilings", []):
        c4 = next((c for c in ref["measured"]["int_mm_ceilings"] if (c["M"], c["K"], c["N"]) == (ca["M"], ca["K"], ca["N"])), None)
        if c4:
            rows.append([f"`_int_mm` ({ca['M']}, {ca['K']}, {ca['N']})", f"{c4['seconds']*1e6:,.0f} us ({c4['peak_fraction']:.4f} of 4090 peak)",
                         f"{ca['seconds']*1e6:,.0f} us ({ca['macs']/DATASHEET_INT8_MACS/ca['seconds']:.4f} of A100 peak)", f"{c4['seconds']/ca['seconds']:.2f}x", "", ""])
            data.append({"kernel": "tensor.int_mm", "shape": [ca["M"], ca["K"], ca["N"]], "ref_s": c4["seconds"], "a100_s": ca["seconds"], "ratio_ref_over_a100_time": c4["seconds"] / ca["seconds"],
                             "a100_fraction_of_int8_peak": ca["macs"] / DATASHEET_INT8_MACS / ca["seconds"]})
    txt = md_table(["INT8-limb round (tensor lane)", "RTX 4090 recorded", "A100 measured", "speed ratio A100/4090", "A100 TC floor", "fraction of INT8 peak"], rows)
    if a100.get("agreement_test_tail"):
        txt += f"\n\nAgreement test on the A100: {a100['agreement_test_tail']}"
    return txt, data


# ---------------------------------------------------------------------------------------------------------------------
# A vs B on the A100
# ---------------------------------------------------------------------------------------------------------------------


def pow2_pad(w: int) -> int:
    return 1 << max(1, math.ceil(math.log2(w)))


def reprice(meas: dict) -> tuple[str, dict]:
    """``explore.tensor`` designs A and B on the A100 with every A100-measured bucket substituted."""
    fs = tx.FIELDS["babybear"]
    gpu = tx.GPUS["A100"]
    out = {"hardware": A100, "B": tx.B_TARGET, "K": tx.K_TARGET, "security": "2^-128", "proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded",
           "utilisation_modelled": UTIL, "inputs": meas, "designs": {}}
    lines = []
    for dname in ("B", "A"):
        p = tx.price_design(tx.DESIGNS[dname], fs, -128)
        hr = meas.get(f"hash_rate_{dname}") or tx.HASH_RATES["blake3_simt"]["bytes_per_s"]
        spr = meas.get(f"encoder_seconds_per_row_{dname}")
        model = p.times(gpu, "A100", UTIL, tx.HASH_RATES["blake3_simt"]["bytes_per_s"], "rs_matrix_ntt", fs)
        tm = p.times(gpu, "A100", UTIL, hr, "rs_matrix_ntt", fs, encoder_seconds_per_row=spr)
        simt_rate = fs.simt_mult_vs_goldilocks / gpu.simt_field_mults_per_s
        buckets = {"encoding": (tm["t.encoding"], f"measured: {meas.get(f'encoder_{dname}_kernel')} {spr*1e6:.3f} us/row x {p.ligero_rows:,} rows" if spr else "modelled"),
                   "hash": (tm["t.hash"], f"measured: {meas.get(f'hash_{dname}_kernel')} {hr/1e9:.0f} GB/s over {p.hash_bytes/1e9:.2f} GB" if f"hash_rate_{dname}" in meas else "modelled")}
        omitted = ["witness/hint generation (modelled: 2e4 int ops per unit at 25% of the A100 FP32 peak)",
                   "Ligero row combination (modelled matmul @0.172 of the A100 INT8 peak)", "openings and transcript serialisation (not priced)",
                   "prover randomness (modelled PRG at 100 GB/s)"]
        total = tm["t.total"]
        if "bwave" in meas:
            # every B-lane bucket measured on the A100, picked by the b-commit lane's own selector (same rows as reports/b_measured_4090)
            bw = meas["bwave"]
            variants = {}
            for hash in ("blake3", "sha256"):
                for openings in ("store", "reencode"):
                    m, detail = rm.measured_buckets(bw["commit"], bw["buckets"], dname, hash, openings, bw["lintest"], "linear_test_structured_M3_X256_lazy")
                    variants[(hash, openings)] = (m, detail)
            (hash, openings), (m, detail) = min(variants.items(), key=lambda kv: sum(kv[1][0].values()))
            commit_label = f"measured: fused commit_stream encode -> {hash} -> root, {detail['commitment']['name']} ({'cosets kept' if openings == 'store' else 'no store'})"
            sep = meas.get(f"encoder_seconds_per_row_{dname}", 0) * p.ligero_rows + (meas.get(f"hash_{dname}_ms", 0) if dname == "B" else p.hash_bytes / hr * 1e3) * 1e-3
            if sep and sep < m["commitment"] and openings == "store":
                # on the anchor the unfused path (SIMT NTT into HBM, then the Blake3 tree over the stored codeword) beats the fused L2-slice pipeline
                m["commitment"] = sep
                commit_label = (f"measured: UNFUSED encode ({meas.get(f'encoder_{dname}_kernel')}, {meas.get(f'encoder_seconds_per_row_{dname}')*p.ligero_rows*1e3:.1f} ms) + Blake3 tree over the stored "
                                f"codeword ({(meas.get(f'hash_{dname}_ms', 0) if dname == 'B' else p.hash_bytes / hr * 1e3):.1f} ms) -- beats the fused commit_stream "
                                f"({detail['commitment']['name']} {detail['commitment']['ms_median']:.1f} ms) on the A100")
            if dname == "A":
                # A's checker sumcheck and logUp are whole measured buckets (section 4b); the model's matmul + Hadamard components go
                if "packed2_best_batch_s_padded" in meas:
                    m["matmul:gkr_checker"] = meas["packed2_best_batch_s_padded"]
                    m["simt:gkr_hadamards"] = 0.0
                if "logup_gkr_full_vu_s" in meas:
                    m["matmul:logup_fractional_gkr"] = meas["logup_gkr_full_vu_s"]
                    m["simt:logup_hadamards"] = 0.0
                    m["simt:logup_tree_build"] = 0.0          # the measured logUp GKR builds its own tree (tree_ms inside the 412 ms)
            tm2 = p.times(gpu, "A100", UTIL, hr, "rs_matrix_ntt", fs, measured=m)
            buckets = {}
            for k, v in m.items():
                label = {"commitment": commit_label,
                         "matmul:ligero_row_combination": "measured: IRS row combination" + (" + structured linear test M=3 lazy (b-lintest)" if dname == "B" else ""),
                         "simt:quadratic_hadamards": "measured: quadratic test", "simt:helper_inversions": "measured: F_c batch inversion",
                         "extra:openings": f"measured: t=192 column openings ({'gather from store' if openings == 'store' else 're-encode'}) + host Merkle paths",
                         "matmul:gkr_checker": f"measured: {meas.get('packed2_best_label')} six-layer batch, padded (un-padded {meas.get('packed2_best_batch_s_unpadded', 0)*1e3:.0f} ms)",
                         "matmul:logup_fractional_gkr": "measured: a-packed2 logUp fractional GKR, full-VU shape, k=3 (launch-bound; includes its tree build)"}.get(k, "measured")
                if v > 0:
                    buckets[k.split(":")[-1]] = (v, label)
            for k, v in tm2["t.modelled"].items():
                buckets[k.split(":")[-1]] = (v, "modelled" + (" matmul @0.172" if k.startswith("matmul") else " SIMT" if k.startswith("simt") else ""))
            omitted = ["not an end-to-end proof: buckets measured separately on resident pseudo-random data"] + [
                f"{k} (modelled)" for k in tm2["t.modelled"]] + ["transcript serialisation, verifier work, network (not priced)"]
            total = tm2["t.total"]
            per_vu = total / tx.B_TARGET
            row = {"t.total": total, "seconds_per_vu": per_vu, "overhead.vs_native_peak": contract.overhead(per_vu, tx.K_TARGET),
                   "buckets": {k: {"seconds": v[0], "basis": v[1], "share": v[0] / total} for k, v in buckets.items()},
                   "measured_buckets": [k for k, v in buckets.items() if v[1].startswith("measured")], "measured_fraction": tm2["measured_fraction"],
                   "leaf_hash": hash, "openings": openings, "omitted": omitted, "model_only_overhead": model["overhead.vs_native_peak"], "model_only_t.total": model["t.total"],
                   "variants": {f"{h}/{o}": {"t.total": p.times(gpu, "A100", UTIL, hr, "rs_matrix_ntt", fs, measured=mm)["t.total"]} for (h, o), (mm, _) in variants.items()},
                   "ligero_rows": p.ligero_rows, "hash_bytes": p.hash_bytes}
            out["designs"][dname] = row
            lines.append(f"| **{dname}** | {model['overhead.vs_native_peak']:.3g} ({model['t.total']*1e3:.0f} ms) | **{row['overhead.vs_native_peak']:.3g}** ({total*1e3:.0f} ms; {per_vu*1e6:.0f} us/VU; {tm2['measured_fraction']:.1%} measured) | "
                         + "; ".join(f"{k} **{v[0]*1e3:.1f}**" if v[1].startswith("measured") else f"{k} {v[0]*1e3:.1f}" for k, v in buckets.items()) + " |")
            continue
        if dname == "B":
            buckets["row_combination"] = (tm["t.matmul"], "modelled matmul @0.172")
            buckets["linear/quadratic tests + helpers (SIMT)"] = (tm["t.simt_field"], "modelled SIMT")
            omitted += ["linear/quadratic tests, constraint combination, helper inversions (modelled SIMT at the A100 rate)"]
        else:
            t_gkr_mm = p.matmul["gkr_checker"] / (gpu.peak_macs_per_s * UTIL)
            t_gkr_simt = p.simt_field_mults["gkr_hadamards"] * simt_rate
            t_logup_mm = p.matmul["logup_fractional_gkr"] / (gpu.peak_macs_per_s * UTIL)
            t_logup_simt = p.simt_field_mults["logup_hadamards"] * simt_rate
            buckets["row_combination"] = (tm["t.matmul"] - t_gkr_mm - t_logup_mm, "modelled matmul @0.172")
            buckets["other SIMT (logUp tree build, ...)"] = (tm["t.simt_field"] - t_gkr_simt - t_logup_simt, "modelled SIMT")
            cands = {}
            if "fused_layer_graphed_s" in meas:
                cands["fused SIMT layer model x96 (graphed)"] = meas["fused_layer_graphed_s"] * tx.UNITS_PER_VU
            if "packed_k3_batch_s_padded" in meas:
                cands["packed k=3 six-layer batch (padded)"] = meas["packed_k3_batch_s_padded"]
            if "packed_k0_batch_s_padded" in meas:
                cands["plain k=0 six-layer batch (padded)"] = meas["packed_k0_batch_s_padded"]
            if "packed2_best_batch_s_padded" in meas:
                cands[meas["packed2_best_label"] + " six-layer batch (padded)"] = meas["packed2_best_batch_s_padded"]
            if cands:
                best_label = min(cands, key=cands.get)
                t_gkr = cands[best_label]
                buckets["gkr_checker_sumcheck"] = (t_gkr, f"measured: {best_label} (model @0.172: {(t_gkr_mm + t_gkr_simt)*1e3:.0f} ms; alternatives "
                                                   + ", ".join(f"{k} {v*1e3:.0f} ms" for k, v in cands.items()) + ")")
                total = total - t_gkr_mm - t_gkr_simt + t_gkr
                if "logup_gkr_full_vu_s" in meas:
                    t_logup = meas["logup_gkr_full_vu_s"]
                    buckets["logup_fractional_gkr"] = (t_logup, f"measured: a-packed2 logUp fractional GKR at the full-VU shape (393,216 units x 306 lookups, 65,536-row table), k=3, "
                                                        f"whole bucket on the A100 (model @0.172: {(t_logup_mm + t_logup_simt)*1e3:.0f} ms; launch-bound: ~0.1-0.2 s of device work)")
                    total = total - t_logup_mm - t_logup_simt + t_logup
                elif "fused_layer_graphed_s" in meas and "logup_over_checker_round" in meas:
                    # per element the measured logUp round costs (logup_us / checker_us) of a fused checker round; the model's MAC ratio
                    # (which carries the 5-vs-3 table factor, hence x3/5) supplies the element count relative to the checker.
                    t_logup = meas["fused_layer_graphed_s"] * tx.UNITS_PER_VU / p.matmul["gkr_checker"] * p.matmul["logup_fractional_gkr"] * meas["logup_over_checker_round"] * 3 / 5
                    buckets["logup_fractional_gkr"] = (t_logup, f"measured rate: fused logUp round {meas['logup_round_2p21_us']:.0f} us = {meas['logup_over_checker_round']:.2f}x a fused checker round "
                                                        f"at 2^21, x the fused checker layer model x96 at the model's logUp/checker element ratio (model @0.172: {(t_logup_mm + t_logup_simt)*1e3:.0f} ms)")
                    total = total - t_logup_mm - t_logup_simt + t_logup
                else:
                    buckets["logup_fractional_gkr"] = (t_logup_mm + t_logup_simt, "modelled @0.172")
                    omitted.append("logUp fractional GKR (modelled @0.172)")
            else:
                buckets["gkr_checker_sumcheck"] = (t_gkr_mm + t_gkr_simt, "modelled @0.172")
                buckets["logup_fractional_gkr"] = (t_logup_mm + t_logup_simt, "modelled @0.172")
                omitted += ["checker GKR sumcheck (modelled @0.172)", "logUp fractional GKR (modelled @0.172)"]
        buckets["hint_generation"] = (tm["t.hint_gen"], "modelled")
        buckets["prg"] = (tm["t.prg"], "modelled")
        per_vu = total / tx.B_TARGET
        row = {"t.total": total, "seconds_per_vu": per_vu, "overhead.vs_native_peak": contract.overhead(per_vu, tx.K_TARGET),
               "buckets": {k: {"seconds": v[0], "basis": v[1], "share": v[0] / total} for k, v in buckets.items()},
               "measured_buckets": [k for k, v in buckets.items() if v[1].startswith("measured")],
               "omitted": omitted, "model_only_overhead": model["overhead.vs_native_peak"], "model_only_t.total": model["t.total"], "ligero_rows": p.ligero_rows, "hash_bytes": p.hash_bytes}
        out["designs"][dname] = row
        lines.append(f"| **{dname}** | {model['overhead.vs_native_peak']:.3g} ({model['t.total']*1e3:.0f} ms) | **{row['overhead.vs_native_peak']:.3g}** ({total*1e3:.0f} ms; {per_vu*1e6:.0f} us/VU) | "
                     + "; ".join(f"{k} **{v[0]*1e3:.1f}**" if v[1].startswith("measured") else f"{k} {v[0]*1e3:.1f}" for k, v in buckets.items()) + " |")
    txt = ("| design | model only (A100 peaks @0.172, Blake3 50 GB/s) | **A100-measured buckets**: overhead (t.total per batch) | buckets, ms (measured in bold) |\n|---|---|---|---|\n"
           + "\n".join(lines))
    return txt, out


# ---------------------------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None, help="default: the recorded runs, from the evidence store")
    ap.add_argument("--out", type=Path, default=REPO / REPORTS / "anchor_a100")
    a = ap.parse_args(argv)
    R = a.results or frozen.path("backends/shared/anchor_a100/results")
    runs = load(R / "runs.json") or {}
    mb = load(R / "microbench" / "microbench.json")
    bw_meas = mb["measured_peaks"]["hbm_copy_bytes_per_s"] if mb else DATASHEET_BW
    doc = {"lane": "anchor-a100", "track": "shared", "hardware": A100, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": runs,
           "labels": {"proof_class": "COMPONENT (kernels) / ARITHMETIC_DIAGNOSTIC (re-price)", "authentication": "excluded", "security.target": -128, "B": 4096, "K": 1536},
           "floors": {"hbm_bytes_per_s_measured_copy": bw_meas, "hbm_bytes_per_s_datasheet": DATASHEET_BW, "int8_macs_per_s_datasheet": DATASHEET_INT8_MACS,
                      "bandwidth_ratio_a100_over_4090_datasheet": DATASHEET_BW / BW_4090, "bandwidth_ratio_a100_over_h100_datasheet": DATASHEET_BW / BW_H100},
           "sections": {}, "kernels": []}
    md = [f"# anchor-a100: every campaign GPU kernel measured on the A100 (the anchor), {doc['generated'][:10]}", "",
          ("**Scope: component measurements** (kernel time on random/patterned data, bit-exact against each lane's CPU reference), then one "
          "`ARITHMETIC_DIAGNOSTIC` re-price of A and B with those numbers.  Nothing here is a proof except section 7 (the b-ligero torch prover, `NON_ZK_PROOF_DIAGNOSTIC`, verifier-accepted on the A100).  Every recorded 4090 / H100 number is the lane's "
          "own (its README / report, run id in the lane's table); every A100 number is a `research run --on vy-a100b --exclusive` of the same bench, same "
          "flags, CUDA events, warm, median of >= 7, bit-exact checks on.  `overhead.vs_native_peak` is defined against *this* card's dense BF16 peak "
          "(`contract.NATIVE_REFERENCE`, 9.85 ps per K=1536 VU), so the last section is the first A-vs-B comparison where the prover and the reference are the same machine."),
          "", f"Device: {mb['hardware']['nvidia_smi'] if mb else A100}; torch {mb['hardware']['torch'] if mb else '?'}, CUDA {mb['hardware']['cuda'] if mb else '?'}, "
          f"{mb['hardware']['sms'] if mb else 108} SMs (sm_80).  Recorded runs: " + ", ".join(f"{k} `{v}`" for k, v in runs.items()) + ".", "",
          ("Bandwidth ratios the earlier reports *scaled* by: A100/4090 = 2,039/1,008 = 2.02; A100/H100 = 2,039/3,350 = 0.61.  Speed ratios below are "
          "recorded-time / A100-time (> 1 = the A100 is faster); a departure from the bandwidth ratio by > 1.5x either way is a finding about the kernel and is flagged."), ""]
    meas = {}
    if mb:
        txt, s = sec_microbench(mb)
        doc["sections"]["microbench"] = s
        md += ["## 0. The anchor's measured roofline inputs", "", txt, "",
               (f"The floors below use the measured copy bandwidth **{bw_meas/1e9:.0f} GB/s** ({bw_meas/DATASHEET_BW:.2f} of the datasheet's 2,039) and the datasheet INT8 peak "
               f"312 T MAC/s; cuBLAS `_int_mm` reaches only {s['int8_macs_per_s_cublas_best']/1e12:.0f} T MAC/s ({s['int8_macs_per_s_cublas_best']/DATASHEET_INT8_MACS:.2f} of it) on this card "
               f"(the 4090 got 0.253 of its peak), while BF16 reaches {s['bf16_flops_per_s_cublas_best']/1e12:.0f} TFLOPS ({s['bf16_flops_per_s_cublas_best']/312e12:.2f} of the peak "
               "that defines the native reference).  The native reference is therefore ~6% conservative on this card, not more."), ""]
    hash_a = load(R / "hash_gpu" / "bench.json")
    hash_ref = load(frozen.path("backends/shared/hash_gpu/results_r20260922-054241-b966.json"))
    if hash_a and hash_ref:
        txt, d = sec_hash(hash_a, hash_ref, bw_meas)
        doc["kernels"] += d
        md += ["## 1. `verity/ml/kernels/hash_gpu` -- Merkle commitment hashing (4090 run `r20260922-054241-b966`)", "", txt, ""]
        sha_b = next(r for r in d if r.get("hash") == "sha256" and r.get("candidate") == "B" and r.get("kernel") == "hash_gpu")
        b3_b = next(r for r in d if r.get("hash") == "blake3" and r.get("candidate") == "B" and r.get("kernel") == "hash_gpu")
        best_b = max((r for r in d if r.get("kernel") == "hash_gpu" and r.get("candidate") == "B" and r.get("mds") is None and r["hash"] in ("sha256", "blake3")), key=lambda r: r["a100_gb_per_s"])
        best_a = max((r for r in d if r.get("kernel") == "hash_gpu" and r.get("candidate") == "A" and r.get("mds") is None and r["hash"] in ("sha256", "blake3")), key=lambda r: r["a100_gb_per_s"])
        meas["hash_rate_B"] = best_b["a100_gb_per_s"] * 1e9
        meas["hash_rate_A"] = best_a["a100_gb_per_s"] * 1e9
        meas["hash_B_kernel"] = best_b["hash"]
        meas["hash_B_ms"] = best_b["a100_ms"]
        meas["hash_A_kernel"] = best_a["hash"]
        md += [(f"**Finding (flagged).**  The SIMT hash kernels do *not* scale with bandwidth: SHA-256 runs at {sha_b['a100_gb_per_s']:.0f} GB/s on the A100 vs 446 on the 4090 "
               f"({sha_b['ratio_ref_over_a100_time']:.2f}x, where the bandwidth scaling assumed 2.02x -- a {2.02/sha_b['ratio_ref_over_a100_time']:.1f}x departure), Poseidon2-16 "
               f"{next(r for r in d if r.get('hash') == 'poseidon2-16' and r.get('candidate') == 'B' and r.get('mds') is None)['ratio_ref_over_a100_time']:.2f}x, Blake3 {b3_b['ratio_ref_over_a100_time']:.2f}x.  "
               "They are INT32-ALU-bound: the A100 has 108 SMs x 64 INT32 lanes at 1.41 GHz = 9.7 T int32 op/s against the 4090's 128 x 64 x 2.52 GHz = 20.6 T (0.47x), and SHA-256 / "
               "Poseidon2 land on that ratio.  Blake3 holds its rate (its chunk-parallel layout was latency-bound on the 4090).  On the anchor the best proof-internal hash is "
               f"**{best_b['hash']} at {best_b['a100_gb_per_s']:.0f} GB/s** ({best_b['a100_ms']:.1f} ms per B batch, {best_b['a100_ms']*1e3/4096:.1f} us/VU); the hash_gpu.md / encode_gpu.json "
               f"A100 rows (SHA-256 x2.02 = 901 GB/s) overstate the A100's hash rate by {901/best_b['a100_gb_per_s']:.1f}x."), ""]
    enc_a = load(R / "encode" / "bench.json")
    enc_ref = load(frozen.path("backends/direct/encode/results_r20260922-062016-6bdf.json"))
    if enc_a and enc_ref:
        txt, d = sec_encode(enc_a, enc_ref, bw_meas)
        doc["kernels"] += d
        md += ["## 2. `archive/direct/encode` -- Reed-Solomon encoders (4090 run `r20260922-062016-6bdf`)", "", txt, ""]
        for shape in ("B", "A"):
            best = min((r for r in d if r["shape"] == shape and r["encoder"].startswith("ntt_simt") and not r["encoder"].endswith("nosys")), key=lambda r: r["a100_ms"])
            meas[f"encoder_seconds_per_row_{shape}"] = best["a100_us_per_row"] * 1e-6
            meas[f"encoder_{shape}_kernel"] = best["encoder"]
        bb = min((r for r in d if r["shape"] == "B" and r["encoder"] == "ntt_simt_v2_t512"), key=lambda r: r["a100_ms"])
        tr = next((r for r in d if r["shape"] == "B" and r["encoder"] == "matdft_int8_triton"), None)
        md += [f"The SIMT NTT (v2, 512 threads) encodes the B batch in **{bb['a100_ms']:.1f} ms** on the A100 ({bb['ratio_ref_over_a100_time']:.2f}x the 4090; bandwidth ratio 2.02), "
               f"{bb['fraction_of_floor']:.2f} of the measured-bandwidth HBM floor ({bb['a100_hbm_floor_ms_measured_bw']:.1f} ms) and {bb['a100_hbm_floor_ms_datasheet']/bb['a100_ms']:.2f} of the datasheet floor.  "
               + (f"The fused Triton INT8 matrix DFT takes {tr['a100_ms']:.0f} ms ({tr['ratio_ref_over_a100_time']:.2f}x the 4090, {tr['a100_fraction_of_int8_peak']:.3f} of the A100 INT8 peak): "
                  f"the SIMT NTT beats it by {tr['a100_ms']/bb['a100_ms']:.1f}x on the anchor too." if tr else ""), ""]
    fused_a = load(R / "fused" / "bench.json")
    fused_ref = load(frozen.path(f"{REPORTS}/fused_sumcheck_4090.json"))
    if fused_a and fused_ref:
        txt, d = sec_fused(fused_a, fused_ref, bw_meas)
        doc["kernels"] += d
        md += ["## 3. `archive/gkr/tensor/fused` -- fused checker sumcheck round (4090 run `r20260922-060339-740b`)", "", txt, ""]
        Ls = fused_a["layers"].get("simt")
        if Ls:
            meas["fused_layer_graphed_s"] = Ls["graphed_seconds"]
            k21 = next((r for r in d if r.get("kernel") == "fused" and r["impl"] == "simt" and r["n"] == 1 << 21 and r["deg_in"] == 6), None)
            if k21:
                meas["fused_checker_round_2p21_us"] = k21["a100_us"]
                md += [(f"The SIMT fused round at 2^21 (ext input, cold L2) runs **{k21['a100_us']:.0f} us** on the A100 ({k21['ratio_ref_over_a100_time']:.2f}x the 4090's 273 us; "
                       f"{k21['a100_gb_per_s']:.0f} GB/s = {k21['fraction_of_floor']:.2f} of the measured-bandwidth floor).  The 130-round layer model: {Ls['graphed_seconds']*1e3:.2f} ms graphed "
                       f"({fused_ref['measured']['layers']['simt']['graphed_seconds']/Ls['graphed_seconds']:.2f}x the 4090's 6.20 ms)."), ""]
    logup_a = load(R / "fused_logup" / "fused_logup.json")
    logup_ref = load(frozen.path(f"{REPORTS}/fused_logup_4090.json"))
    if logup_a and logup_ref:
        txt, d = sec_logup(logup_a, logup_ref, bw_meas)
        doc["kernels"] += d
        md += ["## 3b. `archive/gkr/tensor/fused` -- fused logUp fractional-GKR round (4090 report `fused_logup_4090.json`)", "", txt, ""]
        best = min((r for r in d if r["n"] == 1 << 21), key=lambda r: r["a100_us"], default=None)
        if best and "fused_checker_round_2p21_us" in meas:
            meas["logup_round_2p21_us"] = best["a100_us"]
            meas["logup_over_checker_round"] = best["a100_us"] / meas["fused_checker_round_2p21_us"]
    packed_a = load(R / "packed" / "bench.json")
    packed_ref = load(frozen.path(f"{REPORTS}/packed_sumcheck_h100.json"))
    if packed_a and packed_ref:
        txt, d = sec_packed(packed_a, packed_ref)
        doc["kernels"] += d
        md += ["## 4. `archive/gkr/packed` -- base-field-first packed sumcheck (H100 run `r20260922-062108-4b25`)", "", txt, ""]
        for k in (0, 3):
            La = packed_a["layers"].get(str(k)) or packed_a["layers"].get(k)
            if La:
                meas[f"packed_k{k}_batch_s_padded"] = La["batch_seconds.padded"]
                meas[f"packed_k{k}_per_vu_s"] = La["per_vu_seconds.padded"]
        best_k = min(packed_a["layers"], key=lambda kk: packed_a["layers"][kk]["batch_seconds.padded"])
        Lb = packed_a["layers"][best_k]
        r26 = next((r for r in d if r.get("kernel") == "packed" and r["log2_t"] == 26 and r["k"] == 3), None)
        md += [f"Best k on the A100: **k={best_k}**, {Lb['batch_seconds.padded']*1e3:.0f} ms per six-layer batch = {Lb['per_vu_seconds.padded']*1e6:.1f} us/VU "
               f"(H100: k=3, 257 ms, 62.7 us/VU)." + (f"  At 2^26 k=3: {r26['a100_s']*1e3:.2f} ms vs 9.08 on the H100 ({r26['ratio_ref_over_a100_time']:.2f}x; bandwidth ratio 0.61, "
                                                     f"INT8-peak ratio 0.32)." if r26 else ""), ""]
    p2_a26 = load(R / "wave2_gkr" / "packed26" / "bench.json")
    p2_h26 = load(frozen.path("backends/gkr/packed/results/r20260922-064917-c293/bench.json"))
    p2_a28 = load(R / "wave2_gkr" / "packed28" / "bench.json")
    p2_h28 = load(frozen.path("backends/gkr/packed/results/r20260922-065412-73be/bench.json"))
    lg_a = load(R / "wave2_gkr" / "logup" / "logup_bench.json")
    lg_h = load(frozen.path("backends/gkr/packed/results/r20260922-073439-1f15/logup_bench.json"))
    if p2_a26 and p2_h26 and p2_a28 and p2_h28 and lg_a and lg_h:
        txt, d = sec_packed2(p2_a26, p2_h26, p2_a28, p2_h28, lg_a, lg_h)
        doc["kernels"] += d
        md += ["## 4b. `archive/gkr/packed` second pass (a-packed2, landed on main 07:4xZ): 32-bit Montgomery packed sumcheck, direct 2^28, logUp fractional GKR "
               "(H100 runs `r20260922-064917-c293`, `r20260922-065412-73be`, `r20260922-073439-1f15`; A100 run `r20260922-075147-91a5`)", "", txt, ""]
        best = min((r for r in d if r["kernel"] == "packed2.layers"), key=lambda r: r["a100_batch_s_padded"])
        meas["packed2_best_batch_s_padded"] = best["a100_batch_s_padded"]
        meas["packed2_best_batch_s_unpadded"] = best["a100_batch_s_unpadded"]
        meas["packed2_best_label"] = f"Montgomery packed k={best['k']} from the 2^{best['log2_t_fit']} fit"
        l3 = next((r for r in d if r["kernel"] == "logup_gkr" and r["k"] == 3), None)
        if l3:
            meas["logup_gkr_full_vu_s"] = l3["a100_s"]
        r28 = next((r for r in d if r["kernel"] == "packed2" and r["log2_t"] == 28 and r["k"] == 3), None)
        md += [(f"The Montgomery kernels scale with bandwidth on the A100: 0.52-0.61x the H100 at 2^26 (bandwidth ratio 0.61), k=3 direct 2^28 **{r28['a100_s']*1e3:.1f} ms** "
                f"({r28['ratio_ref_over_a100_time']:.2f}x the H100's 17.7; {r28['a100_fraction_of_int8_peak']:.1%} of the A100 INT8 peak) = **{best['a100_per_vu_s']*1e6:.1f} us/VU, {best['a100_batch_s_padded']*1e3:.0f} ms per six-layer batch** "
                f"padded ({best['a100_batch_s_unpadded']*1e3:.0f} un-padded) -- 1.8x better than the first-pass packed k=3 on the same card (476 ms).  k=2 and k=3 are within 4% of each other on the A100 "
                f"(the base x base Hadamard share is INT32 work at 0.47x the H100's rate).  **Finding (flagged).**  The logUp fractional GKR runs at **{l3['a100_s']*1e3:.0f} ms = {l3['a100_us_per_vu']:.0f} us/VU** "
                f"on the A100 vs 427 on the H100 ({l3['ratio_ref_over_a100_time']:.2f}x: a 1.7x departure from the bandwidth ratio) -- it is host-launch-bound (~1 ms x 351 rounds; the 20 smallest levels are "
                f"{l3['a100_tail_levels_ms']:.0f} of the {l3['a100_s']*1e3:.0f} ms on both cards), so the GPU does not matter yet; the device work is ~0.1-0.2 s.  Fixing the launch loop (CUDA graphs per level) is worth ~2x on either card."), ""]
    tensor_a = load(R / "tensor" / "bench.json")
    tensor_ref = load(frozen.path(f"{REPORTS}/tc_sumcheck_4090.json"))
    if tensor_a and tensor_ref:
        txt, d = sec_tensor(tensor_a, tensor_ref)
        doc["kernels"] += d
        md += ["## 5. `archive/gkr/tensor` -- the original INT8-limb round (4090 run `r20260922-051730-15d4`)", "", txt, ""]
    commit_a = load(R / "wave2_b" / "commit.json")
    commit_4 = load(frozen.path("backends/direct/encode/results_commit_r20260922-064722-07c7.json"))
    buckets_a = load(R / "wave2_b" / "buckets.json")
    buckets_4 = load(frozen.path("backends/direct/encode/results_buckets_r20260922-065537-279b.json"))
    lint_a = load(R / "wave2_b" / "lintest.json")
    lint_4 = load(frozen.path("backends/direct/encode/results_lintest_r20260922-073017-0924.json"))
    if commit_a and commit_4 and buckets_a and buckets_4 and lint_a and lint_4:
        txt, d = sec_bwave(commit_a, commit_4, buckets_a, buckets_4, lint_a, lint_4, bw_meas)
        doc["kernels"] += d
        md += ["## 6. `archive/direct/encode` b-commit + b-lintest -- fused commitment and B's prover buckets (4090 runs `r20260922-064722-07c7`, `r20260922-065537-279b`, "
               "`r20260922-073017-0924`; A100 run `r20260922-080250-cff5`)", "", txt, ""]
        meas["bwave"] = {"commit": commit_a, "buckets": buckets_a, "lintest": lint_a}
        best_c = min((r for r in d if r["kernel"] == "commit" and r["shape"] == "B" and "hash" in r), key=lambda r: r["a100_ms"])
        ref_c = min((r for r in commit_4["results"] if r["shape"] == "B" and r["mode"] == "fused"), key=lambda r: r["ms_median"])
        l3 = next((r for r in d if r["kernel"] == "lintest" and r["name"] == "linear_test_structured_M3_X256_lazy"), None)
        inv = min((r for r in d if r["kernel"] == "buckets" and r["bucket"] == "batch_inverse"), key=lambda r: r["a100_ms"])
        md += [(f"**Finding (flagged).**  The fused commitment does *not* carry over: the best fused variant on the A100 is **{best_c['name']} at {best_c['a100_ms']:.0f} ms** "
                f"({best_c['ratio_ref_over_a100_time']:.2f}x the 4090's best {ref_c['ms_median']:.1f} ms, a {2.02/best_c['ratio_ref_over_a100_time']:.0f}x departure from the bandwidth ratio) against "
                f"separate encode {meas['encoder_seconds_per_row_B']*310602*1e3:.0f} + Blake3 hash {meas['hash_B_ms']:.0f} = {meas['encoder_seconds_per_row_B']*310602*1e3 + meas['hash_B_ms']:.0f} ms measured in sections 1-2 -- fusing gains 1.1x on the anchor "
                f"where it gained 1.8x on the 4090, and the L2-resident small slices that won there lose here (256-row slices 155 ms vs 98 for 65,536-row: 40 MB L2 vs 72 MB, 0.47x the INT32 rate; the 'overlap' variants are 15-30% *slower* on the A100).  The other buckets scale at the INT32 ratio "
                f"(0.4-0.7x) except the batch inversion ({inv['ratio_ref_over_a100_time']:.2f}x: HBM-bound) and the structured linear test at M=1 (1.21x lazy); M=3 lazy = "
                + (f"{l3['a100_ms']:.1f} ms ({l3['ratio_ref_over_a100_time']:.2f}x)." if l3 else "")), ""]
    lig_a = {n: load(R / "ligero" / f"{n}.json") for n in ("ligero_unit_b4096", "ligero_vu_42", "ligero_vu_85")}
    lig_ref = {"ligero_unit_b4096": load(frozen.path("backends/direct/ligero/results/result_unit_b4096.json")), "ligero_vu_42": load(frozen.path("backends/direct/ligero/results/result_vu_42.json")),
               "ligero_vu_85": load(frozen.path("backends/direct/ligero/results/result_vu_85.json"))}
    if all(lig_a.values()) and all(lig_ref.values()):
        txt, d = sec_ligero(lig_a, lig_ref, bw_meas)
        doc["kernels"] += d
        gate = (R / "ligero" / "ligero_gate_vu.log").read_text().strip().splitlines()[-3:] if (R / "ligero" / "ligero_gate_vu.log").exists() else []
        v85 = next(r for r in d if r["bench"] == "ligero_vu_85")
        md += ["## 7. `archive/direct/ligero` -- the b-ligero torch prover: an actual proof of the VU chain on the anchor (L40S record `results/result_*.json`; A100 run `r20260922-081113-f56e`)", "", txt, "",
               ("Gate on the A100 (`run gate-vu --vus 64`): " + "; ".join(gate) + ".") if gate else "", "",
               (f"This is the one **end-to-end proof** in this report (`NON_ZK_PROOF_DIAGNOSTIC`, 2^-128 by accounting, depth 2, authentication excluded, ZK masking not implemented): "
                f"**{v85['a100_seconds_per']*1e3:.2f} ms/VU at 85 VUs = overhead {v85['overhead_vs_native_peak']:.3g}** against the same card's BF16 peak "
                f"({v85['ratio_ref_over_a100_time']:.2f}x the L40S's 2.30 ms/VU; the L40S has 864 GB/s, so the bandwidth ratio would be 2.03x).  The torch-eager encode and tests scale 1.5-1.9x, the hint / witness "
                "generation is launch-bound (0.6-0.9x) and the host verifier is slower on this pod's CPU.  It is 34x the re-priced bucket sum for B below (52 us/VU): the torch prover is unfused eager "
                "torch over a 3,516-row-per-unit constraint system, the buckets are the fused CUDA kernels on the model's B shape -- the gap is the port, not the design."), ""]
    # A vs B
    if meas:
        txt, rp = reprice(meas)
        doc["reprice"] = rp
        md += ["## A vs B on the A100, same batch, same numbers", "",
               ("`explore.tensor` designs A and B (BabyBear, degree-6 challenge, 2^-128, B = 4096, K = 1536, HVZK layer, authentication excluded) priced on the A100 with the "
               "A100-measured buckets substituted; the still-modelled buckets are at 0.172 of the A100 INT8 peak / the model's SIMT rate and are listed as `omitted`.  "
               "Labels: `ARITHMETIC_DIAGNOSTIC`, scope `vu`, hardware NVIDIA A100 SXM4 80GB."), "", txt, "",
               "Measured inputs: " + "; ".join(f"`{k}` = {fmt(v, 4) if isinstance(v, float) else v}" for k, v in meas.items() if k != "bwave"), ""]
        for dname, row in rp["designs"].items():
            md += [f"* **{dname}**: overhead **{row['overhead.vs_native_peak']:.3g}** ({row['t.total']*1e3:.0f} ms per batch, {row['seconds_per_vu']*1e6:.0f} us/VU).", ""]
            md += [f"  * {k}: {v['seconds']*1e3:.1f} ms ({v['share']:.0%}) -- {v['basis']}" for k, v in row["buckets"].items()]
            md += ["", "  Omitted (still modelled): " + "; ".join(row["omitted"]) + ".", ""]
        md.append("")
    # headline table (one row per kernel) inserted after the intro
    K = doc["kernels"]

    def pick(**m):
        return next((r for r in K if all(r.get(k) == v for k, v in m.items())), None)

    def hl(label, r, ref, a, ratio_key="ratio_ref_over_a100_time", frac=None, ref_gpu=None):
        if r is None:
            return None
        fr = r.get(frac) if frac else r.get("fraction_of_floor")
        return [label, f"{ref(r)} ({ref_gpu or r.get('ref_gpu', '')})", a(r), f"{r[ratio_key]:.2f}x", "--" if fr is None else f"{fr:.2f}"]
    ms = lambda k: (lambda r: f"{r[k]:,.1f} ms")
    us = lambda k: (lambda r: f"{r[k]:,.0f} us")
    sec = lambda k: (lambda r: f"{r[k]*1e3:,.1f} ms")
    rows = [hl("hash_gpu: SHA-256 Merkle tree, B matrix (20.4 GB)", pick(kernel="hash_gpu", hash="sha256", mds=None, candidate="B"), lambda r: f"{r['ref_gb_per_s']:.0f} GB/s", lambda r: f"{r['a100_gb_per_s']:.0f} GB/s"),
            hl("hash_gpu: Blake3 Merkle tree, B", pick(kernel="hash_gpu", hash="blake3", mds=None, candidate="B"), lambda r: f"{r['ref_gb_per_s']:.0f} GB/s", lambda r: f"{r['a100_gb_per_s']:.0f} GB/s"),
            hl("hash_gpu: Poseidon2-16 Merkle tree (cuda), B", pick(kernel="hash_gpu", hash="poseidon2-16", mds=None, candidate="B"), lambda r: f"{r['ref_gb_per_s']:.1f} GB/s", lambda r: f"{r['a100_gb_per_s']:.1f} GB/s"),
            hl("encode: SIMT NTT v2 (t1024), B batch", pick(kernel="encode", encoder="ntt_simt_v2_t1024", shape="B"), ms("ref_ms"), ms("a100_ms")),
            hl("encode: Triton INT8 matrix DFT, B", pick(kernel="encode", encoder="matdft_int8_triton", shape="B"), ms("ref_ms"), ms("a100_ms")),
            hl("fused checker round, SIMT, 2^21 ext input", pick(kernel="fused", impl="simt", n=1 << 21, deg_in=6), us("ref_us"), us("a100_us")),
            hl("fused checker round, TC, 2^21 ext input", pick(kernel="fused", impl="tc", n=1 << 21, deg_in=6), us("ref_us"), us("a100_us")),
            hl("fused 130-round layer model, SIMT, graphed", pick(kernel="fused.layers", impl="simt"), sec("ref_graphed_s"), sec("a100_graphed_s")),
            hl("fused logUp round, 2^21 (best block)", min((r for r in K if r.get("kernel") == "fused_logup" and r["n"] == 1 << 21), key=lambda r: r["a100_us"], default=None), us("ref_us"), us("a100_us")),
            hl("packed sumcheck (first pass) 2^26, k=0", pick(kernel="packed", log2_t=26, k=0), sec("ref_s"), sec("a100_s"), frac="a100_fraction_of_int8_peak"),
            hl("packed sumcheck (first pass) 2^26, k=3", pick(kernel="packed", log2_t=26, k=3), sec("ref_s"), sec("a100_s"), frac="a100_fraction_of_int8_peak"),
            hl("Montgomery packed sumcheck 2^26, k=3 (a-packed2)", pick(kernel="packed2", log2_t=26, k=3), sec("ref_s"), sec("a100_s"), frac="a100_fraction_of_int8_peak"),
            hl("Montgomery packed sumcheck direct 2^28, k=3", pick(kernel="packed2", log2_t=28, k=3), sec("ref_s"), sec("a100_s"), frac="a100_fraction_of_int8_peak"),
            hl("logUp fractional GKR, full-VU shape, k=3", pick(kernel="logup_gkr", k=3), sec("ref_s"), sec("a100_s")),
            hl("INT8-limb round (tensor lane), 2^21 deg 6, graphed", pick(kernel="tensor", t=1 << 21, deg=6), us("ref_us"), us("a100_us"), frac="a100_fraction_of_int8_peak_round"),
            hl("INT8-limb 130-round layer model, graphed wall", pick(kernel="tensor.layers", key="total_graphed"), sec("ref_s"), sec("a100_s"), ref_gpu="RTX 4090"),
            hl("fused commitment (encode -> Blake3 -> root), B, best 4090 config (1024-row slices)", pick(kernel="commit", name="blake3_s1024_ov1_keep0_fused", shape="B"), ms("ref_ms"), ms("a100_ms")),
            hl("fused commitment, B, best A100 config (65,536-row slices)", pick(kernel="commit", name="blake3_s65536_ov0_keep0_fused", shape="B"), ms("ref_ms"), ms("a100_ms")),
            hl("IRS row combination, B (chunk 1024)", pick(kernel="buckets", name="irs_row_combination_c1024", shape="B"), ms("ref_ms"), ms("a100_ms")),
            hl("quadratic test, B (s1024)", pick(kernel="buckets", name="quadratic_test_s1024", shape="B"), ms("ref_ms"), ms("a100_ms")),
            hl("structured linear test M=3 lazy, B (b-lintest)", pick(kernel="lintest", name="linear_test_structured_M3_X256_lazy"), ms("ref_ms"), ms("a100_ms")),
            hl("F_c batch inversion, B (T=262144)", pick(kernel="buckets", name="batch_inverse_T262144", shape="B"), ms("ref_ms"), ms("a100_ms")),
            hl("`_int_mm` 8192^3 (cuBLAS INT8 ceiling)", pick(kernel="tensor.int_mm", shape=[8192, 8192, 8192]), lambda r: f"{r['ref_s']*1e3:.1f} ms (0.253 of 4090 peak)", lambda r: f"{r['a100_s']*1e3:.1f} ms", frac="a100_fraction_of_int8_peak", ref_gpu="RTX 4090")]
    rows = [r for r in rows if r]
    head = ["## Headline: one row per kernel", "",
            ("Speed ratio = recorded time / A100 time (> 1: the A100 is faster).  The bandwidth ratios the earlier reports scaled by are **2.02** (vs the 4090) and **0.61** (vs the H100); "
             "`fraction of A100 floor` is the HBM floor at the measured 1,754 GB/s for streaming kernels and the fraction of the datasheet INT8 peak for tensor-core kernels."), "",
            md_table(["kernel", "recorded (card)", "A100 measured", "speed ratio", "fraction of A100 floor"], rows), "",
            ("**Pattern.**  Only the HBM-bound kernels scale with bandwidth (plain k=0 sumcheck 0.62x the H100, Montgomery packed 0.52-0.61x, batch inversion 1.48x the 4090).  Every INT32-SIMT kernel "
             "-- SHA-256, Poseidon2, the SIMT NTT encoder, the fused checker/logUp rounds, the quadratic / linear tests, the fused commitment -- lands near the A100's INT32 issue-rate ratio to the 4090 "
             "(108 SMs x 64 lanes x 1.41 GHz / 128 x 64 x 2.52 GHz = **0.47x**), not the 2.02x bandwidth ratio; the 4090 rows that were reported as 'fraction of HBM floor' were ALU-bound kernels on a "
             "card with a high ALU:bandwidth ratio.  Blake3 (1.05x) and the launch-bound kernels (logUp GKR 1.04x the H100; the torch INT8-limb round 0.6-1.0x) are card-independent.  "
             "Tensor-core kernels reach 4-6% of the A100's INT8 peak (cuBLAS itself only 12%)."), ""]
    intro_end = next(i for i, line in enumerate(md) if line.startswith("## 0.")) if any(line.startswith("## 0.") for line in md) else 8
    md[intro_end:intro_end] = head
    doc["headline"] = rows
    doc["would_not_run"] = WOULD_NOT_RUN
    md += ["## What would not run / caveats", ""] + [f"* {x}" for x in WOULD_NOT_RUN] + [""]
    a.out.with_suffix(".json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    a.out.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
