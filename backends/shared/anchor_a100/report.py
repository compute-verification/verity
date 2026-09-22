"""anchor-a100 lane: the measured A100 column for every GPU kernel of the campaign, and A vs B re-priced on it.

    uv run --frozen --package verity-numerical python backends/shared/anchor_a100/report.py \
        --results backends/shared/anchor_a100/results --out backends/numerical/reports/anchor_a100

Inputs (fetched research runs, copied under ``--results/<kernel>/``): ``microbench/microbench.json``,
``hash_gpu/bench.json``, ``encode/bench.json``, ``fused/bench.json``, ``fused_logup/fused_logup.json``,
``packed/bench.json``, ``tensor/bench.json``.  Missing inputs leave their section out (the report is regenerated after
every kernel lands).  The recorded 4090 / H100 numbers are read from the lanes' own reports in
``backends/numerical/reports`` and ``backends/*/results_*.json``.

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
import time
from pathlib import Path

from verity_numerical.bench import contract
from verity_numerical.explore import tensor as tx
from verity_numerical.security import accounting as acc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPORTS = REPO / "backends" / "numerical" / "reports"
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


def sec_tensor(a100: dict, ref: dict) -> tuple[str, list[dict]]:
    rows, data = [], []
    for r4 in ref["measured"]["rounds"]:
        ra = next((r for r in a100["rounds"] if r["t"] == r4["t"] and r["deg"] == r4["deg"]), None)
        if not ra or r4["t"] < 4096:
            continue
        t4, ta = r4["graphed_round_seconds"] * 1e6, ra["graphed_round_seconds"] * 1e6
        rows.append([f"t=2^{int(math.log2(r4['t']))} deg {r4['deg']} round (graphed)", f"{t4:,.0f} us", f"{ta:,.0f} us", f"{t4/ta:.2f}x",
                     f"{ra['macs']/DATASHEET_INT8_MACS*1e6:.1f} us TC@312T", f"{ra['macs']/DATASHEET_INT8_MACS/ra['graphed_round_seconds']:.4f}"])
        data.append({"kernel": "tensor", "t": r4["t"], "deg": r4["deg"], "ref_gpu": "RTX 4090", "ref_us": t4, "a100_us": ta, "ratio_ref_over_a100_time": t4 / ta,
                     "a100_fraction_of_int8_peak_round": ra["macs"] / DATASHEET_INT8_MACS / ra["graphed_round_seconds"]})
    L4, La = ref["measured"]["layer_model_seconds"], a100["layer_model_seconds"]
    for key, label in (("graphed_wall", "layer model graphed wall (130 rounds)"), ("kernel_total", "layer model kernel time"), ("matmul", "  of which matmul kernels")):
        if key in L4 and key in La:
            rows.append([label, f"{L4[key]*1e3:.1f} ms", f"{La[key]*1e3:.1f} ms", f"{L4[key]/La[key]:.2f}x", "", ""])
            data.append({"kernel": "tensor.layers", "key": key, "ref_s": L4[key], "a100_s": La[key], "ratio_ref_over_a100_time": L4[key] / La[key]})
    for ca in a100.get("int_mm_ceilings", []):
        c4 = next((c for c in ref["measured"]["int_mm_ceilings"] if c.get("shape") == ca.get("shape")), None)
        if c4:
            rows.append([f"`_int_mm` {ca['shape']}", f"{c4['seconds']*1e6:,.0f} us ({c4['fraction_of_peak']:.3f} of 4090 peak)", f"{ca['seconds']*1e6:,.0f} us ({ca['macs']/DATASHEET_INT8_MACS/ca['seconds']:.3f} of A100 peak)",
                         f"{c4['seconds']/ca['seconds']:.2f}x", "", ""])
    txt = md_table(["INT8-limb round (tensor lane)", "RTX 4090 recorded", "A100 measured", "speed ratio A100/4090", "A100 TC floor", "fraction of INT8 peak"], rows)
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
        buckets = {"t.encoding": tm["t.encoding"], "t.hash": tm["t.hash"], "t.matmul_modelled": tm["t.matmul"], "t.simt_field_modelled": tm["t.simt_field"],
                   "t.hint_gen_modelled": tm["t.hint_gen"], "t.prg_modelled": tm["t.prg"]}
        measured = {"encoding": spr is not None, "hash": f"hash_rate_{dname}" in meas}
        omitted = ["witness/hint generation (modelled: 2e4 int ops per unit at 25% of the A100 FP32 peak)",
                   "Ligero row combination (modelled matmul @0.172 of the A100 INT8 peak)", "openings and transcript serialisation (not priced)",
                   "prover randomness (modelled PRG at 100 GB/s)"]
        total = tm["t.total"]
        if dname == "B":
            omitted += ["linear/quadratic tests, constraint combination, helper inversions (modelled SIMT at the A100 rate)"]
        else:
            t_gkr_model = p.matmul["gkr_checker"] / (gpu.peak_macs_per_s * UTIL) + p.simt_field_mults["gkr_hadamards"] * fs.simt_mult_vs_goldilocks / gpu.simt_field_mults_per_s
            t_logup_model = p.matmul["logup_fractional_gkr"] / (gpu.peak_macs_per_s * UTIL) + p.simt_field_mults["logup_hadamards"] * fs.simt_mult_vs_goldilocks / gpu.simt_field_mults_per_s
            buckets["t.gkr_checker_modelled"] = t_gkr_model
            buckets["t.logup_modelled"] = t_logup_model
            cands = {}
            if "fused_layer_graphed_s" in meas:
                cands["fused SIMT layer model x96 (graphed)"] = meas["fused_layer_graphed_s"] * tx.UNITS_PER_VU
            if "packed_k3_batch_s_padded" in meas:
                cands["packed k=3 six-layer batch (padded)"] = meas["packed_k3_batch_s_padded"]
            if "packed_k0_batch_s_padded" in meas:
                cands["plain k=0 six-layer batch (padded)"] = meas["packed_k0_batch_s_padded"]
            if cands:
                best_label = min(cands, key=cands.get)
                t_gkr = cands[best_label]
                measured["gkr_checker"] = best_label
                buckets["t.gkr_checker_measured"] = t_gkr
                buckets["gkr_checker_candidates"] = cands
                total = total - t_gkr_model + t_gkr
                # logUp at the fused kernels' measured per-element rates: the logUp round costs (logup_us / checker_us) of a checker round
                # over the same element count; the model's MAC ratio carries the element count.
                if "fused_layer_graphed_s" in meas and "logup_over_checker_round" in meas:
                    t_logup = meas["fused_layer_graphed_s"] * tx.UNITS_PER_VU / p.matmul["gkr_checker"] * p.matmul["logup_fractional_gkr"] * meas["logup_over_checker_round"] * 3 / 5
                    buckets["t.logup_measured_rate"] = t_logup
                    measured["logup"] = "fused logUp round rate (A100) x model element count"
                    total = total - t_logup_model + t_logup
                else:
                    omitted.append("logUp fractional GKR (modelled @0.172)")
            else:
                omitted += ["checker GKR sumcheck (modelled @0.172)", "logUp fractional GKR (modelled @0.172)"]
        per_vu = total / tx.B_TARGET
        row = {"t.total": total, "seconds_per_vu": per_vu, "overhead.vs_native_peak": contract.overhead(per_vu, tx.K_TARGET), "buckets": buckets, "measured": measured,
               "omitted": omitted, "model_only_overhead": model["overhead.vs_native_peak"], "model_only_t.total": model["t.total"], "ligero_rows": p.ligero_rows, "hash_bytes": p.hash_bytes}
        out["designs"][dname] = row
        lines.append(f"| {dname} | {model['overhead.vs_native_peak']:.3g} ({model['t.total']*1e3:.0f} ms) | {row['overhead.vs_native_peak']:.3g} ({total*1e3:.0f} ms) | "
                     + "; ".join(f"{k.replace('t.', '')} {v*1e3:.1f} ms" for k, v in buckets.items() if isinstance(v, float)) + " | "
                     + "; ".join(f"{k}: {v}" for k, v in measured.items()) + " |")
    txt = "| design | model only (A100 peaks @0.172, Blake3 50 GB/s) | **A100-measured buckets** overhead (t.total) | buckets | measured buckets |\n|---|---|---|---|---|\n" + "\n".join(lines)
    return txt, out


# ---------------------------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=REPORTS / "anchor_a100")
    a = ap.parse_args(argv)
    R = a.results
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
          "`ARITHMETIC_DIAGNOSTIC` re-price of A and B with those numbers.  Nothing here is a proof.  Every recorded 4090 / H100 number is the lane's "
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
    hash_ref = load(REPO / "backends/shared/hash_gpu/results_r20260922-054241-b966.json")
    if hash_a and hash_ref:
        txt, d = sec_hash(hash_a, hash_ref, bw_meas)
        doc["kernels"] += d
        md += ["## 1. `backends/shared/hash_gpu` -- Merkle commitment hashing (4090 run `r20260922-054241-b966`)", "", txt, ""]
        sha_b = next(r for r in d if r.get("hash") == "sha256" and r.get("candidate") == "B" and r.get("kernel") == "hash_gpu")
        b3_b = next(r for r in d if r.get("hash") == "blake3" and r.get("candidate") == "B" and r.get("kernel") == "hash_gpu")
        best_b = max((r for r in d if r.get("kernel") == "hash_gpu" and r.get("candidate") == "B" and r.get("mds") is None and r["hash"] in ("sha256", "blake3")), key=lambda r: r["a100_gb_per_s"])
        best_a = max((r for r in d if r.get("kernel") == "hash_gpu" and r.get("candidate") == "A" and r.get("mds") is None and r["hash"] in ("sha256", "blake3")), key=lambda r: r["a100_gb_per_s"])
        meas["hash_rate_B"] = best_b["a100_gb_per_s"] * 1e9
        meas["hash_rate_A"] = best_a["a100_gb_per_s"] * 1e9
        meas["hash_B_kernel"] = best_b["hash"]
        meas["hash_A_kernel"] = best_a["hash"]
        md += [(f"**Finding (flagged).**  The SIMT hash kernels do *not* scale with bandwidth: SHA-256 runs at {sha_b['a100_gb_per_s']:.0f} GB/s on the A100 vs 446 on the 4090 "
               f"({sha_b['ratio_ref_over_a100_time']:.2f}x, where the bandwidth scaling assumed 2.02x -- a {2.02/sha_b['ratio_ref_over_a100_time']:.1f}x departure), Poseidon2-16 "
               f"{next(r for r in d if r.get('hash') == 'poseidon2-16' and r.get('candidate') == 'B' and r.get('mds') is None)['ratio_ref_over_a100_time']:.2f}x, Blake3 {b3_b['ratio_ref_over_a100_time']:.2f}x.  "
               "They are INT32-ALU-bound: the A100 has 108 SMs x 64 INT32 lanes at 1.41 GHz = 9.7 T int32 op/s against the 4090's 128 x 64 x 2.52 GHz = 20.6 T (0.47x), and SHA-256 / "
               "Poseidon2 land on that ratio.  Blake3 holds its rate (its chunk-parallel layout was latency-bound on the 4090).  On the anchor the best proof-internal hash is "
               f"**{best_b['hash']} at {best_b['a100_gb_per_s']:.0f} GB/s** ({best_b['a100_ms']:.1f} ms per B batch, {best_b['a100_ms']*1e3/4096:.1f} us/VU); the hash_gpu.md / encode_gpu.json "
               f"A100 rows (SHA-256 x2.02 = 901 GB/s) overstate the A100's hash rate by {901/best_b['a100_gb_per_s']:.1f}x."), ""]
    enc_a = load(R / "encode" / "bench.json")
    enc_ref = load(REPO / "backends/direct/encode/results_r20260922-062016-6bdf.json")
    if enc_a and enc_ref:
        txt, d = sec_encode(enc_a, enc_ref, bw_meas)
        doc["kernels"] += d
        md += ["## 2. `backends/direct/encode` -- Reed-Solomon encoders (4090 run `r20260922-062016-6bdf`)", "", txt, ""]
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
    fused_ref = load(REPORTS / "fused_sumcheck_4090.json")
    if fused_a and fused_ref:
        txt, d = sec_fused(fused_a, fused_ref, bw_meas)
        doc["kernels"] += d
        md += ["## 3. `backends/gkr/tensor/fused` -- fused checker sumcheck round (4090 run `r20260922-060339-740b`)", "", txt, ""]
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
    logup_ref = load(REPORTS / "fused_logup_4090.json")
    if logup_a and logup_ref:
        txt, d = sec_logup(logup_a, logup_ref, bw_meas)
        doc["kernels"] += d
        md += ["## 3b. `backends/gkr/tensor/fused` -- fused logUp fractional-GKR round (4090 report `fused_logup_4090.json`)", "", txt, ""]
        best = min((r for r in d if r["n"] == 1 << 21), key=lambda r: r["a100_us"], default=None)
        if best and "fused_checker_round_2p21_us" in meas:
            meas["logup_round_2p21_us"] = best["a100_us"]
            meas["logup_over_checker_round"] = best["a100_us"] / meas["fused_checker_round_2p21_us"]
    packed_a = load(R / "packed" / "bench.json")
    packed_ref = load(REPORTS / "packed_sumcheck_h100.json")
    if packed_a and packed_ref:
        txt, d = sec_packed(packed_a, packed_ref)
        doc["kernels"] += d
        md += ["## 4. `backends/gkr/packed` -- base-field-first packed sumcheck (H100 run `r20260922-062108-4b25`)", "", txt, ""]
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
    tensor_a = load(R / "tensor" / "bench.json")
    tensor_ref = load(REPORTS / "tc_sumcheck_4090.json")
    if tensor_a and tensor_ref:
        txt, d = sec_tensor(tensor_a, tensor_ref)
        doc["kernels"] += d
        md += ["## 5. `backends/gkr/tensor` -- the original INT8-limb round (4090 run `r20260922-051730-15d4`)", "", txt, ""]
    # A vs B
    if meas:
        txt, rp = reprice(meas)
        doc["reprice"] = rp
        md += ["## A vs B on the A100, same batch, same numbers", "",
               ("`explore.tensor` designs A and B (BabyBear, degree-6 challenge, 2^-128, B = 4096, K = 1536, HVZK layer, authentication excluded) priced on the A100 with the "
               "A100-measured buckets substituted; the still-modelled buckets are at 0.172 of the A100 INT8 peak / the model's SIMT rate and are listed as `omitted`.  "
               "Labels: `ARITHMETIC_DIAGNOSTIC`, scope `vu`, hardware NVIDIA A100 SXM4 80GB."), "", txt, "",
               "Measured inputs: " + "; ".join(f"`{k}` = {fmt(v, 4) if isinstance(v, float) else v}" for k, v in meas.items()), ""]
        for dname, row in rp["designs"].items():
            md += [f"* **{dname}**: overhead **{row['overhead.vs_native_peak']:.3g}** ({row['t.total']*1e3:.0f} ms per batch, {row['seconds_per_vu']*1e6:.0f} us/VU).  Measured: "
                   + "; ".join(f"{k} = {v}" for k, v in row["measured"].items()) + ".  Omitted (modelled): " + "; ".join(row["omitted"]) + "."]
        md.append("")
    a.out.with_suffix(".json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    a.out.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
