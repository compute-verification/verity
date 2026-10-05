"""Laptop side: a fetched ``bench.json`` of ``fused/bench.py`` -> A-GKR re-priced with the measured fused round.

    uv run --package verity-numerical python archive/gkr/tensor/fused/reprice.py ~/.research/runs/<run>/bench.json \
        --report backends/numerical/reports/fused_sumcheck_4090

Substitution (``explore.tensor`` A row, BabyBear^6, 2^-128, Blake3 on SIMT at 50 GB/s, other buckets kept at the
model's own rates): the model's checker-GKR time (its ``gkr_checker`` matmul bucket at the assumed utilisation plus the
``gkr_hadamards`` SIMT bucket) is replaced by the measured 6-layer / 130-round fused sequence (CUDA-graphed wall time).
Two variants: the measured layer model uses 2^ceil(log2 width) tables (what the previous lane measured too); the
"census" variant scales it by ``sum(width) / sum(padded width)`` = the model's own element count.  The logUp fractional
GKR (same round shape, 5 tables) is priced two ways: the model's bucket at the assumed utilisation, and at the fused
kernel's measured per-element rate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from verity_numerical.bench import contract
from verity_numerical.explore import tensor as tx
from verity_numerical.security import accounting as acc

LAYER_WIDTHS = acc.CHECKER_LAYER_WIDTHS_A
UNITS = 4096


def pow2_pad(w: int) -> int:
    return 1 << max(1, math.ceil(math.log2(w)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", type=Path)
    ap.add_argument("--report", type=Path, default=Path("backends/numerical/reports/fused_sumcheck_4090"))
    ap.add_argument("--assumed", type=float, default=0.172)
    a = ap.parse_args(argv)
    doc = json.loads(a.bench.read_text())
    res_path = a.bench.with_name("result.json")
    res = json.loads(res_path.read_text()) if res_path.exists() else None
    problems = contract.validate(res) if res else ["result.json not beside bench.json"]
    peaks = doc["peaks"]
    gpu4090 = tx.Gpu(doc["hardware"]["gpu"]["name"], peaks["int8_dense_tops"], peaks["bf16_dense_tflops"], peaks["fp32_tflops"],
                     peaks["hbm_bytes_per_s"], 1.0e11 * peaks["fp32_tflops"] / 19.5, peaks["source"])
    fs = tx.FIELDS["babybear"]
    hr = tx.HASH_RATES["blake3_simt"]["bytes_per_s"]
    p = tx.price_design(tx.DESIGNS["A"], fs, -128)
    base = p.times(gpu4090, "A100", a.assumed, hr, "rs_matrix_ntt", fs)
    peak_macs = gpu4090.peak_macs_per_s
    t_gkr_model_mm = p.matmul["gkr_checker"] / (peak_macs * a.assumed)
    t_gkr_model_simt = p.simt_field_mults["gkr_hadamards"] * fs.simt_mult_vs_goldilocks / gpu4090.simt_field_mults_per_s
    t_logup_model_mm = p.matmul["logup_fractional_gkr"] / (peak_macs * a.assumed)
    t_logup_model_simt = p.simt_field_mults["logup_hadamards"] * fs.simt_mult_vs_goldilocks / gpu4090.simt_field_mults_per_s
    census_scale = sum(LAYER_WIDTHS) / sum(pow2_pad(w) for w in LAYER_WIDTHS)
    # the measured layer model is 4096 units = one VU-column of the batch; the B = 4096 VU batch has 96 x 4096 units.
    # (a) 96 x the measured sequence (conservative: 96 copies of the small-round tail); (b) the batch's bytes at the
    # sustained large-kernel bandwidth (the big-table estimate: tails amortised).
    scale = tx.UNITS_PER_VU
    big = [k for k in doc["kernels"] if k["n"] >= 2**22 and k["deg_in"] == 6 and k["mode"] == "fold+eval"]
    rows = {}
    for impl, L in doc["layers"].items():
        gbs = max(k["gb_per_s"] for k in big if k["impl"] == impl) * 1e9 if any(k["impl"] == impl for k in big) else None
        variants = [("graphed.x96", L["graphed_seconds"] * scale), ("graphed.x96.census_widths", L["graphed_seconds"] * scale * census_scale),
                    ("eager.x96", L["eager_seconds"] * scale), ("hbm_floor.x96", L["hbm_floor_seconds"] * scale)]
        if gbs:
            variants.append(("bytes.x96_at_sustained_gbs", L["bytes"] * scale / gbs))
        for label, t_meas in variants:
            # per-element rate of the fused round -> the logUp fractional GKR at the same rate (same kernel shape, 5 tables)
            rate = t_meas / p.matmul["gkr_checker"]
            t_logup_fused = p.matmul["logup_fractional_gkr"] * rate
            for logup_label, t_logup in (("logup modelled", t_logup_model_mm + t_logup_model_simt), ("logup at fused rate", t_logup_fused)):
                total = base["t.total"] - t_gkr_model_mm - t_gkr_model_simt + t_meas
                if logup_label == "logup at fused rate":
                    total = total - t_logup_model_mm - t_logup_model_simt + t_logup
                per_vu = total / tx.B_TARGET
                rows[f"{impl}.{label}.{logup_label.replace(' ', '_')}"] = {
                    "impl": impl, "gkr_time": label, "logup": logup_label, "t.gkr_checker": t_meas, "t.logup_fractional_gkr": t_logup,
                    "t.total": total, "overhead.vs_native_peak": contract.overhead(per_vu, tx.K_TARGET),
                    "gkr_share_of_time": t_meas / total,
                    "model_equivalent_utilisation": p.matmul["gkr_checker"] / (peak_macs * t_meas),
                    "sustained_gb_per_s_used": gbs}
    out = {"source": {"bench": str(a.bench), "run_id": res["run_id"] if res else None, "commit": doc["software"]["commit"]},
           "labels": {"proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "security.target": -128, "hardware": doc["hardware"], "software": doc["software"]},
           "contract_validation_problems": problems,
           "model": {"assumed_utilisation": a.assumed, "t.total": base["t.total"], "overhead.vs_native_peak": base["overhead.vs_native_peak"],
                     "t.gkr_checker.matmul": t_gkr_model_mm, "t.gkr_hadamards.simt": t_gkr_model_simt, "t.logup.matmul": t_logup_model_mm,
                     "t.logup_hadamards.simt": t_logup_model_simt, "t.matmul": base["t.matmul"], "t.simt_field": base["t.simt_field"], "t.hash": base["t.hash"],
                     "gkr_checker_limb_macs": p.matmul["gkr_checker"], "logup_limb_macs": p.matmul["logup_fractional_gkr"], "census_scale": census_scale,
                     "units_per_vu": scale, "note": "measured layer model = 4096 units; the batch = 96 x that"},
           "measured": {"layers": doc["layers"], "kernels": doc["kernels"], "launch": doc["launch"], "previous_composition": doc.get("previous_composition")},
           "repriced": rows}
    a.report.with_suffix(".json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"model A @ u={a.assumed}: t.total {base['t.total']:.3f} s, overhead {base['overhead.vs_native_peak']:.3g}; gkr_checker matmul {t_gkr_model_mm*1e3:.1f} ms + hadamards {t_gkr_model_simt*1e3:.1f} ms; "
          f"logup matmul {t_logup_model_mm*1e3:.1f} ms + hadamards {t_logup_model_simt*1e3:.1f} ms; hash {base['t.hash']*1e3:.1f} ms")
    for k, v in rows.items():
        print(f"{k:50} gkr {v['t.gkr_checker']*1e3:7.2f} ms  logup {v['t.logup_fractional_gkr']*1e3:7.2f} ms  t.total {v['t.total']:.3f} s  overhead {v['overhead.vs_native_peak']:.3g}  "
              f"u_eq {v['model_equivalent_utilisation']:.3f}")
    if problems:
        print("contract problems:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
