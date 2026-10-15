"""Summarize a credit-ceilings run: each route's measured share of the FP8 tensor rate against M5's credit.

Usage: python3 summarize.py OUT_DIR > summary.md. Reads OUT_DIR/results.jsonl (credit_ceilings, ffma2_probe) and
OUT_DIR/smi-samples.csv (nvidia-smi clocks.sm every 200 ms during the lease). Shares are over 1,024 FP8 MACs a SM a
clock, M5's tensor capacity and the census's measured 1,023.8.
"""

import json
import statistics
import sys
from pathlib import Path

# M5 (note:20261015T0438Z-draft-gpu-model-m5, section 1.4): the credit a route's row grants, as a share.
CREDIT = {
    "dp4a_int8": ("dp4a", 0.250),
    "dp4a_karatsuba3": ("dp4a", 0.250),
    "imad_swar2": ("integer SWAR", 0.109),
    "smem_lut": ("shared-memory lookups", 0.094),
    "ffma_fp32": ("(FMA pipe; v2 fma weight)", 0.250),
    "combo_dp4a_ffma": ("dp4a + SWAR + lookups (issue rows)", 0.453),
}


def main(out: Path) -> None:
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines() if line.startswith("{")]
    smi = []
    csv = out / "smi-samples.csv"
    if csv.exists():
        for line in csv.read_text().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) > 1 and parts[1].endswith("MHz"):
                smi.append(float(parts[1].split()[0]))
    smi_txt = f"{statistics.median(smi):.0f} MHz (min {min(smi):.0f}, max {max(smi):.0f}, n={len(smi)})" if smi else "n/a"
    print(f"# credit ceilings: {out}\n\nnvidia-smi clocks.sm over the lease: {smi_txt}\n")
    print("| route | inputs | M x N x K | MACs/SM/clk | share | M5 row | credit | measured/credit | clock64 MHz | checked | mismatches |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") != "route":
            continue
        row, credit = CREDIT.get(r["route"], ("", float("nan")))
        print(
            f"| {r['route']} | {r['dist']} | {r['M']}x{r['N']}x{r['K']} | {r['macs_per_sm_clk']:.1f} | "
            f"{r['share_of_1024']:.4f} | {row} | {credit:.3f} | {r['share_of_1024'] / credit:.3f} | {r['sm_mhz']:.0f} | "
            f"{r['checked']} | {r['mismatches']} |"
        )
    print("\n| probe | warp instr/SM/clk (total) | per op | MAC-equivalent/SM/clk | clock64 MHz |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") == "probe":
            print(
                f"| {r['ops']} | {r['warp_instr_per_sm_clk']:.3f} | {r.get('per_op', r['warp_instr_per_sm_clk']):.3f} | "
                f"{r['mac_equiv_per_sm_clk']:.1f} | {r.get('sm_mhz', 0):.0f} |"
            )
    for r in rows:
        if r.get("kind") in ("decoder_selftest", "device"):
            print(f"\n`{json.dumps(r)}`")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
