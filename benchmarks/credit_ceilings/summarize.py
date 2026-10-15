"""Summarize a credit-ceilings run: each route's measured share of the FP8 tensor rate against M5's credit.

Usage: python3 summarize.py OUT_DIR > summary.md. Reads OUT_DIR/results.jsonl and OUT_DIR/smi-samples.csv (nvidia-smi
clocks.sm every 200 ms during the lease). Shares are over 1,024 FP8 MACs a SM a clock, M5's tensor capacity and the
census's measured 1,023.8.
"""

import json
import statistics
import sys
from pathlib import Path

# M5 (note:20261015T0438Z-draft-gpu-model-m5, section 1.4): the row a route falls under and the share it grants.
CREDIT = {
    "dp4a_int8": ("dp4a", 0.250),
    "dp4a_karatsuba3": ("dp4a", 0.250),
    "ffma_fp32": ("dp4a (the FMA pipe)", 0.250),
    "imad_swar2": ("integer SWAR", 0.109),
    "smem_lut": ("shared-memory lookups", 0.094),
    "smem_lut8": ("shared-memory lookups", 0.094),
    "corun_dp4a_ffma": ("dp4a + SWAR + lookups", 0.453),
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
    print("| route | inputs | shape | config | MACs/SM/clk | share | M5 row | credit | measured/credit | clock64 MHz | blocks/SM | checked | mismatches |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") != "route":
            continue
        row, credit = CREDIT.get(r["route"], ("", float("nan")))
        shape = f"{r['M']}x{r['N']}x{r['K']}" if "Kd" not in r else f"Kd {r['Kd']}, Kf {r['Kf']}"
        print(
            f"| {r['route']} | {r['dist']} | {shape} | {r.get('config', '')} | {r['macs_per_sm_clk']:.1f} | {r['share_of_1024']:.4f} | {row} | "
            f"{credit:.3f} | {r['share_of_1024'] / credit:.3f} | {r['sm_mhz']:.0f} | {r['blocks_per_sm']} | "
            f"{r['checked']} | {r['mismatches']} |"
        )
    for r in rows:
        if r.get("route") == "corun_dp4a_ffma":
            print(
                f"\nCo-run: dp4a {r['dp4a_macs_per_sm_clk']:.1f} + FFMA {r['ffma_macs_per_sm_clk']:.1f} MACs/SM/clk "
                f"while both ran (alone at one block a SM: {r['dp4a_alone_macs_per_sm_clk']:.1f} and "
                f"{r['ffma_alone_macs_per_sm_clk']:.1f}); co-run time over the sum of alone times "
                f"{r['corun_over_sum_of_alone']:.3f}."
            )
    print("\n| probe | mix (dp4a, imad, ffma, lop3, ffma2) | warp instr/SM/clk | dp4a | imad | ffma | lop3 | ffma2 | MAC-equivalent/SM/clk | clock64 MHz |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") == "probe":
            print(
                f"| {r['ops']} | {r['mix']} | {r['warp_instr_per_sm_clk']:.3f} | {r['dp4a']:.3f} | {r['imad']:.3f} | "
                f"{r['ffma']:.3f} | {r['lop3']:.3f} | {r['ffma2']:.3f} | {r['mac_equiv_per_sm_clk']:.1f} | {r['sm_mhz']:.0f} |"
            )
    print("\n| split warps (even: dp4a, odd: FFMA/FFMA2) | mix (dp4a, ffma, ffma2) | dp4a | ffma | ffma2 | dp4a cycles | fma cycles | block cycles | MAC-equivalent/SM/clk |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("kind") == "split":
            print(
                f"| {r['ops']} | {r['mix']} | {r['dp4a']:.3f} | {r['ffma']:.3f} | {r['ffma2']:.3f} | {r['dp4a_cycles']:.0f} | "
                f"{r['fma_cycles']:.0f} | {r['block_cycles']:.0f} | {r['mac_equiv_per_sm_clk']:.1f} |"
            )
    for r in rows:
        if r.get("kind") in ("decoder_selftest", "device"):
            print(f"\n`{json.dumps(r)}`")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
