"""Roofline arithmetic for RS-encoding the Ligero witness matrix on an RTX 4090 (``ROOFLINE.md`` is generated from
this: ``python roofline.py``).  Pure Python, laptop-safe."""
from __future__ import annotations

import math

N, K = 16384, 4096
ROWS = {"B": 310_602, "A": 48_873}
ELEM = 4
RTX4090 = {
    "hbm_bytes_per_s": 1.008e12,
    "int8_macs_per_s": 330.3e12,             # 660.6 TOPS dense
    "fp32_flops": 82.6e12,
    "int32_ops_per_s": 128 * 64 * 2.52e9,    # 128 SMs x 64 INT32 lanes x 2.52 GHz = 20.6 T int32 op/s (IMAD-class; the FP32-only lanes cannot do integer)
    "smem_bytes_per_s": 128 * 128 * 2.52e9,  # 128 B/clk/SM shared-memory bandwidth = 41 TB/s aggregate
}
BUTTERFLY_INT32_OPS = 9      # Montgomery mul (2 x mul.lo + 1 x mul.hi + add + shift + cmp/sub ~ 6) + add/sub with correction (~3)
COSET_MUL_OPS = 6


def per_row() -> dict:
    lk, ln = int(math.log2(K)), int(math.log2(N))
    bf_textbook = (K // 2) * lk + (N // 2) * ln              # INTT_k + NTT_n (zero-padded)
    bf_coset = 4 * (K // 2) * lk                              # INTT_k + 3 x NTT_k (systematic: coset 0 is the input)
    sq = 1 << (lk // 2)
    return {
        "bytes_in": K * ELEM, "bytes_out": N * ELEM, "bytes_out_cosets_only": (N - K) * ELEM,
        "butterflies_textbook": bf_textbook, "butterflies_coset": bf_coset, "coset_twiddle_muls": 3 * K,
        "matdft_base_macs_model": K * 128 + N * 256,          # tensor.py ENCODERS['rs_matrix_ntt']: 64x64 interpolate + zero-padded 128x128 evaluate
        "matdft_base_macs_coset": 4 * 2 * K * sq,             # 4 DFTs x 2 stages x k x sqrt(k)
        "matdft_base_macs_zero_skipped": 2 * K * 64 + (128 * 32 * 128 + 128 * 128 * 128),   # interpolate 64x64 + evaluate with the 3/4-zero first stage skipped
        "dense_generator_macs": N * K,                        # 16384 x 4096 generator matrix
        "dense_generator_macs_nonsystematic_part": (N - K) * K,
        "expander_macs_per_symbol": 20,
        "expander_macs": 20 * N,
        "limb_factor": 16,
    }


def floors(rows: int, g: dict = RTX4090) -> dict:
    r = per_row()
    out = {"rows": rows}
    bin_, bout = rows * r["bytes_in"], rows * r["bytes_out"]
    out["bytes_in"], out["bytes_out"] = bin_, bout
    out["hbm_floor_s"] = (bin_ + bout) / g["hbm_bytes_per_s"]
    out["hbm_floor_write_only_s"] = bout / g["hbm_bytes_per_s"]
    out["hbm_floor_cosets_only_s"] = (bin_ + rows * r["bytes_out_cosets_only"]) / g["hbm_bytes_per_s"]
    # (a) radix-2 NTT on SIMT
    for tag in ("textbook", "coset"):
        bf = rows * r[f"butterflies_{tag}"]
        ops = bf * BUTTERFLY_INT32_OPS + (rows * r["coset_twiddle_muls"] * COSET_MUL_OPS if tag == "coset" else 0)
        out[f"ntt_{tag}_butterflies"] = bf
        out[f"ntt_{tag}_int32_ops"] = ops
        out[f"ntt_{tag}_simt_floor_s"] = ops / g["int32_ops_per_s"]
        # every butterfly reads and writes 2 words of shared memory
        out[f"ntt_{tag}_smem_floor_s"] = bf * 4 * ELEM / g["smem_bytes_per_s"]
    # (b) two-stage matrix DFT in INT8 limbs on tensor cores
    for tag in ("model", "coset", "zero_skipped"):
        macs = rows * r[f"matdft_base_macs_{tag}"] * r["limb_factor"]
        out[f"matdft_{tag}_limb_macs"] = macs
        out[f"matdft_{tag}_tc_floor_s"] = macs / g["int8_macs_per_s"]
        out[f"matdft_{tag}_at_0.172_s"] = macs / (g["int8_macs_per_s"] * 0.172)
        out[f"matdft_{tag}_at_0.25_s"] = macs / (g["int8_macs_per_s"] * 0.25)   # cuBLAS _int_mm 8192^3 ceiling measured in tc_sumcheck_4090
    # unfused INT32 intermediate traffic of the two-stage DFT: per element per stage 4 B limbs in, 64 B (16 INT32) out,
    # 64 B back in for recombination, 4 B out; 8 stages (4 DFTs x 2) per row of k elements
    out["matdft_unfused_bytes"] = rows * 8 * K * (4 + 64 + 64 + 4)
    out["matdft_unfused_hbm_floor_s"] = out["matdft_unfused_bytes"] / g["hbm_bytes_per_s"]
    # (c) dense generator matrix
    macs = rows * r["dense_generator_macs_nonsystematic_part"] * r["limb_factor"]
    out["dense_generator_limb_macs"] = macs
    out["dense_generator_tc_floor_s"] = macs / g["int8_macs_per_s"]
    # (d) Brakedown / Spielman expander (gather)
    macs = rows * r["expander_macs"] * r["limb_factor"]
    out["expander_limb_macs"] = macs
    out["expander_tc_floor_s"] = macs / g["int8_macs_per_s"]
    out["expander_simt_floor_s"] = rows * r["expander_macs"] * 8 / g["int32_ops_per_s"]     # ~8 int32 ops per base MAC (mul + accumulate, lazy reduction)
    out["expander_gather_bytes"] = rows * r["expander_macs"] * ELEM                          # one 4-B gather per MAC, uncoalesced
    out["expander_gather_floor_s"] = out["expander_gather_bytes"] / g["hbm_bytes_per_s"] * 8   # 32-B sectors for 4-B loads
    return out


def fmt_ms(s: float) -> str:
    return f"{s * 1e3:.1f} ms"


def table() -> str:
    lines = []
    r = per_row()
    lines.append("per row (k = 4096 -> n = 16384, 4-byte elements):")
    for k_, v in r.items():
        lines.append(f"  {k_:40} {v:,}")
    for cand, rows in ROWS.items():
        f = floors(rows)
        lines.append(f"\nCandidate {cand}: {rows:,} rows, {f['bytes_in'] / 1e9:.2f} GB in, {f['bytes_out'] / 1e9:.2f} GB out")
        lines.append(f"  HBM floor (read in + write full codeword)      {fmt_ms(f['hbm_floor_s'])}   (write alone {fmt_ms(f['hbm_floor_write_only_s'])}; cosets only {fmt_ms(f['hbm_floor_cosets_only_s'])})")
        lines.append(f"  (a) NTT textbook  {f['ntt_textbook_butterflies']:.3g} butterflies  SIMT floor {fmt_ms(f['ntt_textbook_simt_floor_s'])}  smem floor {fmt_ms(f['ntt_textbook_smem_floor_s'])}")
        lines.append(f"  (a) NTT coset     {f['ntt_coset_butterflies']:.3g} butterflies  SIMT floor {fmt_ms(f['ntt_coset_simt_floor_s'])}  smem floor {fmt_ms(f['ntt_coset_smem_floor_s'])}")
        for tag in ("model", "coset", "zero_skipped"):
            lines.append(f"  (b) matrix DFT {tag:13} {f[f'matdft_{tag}_limb_macs']:.3g} limb MACs  TC floor {fmt_ms(f[f'matdft_{tag}_tc_floor_s'])}  "
                         f"@0.25 {fmt_ms(f[f'matdft_{tag}_at_0.25_s'])}  @0.172 {fmt_ms(f[f'matdft_{tag}_at_0.172_s'])}")
        lines.append(f"  (b) unfused INT32 intermediates: {f['matdft_unfused_bytes'] / 1e9:.0f} GB of HBM traffic = {fmt_ms(f['matdft_unfused_hbm_floor_s'])}")
        lines.append(f"  (c) dense generator {f['dense_generator_limb_macs']:.3g} limb MACs  TC floor {fmt_ms(f['dense_generator_tc_floor_s'])}")
        lines.append(f"  (d) expander 20 MAC/symbol {f['expander_limb_macs']:.3g} limb MACs  TC floor {fmt_ms(f['expander_tc_floor_s'])}  "
                     f"SIMT {fmt_ms(f['expander_simt_floor_s'])}  gather (32-B sectors) {fmt_ms(f['expander_gather_floor_s'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
