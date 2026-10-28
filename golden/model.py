#!/usr/bin/env python3
"""GD-MODEL: the golden demo's cost model, J(design).

    J = (t_remote / t_stock - 1) + (proving GPU-s a window) / (stock GPU-s a window)

at the measured clock, with the verification cost beside it. `J(design)` returns the serve-overhead parts (ms a step,
by term), the proving GPU-s a window (by part), the verification cost, the sampling draws and J. Every number the
model reads is in `INPUTS`, each with its value, its status (M measured, with its run id; E estimate; D derived from
other inputs) and its source. Stdlib only, so GD-SEARCH can import it anywhere: `from golden.model import J, design`.

The experiment: verity/remote/v0 (sampled proofs + PoUW, end to end on one served window), Llama-3.2-1B int8, workload
B4, one H200, window circuit 4,32,4; the demo's point is T 512 with 64 real tokens, and T 8,192 is the throughput point.

The design dict (`design(**overrides)` builds one; `SCHEMA` lists every key, its default and its allowed values;
`normalize` refuses an unknown key):

    point     T, real_tokens, seqs, steps_per_window, async, window_class
    forward   kernels ("stock" | "commit_taps"), gemm_eff, kfa3, graphs
    commit    on, values ("every_op" | "dedup" | "fused_mlp" | "layer" | "layer_kv" | "partition"), bytes_per_token,
              bits_per_word, row_bytes, hash_eff, corun, seal ("tree" | "none" | "roots"), host ("today" | "fixed"),
              gather, padded
    pouw      scheme ("off" | "j10r" | "jr2" | "dither"), K, L, L_prime, noise ("periodic" | "fresh" | "permuted"),
              record_bytes_per_cell, trees, h1_host ("pre" | "main" | "main+6192" | "stack" | "floor"), kdhash,
              hash_at_bar, coin_off_loop, noise_fused, denoise_g_frac, value_row_bytes, unit_cells
    sampling  delta, eps, pouw_sampled
    partition units: [{name, per_token, work, extra_ands, open_rows, out_bits}] or None for the preset of
              `commit.values`
    prover    o_per_and, staged_per_and, stmt_fixed, stmt_per_and, speed, m_max, shared
    lowering  gates_per_and, s_per_gate, bytes_per_gate, per ("template" | "instance"), ram_gib
    verify    stmt_fixed, stmt_per_and
    price     cpu_s_in_gpu_s, cpu_in_J, baseline ("same" | "fastest")

Units: ms for serve time a step, GPU-s and CPU-s for proving and verification a window, bytes and ANDs as counted.
"""
from __future__ import annotations

import copy
import json
import math
import sys

# ---------------------------------------------------------------------------------------------------------------------
# Inputs. (value, status, source). M = measured (run id), E = estimate, D = derived.
# ---------------------------------------------------------------------------------------------------------------------

B57F, B921, A1FC = "r20261028-080559-b57f", "r20261028-084012-b921", "r20261028-084022-a1fc"
C2CD6, T966F, TF746 = "r20261028-075211-2cd6", "r20261028-080609-966f", "r20261028-080612-f746"
H1POST, H1POST6192 = "r20261028-164426-5430", "r20261028-164438-65ca"
EEB0 = "r20261028-112958-eeb0 (tensor_util --steady re-read of b57f, 2cd6, b921, a1fc)"
COMMIT_BYTES = "note:compute-research/20261028T1600Z-finding-commit-bytes"
H1_COST = "note:compute-research/20261028T1605Z-finding-h1-step-cost"
LEVERS = "note:compute-research/20261028T1715Z-report-p1a-levers (art:bac79898511f50b4)"
FLOOR = "note:compute-research/20261028T0523Z-finding-dither-noised-floor"
ZK_RUN = "r20261018-060433-b55e (C-Flock --zk sessions of ScaledMmS8Coordinate_v1, sm_120; costfit over " \
         "benchmarks/sampling_plan/inputs.py)"

INPUTS: dict[str, tuple[float, str, str]] = {
    # the card
    "sms": (132, "M", "H200 SXM: 132 SMs"),
    "int8_macs_per_sm_clock": (4096, "M", "H200 dense s8 wgmma: 4,096 MACs an SM-clock (the ruler of BENCH-34)"),
    "hbm_GBs": (4800, "M", "H200 HBM3e data sheet, 4.8 TB/s"),
    # the stock forward (sync stack, Commit off), T 8,192, B4 prefill mix: 8 sequences of 1,024
    "gemm_eff_per_clock": (0.8610, "M", f"{B57F}: 7.9715e12 int8 MACs in 10.362 ms at 1,652.6 MHz (eeb0)"),
    "gemm_mhz": (1652.6, "M", f"{B57F} int8 GEMM's clock (eeb0)"),
    "attn_eff_per_clock": (0.2727, "M", f"{B57F}: 2.751e11 bf16 MACs (half rate) in 2.139 ms at 1,744.8 MHz"),
    "attn_mhz": (1744.8, "M", f"{B57F} attention's clock"),
    "ew_us_per_token": (0.5924, "M", f"{B57F}: activation, norms, rope, quantize, other, sampling, memcpy 4.853 ms "
                                     f"at T 8,192; {T966F} 0.581 and {TF746} 0.583 us a token"),
    "head_ms_per_seq": (0.0486, "M", f"{B57F}: LM head 0.389 ms over 8 sequences"),
    "host_gap_ms0": (2.42, "M", f"sync host gap between forwards, fit over {B57F}, {T966F}, {TF746} (7.88, 13.45, "
                                f"24.25 ms at T 8k, 16k, 32k): 2.42 ms + 0.666 us a token ({LEVERS})"),
    "host_gap_us_per_token": (0.666, "M", "as host_gap_ms0"),
    "async_gap_ratio": (2.290 / 7.882, "M", f"{C2CD6} (async stack) host gap 2.290 ms against {B57F}'s 7.882"),
    "launch_gap_ms0": (0.696, "M", f"{B57F} launch gaps in the forward at T 8,192 (0.70 ms; 2.21 at 16k, 3.73 at "
                                   f"32k), scaled with tokens above 8k"),
    "launch_us": (4.0, "E", f"a launch's floor at small T, 4 us ({LEVERS}'s V3 scaling)"),
    "fwd_launches": (113 + 64 + 32 + 1, "M", f"{B57F}: launches a step (elementwise 113, GEMM 64, attention 32, head 1)"),
    # the Commit (rows-sha512, sealed, --retain device), b921
    "commit_B_per_token": (1_663_627, "M", f"{B921}: 13,628,428,480 B a full step / 8,192 ({COMMIT_BYTES})"),
    "sha_bar_per_sm_clock": (0.01803, "M", "HASH-38's bar: 0.01803 SHA-512 compressions an SM-clock (r20261027-042422-"
                                           "85b2: 4.354 G/s at 1,830 MHz, 521 GB/s at 16 KiB rows)"),
    "hash_eff_today": (0.4966, "M", f"{B921}: 149.0 M compressions in 64.12 ms of hash kernels at 1,966 MHz ({LEVERS})"),
    "hash_overlap_today": (0.2245, "M", f"{B921}: 1 - 49.72 ms exclusive / 64.12 ms kernel time of the Commit's hashing"),
    "hash_mhz": (1966.0, "M", f"{B921} the Commit's hashing clock"),
    "corun_share": (0.167, "E", "r20261028-012220-309e: beside a synthetic GEMM at 98.8% of peak, 2 chain-kernel hash "
                                "warps an SM get 0.167 of the bar at kappa 1.001 (66.5 GB/s at 4 KiB rows); a model for "
                                "the Commit's row kernel beside the served GEMM until GD-HASH measures it"),
    "gemm_corun_penalty_per_hash_cycle": (0.347, "D", f"{B921}'s int8 GEMM 26.95 M cycles against {B57F}'s 17.12 M "
                                                      f"(+9.83 M), over the 28.3 M hash cycles that overlapped it"),
    "taps_ms": (1.41, "M", f"{B921}'s attention and elementwise kernels 8.78 ms against {B57F}'s 7.37 (the Commit's taps)"),
    "gather_ms": (3.371, "M", f"{B921} the Commit's gather, exclusive, at 13.63 GB a step"),
    "commit_copies_ms": (16.514, "M", f"{B921} memcpy, exclusive (0.56 GB D2H, 0.97 GB H2D host copies a step)"),
    "commit_launch_gap_ms": (107.81, "M", f"{B921} launch gaps 108.51 ms less {B57F}'s 0.70: 69 cudaEventSynchronize "
                                          f"(98.5 ms) and short gaps ({COMMIT_BYTES})"),
    "seal_rest_ms": (8.23, "M", f"{B921} host gap 1,350.91 less the seal wait 1,334.8 less {B57F}'s 7.88"),
    "seal_GBs": (0.3607, "D", f"{B921}: the seal wait 1,334.8 ms a step for its tree's 7.52 M leaves and nodes at 64 B "
                              f"(0.481 GB); COMMIT-BYTES measured 1.31 s a seal at 0.39 GB/s"),
    "commit_launches": (400 + 93, "M", f"{B921}: 399.6 hash and 92.8 gather launches a step"),
    # H1 (J10-R at K 2,048), a1fc and after the landed fixes
    "h1_value_check_ns_per_cell": (12.2, "M", f"{H1_COST}: 34.4 s over 2.82 G cells, host (11d1, {A1FC}, e6cb)"),
    "h1_tables_s_8k": (9.32, "M", f"{H1_COST}: host tables, zy words, registration, public rows, {A1FC}"),
    "h1_copy_GBs_records": (2.09, "M", f"{H1_COST}: 16.9 GB of records to pageable host memory in 8.10 s"),
    "h1_copy_y_s_8k": (2.55, "M", f"{H1_COST}: y to the host, {A1FC}"),
    "h1_copy_h2d_s_8k": (2.05, "M", f"{H1_COST}: zy table chunks host to device, {A1FC}"),
    "h1_copy_other_s_8k": (0.81 + 0.03 + 0.39, "M", f"{H1_COST}: other D2H, CUDA calls, rest of the forward, {A1FC}"),
    "h1_device_ms_8k": (774.5, "M", f"{LEVERS}: H1's kernels 764.4 ms (k_trees 408, k_leaves 134, k_segments 99, "
                                    f"k_rows 36, elementwise 35) plus its forward extras, {A1FC}, overlapped with host"),
    "h1_post_step_s_8k": (16.504, "M", f"{H1POST}: H1 on at T 8,192 after #5975 and #6016 (tree bc30456e: main b041 + "
                                       f"#6248's head, which holds #6016, #5975 and #6026), the Commit unsealed: 16.50 s"),
    "h1_post6192_step_s_8k": (9.73, "M", f"{H1POST6192}: as {H1POST} plus #6192 (the A table once): 9.73 s"),
    "h1_post_s_512": (16 * 0.0882, "M", "#6016's check (r20261028-034834-7b4b): one layer's four linears at T = m = 512, "
                                        "traced and judged, 0.0882 s; x 16 layers (a test harness, not a served step)"),
    "h1_coin_ms": (200.0, "M", "note:vllm-h1-async/20261027T2205Z-report-h1-step-ab: the round coin's wait, 0.20 s a "
                               "step on the engine thread at T 512"),
    "h1_floor_hash_GBs": (229.0, "M", f"{H1_COST}: the Commit's SHA-512 rate in the same capture (11.9 GB in 52 ms)"),
    "kdhash_ratio": (161.1 / 825.3, "M", "#6182's gate r20261028-082257-6d38: H1 kernels 825.3 -> 161.1 ms a step"),
    "h1_rest_floor_ms_8k": (10.0 + 4.7 + 3.5, "E", f"{H1_COST}'s floors: tables 10 ms, value check 4.7, record write 3.5"),
    # PoUW: the Dither floor and noise
    "noise_amp": (80, "M", "Dither v3.2: U, V uniform on +-80 (verity/core/protocols/pouw/dither/PROTOCOL.md)"),
    "b1_rate_vs_s8": (8.0, "E", f"{FLOOR}: the b1 product at 8x the s8 rate a clock, the coordinator's figure, "
                                f"unmeasured on Hopper (mma.sync only)"),
    "wrap_g_frac": (0.25, "E", "the wrap terms 256(w_X W'^T + X w_W^T + ...) of Dither's identity, as a share of G: an "
                               "estimate pending GD-DENOISE (sparse ternary w_X at the noise's wrap rate)"),
    "noise_fused_keep": (0.1, "E", "noise added in the GEMM's operand load keeps 10% of a separate HBM pass (estimate)"),
    # proving: C-Flock --zk, costfit over ZK_RUN. Measured on sm_120 at ~3-6 M ANDs a block; estimates for the H200
    # (and for larger blocks) until GD-PROVER's H200 fit lands.
    "o_per_and": (1.0870e-8, "E", f"{ZK_RUN}: GPU-s a drawn proof block's AND (prove fit, sigma 0.149)"),
    "staged_per_and": (1.0160e-10, "E", f"{ZK_RUN}: GPU-s a staged proof block's AND"),
    "stmt_fixed": (3.029, "E", f"{ZK_RUN}: GPU-s a touched statement"),
    "stmt_per_and": (3.5095e-6, "E", f"{ZK_RUN}: GPU-s a statement per AND of its proof block (its build)"),
    "stage_fixed": (1.571, "E", f"{ZK_RUN}: CPU-s to stage a statement"),
    "stage_per_and": (5.944e-7, "E", f"{ZK_RUN}: CPU-s a statement per AND of its proof block"),
    "stage_per_and_staged": (2.855e-9, "E", f"{ZK_RUN}: CPU-s a staged proof block's AND"),
    "verify_fixed": (12.436, "E", f"{ZK_RUN}: Lean verify --zk CPU-s a statement (sigma 0.108)"),
    "verify_per_and": (6.710e-6, "E", f"{ZK_RUN}: Lean verify --zk CPU-s a statement per AND of its proof block"),
    "h200_vs_sm120": (1.0, "E", "C-Flock's prover on the H200 against the RTX PRO 6000: taken equal until GD-PROVER"),
    "m_max": (35, "M", "C-Flock circuit.rs M_MAX: m = k_log + nbl <= 35 (note:proofs/20261028T1837Z-report-partopt)"),
    "sha512_ands": (58_120, "M", "optimize.SHA512_ANDS: one SHA-512 compression in Boolean form (hm96 PROTOCOL 5)"),
    "hm96_ands": (116_240, "M", "optimize.HM96_ANDS: the hiding part of an hm96-sha512 row in circuit"),
    "s8_step_ands": (4_190, "M", "benchmarks/sampling_plan/inputs.py S8_STEP: ANDs of 32 s8 MACs (#2391's pin)"),
    "s8_epilogue_ands": (5_000, "M", "benchmarks/sampling_plan/inputs.py EPILOGUE"),
    "ew_ands_per_elem": (2_000, "E", "an elementwise op's ANDs an element (norm, SiLU, RoPE, quantize, bf16 ops): "
                                     "estimate pending GD-PARTITION"),
    "attn_ands_per_mac": (600, "E", "bf16 attention's ANDs a MAC in circuit: estimate pending GD-PARTITION"),
    "h1_tile_gates": (1.0e9, "M", "the brief: the K 2,048 32x32 H1 tile is about 1e9 gates, OOM at 600 GiB (TOP-138)"),
    # lowering (TOP-138) and prices
    "lower_bytes_per_gate": (16 * 2 ** 30 / 1.0e9, "E", "TOP-138's target: the 1e9-gate H1 tile in 16 GiB (17.2 B a "
                                                        "gate); today's figure in lower_bytes_per_gate_today"),
    "lower_bytes_per_gate_today": (600.0, "M", "the brief: about 600 B a gate at the H1 tile today (OOM at 600 GiB), "
                                               "TOP-138; reported beside, as lower_peak_GiB_today"),
    "lower_s_per_gate": (0.9e-6, "E", "TOP-138's target, 1e9 gates in 15 min; today's figure pending TOP-138"),
    "gates_per_and": (3.5, "E", "the H1 tile's ~1e9 gates over its ~2.8e8 ANDs (1,024 cells x 64 s8 steps x 4,190)"),
    "cpu_s_in_gpu_s": (1 / 55, "M", "benchmarks/sampling_plan/inputs.py: vCPU $0.033/h against sm_120 $1.80/h"),
}


def I(key: str) -> float:
    return INPUTS[key][0]


# ---------------------------------------------------------------------------------------------------------------------
# The model: Llama-3.2-1B int8
# ---------------------------------------------------------------------------------------------------------------------

LAYERS, HIDDEN, INTER, QH, KVH, HD, VOCAB = 16, 2048, 8192, 32, 8, 64, 128_256
QKV = (QH + 2 * KVH) * HD                                 # 3,072
LINEARS = (("qkv", HIDDEN, QKV), ("o", HIDDEN, HIDDEN), ("gate_up", HIDDEN, 2 * INTER), ("down", INTER, HIDDEN))
INT8_MACS_PER_TOKEN = LAYERS * sum(k * n for _, k, n in LINEARS)          # 973.1 M
H1_CELLS_PER_TOKEN = LAYERS * sum(n for _, k, n in LINEARS if k == 2048)  # the K = 2,048 linears' cells: 344,064
WEIGHT_BYTES = INT8_MACS_PER_TOKEN                                         # int8: one byte a MAC a token

# COMMIT-BYTES' itemization a token (16 layers), at the committer's widths: (name, bf16 elements, other bytes)
STREAM = (
    ("gate_up output", LAYERS * 2 * INTER, 0),
    ("RMSNorm outputs and residuals, final norm", LAYERS * 2 * 2 * HIDDEN + 2 * HIDDEN, 0),
    ("SiluAndMul output", LAYERS * INTER, 0),
    ("int8 GEMM inputs x_q and row scales", 0, LAYERS * (3 * HIDDEN + INTER + HIDDEN * 0 + 4 * 4)),
    ("o_proj and down outputs", LAYERS * 2 * HIDDEN, 0),
    ("qkv output", LAYERS * QKV, 0),
    ("RoPE q, k", LAYERS * (QH + KVH) * HD, 0),
    ("attention output", LAYERS * HIDDEN, 0),
    ("embedding, norm scales, ids, positions", 0, 4_240),
)
# The presets of what's committed (COMMIT-BYTES section 4): which items, at bits_per_word for the bf16 ones.
VALUES = {
    "every_op": None,
    "dedup": ("SiluAndMul output", "RMSNorm outputs and residuals, final norm"),   # lever 11, approximately 1.27 MB
    "fused_mlp": ("gate_up output", "SiluAndMul output"),                          # lever 12, 0.88 MB
}
LAYER_BOUNDARY_ELEMS = (LAYERS + 1) * HIDDEN            # lever 13: each layer's residual stream, 69.6 KB at bf16
KV_ELEMS = LAYERS * 2 * KVH * HD                        # post-RoPE K and V, +32.8 KB at bf16


def stream_bytes_per_token(values: str, bits: int) -> float:
    """Bytes a token the Commit commits, from COMMIT-BYTES' itemization. The code's widths give 1,667,472 B at bf16,
    0.23% over b921's measured 1,663,627, so every preset is scaled by measured / code."""
    scale = I("commit_B_per_token") / sum(e * 2 + o for _, e, o in STREAM)
    if values == "layer":
        return LAYER_BOUNDARY_ELEMS * bits / 8
    if values == "layer_kv":
        return (LAYER_BOUNDARY_ELEMS + KV_ELEMS) * bits / 8
    drop = VALUES.get(values) or ()
    if values == "dedup":
        # drop SiLU's output and the bf16 norm outputs whose only reader is the int8 quantizer (3 of 4 a layer's 2)
        total = 0.0
        for name, e, o in STREAM:
            if name == "SiluAndMul output":
                continue
            if name.startswith("RMSNorm"):
                e = e - LAYERS * 2 * HIDDEN                       # the two norm outputs; residuals stay
            total += e * bits / 8 + o
        return total * scale
    return scale * sum(e * bits / 8 + o for name, e, o in STREAM if name not in drop)


def row_ands(nbytes: float) -> float:
    """verity.core.protocols.sampled_proofs.optimize.row_ands: in-circuit ANDs to open one hm96-sha512 row."""
    return I("hm96_ands") + I("sha512_ands") * math.ceil((nbytes + 17) / 128)


def compressions_per_row(row_bytes: int) -> int:
    """SHA-512 compressions a row (COMMIT-BYTES section 1): the row's digest (its data blocks and padding), the salt
    (3), c (2), h (2) and the frame leaf (2): 42 at 4 KiB, 138 at 16 KiB."""
    return math.ceil((row_bytes + 17) / 128) + 9


def tree_compressions(nbytes: float, row_bytes: int) -> dict:
    """One binary frame-v3-sha512 tree over nbytes cut in rows: each row at its compressions, each pad leaf at 1 and
    each node at 2 (b921: 3,327,254 rows, 867,050 pads, 4,194,303 nodes, 149.0 M compressions)."""
    rows = max(1, math.ceil(nbytes / row_bytes))
    width = 1 << (rows - 1).bit_length()
    return {"rows": rows, "pads": width - rows, "nodes": width - 1,
            "compressions": rows * compressions_per_row(row_bytes) + (width - rows) + 2 * (width - 1)}


# ---------------------------------------------------------------------------------------------------------------------
# The design dict
# ---------------------------------------------------------------------------------------------------------------------

SCHEMA = {
    "point": {
        "T": (8192, "tokens a step the window circuit holds (the throughput point 8,192; the demo 512)"),
        "real_tokens": (None, "tokens the forward computes (None: T); the demo 64"),
        "seqs": (8, "sequences a step (B4's prefill mix: 8 x 1,024 at T 8,192; the demo 4 x 16)"),
        "steps_per_window": (1, "engine steps a window (b921, a1fc, 5430 seal every step: 1); the prove term scales "
                                "as 1 / steps_per_window, since a window's draws are ~Lambda / eps whatever its size"),
        "async": (False, "LAG 1: the async stack's host gap (a new Program digest, SYNC-PATH a)"),
        "window_class": ("8,1024,1", "the window class of the point's load (slots, prompt ids, outputs): the "
                                     "throughput point's load is 8 prompts of 1,024 ids, max_tokens 1 (b57f, b921, "
                                     "a1fc); the demo's is 4,32,4. Reported, not priced"),
    },
    "forward": {
        "kernels": ("stock", "'stock' or 'commit_taps' (the Commit's norm and rope taps, b921's forward)"),
        "gemm_eff": (None, "int8 GEMM MACs a clock over 4,096 (None: b57f's 0.861)"),
        "kfa3": (False, "attention at 0.45 of bf16 a clock (not byte-equal; Daniel's)"),
        "graphs": (False, "CUDA graphs: launch gaps in the forward to 0"),
    },
    "commit": {
        "on": (True, "the Commit (rows-sha512) on"),
        "values": ("every_op", "every_op | dedup | fused_mlp | layer | layer_kv | partition (from the units' outputs)"),
        "bytes_per_token": (None, "overrides values"),
        "bits_per_word": (16, "bits a committed word for the bf16 values (int8 and f32 entries unchanged)"),
        "row_bytes": (4096, "hm96 row size of the Commit's tree (4 KiB today; 16 KiB: lever 11 of P1a)"),
        "hash_eff": (None, "share of HASH-38's bar the hash kernels reach (None: b921's 0.497; KCOMMIT: 1.0)"),
        "corun": (False, "hash at the no-cost co-run share beside the GEMMs and at the bar in host gaps"),
        "seal": ("tree", "tree (per-window tree sealed, b921) | none (5430) | roots (lever 1: seal roots, binding map "
                         "and records; an estimate: the tree's wait dropped, b921's other seal host time kept, since "
                         "nobody has split the seal's 1.31 s)"),
        "host": ("today", "today (69 event syncs, host copies, launch gaps of b921) | fixed (at stock's)"),
        "gather": (True, "the gather copy before hashing (False: hash rows in place)"),
        "padded": (True, "commit all T rows (True) or only the real tokens' (False)"),
    },
    "pouw": {
        "scheme": ("dither", "off | j10r (as served: plain s8 + z/y records, noise only in drawn units; values-only, "
                             "earns 0 under guarantee 2) | jr2 (U fresh along K) | dither (noised operands, T_s every "
                             "L: Dither's no-fold receipt)"),
        "K": (2048, "inner width of the credited linears (H1 credits the K = 2,048 linears)"),
        "L": (1024, "dither: segment, a 16-bit word a cell every L (L >= K: O alone, un-noised, credit 0)"),
        "L_prime": (None, "noise period (None: dither 128, PROTOCOL.md's static setting at L 1,024, t 8; J10-R 32)"),
        "noise": ("periodic", "periodic | fresh (along K) | permuted (U's columns re-permuted each tile)"),
        "record_bytes_per_cell": (6, "J10-R's z/y records, 6 B a cell"),
        "trees": (2, "trees H1's records are hashed into (2 today: z table and zy@c; 1: Daniel's)"),
        "h1_host": ("main", "pre (a1fc) | main (#5975, #6016: 5430) | main+6192 (65ca) | stack (every H1 host lever) | "
                            "floor (H1-COST's floor)"),
        "kdhash": (False, "#6182: H1's tables hashed in one pass"),
        "hash_at_bar": (False, "H1's hashing at HASH-38's bar"),
        "coin_off_loop": (False, "the round coin's wait off the engine thread"),
        "noise_fused": (True, "dither: noise added in the GEMM's operand load"),
        "denoise_g_frac": (None, "dither: de-noising's cost as a share of G (None: from noise and L')"),
        "value_row_bytes": (16384, "dither: the value root's hm96 row (16 KiB pinned in PROTOCOL.md)"),
        "unit_cells": (1, "PoUW proof block: cells (dither: 1 cell x 1 segment; J10-R tile: 32 x 32 = 1,024)"),
    },
    "sampling": {
        "delta": (2.0 ** -40, "failure probability a window"),
        "eps": (0.01, "share of credited work allowed wrong"),
        "pouw_sampled": (True, "PoUW's units drawn at (eps, delta) a window too (guarantee 2)"),
    },
    "partition": {
        "units": (None, "list of unit classes {name, per_token, work, extra_ands, open_rows, out_bits}; None: the "
                        "preset per op of every_op (one s8 coordinate a unit, an element a unit elsewhere)"),
    },
    "prover": {
        "o_per_and": (None, "GPU-s a drawn block's AND (None: costfit)"),
        "staged_per_and": (None, "GPU-s a staged block's AND"),
        "stmt_fixed": (None, "GPU-s a touched statement"),
        "stmt_per_and": (None, "GPU-s a statement per AND of its block"),
        "speed": (None, "prover speed on the H200 over sm_120's (None: 1.0, estimate)"),
        "m_max": (35, "C-Flock's m = k_log + nbl cap"),
        "shared": (True, "statements hold only the drawn blocks (IntegrityPeriod's hidden window)"),
    },
    "lowering": {
        "gates_per_and": (None, "gates a proof block's AND (None: 3.5, the H1 tile)"),
        "s_per_gate": (None, "CPU-s a gate lowered (None: TOP-138's target 0.9 us)"),
        "bytes_per_gate": (None, "peak bytes a gate (None: TOP-138's target 17.2; today's 600 is reported beside)"),
        "per": ("template", "template (lowered once a shape, cached) | instance (every drawn block)"),
        "ram_gib": (16.0, "RAM a lowering may use (TOP-138's target 16 GiB)"),
    },
    "verify": {
        "stmt_fixed": (None, "CPU-s a statement (None: costfit's 12.4)"),
        "stmt_per_and": (None, "CPU-s a statement per AND of its block"),
    },
    "price": {
        "cpu_s_in_gpu_s": (None, "GPU-s a CPU-second (None: 1/55)"),
        "cpu_in_J": (True, "count lowering and staging CPU at the price ratio in J's proving term"),
        "baseline": ("same", "t_stock: same (the stock forward at the design's point and scheduler: b57f's sync "
                             "stack, or 2cd6's if point.async, so the serve term is verification's overhead alone) | "
                             "fastest (the async stock, 2cd6's stack: the fastest unverified arm measured)"),
    },
}

PRESETS = {
    "throughput": {},
    "demo": {"point": {"T": 512, "real_tokens": 64, "seqs": 4, "window_class": "4,32,4"}},
}


def design(preset: str = "throughput", **sections) -> dict:
    """A full design dict: SCHEMA's defaults, then the preset, then each section's overrides
    (`design("demo", commit={"seal": "roots"})`)."""
    d = {s: {k: v[0] for k, v in keys.items()} for s, keys in SCHEMA.items()}
    for src in (PRESETS[preset], sections):
        for s, over in src.items():
            if s not in d:
                raise KeyError(f"unknown section {s!r}; sections: {sorted(d)}")
            for k, v in over.items():
                if k not in d[s]:
                    raise KeyError(f"unknown key {s}.{k}; keys: {sorted(d[s])}")
                d[s][k] = v
    return d


def normalize(d: dict) -> dict:
    """A copy of d with every missing key at its default; refuses unknown sections and keys."""
    out = design()
    for s, over in d.items():
        if s not in out:
            raise KeyError(f"unknown section {s!r}; sections: {sorted(out)}")
        for k, v in over.items():
            if k not in out[s]:
                raise KeyError(f"unknown key {s}.{k}; keys: {sorted(out[s])}")
            out[s][k] = copy.deepcopy(v)
    return out


def _get(d, sec, key, default_input):
    v = d[sec][key]
    return I(default_input) if v is None else v


# ---------------------------------------------------------------------------------------------------------------------
# Serve time a step
# ---------------------------------------------------------------------------------------------------------------------

def theta(mhz: float) -> float:
    """int8 MACs a ms at mhz."""
    return I("int8_macs_per_sm_clock") * I("sms") * mhz * 1e3


def stock_parts(d: dict) -> dict:
    """The stock forward's parts, ms a step: B4's sync stack with the Commit off (b57f at T 8,192), each scaled to the
    design's tokens with a launch floor at small T."""
    p = d["point"]
    tok = p["real_tokens"] or p["T"]
    seqs = p["seqs"]
    seq_len = tok / seqs
    eff = d["forward"]["gemm_eff"] or I("gemm_eff_per_clock")
    lu = I("launch_us") / 1e3
    gemm = max(INT8_MACS_PER_TOKEN * tok / (theta(I("gemm_mhz")) * eff),
               WEIGHT_BYTES / (I("hbm_GBs") * 1e6), 64 * lu)
    attn_macs = LAYERS * QH * HD * seqs * seq_len ** 2
    attn_eff = 0.45 if d["forward"]["kfa3"] else I("attn_eff_per_clock")
    attn = max(2 * attn_macs / (theta(I("attn_mhz")) * attn_eff), 32 * lu)
    ew = max(I("ew_us_per_token") * tok / 1e3, 112 * lu) + max(I("head_ms_per_seq") * seqs, lu)
    gap = I("host_gap_ms0") + I("host_gap_us_per_token") * tok / 1e3
    if p["async"]:
        gap *= I("async_gap_ratio")
    launch = 0.0 if d["forward"]["graphs"] else I("launch_gap_ms0") * max(tok, 512) / 8192
    return {"int8 GEMM": gemm, "attention": attn, "elementwise and head": ew, "host gap between forwards": gap,
            "launch gaps in the forward": launch}


def commit_bytes_per_token(d: dict, units: list) -> float:
    c = d["commit"]
    if not c["on"]:
        return 0.0
    if c["bytes_per_token"] is not None:
        return c["bytes_per_token"]
    if c["values"] == "partition":
        return sum(u["per_token"] * u["out_bits"] for u in units) / 8
    return stream_bytes_per_token(c["values"], c["bits_per_word"])


def hash_ms(compressions: float, eff: float, mhz: float) -> float:
    return compressions / (I("sha_bar_per_sm_clock") * I("sms") * mhz * 1e3 * eff)


def commit_parts(d: dict, stock: dict, units: list) -> tuple[dict, dict]:
    """The Commit's parts, ms a step, and what they rest on: its tree's compressions at the hash kernels' share of the
    bar, less what overlaps (today: 22% of the kernel time under tensor work; co-run: 0.167 of the bar under the
    GEMMs and the whole bar in host gaps), the GEMM's co-run loss, the taps, the gather, the host copies, the event
    syncs and the seal, each scaled from b921 by bytes or held per step."""
    c, p = d["commit"], d["point"]
    bpt = commit_bytes_per_token(d, units)
    if bpt <= 0:
        return {}, {"bytes_per_token": 0.0}
    S = p["steps_per_window"]
    tok_c = p["T"] if c["padded"] else (p["real_tokens"] or p["T"])
    step_bytes = bpt * tok_c
    tree = tree_compressions(step_bytes * S, c["row_bytes"])
    comp = tree["compressions"] / S
    eff = c["hash_eff"] or I("hash_eff_today")
    mhz = I("hash_mhz")
    kernel = max(hash_ms(comp, eff, mhz), I("commit_launches") * 0.8 * I("launch_us") / 1e3)
    b921_bytes = I("commit_B_per_token") * 8192
    scale = step_bytes / b921_bytes
    if c["corun"]:
        tensor_ms = stock["int8 GEMM"] + stock["attention"]
        idle_ms = stock["host gap between forwards"] + stock["launch gaps in the forward"]
        free = I("corun_share") * tensor_ms * I("gemm_mhz") / mhz + idle_ms
        exposed = max(0.0, kernel - free)
        penalty = 0.0
    else:
        exposed = kernel * (1 - I("hash_overlap_today"))
        overlapped_cycles = kernel * I("hash_overlap_today") * mhz
        penalty = I("gemm_corun_penalty_per_hash_cycle") * overlapped_cycles / I("gemm_mhz")
    parts = {"Commit hashing (exposed)": exposed, "GEMM's loss to co-running hash": penalty}
    if d["forward"]["kernels"] == "commit_taps":
        parts["the Commit's taps"] = I("taps_ms")
    if c["gather"]:
        parts["Commit gather"] = I("gather_ms") * scale
    if c["host"] == "today":
        parts["Commit host copies"] = I("commit_copies_ms") * scale
        parts["Commit event syncs and launch gaps"] = I("commit_launch_gap_ms")
        if c["seal"] in ("tree", "roots"):
            parts["seal: rest of the host gap"] = I("seal_rest_ms")
    if c["seal"] == "tree":
        seal_bytes = (tree["rows"] + tree["nodes"]) * 64
        parts["seal wait"] = seal_bytes / (I("seal_GBs") * 1e6) / S
    info = {"bytes_per_token": bpt, "step_bytes": step_bytes, "tree": tree, "compressions_per_step": comp,
            "hash_kernel_ms": kernel, "hash_eff": eff,
            "corun_free_GBs_at_row": I("corun_share") * I("sha_bar_per_sm_clock") * I("sms") * 1830e6
            * c["row_bytes"] / compressions_per_row(c["row_bytes"]) / 1e9}
    return parts, info


def l_prime(w: dict) -> int:
    return w["L_prime"] or (128 if w["scheme"] == "dither" else 32)


def pouw_forced_frac(d: dict) -> float:
    """The least work a credited call forces, as a share of its plain s8 GEMM G, under SAK on noised operands, at the
    best attack found. Un-noised calls earn 0 (the brief): J10-R as served adds no noise at serve time, and at L >= K a
    dither cell commits O alone. Dither, periodic or permuted (v3.2: V, U periodic in L' with pi fresh each tile):
    min(1, 1/8 + 3/t) (the floor note); fresh noise along K, or JR2: G (estimate: no shared period to separate)."""
    w = d["pouw"]
    if w["scheme"] in ("off", "j10r"):
        return 0.0
    if w["scheme"] == "jr2":
        return 1.0
    L, Lp, K = w["L"], l_prime(w), w["K"]
    if L >= K:
        return 0.0
    if w["noise"] == "fresh":
        return 1.0
    t = max(1, L // Lp)
    return min(1.0, 1.0 / I("b1_rate_vs_s8") + 3.0 / t)


def pouw_credit(d: dict) -> dict:
    """Guarantee 2's credit as a share of the four linears' int8 MACs: a dither call with L < K credits (K - L) / K of
    its MACs (the floor note; S15 of note:compute-research/20261028T1900Z-report-gd-smell), and the work that credit
    forces is the floor's share of it, at the best attack found (an upper bound on the floor, not a bound)."""
    w = d["pouw"]
    total = sum(K * N for _, K, N in LINEARS)
    if w["scheme"] == "dither":
        cred = sum(K * N * (K - w["L"]) / K for _, K, N in LINEARS if w["L"] < K) / total
    elif w["scheme"] == "jr2":
        cred = sum(K * N for _, K, N in LINEARS if K == w["K"]) / total
    else:
        cred = 0.0
    return {"credited_share_of_linear_macs": cred,
            "forced_share_at_best_attack": cred * pouw_forced_frac(d),
            "earns_credit": cred > 0 and pouw_forced_frac(d) > 0,
            "basis": "best attack found (an upper bound on the floor)" if w["scheme"] == "dither" else
                     ("estimate (JR2: fresh U, no floor note)" if w["scheme"] == "jr2" else
                      "un-noised at serve time: 0 under guarantee 2")}


def _h1_main_ms_8k(h1_host: str) -> float:
    """H1's own time in a measured post-fix step at T 8,192: the step (5430, or 65ca with #6192) less the modelled
    stock forward and unsealed Commit of the same configuration, and less the round coin, which `pouw_parts` adds."""
    s = I("h1_post_step_s_8k" if h1_host == "main" else "h1_post6192_step_s_8k") * 1e3
    d = design(forward={"kernels": "commit_taps"}, commit={"seal": "none"}, pouw={"scheme": "off"})
    stock = stock_parts(d)
    rest = sum(stock.values()) + sum(commit_parts(d, stock, preset_units(d))[0].values())
    return s - rest - I("h1_coin_ms")


def _dither_calls(tok: float, L: int):
    """Per credited call class of one token: (cells, segments). Dither credits all four linears."""
    for _, K, N in LINEARS:
        yield LAYERS * N * tok, math.ceil(K / L)


def pouw_parts(d: dict, stock: dict) -> tuple[dict, dict]:
    """PoUW's serve-time parts, ms a step. J10-R: H1's host path after the chosen fixes and its device hashing, the
    larger of the two (they overlap), and the round coin. JR2: J10-R's path plus the noised product and its
    de-noising at serve time. Dither: noise generation and application, de-noising and the value roots' hashing."""
    w, p = d["pouw"], d["point"]
    if w["scheme"] == "off":
        return {}, {}
    tok = p["real_tokens"] or p["T"]
    G_ms = INT8_MACS_PER_TOKEN * tok / (theta(I("gemm_mhz")) * I("gemm_eff_per_clock"))
    k_share = sum(k * n for _, k, n in LINEARS if k == w["K"]) / sum(k * n for _, k, n in LINEARS)
    parts, info = {}, {"forced_frac_of_G": pouw_forced_frac(d), "credit": pouw_credit(d)}
    if w["scheme"] in ("j10r", "jr2"):
        cells = H1_CELLS_PER_TOKEN * tok
        cells8k = H1_CELLS_PER_TOKEN * 8192
        rec_gb = cells * w["record_bytes_per_cell"] / 1e9
        dev = I("h1_device_ms_8k") * cells / cells8k * w["trees"] / 2
        if w["kdhash"]:
            dev *= I("kdhash_ratio")
        if w["hash_at_bar"]:
            bar_GBs = I("sha_bar_per_sm_clock") * I("sms") * 1.83 * 91.5
            dev = rec_gb * w["trees"] / bar_GBs * 1e3 + 3.5 * cells / cells8k
        coin = 0.0 if w["coin_off_loop"] else I("h1_coin_ms")
        h = w["h1_host"]
        if h == "pre":
            host = (I("h1_value_check_ns_per_cell") * cells / 1e6
                    + (cells * 6 / 1e9) / I("h1_copy_GBs_records") * 1e3
                    + (I("h1_tables_s_8k") + I("h1_copy_y_s_8k") + I("h1_copy_h2d_s_8k")
                       + I("h1_copy_other_s_8k")) * 1e3 * cells / cells8k)
            coin = 0.0                                 # inside a1fc's host path
        elif h in ("main", "main+6192"):
            # a + b cells, through the T 8,192 step (5430 or 65ca) and #6016's T 512 check, scaled alike
            hi = _h1_main_ms_8k(h)
            lo = I("h1_post_s_512") * 1e3 * hi / _h1_main_ms_8k("main")
            c512 = H1_CELLS_PER_TOKEN * 512
            b = (hi - lo) / (cells8k - c512)
            host = lo + b * (cells - c512)
            info["h1_host_fit"] = {"ms_at_512": lo, "ns_per_cell": b * 1e6, "ms_at_8192": hi}
        elif h in ("stack", "floor"):
            host = I("h1_rest_floor_ms_8k") * cells / cells8k
            if h == "floor":
                dev = min(dev, rec_gb * w["trees"] / I("h1_floor_hash_GBs") * 1e3)
        else:
            raise ValueError(f"pouw.h1_host {h!r}")
        parts["H1 host path"] = host
        parts["H1 device, beyond the host path"] = max(0.0, dev - host)
        parts["H1 round coin (engine thread)"] = coin
        info.update({"cells": cells, "records_GB": rec_gb, "h1_device_ms": dev,
                     "j10r_floor_frac": 1 / 8 + l_prime(w) / w["K"]})
        if w["scheme"] == "jr2":
            dg = w["denoise_g_frac"] if w["denoise_g_frac"] is not None else 1.0
            parts["JR2: de-noising the K-credited calls (fresh U: estimate 1 G)"] = dg * G_ms * k_share
        return parts, info
    # dither
    L, Lp = w["L"], l_prime(w)
    act_entries = LAYERS * tok * (3 * HIDDEN + INTER)          # the four linears' A operands, one int8 entry each
    pass_ms = 2 * (act_entries + WEIGHT_BYTES) / (I("hbm_GBs") * 1e6)
    parts["dither: noise and dither added to A and B"] = pass_ms * (I("noise_fused_keep") if w["noise_fused"] else 1)
    dither_bytes = (act_entries + WEIGHT_BYTES) / 8            # d_A and d_B: one bit an operand entry
    parts["dither: d_A, d_B expanded from the coins (SHA-512 PRF)"] = \
        hash_ms(dither_bytes / 64, 1.0, I("hash_mhz")) * (1 - I("corun_share"))
    if w["denoise_g_frac"] is not None:
        dg = w["denoise_g_frac"]
    elif w["noise"] == "fresh":
        dg = 1.0
    else:
        dg = min(1.0, 2 * Lp / w["K"] + I("wrap_g_frac"))
    parts["dither: de-noising O"] = dg * G_ms
    calls = list(_dither_calls(tok, L))
    cells = sum(c for c, _ in calls)
    words = sum(c * (2 * (s - 1) + 4) for c, s in calls)
    vt = tree_compressions(words, w["value_row_bytes"])
    vh = hash_ms(vt["compressions"], d["commit"]["hash_eff"] or I("hash_eff_today"), I("hash_mhz"))
    parts["dither: value roots hashed"] = vh * (1 - I("hash_overlap_today"))
    info.update({"cells": cells, "t": max(1, L // Lp), "value_bytes": words, "value_tree": vt, "denoise_g_frac": dg})
    return parts, info


# ---------------------------------------------------------------------------------------------------------------------
# Sampling, proving, lowering, verification a window
# ---------------------------------------------------------------------------------------------------------------------

def s8_coord_work(K: int) -> float:
    return K // 32 * I("s8_step_ands") + I("s8_epilogue_ands")


def preset_units(d: dict) -> list:
    """The per-op partition (every committed value a unit's output, at most 16 bits, one unit an output word):
    one s8 coordinate a unit (opening its two K-byte operand rows and three one-block rows, costfit's
    ScaledMmS8Coordinate_v1), one element a unit for the elementwise ops (opening one input row), one (head, query)
    a unit for attention. Rows opened are the Commit's rows, so they grow with commit.row_bytes."""
    R = d["commit"]["row_bytes"]
    p = d["point"]
    seq_len = (p["real_tokens"] or p["T"]) / p["seqs"]
    units = []
    for name, K, N in LINEARS:
        units.append({"name": f"{name} coordinate (K {K})", "per_token": LAYERS * N, "work": s8_coord_work(K),
                      "extra_ands": 0.0, "open_rows": [min(K, R), min(K, R), 1, 1, 1], "out_bits": 16})
    attn_work = I("attn_ands_per_mac") * 2 * HD * seq_len / 2
    units.append({"name": "attention (head, query)", "per_token": LAYERS * QH, "work": attn_work, "extra_ands": 0.0,
                  "open_rows": [R, R, R], "out_bits": 16 * HD})
    ew = (LAYERS * (2 * HIDDEN + 2 * HIDDEN + INTER + (QH + KVH) * HD + 3 * HIDDEN + INTER))
    units.append({"name": "elementwise element", "per_token": ew, "work": I("ew_ands_per_elem"), "extra_ands": 0.0,
                  "open_rows": [R], "out_bits": 16})
    return units


def pouw_units(d: dict) -> list:
    """PoUW's proof blocks. J10-R/JR2: a 32 x 32 tile at K (the brief's ~1e9 gates at K 2,048). Dither: one segment of
    one cell (PROTOCOL.md), opening the rows of A's and B's segment and of T_s and T_{s-1} (or O)."""
    w = d["pouw"]
    if w["scheme"] == "off" or not d["sampling"]["pouw_sampled"]:
        return []
    cells = H1_CELLS_PER_TOKEN
    if w["scheme"] in ("j10r", "jr2"):
        tile = w["unit_cells"] if w["unit_cells"] > 1 else 1024
        gates = I("h1_tile_gates") * tile / 1024 * w["K"] / 2048
        work = tile * s8_coord_work(w["K"])
        return [{"name": f"H1 tile ({tile} cells, K {w['K']})", "per_token": cells / tile,
                 "work": work, "extra_ands": max(0.0, gates / I("gates_per_and") - work),
                 "open_rows": [], "out_bits": 16, "gates": gates}]
    L, Rv = w["L"], w["value_row_bytes"]
    noise_ands = 2 * L * I("sha512_ands") / 64                 # V and U pi entries expanded from the coins (estimate)
    units = []
    for name, K, N in LINEARS:
        if L >= K:
            continue                                           # O alone: un-noised, credits 0, not drawn
        S_seg = math.ceil(K / L)
        units.append({"name": f"dither block, {name} (1 cell, L {L} of K {K})", "per_token": LAYERS * N * S_seg,
                      "work": L / 32 * I("s8_step_ands"), "extra_ands": noise_ands,
                      "open_rows": [min(L, Rv), min(L, Rv), Rv, Rv], "out_bits": 16})
    return units


def proving(d: dict, units: list, label: str) -> dict:
    """A window's draws and proving cost for one sampled population. Achievability's law q_u = min(1, Lambda w_u / C)
    with C = eps F, so draws ~ Lambda F / C = Lambda / eps; each drawn block costs o (ANDs + row openings) and each
    touched statement stmt_fixed + stmt_per_and ANDs, with ceil(draws / cap) statements a class (cap = 2^(m_max -
    k_log), k_log = ceil(log2 ANDs) + 1). Lowering is per class (template) or per draw (instance)."""
    p, s = d["point"], d["sampling"]
    S = p["steps_per_window"]
    T = p["real_tokens"] or p["T"]                             # units cover what the forward computes
    lam = math.log(1 / s["delta"])
    pv = d["prover"]
    o = (pv["o_per_and"] if pv["o_per_and"] is not None else I("o_per_and"))
    so = (pv["staged_per_and"] if pv["staged_per_and"] is not None else I("staged_per_and"))
    f0 = (pv["stmt_fixed"] if pv["stmt_fixed"] is not None else I("stmt_fixed"))
    f1 = (pv["stmt_per_and"] if pv["stmt_per_and"] is not None else I("stmt_per_and"))
    speed = pv["speed"] if pv["speed"] is not None else I("h200_vs_sm120")
    lo = d["lowering"]
    gpa = lo["gates_per_and"] if lo["gates_per_and"] is not None else I("gates_per_and")
    spg = lo["s_per_gate"] if lo["s_per_gate"] is not None else I("lower_s_per_gate")
    bpg = lo["bytes_per_gate"] if lo["bytes_per_gate"] is not None else I("lower_bytes_per_gate")
    vf = d["verify"]["stmt_fixed"] if d["verify"]["stmt_fixed"] is not None else I("verify_fixed")
    va = d["verify"]["stmt_per_and"] if d["verify"]["stmt_per_and"] is not None else I("verify_per_and")
    F = sum(u["per_token"] * T * S * u["work"] for u in units)
    C = s["eps"] * F
    out = {"population": label, "F_ands": F, "C_ands": C, "Lambda": lam, "classes": [], "draws": 0.0, "gpu_s": 0.0,
           "statements": 0, "lower_cpu_s": 0.0, "stage_cpu_s": 0.0, "verify_cpu_s": 0.0, "lower_peak_GiB": 0.0,
           "infeasible": []}
    if F <= 0:
        return out
    for u in units:
        n = u["per_token"] * T * S
        q = min(1.0, lam * u["work"] / C)
        draws = n * q
        ands = u["work"] + u["extra_ands"] + sum(row_ands(r) for r in u["open_rows"])
        k_log = math.ceil(math.log2(max(ands, 2))) + 1
        cap = 1 << max(0, pv["m_max"] - k_log)
        if k_log > pv["m_max"]:
            out["infeasible"].append(f"{u['name']}: {ands:.3g} ANDs needs k_log {k_log} > m_max {pv['m_max']}")
        stmts = math.ceil(draws / cap) if draws > 0 else 0
        staged = draws if pv["shared"] else min(n, stmts * cap)
        gpu = (draws * o * ands + staged * so * ands + stmts * (f0 + f1 * ands)) / speed
        gates = u.get("gates", ands * gpa)
        lowerings = (1 if draws > 0 else 0) if lo["per"] == "template" else draws
        lower = lowerings * gates * spg
        peak = gates * bpg / 2 ** 30
        if peak > lo["ram_gib"]:
            out["infeasible"].append(f"{u['name']}: lowering needs {peak:.3g} GiB > {lo['ram_gib']} GiB")
        stage = stmts * (I("stage_fixed") + I("stage_per_and") * ands) + staged * I("stage_per_and_staged") * ands
        verify = stmts * (vf + va * ands)
        out["classes"].append({"name": u["name"], "units": n, "work": u["work"], "ands_with_openings": ands,
                               "openings_share": 1 - (u["work"] + u["extra_ands"]) / ands, "q": q, "draws": draws,
                               "cap": cap, "statements": stmts, "gpu_s": gpu, "lower_cpu_s": lower,
                               "lower_peak_GiB": peak,
                               "lower_peak_GiB_today": gates * I("lower_bytes_per_gate_today") / 2 ** 30,
                               "stage_cpu_s": stage, "verify_cpu_s": verify})
        out["draws"] += draws
        out["gpu_s"] += gpu
        out["statements"] += stmts
        out["lower_cpu_s"] += lower
        out["stage_cpu_s"] += stage
        out["verify_cpu_s"] += verify
        out["lower_peak_GiB"] = max(out["lower_peak_GiB"], peak)
    out["lower_peak_GiB_today"] = max((c["lower_peak_GiB_today"] for c in out["classes"]), default=0.0)
    out["draws_bound"] = lam / s["eps"]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# J
# ---------------------------------------------------------------------------------------------------------------------

def J(dsg: dict | None = None) -> dict:
    """The cost of a design: serve-overhead parts (ms a step), proving (GPU-s and CPU-s a window), verification
    (CPU-s a window), the draws, and J = (t_remote / t_stock - 1) + proving GPU-s / stock GPU-s a window."""
    d = normalize(dsg or {})
    stock = stock_parts(d)
    if d["price"]["baseline"] == "fastest":
        base = normalize({**d, "point": {**d["point"], "async": True}})
        t_stock = sum(stock_parts(base).values())
        anchor = f"async stock at the design's point (2cd6's stack, {C2CD6})"
    elif d["price"]["baseline"] == "same":
        t_stock = sum(stock.values())
        anchor = (f"async stock (2cd6's stack, {C2CD6})" if d["point"]["async"] else
                  f"sync stock (b57f's stack, {B57F})") + " at the design's point"
    else:
        raise ValueError(f"price.baseline {d['price']['baseline']!r}")
    units = d["partition"]["units"] or preset_units(d)
    remote = dict(stock)
    if d["forward"]["kernels"] == "commit_taps" and not d["commit"]["on"]:
        remote["the Commit's taps"] = I("taps_ms")
    cparts, cinfo = commit_parts(d, stock, units)
    wparts, winfo = pouw_parts(d, stock)
    remote.update(cparts)
    remote.update(wparts)
    t_remote = sum(remote.values())
    S = d["point"]["steps_per_window"]
    stock_gpu_s = t_stock * S / 1e3
    values = proving(d, units, "values (sampled proofs)")
    pouw = proving(d, pouw_units(d), "PoUW")
    price = d["price"]["cpu_s_in_gpu_s"] if d["price"]["cpu_s_in_gpu_s"] is not None else I("cpu_s_in_gpu_s")
    gpu_s = values["gpu_s"] + pouw["gpu_s"]
    cpu_s = values["lower_cpu_s"] + values["stage_cpu_s"] + pouw["lower_cpu_s"] + pouw["stage_cpu_s"]
    proving_gpu_s = gpu_s + (cpu_s * price if d["price"]["cpu_in_J"] else 0.0)
    verify_cpu_s = values["verify_cpu_s"] + pouw["verify_cpu_s"]
    serve_term = t_remote / t_stock - 1
    prove_term = proving_gpu_s / stock_gpu_s
    return {
        "J": serve_term + prove_term,
        "serve_term": serve_term,
        "prove_term": prove_term,
        "t_stock_ms": t_stock,
        "t_stock_anchor": anchor,
        "window_class": d["point"]["window_class"],
        "t_remote_ms": t_remote,
        "stock_parts_ms": stock,
        "serve_overhead_ms": {k: v for k, v in remote.items() if k not in stock or remote[k] != stock[k]},
        "commit": cinfo,
        "pouw": winfo,
        "stock_gpu_s_per_window": stock_gpu_s,
        "proving": {"gpu_s": gpu_s, "cpu_s": cpu_s, "cpu_in_gpu_s": cpu_s * price, "total_gpu_s": proving_gpu_s,
                    "values": values, "pouw": pouw},
        "verification": {"cpu_s": verify_cpu_s, "gpu_s_equiv": verify_cpu_s * price,
                         "per_stock_gpu_s": verify_cpu_s * price / stock_gpu_s,
                         "statements": values["statements"] + pouw["statements"]},
        "draws": {"values": values["draws"], "pouw": pouw["draws"], "bound_each": values.get("draws_bound")},
        "infeasible": values["infeasible"] + pouw["infeasible"],
        "u_1830": ideal_ms_1830(d) / t_remote,
        "u_stock_1830": ideal_ms_1830(d) / t_stock,
    }


def ideal_ms_1830(d: dict) -> float:
    """BENCH-34's ruler: declared int8 MACs plus bf16 attention and the LM head at half rate, at 1,830 MHz."""
    p = d["point"]
    tok = p["real_tokens"] or p["T"]
    attn = LAYERS * QH * HD * p["seqs"] * (tok / p["seqs"]) ** 2
    head = p["seqs"] * HIDDEN * VOCAB
    return (INT8_MACS_PER_TOKEN * tok + 2 * (attn + head)) / theta(1830.0)


# ---------------------------------------------------------------------------------------------------------------------
# Calibration against the measured steps
# ---------------------------------------------------------------------------------------------------------------------

CALIBRATION = {
    "b57f": ("stock: sync stack, Commit off, T 8,192", B57F, 26.3198,
             {"commit": {"on": False}, "pouw": {"scheme": "off"}}),
    "b921": ("Commit sealed (rows-sha512, --retain device), H1 off", B921, 1553.254,
             {"forward": {"kernels": "commit_taps"}, "pouw": {"scheme": "off"}}),
    "a1fc": ("Commit sealed, H1 under J10-R at K 2,048, before #5975 and #6016", A1FC, 58277.807,
             {"forward": {"kernels": "commit_taps"}, "pouw": {"scheme": "j10r", "h1_host": "pre"}}),
    "5430": ("Commit unsealed, H1 after #5975 and #6016 (main's H1 path; an identity: this step defines main's H1 "
             "host term)", H1POST, 16504.15,
             {"forward": {"kernels": "commit_taps"}, "commit": {"seal": "none"},
              "pouw": {"scheme": "j10r", "h1_host": "main"}}),
}


# Steps no input was fitted on (out of sample): the stock forward at T 16k and 32k, where the GEMM's efficiency and
# clock are b57f's; H1 before the fixes at T 512, whose per-cell rates are T 8,192's.
VALIDATION = {
    "966f": ("stock at T 16,384 (16 x 1,024)", T966F, 50.668,
             {"point": {"T": 16384, "seqs": 16}, "commit": {"on": False}, "pouw": {"scheme": "off"}}),
    "f746": ("stock at T 32,768 (32 x 1,024)", TF746, 97.973,
             {"point": {"T": 32768, "seqs": 32}, "commit": {"on": False}, "pouw": {"scheme": "off"}}),
    "2cd6": ("async stack, Commit off, T 8,192", C2CD6, 21.6008,
             {"point": {"async": True}, "commit": {"on": False}, "pouw": {"scheme": "off"}}),
    "h1-512-pre": ("H1 before #5975 at T 512: one layer's four linears 0.218 s x 16 (#5975's check "
                   "r20261028-031545-d474, a harness: H1's host path only)", "r20261028-031545-d474", 16 * 218.0,
                   {"point": {"T": 512, "real_tokens": 512, "seqs": 1}, "commit": {"on": False},
                    "pouw": {"scheme": "j10r", "h1_host": "pre"}, "_h1_only": True}),
    "h1-512-async": ("H1 at T 512 decode step before the first fix, measured 3.97 s (note:vllm-h1-async/"
                     "20261027T2205Z-report-h1-step-ab; includes the forward and the coin)", "vllm-h1-async", 3970.0,
                     {"point": {"T": 512, "real_tokens": 512, "seqs": 1}, "commit": {"on": False},
                      "pouw": {"scheme": "j10r", "h1_host": "pre"}}),
}


def validate() -> dict:
    out = {}
    for key, (what, run, measured, over) in VALIDATION.items():
        over = dict(over)
        h1_only = over.pop("_h1_only", False)
        r = J(design(**over))
        model = sum(v for k, v in r["serve_overhead_ms"].items() if k.startswith("H1 host")) if h1_only \
            else r["t_remote_ms"]
        out[key] = {"what": what, "run": run, "measured_ms": measured, "model_ms": model, "ratio": model / measured}
    return out


def calibrate() -> dict:
    out = {}
    for key, (what, run, measured, over) in CALIBRATION.items():
        r = J(design(**over))
        out[key] = {"what": what, "run": run, "measured_ms": measured, "model_ms": r["t_remote_ms"],
                    "ratio": r["t_remote_ms"] / measured,
                    "parts_ms": {k: round(v, 3) for k, v in {**r["stock_parts_ms"], **r["serve_overhead_ms"]}.items()}}
    return out


def main(argv: list[str]) -> int:
    if argv[1:2] == ["calibrate"]:
        cal = calibrate()
        for k, v in cal.items():
            print(f"{k:5s} measured {v['measured_ms']:10.2f} ms  model {v['model_ms']:10.2f} ms  "
                  f"ratio {v['ratio']:.3f}  ({v['what']}, {v['run']})")
        print("out of sample:")
        val = validate()
        for k, v in val.items():
            print(f"{k:12s} measured {v['measured_ms']:10.2f} ms  model {v['model_ms']:10.2f} ms  "
                  f"ratio {v['ratio']:.3f}  ({v['what']})")
        if "--json" in argv:
            print(json.dumps({"calibration": cal, "validation": val}, indent=1))
        return 0
    if argv[1:2] == ["schema"]:
        print(json.dumps({s: {k: {"default": v[0], "doc": v[1]} for k, v in keys.items()} for s, keys in
                          SCHEMA.items()}, indent=1, default=str))
        return 0
    if argv[1:2] == ["inputs"]:
        print(json.dumps({k: {"value": v[0], "status": v[1], "source": v[2]} for k, v in INPUTS.items()}, indent=1))
        return 0
    preset = argv[1] if len(argv) > 1 else "throughput"
    print(json.dumps(J(design(preset)), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
