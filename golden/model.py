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

    point     T, real_tokens, seqs, steps_per_window, async, decode_steps, window_class
    forward   kernels ("stock" | "commit_taps"), gemm_eff, kfa3, graphs
    commit    on, values ("every_op" | "dedup" | "fused_mlp" | "layer" | "layer_kv" | "e1" | "e2" | "partition"),
              bytes_per_token,
              bits_per_word, row_bytes, hash_eff, corun, seal ("tree" | "none" | "roots"), host ("today" | "fixed"),
              gather, padded
    pouw      scheme ("off" | "j10r" | "jr2" | "dither"), K, L, L_prime,
              noise ("periodic" | "fresh" | "permuted" | "segment"), u_draw ("period" | "call"),
              record_bytes_per_cell, trees, h1_host ("pre" | "main" | "main+6192" | "stack" | "floor"), kdhash,
              hash_at_bar, coin_off_loop, noise_fused, denoise_g_frac, value_row_bytes, tile (R, C),
              form ("tile" | "lanes"), credit, word_bits, agg, joint
    sampling  delta, eps, pouw_sampled, law ("achievability" | "uniform" | "stratified"), retain ("device" | "host"),
              retain_GB, period
    partition units: [{name, per_token, work, extra_ands, open_rows, out_bits, blocks?}] or None for the preset of
              `commit.values`; a unit drawn as a group of proof blocks lists them, blocks: [{name, count, ands}];
              tile, kchunk, attn_keys, census (None | "b4" | "b4-buckets")
    prover    device ("h200" | "sm120"), reps (1 | 2), o_per_and, staged_per_and, stmt_fixed, stmt_per_and, speed,
              m_max, k_max, k_pad, shared, stmt ("template" | "universal")
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
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------------
# Inputs. (value, status, source). M = measured (run id), E = estimate, D = derived.
# ---------------------------------------------------------------------------------------------------------------------

B57F, B921, A1FC = "r20261028-080559-b57f", "r20261028-084012-b921", "r20261028-084022-a1fc"
C2CD6, T966F, TF746 = "r20261028-075211-2cd6", "r20261028-080609-966f", "r20261028-080612-f746"
MAIN_ASYNC = "r20261028-080545-d0a8"
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
    # GD-HASH (note:compute-research/20261028T1945Z-report-gd-hash): nothing is free beside the GEMM. h_b / h_x, the
    # GEMM's loss a byte hashed beside it over the same byte hashed alone, A-B-A r20261028-192103-b3eb at the
    # cheapest phases; beside capacity, GB a forward below the whole grid against the 64 linears' 14.5 ms at T 8,192
    "hash_eff_rows": (0.92, "M", "the row kernel's leaves at stream scale: 0.91-0.93 of HASH-38's bar a clock "
                                 "(GD-HASH, r20261028-185933-4435); h_x 2,703 / 2,123 / 1,964 MAC-times a byte at "
                                 "4 / 16 / 64 KiB rows is compressions_per_row at this share"),
    "beside_4k": (2223 / 2703, "M", "h_b / h_x at 4 KiB rows, T 8,192 (GD-HASH: 2,223 / 2,703)"),
    "beside_16k": (1465 / 2123, "M", "h_b / h_x at 16 KiB rows, T 8,192 (1,465 / 2,123)"),
    "beside_64k": (1855 / 1964, "M", "h_b / h_x at 64 KiB rows, T 8,192 (1,855 / 1,964)"),
    "beside512_4k": (1690 / 2703, "M", "h_b / h_x at 4 KiB rows, T 512, best cap (1,681-1,700)"),
    "beside512_16k": (1350 / 2123, "M", "h_b / h_x at 16 KiB rows, T 512, best caps (1,308-1,393)"),
    "beside512_64k": (1494 / 1964, "M", "h_b / h_x at 64 KiB rows, T 512 (1,494)"),
    "beside_GB_4k": (0.7, "M", "GB a forward beside the GEMM below the whole grid at 4 KiB rows, T 8,192"),
    "beside_GB_16k": (1.9, "M", "... at 16 KiB rows"),
    "beside_GB_64k": (3.1, "M", "... at 64 KiB rows"),
    "beside_gemm_ms": (14.5, "M", "GD-HASH's GEMM loop alone, a forward at T 8,192 (14.43-14.52 ms); capacity scales "
                                  "with the model's int8 GEMM ms (T 512: 77-199 MB in 1.75 ms, the same rate)"),
    "guess_p16": (2.79e-5, "E", "Dither PROTOCOL.md: the most predictable 16-bit word's free-guess rate on the H200; "
                                "x 2^(16 - bits) down to 9 bits, 2^-bits at 8 and under (GD-BITS)"),
    "mix4_ands": (214, "M", "GD-AGG's mix4 (four xorshift-multiply rounds mod 2^16) in circuit, a cell, counted "
                            "(art:10db9c0c3d71...; note:compute-research/20261028T1945Z-report-gd-agg)"),
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
    "b1_rate_vs_s8": (8.0, "M", f"{FLOOR}: the b1 product at 8x the s8 rate a clock; H200 mma.sync m16n8k256 "
                                f".and.popc 21,845 against s8 m16n8k32 2,731 MACs an SM-clock, r20261028-090323-2c56 "
                                f"(#6216, note:kernels-kfa3/20261028T0912Z-report-h200-b1-mma-rate)"),
    "permuted_limbs": (2.54, "E", "GD-NOISE finding 1 (note:compute-research/20261028T1932Z-report-gd-noise): pi re-"
                                  "drawn each tile, the merged factor's high limb (11.7% dense, 0.16% over 2:4 at "
                                  "+-80, t 3) on 2:4 sparse s8 at half price: 1 + 0.5 + 0.05 + 1 limb-products a word"),
    "jr2_noise_g": (1.32, "E", "GD-NOISE's table, JR2 per period at m 8,192 (qkv): the noised product 1.00 G, CUDA "
                               "0.18 and d_A's PRG 0.14; the lane receipts' hashing ~0 beside H1's"),
    "jr2_call_extra_g": (0.01, "E", "GD-NOISE's table, JR2 per call (#6369) over per period, without W-PRG: CUDA "
                                    "0.19 against 0.18"),
    "jr2_wprg_g_m": (3492.0, "E", "GD-NOISE finding 6: U drawn every call needs n K PRG bytes, 3,492 / m G (7.7 G at "
                                  "m 512, 982 at m 4)"),
    "noise_fused_keep": (0.1, "E", "X formed in the GEMM's operand load keeps 10% of GD-DENOISE's separate X arm"),
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
    "o_per_and_h200": (3.18e-7, "M", "GD-PROVER on the H200 (r20261028-191928-57a0, firewall on, K 1,024 blocks of "
                                     "1.94 M ANDs, warm sessions): 47.0 s at 64 drawn against 12.45 s at 8, so 34.5 s "
                                     "for 56 padded draws; 29x the sm_120 fit (sm_90's GF(2^128) multiply possibly "
                                     "emulated: GD-PROVER's unchecked cause). The statement term agrees with costfit's "
                                     "(7.5 s against 9.8 s at that block)"),
    "sm120_gpu_price": (0.5, "E", "an RTX PRO 6000 GPU-s priced in H200 GPU-s, when the prover runs on sm_120 "
                                  "(prover.device): a judgement, not a quote"),
    "rep_share": (0.27, "E", "the second rep's share of a two-rep proof's drawn-AND time: zerocheck 10.0 s and ring "
                             "switch 4.8 s of 27.6 s GPU arithmetic (GD-PROVER's H200 K 1,024 k 64 session), halved, "
                             "taking those per rep and ligerito's commitment shared"),
    "one_rep_query_cost": (0.03, "E", "GD-KNOBS row 22: one rep clears 100 bits with Q a level up, about +3% queries"),
    "soundness_bits_2reps": (195.5, "M", "C-Flock fast100x2: two reps on one root give 2^-195.5 a table "
                                         "(verifier/PROTOCOL.md section 15; GD-KNOBS row 22)"),
    "soundness_floor_bits": (100.0, "M", "the 100-bit floor one rep is raised to (one rep alone: 2^-97.8)"),
    "m_max": (35, "M", "C-Flock circuit.rs M_MAX: m = k_log + nbl <= 35 (note:proofs/20261028T1837Z-report-partopt)"),
    "sha512_ands": (81_402, "M", "C-Flock's SHA-512 circuit, rows a compression (GD-SIS reproduced the pin, "
                                 "note:compute-research/20261028T1938Z-report-gd-sis; inbox 20:00Z), 1.40x "
                                 "optimize.SHA512_ANDS' 58,120"),
    "hm96_ands": (2 * 81_402, "M", "the hiding part of an hm96-sha512 row in circuit: two compressions, as "
                                   "optimize.HM96_ANDS = 2 x SHA512_ANDS"),
    "s8_step_ands": (4_190, "M", "benchmarks/sampling_plan/inputs.py S8_STEP: ANDs of 32 s8 MACs (#2391's pin)"),
    "s8_epilogue_ands": (5_000, "M", "benchmarks/sampling_plan/inputs.py EPILOGUE"),
    "ew_ands_per_elem": (1_660, "E", "an elementwise op's ANDs an element, weighted over a layer's elements: "
                                     "benchmarks/prove_path/class_size.py MEASURED where measured (RMSNorm 4.6-5.1 K, "
                                     "SiluMul 1.38 K, RoPE 3.0 K an element), the quantizer 1,000 and the residual add "
                                     "500 (estimates); GD-PARTITION's units replace it"),
    "attn_ands_per_mac": (566, "E", "class_size.MEASURED's AttnBlock_v5{D=64,NVIS=128,BN=128}, 9.28 M ANDs, over a "
                                    "query's 128 keys x 64 x 2 MACs (taking the block as one query's key block)"),
    "attn_keys_per_block": (128, "M", "class_size.MEASURED's AttnBlock_v5 BN: keys an attention proof block"),
    "h1_tile_gates": (1.0e9, "M", "the brief: the K 2,048 32x32 H1 tile is about 1e9 gates, OOM at 600 GiB (TOP-138)"),
    # lowering (TOP-138) and prices
    "lower_bytes_per_gate": (16 * 2 ** 30 / 1.0e9, "E", "TOP-138's target: the 1e9-gate H1 tile in 16 GiB (17.2 B a "
                                                        "gate); today's figure in lower_bytes_per_and_today"),
    "lower_bytes_per_and_today": (4.912 * 2 ** 30 / 2_384_742, "M", "r20261028-184657-1d80 (TOP-138 harness, note:top/"
                                  "20261028T1900Z-report-lowering-harness): the (64, 4x8) H1 unit, flat, 4.912 GiB peak "
                                  "for 2,384,742 ANDs, 2,212 B an AND, fixed overheads included; reported beside as "
                                  "lower_peak_GiB_today (the brief's 600 B a gate at the 1e9-gate tile is a floor: it OOMed)"),
    "lower_s_per_and_today": (107.4 / 2_384_742, "M", "r20261028-184657-1d80: 1.79 min for the same unit, 45 us an AND"),
    "lower_s_per_gate": (0.9e-6, "E", "TOP-138's target, 1e9 gates in 15 min"),
    "gates_per_and": (1.0e9 / (32 * 23e6), "E", "the K 2,048 32x32 tile's ~1e9 gates over its 32 lanes of ~23 M ANDs "
                                               "(note:top/20261028T1845Z-report-lowering-struct): 1.36"),
    "k_max": (27, "M", "circuit.compose's K_MAX and the prover's IN_RANGE_K_LOG (live/src/circuit.rs:120): a block with "
                       "k_log > 27 is refused (note:top/20261028T1845Z-report-lowering-struct)"),
    "k_pad": (2, "E", "bits past ceil(log2 ANDs) a block's k_log takes: compose pads each range (gates, hm96 slots, "
                      "sha512x3 slots, mask) to a power of two and lays them side by side (circuit._ranges). "
                      "class_size.MEASURED's staged units at Llama-3.2-1B give 1 to 3, median 2, for units whose gates "
                      "dominate (Gemm_v2{2048,16}: 16.9 M ANDs at k_log 27; AttnBlock_v5 9.3 M at 26; SiluMul 11.3 M "
                      "at 26). The struct worker's 'a lane fits 2^27 to K about 5,041' (~57 M ANDs) is 1; the "
                      "inbox's '2^27 ANDs' (top, 19:16Z) is 0"),
    "k_pad_rows": (1, "M", "k_pad of a block whose row openings are at least half its ANDs: GD-PROVER's composed s8 "
                           "blocks, 88-93% openings (note:compute-research/20261028T1955Z-report-gd-prover: K 1,024's "
                           "1.94 M ANDs at k_log 22, K 65,536's 69.0 M at 27, counted at 58,120 a compression), need at "
                           "most 0.66 and 0.53 bits past log2 of their ANDs counted at 81,402, which already holds most "
                           "of SHA-512's 1.50 bits an AND"),
    "k_pad_h1_lane": (1, "E", "k_pad of an H1 lane or combine block: the struct worker's 'a lane fits 2^27 to K about "
                              "5,041' (h1unit.stager; note:top/20261028T1845Z-report-lowering-struct)"),
    "h1_ands_per_mac": (300, "E", "J10-R's H1 leaf, ANDs a MAC before its openings (inbox 19:32Z: 'about 300 ANDs a "
                                  "MAC'; with the tile's openings, the 32x32 K 2,048 leaf is the inbox's ~736 M) until "
                                  "GD-PROVER"),
    "prover_bytes_per_row": (474, "E", "C-Flock prover memory a block row: twice the 31.8 GB circuit at k_log 27 "
                                       "(test_universal_same_shape's docstring)"),
    "prover_mem_GB": (141, "M", "the H200's HBM3e"),
    "cap_cost_per_bit": (0.05, "E", "per k_log past 27 (a fast100 schedule and Stmt.InRange extended, inbox 19:32Z): "
                                    "more queries and a bigger proof, on a block's per-AND prove cost and its "
                                    "statement's verify cost"),
    "retain_GB": (100.0, "E", "GB a window may retain until its draw: the H200's 141 GB less weights, KV cache and "
                              "activations under --retain device; VERITY_RETAIN=host (served_verdicts.sh) is bounded by "
                              "the pod's host cgroup instead (host_retention.py), size not looked up"),
    "host_retain_GB": (200.0, "E", "GB of host memory a GPU's window may retain under VERITY_RETAIN=host: an eighth of "
                                   "a node's, at the Nebius RTX PRO 6000 nodes' 1,690 GiB (infra/cluster/descriptions/"
                                   "nebius.toml) less the system; the H200 node's size isn't in the repo"),
    "d2h_GBs": (57.1, "M", "pinned D2H over PCIe 5.0 x16 (64 MiB pieces), r20261012-070151-be14 on the RTX PRO 6000 "
                           "(note:ra-pcie-exhaust/20261012T0642Z-report-ra-pcie-exhaust); the H200 SXM's host link is "
                           "the same Gen5 x16 (estimate)"),
    "dma_hbm_cost": (0.07, "M", "a DMA'd stream costs 7% of HBM read (1,531 to 1,423 GB/s), r20261012-070151-be14; "
                                "charged on the step's time the copy overlaps (estimate: as if the step were all "
                                "memory-bound)"),
    "main_async_vs_2cd6": (28.41 / 21.60, "M", "main's async stock over 2cd6's at T 8,192, 2,048 requests: "
                                               "stock-async-n2048 r20261028-080545-d0a8 (vllm's 227-step record) "
                                               "28.41 ms against r20261028-075211-2cd6's 21.60 (BENCH-34, art:a5ea06dd"
                                               "…); 1.38 and 1.36 at T 16,384 and 32,768. Taken at every T, an "
                                               "estimate below 8,192"),
    "partial_words_per_ks": (2, "M", "leaf_lanes: a lane partial is 2 (KS + m) u32, KS = K / 32 segments"),
    "cpu_s_in_gpu_s": (1 / 55, "M", "benchmarks/sampling_plan/inputs.py: vCPU $0.033/h against sm_120 $1.80/h"),
}


_OVERRIDE: dict[str, float] = {}


def I(key: str) -> float:
    return _OVERRIDE[key] if key in _OVERRIDE else INPUTS[key][0]


class inputs:
    """`with inputs(name=value, ...):` evaluates under those input values (GD-SEARCH's scenarios); unknown names are
    refused."""

    def __init__(self, values: dict | None = None, **kw):
        self.values = {**(values or {}), **kw}
        unknown = set(self.values) - set(INPUTS)
        if unknown:
            raise KeyError(f"unknown inputs {sorted(unknown)}")

    def __enter__(self):
        self.saved = dict(_OVERRIDE)
        _OVERRIDE.update(self.values)
        return self

    def __exit__(self, *exc):
        _OVERRIDE.clear()
        _OVERRIDE.update(self.saved)
        return False


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
        "decode_steps": (0, "decode steps a window beside its steps_per_window, each computing one token a sequence "
                            "(the demo's 4 outputs: 3 decode steps after the prefill)"),
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
        "values": ("every_op", "every_op | dedup | fused_mlp | layer | layer_kv | e1 | e2 (GD-PARTITION's measured "
                               "bytes) | partition (from the units' outputs)"),
        "bytes_per_token": (None, "overrides values"),
        "bits_per_word": (16, "bits a committed word for the bf16 values (int8 and f32 entries unchanged)"),
        "row_bytes": (4096, "hm96 row size of the Commit's tree (4 KiB today; 16 KiB: lever 11 of P1a)"),
        "hash_eff": (None, "share of HASH-38's bar the hash kernels reach (None: b921's 0.497; the row kernel: "
                           "INPUTS hash_eff_rows, 0.92)"),
        "corun": (False, "False (b921's measured overlap) | True or 'corun4' (GD-HASH's beside curve: bytes up to the "
                         "beside capacity cost the GEMM h_b / h_x of their exclusive time, the rest run alone; hashing "
                         "in host gaps free, an estimate)"),
        "open": ("whole", "whole (an opened byte range pays every row_bytes row it touches: hm96 hashes whole rows) | "
                          "exact (each read its own row of its size, as costfit's sessions ran)"),
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
        "noise": ("periodic", "periodic | fresh (along K) | permuted (v3.2: U's columns re-permuted each tile; "
                              "GD-NOISE's 2:4 route forces 0.97 G at t 3) | segment (pi re-drawn each segment, "
                              "which closes that route)"),
        "u_draw": ("period", "jr2: U drawn once a period, W' and U stored (GD-NOISE's 'JR2 per period') | call "
                             "(#6369: U drawn every call, n K PRG bytes, SHA-bound at small m)"),
        "record_bytes_per_cell": (6, "J10-R's z/y records, 6 B a cell"),
        "trees": (2, "trees H1's records are hashed into (2 today: z table and zy@c; 1: Daniel's)"),
        "h1_host": ("main", "pre (a1fc) | main (#5975, #6016: 5430) | main+6192 (65ca) | stack (every H1 host lever) | "
                            "floor (H1-COST's floor)"),
        "kdhash": (False, "#6182: H1's tables hashed in one pass"),
        "hash_at_bar": (False, "H1's hashing at HASH-38's bar"),
        "coin_off_loop": (False, "the round coin's wait off the engine thread"),
        "noise_fused": (False, "dither: X formed in the GEMM's operand load (False: GD-DENOISE's measured separate "
                               "arm)"),
        "denoise_g_frac": (None, "dither: de-noising's cost as a share of G (None: from noise and L')"),
        "value_row_bytes": (16384, "dither: the value root's hm96 row (16 KiB pinned in PROTOCOL.md)"),
        "tile": (None, "PoUW proof block's (R, C) cells (None: dither 1 x 1 over one segment; J10-R and JR2 32 x 32, "
                       "the H1 kernel's warp tile). Registered operand rows are K + 4 bytes (chunk rows past K 4,096), "
                       "opened whole; dither's A segment opens the Commit's rows"),
        "credit": ("all4", "dither's credited linears: all4 (qkv, o, gate_up and down) | k2048 (the K 2,048 three)"),
        "word_bits": (16, "dither: bits a T_s word (O stays 32)"),
        "agg": (1, "dither: cells a T_s word"),
        "joint": (False, "dither: each credited values tile also checks its cells' segments, drawn at the larger of "
                         "the two laws (no separate PoUW blocks)"),
        "form": ("tile", "j10r/jr2 leaf's proof: tile (one block; ~2^30 at K 2,048, refused by compose) | lanes "
                         "(TOP-138, pouw.leaf_lanes: its 32 lanes of 4 x 8 cells, each in weights.chunks(K) chunks, and "
                         "a combine over the lanes' partials, drawn as one group: delta counts leaves)"),
    },
    "sampling": {
        "delta": (2.0 ** -40, "failure probability a window"),
        "eps": (0.01, "share of credited work allowed wrong"),
        "pouw_sampled": (True, "PoUW's units drawn at (eps, delta) a window too (guarantee 2)"),
        "law": ("achievability", "achievability (q_u = min(1, Lambda w_u / C), Partitioning/Guarantees.lean) | uniform "
                                 "(the deployed audit's uniform kd-subset over a population's units, SampledKSLiveV3, "
                                 "at eps w_avg / w_max of the units, so that eps still bounds the wrong work) | "
                                 "stratified (the deployed IntegrityPeriod law, benchmarks/sampling_plan/h200.py: each "
                                 "unit class its own uniform kd-subset at eps of its units and delta / S, S classes)"),
        "retain": ("device", "where a window's retained bytes wait for the draw: device (HBM, retain_GB) | host "
                             "(VERITY_RETAIN=host, served_verdicts.sh: host_retain_GB, each step's bytes copied over "
                             "PCIe on a copy engine beside the step, the excess and the DMA's HBM share exposed)"),
        "retain_GB": (None, "GB a window may hold until its draw (the Commit's step bytes, Dither's values, H1's "
                            "records, times steps a window; None: INPUTS retain_GB or host_retain_GB); over it, the "
                            "design is listed infeasible"),
        "period": (1, "windows certified together (the accounting period, ASK-64's form: Properties/Pouw/Period.lean, "
                      "eps(j) = Lambda ln(1/delta) / j): draws ~ Lambda / eps a period, so proving a window falls as "
                      "1 / period, while eps bounds the period's work, not a window's (a window's share up to "
                      "eps x period). Coins stay per unit and live, so a window retains only its own bytes"),
    },
    "partition": {
        "units": (None, "list of unit classes {name, per_token, work, extra_ands, open_rows, out_bits}; None: the "
                        "preset per op of every_op (an r x c tile of a linear a unit, an element a unit elsewhere)"),
        "tile": ((1, 1), "a linear's proof block: r tokens x c output columns (1 x 1: one s8 coordinate)"),
        "kchunk": (1, "a linear's K cut into chunks, each a block committing its running sums (4 B a cell a cut)"),
        "attn_keys": (64, "keys an attention proof block (None: class_size's measured 128, whose block with its K and "
                          "V rows' openings at C-Flock's SHA-512 passes 2^27 at k_pad 2; 64 fits, at "
                          "attn_ands_per_mac, an estimate below the measured block)"),
        "census": (None, "None: the model's classes (attention in attn_keys blocks, one elementwise class) | b4: "
                         "GD-PLANNER's census of the served 4,32,4 window (census_b4.json), every template its own "
                         "class beside the model's linear tiles, lm_head rebound at N 1 | b4-buckets: b4 with the "
                         "35 attention templates padded into the widest (the planner's pad:35)"),
    },
    "prover": {
        "device": ("h200", "the prover's GPU: h200 (GD-PROVER's measured drawn-AND cost, o_per_and_h200) | sm120 (the "
                           "RTX PRO 6000's costfit, its GPU-s at sm120_gpu_price)"),
        "reps": (2, "C-Flock reps on one root: 2 (fast100x2, 195.5 bits) | 1 (Q a level up to clear the 100-bit "
                    "floor: rep_share less drawn-AND time, one_rep_query_cost more)"),
        "o_per_and": (None, "GPU-s a padded drawn block's AND (None: the device's)"),
        "staged_per_and": (None, "GPU-s a staged block's AND"),
        "stmt_fixed": (None, "GPU-s a session (touched statement)"),
        "stmt_per_and": (None, "GPU-s a session per AND of its block"),
        "speed": (None, "a further divisor on proving GPU-s (None: 1.0)"),
        "m_max": (35, "C-Flock's m = k_log + nbl cap (a block's k_log itself is refused above k_max)"),
        "k_max": (None, "the block cap's k_log (None: 27, compose's K_MAX; 28-30 with a fast100 schedule and an "
                        "extended Stmt.InRange, each bit past 27 at cap_cost_per_bit and prover memory 2^k_log x "
                        "prover_bytes_per_row against prover_mem_GB)"),
        "k_pad": (None, "bits past log2 ANDs a block's k_log takes, k_log = ceil(log2 ANDs + k_pad) (None: INPUTS "
                        "k_pad_rows, 1, where openings are at least half the block, else k_pad, 2, from class_size's "
                        "gate-dominated staged units; 0: the inbox's 2^27 ANDs a block)"),
        "shared": (True, "statements hold only the drawn blocks (IntegrityPeriod's hidden window)"),
        "stmt": ("template", "template (a statement a block class) | universal (private_circuits' universal unit: each "
                             "block at its power-of-two size class, statements shared by a size class; no universal "
                             "overhead counted, an estimate)"),
    },
    "lowering": {
        "gates_per_and": (None, "gates a proof block's AND (None: 1.36, the H1 tile's gates over its lanes' ANDs)"),
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
        "baseline": ("main", "t_stock: main (the fastest unverified arm measured on main's stack with the served "
                             "engine options, stock-async-n2048: the async stock at the design's point times "
                             "main_async_vs_2cd6; the coordinator, 20:22Z) | same (the stock forward at the design's "
                             "point and scheduler: b57f's sync stack, or 2cd6's if point.async, both on next) | "
                             "fastest (the async stock, 2cd6's stack on next)"),
    },
}

PRESETS = {
    "throughput": {},
    # served_verdicts.sh: 4 requests of 16 to 24 prompt ids and 4 outputs, so a prefill step and 3 decode steps
    # values are proved over the class circuit's census (its caps: 4 x 32 prompt rows and 4 x 3 decode rows)
    "demo": {"point": {"T": 512, "real_tokens": 64, "seqs": 4, "decode_steps": 3, "window_class": "4,32,4"},
             "partition": {"census": "b4"}},
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


#: GD-PARTITION's committed bytes a token (r20261028-192825-df06, the served decode step at T 35, layers and
#: embedding): e1 epilogue fusion with the residual once a layer, e2 that with max-first recompute. Taken a prefill
#: token too (an estimate); proof blocks stay the preset's
PARTITION_BETA = {"e1": 823_488, "e2": 563_456}


def commit_bytes_per_token(d: dict, units: list) -> float:
    c = d["commit"]
    if not c["on"]:
        return 0.0
    if c["bytes_per_token"] is not None:
        return c["bytes_per_token"]
    if c["values"] == "partition":
        return sum(u["per_token"] * u["out_bits"] for u in units) / 8
    sums = LAYERS * sum(4 * N for _, K, N in LINEARS) * (d["partition"]["kchunk"] - 1)
    if c["values"] in PARTITION_BETA:
        return PARTITION_BETA[c["values"]] + sums
    return stream_bytes_per_token(c["values"], c["bits_per_word"]) + sums


def hash_ms(compressions: float, eff: float, mhz: float) -> float:
    return compressions / (I("sha_bar_per_sm_clock") * I("sms") * mhz * 1e3 * eff)


def commit_parts(d: dict, stock: dict, units: list) -> tuple[dict, dict]:
    """The Commit's parts, ms a step, and what they rest on: its tree's compressions at the hash kernels' share of the
    bar, less what overlaps (today: b921's 22% of the kernel time under tensor work, at its measured GEMM loss;
    co-run: GD-HASH's beside curve, nothing free under the GEMM, host gaps free), the taps, the gather, the host
    copies, the event syncs and the seal, each scaled from b921 by bytes or held per step."""
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
    ratio, beside_gb = beside(c["row_bytes"], tok_c)
    if c["corun"]:
        idle = min(kernel, stock["host gap between forwards"] + stock["launch gaps in the forward"])
        fits = min(1.0, beside_gb * 1e9 * stock["int8 GEMM"] / I("beside_gemm_ms") / step_bytes)
        side = min(kernel - idle, kernel * fits)
        exposed = kernel - idle - side
        penalty = ratio * side
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
            parts["seal: rest of the host gap"] = I("seal_rest_ms") / S
    if c["seal"] == "tree":
        seal_bytes = (tree["rows"] + tree["nodes"]) * 64
        parts["seal wait"] = seal_bytes / (I("seal_GBs") * 1e6) / S
    info = {"bytes_per_token": bpt, "step_bytes": step_bytes, "tree": tree, "compressions_per_step": comp,
            "hash_kernel_ms": kernel, "hash_eff": eff, "beside_h_ratio": ratio,
            "beside_GB_a_step": beside_gb * stock["int8 GEMM"] / I("beside_gemm_ms")}
    return parts, info


def _log_interp(x: float, xs: tuple, ys: list) -> float:
    """Piecewise linear in log2 x through (xs, ys), held flat outside."""
    lx, pts = math.log2(x), [(math.log2(a), b) for a, b in zip(xs, ys)]
    if lx <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if lx <= x1:
            return y0 + (y1 - y0) * (lx - x0) / (x1 - x0)
    return pts[-1][1]


def beside(row_bytes: int, tok: float) -> tuple[float, float]:
    """GD-HASH's beside curve at a row size and T: (h_b / h_x, GB a forward that fit beside the GEMM below the whole
    grid at its 14.5 ms). Rows under 4 KiB take 4 KiB's figures (unmeasured)."""
    rows = (4096, 16384, 65536)
    hi = _log_interp(row_bytes, rows, [I("beside_4k"), I("beside_16k"), I("beside_64k")])
    lo = _log_interp(row_bytes, rows, [I("beside512_4k"), I("beside512_16k"), I("beside512_64k")])
    gb = _log_interp(row_bytes, rows, [I("beside_GB_4k"), I("beside_GB_16k"), I("beside_GB_64k")])
    return _log_interp(tok, (512, 8192), [lo, hi]), gb


def l_prime(w: dict) -> int:
    return w["L_prime"] or (128 if w["scheme"] == "dither" else 32)


def pouw_forced_frac(d: dict) -> float:
    """The least work a credited call forces, as a share of its plain s8 GEMM G, under SAK on noised operands, at the
    best attack found. Un-noised calls earn 0 (the brief): J10-R as served adds no noise at serve time, and at L >= K a
    dither cell commits O alone. Dither, periodic or with pi re-drawn each segment: min(1, 1/8 + 3/t) (the floor note;
    GD-NOISE finding 2). Pi re-drawn each tile (v3.2): the merged factor's high limb runs on 2:4 sparse s8 at half
    price, 2.54 limb-products a word, min(1, 1/8 + 2.54/t) (GD-NOISE finding 1, shown at t 3: 0.97 G). Noise fresh
    along K, or JR2: G, the one-word bound's second product (GD-NOISE finding 3; GD-BITS, if JR2's proof holds)."""
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
    if w["noise"] not in ("periodic", "permuted", "segment"):
        raise ValueError(f"pouw.noise {w['noise']!r}")
    t = max(1, L // Lp)
    limbs = I("permuted_limbs") if w["noise"] == "permuted" else 3.0
    return min(1.0, 1.0 / I("b1_rate_vs_s8") + limbs / t)


def guess_p(bits: int) -> float:
    """A word's free-guess rate: 2^-b to 8 bits (GD-BITS, r20261028-194424-e326: sampling pays 1 / (1 - p), x1.005 at
    8 bits, x1.07 at 4, x2.0 at 1), the measured 16-bit rate scaled above. A mix4 aggregate of g >= 8 cells keeps
    its word's rate (GD-AGG)."""
    return 2.0 ** -bits if bits <= 8 else I("guess_p16") * 2.0 ** (16 - bits)


def dither_linears(d: dict):
    """The linears dither credits: (name, K, N)."""
    return [x for x in LINEARS if d["pouw"]["credit"] == "all4" or x[1] == 2048]


def pouw_credit(d: dict) -> dict:
    """Guarantee 2's credit as a share of the four linears' int8 MACs: a dither call with L < K credits (K - L) / K of
    its MACs (the floor note; S15 of note:compute-research/20261028T1900Z-report-gd-smell), and the work that credit
    forces is the floor's share of it, at the best attack found (an upper bound on the floor, not a bound)."""
    w = d["pouw"]
    total = sum(K * N for _, K, N in LINEARS)
    if w["scheme"] == "dither":
        cred = sum(K * N * (K - w["L"]) / K for _, K, N in dither_linears(d) if w["L"] < K) / total
    elif w["scheme"] == "jr2":
        cred = sum(K * N for _, K, N in LINEARS if K == w["K"]) / total
    else:
        cred = 0.0
    forced = pouw_forced_frac(d)
    return {"credited_share_of_linear_macs": cred,
            "forced_share_at_best_attack": cred * forced,
            "earns_credit": cred > 0 and forced >= 1 - d["sampling"]["eps"],
            "basis": "best attack found (an upper bound on the floor; no proved lower bound yet)"
                     if w["scheme"] == "dither" else
                     ("one-word bound, G (GD-NOISE finding 3, GD-BITS), if JR2's proof holds" if w["scheme"] == "jr2"
                      else "un-noised at serve time: 0 under guarantee 2")}


def _h1_main_ms_8k(h1_host: str) -> float:
    """H1's own time in a measured post-fix step at T 8,192: the step (5430, or 65ca with #6192) less the modelled
    stock forward and unsealed Commit of the same configuration, and less the round coin, which `pouw_parts` adds."""
    s = I("h1_post_step_s_8k" if h1_host == "main" else "h1_post6192_step_s_8k") * 1e3
    d = design(forward={"kernels": "commit_taps"}, commit={"seal": "none"}, pouw={"scheme": "off"})
    stock = stock_parts(d)
    rest = sum(stock.values()) + sum(commit_parts(d, stock, preset_units(d))[0].values())
    return s - rest - I("h1_coin_ms")


#: GD-DENOISE (note:compute-research/20261028T1930Z-report-gd-denoise, run r20261028-190744-7676, one H200,
#: cuBLASLt int8): Dither v3.2's served-call terms as multiples of the plain GEMM, model sums at m 64 / 512 / 8,192.
#: coins at L' 32 / 128 / 512 (d_A bits and V), X, O's recompute on a', b' less the plain GEMM, fresh noise with O
#: recomputed (without T_s), and ts_out: a T_s chunk's int32 output share at K 2,048 (the S 2 shapes' chunk arm less
#: its L / K mainloop; derived)
DENOISE_M = (64, 512, 8192)
DENOISE = {"coins32": (1.483, 0.759, 0.155), "coins128": (1.479, 0.749, 0.205), "coins512": (1.499, 0.782, 0.470),
           "x": (0.161, 0.100, 0.075), "rec": (0.333, 0.258, 0.202), "fresh": (3.21, 3.33, 4.0),
           "ts_out": (0.24, 0.42, 0.49)}


def _dither_calls(d: dict, tok: float):
    """Per credited call class: (cells, segments)."""
    for _, K, N in dither_linears(d):
        yield LAYERS * N * tok, math.ceil(K / d["pouw"]["L"])


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
            if w["u_draw"] not in ("period", "call"):
                raise ValueError(f"pouw.u_draw {w['u_draw']!r}")
            dg = I("jr2_noise_g")
            if w["u_draw"] == "call":
                dg += I("jr2_call_extra_g") + I("jr2_wprg_g_m") / max(1.0, tok)
            if w["denoise_g_frac"] is not None:
                dg = w["denoise_g_frac"]
            parts[f"JR2: the noised product and U's noise (GD-NOISE, U a {w['u_draw']})"] = dg * G_ms * k_share
            info["jr2_noise_g"] = dg
        return parts, info
    # dither: GD-DENOISE's served call over the credited calls' plain GEMM (an L >= K call commits O alone, un-noised)
    L, Lp = w["L"], l_prime(w)
    G_c = {name: G_ms * LAYERS * K * N / WEIGHT_BYTES for name, K, N in dither_linears(d) if L < K}
    G_cred = sum(G_c.values())
    dn = {k: _log_interp(tok, DENOISE_M, list(v)) for k, v in DENOISE.items()}
    if w["noise"] == "fresh":
        parts["dither: fresh coins, noised A and O recomputed"] = dn["fresh"] * G_cred
        dg = dn["fresh"]
    else:
        coins = _log_interp(Lp, (32, 128, 512), [dn["coins32"], dn["coins128"], dn["coins512"]])
        parts["dither: coins (d_A bits and V)"] = coins * G_cred
        parts["dither: noised A (X)"] = dn["x"] * G_cred * (I("noise_fused_keep") if w["noise_fused"] else 1)
        dg = w["denoise_g_frac"] if w["denoise_g_frac"] is not None else dn["rec"]
        parts["dither: O recomputed on a', b' (less the plain GEMM)"] = dg * G_cred
    # the noised GEMM over k < (S - 1) L, a chunk a segment, each writing its T_s words (int32 output share scaled
    # to the word: an estimate below GD-DENOISE's int32 read-back arm)
    parts["dither: noised GEMM over (S-1) L, T_s out"] = sum(
        g * (math.ceil(K / L) - 1) * (L / K + dn["ts_out"] * (2048 / K) ** 0.5 * w["word_bits"] / 32 / w["agg"])
        for (name, K, _), g in ((x, G_c[x[0]]) for x in dither_linears(d) if x[0] in G_c))
    calls = list(_dither_calls(d, tok))
    cells = sum(c for c, _ in calls)
    words = sum(c * ((s - 1) * w["word_bits"] / 8 / w["agg"] + 4) for c, s in calls)
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


def h1_chunks(K: int) -> int:
    """pouw.weights.chunks: the chunks an H1 lane at K is proved in (1 to K 4,096, CHUNK_MAX; then the least n >= 2
    whose chunks are at most 4,096 entries and whole 512-entry noise blocks)."""
    if K <= 4096:
        return 1
    for n in range(-(-K // 4096), K // 512 + 1):
        if K % (n * 512) == 0:
            return n
    raise ValueError(f"K = {K}: no chunking")


def h1_digits(ks: int) -> int:
    """pouw.leaf_lanes.n_digits: H1's digit count m at KS segments."""
    m = 1
    while not (-(128 ** m) <= -128 * ks and 127 * ks // 128 ** (m - 1) <= 127):
        m += 1
    return m


def opened(nbytes: float, R: int, mode: str) -> list:
    """The rows a read of nbytes contiguous bytes opens: every R-byte row it touches (whole, at least one; aligned), or
    one row of its own size (exact)."""
    if mode == "exact":
        return [nbytes]
    return [R] * max(1, math.ceil(nbytes / R))


def dither_check(d: dict, K: int, span: int, cells: int) -> tuple[float, list]:
    """Dither's in-circuit check of `cells` cells over `span` of K: the noised product's ANDs and the coins'
    expansion, and the T_s rows it opens (value rows of value_row_bytes)."""
    w = d["pouw"]
    L, Rv = w["L"], w["value_row_bytes"]
    segs = max(1, span // L)
    noise = 2 * span * I("sha512_ands") / 64                   # V and U pi entries expanded from the coins (estimate)
    words = cells * (segs + 1) * w["word_bits"] / 8 / w["agg"]
    return cells * span / 32 * I("s8_step_ands") + noise, opened(words, Rv, "whole") + [Rv]


def preset_units(d: dict) -> list:
    """The per-op partition (every committed value a unit's output, at most 16 bits a word): an r x c tile of a
    linear's outputs over K / kchunk a unit (r A rows, c W rows, the scales and the r output rows; with K chunks, the
    running sums in and out; at 1 x 1 costfit's ScaledMmS8Coordinate_v1), one element a unit for the elementwise ops
    (opening one input row), one (head, query) a unit for attention. Rows are the Commit's (W's taken at the same
    size, an estimate), opened whole or exact (commit.open). Under pouw.joint each credited tile also checks Dither
    on its cells (`credit_work` its MACs)."""
    R, mode = d["commit"]["row_bytes"], d["commit"]["open"]
    p, w = d["point"], d["pouw"]
    r, c = d["partition"]["tile"]
    kc = d["partition"]["kchunk"]
    seq_len = (p["real_tokens"] or p["T"]) / p["seqs"]
    joint = w["joint"] and w["scheme"] == "dither" and d["sampling"]["pouw_sampled"]
    credited = {x[0] for x in dither_linears(d) if w["L"] < x[1]}
    units = []
    for name, K, N in LINEARS:
        Kc = K // kc
        rows = [b for _ in range(r) for b in opened(Kc, R, mode)] + [b for _ in range(c) for b in opened(Kc, R, mode)]
        rows += opened(4 * r, R, mode) + opened(4 * c, R, mode) + [b for _ in range(r) for b in opened(2 * c, R, mode)]
        if kc > 1:
            rows += [b for _ in range(2 * r) for b in opened(4 * c, R, mode)]
        u = {"name": f"{name} {r}x{c} tile (K {K}{f' / {kc}' if kc > 1 else ''})",
             "per_token": LAYERS * N * kc / (r * c), "work": r * c * s8_coord_work(Kc), "extra_ands": 0.0,
             "open_rows": rows, "out_bits": 16 * r * c}
        if joint and name in credited:
            extra, vrows = dither_check(d, K, Kc, r * c)
            u.update({"extra_ands": extra, "open_rows": rows + vrows, "credit_work": r * c * Kc})
        units.append(u)
    if d["partition"]["census"]:
        return units + census_units(d)
    bn = d["partition"]["attn_keys"] or I("attn_keys_per_block")
    blocks = max(1.0, seq_len / 2 / bn)                        # a query's causal keys, on average, in key blocks
    kb = min(bn, seq_len / 2)
    attn_rows = opened(2 * HD, R, mode) + 2 * opened(kb * HD * 2, R, mode) + 2 * opened(4 * (HD + 2), R, mode)
    units.append({"name": f"attention (head, query, {bn:.0f}-key block)", "per_token": LAYERS * QH * blocks,
                  "work": I("attn_ands_per_mac") * 2 * HD * kb, "extra_ands": 0.0,
                  "open_rows": attn_rows, "out_bits": 32 * (HD + 2)})
    ew = (LAYERS * (2 * HIDDEN + 2 * HIDDEN + INTER + (QH + KVH) * HD + 3 * HIDDEN + INTER))
    units.append({"name": "elementwise element", "per_token": ew, "work": I("ew_ands_per_elem"), "extra_ands": 0.0,
                  "open_rows": [R], "out_bits": 16})
    return units


def registered_rows(K: int) -> list:
    """The registered rows a K-entry operand row is hashed in: one row of K + 4 bytes to K 4,096, then h1_chunks(K)
    chunk rows of K / n + 4 (pouw.weights)."""
    n = h1_chunks(K)
    return [K // n + 4] * n


def pouw_units(d: dict) -> list:
    """PoUW's proof blocks, an R x C tile of cells each (pouw.tile). J10-R/JR2: the tile's H1 leaf at K, ANDs a MAC
    before its openings (h1_ands_per_mac), opening its R A rows and C B rows (registered rows) and its z/y record;
    the lanes form splits it into 4 x 8 lanes (each its own A and B rows, chunked past K 4,096, and a partial row)
    and a combine block, drawn as a group. Dither: the tile over one segment (PROTOCOL.md), opening A's segment rows
    (the Commit's rows), B's registered rows and the tile's T_s and T_{s-1} (or O) value rows, row-major."""
    w = d["pouw"]
    if w["scheme"] == "off" or not d["sampling"]["pouw_sampled"]:
        return []
    R, mode = d["commit"]["row_bytes"], d["commit"]["open"]
    if w["scheme"] in ("j10r", "jr2"):
        r, c = w["tile"] or (32, 32)
        K, kp = w["K"], I("k_pad_h1_lane")
        work = r * c * s8_coord_work(K)
        compute = I("h1_ands_per_mac") * r * c * K
        record = opened(r * c * w["record_bytes_per_cell"], R, "exact")
        unit = {"name": f"H1 leaf ({r}x{c}, K {K}, {w['form']})", "per_token": H1_CELLS_PER_TOKEN / (r * c),
                "work": work, "extra_ands": max(0.0, compute - work), "k_pad": kp, "out_bits": 16,
                "open_rows": [b for _ in range(r + c) for b in registered_rows(K)] + record}
        if w["form"] not in ("tile", "lanes"):
            raise ValueError(f"pouw.form {w['form']!r}")
        lr, lc = min(r, 4), min(c, 8)
        lanes = (r // lr) * (c // lc)
        if w["form"] == "lanes" and lanes > 1:
            n = h1_chunks(K)
            ks = K // 32
            partial = 4 * I("partial_words_per_ks") * (ks + h1_digits(ks))
            # a chunk writes the lane's partial row (the last) or a carry, and a later chunk reads the carry
            rows = [b for _ in range(lr + lc) for b in registered_rows(K)[:1]] + [partial] * (2 if n > 1 else 1)
            combine = (lanes * row_ands(partial) + row_ands(16) + sum(row_ands(b) for b in record)
                       + (lanes * n * row_ands(2) if n > 1 else 0))
            unit["blocks"] = [{"name": "lane chunk" if n > 1 else "lane", "count": lanes * n, "k_pad": kp,
                               "ands": compute / (lanes * n) + sum(row_ands(b) for b in rows)},
                              {"name": "combine", "count": 1, "ands": combine, "k_pad": kp}]
        return [unit]
    if w["joint"]:
        return []                                              # checked inside the values tiles (preset_units)
    r, c = w["tile"] or (1, 1)
    g = w["agg"]
    if g > r * c:                                              # a block opens whole words: at least a group's cells
        r = 1 << (g.bit_length() - 1) // 2
        c = -(-g // r)
    L, Rv = w["L"], w["value_row_bytes"]
    noise_ands = (r + c) * L * I("sha512_ands") / 64           # V and U pi entries expanded from the coins (estimate)
    mix = r * c * I("mix4_ands") if g > 1 else 0.0
    t_rows = opened(c * w["word_bits"] / 8 / g, Rv, "whole")
    units = []
    for name, K, N in dither_linears(d):
        if L >= K:
            continue                                           # O alone: un-noised, credits 0, not drawn
        S_seg = math.ceil(K / L)
        b_rows = registered_rows(K)[:1] if mode == "whole" else [L]
        units.append({"name": f"dither block, {name} ({r}x{c}, L {L} of K {K})",
                      "per_token": LAYERS * N * S_seg / (r * c), "work": r * c * L / 32 * I("s8_step_ands"),
                      "extra_ands": noise_ands + mix, "draw_factor": 1 / (1 - guess_p(w["word_bits"])),
                      "open_rows": [b for _ in range(r) for b in opened(L, R, mode)] + b_rows * c + t_rows * 2 * r,
                      "out_bits": 16})
    return units


CENSUS_B4 = Path(__file__).with_name("census_b4.json")


def census_units(d: dict) -> list:
    """GD-PLANNER's census of the served 4,32,4 window as unit classes, a template a class at its own ANDs and rows,
    counted a window: every template but the int8 linears (ScaledMmS8_v1), which stay the model's tiles. lm_head
    (Gemm_v2 at N 16, all 140 rows) is rebound at N 1, the planner's s*, into Gemm_v2{K=2048,N=1}'s template.
    b4-buckets pads the 35 AttnBlockFirst_v2 templates into their widest (NVIS 35)."""
    mode = d["partition"]["census"]
    if mode not in ("b4", "b4-buckets"):
        raise ValueError(f"partition.census {mode!r}")
    if d["point"]["window_class"] != "4,32,4":
        raise ValueError(f"partition.census {mode!r} is the 4,32,4 window's; the point is "
                         f"{d['point']['window_class']!r}")
    rows = {}
    for t in json.loads(CENSUS_B4.read_text())["templates"]:
        t = dict(t)
        if t["label"].startswith("ScaledMmS8_v1"):
            continue
        if t["label"] == "Gemm_v2{K=2048,N=16}":
            t = {**t, "label": "Gemm_v2{K=2048,N=1}", "units": t["units"] * 16, "ands": None}
        if mode == "b4-buckets" and t["label"].startswith("AttnBlockFirst_v2"):
            t["label"] = "AttnBlockFirst_v2{pad:35}"
        r = rows.setdefault(t["label"], {**t, "units": 0})
        r["units"] += t["units"]
        if t["ands"] is not None and (r["ands"] is None or t["ands"] > r["ands"]):
            r.update({k: t[k] for k in ("ands", "in", "out", "out_bits")})
    tok = window_tokens(d)
    return [{"name": f"{r['label']} (census)", "per_token": r["units"] / tok, "work": float(r["ands"]),
             "extra_ands": 0.0, "open_rows": list(r["in"]) + list(r["out"]), "out_bits": r["out_bits"]}
            for r in rows.values()]


def window_tokens(d: dict) -> float:
    """Tokens a window's forwards compute: steps_per_window steps of real_tokens, and decode_steps of one a sequence."""
    p = d["point"]
    return p["steps_per_window"] * (p["real_tokens"] or p["T"]) + p["decode_steps"] * p["seqs"]


def uniform_kd(n: float, eps: float, delta: float) -> float:
    """The deployed audit's uniform kd-subset (Flock.Guarantees.SampledKSLiveV3): the least kd with
    C(n - K0, kd) / C(n, kd) <= delta, K0 = floor(eps n) + 1. The ratio is C(n - kd, K0) / C(n, K0), summed term by term
    for K0 <= 10^4 and at the midpoint n - K0/2 above (relative error well under 1e-3 there)."""
    n = int(round(n))
    k0 = math.floor(eps * n) + 1
    if k0 > n:
        return float(n)

    def ok(kd: int) -> bool:
        if kd > n - k0:
            return True
        if k0 <= 10_000:
            r = sum(math.log1p(-kd / (n - i)) for i in range(k0))
        else:
            r = k0 * math.log1p(-kd / (n - k0 / 2))
        return r <= math.log(delta)

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return float(lo)


def proving(d: dict, units: list, label: str) -> dict:
    """A window's draws and proving cost for one sampled population. Achievability's law q_u = min(1, Lambda w_u / C)
    with C = eps F, so draws ~ Lambda F / C = Lambda / eps; or the deployed uniform kd-subset over the population's
    units (ε then counts units, not ANDs). Each drawn block costs o (ANDs + row openings) and each touched statement
    stmt_fixed + stmt_per_and ANDs, with ceil(draws / cap) statements a class (cap = 2^(m_max - k_log), k_log =
    ceil(log2 ANDs) + k_pad, refused above prover.k_max). A k_log past 27 costs cap_cost_per_bit a bit on o and on
    verification, and every block's prover memory is 2^k_log x prover_bytes_per_row against prover_mem_GB. A unit
    drawn as a group (its `blocks`, the lanes form) pays each of its blocks on every draw, each block class its own
    k_log, cap and statements. Lowering is per class (template) or per draw (instance)."""
    s = d["sampling"]
    P = s["period"]
    T, S = window_tokens(d) * P, 1                             # a period's units: P windows certified together
    lam = math.log(1 / s["delta"])
    pv = d["prover"]
    if pv["device"] not in ("h200", "sm120"):
        raise ValueError(f"prover.device {pv['device']!r}")
    if pv["reps"] not in (1, 2):
        raise ValueError(f"prover.reps {pv['reps']!r}")
    o = (pv["o_per_and"] if pv["o_per_and"] is not None else
         I("o_per_and_h200" if pv["device"] == "h200" else "o_per_and"))
    so = (pv["staged_per_and"] if pv["staged_per_and"] is not None else I("staged_per_and"))
    f0 = (pv["stmt_fixed"] if pv["stmt_fixed"] is not None else I("stmt_fixed"))
    f1 = (pv["stmt_per_and"] if pv["stmt_per_and"] is not None else I("stmt_per_and"))
    speed = pv["speed"] if pv["speed"] is not None else 1.0
    price = 1.0 if pv["device"] == "h200" else I("sm120_gpu_price")
    rep = 1.0 if pv["reps"] == 2 else (1 - I("rep_share")) * (1 + I("one_rep_query_cost"))
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
           "prover_peak_GB": 0.0, "period": P, "soundness_bits": I("soundness_bits_2reps" if pv["reps"] == 2 else
                                                                    "soundness_floor_bits"), "infeasible": []}
    if F <= 0:
        return out
    k_cap = round(pv["k_max"] if pv["k_max"] is not None else I("k_max"))
    if s["law"] == "uniform":
        # eps counts units under the kd-subset, so K0 wrong units hold up to K0 w_max of the work: eps of the work
        # (the brief's guarantee) needs eps w_avg / w_max of the units
        n_all = sum(u["per_token"] * T * S for u in units)
        eps_units = s["eps"] * (F / n_all) / max(u["work"] for u in units)
        kd = uniform_kd(n_all, eps_units, s["delta"])
        out.update({"uniform_kd": kd, "uniform_eps_units": eps_units})
    elif s["law"] == "stratified":
        strata = sum(1 for u in units if u["per_token"] > 0)
        out["strata"] = strata
    elif s["law"] != "achievability":
        raise ValueError(f"sampling.law {s['law']!r}")
    universal = pv["stmt"] == "universal"
    if pv["stmt"] not in ("template", "universal"):
        raise ValueError(f"prover.stmt {pv['stmt']!r}")
    W_cred = sum(u["per_token"] * T * S * u.get("credit_work", 0.0) for u in units)
    out["draws_compute"] = 0.0
    plan = []
    for u in units:
        n = u["per_token"] * T * S
        if s["law"] == "uniform":
            q = kd / n_all
        elif s["law"] == "stratified":
            q = uniform_kd(n, s["eps"], s["delta"] / out["strata"]) / n if n > 0 else 0.0
        else:
            q = min(1.0, lam * u["work"] / C)
        q = min(1.0, q * u.get("draw_factor", 1.0))
        if u.get("credit_work") and W_cred > 0:
            qc = min(1.0, lam * u["credit_work"] / (s["eps"] * W_cred))
            out["draws_compute"] += n * qc
            q = max(q, qc)
        opened_ands = sum(row_ands(r) for r in u["open_rows"])
        own = u["work"] + u["extra_ands"] + opened_ands
        k_pad = (pv["k_pad"] if pv["k_pad"] is not None else
                 I("k_pad_rows") if opened_ands >= own / 2 else I("k_pad"))
        blocks = []
        whole = {"name": u["name"], "count": 1, "ands": own, **({"k_pad": u["k_pad"]} if "k_pad" in u else {})}
        for b in u.get("blocks") or [whole]:
            ands = b["ands"]
            if universal:
                ands = 2.0 ** math.ceil(math.log2(max(ands, 2)))
            k_log = math.ceil(math.log2(max(ands, 2)) + b.get("k_pad", k_pad) - 1e-9)
            blocks.append({**b, "ands": ands, "k_log": k_log, "cap": 1 << max(0, pv["m_max"] - k_log),
                           "nb": n * q * b["count"]})
        plan.append((u, n, q, own, blocks))
    pooled: dict[int, float] = {}
    for _, _, _, _, blocks in plan:
        for b in blocks:
            pooled[b["k_log"]] = pooled.get(b["k_log"], 0.0) + b["nb"]
    for u, n, q, own, blocks in plan:
        draws = n * q
        cls = {"name": u["name"], "units": n, "work": u["work"], "q": q, "draws": draws, "statements": 0,
               "gpu_s": 0.0, "lower_cpu_s": 0.0, "lower_peak_GiB": 0.0, "lower_peak_GiB_today": 0.0,
               "lower_cpu_s_today": 0.0, "stage_cpu_s": 0.0, "verify_cpu_s": 0.0, "blocks": []}
        for b in blocks:
            ands, nb, k_log, cap = b["ands"], b["nb"], b["k_log"], b["cap"]
            where = u["name"] if len(blocks) == 1 else f"{u['name']}, {b['name']}"
            if k_log > k_cap:
                out["infeasible"].append(f"{where}: {ands:.3g} ANDs needs k_log {k_log} > {k_cap} (compose "
                                         f"and the prover refuse it)")
            mem = 2.0 ** k_log * I("prover_bytes_per_row") / 1e9
            out["prover_peak_GB"] = max(out["prover_peak_GB"], mem)
            if mem > I("prover_mem_GB"):
                out["infeasible"].append(f"{where}: prover memory {mem:.3g} GB at 2^{k_log} rows > "
                                         f"{I('prover_mem_GB')} GB")
            grow = 1 + I("cap_cost_per_bit") * max(0, k_log - I("k_max"))
            per = o * ands * grow * rep
            if universal:
                ns, padded = sessions(pooled[k_log], cap, f0 + f1 * ands, per)
                share = nb / pooled[k_log] if nb > 0 else 0.0
                stmts, padded = ns * share, padded * share
            else:
                stmts, padded = sessions(nb, cap, f0 + f1 * ands, per)
            staged = nb if pv["shared"] else min(n * b["count"], stmts * cap)
            gpu = (padded * per + staged * so * ands + stmts * (f0 + f1 * ands)) * price / speed
            gates = b.get("gates", u.get("gates") if len(blocks) == 1 else None) or ands * gpa
            lowerings = (1 if nb > 0 else 0) if lo["per"] == "template" else nb
            peak = gates * bpg / 2 ** 30
            if peak > lo["ram_gib"]:
                out["infeasible"].append(f"{where}: lowering needs {peak:.3g} GiB > {lo['ram_gib']} GiB")
            part = {"statements": stmts, "gpu_s": gpu, "lower_cpu_s": lowerings * gates * spg,
                    "lower_cpu_s_today": lowerings * ands * I("lower_s_per_and_today"),
                    "stage_cpu_s": stmts * (I("stage_fixed") + I("stage_per_and") * ands)
                    + staged * I("stage_per_and_staged") * ands,
                    "verify_cpu_s": stmts * (vf + va * ands) * grow}
            for k, v in part.items():
                cls[k] += v
            cls["lower_peak_GiB"] = max(cls["lower_peak_GiB"], peak)
            cls["lower_peak_GiB_today"] = max(cls["lower_peak_GiB_today"],
                                              ands * I("lower_bytes_per_and_today") / 2 ** 30)
            cls["blocks"].append({"name": b["name"], "count": b["count"], "ands": ands, "k_log": k_log, "cap": cap,
                                  "padded_draws": padded, **part})
        group = sum(b["count"] * b["ands"] for b in blocks)
        cls.update({"ands_with_openings": group, "openings_share": 1 - (u["work"] + u["extra_ands"]) / group,
                    "k_log": max(b["k_log"] for b in cls["blocks"]), "cap": min(b["cap"] for b in cls["blocks"])})
        out["classes"].append(cls)
        out["draws"] += draws
        for k in ("gpu_s", "statements", "lower_cpu_s", "stage_cpu_s", "verify_cpu_s"):
            out[k] += cls[k]
        out["lower_peak_GiB"] = max(out["lower_peak_GiB"], cls["lower_peak_GiB"])
    out["lower_peak_GiB_today"] = max((c["lower_peak_GiB_today"] for c in out["classes"]), default=0.0)
    out["draws_bound"] = lam / s["eps"]
    per_window = ("units", "draws", "statements", "gpu_s", "lower_cpu_s", "lower_cpu_s_today", "stage_cpu_s",
                  "verify_cpu_s", "padded_draws")
    for c in out["classes"]:
        for b in c["blocks"]:
            for k in per_window:
                if k in b:
                    b[k] /= P
        for k in per_window:
            if k in c:
                c[k] /= P
    for k in ("F_ands", "C_ands", "draws", "gpu_s", "statements", "lower_cpu_s", "stage_cpu_s", "verify_cpu_s",
              "draws_compute", "draws_bound"):
        out[k] /= P
    return out


def sessions(k: float, cap: int, fixed: float, per: float) -> tuple[float, float]:
    """C-Flock's sessions for k draws of one block class (GD-PROVER, note:compute-research/20261028T1955Z-report-gd-
    prover): a session pads its draws to P = max(8, 2^ceil(log2 draws)), at most cap, paying `fixed` a session and
    `per` a padded draw. The cheapest split into such sessions: (sessions, padded draws)."""
    k = math.ceil(k - 1e-9)
    if k <= 0:
        return 0.0, 0.0
    full, r = divmod(k, cap)

    def best(r: int) -> tuple[int, int]:
        if r <= 0:
            return 0, 0
        one = (1, min(cap, max(8, 1 << (r - 1).bit_length())))
        lo = 1 << (r.bit_length() - 1)
        if lo < 8 or lo == r:
            return one
        s2, p2 = best(r - lo)
        return min(one, (1 + s2, lo + p2), key=lambda x: x[0] * fixed + x[1] * per)

    s_, p_ = best(r)
    return float(full + s_), float(full * cap + p_)


# ---------------------------------------------------------------------------------------------------------------------
# J
# ---------------------------------------------------------------------------------------------------------------------

def _step(d: dict, units: list) -> tuple:
    """One engine step of d: (stock parts, t_stock, its anchor, remote parts, Commit info, PoUW info)."""
    stock = stock_parts(d)
    if d["price"]["baseline"] in ("main", "fastest"):
        t_stock = sum(stock_parts(normalize({**d, "point": {**d["point"], "async": True}})).values())
        anchor = f"async stock at the design's point (2cd6's stack on next, {C2CD6})"
        if d["price"]["baseline"] == "main":
            t_stock *= I("main_async_vs_2cd6")
            anchor = (f"main's async stock at the design's point (stock-async-n2048, {MAIN_ASYNC}: 2cd6's modelled "
                      f"async stock x {I('main_async_vs_2cd6'):.3f})")
    elif d["price"]["baseline"] == "same":
        t_stock = sum(stock.values())
        anchor = (f"async stock (2cd6's stack, {C2CD6})" if d["point"]["async"] else
                  f"sync stock (b57f's stack, {B57F})") + " at the design's point"
    else:
        raise ValueError(f"price.baseline {d['price']['baseline']!r}")
    remote = dict(stock)
    if d["forward"]["kernels"] == "commit_taps" and not d["commit"]["on"]:
        remote["the Commit's taps"] = I("taps_ms")
    cparts, cinfo = commit_parts(d, stock, units)
    wparts, winfo = pouw_parts(d, stock)
    remote.update(cparts)
    remote.update(wparts)
    where = d["sampling"]["retain"]
    if where not in ("device", "host"):
        raise ValueError(f"sampling.retain {where!r}")
    if where == "host":
        step_bytes = (cinfo.get("step_bytes", 0.0) + winfo.get("value_bytes", 0.0)
                      + winfo.get("records_GB", 0.0) * 1e9)
        copy = step_bytes / (I("d2h_GBs") * 1e9) * 1e3
        t = sum(remote.values())
        remote["retain: D2H to the host, beyond the step and the DMA's HBM share"] = (
            max(0.0, copy - t) + I("dma_hbm_cost") * min(copy, t))
        cinfo = {**cinfo, "d2h_ms_a_step": copy}
    return stock, t_stock, anchor, remote, cinfo, winfo


def J(dsg: dict | None = None) -> dict:
    """The cost of a design: serve-overhead parts (ms a step), proving (GPU-s and CPU-s a window), verification
    (CPU-s a window), the draws, and J = (t_remote / t_stock - 1) + proving GPU-s / stock GPU-s a window."""
    d = normalize(dsg or {})
    units = d["partition"]["units"] or preset_units(d)
    stock, t_stock, anchor, remote, cinfo, winfo = _step(d, units)
    t_remote = sum(remote.values())
    p = d["point"]
    S, nd = p["steps_per_window"], p["decode_steps"]
    win_stock_ms, win_remote_ms = S * t_stock, S * t_remote
    retained = S * (cinfo.get("step_bytes", 0.0) + winfo.get("value_bytes", 0.0) + winfo.get("records_GB", 0.0) * 1e9)
    decode = None
    if nd:
        dd = normalize({**d, "point": {**p, "real_tokens": p["seqs"], "steps_per_window": 1, "decode_steps": 0}})
        dstock, dt_stock, _, dremote, dcinfo, dwinfo = _step(dd, units)
        dremote.pop("seal: rest of the host gap", None)        # the seal's host work is once a window
        decode = {"t_stock_ms": dt_stock, "t_remote_ms": sum(dremote.values()),
                  "serve_overhead_ms": {k: v for k, v in dremote.items() if dstock.get(k) != v}}
        win_stock_ms += nd * dt_stock
        win_remote_ms += nd * decode["t_remote_ms"]
        retained += nd * (dcinfo.get("step_bytes", 0.0) + dwinfo.get("value_bytes", 0.0)
                          + dwinfo.get("records_GB", 0.0) * 1e9)
    retained /= 1e9
    stock_gpu_s = win_stock_ms / 1e3
    values = proving(d, units, "values (sampled proofs)")
    pouw = proving(d, pouw_units(d), "PoUW")
    price = d["price"]["cpu_s_in_gpu_s"] if d["price"]["cpu_s_in_gpu_s"] is not None else I("cpu_s_in_gpu_s")
    gpu_s = values["gpu_s"] + pouw["gpu_s"]
    cpu_s = values["lower_cpu_s"] + values["stage_cpu_s"] + pouw["lower_cpu_s"] + pouw["stage_cpu_s"]
    proving_gpu_s = gpu_s + (cpu_s * price if d["price"]["cpu_in_J"] else 0.0)
    verify_cpu_s = values["verify_cpu_s"] + pouw["verify_cpu_s"]
    serve_term = win_remote_ms / win_stock_ms - 1
    prove_term = proving_gpu_s / stock_gpu_s
    infeasible = values["infeasible"] + pouw["infeasible"]
    retain_GB = _get(d, "sampling", "retain_GB", "retain_GB" if d["sampling"]["retain"] == "device"
                     else "host_retain_GB")
    if retained > retain_GB:
        infeasible.append(f"a window retains {retained:,.1f} GB for its draw > {retain_GB} GB on the "
                          f"{d['sampling']['retain']}")
    if d["pouw"]["scheme"] == "dither" and 1 < d["pouw"]["agg"] < 8:
        infeasible.append(f"pouw.agg {d['pouw']['agg']}: an aggregate word under 8 cells loses the floor to the "
                          f"attacks GD-AGG measured (safe from 8 with mix4)")
    cr = pouw_credit(d)
    linears_gpu_s = INT8_MACS_PER_TOKEN * window_tokens(d) / (theta(I("gemm_mhz")) * I("gemm_eff_per_clock")) / 1e3
    forced_gpu_s = linears_gpu_s * cr["forced_share_at_best_attack"]
    return {
        "J": serve_term + prove_term,
        "serve_term": serve_term,
        "prove_term": prove_term,
        "credited_gpu_s": forced_gpu_s if cr["earns_credit"] else 0.0,
        "forced_gpu_s": forced_gpu_s,
        "t_stock_ms": t_stock,
        "t_stock_anchor": anchor,
        "window_class": d["point"]["window_class"],
        "t_remote_ms": t_remote,
        "t_stock_window_ms": win_stock_ms,
        "t_remote_window_ms": win_remote_ms,
        "decode_step": decode,
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
        "period": d["sampling"]["period"],
        "eps_of_a_window": min(1.0, d["sampling"]["eps"] * d["sampling"]["period"]),
        "soundness_bits": values["soundness_bits"],
        "retained_GB_per_window": retained,
        "infeasible": infeasible,
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
