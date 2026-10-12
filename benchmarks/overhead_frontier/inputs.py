"""Every input to the overhead frontier: its value, a realistic improvement, and the run that measured it.

A value is a number, a table by batch (``((m, value), ...)``, interpolated linearly in m and held flat outside), or a
dict by rounds k of either. ``improved`` is the value one realistic piece of work would bring (``lever`` names it);
an input without one is a fixed fact or a ruling. ``status`` is ``measured``, ``derived`` (arithmetic on measured
values), ``estimate`` (no measurement yet), ``assumption`` (a named assumption of the design) or ``policy``.

Units: bandwidths in GB/s (1e9 B/s), bytes in B, a cost "per pass" in native read passes over the bytes it covers (0.36
a weight pass is 0.36 × the time to read those weights once at R).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Input:
    value: Any
    unit: str
    source: str
    status: str
    improved: Any = None
    lever: str = ""


INPUTS: dict[str, Input] = {
    # Node 2 (vy-nebius-2): 8 RTX PRO 6000 (sm_120), 2 × Xeon 6776P, PCIe 5, no NVLink.
    "gpus": Input(8, "GPUs", "node 2", "measured"),
    "R": Input(1540.0, "GB/s per GPU", "ra-codec-tensor 17:58Z: the pass's read (plain attention reads 1,495)", "measured"),
    "H": Input(93.9e9, "B per GPU", "ra-codec-tensor: HBM stored, less the 2.6–2.8 GB of step buffers", "measured"),
    "phi": Input(370e12, "FLOP/s per GPU, FP8", "ra-word-codec: cuBLASLt's sm89 FP8 kernels at B160 (datasheet dense "
                 "935.6 TFLOPS at 2,430 MHz)", "measured", 600e12, "a native sm_120 FP8 GEMM"),
    "s_chip": Input(263.2e6, "B per GPU", "ra-pcie-exhaust: SMEM + L2 + registers (art:f3457bd3)", "measured"),
    "b_host": Input(372.0, "GB/s, node", "red team F28: 186 on one socket at 48 threads (r20261012-165009-4f56), "
                    "two sockets at that rate; 819 at DDR5 spec", "measured"),
    "b_zero_copy": Input(51.0, "GB/s per GPU", "ra-pcie-exhaust: zero-copy host read (H2D 56.4, r20261012-165405-1810)",
                         "measured"),
    "tau_host": Input(0.06, "of the step at b_zero_copy per GPU", "ra-pcie-exhaust: the HBM receipt pass 0.977–0.987 → "
                      "1.043–1.063 ms beside the 8-SM host fold (r20261012-114836-4a56, art:f3457bd3)", "measured",
                      0.03, "a host-tier receipt lighter than the 8-SM fold"),
    "b_pcie": Input(56.4, "GB/s per GPU", "red team: PCIe H2D per GPU, 451 total (r20261012-165405-1810)", "measured"),
    "b_nic": Input(50.0, "GB/s, node", "red team: one ConnectX VF at 400 Gb/s (r20261012-171910-14ba)", "measured"),
    "b_p2p": Input(25.0, "GB/s", "PCIe peer all-reduce bus bandwidth, no NVLink", "estimate"),
    "t_allreduce": Input(40e-6, "s per all-reduce", "NCCL small-message all-reduce over PCIe", "estimate"),
    # The adversary's rebuild (μ_rebuild).
    "x1": Input(175.0, "read-times per round", "ra-codec-tensor's p512 encode at the max clock "
                "(r20261012-095329-3aac ÷ 1.16), the red team's x₁", "measured"),
    "s_adv": Input(1.2, "×", "red team assumption 5: the adversary's squaring ≤ 1.2× the best measured (the multiply "
                   "pipe at 100%; 1.45 the issue bound)", "assumption", 1.0,
                   "a measured or proved squaring floor at 512 bits"),
    "h_host_sq": Input(0.003, "of the GPUs' squaring rate", "red team: host CPUs with GMP, 1.12e7 squarings/s/core × "
                       "128 (IFMA port bound 0.067)", "measured"),
    "mu_partial_k2": Input(0.0015, "of capacity", "red team: partial completion at 512 bits, k = 2 (margin 0.75× at "
                           "the GPU floor's low end, 0.12–0.17 points)", "estimate", 0.0,
                           "measure partial completion at 512 bits, k = 2"),
    # The honest prover's codec costs, per pass over the bytes they cover.
    "o_w": Input({2: ((8, 0.36), (32, 0.73)), 3: ((8, 0.97), (32, 1.00))}, "per weight pass",
                 "ra-codec-tensor 17:58Z: wt.cu decode at M=8 (70B, r20261012-175923-0080) and M=32 (8B, "
                 "r20261012-175841-5578)", "measured", {2: ((8, 0.20), (32, 0.20)), 3: ((8, 0.65), (32, 0.65))},
                 "port the KV path's warp-specialized mma-ws decode to the weight GEMM"),
    "tau_w": Input({2: ((8, 0.50), (32, 0.70)), 3: ((8, 0.22), (32, 0.71))}, "per weight pass, k_s = 16, n = 4 KiB",
                   "ra-codec-tensor 17:58Z: the sketch receipt with its commit in the p512 step (M=8 70B, M=32 8B; "
                   "at 70B k = 3 it hides partly under the decode)", "measured",
                   {2: ((8, 0.07), (32, 0.07)), 3: ((8, 0.07), (32, 0.07))},
                   "ra-sketch-tag's sketch warps and deferred commit (+3.6..4.6 + 0.4..2.6 points on an encoded "
                   "store, r20261012-131750-2a7a) in wt.cu"),
    "tau_kv": Input({2: ((8, 0.50), (32, 0.70)), 3: ((8, 0.22), (32, 0.71))}, "per KV pass, k_s = 16, n = 4 KiB",
                    "unmeasured: the KV is not receipted in any step yet; taken equal to the weights'", "estimate",
                    {2: ((8, 0.07), (32, 0.07)), 3: ((8, 0.07), (32, 0.07))},
                    "a sketch-warp receipt in the mma-ws attention kernel"),
    "ks_ratio": Input({8: 0.73, 12: 0.87, 16: 1.0, 32: 3.54}, "× the k_s = 16 receipt",
                      "ra-sketch-tag: encoded store M=1, +21.1 / +22.35 / +34.3% at k_s 8 / 16 / 32 over the codec's "
                      "17.65%", "measured"),
    "tau_fill": Input({8: 0.002, 12: 0.010, 16: 0.019, 32: 0.085}, "per filler pass by k_s",
                      "ra-sketch-tag: the sketch receipt on a store that is read and not decoded (plain store M=32, "
                      "r20261012-133311-b8e1)", "measured"),
    "tau_bind": Input(0.03, "per pass", "binding z_{j−1} into the sketch (a ±1 fold of z, k_s/32 ops a byte); "
                      "ra-bound-receipt's question", "estimate", 0.01, "fold z into the sign matrix once a tile"),
    "o_kv": Input({2: ((32, 0.225), (40, 0.177), (80, 0.162), (160, 0.159)),
                   3: ((32, 0.761), (40, 0.670), (80, 0.651), (160, 0.649))}, "per KV pass",
                  "ra-word-codec mma-ws (r20261012-172911-1184, art:47c171ef) at B40–160; ra-codec-tensor at B32 "
                  "ctx 20k (r20261012-175841-5578)", "measured",
                  {2: ((32, 0.087), (160, 0.087)), 3: ((32, 0.534), (160, 0.534))},
                  "a persistent mma-ws attention kernel (the 16-warp floor, 0.087 / 0.534)"),
    "x1_h": Input(240.0, "read-times per round", "the honest KV append: ra-word-codec 226 (3.39 GB/s at k = 2), "
                  "ra-codec-tensor's implied 232–257", "measured"),
    "beta": Input(0.5, "of tensor-bound time", "ra-word-codec: about 52% of the KV encode hides beside the FP8 GEMM "
                  "at B80–160", "measured"),
    "e_attn": Input(1540.0 / 1495.0, "reads", "ra-codec-tensor: plain attention at 1,495 of 1,540 GB/s", "measured"),
    "e_mem": Input(((1, 1.05), (32, 1.25)), "weight reads by M", "ra-codec-tensor: the plain step 1.05–1.08 passes; "
                   "native linears at M=32 read the weights 1.25 times", "measured"),
    "rho_l2": Input(0.45, "of activation traffic", "calibrated to stock 70B's uncertified share (ra-vllm-encoded "
                    "r20261012-104353-3dc6, art:6dd5489e)", "estimate", 0.9,
                    "evict-first hints on the weight and KV streams, so activations stay in L2"),
    # The verifier's sample (μ_sample) and the counting slack (λ).
    "v_budget": Input(2.0, "Gb/s per node", "the verifier's download for openings; s ≈ 3 × 10^4 a window at 4 KiB "
                      "(the design's μ_sample ≈ 0.1%)", "policy", 10.0, "a 10 Gb/s verifier link"),
    "t_window": Input(1.0, "s", "the window", "policy"),
    "log2_inv_delta": Input(40.0, "bits", "δ = 2^-40 (ra-bandwidth-lean)", "policy"),
    "log2_q": Input(64.0, "bits", "q ≤ 2^64 oracle queries (ra-bandwidth-protocol 4c3352875: λ = 64k + 41)", "policy"),
    "x_from_transcript": Input(False, "switch", "the verifier takes z_{j−1} for an opened block from PoUW's "
                               "transcript instead of with the opening", "estimate", True,
                               "read z from PoUW's transcript"),
    "receipt_bits": Input(512, "bits", "a SHA-512 receipt", "policy"),
    "eta_reserve": Input(0.0, "of capacity", "the honest read shortfall under cap: unmeasured (plain attention alone "
                         "reads at 97.1% of the pass's rate; top-up beside it may close the gap)", "estimate"),
}

# Values an input may take that make the frontier worse, each with why it is plausible.
RISKS: dict[str, tuple[Any, str]] = {
    "eta_reserve": (0.0025, "the honest prover reads at 99.75% of the adversary's best stream"),
    "b_host": (819.0, "the host DRAM at DDR5-6400 spec (no trusted hardware: the adversary may provision it)"),
    "s_adv": (1.45, "the adversary's squaring at the issue bound"),
    "rho_l2": (0.0, "every activation byte goes to HBM"),
    "v_budget": (1.0, "a 1 Gb/s verifier link"),
}


def values(inputs: dict[str, Input] = INPUTS) -> dict[str, Any]:
    return {name: i.value for name, i in inputs.items()}


def improved(name: str, inputs: dict[str, Input] = INPUTS) -> dict[str, Input]:
    out = dict(inputs)
    out[name] = replace(inputs[name], value=inputs[name].improved)
    return out


def all_improved(inputs: dict[str, Input] = INPUTS) -> dict[str, Input]:
    return {n: (replace(i, value=i.improved) if i.improved is not None else i) for n, i in inputs.items()}


def levers(inputs: dict[str, Input] = INPUTS) -> list[str]:
    return [n for n, i in inputs.items() if i.improved is not None]
