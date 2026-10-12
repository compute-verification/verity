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
    "b_host": Input(819.0, "GB/s, node", "the adversary's host DRAM read at DDR5-6400 spec (red team F28): half of "
                    "socket 0 already reads 266 with two GPUs' copy engines and still rises with cores (ra-host-tier "
                    "r20261012-183019-a547), so the spec is the bound until the whole-node run", "derived"),
    "b_zero_copy": Input(51.0, "GB/s per GPU", "ra-pcie-exhaust: zero-copy host read (H2D 56.4, r20261012-165405-1810)",
                         "measured"),
    "tau_host": Input(0.06, "of the step at b_zero_copy per GPU", "ra-pcie-exhaust: the HBM receipt pass 0.977–0.987 → "
                      "1.043–1.063 ms beside the 8-SM host fold (r20261012-114836-4a56, art:f3457bd3)", "measured",
                      0.03, "a host-tier receipt lighter than the 8-SM fold"),
    "b_pcie": Input(56.4, "GB/s per GPU", "red team: PCIe H2D per GPU, 451 total (r20261012-165405-1810)", "measured"),
    "b_pcie_down": Input(25.6, "GB/s per GPU", "ra-host-tier: D2H beside a zero-copy read stream, 25.2–25.7 "
                         "(r20261012-183019-a547; 57.3 alone), the PCIe-down F30 has for z", "measured"),
    "host_cpu": Input(0.89, "of the host DRAM ceiling", "ra-host-tier v0: the honest loop (GPU zero-copy receipts + 44 "
                      "CPU threads' receipts under the latest z) reads 228–229 GB/s against the adversary's 253–256 on "
                      "the same CPUs and GPU, half a socket (r20261012-181656-d1cc); 0.89 with CPUs alone too; the node "
                      "taken at the same ratio", "measured", 0.98,
                      "ra-host-tier v1: CPU receipts four streams wide (10.4 against 8.2 GB/s a thread on the agent "
                      "VM; μ_host ≤ 0.1% needs 0.985–0.989), estimate"),
    "b_nic": Input(100.0, "GB/s, node", "ra-link-pricing: one ConnectX VF at 400 Gb/s each way, and inbound bytes "
                   "into a small GPU or host ring land in neither counted tier, so both directions count "
                   "(r20261012-183152-39a5, art:d1040f91)", "measured"),
    "s_hostcache": Input(0.906e9, "B, node", "ra-link-pricing §4: the host's L3 672 MiB + L2 192 MiB, a store outside "
                         "both tiers, read back at 51–57 GB/s a GPU (r20261012-181459-46ae)", "derived"),
    "b_p2p": Input(25.0, "GB/s", "PCIe peer all-reduce bus bandwidth, no NVLink", "estimate"),
    "t_allreduce": Input(40e-6, "s per all-reduce", "NCCL small-message all-reduce over PCIe", "estimate"),
    # The adversary's rebuild (μ_rebuild).
    "x1": Input(175.0, "read-times per round", "ra-codec-tensor's p512 encode at the max clock "
                "(r20261012-095329-3aac ÷ 1.16), the red team's x₁", "measured"),
    "s_adv": Input(1.38, "× 5.13e10/s, clock-matched", "assumption 5 as ra-squaring-ceiling's final restates it: the "
                   "column form's ceiling at the measured instruction mix (a fit over 11 forms, 1.34–1.39); FOLD 5 "
                   "measures 1.27 (6.53e10/s, r20261012-182613-5df1, r20261012-183030-b044, art:38bdc4ec), so no "
                   "measurement can lower it", "assumption"),
    "h_host_sq": Input(0.129, "of the GPUs' squaring rate", "assumption 6 as ra-squaring-ceiling's final restates it "
                       "(h ≤ 13%, the IFMA-only bound); AVX-512 IFMA measures 2.84e8 squarings/s a physical core × "
                       "128, h = 0.076 (r20261012-183544-2d44)", "assumption"),
    "theta": Input(0.75, "of the per-bit claim", "red team: partial completion's cost at the 4D reduction's floor on "
                   "the GPU at 512 bits, k = 2, low end (centre 1.2, range 0.75–2.7, r20261012-170958-f49a); "
                   "ra-partial-lean's WorkHardness at θ·c_root divides the rebuild term by θ. k = 3 taken at k = 2's "
                   "θ (its completion window is narrower; unpriced)", "estimate", 1.0,
                   "price assumption 7 at or above the per-bit claim (the floor's centre is 1.2; no agent may "
                   "measure it)"),
    "bits": Input(512, "bits", "the codec's word, p = 2^512 − 569 (ruling); `frontier.widths` sweeps it", "policy"),
    "masked": Input(False, "switch", "plain Sloth rounds, the ruling's codec; `frontier` also runs ra-codec-mix's "
                    "σ_K at k = 2 (``k2m``)", "policy"),
    "mask_cost": Input({2: 0.032, 3: 0.059}, "per decoded pass", "ra-codec-mix: σ_K, a public XOR between rounds, "
                       "fast path in mma-ws attention, +0.192 / +0.709 against +0.160 / +0.650 a KV byte at k = 2 / 3 "
                       "(r20261012-182332-24e8, art:9c3b9e2724d6ca52); the weights taken at the same cost a byte",
                       "measured"),
    "theta_masked": Input(1.0, "of the per-bit claim", "ra-codec-mix: with σ_K the red team's 4D lattice completes "
                          "0/200 words at every width (200/200 plain); a completion must compute the inner root, "
                          "≥ 1.515 at the floor and ≥ 1.0 with Coppersmith to b/2 for free (art:ffca24ef); assumption "
                          "7 becomes 'no completion skips the inner root'", "estimate"),
    "sq_speed": Input(1.0, "× kv.cu's squaring", "the honest kernels' squaring: kv.cu's carry chains (FOLD 1 in "
                      "ra-squaring-ceiling's harness, 5.71e10/s)", "measured", 1.14,
                      "FOLD 5's column-chained squaring (6.53e10/s, ra-squaring-ceiling, art:38bdc4ec) in the decode "
                      "and encode kernels"),
    "decode_floor": Input({2: 0.052, 3: 0.503}, "per pass at 512 bits", "ra-word-codec: the standalone "
                          "read-and-decode at 24 warps, c·k − 1 with c = 0.526 / 0.501 (r20261012-172911-1184); "
                          "below it a form exposes a share of c·k (mma-ws: 0.10)", "measured"),
    # The honest prover's codec costs, per pass over the bytes they cover.
    "o_w": Input({2: ((8, 0.36), (32, 0.73)), 3: ((8, 0.97), (32, 1.00))}, "per weight pass",
                 "ra-codec-tensor 17:58Z: wt.cu decode at M=8 (70B, r20261012-175923-0080) and M=32 (8B, "
                 "r20261012-175841-5578)", "measured", {2: ((8, 0.20), (32, 0.20)), 3: ((8, 0.65), (32, 0.65))},
                 "port the KV path's warp-specialized mma-ws decode to the weight GEMM"),
    "tau_w": Input({2: ((8, 0.232), (32, 0.70)), 3: ((8, 0.22), (32, 0.71))}, "per weight pass, k_s = 16, n = 4 KiB",
                   "ra-codec-tensor: the sketch receipt with its commit in the p512 step; k = 2 at M=8 (70B) re-keyed "
                   "every pass (F29) on sketch warps of its own, 19:25Z (r20261012-190554-dd7c); the rest 17:58Z, "
                   "before the sketch warps (M=32 8B; at 70B k = 3 it hides partly under the decode)", "measured",
                   {2: ((8, 0.07), (32, 0.07)), 3: ((8, 0.07), (32, 0.07))},
                   "ra-sketch-tag's sketch warps and deferred commit (+3.6..4.6 + 0.4..2.6 points on an encoded "
                   "store, r20261012-131750-2a7a) in wt.cu"),
    "tau_kv": Input({2: 0.159, 3: 0.158}, "per KV pass, k_s = 16",
                    "ra-word-codec 19:10Z: an unbound sketch-then-hash receipt on every 512-token KV block inside "
                    "mma-ws attention, +0.319 / +0.808 against +0.160 / +0.650 unreceipted at k = 2 / 3 "
                    "(r20261012-185147-b5db, art:830335f2)", "measured", {2: 0.07, 3: 0.07},
                    "the sketch on warps of its own and the hash out of the decoders' issue slots, as wt.cu's"),
    "ks_ratio": Input({8: 0.73, 12: 0.87, 16: 1.0, 32: 3.54}, "× the k_s = 16 receipt",
                      "ra-sketch-tag: encoded store M=1, +21.1 / +22.35 / +34.3% at k_s 8 / 16 / 32 over the codec's "
                      "17.65%", "measured"),
    "tau_fill": Input({8: 0.002, 12: 0.010, 16: 0.019, 32: 0.085}, "per filler pass by k_s",
                      "ra-sketch-tag: the sketch receipt on a store that is read and not decoded (plain store M=32, "
                      "r20261012-133311-b8e1)", "measured"),
    "tau_bind": Input(0.05, "per weight pass", "ra-bound-receipt (ro-bound-sketch-blake3/v0, B = R ⊕ M): binding adds "
                      "11–28% of the unbound receipt's own instructions, a static SASS count, not a timing; × τ_W ≈ "
                      "0.23", "estimate", 0.02, "the bound sketch on wt.cu's sketch warps (the R ⊕ M functor, "
                      "ra-codec-tensor's SkB interface)"),
    "tau_bind_kv": Input({2: 0.069, 3: 0.063}, "per KV pass", "ra-word-codec: the receipt bound to attention's prior "
                         "PV output (ro-bound-sketch/v0), +0.388 / +0.871 against the unbound +0.319 / +0.808 "
                         "(r20261012-185147-b5db)", "measured", {2: 0.03, 3: 0.03},
                         "the bound sketch on sketch warps of its own"),
    "rho": Input(0.5, "of min(H(z), 8|E|)", "ra-bound-receipt §2: a relay that makes one bound receipt exchanges at "
                 "least ½·min(H(z), 8|E|) bits (Lindsey's lemma over F₂₅₇)", "derived"),
    "z_entropy": Input(0.75, "distinct z bytes over |z|", "ra-bound-receipt §4: 70B's layer, k-slice binding caps at "
                       "0.75 (each 512-column chunk's first word has no earlier accumulator)", "derived"),
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
    # The step's ephemeral DRAM traffic, in bytes per layer × hidden: (a fixed part, a part per sequence in the batch).
    "eph_read": Input({"8B": (106.0, 9.4), "70B": (54.0, 17.9), "405B": (106.0, 17.9)}, "B per layer·hidden",
                      "ra-ephemeral: Nsight DRAM reads beyond the linears, the lm_head and the live KV, stock vLLM, "
                      "full depth (8B r20261012-181016-6cbb, 70B r20261012-181715-5045; fit within ~10% over B1–64 × "
                      "2k–32k); 405B the larger of each", "measured",
                      {"8B": (0.0, 9.4), "70B": (0.0, 17.9), "405B": (0.0, 17.9)},
                      "the linears reduce on chip: no split-K or stream-K workspace read back"),
    "eph_write": Input({"8B": (754.0, 47.2), "70B": (692.0, 43.8), "405B": (754.0, 47.2)}, "B per layer·hidden",
                       "ra-ephemeral: Nsight DRAM writes (GEMM workspaces, split-KV partials, the activations' "
                       "write-backs), the same runs, scaled to full depth; 1.3–2.8% of 8B's reads", "measured",
                       {"8B": (0.0, 11.8), "70B": (0.0, 11.0), "405B": (0.0, 11.8)},
                       "no workspaces at decode, and evict-first on the weight and KV streams so the activations' "
                       "lines are overwritten in L2 instead of written back (a quarter of them still are: estimate)"),
    "write_bus": Input(1.0, "read-times per written byte", "a DRAM write holds the bus as long as a read (turnaround "
                       "not counted)", "estimate"),
    # The verifier's sample (μ_sample) and the counting slack (λ).
    "v_budget": Input(2.0, "Gb/s per node", "the verifier's download for openings; s ≈ 3 × 10^4 a window at 4 KiB "
                      "(the design's μ_sample ≈ 0.1%)", "policy", 10.0, "a 10 Gb/s verifier link"),
    "t_window": Input(1.0, "s", "the window", "policy"),
    "opening_bytes": Input(1850.0, "B an opening beyond the block and z",
                           "ra-openings §3: one multiproof per tree, 5,959 B at 4 KiB and s = 18,470 (store paths 793, "
                           "receipt paths 1,006, the receipt 64); 4.0 KB at 2 KiB, 9.8 KB at 8 KiB", "derived"),
    "lambda_named": Input(False, "switch", "λ = (log₂q + log₂e − log₂N)/(8n), the proved theorem with C(q, k) ≤ "
                          "(eq/k)^k (ra-openings §2; a change to audit.bound, nothing in Lean)", "derived", True,
                          "name the certified blocks in the compression (ra-openings' proposal, λ ≈ H₂(γ)/(8n); "
                          "needs a Lean statement)"),
    "horizon": Input(1, "windows", "a claim per 1-s window", "policy", 60,
                     "a claim averaged over a minute (ra-openings' causal horizon bound; a window keeps its ≥ 2,761 "
                     "openings, so each is still ≤ 1% alone; needs its Lean statement and Daniel's ruling)"),
    "log2_inv_delta": Input(40.0, "bits", "δ = 2^-40 (ra-bandwidth-lean)", "policy"),
    "log2_q": Input(64.0, "bits", "q ≤ 2^64 oracle queries (ra-bandwidth-protocol 4c3352875: λ = 64k + 41)", "policy"),
    "x_from_transcript": Input(False, "switch", "the verifier takes z_{j−1} for an opened block from PoUW's "
                               "transcript instead of with the opening", "estimate", True,
                               "read z from PoUW's transcript"),
    "receipt_bits": Input(512, "bits", "a SHA-512 receipt", "policy"),
    "eta_reserve": Input(0.0, "of capacity", "the audited engine's read shortfall under cap, unmeasured: the frontier "
                         "reports what γ leaves for it. Stock vLLM 8B leaves η = 0.30–0.42 of R·T on two bursty "
                         "traces, 0.27–0.34 with idle filler (ra-kv-filler 19:20Z): decode at 0.75–0.80 R, prefill, "
                         "host gaps; ≤ 1% needs every phase to read at R itself", "estimate"),
    "mu_cpu": Input(0.0, "of capacity", "the host CPUs' tile work (Daniel 18:55Z: a capacity to exhaust or count). "
                    "Under C they are counted: their squarings in μ_rebuild (h), their DRAM in ΣR, their caches as "
                    "per-pass state, and tile work buys nothing without bytes from off the node, which C excludes; "
                    "ra-host-cpu, which would price it, is stopped by the usage filter", "policy"),
    # Chained passes (μ_timing): a pass is one step's plan, and its seed needs the previous pass's root.
    "t_gap": Input(100e-6, "s per pass", "ra-remote-driver: the root's reduction (an all-gather of the GPUs' subtree "
                   "roots, one SHA-512, a broadcast), 25–100 µs, unmeasured", "estimate"),
    "prefetch": Input(False, "switch", "the GPUs wait for the seed", "estimate", True,
                      "prefetch the next pass's first blocks on chip across the gap (ra-remote-driver: 0.52% → "
                      "0.04% at 20 ms passes), hiding up to S_chip/R"),
    "rtt": Input(0.020, "s", "ra-remote-driver's verifier round trip", "policy"),
    "n_session": Input(100, "windows", "ra-remote-driver's session of chained windows", "policy", 1000,
                       "sessions of 1,000 chained windows (17 minutes), so a padded pass's tail costs a tenth"),
    "p_over_o": Input(0.0, "prompt tokens per output token", "the frontier's KV is written by decode: a step encodes "
                      "only the token it appends", "policy"),
}

# The floor: each lever past its realistic improvement to the best any measurement or bound supports, so that the
# frontier at the floor bounds how fast the audit can get on this node with this codec.
FLOOR: dict[str, tuple[Any, str]] = {
    "o_w": ({2: ((8, 0.052), (32, 0.052)), 3: ((8, 0.505), (32, 0.505))},
            "the mma-ws decode floor with 24 warps (ra-word-codec r20261012-172911-1184) on the weights too"),
    "o_kv": ({2: ((32, 0.052), (160, 0.052)), 3: ((32, 0.505), (160, 0.505))}, "the same floor on the KV"),
    "tau_w": ({2: ((8, 0.019), (32, 0.019)), 3: ((8, 0.019), (32, 0.019))},
              "the sketch's cost on a store that is only read (ra-sketch-tag plain store M=32, +1.7..1.9%)"),
    "tau_kv": ({2: ((8, 0.019), (32, 0.019)), 3: ((8, 0.019), (32, 0.019))}, "the same on the KV"),
    "tau_bind": (0.0, "binding folded into the sign matrix for free"),
    "tau_bind_kv": ({2: 0.0, 3: 0.0}, "the same in attention"),
    "tau_host": (0.0, "the host tier's stream costs the HBM stream nothing"),
    "eph_read": ({"8B": (0.0, 0.0), "70B": (0.0, 0.0), "405B": (0.0, 0.0)}, "no ephemeral byte read from DRAM"),
    "eph_write": ({"8B": (0.0, 0.0), "70B": (0.0, 0.0), "405B": (0.0, 0.0)}, "no ephemeral byte written to DRAM"),
    "host_cpu": (1.0, "the honest loop reads host DRAM at the adversary's ceiling"),
    "v_budget": (10.0, "a 10 Gb/s verifier link"),
    "x_from_transcript": (True, "z from PoUW's transcript"),
    "lambda_named": (True, "named λ"),
    "horizon": (600, "a claim averaged over 10 minutes"),
}

# Values an input may take that make the frontier worse, each with why it is plausible.
RISKS: dict[str, tuple[Any, str]] = {
    "eta_reserve": (0.0025, "the honest prover reads at 99.75% of the adversary's best stream"),
    "b_host": (1024.0, "ra-host-tier's four halves of a socket at 256 GB/s (above DDR5-6400 spec, so only with faster "
                       "DIMMs the adversary may provision)"),
    "s_adv": (1.61, "the column form with every add free (ra-squaring-ceiling)"),
    "mu_cpu": (0.005, "the host CPUs' tile work counted against this audit at ra-pouw-zero's low end (a dual-socket "
                      "AMX host is 0.5–2% of node HBM, extrapolated and untuned, r20261012-181859-f096)"),
    "write_bus": (1.5, "a write costs half a read more in bus turnaround"),
    "v_budget": (1.0, "a 1 Gb/s verifier link"),
    "t_gap": (500e-6, "a root reduction of 500 µs (ra-remote-driver's worst row)"),
    "p_over_o": (8.0, "prompts 8× the output: each step also encodes 8 prompt tokens a sequence (ra-kv-filler: the "
                      "prompt encode is +17% / +25% of busy time at k = 2 / 3 on a vLLM trace)"),
}


def values(inputs: dict[str, Input] = INPUTS) -> dict[str, Any]:
    return {name: i.value for name, i in inputs.items()}


def improved(name: str, inputs: dict[str, Input] = INPUTS) -> dict[str, Input]:
    out = dict(inputs)
    out[name] = replace(inputs[name], value=inputs[name].improved)
    return out


def all_improved(inputs: dict[str, Input] = INPUTS) -> dict[str, Input]:
    return {n: (replace(i, value=i.improved) if i.improved is not None else i) for n, i in inputs.items()}


def at_floor(inputs: dict[str, Input] = INPUTS) -> dict[str, Input]:
    out = all_improved(inputs)
    for name, (value, _) in FLOOR.items():
        out[name] = replace(inputs[name], value=value)
    return out


def levers(inputs: dict[str, Input] = INPUTS) -> list[str]:
    return [n for n, i in inputs.items() if i.improved is not None]
