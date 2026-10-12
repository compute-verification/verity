"""Every input to the overhead frontier: its value, a realistic improvement, and the run that measured it.

A value is a number, a table by batch or by word width (``((m, value), ...)``, interpolated linearly and held flat
outside), or a dict by rounds k of either. ``improved`` is the value one realistic piece of work would bring (``lever`` names it);
an input without one is a fixed fact or a ruling. ``status`` is ``measured``, ``derived`` (arithmetic on measured
values), ``estimate`` (no measurement yet), ``assumption`` (a named assumption of the design), ``proved`` (a
guarantee in Lean) or ``policy``.

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
    "capacity": Input("hbm", "tier", "Daniel 21:00Z: the remote guarantee is Conservation alone, and the declared "
                      "capacity HBM alone, so the host tier, μ_link and the identity terms leave γ", "policy"),
    "b_p2p": Input(25.0, "GB/s", "PCIe peer all-reduce bus bandwidth, no NVLink", "estimate"),
    "t_allreduce": Input(40e-6, "s per all-reduce", "NCCL small-message all-reduce over PCIe", "estimate"),
    # The adversary's rebuild (μ_rebuild), by word width: ra-squaring-ceiling's `wide.py` (cursor/ra-squaring-ceiling-e3cd
    # at 687c86282; STEERING 21:22Z).
    "x1": Input(((512, 175.0), (768, 338.6), (1024, 600.0)), "read-times per round",
                "the red team's 175 at p512 (ra-codec-tensor's encode at the max clock, r20261012-095329-3aac ÷ 1.16), "
                "scaled by kv.cu's root bytes a second at each width (`wide.x1_ref`; ra-codec-mix's kv-codec.json, "
                "r20261012-183859-2055, art:c1625f3cebececb0)", "measured"),
    "s_adv": Input(((512, 1.388), (768, 1.394), (1024, 1.420)), "× R_ref(b), clock-matched",
                   "assumption 5 per width (`wide.adversaries`' stated row): the best form less its MOV and SEL; the "
                   "best forms measure 1.268 / 1.260 / 1.309 (FOLD 5 at 512, r20261012-183030-b044; FOLD 6 at 768, "
                   "r20261012-194528-b277, and at 1,024, r20261012-194557-c72f), so no measurement can lower it",
                   "assumption"),
    "h_host_sq": Input(((512, 0.129), (768, 0.120), (1024, 0.125)), "of the GPUs' squaring rate",
                       "assumption 6 per width (h at the host's IFMA bound, `wide.h_of`); the host's IFMA measures "
                       "7.6% / 7.5% / 6.4% (r20261012-183544-2d44, r20261012-195420-b71d, r20261012-195829-3929)",
                       "assumption"),
    "theta": Input(0.75, "of the per-bit claim", "red team: partial completion's cost at the 4D reduction's floor on "
                   "the GPU at 512 bits, k = 2, low end (centre 1.2, range 0.75–2.7, r20261012-170958-f49a); "
                   "ra-partial-lean's WorkHardness at θ·c_root divides the rebuild term by θ. k = 3 taken at k = 2's "
                   "θ (its completion window is narrower; unpriced)", "estimate", 1.0,
                   "price assumption 7 at or above the per-bit claim (the floor's centre is 1.2; no agent may "
                   "measure it)"),
    "bits": Input(1024, "bits", "the stated codec's word, p = 2^1024 − 105 (Daniel 20:55Z: secure and slow first); "
                  "512 bits is a later upgrade, and `frontier.widths` sweeps 512–1,024", "policy"),
    "masked": Input(True, "switch", "σ_K between rounds, the stated codec (Daniel 20:55Z): masked p1024 at k = 3, "
                    "with k = 2 the fallback if k = 3's step exceeds about 4×", "policy"),
    "mask_cost": Input({2: 0.032, 3: 0.059}, "per decoded pass", "ra-codec-mix: σ_K, a public XOR between rounds, "
                       "fast path in mma-ws attention, +0.192 / +0.709 against +0.160 / +0.650 a KV byte at k = 2 / 3 "
                       "(r20261012-182332-24e8, art:9c3b9e2724d6ca52); the weights taken at the same cost a byte",
                       "measured"),
    "theta_masked": Input(1.0, "of the per-bit claim", "ra-codec-mix: with σ_K the red team's 4D lattice completes "
                          "0/200 words at every width (200/200 plain); a completion must compute the inner root, "
                          "≥ 1.515 at the floor and ≥ 1.0 with Coppersmith to b/2 for free (art:ffca24ef); assumption "
                          "7 becomes 'no completion skips the inner root'", "estimate"),
    "sq_speed": Input(1.0, "× kv.cu's squaring", "the honest kernels' squaring: kv.cu's carry chains", "measured",
                      ((512, 1.162), (768, 1.154), (1024, 1.226)),
                      "the best measured squaring forms (FOLD 5 at 512, FOLD 6 at 768 and 1,024: `wide.py`'s "
                      "\"× kv.cu\", r20261012-183030-b044, -194528-b277, -194557-c72f) in the decode and encode "
                      "kernels"),
    "dec_ratio": Input(((512, 1.0), (768, 1.33), (1024, 1.68)), "× p512's decode a byte at equal k",
                       "1,024: ra-codec-mix's masked attention at 8B B80, +0.989 / +1.829 a KV byte at k = 2 / 3 "
                       "(r20261012-190027-eaee, art:f940d8b9a4b6c43b), 1.684 / 1.678 through `decode` (kv.cu's "
                       "squaring alone gives 1.52 / 1.54); 768: kv.cu's 1.21 at the same 1.10 (a tensor-core layout's "
                       "projection, estimate); the weights taken at attention's ratio (no wide weight kernel yet)",
                       "measured"),
    "decode_floor": Input({2: 0.052, 3: 0.503}, "per pass at 512 bits", "ra-word-codec: the standalone "
                          "read-and-decode at 24 warps, c·k − 1 with c = 0.526 / 0.501 (r20261012-172911-1184); "
                          "below it a form exposes a share of c·k (mma-ws: 0.10)", "measured"),
    # The honest prover's codec costs, per pass over the bytes they cover.
    "o_w": Input({2: ((1, 0.344), (8, 0.300), (32, 0.384)), 3: ((1, 0.882), (8, 0.849), (32, 0.965))},
                 "per weight pass", "ra-codec-tensor 21:35Z (e55995f5f, hbm-step-points/v0): the decode in the step, "
                 "70B at M = 8 (r20261012-203305-7dfa) and M = 32 (r20261012-203315-5252, r20261012-212703-cb89), "
                 "8B at M = 1 (r20261012-212713-6524)", "measured", {2: ((8, 0.20), (32, 0.20)), 3: ((8, 0.64), (32, 0.64))},
                 "the fused kernel at its decode alone (1.20 / 1.64 reads at k = 2 / 3, ra-codec-tensor §2)"),
    "tau_w": Input({2: ((8, 0.083), (32, 0.134)), 3: ((8, 0.074), (32, 0.079))},
                   "per weight pass, k_s = 16, n = 4 KiB",
                   "ra-codec-tensor 21:35Z (e55995f5f): the receipt re-keyed every pass in the step, its SHA-512 commit "
                   "(0.030) included, 70B at M = 8 (r20261012-203305-7dfa) and M = 32 (r20261012-203315-5252); 8B at "
                   "M = 1 pays 0.17, per-pass costs that don't shrink with the weights (inferred)",
                   "measured",
                   {2: ((8, 0.083), (32, 0.083)), 3: ((8, 0.074), (32, 0.079))},
                   "a 21-warp kernel at M = 32 (ra-codec-tensor, estimate, not built); the commit stays SHA-512 "
                   "(21:45Z), so BLAKE3's 0.02 is not a lever"),
    "tau_kv": Input({2: 0.107, 3: 0.115}, "per KV pass, k_s = 16",
                    "ra-word-codec 21:35Z: an unbound sketch inside mma-ws attention, then two receipts a block of 8 "
                    "vectors each, +0.268 / +0.765 against +0.161 / +0.650 unreceipted at k = 2 / 3 "
                    "(r20261012-212502-016a, art:96819a51); hiding the hash beside the next layer's attention "
                    "measured no gain (r20261012-211626-0945)", "measured"),
    "ks_ratio": Input({8: 0.73, 12: 0.87, 16: 1.0, 32: 3.54}, "× the k_s = 16 receipt",
                      "ra-sketch-tag: encoded store M=1, +21.1 / +22.35 / +34.3% at k_s 8 / 16 / 32 over the codec's "
                      "17.65%", "measured"),
    "tau_fill": Input({8: 0.002, 12: 0.010, 16: 0.019, 32: 0.085}, "per filler pass by k_s",
                      "ra-sketch-tag: the sketch receipt on a store that is read and not decoded (plain store M=32, "
                      "r20261012-133311-b8e1)", "measured"),
    "bound": Input(False, "switch", "Daniel 21:00Z: binding receipts to z isn't needed for Conservation; it stays the "
                   "optional PoUW join (one hash), priced by ``tau_bind`` and ``tau_bind_kv`` when on", "policy"),
    "tau_bind": Input(0.06, "per weight pass", "ra-bound-receipt (ro-bound-sketch-blake3/v0, B = R ⊕ M) on "
                      "Llama-3.1-70B's linears, against cuBLASLt FP8: the bound receipt over the unbound, +0.05–0.08 "
                      "at every mode and batch (r20261012-191645-25bd)", "measured"),
    "tau_bind_kv": Input({2: 0.019, 3: 0.016}, "per KV pass", "ra-word-codec 21:35Z: the block receipts bound to "
                         "attention's PV output two tiles back, +0.287 / +0.781 against the unbound +0.268 / +0.765 "
                         "(r20261012-212502-016a)", "measured"),
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
    "eph_read": Input({"8B": (107.0, 2.3), "70B": (60.0, 2.5), "405B": (107.0, 2.5)}, "B per layer·hidden",
                      "ra-ephemeral's fused arm (vLLM's default Inductor compilation): Nsight DRAM reads beyond the "
                      "linears, the lm_head and the live KV, per graph at full depth, 8B 14–33 MB and 70B 41–164 MB "
                      "at B1–64 2k (r20261012-184939-520e, art:8c12a22f; μ_eph 0.05–0.06% at 70B and 0.06–0.16% at "
                      "8B where timed, r20261012-184455-2823); 405B the larger of each", "measured",
                      {"8B": (0.0, 2.3), "70B": (0.0, 2.5), "405B": (0.0, 2.5)},
                      "the fused step's fixed reads kept on chip (GEMM inputs, Triton's and attention's partials: "
                      "estimate)"),
    "eph_write": Input({"8B": (900.0, 6.0), "70B": (700.0, 6.5), "405B": (900.0, 6.5)}, "B per layer·hidden",
                       "ra-ephemeral 21:40Z, stock kernels as the upper bound (21:50Z): DRAM writes are 0.6–0.8% of "
                       "reads at 70B and 0.4–1.6% at 8B (r20261012-184939-520e), 92% of them the CUTLASS FP8 GEMM's "
                       "and XQA's local-memory stores (2361 art:5c8baf5e, 6732 art:1d638a4b); with write bus time "
                       "μ_eph is 0.58–0.66% at 70B and 0.32–1.30% at 8B; 405B the larger of each", "derived",
                       {"8B": (0.0, 6.0), "70B": (0.0, 6.5), "405B": (0.0, 6.5)},
                       "the encoded GEMM and attention pass the 21:50Z gate (STACK:0, LOCAL:0, L2 writes no more than "
                       "their outputs and declared partials), leaving the activations' write-backs: about 0.1% at 70B "
                       "batch 1 (ra-ephemeral, estimate)"),
    "eph_split": Input(0.0, "of the bytes divided across a step's cards", "ra-ephemeral: a card's rest barely splits "
                       "(38 MB a graph at TP 2 against 41 MB at TP 1), and 70B at TP 2 writes 0.79–1.51% of a card's "
                       "bus time (cd24 art:ccbe15a6), above even unsplit bytes here (0.36–1.16%); TP 4 and 8 are "
                       "unmeasured", "measured"),
    "write_bus": Input(1.0, "read-times per written byte", "a DRAM write holds the bus as long as a read (turnaround "
                       "not counted)", "estimate"),
    # The verifier's sample (μ_sample) and the counting slack (λ).
    "v_budget": Input(2.0, "Gb/s per node", "the verifier's download for openings, a cap on the hybrid's", "policy"),
    "t_window": Input(1.0, "s", "the window", "policy"),
    "window_max": Input(0.01, "of a window", "the hybrid (Daniel 20:55Z): every window's sampling at most 1%, so "
                        "ln(2^40)/0.01 = 2,773 openings a window (ra-openings §1: 2,761, 12.5 MB/s at 2 KiB)",
                        "policy"),
    "opening_bytes": Input(2500.0, "B an opening beyond the block and z",
                           "ra-openings §1's hybrid row: 4,522 B an opening at 2 KiB and s = 2,761, multiproofs per "
                           "tree and one node for an adjacent PoUW leaf included", "derived"),
    "mu_serve": Input(0.0001, "of the reads", "ra-openings §9: serving the hybrid's draw, 64 receipts a kept node "
                      "(3.7 GB kept a window), reads no pass certifies: 0.010%", "derived"),
    "lambda_named": Input(True, "switch", "named λ, N + 40 bits a pass at every count, 0.003% at 4 KiB "
                          "(`RatePaidNamedRO`; `RateConservationNamedRO` gives N·H(μ) + 40), PR #1900 (STEERING "
                          "21:00Z)", "proved"),
    "horizon": Input(600, "windows", "the hybrid's 10-minute horizon (Daniel 20:55Z), by the causal horizon bound "
                     "(`HorizonSampling`, `RateHorizon`, PR #1900)", "proved"),
    "log2_inv_delta": Input(40.0, "bits", "δ = 2^-40 (ra-bandwidth-lean)", "policy"),
    "log2_q": Input(64.0, "bits", "q ≤ 2^64 oracle queries (ra-bandwidth-protocol 4c3352875: λ = 64k + 41)", "policy"),
    "x_from_transcript": Input(False, "switch", "a bound receipt's opening carries its z; the verifier could take "
                               "z_{j−1} from PoUW's transcript instead", "estimate"),
    "receipt_bits": Input(512, "bits", "a SHA-512 receipt", "policy"),
    # η, the share of R·T the honest engine leaves unread, by phase on the GPU's clock: stock vLLM 8B on ra-kv-filler's
    # bursty trace 1 under async scheduling (trace 2, decode-heavy, in each source).
    "eta_counted": Input(True, "switch", "\"99%\" is against R·T (Daniel 20:55Z): γ adds the honest engine's "
                         "unread share by phase, which ra-fill-bandwidth and ra-fill-compute hill-climb", "policy"),
    "eta_decode": Input(0.223, "of R·T", "ra-kv-filler §7 on the GPU's clock: decode steps read at 0.72 R (trace 2: "
                        "0.75 R, 0.234), the host's gaps between steps included, which async scheduling keeps GPU-busy "
                        "(art:5448c72f); §8: the native decode step reads at 0.73–0.87 R of its span "
                        "(r20261012-211642-4dbc, art:be9b8dfd)", "measured", 0.05,
                        "every read native (0.79–0.88 R, §8) and the rest, 6–11% of the step (quantization, norms, "
                        "rotary, sampling), fused into the GEMMs' and attention's prologues and epilogues: 0.95 R "
                        "(estimate; a side stream nets only 0.0075 R per 1% of step time, ra-kv-filler)"),
    "eta_prefill": Input(0.099, "of R·T", "ra-kv-filler §7 on the GPU's clock: prefill reads at 0.038 R (trace 2: "
                         "0.085 R, 0.037; art:5448c72f)", "measured", 0.005,
                         "weights re-streamed and sketched beside the compute-bound prefill at 0.95 R, about +0.9% of "
                         "busy time (ra-kv-filler, estimate)"),
    "eta_gaps": Input(0.0, "of R·T", "ra-kv-filler §7: under async scheduling the GPU runs the queued step through "
                      "the host's gaps (0.040 of R·T on the host's clock, 0.065 on trace 2), so they are decode's and "
                      "prefill's GPU time; synchronous scheduling would add 0.07 of idle (art:5448c72f)", "measured"),
    "eta_idle": Input(0.095, "of R·T", "ra-kv-filler §7 on the GPU's clock: no request running (trace 2: 0.034; "
                      "art:5448c72f)", "measured", 0.005,
                      "the idle top-up's plain read, which fills 0.88–0.95 of it at +0.1% TPOT (ra-kv-filler)"),
    # Chained passes (μ_timing): a pass is one step's plan, and its seed needs the previous pass's root.
    "t_gap": Input(100e-6, "s per pass", "ra-remote-driver: the root's reduction (an all-gather of the GPUs' subtree "
                   "roots, one SHA-512, a broadcast), 25–100 µs, unmeasured", "estimate"),
    "prefetch": Input(False, "switch", "the GPUs wait for the seed", "estimate", True,
                      "prefetch the next pass's first blocks on chip across the gap (ra-remote-driver: 0.52% → "
                      "0.04% at 20 ms passes), hiding up to S_chip/R"),
    "copies": Input(1, "copies of the weights and KV, at most", "a pass is one step's plan, which reads a block at most "
                    "once (ra-remote-driver)", "policy", 8,
                    "c differently keyed copies of the weights and KV where they fit, one read a step, so a pass spans "
                    "c steps and pays its carried state once (design: the append encodes c times; needs the driver's "
                    "plan format and the red team)"),
    "rtt": Input(0.020, "s", "ra-remote-driver's verifier round trip", "policy"),
    "n_session": Input(1000, "windows", "sessions of at least 1,000 chained windows, the default policy "
                       "(Daniel 20:55Z, this lane's Ask 2)", "policy"),
    "p_over_o": Input(0.47, "prompt tokens per output token", "ra-kv-filler's decode-heavy trace 2 (93,312 prompt and "
                      "200,192 output tokens, every prompt token computed): each step also encodes P/O prompt tokens "
                      "a sequence, k·x₁ a byte over the reads it gets, with P the tokens computed rather than served "
                      "from a prefix cache (21:45Z; ra-kv-filler §6); the prompt encode was +7.3% / +11.0% of busy "
                      "time at k = 2 / 3 "
                      "(r20261012-183434-0d4b, r20261012-190215-6752, art:231e0b01)", "measured"),
}

# The floor: each lever past its realistic improvement to the best any measurement or bound supports, so that the
# frontier at the floor bounds how fast the audit can get on this node with this codec.
FLOOR: dict[str, tuple[Any, str]] = {
    "o_w": ({2: ((8, 0.052), (32, 0.052)), 3: ((8, 0.505), (32, 0.505))},
            "the mma-ws decode floor with 24 warps (ra-word-codec r20261012-172911-1184) on the weights too"),
    "o_kv": ({2: ((32, 0.052), (160, 0.052)), 3: ((32, 0.505), (160, 0.505))}, "the same floor on the KV"),
    "tau_w": ({2: ((8, 0.049), (32, 0.049)), 3: ((8, 0.049), (32, 0.049))},
              "the sketch's cost on a store that is only read (ra-sketch-tag plain store M=32, +1.7..1.9%) and the "
              "SHA-512 commit, which stays (0.030, ra-codec-tensor)"),
    "tau_kv": ({2: ((8, 0.046), (32, 0.046)), 3: ((8, 0.046), (32, 0.046))},
               "the same sketch on the KV and two 8-vector hashes a block (0.027, ra-word-codec)"),
    "eph_read": ({"8B": (0.0, 0.0), "70B": (0.0, 0.0), "405B": (0.0, 0.0)}, "no ephemeral byte read from DRAM"),
    "eph_write": ({"8B": (0.0, 0.0), "70B": (0.0, 0.0), "405B": (0.0, 0.0)}, "no ephemeral byte written to DRAM"),
    "eta_decode": (0.0, "every phase reads at R, as the frontier's step does"),
    "eta_prefill": (0.0, "every phase reads at R"),
    "eta_idle": (0.0, "every phase reads at R"),
}

# Values an input may take that make the frontier worse, each with why it is plausible.
RISKS: dict[str, tuple[Any, str]] = {
    "s_adv": (((512, 1.535), (768, 1.514), (1024, 1.676)), "Karatsuba at the probe's mix at its best depth "
              "(`wide.adversaries`' conservative row, 1.885 at 1,024 bits with h at the bound)"),
    "dec_ratio": (((512, 1.0), (768, 1.42), (1024, 1.83)), "the decode a byte at ra-squaring-ceiling's cycle model "
                  "(a squaring's cycles over the word's bytes), above both measurements"),
    "write_bus": (1.5, "a write costs half a read more in bus turnaround"),
    "v_budget": (0.2, "a 200 Mb/s verifier link, which carries the hybrid's 2,773 openings a window at blocks of "
                 "4 KiB at most"),
    "t_gap": (500e-6, "a root reduction of 500 µs (ra-remote-driver's worst row)"),
    "p_over_o": (3.63, "ra-kv-filler's trace 1 (160,256 prompt and 44,160 output tokens), where the prompt encode was "
                       "+17% / +25% of busy time at k = 2 / 3"),
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
