"""The int8 Llama-3.1-8B row's inputs to the sampling planner: each a value, a low and a high, its source and its status (M
measured, A assumed, D derived from the others), in one currency, GPU-seconds of an RTX PRO 6000 (sm_120)."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from verity.core.protocols.sampled_proofs.costfit import Observation


@dataclass(frozen=True)
class Input:
    value: float
    low: float
    high: float
    source: str
    status: str


I = {
    "cpu": Input(1 / 55, 1 / 110, 1 / 27, "vCPU $0.033/h against sm_120 $1.80/h (note:compute-accounting/"
                 "20261018T0349Z-draft-budget, X = 55)", "M"),
    "round": Input(12.6, 12.6, 12.6, "a round (BUDGET)", "A"),
    "rounds": Input(48_000, 48_000, 48_000, "a week of rounds (BUDGET's S)", "A"),
    "tokens_per_s": Input(20_000, 8_000, 40_000, "int8 Llama-3.1-8B on one RTX PRO 6000: 7.2G MACs a token at 2K context "
                          "against 5e14 MAC/s dense int8 at 30-40% (prefill) down to decode's weight-bandwidth bound", "A"),
    "context": Input(2048, 2048, 2048, "prefill at 2K, positions uniform (BUDGET's binding cell)", "A"),
    "step_tokens": Input(256, 64, 2048, "tokens a forward step (a replay unit is one layer of one step)", "A"),
    "replay_layer_step": Input(5.2e-4, 1.0e-4, 2.0e-3, "one layer of one step re-executed from its committed input: its "
                               "weights' read (218 MB at 1.8 TB/s) plus 256 tokens at the plain rate; Qwen3-8B on H200 "
                               "is 0.18 ms a layer-step at B32 (note:circuits/20261018T0315Z-report-commit-defaults-"
                               "overlap)", "A"),
    "commit_per_byte": Input(4.34e-12, 2.2e-12, 1.3e-11, "SHA-512 rows on H200: 3.62 ps a byte plus 2.94 ns a 4 KiB leaf "
                             "(note:circuits/20261017T2200Z-report-commit-speed); sm_120 taken at 0.5-3x", "M"),
    "commit_floor": Input(0.060, 0.045, 0.101, "the served Program's own commitment, SHA-512 tile leaves at R = 128 on "
                          "sm_120 (COMMIT-GPU r5 via BUDGET), paid by every plan", "M"),
    "noise_bits": Input(1, 1, 2, "noise bits an entry (compute accounting, decision 105)", "A"),
    "zero_work_templates": Input(30, 15, 60, "the layer's templates that carry no work (quantizers, softmax, norms, "
                                 "the epilogues' pieces), each held at its floor", "A"),
}

#: ANDs of the pieces (#2391's pins and hm96 PROTOCOL §5)
S8_STEP = 4_190
EPILOGUE = 5_000
SHA512 = 58_120
HM96 = 116_240

#: Llama-3.1-8B: hidden 4096, intermediate 14336, 32 layers, 32 query heads and 8 KV heads of 128
LINEARS = {4096: 6144 + 4096 + 28672, 14336: 4096}
LAYERS, HEADS, HEAD_DIM = 32, 32, 128



def s8_coord_ands(k: int) -> int:
    """One s8 coordinate's ANDs with its rows' openings, as `costfit` counts them: K/32 steps, the epilogue, two K-byte
    operand rows and three one-block rows (the scales and the output)."""
    return k // 32 * S8_STEP + EPILOGUE + 2 * (HM96 + SHA512 * ceil((k + 17) / 128)) + 3 * (HM96 + SHA512)


#: the measurements the cost model is fitted from (`costfit`): benchmarks/one_stage/zk_cost.py's sessions of
#: `ScaledMmS8Coordinate_v1{K}` by `flock-circuit prove --zk --gpu` with the firewall's send check, per (K, staged units,
#: drawn units): the mean wall of its two cold prover processes (GPU-s: one touched statement's prover today, its statement
#: build included), the mean wall of Lean `verify --zk` on its sessions and its statement's staging (CPU-s)
ZK_RUN = "r20261018-054315-8879+r20261018-060433-b55e"
ZK: dict[tuple[int, int, int], tuple[float, float, float]] = {
    (4096, 1, 1): (22.34, 52.22, 4.7),
    (4096, 8, 1): (21.57, 49.69, 4.5),
    (4096, 8, 8): (19.33, 46.50, 4.5),
    (4096, 64, 1): (18.55, 47.66, 7.2),
    (4096, 64, 8): (25.67, 39.35, 7.2),
    (4096, 64, 64): (27.75, 38.54, 7.2),
    (4096, 512, 1): (21.68, 39.20, 10.2),
    (4096, 512, 8): (21.22, 35.67, 10.2),
    (4096, 512, 64): (30.57, 36.83, 10.2),
    (4096, 512, 512): (62.17, 58.45, 10.2),
    (14336, 1, 1): (50.50, 159.68, 9.6),
    (14336, 8, 1): (54.69, 148.43, 9.9),
    (14336, 8, 8): (54.68, 162.62, 9.9),
    (14336, 64, 1): (53.68, 163.31, 16.1),
    (14336, 64, 8): (54.06, 172.87, 16.1),
    (14336, 64, 64): (65.44, 164.61, 16.1),
    (14336, 512, 1): (54.45, 168.34, 41.6),
    (14336, 512, 8): (54.39, 163.51, 41.6),
    (14336, 512, 64): (66.22, 170.70, 41.6),
    (14336, 512, 512): (131.07, 171.62, 41.6)}


def observations() -> list[Observation]:
    return [Observation(ZK_RUN, f"ScaledMmS8Coordinate_v1{{K={k}}}", s8_coord_ands(k), c, d, prove=p, verify=v, stage=s)
            for (k, c, d), (p, v, s) in ZK.items()]


KEY = {"prover": "flock-circuit bin-dec484d021e461a0-g1-sm120 (binary 21944fa201b5, CUDA 13.3.1), --zk --gpu, "
                  "FC_HOST_WITNESS=1, the firewall's send check on",
       "verifier": "flock-verify lean-3a9d4dc43048fdae (binary b3b257d5ac0e), verify --zk, one CPU",
       "hardware": "RTX PRO 6000 Blackwell Server Edition (driver 580.173.02), Xeon 6776P, vy-nebius-2",
       "templates": "ScaledMmS8Coordinate_v1{K=4096|14336}"}
