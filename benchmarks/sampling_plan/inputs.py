"""The int8 Llama-3.1-8B row's inputs to the sampling planner: each a value, a low and a high, its source and its status (M
measured, A assumed, D derived from the others), in one currency, GPU-seconds of an RTX PRO 6000 (sm_120)."""
from __future__ import annotations

from dataclasses import dataclass

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
    "zk_unit": Input(2.0, 1.0, 4.0, "a zk unit's proving over the non-zk sweep's (A6 is zk, the sweep is not)", "A"),
    "noise_bits": Input(1, 1, 2, "noise bits an entry (compute accounting, decision 105)", "A"),
    "zero_work_templates": Input(30, 15, 60, "the layer's templates that carry no work (quantizers, softmax, norms, "
                                 "the epilogues' pieces), each held at its floor", "A"),
}

#: ANDs of the pieces (#2391's pins and hm96 PROTOCOL §5)
S8_STEP = 4_190
EPILOGUE = 5_000
SHA512 = 58_120
BF16_STEP = 8_233

#: Llama-3.1-8B: hidden 4096, intermediate 14336, 32 layers, 32 query heads and 8 KV heads of 128
LINEARS = {4096: 6144 + 4096 + 28672, 14336: 4096}
LAYERS, HEADS, HEAD_DIM = 32, 32, 128

#: the measurements the cost model is fitted from (`costfit`); ands are a unit's with its rows' openings
SWEEP_ANDS = 384 * BF16_STEP + 77 + 2 * (116_240 + SHA512 * 97) + (116_240 + SHA512)
SWEEP = {1: (0.6769, 224.84), 2: (0.6319, 224.58), 4: (0.6406, 225.2), 8: (0.6400, 222.6), 16: (0.6849, 220.6),
         32: (0.6288, 219.0), 64: (0.6609, 200.2), 128: (1.048, 193.6), 256: (1.019, 193.6), 512: (1.369, 190.5),
         1024: (2.081, 169.4)}
A6_ANDS = {2048: 64 * S8_STEP + EPILOGUE + 2 * (116_240 + SHA512 * 17) + 3 * (116_240 + SHA512),
           8192: 256 * S8_STEP + EPILOGUE + 2 * (116_240 + SHA512 * 65) + 3 * (116_240 + SHA512)}


def observations() -> list[Observation]:
    """benchmarks/proof_throughput's bf16-k6144 sweep (non-zk, a 1,024-unit statement, runs in its baseline.json, prove
    run r20261017-103527-eb06); A6's int8 statements (zk, run r20261017-202742-da01: 33 s and 69 s staging, 32 s and 83 s
    Lean verify, 1.4 s and 2.2 s a session of about 11 and 6 drawn units); PRF-57's small CPU sessions (about 1.15 s Lean
    verify, note:proofs/20261017T0552Z-report-proof-service-cost-model)."""
    out = [Observation(f"proof_throughput:B{b}", "GemmCoordinate_v2{K=6144}", SWEEP_ANDS, 1024, b, prove=p)
           for b, (p, _) in SWEEP.items()]
    out += [Observation("r20261017-202742-da01", "ScaledMmS8Coordinate{K=2048}", A6_ANDS[2048], 4096, 11, prove=1.4,
                        verify=32.0, stage=33.0),
            Observation("r20261017-202742-da01", "ScaledMmS8Coordinate{K=8192}", A6_ANDS[8192], 2048, 6, prove=2.2,
                        verify=83.0, stage=69.0),
            Observation("note:proofs/20261017T0552Z-report-proof-service-cost-model", "Pc4Tile_v1{K=4096}",
                        A6_ANDS[2048] * 2, 16, 16, verify=1.15)]
    return out


KEY = {"prover": "flock-circuit 1216db812a (binary dff9f6fbc8bb) for the per-unit slope; source 7107bab611 (A6) for the "
                 "zk session",
       "verifier": "flock-verify 7107bab611 (A6); build_key 4163b0b7fc3d3cfc in the sweep (not fitted: an older parse)",
       "hardware": "RTX PRO 6000 Blackwell Server Edition; Xeon 6776P (sweep), EPYC-Genoa (A6)",
       "templates": "GemmCoordinate_v2{K=6144,DOT=HopperBF16WgmmaDot16_v1} (statement 6b661a3d), "
                    "ScaledMmS8Coordinate_v1{K=2048|8192} (A6)"}
