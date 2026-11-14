"""Per-type levers for the repriced break-even: the fused per-expert block's price, the statement cache, K provers
overlapped on one GPU, a faster GPU prover, the dense types' host witness, Flock without --zk, the Lean verifier's CPU
priced in GPU-seconds, and the 128k workload over vLLM's 128k serve plan.

Every factor multiplies the card's end-to-end prices (fixed seconds a statement, seconds a block, blocks a statement).
Each table's comment says whether its values are measured, modeled or assumed.
"""
import json
import math
import re
import statistics
from dataclasses import replace

from common import SCOPES, c, f1, tc

#: The statement cache on the H200: (fixed after over before, host share of after) by block prefix: three types
#: measured warm on one H200 against the card's fixed, quant K6144 (a 2-block statement 8.56 s, 8.30 s of it host,
#: against 98.16 s), FlashInfer K2048 (5.12 s, 4.95 s of it host, against 112.3 s) and quant K2048 (2.04 s, 0.115 s of
#: it proving, against 47.8 s); one priced from its uncached H200 split and two (quant K1536, combine N5) from their
#: CPU shares. "Host" is end to end less GPU proving.
CACHE = {
    "Fp8GroupQuantUe8m0_v1{K=6144,": (8.56 / 98.16, 8.30 / 8.56),
    "FlashInferFp8Linear_v1{K=2048,": (5.12 / 112.3, 4.95 / 5.12),
    "AllReduceMultimem_v1{": (3.0 / 42.6, 0.27 / 3.0),
    "Fp8GroupQuantUe8m0_v1{K=2048,": (2.04 / 47.8, (2.04 - 0.115) / 2.04),
    "Fp8GroupQuantUe8m0_v1{K=1536,": (6.3 / 39.6, 6.0 / 6.3),
    "SparseMlaCombine_v1{": (7.7 / 42.7, 7.0 / 7.7),
}
#: seconds a block measured warm on one H200 with the statement cache on, in place of the card's, and the GPU's share
#: of them (the rest the host's): FlashInfer K2048 at 7.16 ms, 0.18 ms of it GPU proving
MARGINAL = {"FlashInferFp8Linear_v1{K=2048,": (7.16e-3, 0.18 / 7.16)}
#: the priced type whose factors a type the cache wasn't priced on takes, by prefix (assumed)
FAMILY = (("AllReduce", "AllReduceMultimem_v1{"), ("RmsNormMnnvl", "AllReduceMultimem_v1{"),
          ("Fp8GroupQuant", "Fp8GroupQuantUe8m0_v1{K=2048,"), ("FlashInferFp8Linear", "FlashInferFp8Linear_v1{K=2048,"),
          ("ScaledMm", "FlashInferFp8Linear_v1{K=2048,"), ("SparseMlaCombine", "SparseMlaCombine_v1{"))
#: any other type: the priced types' median factor, and a host share of after like the dense types' (assumed)
DEFAULT = (0.1, 0.9)
#: the fused and per-expert GEMMs, K6144 (w13) and K256 (w2): the cache alone at one prover, end to end (fixed after
#: over before), and with verify-ahead and four provers on one GPU (MPS for w2): fixed a statement, seconds a block;
#: four provers fit one H200 at a quarter of a full statement each, or at full statements once each prover's peak is
#: about 33 GiB (`full`)
EXPERT = {"w13": {"cache": 0.56 / 5.73, "fixed": 0.045, "marginal": 0.58},
          "w2": {"cache": 0.67 / 5.54, "fixed": 0.043, "marginal": 0.54}}
PROVERS = 4
#: four provers with the cache and verify-ahead against the cache alone at one, on w13, taken for every other type
#: (assumed): fixed a statement, seconds a block
OVERLAP_OTHER = {"fixed": 0.25 / 0.56, "marginal": 0.58}
#: Flock without --zk: the share of a session --zk costs, by prefix: the fused blocks' measured on one H200 at 1024
#: blocks, the rest on CPU; every other type none
PLAIN = {"AllReduceMultimem_v1{": 0.35, "AllReduceResidualMnnvl": 0.35, "AllReduceMnnvl": 0.35, "RmsNormMnnvl": 0.35,
         "Fused[EmbeddingRead_v1{V=256,H=4096}": 0.235, "Fused[EmbeddingRead_v1{V=256,H=3072}": 0.208,
         "Fp8GroupQuantUe8m0_v1{K=1536,": 0.11, "Fp8GroupQuantUe8m0_v1{K=2048,": 0.08, "SparseMlaCombine_v1{": 0.08}
#: the dense types whose host witness dominates their fixed cost after the cache
DENSE = ("Fp8GroupQuantUe8m0_v1", "FlashInferFp8Linear_v1")
#: the linear layers' blocks, against which the indexer's MACs at 128k are stated: every cut's MoE expert GEMMs among
#: them (the served cut's `MoeExpertFp8CoordinateChunk_v1` and `MoeExpertFp8GemmW_v1`, the per-expert cuts' BlockGemm and
#: ReadGemm), since `indexer_factor` refuses an MoE expert block with MACs that no entry names
LINEARS = ("Fused[", "MoeExpertFp8BlockGemm", "MoeExpertFp8ReadGemm", "MoeExpertFp8CoordinateChunk", "MoeExpertFp8Gemm",
           "FlashInferFp8Linear", "ScaledMmFp8", "BatchedGemm", "CublasSplitKGemm", "LlBf16DotprodGemm", "Gemm_v2")
MOE_EXPERT = "MoeExpert"
INDEXER = "DsaMqaLogits_v3"
#: the DSA indexer's MACs a decode token at 131,072 keys over the linears'
INDEXER_AT_MAX = 2.2
MAX_KEYS = 131072
#: the fused per-expert blocks staged as one type each (`MoeExpertFp8ReadGemm(W)_v1`), at this k_log
FUSED_K_LOG = 25
#: their end-to-end seconds over the unfused pair's (the GEMM block and the read block of one row), measured on CPU
#: at 16 blocks a statement with the cache on; taken for the fixed and the marginal parts alike
FUSED_CPU = {"w13": 0.75, "w2": 0.59}
#: their prices measured on one H200 with the statement cache, (fixed seconds a statement, seconds a block): one
#: prover end to end at 512 and 1024 blocks a statement, and four provers with verify-ahead at 128 and 256 blocks,
#: from their throughput (the GPU's seconds a statement)
FUSED_H200 = {"w13": {"one": (0.7374, 2.8488e-3), "four": (128 / 219 - 128 * (256 / 329 - 128 / 219) / 128,
                                                           (256 / 329 - 128 / 219) / 128)},
              "w2": {"one": (0.6792, 2.9751e-3), "four": (128 / 217 - 128 * (256 / 307 - 128 / 217) / 128,
                                                          (256 / 307 - 128 / 217) / 128)}}

#: the Lean verifier (`verify --zk --partition --program`), one core, one statement, setup included: seconds by block
#: prefix, measured on today's flat statement (the verifier_cost card), and its fixed part; the rest grows with the
#: unit's matrix entries
_VERIFIER_CARD = tc.spot_cards.load("verifier_cost", "GLM-5.3")["value"]
VERIFY = {t["block"]: t["core_seconds"] for t in _VERIFIER_CARD["types"]}
VERIFY_FIXED = _VERIFIER_CARD["fixed_seconds"]
#: (types, measured member, size parameter, the member's size): a family's other types scale linearly in the size
#: above VERIFY_FIXED, or take the member's seconds when the size is None (assumed)
VERIFY_FAMILY = ((("Fp8GroupQuantUe8m0_v1",), "Fp8GroupQuantUe8m0_v1{K=2048,", "K", 2048),
                 (("FlashInferFp8Linear_v1", "ScaledMmFp8BlockSharedScale_v1"), "FlashInferFp8Linear_v1{K=2048,", "K",
                  2048),
                 (("RMSNormCudaVector8_v1", "DsaRMSNorm_v1", "ResidualAdd_v1"), "RMSNormFusedCudaVector8_v1{N=6144,",
                  "N", 6144),
                 (("AllReduce", "RmsNormMnnvl_v1"), "AllReduceMultimem_v1{", None, None),
                 (("MoeExpertFp8", "Fused["), "MoeExpertFp8GemmW_v1{", None, None))
#: any other type: the measured types' median (assumed)
VERIFY_DEFAULT = statistics.median(VERIFY.values())
#: the templated, served verifier's estimate on quant K6144 (125 s) over today's, taken for every type (assumed)
VERIFY_TEMPLATED = 125.0 / 2100.0
#: core-seconds that cost one GPU-second, from Nebius AI Cloud's on-demand list prices
#: (docs.nebius.com/compute/resources/pricing): an H200 NVLink at $5.40 a GPU-hour against one non-GPU AMD EPYC Genoa
#: core, two vCPUs at $0.015 a vCPU-hour, with its 8 GiB at $0.0045 a GiB-hour
CORE_PER_GPU = 5.40 / (2 * 0.015 + 8 * 0.0045)
#: the flat Lean verifier with its proved twins (`fast`): a full statement in the proof server's verdict pool at K = 1
#: on the fused blocks (w13 48.2 s, w2 52.7 s), RMSNorm v8 (255.2 s) and the all-reduce (91.96 s), and on the server's
#: last K = 1 session record on the light types (the FP8 scaled matmuls 8.8 and 14.2 s, the DSA logits 11.6 s, the
#: residual add 37.1 s); today's column times the after-over-before ratio measured in one job a type on FlashInfer K2048
#: (0.207), quant K6144 (0.217) and the MoE GEMM (0.577); the other four by a fixed-plus-per-entry fit to those two
#: (0.21 to 0.23, assumed)
VERIFY_FAST = {"Fused[EmbeddingRead_v1{V=256,H=3072}": 48.2, "Fused[EmbeddingRead_v1{V=256,H=4096}": 52.7,
               "FlashInferFp8Linear_v1{K=2048,": 513.0, "Fp8GroupQuantUe8m0_v1{K=6144,": 456.0,
               "SparseMlaCombine_v1{NSPLIT=12,": 359.0, "RMSNormFusedCudaVector8_v1{N=6144,": 255.2,
               "Fp8GroupQuantUe8m0_v1{K=2048,": 150.0, "Fp8GroupQuantUe8m0_v1{K=1536,": 112.0,
               "SparseMlaCombine_v1{NSPLIT=2,": 75.0, "AllReduceMultimem_v1{": 91.96, "MoeExpertFp8GemmW_v1{": 8.7,
               "ScaledMmFp8BlockSharedScale_v1{K=2048,N=1,G=128}": 8.8,
               "ScaledMmFp8BlockSharedScale_v1{K=6144,N=1,G=128}": 14.2, "DsaMqaLogits_v3{N=1,H=32,D=128}": 11.6,
               "ResidualAdd_v1{N=6144}": 37.1}
VERIFY_FAST_MEASURED = ("Fused[EmbeddingRead_v1{V=256,H=3072}", "Fused[EmbeddingRead_v1{V=256,H=4096}",
                        "RMSNormFusedCudaVector8_v1{N=6144,", "AllReduceMultimem_v1{", "FlashInferFp8Linear_v1{K=2048,",
                        "Fp8GroupQuantUe8m0_v1{K=6144,", "MoeExpertFp8GemmW_v1{",
                        "ScaledMmFp8BlockSharedScale_v1{K=2048,N=1,G=128}",
                        "ScaledMmFp8BlockSharedScale_v1{K=6144,N=1,G=128}", "DsaMqaLogits_v3{N=1,H=32,D=128}",
                        "ResidualAdd_v1{N=6144}")
#: the verifier's families its end-to-end measurements are taken for: FlashInfer K2048's and RMSNorm v1 N=6144's
FI_FAMILY = ("FlashInferFp8Linear_v1", "ScaledMmFp8BlockSharedScale_v1")
RMS_FAMILY = ("RMSNorm", "DsaRMSNorm", "ResidualAdd", "RmsNormMnnvl")
#: the flat column's cut a statement (`verify_cut`): the flat verifier with its file reading cut and the HmOut.delta
#: twin over the fast column's, one core on the same flat files, by family (FlashInfer K2048 304.3 s against 350.5 s,
#: RMSNorm v1 N=6144 138.9 s against 214.7 s); every other type at the file reading cut alone, measured on FlashInfer
#: K2048 and quant K6144 (assumed), but the MoE family's, which keeps its own
VERIFY_CUT_MEASURED = ((FI_FAMILY, 304.3 / 350.5), (RMS_FAMILY, 138.9 / 214.7))
VERIFY_CUT = 0.85
VERIFY_CUT_KEEPS = ("MoeExpertFp8", "Fused[")
#: today's column with the fused blocks at their fast seconds over the MoE GEMM's after over before (assumed), where
#: the card takes its MoE member's
VERIFY_TODAY = VERIFY | {p: VERIFY_FAST[p] / 0.577 for p in VERIFY_FAST if p.startswith("Fused[")}
VERIFY_COLUMNS = {"today": (VERIFY_TODAY, VERIFY_FIXED), "fast": (VERIFY_FAST, min(VERIFY_FAST.values()))}
#: the templated (structural) session check's whole verify over the flat one's, one core on the same statements, the
#: flat with its cut, by family, applied to the fast column: FlashInfer K2048 31.1 s against 304.3 s, RMSNorm v1
#: N=6144 96.1 s against 138.9 s; any other type keeps the flat (unmeasured)
TMPL_MEASURED = ((FI_FAMILY, 31.1 / 304.3), (RMS_FAMILY, 96.1 / 138.9))
#: the families quant K6144's estimate isn't taken for (`goal`), which keep their measured ratio
TMPL_GOAL_EXCLUDES = ("RMSNorm", "DsaRMSNorm", "RmsNormMnnvl")

#: the long-running proof server on one H200, one prover, the statement cache warm, on the final driver (each session
#: timed when its verifier files it): (seconds a full statement, its proving share 1 - front/e2e) by block prefix
SERVER_K1 = {"ScaledMmFp8BlockSharedScale_v1{K=6144,N=1,G=128}": (1 / 0.500, 1 - 0.79 * 0.500),
             "ScaledMmFp8BlockSharedScale_v1{K=2048,N=1,G=128}": (1 / 0.483, 1 - 0.91 * 0.483),
             "DsaMqaLogits_v3{N=1,H=32,D=128}": (1 / 0.363, 1 - 0.80 * 0.363),
             "ResidualAdd_v1{N=6144}": (1 / 0.405, 1 - 0.48 * 0.405),
             "RMSNormFusedCudaVector8_v1{N=6144,": (1 / 0.2525, 1.19 / 3.88),
             "AllReduceMultimem_v1{WORLD=8,SIZE=3072,": (1 / 0.1152, 7.78 / 8.64)}
#: each type's split of a statement into fixed seconds and seconds a block, an earlier driver's two-point fit at full
#: and a quarter of a statement (rmsnorm and the all-reduce at full and half), and blocks a full statement
SERVER_FIT = {"ScaledMmFp8BlockSharedScale_v1{K=6144,N=1,G=128}": (0.98, 1.41e-3, 2048),
              "ScaledMmFp8BlockSharedScale_v1{K=2048,N=1,G=128}": (0.79, 0.74e-3, 4096),
              "DsaMqaLogits_v3{N=1,H=32,D=128}": (0.96, 0.71e-3, 4096),
              "ResidualAdd_v1{N=6144}": (1.95, 5.33e-3, 512),
              "RMSNormFusedCudaVector8_v1{N=6144,": (5.11, 10.38e-3, 256),
              "AllReduceMultimem_v1{WORLD=8,SIZE=3072,": (3.79, 9.40e-3, 256)}
#: (fixed GPU-seconds a statement, GPU-seconds a block, proving share, blocks a full statement): the fit scaled to the
#: final driver's seconds a full statement
SERVER = {p: (f * (s / (f + n * m)), m * (s / (f + n * m)), share, n)
          for p, (s, share) in SERVER_K1.items() for f, m, n in [SERVER_FIT[p]]}
#: the fused blocks on node A, one prover at full statements: sessions a second, and prove_total's share of a session
#: (M1's 1.101 s at main's prover times its 0.570 sessions a second); w13's fit scaled to those sessions, and w2 by
#: the server's w2 over w13 (assumed)
FUSED_SESSIONS_S = (0.53 + 0.59) / 2
FUSED_PROVE_SHARE = 1.101 * 0.570
FUSED_SERVER = {e: (f * k, m * k, FUSED_PROVE_SHARE, 1 << (35 - FUSED_K_LOG))
                for e, (f, m), k in (("w13", (1.24, 2.77e-3), 1 / FUSED_SESSIONS_S / (1.24 + 1024 * 2.77e-3)),
                                     ("w2", (1.30, 2.93e-3), 1 / FUSED_SESSIONS_S / (1.24 + 1024 * 2.77e-3)))}
#: the types kept at one prover an H200: the sparse-MLA head block's host memory is unmeasured, and so is its scores
#: unit's, which stages at most at the head block's k_log, 27 at 64 keys (assumed: its values units, at 24 or less,
#: take the default)
ONE_PROVER = ("SparseMlaHeadBlock_v1", "SparseMlaHeadScores_v1")
#: K provers on one H200, sessions a second over one prover's, measured on M1 (the fused w13 at full statements) on
#: node A; four under MPS; the dense types fit two provers a GPU in host memory today (`dense_k`)
K_GAIN = {2: 0.768 / 0.570, 4: 0.903 / 0.570}
K_GAIN_MPS = 1.77
#: each type's own statements a second at K provers over one's without MPS on the final driver, by block prefix or
#: expert (`k_gain="types"`; the rest at K_GAIN); fr-w13 and rmsnorm-v8 in the same run, fr-w2's K = 4 the mean of two
K_GAIN_TYPES = {"ScaledMmFp8BlockSharedScale_v1{K=6144,N=1,G=128}": {2: 1.04, 4: 1.10},
                "ScaledMmFp8BlockSharedScale_v1{K=2048,N=1,G=128}": {2: 1.17, 4: 1.16},
                "DsaMqaLogits_v3{N=1,H=32,D=128}": {2: 1.10, 4: 1.13},
                "ResidualAdd_v1{N=6144}": {2: 1.09, 4: 1.20},
                "RMSNormFusedCudaVector8_v1{N=6144,": {2: 1.60},
                "AllReduceMultimem_v1{WORLD=8,SIZE=3072,": {2: 1.02, 4: 1.00},
                "w13": {2: 1.20, 4: 1.34}, "w2": {2: 1.17, 4: 1.45}}
#: the most provers one H200's device memory holds, in every mode: rmsnorm-v8's four need all of its 140 GiB, and the
#: all-reduce's four come within 1 GiB of it
K_MAX = {"RMSNormFusedCudaVector8_v1{N=6144,": 2, "AllReduceMultimem_v1{WORLD=8,SIZE=3072,": 2}
#: each type's statements a second at K provers under MPS over one prover's without it, both in one run on one H200
#: (`k_gain="mps_types"`), never capped by the device's share: MPS overlaps device work, and the device-bound types
#: gain most; fr-w2, which has no MPS row, at its own without MPS
K_MPS = {"AllReduceMultimem_v1{WORLD=8,SIZE=3072,": {2: 1.76, 4: 2.99},
         "ResidualAdd_v1{N=6144}": {2: 1.58, 4: 2.38},
         "DsaMqaLogits_v3{N=1,H=32,D=128}": {2: 1.16, 4: 1.69},
         "ScaledMmFp8BlockSharedScale_v1{K=2048,N=1,G=128}": {2: 1.17, 4: 1.30},
         "ScaledMmFp8BlockSharedScale_v1{K=6144,N=1,G=128}": {2: 1.22, 4: 1.29},
         "RMSNormFusedCudaVector8_v1{N=6144,": {2: 1.71},
         "w13": {2: 1.29, 4: 1.50}, "w2": K_GAIN_TYPES["w2"]}
#: under MPS, a type without a row of its own: the GEMM rows' lowest, sm-fp8's (assumed)
K_MPS_FLOOR = {2: 1.17, 4: 1.29}
#: the busy-share model K provers' gain is held to as its goal: chains overlapping at the light types' statements a
#: second at K = 4 over K = 1 (2.87) over 4, on an earlier driver
OVERLAP_EFFICIENCY = 2.87 / 4
#: the GPU prover: M1's prove_total at main's prover (1.101 s), with kernels' marked prover PRs (0.747 s), and their
#: modeled target, 4x the prover main had before its first prover PRs (2.088 s)
PROVE_OLD_MAIN = 2.088
PROVE_MAIN, PROVE_MARKED, PROVE_TARGET = 1.101, 0.747, PROVE_OLD_MAIN / 4
#: the quant K6144's price a block at the floor its K2048 sibling's runs show
QUANT_FLOOR = 0.003532


def gpu_factor(busy: float, prover: float = 1.0, k: int = 1, front: bool = False, gain: float | None = None) -> float:
    """A statement's GPU time over one prover's: the proving share `busy` sped up `prover` times, the host share never;
    the next session's front run beside this one's device work (`front`), and `k` provers sharing the GPU at `gain`
    times one prover's sessions a second (the busy-share model's k·OVERLAP_EFFICIENCY when None), never past the
    device's share."""
    device, host = busy / prover, 1.0 - busy
    chain = max(device, host) if front else device + host
    return chain if k == 1 else max(device, chain / (gain or k * OVERLAP_EFFICIENCY))


def templated_seconds(x, how: str) -> float:
    """The templated session check's core-seconds a statement of `x`'s type: `measured`, the fast column times its
    family's measured ratio (TMPL_MEASURED), none for a type outside them; `goal`, the less of that and quant K6144's
    estimate over today's, which TMPL_GOAL_EXCLUDES' families don't take."""
    ratio = next((r for types, r in TMPL_MEASURED if x["block"].startswith(types)), math.inf)
    measured = ratio * verifier_seconds(x, *VERIFY_COLUMNS["fast"])[0]
    if how != "goal" or x["block"].startswith(TMPL_GOAL_EXCLUDES):
        return measured
    return min(measured, VERIFY_TEMPLATED * verifier_seconds(x, VERIFY_TODAY, VERIFY_FIXED)[0])


def verify_per_and(strata, column="fast") -> float:
    """The Lean verifier's core-seconds an AND: the median over the types `column` measures of a full statement's
    seconds over its ANDs."""
    table, floor = VERIFY_COLUMNS[column]
    rates = [verifier_seconds(x, table, floor)[0] / (x["ands"] * x["statement_blocks"]) for x in strata
             if x.get("ands") and verifier_seconds(x, table, floor)[1] == "measured"]
    return statistics.median(rates)


def server_price(x) -> tuple[float, float, float, int] | None:
    e = expert(x) if x.get("kind") == "fused" else None
    if e:
        return FUSED_SERVER[e]
    return next((v for p, v in SERVER.items() if x["block"].startswith(p)), None)


#: what a steady-state row holds (`steady`): the block type as the strata name it, a statement's GPU-seconds, the
#: blocks a statement held, the provers that shared the GPU, and the run or art that holds the measurement
STEADY_FIELDS = ("block", "statement_seconds", "blocks", "provers", "record")


def steady(path) -> dict[str, dict]:
    """The proof server's steady-state rows in a file, by block type: `{"rows": [...]}`, each row with
    STEADY_FIELDS. They are measurements a run made, so they arrive as a file a case names (`sweep.Case.steady`) and
    never as a table here."""
    rows = json.load(open(path))["rows"]
    for r in rows:
        if lacks := [k for k in STEADY_FIELDS if k not in r]:
            raise ValueError(f"{path}: the row of `{r.get('block')}` lacks {', '.join(lacks)}")
        if not (r["statement_seconds"] > 0 and r["blocks"] >= 1 and r["provers"] >= 1):
            raise ValueError(f"{path}: the row of `{r['block']}` is no measurement: {r}")
    by = {r["block"]: r for r in rows}
    if len(by) != len(rows):
        raise ValueError(f"{path}: two rows price one block type")
    return by


def basis(x) -> str:
    """What prices a stratum's proving off the steady state, by the table's name: a server price, or the card's
    seconds with the statement cache's factor from CACHE, from FAMILY's member or from DEFAULT."""
    if server_price(x):
        return "reprice.FUSED_SERVER" if x.get("kind") == "fused" else "reprice.SERVER"
    if expert(x):
        return "reprice.EXPERT"
    how = cache_factor(x)[2]
    return {"measured": "reprice.CACHE", "CPU share": "reprice.CACHE (a CPU share)", "family": "reprice.FAMILY",
            "assumed": "reprice.DEFAULT"}[how]


def basis_shares(strata) -> dict[str, float]:
    """Each basis's share of the proving a token takes with every block proved in full statements (a block's seconds
    and its share of a statement's fixed seconds, the verifier's included), the largest first; a steady-state row's
    basis is `steady: ` and its record."""
    cost: dict[str, float] = {}
    for x in strata:
        s = x["per_token"] * (x["seconds_per_block"] + x["fixed_seconds"] / max(1, x["statement_blocks"]))
        cost[x.get("basis", "the card")] = cost.get(x.get("basis", "the card"), 0.0) + s
    total = sum(cost.values())
    return {k: v / total for k, v in sorted(cost.items(), key=lambda kv: -kv[1])} if total else {}


def apply_server(strata, prover=1.0, k4=False, front=False, quant6144=None, host=1.0, plain=False, verify=0.0,
                 column="fast", tmpl=False, fused_busy=None, verify_other=None, k_gain="measured", dense_k=2,
                 verify_cut=False, cold=0.0, steady=None):
    """The strata priced as the long-running proof server proves them, with the statement cache warm. A type with a
    steady-state row (`steady`, by block type) costs what its row measured and nothing else: its statement's
    GPU-seconds over the statement's blocks for each block, at the prover count the row ran, in statements of the
    row's blocks, with no fixed seconds of its own (a statement of fewer blocks is charged in proportion, which no
    row measured) and no lever below but the Lean verifier's. Of the rest, the measured
    types at their server prices and full statements, the rest at the card with the cache's factor and MARGINAL's
    seconds a block (quant K6144 at `quant6144` when given); then a GPU prover `prover` times faster on proving
    shares, K provers an H200 on full statements (`k4`: four, `dense_k` on the dense types, one on
    ONE_PROVER's, never past K_MAX; at K_GAIN, K_GAIN_MPS with `k_gain="mps"`, each type's own with `"types"`, each
    type's own under MPS with `"mps_types"` (K_MPS over the whole chain, the rest at K_MPS_FLOOR), or the busy-share
    model with `"model"`), the next session's
    front beside this one's device work (`front`), the dense types' host witness `host` times lower, Flock without
    --zk, every prover's proving `1 + cold` times longer (the share of its time a cold session takes), and the Lean
    verifier's core-seconds a statement from `column` times `verify`, cut by its family's measured cut with
    `verify_cut` (VERIFY_CUT_MEASURED, VERIFY_CUT elsewhere, the MoE family's uncut; the templated check where it is
    cheaper, `tmpl` = `measured` or `goal`, see `templated_seconds`; the assumed types at `verify_other` when given).
    Each stratum says what priced its proving (`basis`): `steady: ` and its row's record, or `basis`'s table."""
    table, floor = VERIFY_COLUMNS[column]
    out = []
    for x in strata:
        fixed, marginal, cap = x["fixed_seconds"], x["seconds_per_block"], x["statement_blocks"]
        row = steady.get(x["block"]) if steady else None
        sp = None if row else server_price(x)
        if row:
            fixed, marginal, cap, bf, bm = 0.0, row["statement_seconds"] / row["blocks"], row["blocks"], 0.0, 0.0
        elif sp:
            fixed, marginal, b, cap = sp
            if fused_busy is not None and x.get("kind") == "fused":
                b = fused_busy
            bf = bm = b
        else:
            if quant6144 is not None and x["block"].startswith("Fp8GroupQuantUe8m0_v1{K=6144,"):
                marginal = quant6144
            host_m = 0.0
            if m := next((v for p, v in MARGINAL.items() if x["block"].startswith(p)), None):
                marginal, host_m = m[0], m[0] * (1 - m[1])
            e = expert(x)
            f, hs, _ = (EXPERT[e]["cache"], 0.0, "") if e else cache_factor(x)
            fixed *= f
            host_s = hs * fixed
            if host != 1.0 and x["block"].startswith(DENSE):
                fixed, host_s = fixed - host_s * (1 - 1 / host), host_s / host
                marginal, host_m = marginal - host_m * (1 - 1 / host), host_m / host
            bf = 1 - host_s / fixed if fixed else 1.0
            bm = 1 - host_m / marginal if marginal else 1.0
        k = 1
        if k4 and not row and not x["block"].startswith(ONE_PROVER):
            k = dense_k if x["block"].startswith(DENSE) else 4
            k = min(k, next((m for p, m in K_MAX.items() if x["block"].startswith(p)), k))
        if k_gain == "mps_types" and k > 1:
            g = mps_gain(x, k)
            fixed *= (bf / prover + 1 - bf) / g
            marginal *= (bm / prover + 1 - bm) / g
        else:
            gain = (None if k_gain == "model" or k == 1 else K_GAIN_MPS if k_gain == "mps" and k == 4
                    else (k_gain == "types" and type_gain(x, k)) or K_GAIN[k])
            fixed *= gpu_factor(bf, prover, k, front, gain)
            marginal *= gpu_factor(bm, prover, k, front, gain)
        if plain and not row:
            p = 1 - share(PLAIN, x)
            fixed, marginal = fixed * p, marginal * p
        if not row:
            fixed, marginal = fixed * (1 + cold), marginal * (1 + cold)
        if verify:
            v, how = verifier_seconds(x, table, floor)
            if tmpl:
                v = min(v, templated_seconds(x, "goal" if tmpl is True else tmpl))
            if how == "assumed" and verify_other is not None:
                v = verify_other
            if verify_cut and not x["block"].startswith(VERIFY_CUT_KEEPS):
                v *= next((r for types, r in VERIFY_CUT_MEASURED if x["block"].startswith(types)), VERIFY_CUT)
            fixed += verify * v / CORE_PER_GPU
        out.append(x | {"fixed_seconds": fixed, "seconds_per_block": marginal, "statement_blocks": int(cap),
                        "priced": "steady" if row else "measured" if sp else x["priced"],
                        "basis": f"steady: {row['record']}" if row else basis(x)})
    return out


def type_gain(x, k: int) -> float | None:
    key = expert(x) if x.get("kind") == "fused" else next((p for p in K_GAIN_TYPES if x["block"].startswith(p)), None)
    return K_GAIN_TYPES.get(key, {}).get(k)


def mps_key(x) -> str | None:
    """The K_MPS row `x`'s type is priced by, None for one at K_MPS_FLOOR."""
    key = expert(x) if x.get("kind") == "fused" else next((p for p in K_MPS if x["block"].startswith(p)), None)
    return key if key in K_MPS else None


def mps_gain(x, k: int) -> float:
    key = mps_key(x)
    return K_MPS[key][k] if key and k in K_MPS[key] else K_MPS_FLOOR[k]


def expert(x) -> str | None:
    b = x["block"]
    if b.startswith(("Fused[EmbeddingRead_v1{V=256,H=3072}", "MoeExpertFp8BlockGemm_v4{K=6144")):
        return "w13"
    if b.startswith(("Fused[EmbeddingRead_v1{V=256,H=4096}", "MoeExpertFp8BlockGemmW_v4{K=256")):
        return "w2"
    return None


def cache_factor(x, table=CACHE) -> tuple[float, float, str]:
    """(fixed after over before, host share of after, label) for a census type."""
    b = x["block"]
    for p, f in table.items():
        if b.startswith(p):
            return (*f, "measured" if "1536" not in p and "Combine" not in p else "CPU share")
    for fam, p in FAMILY:
        if b.startswith(fam):
            return (*table[p], "family")
    return (*DEFAULT, "assumed")


def share(table, x) -> float:
    return next((v for p, v in table.items() if x["block"].startswith(p)), 0.0)


def verifier_seconds(x, table=VERIFY, floor=VERIFY_FIXED) -> tuple[float, str]:
    """The Lean verifier's core-seconds a statement of `x`'s type from a column of measured types, and how they are
    known."""
    b = x["block"]
    for p, t in table.items():
        if b.startswith(p):
            return t, "measured"
    if m := re.match(r"SparseMlaCombine_v1\{NSPLIT=(\d+),", b):
        lo, hi = table["SparseMlaCombine_v1{NSPLIT=2,"], table["SparseMlaCombine_v1{NSPLIT=12,"]
        return lo + (int(m[1]) - 2) * (hi - lo) / 10, "interpolated"
    for types, member, size, at in VERIFY_FAMILY:
        if b.startswith(types):
            m = size and re.search(rf"[{{,]{size}=(\d+)", b)
            if not m:
                return table[member], "family"
            return floor + (table[member] - floor) * int(m[1]) / at, "family"
    return statistics.median(table.values()), "assumed"


def price_fused(strata, how="h200"):
    """The fused per-expert blocks at their staged k_log, priced from the unfused pair's summed fixed and marginal
    seconds (`levers.fuse_reads(fixed="sum")`) times the end-to-end ratio measured on CPU (`"cpu"`), or at their
    H200 prices (`"h200"`): one prover's with the cache, its fixed seconds over the cache's factor before it, and four
    provers' kept for the overlap (`"four"`)."""
    assert how in ("cpu", "h200"), how
    out = []
    for x in strata:
        e = expert(x) if x.get("kind") == "fused" else None
        if e:
            x = x | {"k_log": FUSED_K_LOG, "statement_blocks": max(1, tc.STATEMENT_BITS >> FUSED_K_LOG),
                     "table_blocks": tc.table_blocks(FUSED_K_LOG)}
            if how == "cpu":
                r = FUSED_CPU[e]
                x |= {"fixed_seconds": x["fixed_seconds"] * r, "seconds_per_block": x["seconds_per_block"] * r,
                      "priced": "CPU ratio"}
            else:
                (f, m), four = FUSED_H200[e]["one"], FUSED_H200[e]["four"]
                x |= {"fixed_seconds": f / EXPERT[e]["cache"], "seconds_per_block": m, "four": four,
                      "priced": "measured (H200)"}
        out.append(x)
    return out


def apply(strata, cache=False, overlap="", prover=1.0, full=False, host=1.0, plain=False, verify=0.0,
          verify_expert=None, verify_other=None):
    """The strata with each lever applied in turn: the cache's fixed seconds (with its host part kept apart), K provers
    overlapped on the expert types (`overlap="experts"`) or every type (`"all"`), a GPU prover `prover` times faster on
    device parts, full statements under K provers (`full`: four provers an H200 at about 33 GiB each), the dense types'
    host witness `host` times lower, Flock without --zk, and the Lean verifier's core-seconds a statement times
    `verify`, in GPU-seconds at CORE_PER_GPU (the expert types' at `verify_expert` core-seconds and the assumed types'
    at `verify_other` when given)."""
    assert not overlap or cache, "the overlap is measured with the cache on"
    out = []
    for x in strata:
        e = expert(x)
        fixed, marginal, cap = x["fixed_seconds"], x["seconds_per_block"], x["statement_blocks"]
        host_s = 0.0
        if cache:
            if e:
                fixed *= EXPERT[e]["cache"]
            else:
                f, hs, _ = cache_factor(x)
                fixed *= f
                host_s = hs * fixed
        if overlap and (e or overlap == "all"):
            if e and "four" in x:
                fixed, marginal = x["four"]
            elif e:
                fixed, marginal = x["fixed_seconds"] * EXPERT[e]["fixed"], marginal * EXPERT[e]["marginal"]
            else:
                fixed, host_s = fixed * OVERLAP_OTHER["fixed"], host_s * OVERLAP_OTHER["fixed"]
                marginal *= OVERLAP_OTHER["marginal"]
            if not full:
                cap = max(1, cap // PROVERS)
        if host != 1.0 and x["block"].startswith(DENSE):
            fixed, host_s = fixed - host_s * (1 - 1 / host), host_s / host
        if prover != 1.0:
            fixed, marginal = host_s + (fixed - host_s) / prover, marginal / prover
        if plain:
            p = 1 - share(PLAIN, x)
            fixed, marginal = fixed * p, marginal * p
        if verify:
            v, how = verifier_seconds(x)
            if e and verify_expert is not None:
                v = verify_expert
            elif how == "assumed" and verify_other is not None:
                v = verify_other
            fixed += verify * v / CORE_PER_GPU
        out.append(x | {"fixed_seconds": fixed, "seconds_per_block": marginal, "statement_blocks": int(cap)})
    return out


# ---- the 128k workload ---------------------------------------------------------------------------------------------
#: vLLM's 128k serve plan, its 651 requests within the span: computed tokens a request (a whole-prefix cache hit's
#: tokens left out), the token-weighted share of 131,072 keys each computed token's indexer reads, and the mean of
#: log2(max_tokens) + 1 a request (`mix` derives them from the plan)
C343 = {"tokens": 3320.5, "rho": 0.2536, "advice_bits": 8.75}
#: the same plan as served, its seven rows of 123,992 tokens or more aborted by the client
C343_SERVED = {"tokens": 2358.7, "rho": 0.1534, "advice_bits": 8.74}
#: a sampled request's replay at 128k in GPU-seconds, compute-weighted over the plan's served requests: its lifetime
#: alone, or every 600 s machine block its lifetime spans from the edge before it
LIFETIME_128K = 458.0
BLOCKS_128K = 5258.0
#: every full K/V page hashed when written, a share of one GPU
COMMIT_WRITE = 0.00045


def mix(plan_path: str, meta_path: str, within: int = 651) -> dict:
    rows = [json.loads(line) for line in open(plan_path)][:within]
    hits = {r["i"]: r["model"].get("prefix_hit", 0) for r in json.load(open(meta_path))["long_rows"]}
    tokens = keys = 0.0
    for r in rows:
        p, o = r["prompt_len"], r["max_tokens"]
        h = hits.get(r["i"], 0) if r["kind"].startswith("long128k") else 0
        tokens += p - h + o
        keys += (p * (p + 1) - h * (h + 1)) / 2 + o * p + o * (o + 1) / 2
    return {"tokens": tokens / len(rows), "rho": keys / tokens / MAX_KEYS,
            "advice_bits": sum(math.log2(r["max_tokens"]) + 1 for r in rows) / len(rows)}


def indexer_factor(strata, rho: float) -> float:
    """How many times the census request's indexer blocks a token the 128k mix reads: the indexer's MACs a token at
    `INDEXER_AT_MAX` times the linears' at 131,072 keys, in proportion to the keys read, over the census's.  An MoE
    expert block with MACs (by its block or its kind) that no entry of `LINEARS` names is refused by name: its MACs would
    drop out of the linears."""
    if missed := sorted({x["block"].split("{")[0] for x in strata if x.get("macs") and not x["block"].startswith(LINEARS)
                         and (x["block"].startswith(MOE_EXPERT) or str(x.get("kind") or "").startswith(MOE_EXPERT))}):
        raise ValueError(f"indexer_factor: the MoE expert blocks {', '.join(missed)} match no entry of reprice.LINEARS, so "
                         "the 128k indexer would be stated without their MACs; add each one's prefix to LINEARS")
    linears = sum(x["per_token"] * x["macs"] for x in strata if x["block"].startswith(LINEARS))
    census = sum(x["per_token"] * x["macs"] for x in strata if x["block"].startswith(INDEXER))
    return INDEXER_AT_MAX * rho * linears / census


def workload(L, mix_: dict = C343):
    """The line at the 128k mix: its indexer blocks a token scaled by `indexer_factor`, the served MACs a token with
    them, the mix's tokens a request, and the scopes' requests recomputed from both."""
    name, i, (tb, fl, oh), comp, conf, replay_gpu_s, extra, *more = L
    strata = i["partition"].value["strata"]
    k = indexer_factor(strata, mix_["rho"])
    census = sum(x["per_token"] * x["macs"] for x in strata if x["block"].startswith(INDEXER))
    strata = [x | {"per_token": x["per_token"] * k} if x["block"].startswith(INDEXER) else x for x in strata]
    served = i["partition"].value["served_macs_per_token"] + (k - 1) * census
    part = i["partition"]
    tokens = f1.InputCard(mix_["tokens"], "token", "modeled", "vLLM's 128k serve plan: computed tokens a request")
    i2 = i | {"partition": replace(part, value=part.value | {"strata": strata, "served_macs_per_token": served}),
              "tokens_per_request": tokens}
    line = (f1.TokenBlocks.of(strata, served), f1.Fill.of(strata, i["proof_blocks"].value), oh)
    comp2 = tuple(x["priced"] == "computed" for x in strata if x["per_token"] > 0)
    scopes = {s.name: s for s in f1.scopes(c | {"tokens_per_request": tokens}, line[0])}
    opts = (more[0] if more else {}) | {"scopes": scopes, "indexer_factor": k, "served_macs": served}
    return (name, i2, line, comp2, conf, replay_gpu_s, extra, opts)


def advice_bits(L, scope: str, bits_a_request: float) -> float:
    """The advice's charge over a scope: the line's requests there times the bits a request, until the length and
    finish are checked in-circuit."""
    scopes = L[7]["scopes"] if len(L) > 7 and "scopes" in L[7] else SCOPES
    return scopes[scope].requests * bits_a_request
