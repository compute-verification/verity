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

#: The statement cache on the H200: (fixed after over before, host share of after) by block prefix, five types priced
#: from their H200 splits and two (quant K1536, combine N5) from their CPU shares.
CACHE = {
    "Fp8GroupQuantUe8m0_v1{K=6144,": (13.3 / 97.7, 12.8 / 13.3),
    "FlashInferFp8Linear_v1{K=2048,": (11.1 / 112.3, 10.8 / 11.1),
    "AllReduceMultimem_v1{": (3.0 / 42.6, 0.27 / 3.0),
    "Fp8GroupQuantUe8m0_v1{K=2048,": (4.7 / 47.8, 4.4 / 4.7),
    "Fp8GroupQuantUe8m0_v1{K=1536,": (6.3 / 39.6, 6.0 / 6.3),
    "SparseMlaCombine_v1{": (7.7 / 42.7, 7.0 / 7.7),
}
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
#: the linear layers' blocks, against which the indexer's MACs at 128k are stated
LINEARS = ("Fused[", "MoeExpertFp8BlockGemm", "FlashInferFp8Linear", "ScaledMmFp8", "BatchedGemm", "CublasSplitKGemm",
           "LlBf16DotprodGemm", "Gemm_v2")
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
#: prefix, measured on today's flat statement
VERIFY = {"FlashInferFp8Linear_v1{K=2048,": 2474.0, "Fp8GroupQuantUe8m0_v1{K=6144,": 2100.0,
          "SparseMlaCombine_v1{NSPLIT=12,": 1634.0, "RMSNormFusedCudaVector8_v1{N=6144,": 769.0,
          "Fp8GroupQuantUe8m0_v1{K=2048,": 681.0, "Fp8GroupQuantUe8m0_v1{K=1536,": 508.0,
          "SparseMlaCombine_v1{NSPLIT=2,": 324.0, "AllReduceMultimem_v1{": 314.0, "MoeExpertFp8GemmW_v1{": 15.0}
#: its fixed part; the rest grows with the unit's matrix entries
VERIFY_FIXED = 10.0
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
#: the flat Lean verifier with its proved twins (`fast`): today's column times the after-over-before ratio measured in
#: one job a type on FlashInfer K2048 (0.207), quant K6144 (0.217) and the MoE GEMM (0.577); the other six by a
#: fixed-plus-per-entry fit to those two (0.21 to 0.23, assumed)
VERIFY_FAST = {"FlashInferFp8Linear_v1{K=2048,": 513.0, "Fp8GroupQuantUe8m0_v1{K=6144,": 456.0,
               "SparseMlaCombine_v1{NSPLIT=12,": 359.0, "RMSNormFusedCudaVector8_v1{N=6144,": 169.0,
               "Fp8GroupQuantUe8m0_v1{K=2048,": 150.0, "Fp8GroupQuantUe8m0_v1{K=1536,": 112.0,
               "SparseMlaCombine_v1{NSPLIT=2,": 75.0, "AllReduceMultimem_v1{": 72.0, "MoeExpertFp8GemmW_v1{": 8.7}
VERIFY_FAST_MEASURED = ("FlashInferFp8Linear_v1{K=2048,", "Fp8GroupQuantUe8m0_v1{K=6144,", "MoeExpertFp8GemmW_v1{")
#: the types whose templated session check measured costlier than the flat verifier, which keep the flat
VERIFY_FLAT_CHEAPER = ("RMSNorm", "DsaRMSNorm", "RmsNormMnnvl")
VERIFY_COLUMNS = {"today": (VERIFY, VERIFY_FIXED), "fast": (VERIFY_FAST, min(VERIFY_FAST.values()))}
#: the templated (structural) session check over the flat one, both before the flat verifier's proved twins, by family:
#: Lean verify's CPU-seconds under --zk with the Program on FlashInfer K2048 (1,382 against 1,751) and RMSNorm v3
#: N=6144 (454.9 against 713.5); any other type the smaller ratio (assumed)
TMPL_MEASURED = ((("FlashInferFp8Linear_v1", "ScaledMmFp8BlockSharedScale_v1"), 1382.0 / 1751.0),
                 (("RMSNorm", "DsaRMSNorm", "ResidualAdd", "RmsNormMnnvl"), 454.9 / 713.5))

#: the long-running proof server on one H200, one prover, the statement cache warm: (fixed GPU-seconds a statement,
#: GPU-seconds a block, the device's busy share, blocks a full statement) by block prefix, a two-point fit at full and a
#: quarter of a statement (rmsnorm and the all-reduce at full and half), host work and stalls included; the fused
#: blocks' w13 from its sessions a second at full statements, split as the server's fit splits it, and w2 by the
#: server's w2 over w13 (assumed)
SERVER = {"ScaledMmFp8BlockSharedScale_v1{K=6144,N=1,G=128}": (0.98, 1.41e-3, 0.28, 2048),
          "ScaledMmFp8BlockSharedScale_v1{K=2048,N=1,G=128}": (0.79, 0.74e-3, 0.28, 4096),
          "DsaMqaLogits_v3{N=1,H=32,D=128}": (0.96, 0.71e-3, 0.28, 4096),
          "ResidualAdd_v1{N=6144}": (1.95, 5.33e-3, 0.28, 512),
          "RMSNormFusedCudaVector8_v1{N=6144,": (5.11, 10.38e-3, 0.27, 256),
          "AllReduceMultimem_v1{WORLD=8,SIZE=3072,": (3.79, 9.40e-3, 0.27, 256)}
FUSED_SESSIONS_S = (0.53 + 0.59) / 2
FUSED_SERVER = {e: (f * k, m * k, 0.87, 1 << (35 - FUSED_K_LOG))
                for e, (f, m), k in (("w13", (1.24, 2.77e-3), 1 / FUSED_SESSIONS_S / (1.24 + 1024 * 2.77e-3)),
                                     ("w2", (1.30, 2.93e-3), 1 / FUSED_SESSIONS_S / (1.24 + 1024 * 2.77e-3)))}
#: the types whose prover and verifier need 100 GiB or more of host memory a copy: one prover an H200
HOST_BOUND = ("Fp8GroupQuantUe8m0_v1{K=6144,", "FlashInferFp8Linear_v1{K=2048,", "FlashInferFp8Linear_v1{K=6144,",
              "SparseMlaSplitPart_v1")
#: K provers' chains overlap at this efficiency: the light types' statements a second at K = 4 over K = 1 (2.87) over 4
OVERLAP_EFFICIENCY = 2.87 / 4
#: the quant K6144's price a block at the floor its K2048 sibling's runs show
QUANT_FLOOR = 0.003532


def gpu_factor(busy: float, prover: float = 1.0, k: int = 1, front: bool = False) -> float:
    """A statement's GPU time over one prover's: the device share `busy` sped up `prover` times, the host share never;
    the next session's front run beside this one's device work (`front`), and `k` provers sharing the GPU, their chains
    overlapping at OVERLAP_EFFICIENCY."""
    device, host = busy / prover, 1.0 - busy
    chain = max(device, host) if front else device + host
    return chain if k == 1 else max(device, chain / (k * OVERLAP_EFFICIENCY))


def templated_seconds(x, how: str) -> float:
    """The templated session check's core-seconds a statement of `x`'s type: `measured`, today's column times its
    family's measured ratio (TMPL_MEASURED); `goal`, quant K6144's estimate over today's for every type but the
    RMSNorm family, which keeps the flat."""
    today = verifier_seconds(x)[0]
    if how == "goal":
        return math.inf if x["block"].startswith(VERIFY_FLAT_CHEAPER) else VERIFY_TEMPLATED * today
    ratio = next((r for types, r in TMPL_MEASURED if x["block"].startswith(types)),
                 min(r for _, r in TMPL_MEASURED))
    return ratio * today


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


def apply_server(strata, prover=1.0, k4=False, front=False, quant_floor=True, host=1.0, plain=False, verify=0.0,
                 column="fast", tmpl=False, fused_busy=None, verify_other=None):
    """The strata priced as the long-running proof server proves them, with the statement cache warm: the measured
    types at their server prices and full statements, the rest at the card with the cache's factor; then a GPU prover
    `prover` times faster on device shares, four provers an H200 on full statements (`k4`, none on HOST_BOUND types),
    the next session's front beside this one's device work (`front`), the dense types' host witness `host` times
    lower, Flock without --zk, and the Lean verifier's core-seconds a statement from `column` times `verify` (the
    templated check where it is cheaper, `tmpl` = `measured` or `goal`, see `templated_seconds`; the assumed types at
    `verify_other` when given)."""
    table, floor = VERIFY_COLUMNS[column]
    out = []
    for x in strata:
        fixed, marginal, cap = x["fixed_seconds"], x["seconds_per_block"], x["statement_blocks"]
        sp = server_price(x)
        if sp:
            fixed, marginal, b, cap = sp
            if fused_busy is not None and x.get("kind") == "fused":
                b = fused_busy
            bf = bm = b
        else:
            if quant_floor and x["block"].startswith("Fp8GroupQuantUe8m0_v1{K=6144,"):
                marginal = QUANT_FLOOR
            e = expert(x)
            f, hs, _ = (EXPERT[e]["cache"], 0.0, "") if e else cache_factor(x)
            fixed *= f
            host_s = hs * fixed
            if host != 1.0 and x["block"].startswith(DENSE):
                fixed, host_s = fixed - host_s * (1 - 1 / host), host_s / host
            bf, bm = (1 - host_s / fixed if fixed else 1.0), 1.0
        k = 4 if k4 and not x["block"].startswith(HOST_BOUND) else 1
        fixed *= gpu_factor(bf, prover, k, front)
        marginal *= gpu_factor(bm, prover, k, front)
        if plain:
            p = 1 - share(PLAIN, x)
            fixed, marginal = fixed * p, marginal * p
        if verify:
            v, how = verifier_seconds(x, table, floor)
            if tmpl:
                v = min(v, templated_seconds(x, "goal" if tmpl is True else tmpl))
            if how == "assumed" and verify_other is not None:
                v = verify_other
            fixed += verify * v / CORE_PER_GPU
        out.append(x | {"fixed_seconds": fixed, "seconds_per_block": marginal, "statement_blocks": int(cap),
                        "priced": "measured" if sp else x["priced"]})
    return out


def expert(x) -> str | None:
    b = x["block"]
    if b.startswith(("Fused[EmbeddingRead_v1{V=256,H=3072}", "MoeExpertFp8BlockGemm_v4{K=6144")):
        return "w13"
    if b.startswith(("Fused[EmbeddingRead_v1{V=256,H=4096}", "MoeExpertFp8BlockGemmW_v4{K=256")):
        return "w2"
    return None


def cache_factor(x) -> tuple[float, float, str]:
    """(fixed after over before, host share of after, label) for a census type."""
    b = x["block"]
    for p, f in CACHE.items():
        if b.startswith(p):
            return (*f, "measured" if "1536" not in p and "Combine" not in p else "CPU share")
    for fam, p in FAMILY:
        if b.startswith(fam):
            return (*CACHE[p], "family")
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
            x = x | {"k_log": FUSED_K_LOG, "statement_blocks": max(1, tc.STATEMENT_BITS >> FUSED_K_LOG)}
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
    `INDEXER_AT_MAX` times the linears' at 131,072 keys, in proportion to the keys read, over the census's."""
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
