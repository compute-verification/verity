"""Per-type levers for the repriced break-even: the statement cache, K provers overlapped on one GPU, a faster GPU
prover, the dense types' host witness, Flock without --zk, and the 128k workload over vLLM's 128k serve plan.

Every factor multiplies the card's end-to-end prices (fixed seconds a statement, seconds a block, blocks a statement).
`LABELS` says which are measured, modeled or assumed.
"""
import json
import math
from dataclasses import replace

from common import SCOPES, c, f1

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
#: each prover's device arena then holds a quarter of a full statement
EXPERT = {"w13": {"cache": 0.56 / 5.73, "fixed": 0.045, "marginal": 0.58},
          "w2": {"cache": 0.67 / 5.54, "fixed": 0.043, "marginal": 0.54}}
PROVERS = 4
#: four provers with the cache and verify-ahead against the cache alone at one, on w13, taken for every other type
#: (assumed): fixed a statement, seconds a block
OVERLAP_OTHER = {"fixed": 0.25 / 0.56, "marginal": 0.58}
#: Flock without --zk: the share of a session --zk costs on CPU, by prefix; the fused blocks' from their parts (the
#: read 11%, w2 18%, w13 0%); every other type none
PLAIN = {"AllReduceMultimem_v1{": 0.35, "AllReduceResidualMnnvl": 0.35, "AllReduceMnnvl": 0.35, "RmsNormMnnvl": 0.35,
         "Fused[EmbeddingRead_v1{V=256,H=4096}": 0.15, "Fused[EmbeddingRead_v1{V=256,H=3072}": 0.05,
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


def apply(strata, cache=False, overlap="", prover=1.0, arena=False, host=1.0, plain=False):
    """The strata with each lever applied in turn: the cache's fixed seconds (with its host part kept apart), K provers
    overlapped on the expert types (`overlap="experts"`) or every type (`"all"`), a GPU prover `prover` times faster on
    device parts, full statements under K provers (`arena`), the dense types' host witness `host` times lower, and
    Flock without --zk."""
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
            if e:
                fixed, marginal = x["fixed_seconds"] * EXPERT[e]["fixed"], marginal * EXPERT[e]["marginal"]
            else:
                fixed, host_s = fixed * OVERLAP_OTHER["fixed"], host_s * OVERLAP_OTHER["fixed"]
                marginal *= OVERLAP_OTHER["marginal"]
            if not arena:
                cap = max(1, cap // PROVERS)
        if host != 1.0 and x["block"].startswith(DENSE):
            fixed, host_s = fixed - host_s * (1 - 1 / host), host_s / host
        if prover != 1.0:
            fixed, marginal = host_s + (fixed - host_s) / prover, marginal / prover
        if plain:
            p = 1 - share(PLAIN, x)
            fixed, marginal = fixed * p, marginal * p
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
